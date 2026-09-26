#!/bin/bash -l
#SBATCH --job-name=uprmvs_bld
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# moa1 (DTU, MOA_E15/model/latest.pth) 在 BlendedMVS 上微调 30k 步 —— 单卡 A100, sbatch 提交。
#
#   cd /scr/user/qinglong/projects/upr-mvs01 && git pull && mkdir -p logs
#   sbatch scripts/train_blended_umhpc.sh
#
# 作业里依次做三件事 (前两步都可断点续跑, 已完成时几秒钟就过去):
#   1. scripts/blended_lists.py —— 检查 BLENDED_ROOT 每个场景的文件, 写
#      lists/blended_plus/{train,val,all}.txt (val = 7 个场景, 排序后等距抽取, 已有就沿用)
#   2. DA3 缓存 log/da3_cache_blended (原生 768, DA3_SHARDS 个进程并行; 有失败就停)
#   3. train_blended.py —— 架构取自 INIT_CKPT 的快照, 并固定成 moa1 的行为:
#      warp 128/64/32/16, global_solver=huber, moa_gain=1,1,1, edge_snap=off
#      (即不带 moa2 的逐级降权 / 边缘硬选面)。只加载权重 (严格: 缺一个就报错),
#      新优化器 + 新 warmup/cosine, horizon = STEPS。
#
# 验证指标: abs_err / acc_2mm 按 "DTU 深度范围下的 mm" 归一化 (见 data/blended_moa.py),
# 与 DTU 的数量级可比; best.pth 按它选。
#
# 常用覆盖:  LR=2e-4 sbatch ...   PER_GPU_BATCH=2 sbatch ...   RUN_NAME=xxx sbatch ...
# 超时: 提前 15 分钟 USR1 -> 写 latest.pth -> 以 FRESH=0 自动续投 (最多 MAX_CHAIN 次)。
# 手动续训: FRESH=0 sbatch scripts/train_blended_umhpc.sh
# 训完测试: sbatch scripts/test_tnt_umhpc.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$PROJECT_DIR/train_blended.py" ]]; then
    echo "PROJECT_DIR=$PROJECT_DIR 不像是本仓库的根目录 (没有 train_blended.py); 请在仓库根目录 sbatch" >&2
    exit 2
fi
cd "$PROJECT_DIR"

RUN_NAME=${RUN_NAME:-MOA1_BLD_30K}
INIT_CKPT=${INIT_CKPT:-$PROJECT_DIR/log/experiments/MOA_E15/model/latest.pth}
BLENDED_ROOT=${BLENDED_ROOT:-/scr/user/qinglong/dataset/BlendedMVS_plus}
LIST_DIR=${LIST_DIR:-$PROJECT_DIR/lists/blended_plus}
DA3_ROOT=${DA3_ROOT:-$PROJECT_DIR/log/da3_cache_blended}
DA3_SHARDS=${DA3_SHARDS:-3}
STEPS=${STEPS:-30000}
PER_GPU_BATCH=${PER_GPU_BATCH:-4}       # 与 moa1 相同; Blended 最大裁剪 576x768 < DTU 的 640x896
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
NUM_VIEWS=${NUM_VIEWS:-5}
NUM_WORKERS=${NUM_WORKERS:-12}
LR=${LR:-1e-4}                          # 微调: moa1 从头训练是 4.243e-4 @ batch 4
WARMUP_STEPS=${WARMUP_STEPS:-500}
VAL_INTERVAL=${VAL_INTERVAL:-2000}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-600}
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
SEED=${SEED:-20260526}
DETERMINISTIC=${DETERMINISTIC:-1}
FRESH=${FRESH:-1}
CHAIN=${CHAIN:-0}
MAX_CHAIN=${MAX_CHAIN:-4}

set +u
source ~/.bashrc
conda activate uprmvs
set -u

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p logs

[[ -f "$INIT_CKPT" ]] || { echo "INIT_CKPT 不存在: $INIT_CKPT" >&2; exit 2; }
[[ -d "$BLENDED_ROOT" ]] || { echo "BLENDED_ROOT 不存在: $BLENDED_ROOT" >&2; exit 2; }

RUN_DIR="log/experiments/$RUN_NAME"
if [[ "$FRESH" == "1" ]]; then
    if [[ -e "$RUN_DIR" ]]; then
        RECENT=$(find "$RUN_DIR" -type f -mmin -60 -print -quit 2>/dev/null || true)
        if [[ -n "$RECENT" && "${FORCE_ARCHIVE:-0}" != "1" ]]; then
            echo "拒绝归档: $RUN_DIR 最近 60 分钟内仍有写入 ($RECENT); 换 RUN_NAME 或 FORCE_ARCHIVE=1" >&2
            exit 2
        fi
        ARCHIVE="log/experiments/_archive/${RUN_NAME}_$(date -u +%Y%m%d_%H%M%S)"
        mkdir -p "$(dirname "$ARCHIVE")"
        mv "$RUN_DIR" "$ARCHIVE"
        echo "=== 上一轮已归档 (未删除): $RUN_DIR -> $ARCHIVE ==="
    fi
    RESUME=off
else
    RESUME=auto
fi

echo "=================================================================="
echo " BlendedMVS fine-tune  run=$RUN_NAME  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)  chain=$CHAIN/$MAX_CHAIN  resume=$RESUME"
echo " init=$INIT_CKPT"
echo " data=$BLENDED_ROOT  lists=$LIST_DIR  da3=$DA3_ROOT"
echo " steps=$STEPS batch=$PER_GPU_BATCH views=$NUM_VIEWS lr=$LR warmup=$WARMUP_STEPS"
echo "=================================================================="
nvidia-smi -L || true
python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA 不可用 —— 需要 --gres=gpu:1")'

# ---- 1. scene lists ---------------------------------------------------------
python scripts/blended_lists.py --root "$BLENDED_ROOT" --out "$LIST_DIR"

# ---- 2. DA3 cache (native 768; resumable) -----------------------------------
DA3_ARGS=(--dataset blended --root "$BLENDED_ROOT" --scenes-file "$LIST_DIR/all.txt" --out "$DA3_ROOT")
python scripts/build_da3_cache_mvs.py "${DA3_ARGS[@]}" --dry-run
pids=()
for ((i = 0; i < DA3_SHARDS; i++)); do
    python scripts/build_da3_cache_mvs.py "${DA3_ARGS[@]}" --shard "$i/$DA3_SHARDS" \
        > "logs/da3_blended_${SLURM_JOB_ID:-manual}_$i.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 2 logs/da3_blended_${SLURM_JOB_ID:-manual}_*.log || true
[[ $fail == 0 ]] || { echo "DA3 缓存有分片失败, 见 logs/da3_blended_${SLURM_JOB_ID:-manual}_*.log; 重新 sbatch 会补齐" >&2; exit 1; }

# ---- 3. train ---------------------------------------------------------------
args=(
    --profile umhpc
    --name "$RUN_NAME"
    --init-from "$INIT_CKPT"
    --blended-root "$BLENDED_ROOT"
    --train-list "$LIST_DIR/train.txt"
    --val-list "$LIST_DIR/val.txt"
    --da3-root "$DA3_ROOT"
    --da3-missing error
    --moa on --moa1-semantics on
    --max-steps "$STEPS"
    --batch-size "$PER_GPU_BATCH"
    --val-batch-size "$VAL_BATCH_SIZE"
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --warmup-steps "$WARMUP_STEPS"
    --amp on --amp-dtype bf16
    --multi-scale on
    --seed "$SEED"
    --log-interval "$LOG_INTERVAL"
    --val-interval "$VAL_INTERVAL"
    --max-val-samples "$MAX_VAL_SAMPLES"
    --ckpt-interval "$CKPT_INTERVAL"
    --resume "$RESUME"
)
[[ "$DETERMINISTIC" == "1" ]] && args+=(--deterministic)

python train_blended.py "${args[@]}" &
PID=$!
trap 'echo "=== 收到 USR1 (即将超时), 通知训练进程存档 ==="; kill -USR1 "$PID" 2>/dev/null || true' USR1
set +e
wait "$PID"
RC=$?
if [[ $RC -gt 128 ]]; then
    wait "$PID"
    RC=$?
fi
set -e

if [[ $RC -eq 124 ]]; then
    if [[ $CHAIN -ge $MAX_CHAIN ]]; then
        echo "=== 已续投 $CHAIN 次, 不再自动续投; 手动: FRESH=0 sbatch $0 ===" >&2
        exit 1
    fi
    NEXT=$((CHAIN + 1))
    echo "=== 超时存档完成, 续投第 $NEXT 次 ==="
    sbatch --export=ALL,FRESH=0,CHAIN=$NEXT,RUN_NAME=$RUN_NAME,INIT_CKPT=$INIT_CKPT,BLENDED_ROOT=$BLENDED_ROOT,LIST_DIR=$LIST_DIR,DA3_ROOT=$DA3_ROOT,STEPS=$STEPS,PER_GPU_BATCH=$PER_GPU_BATCH,LR=$LR,WARMUP_STEPS=$WARMUP_STEPS \
        "$PROJECT_DIR/scripts/train_blended_umhpc.sh"
    exit 0
fi
echo "=== train_blended.py 退出码 $RC ==="
exit $RC
