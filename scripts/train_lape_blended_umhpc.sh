#!/bin/bash -l
#SBATCH --job-name=uprmvs_lape_bld
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
# LAPE 在 BlendedMVS + DTU 上均衡混合微调 10 个 epoch (MVSFormer++ 的 --balanced_training)。
# 单卡 A100-80GB, sbatch 提交。前提: DTU 训练 (train_lape_umhpc.sh) 已完成。
#
#   cd <本 checkout 的根目录> && git pull && mkdir -p logs
#   sbatch scripts/train_lape_blended_umhpc.sh                           # 默认从 LAPE_DTU_E10 的 latest.pth 起
#   INIT_CKPT=log/experiments/XXX/model/latest.pth RUN_NAME=XXX_BLD sbatch scripts/train_lape_blended_umhpc.sh
#   MIX_DTU=off sbatch scripts/train_lape_blended_umhpc.sh               # 只用 BlendedMVS
#
# 数据: /scr/user/qinglong/dataset/BlendedMVS_lowres —— 原版 BlendedMVS 低分辨率 (576x768), 场景目录名是
# 官方 24 位 id; 划分用 MVSFormer++ 的官方 106/7 列表 lists/blended/{training,validation}_list.txt。
# 混合方式照 MVSFormer++ (reference/MVSFormerPlusPlus: datasets/balanced_sampling.py,
# config/mvsformer++_ft.json): 每个 epoch 从 DTU (lists/dtu/trainval.txt) 和 Blended-train 各取
# min(两者长度) 个样本混洗; 一个 batch 内共用一个裁剪尺度 (两边都用 Blended 的多尺度列表, 最大 576x768);
# 源视角从 pair 前 7 (Blended) / 全部 (DTU) 随机取; 随机裁剪落到全空 GT 时重抽;
# lr 1e-4, warmup 500, 10 epoch。验证只在 Blended-val 上 (误差按假设间隔计, acc_2mm = 2 个间隔内)。
#
# DA3 在网络里在线推理, **不再需要** log/da3_cache_blended 或 DTU 的 DA3 缓存。
# 架构取自 INIT_CKPT 的快照 (严格加载: 缺一个权重就报错), 新优化器 + 新 warmup/cosine。
#
# 时长: 每 epoch = 2 x min(DTU trainval 样本数, Blended-train 样本数) 个样本 —— 本地实测
# DTU trainval 33271 / Blended-train 16904 -> 每 epoch 33808 个样本, batch 2 下 16904 步,
# 10 epoch = 169040 步 (启动日志 "[mix] ... per dataset per epoch" 一行可核对)。Blended 的最大裁剪
# 576x768 比 DTU 的 640x896 小, 每步略快于 DTU 阶段; A100 速度没有实测, 按 1.3-1.8 s/步估计约
# 60-85 小时, 靠自动续投接着跑 (最多 MAX_CHAIN 次)。只想先看效果可以 EPOCHS=3。
#
# 超时: 提前 15 分钟 USR1 -> 写 latest.pth -> 以 FRESH=0 自动续投。手动续训: FRESH=0 sbatch $0
# 训完测试: CKPT=log/experiments/$RUN_NAME/model/latest.pth sbatch scripts/test_tnt_umhpc.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$PROJECT_DIR/train_blended.py" ]]; then
    echo "PROJECT_DIR=$PROJECT_DIR 不像是本仓库的根目录 (没有 train_blended.py); 请在仓库根目录 sbatch" >&2
    exit 2
fi
cd "$PROJECT_DIR"

RUN_NAME=${RUN_NAME:-LAPE_BLD_E10}
INIT_CKPT=${INIT_CKPT:-$PROJECT_DIR/log/experiments/LAPE_DTU_E10/model/latest.pth}
BLENDED_ROOT=${BLENDED_ROOT:-/scr/user/qinglong/dataset/BlendedMVS_lowres}
TRAIN_LIST=${TRAIN_LIST:-$PROJECT_DIR/lists/blended/training_list.txt}
VAL_LIST=${VAL_LIST:-$PROJECT_DIR/lists/blended/validation_list.txt}
MIX_DTU=${MIX_DTU:-on}
EPOCHS=${EPOCHS:-10}
STEPS=${STEPS:-0}                        # >0 则按步数跑 (覆盖 EPOCHS)
PER_GPU_BATCH=${PER_GPU_BATCH:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
NUM_VIEWS=${NUM_VIEWS:-5}
NUM_WORKERS=${NUM_WORKERS:-12}
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-500}
VAL_INTERVAL=${VAL_INTERVAL:-5000}
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

[[ -f "$INIT_CKPT" ]] || { echo "INIT_CKPT 不存在: $INIT_CKPT (先跑 train_lape_umhpc.sh, 或 INIT_CKPT=... 指定)" >&2; exit 2; }
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
echo " LAPE BlendedMVS fine-tune  run=$RUN_NAME  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)  chain=$CHAIN/$MAX_CHAIN  resume=$RESUME"
echo " init=$INIT_CKPT"
echo " data=$BLENDED_ROOT  mix_dtu=$MIX_DTU  epochs=$EPOCHS steps=$STEPS"
echo " batch=$PER_GPU_BATCH views=$NUM_VIEWS lr=$LR warmup=$WARMUP_STEPS"
echo "=================================================================="
nvidia-smi -L || true
python - "$PER_GPU_BATCH" <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA 不可用 —— 需要 --gres=gpu:1")
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"=== GPU {torch.cuda.get_device_name(0)}  {gib:.0f} GiB ===")
if gib < 70 and int(sys.argv[1]) >= 2:
    sys.exit(f"只分到 {gib:.0f} GiB 的卡, batch {sys.argv[1]} 需要 80GB A100; PER_GPU_BATCH=1 重投")
PY

# 场景检查: 官方列表里的场景在 BLENDED_ROOT 下都在且完整
python scripts/blended_lists.py --root "$BLENDED_ROOT" --check-lists "$TRAIN_LIST" "$VAL_LIST" \
    || { echo "列表里的场景在 $BLENDED_ROOT 下不全 (见上面 !! 行)" >&2; exit 2; }

args=(
    --profile umhpc
    --name "$RUN_NAME"
    --init-from "$INIT_CKPT"
    --blended-root "$BLENDED_ROOT"
    --train-list "$TRAIN_LIST"
    --val-list "$VAL_LIST"
    --mix-dtu "$MIX_DTU"
    --mvsformer-sampling "${MVSFORMER_SAMPLING:-on}"
    --moa on
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
if [[ "$STEPS" -gt 0 ]]; then
    args+=(--max-steps "$STEPS")
else
    args+=(--epochs "$EPOCHS")
fi
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
    sbatch --export=ALL,FRESH=0,CHAIN=$NEXT,RUN_NAME=$RUN_NAME,INIT_CKPT=$INIT_CKPT,BLENDED_ROOT=$BLENDED_ROOT,TRAIN_LIST=$TRAIN_LIST,VAL_LIST=$VAL_LIST,MIX_DTU=$MIX_DTU,EPOCHS=$EPOCHS,STEPS=$STEPS,PER_GPU_BATCH=$PER_GPU_BATCH,LR=$LR,WARMUP_STEPS=$WARMUP_STEPS \
        "$PROJECT_DIR/scripts/train_lape_blended_umhpc.sh"
    exit 0
fi
echo "=== train_blended.py 退出码 $RC ==="
exit $RC
