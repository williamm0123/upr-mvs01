#!/bin/bash -l
#SBATCH --job-name=uprmvs_lape
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
# LAPE (DA3 特征 + 在线 DA3 单目深度 + 五个先验模块) —— DTU 从头训练 10 个 epoch。
# 单卡 A100-80GB, sbatch 提交 (不是 bash)。
#
#   cd <本 checkout 的根目录> && git pull && mkdir -p logs
#   sbatch scripts/train_lape_umhpc.sh                       # 默认: LAPE_DTU_E10, 10 epoch, batch 2
#   LAPE=off sbatch scripts/train_lape_umhpc.sh              # 同口径对照: DA3 特征 + MoA (先验只定中心)
#   EPOCHS=1 RUN_NAME=LAPE_DTU_E1 sbatch scripts/train_lape_umhpc.sh   # 先跑 1 个 epoch 看速度
#
# 与旧 MoA 脚本 (train_moa_umhpc.sh) 的区别:
#   * 特征: DA3MONO-LARGE 的编码器 token 代替 DINOv3 进 SVA (MonoMVSNet 的做法), 其余特征处理不变;
#   * 单目深度: 同一次 DA3 前向的 DPT 头在线给出, 不读 log/da3_cache, 也不需要先建缓存;
#   * LAPE: RAC 多模型对齐 / 3-7-11 局部 expert + 标定 sigma / 法向一致证据 / 低频回拉 / 先验融合,
#     stage1 两遍正则 (见 docs: LAPE 终版方案)。
#   必须从 step 0 训练 (项目惯例), 不能从 moa1/moa2 续训。
#
# 时长: DTU train 27097 个样本, batch 2 -> 每 epoch 13549 步, 10 epoch = 135490 步。
#   本地 5060Ti 实测 (batch 1, 5 视角): 384x512 1.38 s/步 峰值 10.5 GiB, 448x576 1.83 s/步 13.2 GiB;
#   同尺寸 LAPE=off 1.02 s/步 10.2 GiB (LAPE 多 ~36% 时间, 显存几乎不变)。按这两点线性外推,
#   最大训练尺度 640x896 + batch 2 峰值约 52 GiB, 80GB 卡放得下。A100 上的速度**没有实测**,
#   估计约 1.5-2 s/步, 10 epoch 约 60-75 小时 —— 会超过 3 天上限, 靠下面的自动续投接着跑。
#   第一份日志的 "s/step" 和 "mem=" 就是实测值; 如果显存吃紧 (>75G) 用 PER_GPU_BATCH=1 重投。
#   提交前也可以在 interactive 里先测最大尺度:
#     python train_moa.py --profile umhpc --smoke --smoke-steps 5 --batch-size 2 --num-views 5 --smoke-hw 640 896
#
# 超时自动续投: slurm 在超时前 15 分钟发 USR1, train_moa.py 跑完当前 step 写 latest.pth 并以 124
# 退出, 本脚本用 FRESH=0 (--resume auto) 重投自己。手动续训: FRESH=0 sbatch scripts/train_lape_umhpc.sh
#
# 输出: log/experiments/$RUN_NAME/{model/{latest,best}.pth, tensorboard, config.json}
# 训完: DTU 测试 sbatch scripts/test_lape_dtu_umhpc.sh;  BlendedMVS 微调 sbatch scripts/train_lape_blended_umhpc.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$PROJECT_DIR/train_moa.py" ]]; then
    echo "PROJECT_DIR=$PROJECT_DIR 不像是本仓库的根目录 (没有 train_moa.py); 请在仓库根目录 sbatch" >&2
    exit 2
fi
cd "$PROJECT_DIR"

LAPE=${LAPE:-on}
case "$LAPE" in
    on)  RUN_NAME=${RUN_NAME:-LAPE_DTU_E10} ;;
    off) RUN_NAME=${RUN_NAME:-DA3MOA_DTU_E10} ;;
    *) echo "LAPE 只能是 on / off, 收到 '$LAPE'" >&2; exit 2 ;;
esac
EPOCHS=${EPOCHS:-10}
STEPS=${STEPS:-0}                        # >0 则按步数跑 (覆盖 EPOCHS); 0 = 按 EPOCHS
WARP_CHANNELS=${WARP_CHANNELS:-128,128,128,128}
PER_GPU_BATCH=${PER_GPU_BATCH:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
NUM_VIEWS=${NUM_VIEWS:-5}
NUM_WORKERS=${NUM_WORKERS:-12}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-5000}       # 另外每个 epoch 末都验证一次
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
SEED=${SEED:-20260526}
DETERMINISTIC=${DETERMINISTIC:-1}
DA3_RES=${DA3_RES:-518}
FRESH=${FRESH:-1}                        # 1 = 新 run (归档同名旧目录, --resume off); 0 = 续训
CHAIN=${CHAIN:-0}
MAX_CHAIN=${MAX_CHAIN:-4}
# lr: 3e-4 @ 全局 batch 2, 按 sqrt 缩放 (与 train_moa_umhpc.sh 同一规则)
LR_REF=${LR_REF:-3e-4}
LR_REF_BATCH=${LR_REF_BATCH:-2}
LR=${LR:-$(awk -v l="$LR_REF" -v g="$PER_GPU_BATCH" -v r="$LR_REF_BATCH" 'BEGIN{printf "%.4g", l*sqrt(g/r)}')}

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

RUN_DIR="log/experiments/$RUN_NAME"
if [[ "$FRESH" == "1" ]]; then
    if [[ -e "$RUN_DIR" ]]; then
        RECENT=$(find "$RUN_DIR" -type f -mmin -60 -print -quit 2>/dev/null || true)
        if [[ -n "$RECENT" && "${FORCE_ARCHIVE:-0}" != "1" ]]; then
            echo "拒绝归档: $RUN_DIR 最近 60 分钟内仍有写入 ($RECENT); 换 RUN_NAME 或确认结束后 FORCE_ARCHIVE=1" >&2
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

GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
echo "=================================================================="
echo " LAPE DTU  lape=$LAPE  run=$RUN_NAME  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=${GIT_SHA:0:12}  chain=$CHAIN/$MAX_CHAIN  fresh=$FRESH  resume=$RESUME"
echo " epochs=$EPOCHS steps=$STEPS batch=$PER_GPU_BATCH views=$NUM_VIEWS warp=$WARP_CHANNELS da3_res=$DA3_RES"
echo " lr=$LR warmup=$WARMUP_STEPS seed=$SEED"
echo "=================================================================="
nvidia-smi -L || true
python - "$PER_GPU_BATCH" <<'PY' || exit 1
import sys, torch
from base.config import ProjectPaths
if not torch.cuda.is_available():
    sys.exit("CUDA 不可用 —— 这个作业需要 --gres=gpu:1")
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"=== GPU {torch.cuda.get_device_name(0)}  {gib:.0f} GiB ===")
if gib < 70 and int(sys.argv[1]) >= 2:
    sys.exit(f"只分到 {gib:.0f} GiB 的卡, batch {sys.argv[1]} 需要 80GB A100; 换节点或 PER_GPU_BATCH=1 重投")
w = ProjectPaths().da3_weights_file
if not (w / "model.safetensors").is_file():
    sys.exit(f"DA3 权重不在 {w} (需要 config.json + model.safetensors)")
import depth_anything_3.api  # noqa: F401  (fail here, not after the dataset is built)
print(f"=== DA3 weights {w} ===")
PY

args=(
    --profile umhpc
    --name "$RUN_NAME"
    --moa on --lape "$LAPE" --feat-backbone da3 --da3-process-res "$DA3_RES"
    --warp-channels "$WARP_CHANNELS"
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
    --ckpt-interval "$CKPT_INTERVAL"
    --resume "$RESUME"
)
if [[ "$STEPS" -gt 0 ]]; then
    args+=(--max-steps "$STEPS")
else
    args+=(--epochs "$EPOCHS")
fi
[[ "$DETERMINISTIC" == "1" ]] && args+=(--deterministic)

python train_moa.py "${args[@]}" &
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
        echo "=== 已续投 $CHAIN 次 (MAX_CHAIN=$MAX_CHAIN), 不再自动续投; 手动: FRESH=0 sbatch $0 ===" >&2
        exit 1
    fi
    NEXT=$((CHAIN + 1))
    echo "=== 超时存档完成, 续投第 $NEXT 次 ==="
    sbatch --export=ALL,FRESH=0,CHAIN=$NEXT,LAPE=$LAPE,RUN_NAME=$RUN_NAME,PER_GPU_BATCH=$PER_GPU_BATCH,LR=$LR,EPOCHS=$EPOCHS,STEPS=$STEPS,WARP_CHANNELS=$WARP_CHANNELS,DA3_RES=$DA3_RES \
        "$PROJECT_DIR/scripts/train_lape_umhpc.sh"
    exit 0
fi
echo "=== train_moa.py 退出码 $RC ==="
exit $RC
