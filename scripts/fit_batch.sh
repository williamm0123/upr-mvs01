#!/bin/bash -l
# =============================================================================
# 在**目标卡**上量 per-GPU batch 的显存占用, 然后再决定 PER_GPU_BATCH。
# 不训练, 扫完就退出。在已经拿到 GPU 的 interactive shell 里跑。
#
#   ARM=w1 bash scripts/fit_batch.sh
#   ARM=w3b BATCHES=1,2,4,6,8,10 TARGET=0.95 bash scripts/fit_batch.sh
#
# 为什么不能靠算: 显存对 batch 近似线性, 但对**像素数不是**干净的线性 ——
# stage4 全分辨率代价体占大头, 和 FPN/DINO 的缩放规律不一样。小卡上量到的斜率
# 外推到 80GB 有好几个 batch 的不确定度。所以只能在目标卡上量。
#
# 也不要用 scripts/verify_w1.py --mem 来定 batch: 那条路把 dino 和 spre 关掉了,
# 数字系统性偏低。本脚本走的是 train.py 里**与训练完全同一条** cfg 覆盖路径。
#
# 量的是多尺度里**最大**的那个尺度 (640x896)。训练尺度跨度 2.22 倍, 按平均尺度
# 定 batch 的话最大尺度那几步必 OOM。
# =============================================================================
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/uprmvs/bin/python}
TRAIN_PROFILE=${TRAIN_PROFILE:-umhpc}
BATCHES=${BATCHES:-1,2,4,5,6,7,8}
TARGET=${TARGET:-0.95}
FIT_HW=${FIT_HW:-}                  # 空 = 用多尺度里最大的

[[ -d "$PROJECT_DIR" ]] || { echo "找不到项目目录: $PROJECT_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN"  ]] || { echo "找不到解释器: $PYTHON_BIN" >&2; exit 1; }
cd "$PROJECT_DIR"

ARM=${ARM:-w1}
NPROC=${NPROC:-1}
# shellcheck source=scripts/_arm_common.sh
source "$PROJECT_DIR/scripts/_arm_common.sh"

export UPRMVS_MACHINE=${UPRMVS_MACHINE:-umhpc}
export UPRMVS_PROFILE="$TRAIN_PROFILE"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=================================================================="
echo " 显存扫描  arm=$ARM  batches=$BATCHES  target=$TARGET"
echo " host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "=================================================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# lr 基准与 _arm_common.sh 用同一组值, 免得两边的建议对不上
fit_args=(--fit-batch "$BATCHES" --fit-target "$TARGET"
          --fit-lr-ref "$LR_REF" --fit-lr-ref-batch "$LR_REF_BATCH")
[[ -n "$FIT_HW" ]] && fit_args+=(--fit-hw "$FIT_HW")

# --batch-size 在 --fit-batch 模式下不起作用 (扫描列表说了算), 但 cfg 要有个值
exec "$PYTHON_BIN" train.py \
    --gpus 1 \
    --ddp off \
    --batch-size 1 \
    --name "fit_$ARM" \
    "${COMMON_ARGS[@]}" \
    "${ARM_ARGS[@]}" \
    "${fit_args[@]}"
