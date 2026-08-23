#!/bin/bash -l
#SBATCH --job-name=uprmvs_1gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# UPRMVS 改造工单 v3 —— **单卡 A100-80GB** 训练。一个脚本四个 arm。
#
#   ARM=w0 sbatch scripts/sbatch_1gpu.sh     # W0-A 基线 R32_f10 -> 30k
#   ARM=w1 sbatch scripts/sbatch_1gpu.sh     # W1  Depth-core vNext (修订版) -> 30k
#   ARM=w3 sbatch scripts/sbatch_1gpu.sh     # W3  UPRMVS-3D vNext -> 30k (从头)
#   ARM=w3b sbatch scripts/sbatch_1gpu.sh    # W3 + W3-B 源视图可见性监督
#
# 2026-08-23 起**只用单卡**, 双卡 DDP 路径不再使用。原因是双卡下没有哪种配置能
# 同时对齐"全局 batch"和"BN 批量" (FPN 是 BatchNorm2d 且全网无 SyncBN), 单卡
# 反而口径干净。DDP 代码保留可用, 只是不再是默认路径。
#
# 顺序是工单排的, 不要跳:
#   w0 跑完 -> best.pth -> scripts/tail_posterior_stats.py 出 deployable 判据
#   -> 决定 W2 做不做 -> w1 -> w3。W3 **从头训练**而不是在 W1 权重上续训,
#   为的是归因清楚。
#
# -----------------------------------------------------------------------------
# 批量: per-GPU = 全局 = 4 (PER_GPU_BATCH)。lr 按 sqrt(全局/2) 自 3e-4 缩放。
# 换卡或换 arm 之后先量显存:  ARM=w1 bash scripts/fit_batch.sh
#
# 速度参考: batch 5 单卡实测约 3.6 s/step, 30k 步约 30 小时 (--time 给了 48h)。
# =============================================================================

set -euo pipefail

ARM=${ARM:-w1}
PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

# conda 的 activate.d 脚本会读 LD_LIBRARY_PATH 这类未必存在的变量; set -u 下
# 会直接 unbound variable 退出。只在激活期间关掉 nounset。
set +u
source ~/.bashrc
conda activate uprmvs
set -u

# --- arm 与公共参数: 与单卡脚本 (train_umhpc_interactive.sh) 同源 ---
# 两个脚本各抄一份 arg 列表迟早会漂, 那时候两条曲线还长得很像, 但已经不是
# 同一个实验了。唯一该有的差别是进程数与全局 batch。
# NPROC 必须在 source **之前**定好: _arm_common.sh 用它算 GLOBAL_BATCH 和 lr。
NPROC=1
# shellcheck source=scripts/_arm_common.sh
source "$PROJECT_DIR/scripts/_arm_common.sh"

# -------------------------------------------------------- 干净工作树是硬要求
# 跑出来的曲线必须对得上一个 commit, 否则几周后没人能说清那条线是哪版代码。
GIT_SHA=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "工作树有未提交的改动 —— 这一版跑出来的结果对不上任何 commit" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

RUN_DIR="log/experiments/$RUN_NAME"
if [[ -e "$RUN_DIR" ]]; then
    ARCHIVE="log/experiments/_archive/${RUN_NAME}_$(date -u +%Y%m%d_%H%M%S)"
    mkdir -p "$(dirname "$ARCHIVE")"
    mv "$RUN_DIR" "$ARCHIVE"
    echo "=== 上一轮已归档 (未删除): $RUN_DIR -> $ARCHIVE ==="
fi
mkdir -p logs

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=================================================================="
echo " arm=$ARM  run=$RUN_NAME  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=${GIT_SHA:0:12} (clean)"
echo " 单卡  per_gpu_batch=$PER_GPU_BATCH  global_batch=$GLOBAL_BATCH"
echo " steps=$STEPS  horizon=$LR_HORIZON  amp=$AMP_DTYPE  seed=$SEED"
echo " lr=$LR  (${LR_SCALING} 缩放自 $LR_REF @ 全局 batch $LR_REF_BATCH)"
echo " arm_args: ${ARM_ARGS[*]}"
echo "=================================================================="
nvidia-smi -L

exec python train.py \
    --gpus 1 \
    --ddp off \
    --batch-size "$PER_GPU_BATCH" \
    --name "$RUN_NAME" \
    "${COMMON_ARGS[@]}" \
    "${ARM_ARGS[@]}"
