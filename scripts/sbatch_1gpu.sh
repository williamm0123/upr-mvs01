#!/bin/bash -l
#SBATCH --job-name=uprmvs_sva
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
# UPRMVS —— **单卡 A100-80GB 正式训练** (sbatch 排队, 不是 bash)。
#
#   cd /scr/user/qinglong/projects/upr-mvs01
#   git pull
#   sbatch scripts/sbatch_1gpu.sh                 # 默认 ARM=sva, 30k 步
#
# 当前主线 ARM=sva (2026-09-18, 定义在 scripts/_arm_common.sh):
#   * MVSFormer++ 完整 SVA: DINO SVA -> deconv -> + FPN 1/8 -> 普通 FPN (out0 头)
#     -> FMT_with_pathway (1/8 上 2D-PE + self/cross x2, 再加第二条逐级路径)
#   * stage1 候选 44 global + 4 local (RANGE_MIN_GI 已按 43/31 换算)
#   * CVPE 已卸载; 其余沿用 vNext 基座 (legacy_depth / expect / geo_valid / conf_head)
# 其它 arm (w0/w1/w3/w3b) 仍可用: ARM=w0 sbatch scripts/sbatch_1gpu.sh
#
# 提交之前先在 interactive 分配里跑一遍 scripts/smoke_interactive.sh (实现校验 +
# 显存实测)。它会告诉你 batch 4 在最大训练尺度 640x896 上的峰值显存。
#
# -----------------------------------------------------------------------------
# 批量: per-GPU = 全局 = 4 (PER_GPU_BATCH)。lr 按 sqrt(全局/2) 自 3e-4 缩放 = 4.243e-4。
# 本机 (5060 Ti) 在 320x448 上实测: 完整 SVA 让每样本显存 +17%
# (3.49 -> 4.09 GiB; 大头是第二条路径在全分辨率上的 128 通道 3x3 conv)。
# 外推到 640x896 x batch 4 约 63 GiB allocated (~80%), 80GB 卡够用但余量不大 ——
# 以 smoke_interactive.sh 的 [3] 实测为准, 超过 90% 就 PER_GPU_BATCH=3。
#
# 速度参考: 旧 W0 基座 batch 4 单卡 30k 约 18.6 h; 完整 SVA 多了 1/8 上四层线性
# 注意力和一条全分辨率路径, 预计慢 10-25%。--time 给 2 天, --qos=long。
#
# 输出: log/experiments/$RUN_NAME/{model,tensorboard}; 已存在的同名目录会被
# **归档** (移到 log/experiments/_archive/, 不删除)。
# 训练完之后的推理 + fusibile 融合: scripts/test_dtu_fusibile.sh
# =============================================================================

set -euo pipefail

ARM=${ARM:-sva}
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

# 只记一下当前 SHA 进日志, 不做任何拦截。
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)

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
echo " git=${GIT_SHA:0:12}"
echo " 单卡  per_gpu_batch=$PER_GPU_BATCH  global_batch=$GLOBAL_BATCH"
echo " steps=$STEPS  horizon=$LR_HORIZON  amp=$AMP_DTYPE  seed=$SEED"
echo " lr=$LR  (${LR_SCALING} 缩放自 $LR_REF @ 全局 batch $LR_REF_BATCH)"
echo " stage1: global=$NUM_GLOBAL local=$NUM_LOCAL  range_min_gi=$RANGE_MIN_GI"
echo " arm_args: ${ARM_ARGS[*]}"
echo "=================================================================="
nvidia-smi -L || true
# 拿不到卡时 train.py 会退回 CPU 然后慢到看不出是出错 —— 在这里直接失败。
python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || { echo "CUDA 不可用 —— 这个作业需要 --gres=gpu:1" >&2; exit 1; }

exec python train.py \
    --gpus 1 \
    --ddp off \
    --batch-size "$PER_GPU_BATCH" \
    --name "$RUN_NAME" \
    "${COMMON_ARGS[@]}" \
    "${ARM_ARGS[@]}"
