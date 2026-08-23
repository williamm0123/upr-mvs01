#!/bin/bash -l
# =============================================================================
# UPRMVS 工单 v3 —— 单卡真实训练, 在已经分配到 GPU 的 UMHPC interactive shell
# 中直接运行。本脚本不含 sbatch/salloc/srun, 不申请任何资源。
#
#   salloc --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=1-00:00:00
#   cd /scr/user/qinglong/projects/upr-mvs01
#   ARM=w1 bash scripts/train_umhpc_interactive.sh
#
# arm 与**每张卡**的训练参数全部来自 scripts/_arm_common.sh —— 与
# scripts/sbatch_ddp2.sh 是同一份。两个脚本各抄一份 arg 列表迟早会漂, 那时候
# 两条曲线还长得很像, 但已经不是同一个实验了。
#
# -----------------------------------------------------------------------------
# 这一条必须先说清楚: 每张卡的参数相同, **全局 batch 不同**。
#
#     双卡 sbatch_ddp2.sh : 2 进程 x per-GPU 5 = 全局 batch 10
#     单卡 (本脚本)       : 1 进程 x per-GPU 5 = 全局 batch 5
#
# 所以本脚本的曲线**不能**直接和双卡的比 —— 每步看到的样本少一半。lr 会按
# 各自的全局 batch 自动 sqrt 缩放 (见 _arm_common.sh), 所以两边的 lr 也不同,
# 这是有意的。要跟双卡对齐全局 batch 就 PER_GPU_BATCH=10, 但那样每卡参数又
# 不同了 (显存也放不下) —— 两者不可兼得, 自己挑一个并记录。
#
# 显存: 提交长跑之前先量, 不要信默认的 PER_GPU_BATCH=5:
#     ARM=w1 bash scripts/fit_batch.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
# 直接使用 uprmvs 环境，避免依赖 interactive shell 是否执行过 conda activate。
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/uprmvs/bin/python}
TRAIN_PROFILE=${TRAIN_PROFILE:-umhpc}

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory not found: $PROJECT_DIR" >&2
    echo "Override it with: PROJECT_DIR=/path/to/upr-mvs01 bash $0" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    echo "Override it with: PYTHON_BIN=/path/to/uprmvs/bin/python bash $0" >&2
    exit 1
fi
cd "$PROJECT_DIR"

# --- arm 与公共参数: 与双卡脚本同源 ---
ARM=${ARM:-w1}
NPROC=1                       # 必须在 source 之前: 用来算 GLOBAL_BATCH 和 lr
# shellcheck source=scripts/_arm_common.sh
source "$PROJECT_DIR/scripts/_arm_common.sh"
# 单卡 run 名加后缀, 免得和双卡的 run 目录撞在一起被归档掉
RUN_NAME="${RUN_NAME}${RUN_SUFFIX:-_1gpu}"

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE="$TRAIN_PROFILE"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=================================================================="
echo " UMHPC interactive 单卡训练 (真实训练, 不是 smoke)"
echo " arm=$ARM  run=$RUN_NAME  job=${SLURM_JOB_ID:-none}  host=$(hostname)"
echo " project=$PROJECT_DIR"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "=================================================================="
nvidia-smi -L
"$PYTHON_BIN" -c 'import sys, torch; print("python:", sys.executable); print("torch:", torch.__version__); print("cuda available:", torch.cuda.is_available()); print("visible GPUs:", torch.cuda.device_count())'

if ! "$PYTHON_BIN" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)'; then
    echo "No CUDA GPU is visible. Enter a GPU interactive allocation before running this script." >&2
    exit 1
fi

# 跑出来的曲线要对得上一个 commit, 否则几周后没人说得清那条线是哪版代码。
# 单卡是 interactive 探索用的, 所以这里只警告不拦截 (双卡 sbatch 是硬拦)。
if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "!! 工作树有未提交的改动 —— 这一版的结果对不上任何 commit:"
    git status --short --untracked-files=no >&2
else
    echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) (clean)"
fi

echo "per_gpu_batch=$PER_GPU_BATCH  全局 batch=$GLOBAL_BATCH (双卡同 batch 时是 $((2 * PER_GPU_BATCH)))"
echo "steps=$STEPS horizon=$LR_HORIZON lr=$LR (${LR_SCALING} 缩放自 $LR_REF @ 全局 $LR_REF_BATCH) seed=$SEED"
echo "views=$NUM_VIEWS workers=$NUM_WORKERS val_batch=$VAL_BATCH_SIZE val_interval=$VAL_INTERVAL"
echo "build_priors=$BUILD_PRIORS resume=$RESUME"
echo "arm_args: ${ARM_ARGS[*]}"

exec "$PYTHON_BIN" train.py \
    --gpus 1 \
    --ddp off \
    --batch-size "$PER_GPU_BATCH" \
    --name "$RUN_NAME" \
    "${COMMON_ARGS[@]}" \
    "${ARM_ARGS[@]}"
