#!/bin/bash -l

set -euo pipefail

# =============================================================================
# 用户参数区（服务器单张 A100）
# =============================================================================
# 用法：
#   跑通测试：SMOKE=1 bash scripts/train_umhpc_single_gpu.sh
#   两步真数据：BUILD_PRIORS=skip STEPS=2 bash scripts/train_umhpc_single_gpu.sh
#   正常训练：bash scripts/train_umhpc_single_gpu.sh

# 路径与环境
PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
CONDA_ENV=${CONDA_ENV:-mvs}
TRAIN_PROFILE=${TRAIN_PROFILE:-local}
RUN_NAME=${RUN_NAME:-uprmvs_1gpu_${SLURM_JOB_ID:-manual}}

# 核心训练参数（命令行会覆盖 TRAIN_PROFILE 中的同名参数）
BATCH_SIZE=${BATCH_SIZE:-2}       # 单卡 batch size；显存不足时保持 1
NUM_VIEWS=${NUM_VIEWS:-5}         # MVS 总视图数：1 个参考视图 + 2 个源视图
NUM_WORKERS=${NUM_WORKERS:-8}     # DataLoader 进程数；32 CPU 下建议 8
LEARNING_RATE=${LEARNING_RATE:-1e-4}
AMP=${AMP:-on}                    # on/off；A100 建议 on
STEPS=${STEPS:-0}                 # 0=使用 profile 默认值；测试可设 2

# 先验与跑通测试
BUILD_PRIORS=${BUILD_PRIORS:-auto}
# BUILD_PRIORS: auto=补齐缺失先验，force=全部重算，skip=要求缓存已存在
SMOKE=${SMOKE:-0}                 # 1=合成数据跑通测试；0=真实数据训练
SMOKE_STEPS=${SMOKE_STEPS:-2}     # SMOKE=1 时执行的训练步数
# =============================================================================

cd "$PROJECT_DIR"
mkdir -p logs

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE="$TRAIN_PROFILE"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models/vggt:$PROJECT_DIR/models/Depth-Anything-3/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) profile=$TRAIN_PROFILE ==="
nvidia-smi -L
echo "=== batch=$BATCH_SIZE views=$NUM_VIEWS workers=$NUM_WORKERS lr=$LEARNING_RATE amp=$AMP steps=$STEPS build_priors=$BUILD_PRIORS smoke=$SMOKE ==="

train_args=(
    --profile "$TRAIN_PROFILE"
    --gpus 1
    --ddp off
    --batch-size "$BATCH_SIZE"
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --lr "$LEARNING_RATE"
    --amp "$AMP"
    --name "$RUN_NAME"
)

case "$SMOKE" in
    1|true|TRUE|yes|YES)
        train_args+=(
            --smoke
            --smoke-steps "$SMOKE_STEPS"
            --build-priors skip
        )
        ;;
    0|false|FALSE|no|NO)
        train_args+=(
            --steps "$STEPS"
            --build-priors "$BUILD_PRIORS"
        )
        ;;
    *)
        echo "SMOKE must be 0/1, true/false, or yes/no; got: $SMOKE" >&2
        exit 2
        ;;
esac

conda run -n "$CONDA_ENV" --no-capture-output python train.py "${train_args[@]}"
