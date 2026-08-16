#!/bin/bash -l

# 在已经分配到 GPU 的 UMHPC interactive shell 中直接运行。
# 本脚本不包含 sbatch/salloc/srun，也不会申请任何资源。

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
# 直接使用 uprmvs 环境，避免依赖 interactive shell 是否执行过 conda activate。
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/uprmvs/bin/python}
TRAIN_PROFILE=${TRAIN_PROFILE:-umhpc}
RUN_NAME=${RUN_NAME:-uprmvs_interactive_${SLURM_JOB_ID:-manual}}

# 训练参数；均可在命令前通过同名环境变量覆盖。
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_VIEWS=${NUM_VIEWS:-5}
NUM_WORKERS=${NUM_WORKERS:-16}
LEARNING_RATE=${LEARNING_RATE:-3.0e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-500}
AMP=${AMP:-on}
STEPS=${STEPS:-0}                       # 0 = 使用 umhpc profile 默认训练步数

DINO_MODE=${DINO_MODE:-all_view}        # off / all_view / ref_only
FEED_FPN=${FEED_FPN:-on}                # on / off
RELIABILITY=${RELIABILITY:-spre}        # cached / edge / spre
# 2026-08-16: prior 缓存已全部标尺 (29302 个, 未标尺 0, 尺度离群 0),
# *_clean.txt / exclude_*.csv 是当初为了绕开 8.35% 未标尺样本做的剔除表,
# 现在留着只会白少 10 个 train scan (79->69) 和 294 个样本。除非要复现旧
# 实验, 否则保持 off。
CLEAN_LISTS=${CLEAN_LISTS:-off}

RESUME=${RESUME:-off}                   # auto = 从 log/model/latest.pth 续训
# 缓存已完整; auto 只是核验一遍 (models/pre_prior.py 提速后约 5 秒,
# 若集群上还是旧版会读满 98GB/9 分钟, 那就用 skip)。
BUILD_PRIORS=${BUILD_PRIORS:-auto}       # auto / force / skip / only
SMOKE=${SMOKE:-0}
SMOKE_STEPS=${SMOKE_STEPS:-2}

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

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE="$TRAIN_PROFILE"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

case "$DINO_MODE:$FEED_FPN:$RELIABILITY" in
    off:*:spre)
        echo "Invalid config: RELIABILITY=spre requires DINO_MODE=all_view/ref_only" >&2
        exit 2
        ;;
    off:on:*)
        echo "Invalid config: DINO_MODE=off requires FEED_FPN=off" >&2
        exit 2
        ;;
    all_view:off:cached|all_view:off:edge)
        echo "Invalid config: DINO is enabled but neither FPN nor SPRE consumes it" >&2
        exit 2
        ;;
esac

if [[ "$RELIABILITY" == "spre" ]]; then
    SPRE=on
else
    SPRE=off
fi

echo "=== UMHPC interactive single-GPU training ==="
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) project=$PROJECT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
nvidia-smi -L
echo "python=$PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys, torch; print("python:", sys.executable); print("torch:", torch.__version__); print("cuda available:", torch.cuda.is_available()); print("visible GPUs:", torch.cuda.device_count())'

if ! "$PYTHON_BIN" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)'; then
    echo "No CUDA GPU is visible. Enter a GPU interactive allocation before running this script." >&2
    exit 1
fi

echo "batch=$BATCH_SIZE views=$NUM_VIEWS workers=$NUM_WORKERS lr=$LEARNING_RATE"
echo "warmup=$WARMUP_STEPS val_interval=$VAL_INTERVAL amp=$AMP steps=$STEPS"
echo "dino=$DINO_MODE feed_fpn=$FEED_FPN reliability=$RELIABILITY"
echo "resume=$RESUME build_priors=$BUILD_PRIORS smoke=$SMOKE"

train_args=(
    --profile "$TRAIN_PROFILE"
    --gpus 1
    --ddp off
    --batch-size "$BATCH_SIZE"
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --lr "$LEARNING_RATE"
    --warmup-steps "$WARMUP_STEPS"
    --val-interval "$VAL_INTERVAL"
    --amp "$AMP"
    --dino-mode "$DINO_MODE"
    --feed-fpn "$FEED_FPN"
    --reliability "$RELIABILITY"
    --spre "$SPRE"
    --name "$RUN_NAME"
)

case "$CLEAN_LISTS" in
    on|ON|1|true|TRUE|yes|YES) ;;
    off|OFF|0|false|FALSE|no|NO) train_args+=(--no-clean-lists) ;;
    *)
        echo "CLEAN_LISTS must be on/off; got: $CLEAN_LISTS" >&2
        exit 2
        ;;
esac

case "$SMOKE" in
    1|true|TRUE|yes|YES)
        train_args+=(--smoke --smoke-steps "$SMOKE_STEPS" --build-priors skip)
        ;;
    0|false|FALSE|no|NO)
        train_args+=(--steps "$STEPS" --build-priors "$BUILD_PRIORS" --resume "$RESUME")
        ;;
    *)
        echo "SMOKE must be 0/1, true/false, or yes/no; got: $SMOKE" >&2
        exit 2
        ;;
esac

exec "$PYTHON_BIN" train.py "${train_args[@]}"
