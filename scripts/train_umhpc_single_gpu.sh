#!/bin/bash -l

set -euo pipefail


PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
# 直接指定 mvs 环境的解释器，不依赖当前 shell 的 conda activate/PATH。
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/mvs/bin/python}
TRAIN_PROFILE=${TRAIN_PROFILE:-umhpc}
RUN_NAME=${RUN_NAME:-uprmvs_1gpu_${SLURM_JOB_ID:-manual}}

# 核心训练参数（命令行会覆盖 TRAIN_PROFILE 中的同名参数）
BATCH_SIZE=${BATCH_SIZE:-4}       # 单卡 batch size；显存不足时保持 1
NUM_VIEWS=${NUM_VIEWS:-5}         # MVS 总视图数：1 个参考视图 + 4 个源视图
NUM_WORKERS=${NUM_WORKERS:-16}    # DataLoader 进程数；32 CPU 下建议 8~16
LEARNING_RATE=${LEARNING_RATE:-2e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
AMP=${AMP:-on}                    # on/off；A100 建议 on
STEPS=${STEPS:-0}                 # 0=使用 profile 默认值；测试可设 2

# 先验与跑通测试
BUILD_PRIORS=${BUILD_PRIORS:-auto}
# BUILD_PRIORS: auto=补齐缺失先验，force=全部重算，skip=要求缓存已存在，
#               only=只构建先验然后退出（换 val 列表后先跑一次这个）
SMOKE=${SMOKE:-0}                 # 1=合成数据跑通测试；0=真实数据训练
SMOKE_STEPS=${SMOKE_STEPS:-2}     # SMOKE=1 时执行的训练步数
# =============================================================================

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    echo "Override it with: PYTHON_BIN=/path/to/uprmvs/bin/python bash $0" >&2
    exit 1
fi

cd "$PROJECT_DIR"

export UPRMVS_PROFILE="$TRAIN_PROFILE"
# vggt 是 PROJECT_DIR/models 下的顶层 namespace package；不继承外部
# PYTHONPATH，避免误导入 /scr/user/qinglong/projects/vggt。
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTHONUNBUFFERED=1


echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) profile=$TRAIN_PROFILE ==="
nvidia-smi -L
echo "=== python=$PYTHON_BIN ==="
"$PYTHON_BIN" -c 'import importlib.util, sys, torch, huggingface_hub; print("python:", sys.executable); print("torch:", torch.__version__, torch.__file__); print("huggingface_hub:", huggingface_hub.__version__); print("vggt:", importlib.util.find_spec("vggt.models.vggt").origin)'
echo "=== batch=$BATCH_SIZE views=$NUM_VIEWS workers=$NUM_WORKERS lr=$LEARNING_RATE warmup=$WARMUP_STEPS amp=$AMP steps=$STEPS build_priors=$BUILD_PRIORS smoke=$SMOKE ==="

train_args=(
    --profile "$TRAIN_PROFILE"
    --gpus 1
    --ddp off
    --batch-size "$BATCH_SIZE"
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --lr "$LEARNING_RATE"
    --warmup-steps "$WARMUP_STEPS"
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

exec "$PYTHON_BIN" train.py "${train_args[@]}"
