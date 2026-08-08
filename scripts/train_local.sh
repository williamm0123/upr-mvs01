#!/bin/bash -l

set -euo pipefail

# PROJECT_DIR / PYTHON_BIN / PYTHONPATH / UPRMVS_MACHINE 都来自 _common.sh，
# 路径不再写死（这个脚本原来钉在 /home/william/project/uprmvs01）。
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU_ID=${GPU_ID:-0}
RUN_NAME=${RUN_NAME:-uprmvs_local}

# RTX 5060 Ti 16GB 的保守默认值。需要时都可以在命令前用环境变量覆盖。
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_VIEWS=${NUM_VIEWS:-3}
NUM_WORKERS=${NUM_WORKERS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
AMP=${AMP:-on}
STEPS=${STEPS:-0}                       # 0 = 使用 local profile 的 max_steps
RESUME=${RESUME:-auto}
SPRE=${SPRE:-on}                        # on/off：DINOv3 先验可靠度头（SPRE）

# 正式训练前在独立 Python 进程中补齐 prior，避免 VGGT/DA3 占用的显存残留
# 到训练模型初始化阶段。已有完整缓存时检查会很快结束。
PRECOMPUTE_PRIORS=${PRECOMPUTE_PRIORS:-1}

# 快速验证模型、loss、反向传播和 checkpoint；不会读取真实数据或 prior。
# 用法：SMOKE=1 bash scripts/train_local.sh
SMOKE=${SMOKE:-0}
SMOKE_STEPS=${SMOKE_STEPS:-100}

# 其余环境变量来自 _common.sh；这里只加本脚本特有的。
export UPRMVS_PROFILE=local
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

uprmvs_env_banner
echo "=== local training: GPU=$GPU_ID batch=$BATCH_SIZE views=$NUM_VIEWS workers=$NUM_WORKERS amp=$AMP ==="

common_args=(
    --profile local
    --devices "$GPU_ID"
    --ddp off
    --batch-size "$BATCH_SIZE"
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --lr "$LEARNING_RATE"
    --warmup-steps "$WARMUP_STEPS"
    --amp "$AMP"
    --spre "$SPRE"
    --name "$RUN_NAME"
)

case "$SMOKE" in
    1|true|TRUE|yes|YES)
        exec "$PYTHON_BIN" train.py \
            "${common_args[@]}" \
            --smoke \
            --smoke-steps "$SMOKE_STEPS" \
            --build-priors skip \
            --resume off
        ;;
    0|false|FALSE|no|NO)
        ;;
    *)
        echo "SMOKE must be 0/1, true/false, or yes/no; got: $SMOKE" >&2
        exit 2
        ;;
esac

case "$PRECOMPUTE_PRIORS" in
    1|true|TRUE|yes|YES)
        "$PYTHON_BIN" train.py \
            --profile local \
            --devices "$GPU_ID" \
            --ddp off \
            --num-views "$NUM_VIEWS" \
            --build-priors only
        ;;
    0|false|FALSE|no|NO)
        ;;
    *)
        echo "PRECOMPUTE_PRIORS must be 0/1, true/false, or yes/no; got: $PRECOMPUTE_PRIORS" >&2
        exit 2
        ;;
esac

exec "$PYTHON_BIN" train.py \
    "${common_args[@]}" \
    --steps "$STEPS" \
    --build-priors skip \
    --resume "$RESUME"
