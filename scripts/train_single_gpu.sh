#!/bin/bash -l

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/william/project/uprmvs01}
CONDA_ENV=${CONDA_ENV:-uprmvs}
GPU_ID=${GPU_ID:-0}
RUN_NAME=${RUN_NAME:-uprmvs_local}
BUILD_PRIORS=${BUILD_PRIORS:-auto}
STEPS=${STEPS:-0}

cd "$PROJECT_DIR"

export UPRMVS_MACHINE=${UPRMVS_MACHINE:-ubuntu}
export UPRMVS_PROFILE=local
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models/vggt:$PROJECT_DIR/models/Depth-Anything-3/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

conda run -n "$CONDA_ENV" --no-capture-output python train.py \
    --profile local \
    --devices "$GPU_ID" \
    --ddp off \
    --steps "$STEPS" \
    --build-priors "$BUILD_PRIORS" \
    --name "$RUN_NAME"
