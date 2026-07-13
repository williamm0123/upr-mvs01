#!/bin/bash -l
#SBATCH --job-name=uprmvs01
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

set -euo pipefail

PROJECT_DIR=/scr/user/qinglong/projects/upr-mvs01
CONDA_ENV=mvs
RUN_NAME=${RUN_NAME:-uprmvs_${SLURM_JOB_ID:-manual}}

cd "$PROJECT_DIR"
mkdir -p log

source ~/.bashrc
conda activate "$CONDA_ENV"

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models/vggt:$PROJECT_DIR/models/Depth-Anything-3/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NPROC=$(nvidia-smi -L | wc -l)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "=== job=$SLURM_JOB_ID host=$(hostname) gpus=$NPROC port=$MASTER_PORT ==="
nvidia-smi -L

# train.py owns process spawning. Do not wrap it in torchrun.
python train.py \
    --profile umhpc \
    --gpus "$NPROC" \
    --ddp on \
    --master-port "$MASTER_PORT" \
    --build-priors "${BUILD_PRIORS:-auto}" \
    --name "$RUN_NAME"
