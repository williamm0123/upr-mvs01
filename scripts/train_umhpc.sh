#!/bin/bash -l
#SBATCH --job-name=uprmvs01
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /scr/user/qinglong/projects/upr-mvs01
source ~/.bashrc
conda activate mvs

export UPRMVS_PROFILE=umhpc
export PYTHONPATH=/scr/user/qinglong/projects/upr-mvs01:/scr/user/qinglong/projects/upr-mvs01/models:/scr/user/qinglong/projects/upr-mvs01/models/Depth-Anything-3/src
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

# 先在单进程中幂等地补齐缓存。这样先验构建若 OOM/报错会直接显示原始异常，
# 也不会让其他 DDP rank 在 NCCL barrier 中等待并只报告 connection reset。
python train.py \
  --profile umhpc \
  --gpus 1 \
  --ddp off \
  --num-views 5 \
  --build-priors only

# 缓存完整后才启动 DDP；训练进程绝不加载 VGGT/DA3。
python train.py \
  --profile umhpc \
  --gpus 2 \
  --ddp on \
  --batch-size 4 \
  --num-views 5 \
  --num-workers 16 \
  --lr 2e-4 \
  --warmup-steps 1000 \
  --amp on \
  --steps 0 \
  --build-priors skip \
  --resume auto \
  --name uprmvs01
