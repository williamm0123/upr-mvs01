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

cd /scr/user/qinglong/projects/upr-mvs01
source ~/.bashrc
conda activate uprmvs

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH=/scr/user/qinglong/projects/upr-mvs01:/scr/user/qinglong/projects/upr-mvs01/models:/scr/user/qinglong/projects/upr-mvs01/models/Depth-Anything-3/src
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 换过 val 列表(lists/dtu/val.txt)后, 首次需单进程预构建 val 先验缓存,
# 避免 DDP 下 rank0 长时间构建导致 NCCL barrier 超时:
#   python train.py --profile umhpc --gpus 1 --ddp off --build-priors only
python train.py \
  --profile umhpc \
  --gpus 2 \
  --ddp on \
  --batch-size 3 \
  --num-views 5 \
  --num-workers 16 \
  --lr 2e-4 \
  --warmup-steps 1000 \
  --amp on \
  --steps 0 \
  --build-priors auto \
  --resume auto \
  --name uprmvs01
