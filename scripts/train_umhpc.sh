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

python train.py \
  --profile umhpc \
  --gpus 2 \
  --ddp on \
  --batch-size 3 \
  --num-views 5 \
  --num-workers 16 \
  --lr 1e-4 \
  --warmup-steps 1000 \
  --amp on \
  --steps 0 \
  --build-priors skip \
  --name uprmvs01
