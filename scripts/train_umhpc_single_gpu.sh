#!/bin/bash -l
#SBATCH --job-name=uprmvs1g
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

# 必须激活环境，不能直接调用环境里的 python。激活脚本会设置 PyTorch
# 加载 libcusparseLt.so.0 所需的动态库路径。
source ~/.bashrc
conda activate uprmvs

exec python -u train.py \
    --profile umhpc \
    --gpus 1 \
    --ddp off \
    --num-workers 16 \
    --spre on \
    --no-clean-lists \
    --resume off
