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
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

cd /scr/user/qinglong/projects/upr-mvs01
source ~/.bashrc
conda activate mvs

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export OMP_NUM_THREADS=4

torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py
