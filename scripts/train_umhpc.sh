#!/bin/bash -l
#SBATCH --job-name=uprmvs01
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py
