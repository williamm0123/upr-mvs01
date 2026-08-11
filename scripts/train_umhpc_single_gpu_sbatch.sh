#!/bin/bash -l
#SBATCH --job-name=uprmvs01-1gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

# SLURM 会把本脚本复制到 spool 目录后再执行，因此不能依赖 BASH_SOURCE 自动
# 推导 checkout。默认路径与 train_umhpc.sh 保持一致；集群路径不同时可在提交时用
#   sbatch --export=ALL,PROJECT_DIR=/path/to/upr-mvs01 scripts/train_umhpc_single_gpu_sbatch.sh
# 覆盖。
PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_umhpc_single_gpu.sh"

if [[ ! -x "$TRAIN_SCRIPT" ]]; then
    echo "找不到可执行的单卡训练脚本: $TRAIN_SCRIPT" >&2
    echo "请通过 sbatch --export=ALL,PROJECT_DIR=/path/to/upr-mvs01 ... 指定正确路径。" >&2
    exit 1
fi

export PROJECT_DIR

# 训练参数全部由 train_umhpc_single_gpu.sh 读取，因此可沿用相同的环境变量。
# 示例：
#   sbatch --export=ALL,BATCH_SIZE=1,STEPS=30000,TRAIN_PROFILE=umhpc \
#     scripts/train_umhpc_single_gpu_sbatch.sh
exec "$TRAIN_SCRIPT"
