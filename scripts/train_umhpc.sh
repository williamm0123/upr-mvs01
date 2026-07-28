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
# conda 的 activate.d 脚本 (zzz_cusparselt.sh 等) 会读取 LD_LIBRARY_PATH 这类
# 未必存在的变量; 在 set -u 下会直接 "unbound variable" 退出。仅在激活期间
# 关闭 nounset, 激活完成后立刻恢复严格模式。
set +u
source ~/.bashrc
conda activate uprmvs
set -u

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH=/scr/user/qinglong/projects/upr-mvs01:/scr/user/qinglong/projects/upr-mvs01/models:/scr/user/qinglong/projects/upr-mvs01/models/Depth-Anything-3/src
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
# 缓解显存碎片 (与单卡脚本保持一致)。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# NCCL flight recorder: 集合通信超时时把卡住的 op 和调用栈打进日志, 否则只能
# 看到 "Stack trace of the failed collective not found"。
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}

# 先在单进程中幂等地补齐缓存。这样先验构建若 OOM/报错会直接显示原始异常，
# 也不会让其他 DDP rank 在 NCCL barrier 中等待并只报告 connection reset。
python train.py \
  --profile umhpc \
  --gpus 1 \
  --ddp off \
  --num-views 5 \
  --build-priors only

# 缓存完整后才启动 DDP；训练进程只额外加载冻结的 DINOv3（SPRE），绝不加载 VGGT/DA3。
python train.py \
  --profile umhpc \
  --gpus 2 \
  --ddp on \
  --batch-size 2 \
  --num-views 5 \
  --num-workers 16 \
  --lr 3.0e-4 \
  --warmup-steps 1000 \
  --amp on \
  --spre on \
  --steps 0 \
  --build-priors skip \
  --resume auto \
  --name uprmvs01
