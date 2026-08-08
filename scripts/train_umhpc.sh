#!/bin/bash -l
#SBATCH --job-name=uprmvs01
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
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

# =============================================================================
# 可覆盖参数（与 train_umhpc_single_gpu.sh 对齐，便于两边同步调整）
#
# !! 末级假设数 4 -> 8 (base/config.py num_depths_stage4)。stage 4 是全分辨率级、
#    显存占大头, 这一改把最大的那一项翻倍 (512x640 下 cost volume 体素
#    2.54M -> 3.86M, 约 1.5x)。OOM 退让顺序:
#      1) BATCH_SIZE 4 -> 2 -> 1
#      2) UPRMVS_MAX_SCALE=576 (砍掉 augment 里 640x832 / 640x896 两档)
#      3) 把 num_depths_stage4 改回 4 (等于放弃 P5 这个实验)
# =============================================================================
GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}        # 每卡 batch
NUM_VIEWS=${NUM_VIEWS:-5}
NUM_WORKERS=${NUM_WORKERS:-16}
LEARNING_RATE=${LEARNING_RATE:-3.0e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-500}
AMP=${AMP:-on}
STEPS=${STEPS:-0}                  # 0 = profile 默认 (umhpc: 30000)
SPRE=${SPRE:-on}
# FRESH=1 (默认): 训练开始前把 log/model/ 里已有的 .pth 整体挪到
# log/model.bak_<时间戳>/。挪走不是删除, 但本次训练绝对读不到任何旧权重。
# 之所以必须物理隔离而不是靠加载报错兜底: 末级 D4->D8 不改变 state_dict
# (Conv3d 权重是 [out,in,3,3,3], 与 D 无关), 旧 checkpoint 会干净地加载进来
# 然后训练一个不是你要的模型。
FRESH=${FRESH:-1}
# RESUME=auto 在 FRESH=1 之后是安全的: 目录已经被清空, auto 找不到东西, 从
# step 0 开始; 而 SLURM 把作业重排队时 job id 不变, preflight 会认出这是同一次
# 运行、跳过归档, auto 于是正确地续上本次训练自己写的 checkpoint —— 三天的作业
# 不会因为一次重排队而从头再来。真要完全禁用续跑就设 RESUME=off。
RESUME=${RESUME:-auto}
RUN_NAME=${RUN_NAME:-uprmvs01}

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

echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) gpus=$GPUS ==="
nvidia-smi -L

# 确认跑的是改动后的代码, 并（FRESH=1 时）把旧 checkpoint 挪开。DDP 下一个 rank
# 因为旧代码报错, 另一个会卡在 NCCL barrier 上直到 watchdog 超时, 所以这一步必须
# 在起 DDP 之前单进程做完。
preflight_args=()
[[ "$FRESH" == "1" ]] && preflight_args+=(--fresh --run-id "${SLURM_JOB_ID:-manual}")
python scripts/preflight.py "${preflight_args[@]}"

echo "=== gpus=$GPUS batch=$BATCH_SIZE views=$NUM_VIEWS workers=$NUM_WORKERS \
lr=$LEARNING_RATE warmup=$WARMUP_STEPS val_interval=$VAL_INTERVAL amp=$AMP \
steps=$STEPS spre=$SPRE fresh=$FRESH resume=$RESUME max_scale=${UPRMVS_MAX_SCALE:-none} ==="

# 先在单进程中幂等地补齐缓存。这样先验构建若 OOM/报错会直接显示原始异常，
# 也不会让其他 DDP rank 在 NCCL barrier 中等待并只报告 connection reset。
python train.py \
  --profile umhpc \
  --gpus 1 \
  --ddp off \
  --num-views "$NUM_VIEWS" \
  --build-priors only

# 缓存完整后才启动 DDP；训练进程只额外加载冻结的 DINOv3（SPRE），绝不加载 VGGT/DA3。
exec python train.py \
  --profile umhpc \
  --gpus "$GPUS" \
  --ddp on \
  --batch-size "$BATCH_SIZE" \
  --num-views "$NUM_VIEWS" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LEARNING_RATE" \
  --warmup-steps "$WARMUP_STEPS" \
  --val-interval "$VAL_INTERVAL" \
  --amp "$AMP" \
  --spre "$SPRE" \
  --steps "$STEPS" \
  --build-priors skip \
  --resume "$RESUME" \
  --name "$RUN_NAME"
