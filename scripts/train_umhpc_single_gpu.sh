#!/bin/bash -l
#SBATCH --job-name=uprmvs
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# =============================================================================
# 消融 arm 全部由环境变量驱动, 代码不改。提交示例:
#
#   sbatch --job-name=L --export=ALL,RUN_NAME=L,LR=3e-4 \
#          scripts/train_umhpc_single_gpu.sh
#
# 每个 arm 写自己的 log/experiments/$RUN_NAME/{model,tensorboard}, 所以可以
# 并行提交而不互相覆盖 checkpoint。
#
# 时限: 12k 步约 3.5 小时, 默认 --time=06:00:00 够用。跑满 30k 需要约 8.7 小时,
# 提交时覆盖: sbatch --time=12:00:00 --qos=long ...
# (0817 那次就是 8 小时限额 + 30k, 会在 ~27.6k 步被杀掉。)
# =============================================================================

set -e

source ~/.bashrc
conda activate uprmvs

RUN_NAME=${RUN_NAME:?RUN_NAME is required —— 每个 arm 必须有独立名字}

# --- 训练规模 ---
STEPS=${STEPS:-12000}
LR=${LR:-3e-4}
SEED=${SEED:-20260526}
NUM_WORKERS=${NUM_WORKERS:-16}
VAL_INTERVAL=${VAL_INTERVAL:-500}
# 训练 batch: 消融期间必须全 arm 一致 (2 = 7.29/8.16 用的值), 否则 arm 之间不可比。
# 显存实测 0.57 + B*(0.18 + 21.9*Mpx) GiB, 640x896 下 B=2 约 26 GiB。跑最终 30k
# 时没有跨 arm 比较的顾虑, 可以按卡的余量提上去 (40G 卡上 B=3 约 38.8 GiB, 偏满)。
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_VIEWS=${NUM_VIEWS:-5}
# 验证只跑前向且在 no_grad 下, batch 大小不影响任何指标 (按像素加权求和),
# 纯粹是拿空闲显存换时间 —— val 占一个 arm 约四分之一的墙钟。
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-6}

# --- 消融开关 (默认 = 8.16 那次的配置, 也就是 D0) ---
NUM_GLOBAL=${NUM_GLOBAL:-40}
NUM_LOCAL=${NUM_LOCAL:-8}
GATE_LOCAL=${GATE_LOCAL:-on}
BRANCH_PRIOR=${BRANCH_PRIOR:-on}
VISIBILITY=${VISIBILITY:-on}
STAGE1_WEIGHT=${STAGE1_WEIGHT:-1.5}
W_BRANCH=${W_BRANCH:-0.5}
SPRE_BALANCE=${SPRE_BALANCE:-on}

# --- 运行控制 ---
# auto: 被 slurm 杀掉后重排队能接着跑 (checkpoint 已按 run 隔离, 不会串)
RESUME=${RESUME:-auto}
# 并行提交时绝不能让多个任务同时重建先验缓存
BUILD_PRIORS=${BUILD_PRIORS:-skip}
EXPECT_SHA=${EXPECT_SHA:-}

echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) run=$RUN_NAME ==="
nvidia-smi -L
echo "=== git: $(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="
echo "=== steps=$STEPS lr=$LR seed=$SEED batch=$BATCH_SIZE views=$NUM_VIEWS val_batch=$VAL_BATCH_SIZE global/local=$NUM_GLOBAL/$NUM_LOCAL \
gate=$GATE_LOCAL branch_prior=$BRANCH_PRIOR vis=$VISIBILITY \
stage1=$STAGE1_WEIGHT w_branch=$W_BRANCH spre_balance=$SPRE_BALANCE \
resume=$RESUME priors=$BUILD_PRIORS ==="

train_args=(
    --profile umhpc
    --gpus 1
    --ddp off
    --name "$RUN_NAME"
    --steps "$STEPS"
    --lr "$LR"
    --seed "$SEED"
    --deterministic
    --num-workers "$NUM_WORKERS"
    --batch-size "$BATCH_SIZE"
    --num-views "$NUM_VIEWS"
    --val-batch-size "$VAL_BATCH_SIZE"
    --val-interval "$VAL_INTERVAL"
    --num-global "$NUM_GLOBAL"
    --num-local "$NUM_LOCAL"
    --gate-local "$GATE_LOCAL"
    --branch-prior "$BRANCH_PRIOR"
    --visibility "$VISIBILITY"
    --stage1-weight "$STAGE1_WEIGHT"
    --w-branch "$W_BRANCH"
    --spre-balance-corrupt "$SPRE_BALANCE"
    --spre on
    --no-clean-lists
    --resume "$RESUME"
    --build-priors "$BUILD_PRIORS"
)
if [[ -n "$EXPECT_SHA" ]]; then
    train_args+=(--expect-sha "$EXPECT_SHA")
fi

exec python -u train.py "${train_args[@]}"
