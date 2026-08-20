#!/bin/bash -l
#SBATCH --job-name=R30s1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --time=12:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# ===========================================================================
# arm R30-s1: 用 arm_R 的配置跑满 30000 步, seed=20260526 (两个 seed 中的第 1 个)
#
#     sbatch scripts/arm_R30_s1.sh
#
# 与 arm_R.sh 的差别只有三处: STEPS 30000、seed、以及随之而来的时限。其余全部
# 逐字一致 —— 12k 的 arm 和 30k 的复现之间必须只差步数, 否则两张表接不起来。
#
# 时限: 30k 约 8.7 小时。必须 --time=12:00:00 --qos=long。旧脚本的 8 小时上限
# 会在 ~27.6k 步被 slurm 杀掉; 现在 latest.pth 存了 scaler/RNG、resume 也会跳过
# 本 epoch 已消费的 batch, 被杀之后 sbatch 同一个脚本即可精确续上, 不用重来。
#
# 判读: 30k 用最后 10-11 个 val 点的均值, 不是 8k-12k 那个窗口, 也不是 best 单点。
# 两个 seed 跑完用 python scripts/compare_arms.py --lo 25000 --hi 30000 一起看。
#
# 卡: A100-SXM4-80GB。batch 仍然锁死 2 —— 显存有的是余量, 但 batch 一动, 与 12k
# 三个 arm 和 7.29/8.16 的历史成绩就全部不可比了。省时间要调的是 VAL_BATCH_SIZE。
# ===========================================================================

set -e

source ~/.bashrc
conda activate uprmvs

# 长跑更容易碎在 reserved-but-unallocated 上。纯分配器策略, 不改任何数值。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_NAME=R30_s1

# --- arm 配置 (三个脚本只有这里不同) ---
LR=3e-4
NUM_GLOBAL=32
NUM_LOCAL=16
GATE_LOCAL=off
BRANCH_PRIOR=off
VISIBILITY=off
STAGE1_WEIGHT=0.5
W_BRANCH=0
SPRE_BALANCE=on

# --- 以下所有 arm 共用, 不要单独改 ---
STEPS=30000
SEED=20260526
BATCH_SIZE=2          # 消融期间必须全 arm 一致, 否则 arm 之间不可比
NUM_VIEWS=5
VAL_BATCH_SIZE=6      # 验证只前向且在 no_grad 下, 不影响任何指标, 纯省时间
NUM_WORKERS=16
VAL_INTERVAL=500

# 代码版本: 只记录 + 拦"改过但没提交的跟踪文件"。未跟踪的 slurm 输出不算。
GIT_SHA=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "工作树有未提交的改动 —— 这一版跑出来的结果对不上任何 commit" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) run=$RUN_NAME ==="
echo "=== git=${GIT_SHA:0:12} (clean) ==="
nvidia-smi -L
echo "=== steps=$STEPS lr=$LR seed=$SEED batch=$BATCH_SIZE views=$NUM_VIEWS \
global/local=$NUM_GLOBAL/$NUM_LOCAL gate=$GATE_LOCAL branch_prior=$BRANCH_PRIOR \
vis=$VISIBILITY stage1=$STAGE1_WEIGHT w_branch=$W_BRANCH ==="

exec python -u train.py \
    --profile umhpc \
    --gpus 1 \
    --ddp off \
    --name "$RUN_NAME" \
    --steps "$STEPS" \
    --lr "$LR" \
    --seed "$SEED" \
    --deterministic \
    --batch-size "$BATCH_SIZE" \
    --num-views "$NUM_VIEWS" \
    --num-workers "$NUM_WORKERS" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --val-interval "$VAL_INTERVAL" \
    --num-global "$NUM_GLOBAL" \
    --num-local "$NUM_LOCAL" \
    --gate-local "$GATE_LOCAL" \
    --branch-prior "$BRANCH_PRIOR" \
    --visibility "$VISIBILITY" \
    --stage1-weight "$STAGE1_WEIGHT" \
    --w-branch "$W_BRANCH" \
    --spre-balance-corrupt "$SPRE_BALANCE" \
    --spre on \
    --no-clean-lists \
    --resume false \
    --build-priors skip
