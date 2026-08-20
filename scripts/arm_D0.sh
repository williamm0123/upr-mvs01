#!/bin/bash -l
#SBATCH --job-name=D0
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
# arm D0: 8.16 那次的完整配置, lr 2e-4 —— 确定性基线, 一切差值的分母
#
#     sbatch scripts/arm_D0.sh
#
# 单卡 A100, 12000 步, 约 3 小时。结果写 log/experiments/D0/。
# 三个 arm 的差异只有下面 "arm 配置" 那一段, 其余完全一致 —— 这是单变量
# 比较的前提。跑完用 python scripts/compare_arms.py --ref D0 看结果。
# =============================================================================

set -e

source ~/.bashrc
conda activate uprmvs

RUN_NAME=D0

# --- arm 配置 (三个脚本只有这里不同) ---
LR=2e-4
NUM_GLOBAL=40
NUM_LOCAL=8
GATE_LOCAL=on
BRANCH_PRIOR=on
VISIBILITY=on
STAGE1_WEIGHT=1.5
W_BRANCH=0.5
SPRE_BALANCE=on

# --- 以下所有 arm 共用, 不要单独改 ---
STEPS=12000
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
