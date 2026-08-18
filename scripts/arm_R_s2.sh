#!/bin/bash -l
#SBATCH --job-name=Rs2
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
# arm R-s2: 与 arm_R.sh 完全相同, 只换种子 (20260526 -> 20260626)。
#
#     sbatch scripts/arm_R_s2.sh
#
# 单卡 A100, 12000 步, 约 3 小时。结果写 log/experiments/R_s2/。
# 三个 arm 的差异只有下面 "arm 配置" 那一段, 其余完全一致 —— 这是单变量
# 比较的前提。跑完用 python scripts/compare_arms.py --ref D0 看结果。
# =============================================================================

set -e

source ~/.bashrc
conda activate uprmvs

RUN_NAME=R_s2

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
STEPS=12000
SEED=20260626
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
    --resume auto \
    --build-priors skip
