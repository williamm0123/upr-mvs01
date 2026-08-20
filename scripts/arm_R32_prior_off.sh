#!/bin/bash -l
#SBATCH --job-name=R32_prior_off
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
# arm R32_prior_off   —— 第二轮 (bf16 + 30k LR horizon)
#
# 严格无先验对照 —— 课题主张的直接证据。
# prior/conf 置零后 local 分支退化成 guard 范围内的半偏移均匀网格 (一条不含
# 先验的 48 候选轴); meta 的 dist_prior/conf/edge 显式中性化 (只置零 prior
# 不够: dist_prior=(hypos-0)/gi 就成了归一化深度的另一种写法); branch prior
# 与 branch loss 关闭; SPRE 头不参与前向, 因而不向共享 DINO/SVA 特征回传梯度。
# 模块**照常构造**, RNG 流与 prior=on 完全一致 —— 与 R32_base_s1 是配对实验。
# 它与 base 的差值 = 先验通路的净贡献。共位 oracle 的上界是 -0.54mm。
#
#     sbatch scripts/arm_R32_prior_off.sh
#
# 与第一轮的两处口径变化, 所有 arm 一致:
#   * --amp-dtype bf16         第一轮八个 arm 里五个死于 stage4 前向 fp16 溢出
#   * --lr-schedule-steps 30000  12k 只是**停止步数**, 退火轨迹仍是 30k 的。
#                              第一轮 12k arm 在 8k-12k 已经退火到底 (lr 1.6e-5
#                              vs 30k run 同窗口的 2.3e-4), 既不能跟历史横比,
#                              也不能无跳变续训。
# 因此第二轮的数**不能**跟第一轮的数直接比, 只能 arm 之间互比。
#
# 单卡 A100, 12000 步, 约 3 小时。结果写 log/experiments/R32_prior_off/。
# 比较: python scripts/compare_arms.py --ref R32_base_s1
# =============================================================================

set -e

source ~/.bashrc
conda activate uprmvs

RUN_NAME=R32_prior_off

# --- arm 配置 (各脚本只有这里不同) ---
SEED=20260526
RANGE_MIN_GI=0.66,0.20,0.05
PRIOR=off
EXTRA=()

# --- 以下所有 arm 共用, 不要单独改 ---
STEPS=12000
LR_HORIZON=30000
LR=3e-4
AMP_DTYPE=bf16
NUM_GLOBAL=32
NUM_LOCAL=16
GATE_LOCAL=off
BRANCH_PRIOR=off
VISIBILITY=off
STAGE1_WEIGHT=0.5
W_BRANCH=0
SPRE_BALANCE=on
BATCH_SIZE=2
NUM_VIEWS=5
VAL_BATCH_SIZE=6
NUM_WORKERS=16
VAL_INTERVAL=500
LOG_INTERVAL=10

GIT_SHA=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "工作树有未提交的改动 —— 这一版跑出来的结果对不上任何 commit" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

RUN_DIR="log/experiments/$RUN_NAME"
if [[ -e "$RUN_DIR" ]]; then
    ARCHIVE="log/experiments/_archive/${RUN_NAME}_$(date -u +%Y%m%d_%H%M%S)"
    mkdir -p "$(dirname "$ARCHIVE")"
    mv "$RUN_DIR" "$ARCHIVE"
    echo "=== 上一轮已归档 (未删除): $RUN_DIR -> $ARCHIVE ==="
fi

echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) run=$RUN_NAME ==="
echo "=== git=${GIT_SHA:0:12} (clean) ==="
nvidia-smi -L
echo "=== steps=$STEPS horizon=$LR_HORIZON lr=$LR amp=$AMP_DTYPE seed=$SEED \
global/local=$NUM_GLOBAL/$NUM_LOCAL range_min_gi=$RANGE_MIN_GI prior=$PRIOR \
extra=${EXTRA[*]} ==="

exec python -u train.py \
    --profile umhpc \
    --gpus 1 \
    --ddp off \
    --name "$RUN_NAME" \
    --steps "$STEPS" \
    --lr-schedule-steps "$LR_HORIZON" \
    --lr "$LR" \
    --amp-dtype "$AMP_DTYPE" \
    --seed "$SEED" \
    --deterministic \
    --batch-size "$BATCH_SIZE" \
    --num-views "$NUM_VIEWS" \
    --num-workers "$NUM_WORKERS" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --val-interval "$VAL_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --num-global "$NUM_GLOBAL" \
    --num-local "$NUM_LOCAL" \
    --range-min-gi "$RANGE_MIN_GI" \
    --gate-local "$GATE_LOCAL" \
    --branch-prior "$BRANCH_PRIOR" \
    --visibility "$VISIBILITY" \
    --stage1-weight "$STAGE1_WEIGHT" \
    --w-branch "$W_BRANCH" \
    --spre-balance-corrupt "$SPRE_BALANCE" \
    --prior "$PRIOR" \
    --spre on \
    --no-clean-lists \
    --resume off \
    --build-priors skip \
    "${EXTRA[@]}"
