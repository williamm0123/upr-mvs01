#!/bin/bash -l
#SBATCH --job-name=R_s2
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
# arm R_s2
#
# R 换一个种子。所有判读阈值 (<0.05 不解释 / >0.15 确认) 都建立在噪声估计上,
# 而现有的 0.048mm 是从旧的无种子设置里量出来的, 必须在新设置下重新量一次。
# R 与 R_s2 的窗口均值之差就是新的噪声地板。
#
#     sbatch scripts/arm_R_s2.sh
#
# 单卡 A100, 12000 步, 约 3 小时。结果写 log/experiments/R_s2/。
# 全部 arm 的差异只有下面 "arm 配置" 那一段, 其余逐字相同 —— 这是单变量比较的前提。
# 比较: python scripts/compare_arms.py --ref D0
# =============================================================================

set -e

source ~/.bashrc
conda activate uprmvs

RUN_NAME=R_s2

# --- arm 配置 (各脚本只有这里不同) ---
LR=3e-4
NUM_GLOBAL=32
NUM_LOCAL=16
GATE_LOCAL=off
BRANCH_PRIOR=off
VISIBILITY=off
STAGE1_WEIGHT=0.5
W_BRANCH=0
SEED=20260626

# --- 以下所有 arm 共用, 不要单独改 ---
STEPS=12000
SPRE_BALANCE=on
BATCH_SIZE=2          # 消融期间必须全 arm 一致, 否则 arm 之间不可比
NUM_VIEWS=5
VAL_BATCH_SIZE=6      # 验证只前向且在 no_grad 下, 不影响任何指标, 纯省时间
NUM_WORKERS=16
VAL_INTERVAL=500
# 10 而不是默认 50: 梯度范数的尖峰会被 50 步的窗口均值抹平, 而 grad/norm_unclipped、
# grad/amp_scale、grad/nonfinite_frac 这三条正是 L 发散的提前量。全 arm 统一取 10,
# 否则 train/* 曲线的平滑程度不同, 诊断量之间没法横向比。
LOG_INTERVAL=10

# 代码版本: 只记录 + 拦"改过但没提交的跟踪文件"。未跟踪的 slurm 输出不算。
GIT_SHA=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "工作树有未提交的改动 —— 这一版跑出来的结果对不上任何 commit" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

# --- 归档上一轮的同名 run (--resume off 的必要配套) -------------------------
# --resume off 只让训练从 step 0 开始, 它不会动 log/experiments/<run>/ 里已有的
# 文件。不归档的话: best_metric 从 inf 起算会覆盖旧 best.pth; 而 tensorboard 把
# 同一个 run 下所有事件文件按 step 合并 —— 新旧两轮的 step 0-12000 会叠在一起,
# compare_arms.py 读出来就是两轮的混合。移走而不是删掉。
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
    --log-interval "$LOG_INTERVAL" \
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
    --resume off \
    --build-priors skip
