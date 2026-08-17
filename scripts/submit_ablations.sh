#!/bin/bash
# =============================================================================
# 一次提交全部消融 arm。在 HPC 的项目根目录下运行:
#
#     bash scripts/submit_ablations.sh
#
# 前置: 工作树必须干净且已 checkout 到要跑的 commit —— 脚本会把当前 SHA 传给
# 每个任务, 训练启动时校验。排队期间代码被改动过, 任务会直接退出而不是跑出
# 一份不知道对应哪版代码的结果。
#
# 每个 arm 12000 步 (约 3 小时), 写自己的 log/experiments/<RUN_NAME>/。
# 结果比较: python scripts/compare_arms.py
# =============================================================================
set -e
cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain)" ]]; then
    echo "工作树 dirty —— 先提交或 stash, 否则跑出来的结果对不上任何 commit" >&2
    git status --short >&2
    exit 1
fi
SHA=$(git rev-parse HEAD)
STEPS=${STEPS:-12000}
echo "=== 提交消融实验 @ ${SHA:0:12}, 每个 arm $STEPS 步 ==="

# 先验缓存必须是完整的: 所有 arm 都用 --build-priors skip, 不能让 9 个任务
# 同时去重建缓存。这里先确认它非空。
if [[ ! -d log/prior_cache ]] || [[ -z "$(ls -A log/prior_cache 2>/dev/null)" ]]; then
    echo "log/prior_cache 为空 —— 先单独跑一次 --build-priors only" >&2
    exit 1
fi

submit () {
    local name="$1"; shift
    local vars="$*"
    local jid
    jid=$(sbatch --parsable --job-name="$name" \
        --export="ALL,RUN_NAME=$name,STEPS=$STEPS,EXPECT_SHA=$SHA${vars:+,$vars}" \
        scripts/train_umhpc_single_gpu.sh)
    printf '  %-12s job %s\n' "$name" "$jid"
}

# --- 基准与学习率 ---
# D0: 8.16 那次的完整配置 (脚本默认值就是它), 只是现在是确定性的
submit D0  "LR=2e-4"
# L : 只改学习率, 量化 lr 单独的贡献
submit L   "LR=3e-4"

# --- 在 L 之上的单变量拆分 (lr 已确认, 以它为基准逐项回退) ---
submit LG  "LR=3e-4,NUM_GLOBAL=32,NUM_LOCAL=16"
submit LB  "LR=3e-4,BRANCH_PRIOR=off"
submit LS  "LR=3e-4,STAGE1_WEIGHT=0.5,W_BRANCH=0"
submit LV  "LR=3e-4,VISIBILITY=off"

# --- 恢复候选: 六项一起回退 ---
R_VARS="LR=3e-4,NUM_GLOBAL=32,NUM_LOCAL=16,GATE_LOCAL=off,BRANCH_PRIOR=off,VISIBILITY=off,STAGE1_WEIGHT=0.5,W_BRANCH=0"
submit R   "$R_VARS"
# R_s2: 同配置换 seed —— 所有判读阈值都建立在噪声估计上, 而现有的 0.048mm
# 是从旧的无种子设置里量出来的, 必须在新设置下重新量一次
submit R_s2 "$R_VARS,SEED=7"
# R_nospre: R 没恢复时的备选 (spre_balance 会经 q 改变 local 窗宽)
submit R_nospre "$R_VARS,SPRE_BALANCE=off"

echo
echo "全部提交完毕。查看: squeue -u \$USER"
echo "结果比较: python scripts/compare_arms.py --ref D0"
