#!/bin/bash -l
#SBATCH --job-name=uprmvs_ddp2
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# UPRMVS 改造工单 v3 —— 双卡 A100-80GB, torchrun DDP。一个脚本跑三个 arm。
#
#   ARM=w0 sbatch scripts/sbatch_ddp2.sh     # W0-A 基线 R32_f10 -> 30k
#   ARM=w1 sbatch scripts/sbatch_ddp2.sh     # W1  Depth-core vNext -> 30k
#   ARM=w3 sbatch scripts/sbatch_ddp2.sh     # W3  UPRMVS-3D vNext -> 30k (从头)
#   ARM=w3b sbatch scripts/sbatch_ddp2.sh    # W3 + W3-B 源视图可见性监督
#
# 顺序是工单排的, 不要跳:
#   w0 先跑完 -> best.pth -> scripts/tail_posterior_stats.py 出 deployable 判据
#   -> 再决定 W2 做不做 -> w1 -> w3。W3 **从头训练**而不是在 W1 权重上续训,
#   为的是归因清楚。
#
# -----------------------------------------------------------------------------
# 关于批量: 默认 per-GPU batch = 1, **全局 batch = 2**, 与 f10 单卡 batch=2 的
# 全局批量一致。这样 30k 步的语义、lr 3e-4 和验证曲线都能直接和基线比, 双卡
# 只是把 wall-clock 砍一半。
#
# 80GB 的卡确实吃得下更多 (实测 0.57 + B x (0.18 + 21.9 x Mpx) GiB, 最大尺度
# 640x896 下 B=2 约 27GiB, B=4 约 55GiB), 但**把卡吃满不是目标**: 全局 batch=4
# 是另一个训练配置, lr 要重新标定, 30k 步也不再等价于基线的 30k 步。
#     PER_GPU_BATCH=2 ARM=w1 sbatch scripts/sbatch_ddp2.sh    # 需自行调 lr
#
# -----------------------------------------------------------------------------
# BatchNorm 的坑 —— 决定 batch 之前先读这一段。
#
# FPN 里是 nn.BatchNorm2d, 而且它看到的是 **B x V** 个样本 (fpn.py 把所有视角
# flatten 进 batch 维), 同时全网**故意没有用 SyncBN**。于是:
#
#     单卡 batch=2, V=5        -> BN 批量 10
#     双卡 per-GPU batch=1     -> BN 批量 5   (每张卡各算各的)
#     双卡 per-GPU batch=2     -> BN 批量 10  (与单卡基线一致, 但全局 batch=4)
#
# 也就是说**没有**哪个双卡配置能同时对齐"全局 batch"和"BN 批量"。结论:
#   * 与**历史**单卡 f10 数字的严格可比性, 双卡拿不到 —— 别指望;
#   * 工单要的可比性是 W0/W1/W3 三个 arm **互相之间**可比, 而 W0-A 本来就是
#     重新跑一遍拿新基准 (排期第 1 项)。所以只要三个 arm 用**同一个**
#     PER_GPU_BATCH, 比较就是成立的。
#   * 因此: 三个 arm 要么全用默认的 1, 要么全用 2 并且都重标 lr。
#     **不要**给不同 arm 用不同的 batch。
# =============================================================================

set -euo pipefail

ARM=${ARM:-w1}
PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

# conda 的 activate.d 脚本会读 LD_LIBRARY_PATH 这类未必存在的变量; set -u 下
# 会直接 unbound variable 退出。只在激活期间关掉 nounset。
set +u
source ~/.bashrc
conda activate uprmvs
set -u

# ---------------------------------------------------------------- arm 定义
# 三个 arm 只在这一段有差别; 下面的公共部分一个字都不要动, 否则就不是配对实验。
case "$ARM" in
  w0)
    RUN_NAME=${RUN_NAME:-R32_f10_30k}
    ARM_ARGS=(--axis-space legacy_depth --stage4-head expect)
    ;;
  w1)
    RUN_NAME=${RUN_NAME:-W1_vnext}
    ARM_ARGS=(
      --axis-space inverse                       # A 逆深度轴 + E 非均匀轴记账
      --axis-blend-steps "${AXIS_BLEND_STEPS:-2000}"
      --tau-stages "${TAU_STAGES:-0.98,0.95,0.92}"   # C pinball 目标覆盖率
      --rho-max "${RHO_MAX:-8.0}"                # B 半宽的倍率界 [h0/8, 8h0]
      --stage4-head map                          # D 硬 MAP + 逐候选残差
      --mode-window-stages "${MODE_WINDOW_STAGES:-2,2,1,2}"
      --spre-cascade on                          # F SPRE 累乘门贯穿四级
      --spre-gate-init "${SPRE_GATE_INIT:-1.0,0.60,0.35,0.20}"
      --w-range "${W_RANGE:-1.0}" --w-center "${W_CENTER:-0.2}"
      --w-residual "${W_RESIDUAL:-1.0}" --w-oor "${W_OOR:-0.1}"
    )
    ;;
  w3|w3b)
    RUN_NAME=${RUN_NAME:-W3_vnext}
    ARM_ARGS=(
      --axis-space inverse
      --axis-blend-steps "${AXIS_BLEND_STEPS:-2000}"
      --tau-stages "${TAU_STAGES:-0.98,0.95,0.92}"
      --rho-max "${RHO_MAX:-8.0}"
      --stage4-head map
      --mode-window-stages "${MODE_WINDOW_STAGES:-2,2,1,2}"
      --spre-cascade on
      --spre-gate-init "${SPRE_GATE_INIT:-1.0,0.60,0.35,0.20}"
      --w-range "${W_RANGE:-1.0}" --w-center "${W_CENTER:-0.2}"
      --w-residual "${W_RESIDUAL:-1.0}" --w-oor "${W_OOR:-0.1}"
      --geo-valid on                             # W3-A 逐假设几何有效聚合
      --conf-head on --w-conf "${W_CONF:-1.0}"   # W3-C 最终重建置信度
      --conf-tau-mm "${CONF_TAU_MM:-2.0}"
    )
    if [[ "$ARM" == "w3b" ]]; then
      RUN_NAME=${RUN_NAME_B:-W3_vnext_visB}
      # W3-B: 多标签可见性 + 源视图 GT 遮挡监督。dataset 每样本多读 N-1 张
      # PFM + mask, 所以 workers 给足。工单把它排在最后 —— 先有 A+C 的干净
      # 读数, 再看 B 加不加得动。
      ARM_ARGS+=(--visibility on --vis-mode sigmoid --vis-supervise on
                 --w-vis "${W_VIS:-0.1}" --delta-occ-mm "${DELTA_OCC_MM:-2.0}")
      NUM_WORKERS=${NUM_WORKERS:-12}
    fi
    ;;
  *)
    echo "ARM 只能是 w0 / w1 / w3 / w3b, 收到 '$ARM'" >&2; exit 2 ;;
esac

# ------------------------------------------------- 公共部分 (三个 arm 完全一致)
STEPS=${STEPS:-30000}
LR_HORIZON=${LR_HORIZON:-30000}
LR=${LR:-3e-4}
SEED=${SEED:-20260526}
AMP_DTYPE=bf16
NUM_GLOBAL=32
NUM_LOCAL=16
RANGE_MIN_GI=0.66,0.20,0.10      # inverse 下只作为 RangeController 的初始化值
GATE_LOCAL=off
BRANCH_PRIOR=off
STAGE1_WEIGHT=0.5
W_BRANCH=0
SPRE_BALANCE=on
PRIOR=on

PER_GPU_BATCH=${PER_GPU_BATCH:-1}
NPROC=${NPROC:-2}
NUM_VIEWS=${NUM_VIEWS:-5}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-6}
NUM_WORKERS=${NUM_WORKERS:-8}    # per-process; 2 进程 x 8 = 16 个 loader worker
VAL_INTERVAL=${VAL_INTERVAL:-500}
LOG_INTERVAL=${LOG_INTERVAL:-10}
# w3b 之外的 arm 不开 visibility (与 f10 基线一致)
VISIBILITY=${VISIBILITY:-off}
[[ "$ARM" == "w3b" ]] || ARM_ARGS+=(--visibility "$VISIBILITY")

# -------------------------------------------------------- 干净工作树是硬要求
# 跑出来的曲线必须对得上一个 commit, 否则几周后没人能说清那条线是哪版代码。
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
mkdir -p logs

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# NCCL flight recorder: 集合通信超时时把卡住的 op 和调用栈打进日志。
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}

echo "=================================================================="
echo " arm=$ARM  run=$RUN_NAME  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=${GIT_SHA:0:12} (clean)"
echo " nproc=$NPROC  per_gpu_batch=$PER_GPU_BATCH  global_batch=$((NPROC * PER_GPU_BATCH))"
echo " steps=$STEPS  horizon=$LR_HORIZON  lr=$LR  amp=$AMP_DTYPE  seed=$SEED"
echo " arm_args: ${ARM_ARGS[*]}"
echo "=================================================================="
nvidia-smi -L

exec torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" train.py \
    --profile umhpc \
    --ddp on \
    --name "$RUN_NAME" \
    --steps "$STEPS" \
    --lr-schedule-steps "$LR_HORIZON" \
    --lr "$LR" \
    --amp-dtype "$AMP_DTYPE" \
    --seed "$SEED" \
    --deterministic \
    --batch-size "$PER_GPU_BATCH" \
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
    --stage1-weight "$STAGE1_WEIGHT" \
    --w-branch "$W_BRANCH" \
    --spre-balance-corrupt "$SPRE_BALANCE" \
    --prior "$PRIOR" \
    --spre on \
    "${ARM_ARGS[@]}" \
    --no-clean-lists \
    --resume off \
    --build-priors skip
