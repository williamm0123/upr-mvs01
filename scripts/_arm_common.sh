# shellcheck shell=bash
# =============================================================================
# UPRMVS 工单 v3 —— arm 定义与公共训练参数。**不要直接执行**, 由这两个脚本 source:
#
#   scripts/sbatch_ddp2.sh              双卡 A100-80GB, torchrun DDP
#   scripts/train_umhpc_interactive.sh  单卡, interactive 节点直接跑
#
# 为什么要单独一个文件: 单卡和双卡唯一该有的差别是**进程数和全局 batch**,
# 其余每一个参数都必须逐字相同。两个脚本各抄一份 arg 列表, 迟早会有人只改了
# 一边 —— 那时候两条曲线还长得很像, 但已经不是同一个实验了。
#
# 用法 (调用方):
#   ARM=${ARM:-w1}
#   source "$(dirname "$0")/_arm_common.sh"
#   ... "${COMMON_ARGS[@]}" "${ARM_ARGS[@]}"
#
# 提供的变量:
#   ARM_ARGS[]     该 arm 独有的开关
#   COMMON_ARGS[]  三个 arm 完全一致的部分 (不含 --batch-size / --name / --ddp)
#   RUN_NAME       该 arm 的默认 run 名 (调用方可先行覆盖)
#   PER_GPU_BATCH  每张卡的 batch (单卡双卡都用这个值)
#   NUM_WORKERS NUM_VIEWS STEPS LR SEED ...   给调用方 echo 用
# =============================================================================

ARM=${ARM:-w1}

# ---------------------------------------------------------------- arm 定义
case "$ARM" in
  w0)
    RUN_NAME=${RUN_NAME:-R32_f10_30k}
    ARM_ARGS=(--axis-space legacy_depth --stage4-head expect --visibility off)
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
      --visibility "${VISIBILITY:-off}"
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
    else
      ARM_ARGS+=(--visibility "${VISIBILITY:-off}")
    fi
    ;;
  *)
    echo "ARM 只能是 w0 / w1 / w3 / w3b, 收到 '$ARM'" >&2; return 2 2>/dev/null || exit 2 ;;
esac

# ------------------------------------------- 公共部分 (四个 arm 完全一致)
STEPS=${STEPS:-30000}
LR_HORIZON=${LR_HORIZON:-30000}
LR=${LR:-3e-4}
SEED=${SEED:-20260526}
AMP_DTYPE=${AMP_DTYPE:-bf16}
NUM_GLOBAL=${NUM_GLOBAL:-32}
NUM_LOCAL=${NUM_LOCAL:-16}
# inverse 轴下它只是 RangeController 的初始化值, 不再是训练期常数
RANGE_MIN_GI=${RANGE_MIN_GI:-0.66,0.20,0.10}
GATE_LOCAL=${GATE_LOCAL:-off}
BRANCH_PRIOR=${BRANCH_PRIOR:-off}
STAGE1_WEIGHT=${STAGE1_WEIGHT:-0.5}
W_BRANCH=${W_BRANCH:-0}
SPRE_BALANCE=${SPRE_BALANCE:-on}
PRIOR=${PRIOR:-on}

# **每张卡**的 batch。单卡脚本和双卡脚本都用这一个值 —— 这正是"每张卡的训练
# 参数相同"的含义; 两者的全局 batch 因此差一倍 (单卡 1, 双卡 2), 调用方要自己
# 把这件事说清楚。
PER_GPU_BATCH=${PER_GPU_BATCH:-1}
NUM_VIEWS=${NUM_VIEWS:-5}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-6}
NUM_WORKERS=${NUM_WORKERS:-8}
VAL_INTERVAL=${VAL_INTERVAL:-500}
LOG_INTERVAL=${LOG_INTERVAL:-10}
BUILD_PRIORS=${BUILD_PRIORS:-skip}
RESUME=${RESUME:-off}

COMMON_ARGS=(
    --profile "${TRAIN_PROFILE:-umhpc}"
    --steps "$STEPS"
    --lr-schedule-steps "$LR_HORIZON"
    --lr "$LR"
    --amp-dtype "$AMP_DTYPE"
    --seed "$SEED"
    --deterministic
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --val-batch-size "$VAL_BATCH_SIZE"
    --val-interval "$VAL_INTERVAL"
    --log-interval "$LOG_INTERVAL"
    --num-global "$NUM_GLOBAL"
    --num-local "$NUM_LOCAL"
    --range-min-gi "$RANGE_MIN_GI"
    --gate-local "$GATE_LOCAL"
    --branch-prior "$BRANCH_PRIOR"
    --stage1-weight "$STAGE1_WEIGHT"
    --w-branch "$W_BRANCH"
    --spre-balance-corrupt "$SPRE_BALANCE"
    --prior "$PRIOR"
    --spre on
    --no-clean-lists
    --resume "$RESUME"
    --build-priors "$BUILD_PRIORS"
)
