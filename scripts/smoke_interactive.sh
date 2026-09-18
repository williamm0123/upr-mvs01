#!/bin/bash -l
# =============================================================================
# UPRMVS —— 单卡 A100-80GB 上的**实现校验 + 显存实测** (默认 ARM=sva)。
# 在已经拿到 GPU 的 interactive shell 里跑; 本脚本不含 sbatch/salloc/srun,
# 不申请任何资源。
#
#   salloc --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=03:00:00
#   cd /scr/user/qinglong/projects/upr-mvs01 && git pull
#   bash scripts/smoke_interactive.sh
#
# 四步, 回答的都是 "跑不跑得通 / 占多少显存", **不是**性能实验:
#   [1] env    GPU / torch / git / 数据集与先验缓存路径
#   [2] synth  scripts/verify_sva.py (10 条实现校验: PE / 线性注意力 / FMT_with_pathway
#              与 MVSFormer++ 一致、out0 + 普通 top-down、44+4 候选、梯度覆盖、
#              fingerprint 往返、CVPE 已卸载),
#              然后合成数据 train.py --smoke: 构造 + 前向 + 反向 + 存 checkpoint。
#              不碰数据集, 几分钟内出结果。这一步挂了后面不用看。
#   [3] mem    train.py --fit-batch: 在多尺度里**最大**的训练尺度 (640x896) 上
#              逐个 batch 量峰值显存 (合成数据, 前反向两步, 含 AdamW 状态)。
#   [4] real   真实数据 LONG_STEPS 步训练 + 结尾一次完整 val, 后台 nvidia-smi
#              每秒采样。报: 峰值显存 / 占用率 / GPU 利用率 / s/step -> 30k ETA /
#              有无 nan。
#
# 训练参数与 scripts/sbatch_1gpu.sh **同源** (scripts/_arm_common.sh), 所以这里
# 量到的就是正式训练的那一份, 不是另一个配置。
#
# 只跑某一步:  STAGE=mem bash scripts/smoke_interactive.sh
# 调整:        LONG_STEPS=500 FIT_BATCHES=2,4,6 PER_GPU_BATCH=4 bash ...
# =============================================================================
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/uprmvs/bin/python}
TRAIN_PROFILE=${TRAIN_PROFILE:-umhpc}
STAGE=${STAGE:-all}                 # all / env / synth / mem / real
SMOKE_STEPS=${SMOKE_STEPS:-3}
FIT_BATCHES=${FIT_BATCHES:-1,2,4,5,6}
FIT_TARGET=${FIT_TARGET:-0.90}
LONG_STEPS=${LONG_STEPS:-300}

[[ -d "$PROJECT_DIR" ]] || { echo "找不到项目目录: $PROJECT_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN"  ]] || { echo "找不到解释器: $PYTHON_BIN (用 PYTHON_BIN=... 覆盖)" >&2; exit 1; }
cd "$PROJECT_DIR"

ARM=${ARM:-sva}
NPROC=1                       # 必须在 source 之前: 用来算 GLOBAL_BATCH 和 lr
# shellcheck source=scripts/_arm_common.sh
source "$PROJECT_DIR/scripts/_arm_common.sh"
SMOKE_NAME="smoke_${ARM}"

export UPRMVS_MACHINE=${UPRMVS_MACHINE:-umhpc}
export UPRMVS_PROFILE="$TRAIN_PROFILE"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT_DIR="logs/smoke_${ARM}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

run_stage () { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }
train_cmd=("$PYTHON_BIN" train.py --gpus 1 --ddp off "${COMMON_ARGS[@]}" "${ARM_ARGS[@]}")

echo "=================================================================="
echo " UPRMVS 单卡实现校验 + 显存实测   arm=$ARM"
echo " host=$(hostname)  job=${SLURM_JOB_ID:-none}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo " git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (工作树有改动)')"
echo " per_gpu_batch=$PER_GPU_BATCH  lr=$LR  views=$NUM_VIEWS  workers=$NUM_WORKERS"
echo " stage1: global=$NUM_GLOBAL local=$NUM_LOCAL  range_min_gi=$RANGE_MIN_GI"
echo " arm_args: ${ARM_ARGS[*]}"
echo " 日志目录: $OUT_DIR"
echo "=================================================================="

# ------------------------------------------------------------------ [1] env
# 环境检查每次都跑: 它很便宜, 而且没卡的话后面三步都没有意义。
echo; echo "### [1/4] 环境 ###"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
"$PYTHON_BIN" - <<'PY'
import sys, torch
print(f"python {sys.executable}\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("没有可见 GPU —— 先进 GPU interactive 分配 (salloc ... --gres=gpu:1)")
p = torch.cuda.get_device_properties(0)
print(f"GPU0 {p.name}  {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}")
from base.config import ProjectPaths
pp = ProjectPaths()
for k in ("dtu_train_root", "prior_cache_path", "dinov3_weights_file"):
    v = getattr(pp, k)
    print(f"  {k:20s} {v}{'' if v.exists() else '   <<< 不存在'}")
PY

# ------------------------------------------------------------------ [2] synth
if run_stage synth; then
echo; echo "### [2/4] 实现校验 (scripts/verify_sva.py) ###"
set +e
"$PYTHON_BIN" scripts/verify_sva.py 2>&1 | tee "$OUT_DIR/verify_sva.log" | grep -E "^\[verify_sva\]|\[ok\]|Error|assert"
rc=${PIPESTATUS[0]}
set -e
[[ $rc -eq 0 ]] || { echo "verify_sva 失败 (退出码 $rc), 完整日志 $OUT_DIR/verify_sva.log" >&2; exit "$rc"; }
echo; echo "### [2/4] 合成数据 smoke (${SMOKE_STEPS} 步, 构造 + 前向 + 反向 + 存 ckpt) ###"
set +e
"${train_cmd[@]}" --batch-size "$PER_GPU_BATCH" --name "$SMOKE_NAME" \
    --smoke --smoke-steps "$SMOKE_STEPS" 2>&1 | tee "$OUT_DIR/synth.log" | grep -E "^\[smoke|params=|Error"
rc=${PIPESTATUS[0]}
set -e
[[ $rc -eq 0 ]] || { echo "合成数据 smoke 失败 (退出码 $rc), 完整日志 $OUT_DIR/synth.log" >&2; exit "$rc"; }
fi

# ------------------------------------------------------------------ [3] mem
if run_stage mem; then
echo; echo "### [3/4] 显存扫描: 最大训练尺度上 batch = ${FIT_BATCHES} ###"
echo "    (训练是多尺度的, 峰值只出现在最大尺度那几步; 按它定 batch 才不会跑到一半 OOM)"
set +e
"${train_cmd[@]}" --batch-size 1 --name "fit_${ARM}" \
    --fit-batch "$FIT_BATCHES" --fit-target "$FIT_TARGET" \
    --fit-lr-ref "$LR_REF" --fit-lr-ref-batch "$LR_REF_BATCH" 2>&1 | tee "$OUT_DIR/fit_batch.log" \
    | grep -E "^\[fit-batch\]|^ +[0-9]+ |batch +allocated|OOM|卡 ->"
rc=${PIPESTATUS[0]}
set -e
[[ $rc -eq 0 ]] || { echo "显存扫描异常退出 (退出码 $rc), 完整日志 $OUT_DIR/fit_batch.log" >&2; exit "$rc"; }
fi

# ------------------------------------------------------------------ [4] real
if run_stage real; then
echo; echo "### [4/4] 真实数据 ${LONG_STEPS} 步 + 一次完整 val (batch=$PER_GPU_BATCH) ###"
MON="$OUT_DIR/nvidia_smi.csv"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "$MON" 2>/dev/null &
monpid=$!
trap 'kill $monpid 2>/dev/null || true' EXIT
set +e
"${train_cmd[@]}" --batch-size "$PER_GPU_BATCH" --name "${SMOKE_NAME}_real" \
    --steps "$LONG_STEPS" --val-interval 1000000000 --log-interval 10 --resume off \
    2>&1 | tee "$OUT_DIR/real.log" | grep -E "^\[step|^\[val|NonFinite|Error|OOM|out of memory"
rc=${PIPESTATUS[0]}
set -e
kill "$monpid" 2>/dev/null || true
wait "$monpid" 2>/dev/null || true

echo
echo "--- [4] 汇总 ---"
if [[ -s "$MON" ]]; then
    awk -F', *' '{u=$1; t=$2; if (u>m) m=u; s+=$3; n++} END {
        printf "  nvidia-smi 峰值显存 %.1f / %.1f GiB = %.1f%%   平均 GPU 利用率 %.0f%% (%d 个采样, 含 val 与 dataloader 预热)\n",
               m/1024, t/1024, 100*m/t, s/n, n }' "$MON"
fi
last=$(grep -E "^\[step" "$OUT_DIR/real.log" | tail -1 || true)
if [[ -n "$last" ]]; then
    echo "  最后一行: $last"
    sps=$(sed -n 's/.* \([0-9.]*\)s\/step.*/\1/p' <<<"$last")
    [[ -n "$sps" ]] && awk -v s="$sps" -v n="$STEPS" \
        'BEGIN{printf "  %.2f s/step -> %d 步约 %.1f 小时 (不含 val; 每 500 步一次 val)\n", s, n, s*n/3600}'
fi
if grep -E "^\[step" "$OUT_DIR/real.log" | grep -qE "loss=(nan|inf)|abs_err=(nan|inf)"; then
    echo "  !!! loss / abs_err 出现 nan/inf:"
    grep -E "^\[step" "$OUT_DIR/real.log" | grep -E "loss=(nan|inf)|abs_err=(nan|inf)" | head -5
else
    echo "  [ok] 训练行里 loss / abs_err 没有 nan/inf (rescue_err=nan 在窗口里没有被破坏的先验时是正常的)"
fi
grep -E "^\[val" "$OUT_DIR/real.log" | tail -1 || echo "  !!! 没有 val 输出 —— val 没跑完"
[[ $rc -eq 0 ]] || { echo "真实数据短跑失败 (退出码 $rc), 完整日志 $OUT_DIR/real.log" >&2; exit "$rc"; }
fi

echo
echo "=================================================================="
echo " 校验结束 (完整日志在 $OUT_DIR)。提交 30k 之前确认:"
echo "   * [2] verify_sva 全部通过, 合成数据 smoke 打印了 '[smoke] OK'"
echo "   * [3] PER_GPU_BATCH=$PER_GPU_BATCH 那一行的占用率 < 90%"
echo "         (没有就改小: PER_GPU_BATCH=... sbatch scripts/sbatch_1gpu.sh, lr 会自动按 sqrt 缩放)"
echo "   * [4] 没有 nan/inf、val 正常结束; 看一眼 s/step 推算的 30k 时长是否在 --time=2 天之内"
echo " 然后:  sbatch scripts/sbatch_1gpu.sh"
echo "=================================================================="
