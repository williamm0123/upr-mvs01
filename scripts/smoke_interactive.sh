#!/bin/bash -l
# =============================================================================
# UPRMVS 工单 v3 —— interactive 节点上的实现校验 + 显存实测。
# 本脚本不含 sbatch/salloc/srun, 不申请任何资源, 只在已经拿到 GPU 的 shell 里跑。
#
#   salloc --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=02:00:00
#   cd /scr/user/qinglong/projects/upr-mvs01
#   bash scripts/smoke_interactive.sh
#
# 五步, 全部是**实现校验**而不是性能实验:
#   [1] 单元检查            不碰数据集, 秒级
#   [2] legacy 等价性       axis_space=inverse 在 step 0 必须与 legacy 逐元素一致
#   [3] 死参数检查          每个可训练参数都要收到梯度 (白占优化器状态 / DDP 会炸)
#   [4] 显存峰值            合成全分辨率前反向的确切峰值 + batch 扫描
#   [5] 真实数据短跑        四个 arm + 一次长跑 (2026-08-23 起只用单卡)
#
# **[2] 不过就不要往下走。** W1 同时动了轴的空间、interval 的算法、回归方式和
# 损失, 任何一处写错都会以"效果不好"而不是报错的形式出现。
#
# 只跑某一步:  STAGE=mem bash scripts/smoke_interactive.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/uprmvs/bin/python}
SMOKE_STEPS=${SMOKE_STEPS:-1000}        # arm 扫描: 只问"跑不跑得起来"
LONG_STEPS=${LONG_STEPS:-2000}        # 长跑: 工单验收 (4) 要 200-500 步
LONG_ARM=${LONG_ARM:-w1}             # 长跑用哪个 arm (就是你要提交的那个)
BATCH_SIZE=${BATCH_SIZE:-4}          # per-GPU
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_VIEWS=${NUM_VIEWS:-5}
MEM_BATCHES=${MEM_BATCHES:-"1 2 4"}  # 显存扫描的 per-GPU batch
STAGE=${STAGE:-all}                  # all / units / equiv / ddpcheck / mem / run / long

[[ -d "$PROJECT_DIR" ]] || { echo "找不到项目目录: $PROJECT_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN"  ]] || { echo "找不到解释器: $PYTHON_BIN" >&2; exit 1; }
cd "$PROJECT_DIR"

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=================================================================="
echo " UPRMVS 工单 v3 实现校验"
echo " host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "=================================================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
"$PYTHON_BIN" -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available(),"gpus",torch.cuda.device_count())'
NGPU=$("$PYTHON_BIN" -c 'import torch;print(torch.cuda.device_count())')
[[ "$NGPU" -ge 1 ]] || { echo "没有可见 GPU —— 先进 GPU interactive 分配" >&2; exit 1; }

run_stage () { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

# ------------------------------------------------------------------ [1] 单元
if run_stage units; then
echo; echo "### [1/5] 单元检查 (不碰数据集) ###"
"$PYTHON_BIN" scripts/verify_w1.py --units
fi

# ------------------------------------------------------- [2] legacy 等价性
if run_stage equiv; then
echo; echo "### [2/5] legacy 等价性 (step 0 的候选轴必须逐元素一致) ###"
"$PYTHON_BIN" scripts/verify_w1.py --equivalence --tol 1e-5
fi

# ------------------------------------------------------------- [3] DDP 前提
if run_stage ddpcheck; then
echo; echo "### [3/5] 死参数检查: 每个可训练参数都要收到梯度 ###"
echo "    (要梯度却永远收不到 = 白占 AdamW 状态; 而且一旦回到 DDP"
echo "     find_unused_parameters=False 会在第二个 iteration 直接报错)"
"$PYTHON_BIN" scripts/verify_w1.py --ddp-check
fi

# ------------------------------------------------------------- [4] 显存峰值
if run_stage mem; then
echo; echo "### [4/5] 合成全分辨率前反向的**确切**显存峰值 ###"
echo "    torch.cuda.max_memory_allocated —— nvidia-smi 的采样在短跑里经常采不到峰值"
for b in $MEM_BATCHES; do
  echo; echo "--- per-GPU batch = $b ---"
  set +e
  "$PYTHON_BIN" scripts/verify_w1.py --mem --batch "$b" --views "$NUM_VIEWS"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "    batch=$b 失败 (很可能是 OOM) —— 这就是这张卡的上界"
    break
  fi
done
echo
echo "提示: 训练是多尺度的 (最大 640x896, 见 cfg.augment.scales), 上面用的是"
echo "      target 512x640。按 0.57 + B x (0.18 + 21.9 x Mpx) GiB 外推到 640x896:"
echo "      B=1 约 13GiB, B=2 约 26GiB, B=4 约 51GiB, W1/W3 再加约 6%。"
fi

# ------------------------------------------------------- [5] 真实数据短跑
if run_stage run; then
echo; echo "### [5/5] 真实数据短跑 (每个配置 ${SMOKE_STEPS} 步) ###"
common=(
  --profile umhpc --gpus 1 --ddp off
  --batch-size "$BATCH_SIZE" --num-views "$NUM_VIEWS" --num-workers "$NUM_WORKERS"
  --amp on --amp-dtype bf16
  --num-global 32 --num-local 16 --range-min-gi 0.66,0.20,0.10
  --gate-local off --branch-prior off
  --stage1-weight 0.5 --w-branch 0 --prior on --spre on
  --no-clean-lists --smoke --smoke-steps "$SMOKE_STEPS" --build-priors skip
)
W1_ARGS=(--axis-space inverse --stage4-head map --mode-window-stages 2,2,1,2
         --spre-cascade on --tau-stages 0.98,0.95,0.92)
for cfgname in w0 w1 w3 w3b; do
  case "$cfgname" in
    w0)  extra=(--axis-space legacy_depth --stage4-head expect --visibility off) ;;
    w1)  extra=("${W1_ARGS[@]}" --visibility off) ;;
    w3)  extra=("${W1_ARGS[@]}" --visibility off --geo-valid on
                --conf-head on --w-conf 1.0) ;;
    w3b) extra=("${W1_ARGS[@]}" --geo-valid on --conf-head on --w-conf 1.0
                --visibility on --vis-mode sigmoid --vis-supervise on --w-vis 0.1) ;;
  esac
  echo; echo "--- arm: $cfgname ---"
  mon="/tmp/gpu_${cfgname}_$$.csv"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader -l 2 > "$mon" 2>/dev/null &
  monpid=$!
  set +e
  "$PYTHON_BIN" train.py "${common[@]}" "${extra[@]}" --name "smoke_$cfgname" 2>&1 | tail -12
  rc=${PIPESTATUS[0]}
  set -e
  kill "$monpid" 2>/dev/null || true
  wait "$monpid" 2>/dev/null || true
  if [[ -s "$mon" ]]; then
    echo "--- $cfgname nvidia-smi 采样峰值 (2s 间隔, 只是参考; 确切值看 [4]) ---"
    sort -t, -k2 -n -r "$mon" | head -2
  fi
  rm -f "$mon"
  [[ "$rc" -eq 0 ]] || { echo "arm $cfgname 退出码 $rc" >&2; exit "$rc"; }
done

fi

# ------------------------------------------- [5b] 真实数据长跑 (工单验收 (4))
if run_stage long; then
echo; echo "### [5b] 真实数据长跑 ${LONG_STEPS} 步 (arm=$LONG_ARM) ###"
echo "    工单验收 (4): 检查非有限值、梯度范数、张量形状。"
echo "    **不是**性能实验 —— 200 步区分不了'实现有 bug'和'方案不行', 不要拿它排名。"
case "$LONG_ARM" in
  w0)  long_extra=(--axis-space legacy_depth --stage4-head expect --visibility off) ;;
  w1)  long_extra=(--axis-space inverse --stage4-head map --mode-window-stages 2,2,1,2
                   --spre-cascade on --tau-stages 0.98,0.95,0.92 --visibility off) ;;
  w3)  long_extra=(--axis-space inverse --stage4-head map --mode-window-stages 2,2,1,2
                   --spre-cascade on --tau-stages 0.98,0.95,0.92 --visibility off
                   --geo-valid on --conf-head on --w-conf 1.0) ;;
  w3b) long_extra=(--axis-space inverse --stage4-head map --mode-window-stages 2,2,1,2
                   --spre-cascade on --tau-stages 0.98,0.95,0.92
                   --geo-valid on --conf-head on --w-conf 1.0
                   --visibility on --vis-mode sigmoid --vis-supervise on --w-vis 0.1) ;;
  *)   echo "LONG_ARM 只能是 w0/w1/w3/w3b" >&2; exit 2 ;;
esac
LONG_LOG="/tmp/uprmvs_long_${LONG_ARM}_$$.log"
"$PYTHON_BIN" train.py     --profile umhpc --gpus 1 --ddp off     --batch-size "$BATCH_SIZE" --num-views "$NUM_VIEWS" --num-workers "$NUM_WORKERS"     --amp on --amp-dtype bf16     --num-global 32 --num-local 16 --range-min-gi 0.66,0.20,0.10     --gate-local off --branch-prior off     --stage1-weight 0.5 --w-branch 0 --prior on --spre on     "${long_extra[@]}"     --steps "$LONG_STEPS" --lr-schedule-steps 30000     --val-interval "$LONG_STEPS" --log-interval 10     --no-clean-lists --build-priors skip --resume off     --name "long_$LONG_ARM" 2>&1 | tee "$LONG_LOG" | tail -30
echo
echo "--- 非有限值 / 梯度自检 ---"
if grep -qiE "nan|inf" "$LONG_LOG"; then
  echo "  日志里出现 nan/inf, 逐行看一下 (rescue_err=nan 在 branch_prior=off 时是正常的):"
  grep -inE "nan|inf" "$LONG_LOG" | head -10
else
  echo "  [ok] 日志里没有 nan/inf"
fi
grep -E "grad/norm|nonfinite" "$LONG_LOG" | tail -3 || true
echo "  完整日志: $LONG_LOG"
echo "  梯度范数 / amp_scale / nonfinite_frac 三条曲线在 tensorboard 的"
echo "  train/diag_grad_* 下, 事故的提前量都在那里。"
fi

echo
echo "=================================================================="
echo " 校验结束。提交 30k 之前确认:"
echo "   * [2] legacy 等价性 PASS —— 不过就不要提交"
echo "   * [3] 四个配置都没有 '缺梯度' 的参数"
echo "   * [4] 目标 batch 的峰值显存离 80GiB 有余量 (多尺度会再高约 40%)"
echo "   * [5] 四个 arm 都没有非有限值"
echo "   * [5b] ${LONG_STEPS} 步长跑没有 nan/inf, 梯度范数没有发散"
echo "=================================================================="
