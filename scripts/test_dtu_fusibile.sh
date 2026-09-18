#!/bin/bash -l
# =============================================================================
# DTU 测试: 推理出逐视角深度图 -> fusibile 融合成点云 (每个 scan 一个 ply)。
# 在**已经拿到 A100 的 shell** 里直接跑 (interactive), 本脚本不排队:
#
#   salloc --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=04:00:00
#   cd /scr/user/qinglong/projects/upr-mvs01 && git pull
#   bash scripts/test_dtu_fusibile.sh
#
#   # 指定别的权重 / 先只跑 scan1 验流程:
#   CKPT=log/experiments/UPRMVS_SVA_G44L4/model/latest.pth bash scripts/test_dtu_fusibile.sh
#   SMOKE=1 bash scripts/test_dtu_fusibile.sh
#   # 换光度门重融 (不重跑推理, 几分钟一个 scan):
#   PHASE=fuse PHOTO_KEEP_RATIO=0.8 PLY_DIR=log/pred_points_sva_keep080 bash scripts/test_dtu_fusibile.sh
#
# 三步:
#   [1] 先验    test.py --priors-only。BUILD_PRIORS=auto: 缓存齐全时只是 stat 一遍,
#               缺哪个补哪个 (VGGT+DA3, 之后进程退出、显存全部还回来)。
#   [2] 推理    test.py --fusion none: 22 个 scan x 49 个参考视角, 逐视角 depth/conf
#               npz 写到 OUT/depth/, 深度图指标写 OUT/metrics.json。不在这个进程里融合。
#   [3] 融合    points_fusibile.py: 光度门 (conf) -> gipuma 目录树 -> fusibile ->
#               PLY_DIR/mvsnet{scan:03d}_l3.ply (Fast-DTU-Evaluation 认的命名)
# 打分是独立的第四步, 你手动跑 Fast-DTU-Evaluation; 脚本最后打印命令。
#
# 协议 (与之前的点云终审一致, 便于对比):
#   * checkpoint 用 latest.pth (训练结束那一步), **不用** best.pth、也不拿 test 点云
#     反挑 checkpoint —— best.pth 是按 val abs_err 选的, 那是尾巴主导的指标。
#   * RESIZE=0.8 整幅 (960x1280) + 5 视角。A100 显存放得下 1.0, 但之前所有点云都是
#     0.8, 改了就不可比。要试 1.0: RESIZE=1.0 OUT=... PLY_DIR=... (换目录!)
#   * 光度门: 固定保留率 PHOTO_KEEP_RATIO=0.60 (每个参考视角保留置信度最高的 60%)。
#     保留率对置信度的单调变换不敏感, 所以 conf_head 有没有做 Platt 标定都一样。
#     要用阈值门: PHOTO_KEEP_RATIO=0 PHOTO_THRESH=0.5。
#   * fusibile: disp 0.25 / 3 个一致视角 / 法向检查关闭 —— MVSFormer++ 的取值。
# =============================================================================
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/uprmvs/bin/python}

# ---- 模型 --------------------------------------------------------------------
RUN_NAME=${RUN_NAME:-UPRMVS_SVA_G44L4}
CKPT=${CKPT:-log/experiments/$RUN_NAME/model/latest.pth}
MIN_STEP=${MIN_STEP:-30000}         # 拦下训练中断留下的残权重; 确认要用就 ALLOW_STALE=1
ALLOW_STALE=${ALLOW_STALE:-0}

# ---- 数据与推理 ---------------------------------------------------------------
PHASE=${PHASE:-all}                 # all | infer (先验+推理, 不融合) | fuse (只重融)
SPLIT=${SPLIT:-test}
SMOKE=${SMOKE:-0}                   # 1 = 只跑 scan1 (全部 49 个视角, 能融出点云)
NUM_VIEWS=${NUM_VIEWS:-5}
RESIZE=${RESIZE:-0.8}
FULL_IMAGE=${FULL_IMAGE:-1}
PRIOR_RESIZE=${PRIOR_RESIZE:-1.0}   # 与 RESIZE 无关; 改它才要 BUILD_PRIORS=force
BUILD_PRIORS=${BUILD_PRIORS:-auto}
NUM_WORKERS=${NUM_WORKERS:-8}
CONF_SOURCE=${CONF_SOURCE:-auto}    # auto: 有 conf_head 用它, 没有退回 cascade

# ---- 融合 --------------------------------------------------------------------
PHOTO_KEEP_RATIO=${PHOTO_KEEP_RATIO:-0.60}
PHOTO_THRESH=${PHOTO_THRESH:-0.3}   # 只在 PHOTO_KEEP_RATIO=0 时生效
DISP_THRESH=${DISP_THRESH:-0.25}
NUM_CONSISTENT=${NUM_CONSISTENT:-3}
FUSIBILE_EXE=${FUSIBILE_EXE:-}      # 空 = points_fusibile.py 按序找 (umhpc 的在第一位)
FUSIBILE_LIB_DIR=${FUSIBILE_LIB_DIR:-}
FUSE_WORKERS=${FUSE_WORKERS:-8}

TAG=${TAG:-${RUN_NAME}_r${RESIZE}_v${NUM_VIEWS}}
OUT=${OUT:-log/depth_cache/${TAG}_${SPLIT}}
PLY_DIR=${PLY_DIR:-log/pred_points_${TAG}_fusibile}
# =============================================================================

# 数值比较, 不是字符串比较: PHOTO_KEEP_RATIO=0.0 也要算"关掉保留率门"
KEEP_ON=$(awk -v r="$PHOTO_KEEP_RATIO" 'BEGIN{print (r > 0) ? 1 : 0}')

[[ -d "$PROJECT_DIR" ]] || { echo "找不到项目目录: $PROJECT_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN"  ]] || { echo "找不到解释器: $PYTHON_BIN (用 PYTHON_BIN=... 覆盖)" >&2; exit 1; }
cd "$PROJECT_DIR"
case "$PHASE" in all|infer|fuse) ;; *) echo "PHASE 只能是 all / infer / fuse, 收到 '$PHASE'" >&2; exit 2 ;; esac
case "$SMOKE" in 1) MAX_SCANS=1 ;; 0) MAX_SCANS=0 ;; *) echo "SMOKE 只能是 0/1" >&2; exit 2 ;; esac

export UPRMVS_MACHINE=${UPRMVS_MACHINE:-umhpc}
export UPRMVS_PROFILE=${UPRMVS_PROFILE:-umhpc}
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# 整幅推理不设这个会在 reserved-but-unallocated 的碎片上 OOM
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# test.py 在没卡时会**静默**退回 CPU (1078 个整幅前向要跑几天), fusibile 是纯 CUDA
# 程序 —— 两步都必须有卡, 在这里直接失败。
"$PYTHON_BIN" - <<'PY' || { echo "CUDA 不可用 —— 先进 GPU 分配 (salloc ... --gres=gpu:1)" >&2; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"=== GPU: {p.name}  {p.total_memory / 2**30:.0f} GiB  torch {torch.__version__} ===")
PY

# ---- checkpoint: 存在 / 步数 / 架构 ---------------------------------------------
if [[ "$PHASE" != "fuse" ]]; then
    [[ -f "$CKPT" ]] || { echo "找不到 checkpoint: $CKPT  (用 CKPT=... 或 RUN_NAME=... 指定)" >&2; exit 1; }
    CKPT_STEP=$("$PYTHON_BIN" - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
fp = ck.get("fingerprint") or {}
print(f"    step={ck.get('step')}  sva_full={fp.get('sva_full', False)}  "
      f"global/local={fp.get('num_global')}/{fp.get('num_local')}  "
      f"range_min_gi={fp.get('range_min_gi')}  conf_head={fp.get('fusion_conf', False)}  "
      f"cvpe={fp.get('cvpe_enabled', False)}  git={str((ck.get('git') or {}).get('commit', '?'))[:12]}",
      file=sys.stderr)
print(int(ck.get("step", -1)))
PY
)
    echo "=== checkpoint: $CKPT  (step $CKPT_STEP) ==="
    if [[ "$ALLOW_STALE" != "1" && "$CKPT_STEP" -lt "$MIN_STEP" ]]; then
        echo "这份权重只有 $CKPT_STEP 步 (< MIN_STEP=$MIN_STEP), 多半是训练中断留下的。" >&2
        echo "确认要用它就加 ALLOW_STALE=1 (或 MIN_STEP=...)。" >&2
        exit 1
    fi
fi

echo "======================================================================"
echo " DTU 测试 + fusibile 融合   phase=$PHASE  split=$SPLIT  smoke=$SMOKE"
echo " resize=$RESIZE full_image=$FULL_IMAGE views=$NUM_VIEWS  conf_source=$CONF_SOURCE"
echo " 光度门: $( [[ "$KEEP_ON" == 1 ]] && echo "keep_ratio $PHOTO_KEEP_RATIO" || echo "thresh $PHOTO_THRESH" )"
echo " fusibile: disp $DISP_THRESH  num_consistent $NUM_CONSISTENT"
echo " depth cache -> $OUT"
echo " ply         -> $PLY_DIR"
echo " git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)  host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "======================================================================"

# 旧产物**归档**而不是覆盖: 融合会吃下 OUT/depth 下的**全部** npz, 上一次的残留
# (别的分辨率/别的 checkpoint) 会静默混进同一个点云; ply 目录同理。
archive () {
    local d="$1"
    if [[ -e "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null)" ]]; then
        local a="${d%/}_old_$(date +%Y%m%d_%H%M%S)"
        mv "$d" "$a"
        echo "=== 已存在的 $d 归档为 $a (未删除) ==="
    fi
}

common=(--split "$SPLIT" --num-views "$NUM_VIEWS" --resize-scale "$RESIZE"
        --num-workers "$NUM_WORKERS" --max-scans "$MAX_SCANS" --max-refs 0
        --prior-resize-scale "$PRIOR_RESIZE" --out "$OUT")
[[ "$FULL_IMAGE" == "1" ]] && common+=(--full-image)

if [[ "$PHASE" != "fuse" ]]; then
    archive "$OUT"

    echo; echo "### [1/3] 先验 (BUILD_PRIORS=$BUILD_PRIORS) ###"
    "$PYTHON_BIN" test.py "${common[@]}" --build-priors "$BUILD_PRIORS" --priors-only --no-fuse

    echo; echo "### [2/3] 推理: 逐视角深度 + 置信度 -> $OUT/depth ###"
    "$PYTHON_BIN" test.py "${common[@]}" --build-priors skip --ckpt "$CKPT" \
        --fuse --fusion none --conf-source "$CONF_SOURCE"
    n_npz=$(find "$OUT/depth" -name '*.npz' | wc -l)
    echo "=== 推理完成: $n_npz 个逐视角 npz, 深度指标 $OUT/metrics.json ==="
fi

if [[ "$PHASE" == "infer" ]]; then
    echo "=== PHASE=infer, 到此为止。融合: PHASE=fuse OUT=$OUT bash $0 ==="
    exit 0
fi

echo; echo "### [3/3] fusibile 融合 -> $PLY_DIR ###"
[[ -d "$OUT/depth" ]] || { echo "$OUT/depth 不存在 —— 先跑推理 (PHASE=all 或 infer)" >&2; exit 1; }
archive "$PLY_DIR"
fuse_args=(--out "$OUT" --ply-dir "$PLY_DIR" --workers "$FUSE_WORKERS"
           --disp-thresh "$DISP_THRESH" --num-consistent "$NUM_CONSISTENT")
if [[ "$KEEP_ON" == 1 ]]; then
    fuse_args+=(--photo-keep-ratio "$PHOTO_KEEP_RATIO")
else
    fuse_args+=(--photo-thresh "$PHOTO_THRESH")
fi
[[ -n "$FUSIBILE_EXE" ]] && fuse_args+=(--fusibile-exe "$FUSIBILE_EXE")
if [[ -n "$FUSIBILE_LIB_DIR" ]]; then
    IFS=':' read -r -a _libs <<<"$FUSIBILE_LIB_DIR"
    for _l in "${_libs[@]}"; do fuse_args+=(--lib-dir "$_l"); done
fi
"$PYTHON_BIN" points_fusibile.py "${fuse_args[@]}"

n_ply=$(find "$PLY_DIR" -maxdepth 1 -name 'mvsnet*_l3.ply' | wc -l)
echo
echo "======================================================================"
echo " 完成: $n_ply 个点云在 $PROJECT_DIR/$PLY_DIR"
echo "   深度图指标:  $OUT/metrics.json"
echo "   融合参数:    $PLY_DIR/fusibile_manifest.json (含推理端的 run_manifest)"
echo " 打分 (独立的第四步):"
echo "   cd <Fast-DTU-Evaluation> && python eval_dtu.py --method mvsnet --save \\"
echo "       --pred_dir $PROJECT_DIR/$PLY_DIR --gt_dir <DTU GT 根目录>"
echo "======================================================================"
