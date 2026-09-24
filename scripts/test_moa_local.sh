#!/bin/bash -l
# =============================================================================
# MoAMVSNet 本地 DTU 测试: test_moa.py 推理 -> points_fusibile.py 融合点云。
#
#   bash scripts/test_moa_local.sh                      # 全部 22 个 test scan
#   SMOKE=1 bash scripts/test_moa_local.sh              # 只跑 scan1, 验流程
#   PHASE=infer bash scripts/test_moa_local.sh          # 只推理, 不融合
#   PHASE=fuse  bash scripts/test_moa_local.sh          # 只重融 (换阈值时用, 不重跑推理)
#   CKPT=... RESIZE=0.5 bash scripts/test_moa_local.sh
#
# 默认口径与集群一致: 整幅 0.8 (960x1280) + 5 视角, keep_ratio 0.60, disp 0.25,
# 3 视角一致。本地 5060 Ti 16GB 实测推理峰值约 6 GiB。
#
# ⚠ fusibile 必须编进本机 GPU 的 arch。CMakeLists 里那行
#   set_target_properties(fusibile PROPERTIES CUDA_ARCHITECTURES "80;89;90")
# 是 **target 属性**, 会覆盖命令行的 -DCMAKE_CUDA_ARCHITECTURES, 所以要么改它,
# 要么像下面这样追加 gencode。arch 不匹配时 fusibile 会报 PTX 错误、写出 0 顶点的
# ply、然后**退出码 0** —— 本脚本启动时会自检, 不让这种情况静默通过。
#   cd /home/william/project/fusibile
#   conda run -n fusibile_build cmake -S . -B build_sm120 -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_CUDA_FLAGS="-gencode arch=compute_120,code=sm_120"
#   conda run -n fusibile_build make -C build_sm120 -j8
#
# 打分是独立的第三步 (Fast-DTU-Evaluation), 结尾会打印命令。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PROJECT_DIR"

CONDA_ENV=${CONDA_ENV:-uprmvs}
PYTHON_BIN=${PYTHON_BIN:-/home/william/miniconda3/envs/$CONDA_ENV/bin/python}
# 本地 uprmvs 环境 2026-09-18 起缺 typing_extensions / matplotlib / pillow,
# 修好之前用一个临时目录补齐 (pip install --target <dir> --no-deps ...)。
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-}

RUN_NAME=${RUN_NAME:-moa2}
CKPT=${CKPT:-log/$RUN_NAME/checkpoint/best.pth}
PHASE=${PHASE:-all}                 # all | infer | fuse
SPLIT=${SPLIT:-test}
SMOKE=${SMOKE:-0}
NUM_VIEWS=${NUM_VIEWS:-5}
RESIZE=${RESIZE:-0.8}
FULL_IMAGE=${FULL_IMAGE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
CONF_WINDOW=${CONF_WINDOW:-1}
PHOTO_KEEP_RATIO=${PHOTO_KEEP_RATIO:-0.60}
PHOTO_THRESH=${PHOTO_THRESH:-0.3}   # 只在 PHOTO_KEEP_RATIO=0 时生效
DISP_THRESH=${DISP_THRESH:-0.25}
NUM_CONSISTENT=${NUM_CONSISTENT:-3}
FUSE_WORKERS=${FUSE_WORKERS:-2}
FUSIBILE_EXE=${FUSIBILE_EXE:-/home/william/project/fusibile/build_sm120/fusibile}
OUT=${OUT:-log/$RUN_NAME/depth_cache_$SPLIT}
PLY_DIR=${PLY_DIR:-log/$RUN_NAME/ply}
# 推理阶段的 MoA 语义覆盖 (给缺字段的旧 checkpoint 用), 例如 MOA_GAIN=1,1,1
DA3_MISSING=${DA3_MISSING:-error}
SCANS=${SCANS:-}                    # 例如 "1 4 9"; 空 = 列表里全部
MOA_GAIN=${MOA_GAIN:-}
EDGE_SNAP=${EDGE_SNAP:-}

case "$PHASE" in all|infer|fuse) ;; *) echo "PHASE 只能是 all / infer / fuse" >&2; exit 2 ;; esac
case "$SMOKE" in 1) MAX_SCANS=1 ;; 0) MAX_SCANS=0 ;; *) echo "SMOKE 只能是 0/1" >&2; exit 2 ;; esac

export UPRMVS_MACHINE=ubuntu
export UPRMVS_PROFILE=local
export PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$PYTHON_BIN" -c 'import torch, cv2; assert torch.cuda.is_available()' 2>/dev/null || {
    echo "Python 环境不可用 (import torch/cv2 失败或没有 CUDA)。" >&2
    echo "  用 PYTHON_BIN=... / EXTRA_PYTHONPATH=... 指定可用的解释器和补包目录。" >&2
    exit 1
}

# --- fusibile arch 自检: 不匹配就是 0 顶点 ply + 退出码 0, 必须提前拦下 ---
check_fusibile () {
    [[ -x "$FUSIBILE_EXE" ]] || { echo "找不到 fusibile: $FUSIBILE_EXE (用 FUSIBILE_EXE=... 指定)" >&2; exit 1; }
    local cc cuobj elf
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
    cuobj=$(command -v cuobjdump || echo "/home/william/miniconda3/envs/fusibile_build/bin/cuobjdump")
    [[ -n "$cc" && -x "$cuobj" ]] || { echo "[warn] 无法自检 fusibile 的 CUDA arch (缺 nvidia-smi 或 cuobjdump)"; return 0; }
    # 先取完整输出再匹配: `cuobjdump | grep -q` 在 pipefail 下会因为 grep 提前退出
    # 让 cuobjdump 吃到 SIGPIPE(141), 管道状态非零, 于是明明匹配上了也判成失败。
    elf=$("$cuobj" --list-elf "$FUSIBILE_EXE" 2>/dev/null || true)
    if grep -q "sm_${cc}\." <<<"$elf"; then
        echo "[check] fusibile 覆盖 sm_${cc} ✓"
    else
        echo "fusibile ($FUSIBILE_EXE) 没有编进本机 GPU 的 sm_${cc}:" >&2
        sed 's/^/    /' <<<"$elf" >&2
        echo "  不重编的话它会报 PTX 错误、写出 0 顶点的 ply 并**退出码 0**。重编方法见本脚本顶部注释。" >&2
        exit 1
    fi
}

echo "======================================================================"
echo " MoAMVSNet 本地测试  phase=$PHASE split=$SPLIT smoke=$SMOKE"
echo " ckpt=$CKPT  resize=$RESIZE full_image=$FULL_IMAGE views=$NUM_VIEWS"
echo " depth cache -> $OUT"
echo " ply         -> $PLY_DIR"
echo "======================================================================"
[[ "$PHASE" != "infer" ]] && check_fusibile

if [[ "$PHASE" != "fuse" ]]; then
    [[ -f "$CKPT" ]] || { echo "找不到 checkpoint: $CKPT" >&2; exit 1; }
    targs=(--profile local --ckpt "$CKPT" --split "$SPLIT" --num-views "$NUM_VIEWS"
           --resize-scale "$RESIZE" --num-workers "$NUM_WORKERS" --max-scans "$MAX_SCANS"
           --conf-window "$CONF_WINDOW" --da3-missing "$DA3_MISSING" --out "$OUT")
    [[ -n "$SCANS" ]] && targs+=(--scans $SCANS)
    [[ "$FULL_IMAGE" == "1" ]] && targs+=(--full-image)
    [[ -n "$MOA_GAIN" ]] && targs+=(--moa-gain "$MOA_GAIN")
    [[ -n "$EDGE_SNAP" ]] && targs+=(--edge-snap "$EDGE_SNAP")
    echo; echo "### [1/2] 推理 -> $OUT ###"
    "$PYTHON_BIN" test_moa.py "${targs[@]}"
fi
if [[ "$PHASE" == "infer" ]]; then
    echo "=== PHASE=infer 到此为止。融合: PHASE=fuse bash $0 ==="
    exit 0
fi

echo; echo "### [2/2] fusibile 融合 -> $PLY_DIR ###"
[[ -d "$OUT/depth" ]] || { echo "$OUT/depth 不存在 —— 先跑推理 (PHASE=all 或 infer)" >&2; exit 1; }
fargs=(--out "$OUT" --ply-dir "$PLY_DIR" --workers "$FUSE_WORKERS"
       --disp-thresh "$DISP_THRESH" --num-consistent "$NUM_CONSISTENT" --fusibile-exe "$FUSIBILE_EXE")
if awk -v r="$PHOTO_KEEP_RATIO" 'BEGIN{exit !(r > 0)}'; then
    fargs+=(--photo-keep-ratio "$PHOTO_KEEP_RATIO")
else
    fargs+=(--photo-thresh "$PHOTO_THRESH")
fi
"$PYTHON_BIN" points_fusibile.py "${fargs[@]}"

n_ply=$(find "$PLY_DIR" -maxdepth 1 -name 'mvsnet*_l3.ply' 2>/dev/null | wc -l)
echo "======================================================================"
echo " 完成: $n_ply 个点云在 $PROJECT_DIR/$PLY_DIR ; 深度指标 $OUT/metrics.json"
echo " 打分: cd <Fast-DTU-Evaluation> && python eval_dtu.py --method mvsnet --save \\"
echo "       --pred_dir $PROJECT_DIR/$PLY_DIR --gt_dir <DTU GT 根目录>"
echo "======================================================================"
