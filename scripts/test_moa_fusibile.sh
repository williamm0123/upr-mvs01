#!/bin/bash -l
#SBATCH --job-name=moa_test
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# MoAMVSNet DTU 测试: test_moa.py 推理 (深度指标 + 逐视角深度缓存) -> points_fusibile.py 融合。
# sbatch 或在 interactive GPU 分配里 bash 都行:
#
#   RUN_NAME=MOA_E15 sbatch scripts/test_moa_fusibile.sh
#   RUN_NAME=MOA_E15 SMOKE=1 bash scripts/test_moa_fusibile.sh     # 只跑 scan1
#   PHASE=fuse OUT=log/depth_cache/... bash scripts/test_moa_fusibile.sh   # 只重融
#
# 默认沿用已定的测试配置: 整幅 0.8 + 5 视角, keep_ratio 0.60, disp 0.25, 3 视角一致。
# 打分是独立的第三步 (Fast-DTU-Evaluation), 结尾会打印命令。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/uprmvs/bin/python}
PROFILE=${PROFILE:-umhpc}
RUN_NAME=${RUN_NAME:-MOA_E15}
CKPT=${CKPT:-log/experiments/$RUN_NAME/model/best.pth}
PHASE=${PHASE:-all}                 # all | infer | fuse
SPLIT=${SPLIT:-test}
SMOKE=${SMOKE:-0}
NUM_VIEWS=${NUM_VIEWS:-5}
RESIZE=${RESIZE:-0.8}
FULL_IMAGE=${FULL_IMAGE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
CONF_WINDOW=${CONF_WINDOW:-1}
PHOTO_KEEP_RATIO=${PHOTO_KEEP_RATIO:-0.60}
PHOTO_THRESH=${PHOTO_THRESH:-0.3}   # 只在 PHOTO_KEEP_RATIO=0 时生效
DISP_THRESH=${DISP_THRESH:-0.25}
NUM_CONSISTENT=${NUM_CONSISTENT:-3}
FUSIBILE_EXE=${FUSIBILE_EXE:-}
FUSE_WORKERS=${FUSE_WORKERS:-8}
TAG=${TAG:-${RUN_NAME}_r${RESIZE}_v${NUM_VIEWS}}
OUT=${OUT:-log/depth_cache/${TAG}_${SPLIT}}
PLY_DIR=${PLY_DIR:-log/pred_points_${TAG}_fusibile}

cd "$PROJECT_DIR"
case "$PHASE" in all|infer|fuse) ;; *) echo "PHASE 只能是 all / infer / fuse" >&2; exit 2 ;; esac
case "$SMOKE" in 1) MAX_SCANS=1 ;; 0) MAX_SCANS=0 ;; *) echo "SMOKE 只能是 0/1" >&2; exit 2 ;; esac
[[ -x "$PYTHON_BIN" ]] || { echo "找不到解释器: $PYTHON_BIN (用 PYTHON_BIN=... 覆盖)" >&2; exit 1; }

export UPRMVS_MACHINE=${UPRMVS_MACHINE:-umhpc}
export UPRMVS_PROFILE=$PROFILE
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p logs

archive () {
    local d="$1"
    if [[ -e "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null)" ]]; then
        local a="${d%/}_old_$(date +%Y%m%d_%H%M%S)"
        mv "$d" "$a"
        echo "=== 已存在的 $d 归档为 $a (未删除) ==="
    fi
}

echo "======================================================================"
echo " MoAMVSNet DTU test  phase=$PHASE split=$SPLIT smoke=$SMOKE ckpt=$CKPT"
echo " resize=$RESIZE full_image=$FULL_IMAGE views=$NUM_VIEWS"
echo " depth cache -> $OUT    ply -> $PLY_DIR"
echo " git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "======================================================================"

if [[ "$PHASE" != "fuse" ]]; then
    [[ -f "$CKPT" ]] || { echo "找不到 checkpoint: $CKPT (用 CKPT=... 或 RUN_NAME=... 指定)" >&2; exit 1; }
    archive "$OUT"
    targs=(--profile "$PROFILE" --ckpt "$CKPT" --split "$SPLIT" --num-views "$NUM_VIEWS"
           --resize-scale "$RESIZE" --num-workers "$NUM_WORKERS" --max-scans "$MAX_SCANS"
           --conf-window "$CONF_WINDOW" --out "$OUT")
    [[ "$FULL_IMAGE" == "1" ]] && targs+=(--full-image)
    echo; echo "### [1/2] 推理 -> $OUT ###"
    "$PYTHON_BIN" test_moa.py "${targs[@]}"
fi
if [[ "$PHASE" == "infer" ]]; then
    echo "=== PHASE=infer, 到此为止。融合: PHASE=fuse OUT=$OUT bash $0 ==="
    exit 0
fi

echo; echo "### [2/2] fusibile 融合 -> $PLY_DIR ###"
[[ -d "$OUT/depth" ]] || { echo "$OUT/depth 不存在 —— 先跑推理 (PHASE=all 或 infer)" >&2; exit 1; }
archive "$PLY_DIR"
fargs=(--out "$OUT" --ply-dir "$PLY_DIR" --workers "$FUSE_WORKERS"
       --disp-thresh "$DISP_THRESH" --num-consistent "$NUM_CONSISTENT")
if awk -v r="$PHOTO_KEEP_RATIO" 'BEGIN{exit !(r > 0)}'; then
    fargs+=(--photo-keep-ratio "$PHOTO_KEEP_RATIO")
else
    fargs+=(--photo-thresh "$PHOTO_THRESH")
fi
[[ -n "$FUSIBILE_EXE" ]] && fargs+=(--fusibile-exe "$FUSIBILE_EXE")
"$PYTHON_BIN" points_fusibile.py "${fargs[@]}"

n_ply=$(find "$PLY_DIR" -maxdepth 1 -name 'mvsnet*_l3.ply' | wc -l)
echo "======================================================================"
echo " 完成: $n_ply 个点云在 $PROJECT_DIR/$PLY_DIR ; 深度指标 $OUT/metrics.json"
echo " 打分: cd <Fast-DTU-Evaluation> && python eval_dtu.py --method mvsnet --save \\"
echo "       --pred_dir $PROJECT_DIR/$PLY_DIR --gt_dir <DTU GT 根目录>"
echo "======================================================================"
