#!/bin/bash
# 本地一条龙: 推理 -> 融合 -> DTU 打分。
#
#     bash scripts/eval_local.sh                      # 三个 arm 全跑, 跑完打分
#     ARMS="R_geo" bash scripts/eval_local.sh         # 只跑一个
#     SMOKE=1 bash scripts/eval_local.sh              # 只跑 scan1, 验流程 (几分钟)
#     REFUSE=1 PHOTO_THRESH=0.5 bash scripts/eval_local.sh   # 只重融合, 不重跑推理
#     SCORE=0 bash scripts/eval_local.sh              # 只出 ply, 不打分
#
# 为什么用 --fusion dedup 而不是 fusibile:
#   本机 RTX 5060 Ti 是 sm_120, 生成代码需要 CUDA >= 12.8; 而 fusibile 用的
#   texture reference API 在 CUDA 12 已被移除, 退回 CUDA 11 编译能过编译期但
#   CUDA 11 不支持 sm_120 —— 两个约束互斥, 这台机器上编不出可用的二进制。
#   dedup 是同一套思路的纯 torch 实现: 一个 ref 像素被写成点之后, 把它在各源
#   视角里对应的像素标记为已消费, 那些视角轮到当 ref 时跳过。
#
# 推理结果缓存在 log/depth_cache/test_<arm>, 所以调融合参数用 REFUSE=1,
# 不用重跑那 1078 个样本。
set -e
cd "$(dirname "$0")/.."
ROOT=$PWD

PY=${PY:-/home/william/miniconda3/envs/uprmvs/bin/python}
ARMS=${ARMS:-"D0 R_bp R_geo"}
FUSION=${FUSION:-dedup}          # geo | dedup | gipuma
PHOTO_THRESH=${PHOTO_THRESH:-0.3}
GEO_VIEWS=${GEO_VIEWS:-3}
RESIZE=${RESIZE:-0.8}            # 0.8 整幅 + 5 视角是记录在案的口径, 也正好卡住 16G
NUM_VIEWS=${NUM_VIEWS:-5}
NUM_WORKERS=${NUM_WORKERS:-8}
REFUSE=${REFUSE:-0}
SCORE=${SCORE:-1}
SMOKE=${SMOKE:-0}
TAG=${TAG:-$FUSION}              # ply 目录后缀, 便于并存对比不同融合/阈值

EVAL_TOOL=${EVAL_TOOL:-/home/william/Downloads/Fast-DTU-Evaluation}
EVAL_GT=${EVAL_GT:-/home/william/project/dataset/DTU/SampleSet/MVS Data}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="$ROOT:$ROOT/models:$ROOT/models/Depth-Anything-3/src"

SCANS_ARG=()
[[ "$SMOKE" == "1" ]] && SCANS_ARG=(--max-scans 1)

# --- 权重定位: 下载下来的可能是 <arm>/best.pth, 也可能是 <arm>/model/best.pth ---
declare -A CKPT
for a in $ARMS; do
    for c in "log/experiments/$a/best.pth" "log/experiments/$a/model/best.pth"; do
        [[ -f "$c" ]] && { CKPT[$a]=$c; break; }
    done
    [[ -n "${CKPT[$a]:-}" ]] || {
        echo "找不到 $a 的权重。找过:" >&2
        echo "    log/experiments/$a/best.pth" >&2
        echo "    log/experiments/$a/model/best.pth" >&2
        echo "现有: $(ls log/experiments/$a/ 2>/dev/null | tr '\n' ' ')" >&2
        exit 1
    }
done

echo "=== arms=$ARMS  fusion=$FUSION  photo_thresh=$PHOTO_THRESH  geo_views=$GEO_VIEWS ==="
echo "=== resize=$RESIZE  views=$NUM_VIEWS  refuse=$REFUSE  score=$SCORE  smoke=$SMOKE ==="
for a in $ARMS; do echo "    $a -> ${CKPT[$a]}"; done
"$PY" -c "import torch;p=torch.cuda.get_device_properties(0);print(f'    GPU: {p.name} sm_{p.major}{p.minor} {p.total_memory/2**30:.1f} GiB')"

if [[ "$FUSION" == "gipuma" ]]; then
    exe=${FUSIBILE_EXE:-third_party/fusibile/fusibile}
    [[ "$exe" = /* ]] || exe="$ROOT/$exe"
    [[ -x "$exe" ]] || { echo "找不到可执行的 fusibile: $exe" >&2
                         echo "本机编不出来 (见本脚本开头), 用 FUSION=dedup" >&2; exit 1; }
fi

# --- 推理 + 融合 ------------------------------------------------------------
for a in $ARMS; do
    echo ""
    echo "======================================================================"
    echo "=== $a  ($([[ $REFUSE == 1 ]] && echo 只重新融合 || echo 推理+融合)) ==="
    echo "======================================================================"
    common=(--split test --num-views "$NUM_VIEWS" --resize-scale "$RESIZE" --full-image
            --num-workers "$NUM_WORKERS" "${SCANS_ARG[@]}"
            --out "log/depth_cache/test_$a" --ply-dir "log/pred_points_${a}_${TAG}"
            --fuse --fusion "$FUSION" --photo-thresh "$PHOTO_THRESH" --geo-views "$GEO_VIEWS")
    if [[ "$REFUSE" == "1" ]]; then
        "$PY" test.py "${common[@]}" --fuse-only
    else
        "$PY" test.py "${common[@]}" --build-priors skip --vis 0 --ckpt "${CKPT[$a]}"
    fi
done

echo ""
echo "=== ply 大小 ==="
for a in $ARMS; do
    d="log/pred_points_${a}_${TAG}"
    echo "  $a: $(du -sh $d 2>/dev/null | cut -f1)  ($(ls $d/*.ply 2>/dev/null | wc -l) 个)"
done

# --- 打分 -------------------------------------------------------------------
if [[ "$SCORE" != "1" ]]; then
    echo ""
    echo "=== SCORE=0, 跳过打分。手动跑: ==="
    for a in $ARMS; do
        echo "  cd $EVAL_TOOL && python eval_dtu.py --method mvsnet --save \\"
        echo "      --pred_dir $ROOT/log/pred_points_${a}_${TAG} --gt_dir \"$EVAL_GT\""
    done
    exit 0
fi

[[ -f "$EVAL_TOOL/eval_dtu.py" ]] || { echo "找不到 $EVAL_TOOL/eval_dtu.py" >&2; exit 1; }
[[ -d "$EVAL_GT" ]] || { echo "找不到 DTU GT: $EVAL_GT" >&2; exit 1; }

if [[ "$SMOKE" == "1" ]]; then
    SCAN_IDS=(1)
else
    SCAN_IDS=($(sed 's/scan//' lists/dtu/test.txt))
fi

for a in $ARMS; do
    echo ""
    echo "======================================================================"
    echo "=== 打分 $a  (overall 与 Phase-1 的 0.326mm 比) ==="
    echo "======================================================================"
    ( cd "$EVAL_TOOL" && "$PY" eval_dtu.py \
        --scans "${SCAN_IDS[@]}" \
        --method mvsnet \
        --pred_dir "$ROOT/log/pred_points_${a}_${TAG}" \
        --gt_dir "$EVAL_GT" \
        --out_dir "$ROOT/log/dtu_eval/${a}_${TAG}" \
        --save )
done

echo ""
echo "======================================================================"
echo "=== 完成。打分结果在 log/dtu_eval/<arm>_${TAG}/ ==="
echo "=== 三个 arm 之间的差值才是结论; 绝对值受 photo_thresh 影响, 换阈值要三个一起换。"
echo "======================================================================"
