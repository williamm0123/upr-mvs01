#!/bin/bash -l
#
# DTU 测试，两个阶段：
#   阶段 1  建 prior：VGGT + DA3 逐视角算深度先验，存进 log/prior_cache
#   阶段 2  推理+融合：加载训练权重出深度图 -> 光度/几何一致性融合 -> log/pred_points
#
# 拆成两个进程是有意的：阶段 1 结束后 VGGT/DA3 彻底退出，阶段 2 才有完整显存；
# 而且阶段 1 中断可以用 BUILD_PRIORS=auto 续，不用从头再来。
#
# 官方打分是第三步、完全独立，用 Fast-DTU-Evaluation 对着 ply 目录跑，本脚本不掺和。
# 所有数据集路径都来自 base/config.py 的 ProjectPaths（跟着 REPO_ROOT 走，本机和
# 集群自动各自解析），这里只放"这次要怎么跑"的运行时开关。
#
# 用法：
#   BUILD_PRIORS=force bash scripts/test_dtu.sh   # 全部重建 prior 再跑完整流程
#   bash scripts/test_dtu.sh                      # prior 缺啥补啥，然后跑完整流程
#   PHASE=priors bash scripts/test_dtu.sh         # 只建 prior
#   PHASE=infer  bash scripts/test_dtu.sh         # 只推理+融合（要求 prior 齐全）
#   SMOKE=1 bash scripts/test_dtu.sh              # 先跑 scan1 一个场景验证流程
#   FUSE=0 bash scripts/test_dtu.sh               # 只要深度图指标，不融合
#
set -euo pipefail

# 从脚本自身位置推出仓库根目录，本机 / 集群都不用改
PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONDA_ENV=${CONDA_ENV:-uprmvs}
GPU_ID=${GPU_ID:-0}

# ---- 跑哪些数据 -------------------------------------------------------------
PHASE=${PHASE:-all}               # all | priors | infer | fuse
# fuse = 只重新融合已缓存的深度, 不跑先验也不跑推理。换融合方法/调阈值时用它。
# dedup = 内置融合 + fusibile 式跨视角去重 (默认); geo = 不去重的旧行为;
# gipuma = 外部 fusibile 二进制
FUSION=${FUSION:-dedup}
FUSIBILE_EXE=${FUSIBILE_EXE:-third_party/fusibile/fusibile}
GIPUMA_DISP=${GIPUMA_DISP:-0.25}  # MVSFormer++ 的取值
GIPUMA_NC=${GIPUMA_NC:-3}
SPLIT=${SPLIT:-test}              # test = 官方 22 个测试场景；val 用来和训练日志对齐
SMOKE=${SMOKE:-0}                 # 1 = 只跑第一个 scan（scan1），流程验证用
MAX_SCANS=${MAX_SCANS:-0}         # 0 = 全部；SMOKE=1 时强制为 1
MAX_REFS=${MAX_REFS:-0}           # 0 = 每个 scan 全部 49 个参考视角（融合必须用全部）
SCANS=${SCANS:-}                  # 空 = 全部；"1 4 9" = 只跑这几个。多卡切分用：
                                  #   GPU_ID=0 SCANS="1 4 9 10 11 12" OUT=outputs/t0 ...
                                  # ply 按 scan 编号命名，几个作业共用 PLY_DIR 不会撞；
                                  # 但 OUT 必须各不相同（逐视角 npz 缓存 + metrics.json）。

# ---- prior ------------------------------------------------------------------
BUILD_PRIORS=${BUILD_PRIORS:-auto}  # auto=缺啥补啥 / force=全部重建 / skip=要求已齐全
PRIOR_RESIZE=${PRIOR_RESIZE:-1.0}   # prior 按原生 1200x1600 建，与下面的推理分辨率无关。
                                    # 存最高分辨率，推理时下采样使用（升采样才有损），
                                    # 且 SfM 公制尺度标定跑在最清晰的图上。
                                    # 改了这个必须配 BUILD_PRIORS=force —— 缓存文件名
                                    # 只编码 (scan, ref_view, light)，不含分辨率。

# ---- 推理设置 ---------------------------------------------------------------
NUM_VIEWS=${NUM_VIEWS:-5}         # 与训练一致：1 参考 + 4 源视图。降到 3 会明显变差
                                  # （scan1 实测 abs_err 0.493 -> 0.560），别为省显存动它。
RESIZE=${RESIZE:-1.0}             # 1600x1200 -> 1280x960
FULL_IMAGE=${FULL_IMAGE:-1}       # 1 = 整幅不裁剪。裁剪只重建 68% 画面，完整度虚高。
                                  # 5060Ti 16G 上限：0.8+整幅+5视角 = 9.26 GiB 可以；
                                  # 1.0+整幅+5视角 = 14.22 GiB 实跑 OOM。
NUM_WORKERS=${NUM_WORKERS:-8}
VIS=${VIS:-0}                     # 每个 scan 存前 N 张深度可视化 png

# ---- 融合 -------------------------------------------------------------------
FUSE=${FUSE:-1}                   # 0 = 只算深度图指标，不融合
PHOTO_THRESH=${PHOTO_THRESH:-0.3} # stage4 众数概率阈值，调高 -> acc 好 comp 差
GEO_VIEWS=${GEO_VIEWS:-3}         # 最少几个源视图几何一致
GEO_PIX=${GEO_PIX:-1.0}           # 重投影误差上限（像素）
GEO_REL=${GEO_REL:-0.01}          # 相对深度差上限
PLY_DIR=${PLY_DIR:-}              # 留空 = config 的 pred_points_path（<project>/log/pred_points）

# ---- checkpoint -------------------------------------------------------------
CKPT=${CKPT:-}                    # 显式指定权重文件；留空则用 CKPT_DIR 里的 best.pth
CKPT_DIR=${CKPT_DIR:-log/model}    # 训练输出目录，best.pth = val 最好的那一步
MIN_STEP=${MIN_STEP:-1000}        # 低于这个步数直接拦下来，防止拿训练中断的残权重白跑
ALLOW_STALE=${ALLOW_STALE:-0}     # 1 = 跳过上面的检查

OUT=${OUT:-}                      # 逐视角 npz 缓存 + metrics.json 的去处（不是 ply 的去处，
                                  # ply 看 PLY_DIR）。留空 = config 的 depth_cache_path/<split>，
                                  # 即 <project>/log/depth_cache/<split>。多作业切分时要各给一个。
# =============================================================================

cd "$PROJECT_DIR"

# 依次试几个候选：集群上 conda 多半不在 PATH（训练脚本就是刻意绕开它的），
# env 装在 ~/.conda/envs 而不是 ~/miniconda3/envs。任何一个可执行就用哪个。
if [[ -z "${PYTHON_BIN:-}" ]]; then
    for _cand in \
        "${ENV_PREFIX:-}/bin/python" \
        "$HOME/.conda/envs/$CONDA_ENV/bin/python" \
        "$HOME/miniconda3/envs/$CONDA_ENV/bin/python" \
        "$(conda info --base 2>/dev/null)/envs/$CONDA_ENV/bin/python"
    do
        [[ -n "$_cand" && -x "$_cand" ]] && { PYTHON_BIN="$_cand"; break; }
    done
fi

if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
    echo "找不到 $CONDA_ENV 环境的解释器。试过：" >&2
    echo "  \${ENV_PREFIX}/bin/python, ~/.conda/envs/$CONDA_ENV/bin/python," >&2
    echo "  ~/miniconda3/envs/$CONDA_ENV/bin/python, \$(conda info --base)/envs/$CONDA_ENV/bin/python" >&2
    echo "显式指定：PYTHON_BIN=/home/user/qinglong/.conda/envs/uprmvs/bin/python bash $0" >&2
    exit 1
fi
echo "=== python: $PYTHON_BIN ==="

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
# 不设这个，整幅推理会在 reserved-but-unallocated 上碎掉：实测 9 GiB 卡在碎片里，
# 明明还有余量却 OOM。训练脚本一直设着，测试脚本之前漏了。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

case "$PHASE" in
    all|priors|infer|fuse) ;;
    *) echo "PHASE 必须是 all / priors / infer / fuse，收到: $PHASE" >&2; exit 2 ;;
esac

case "$SMOKE" in
    1|true|TRUE|yes|YES) MAX_SCANS=1 ;;
    0|false|FALSE|no|NO) ;;
    *) echo "SMOKE 必须是 0/1，收到: $SMOKE" >&2; exit 2 ;;
esac

# --- 数据路径全部由 config 决定，这里只是打印出来让人看得见 -------------------
"$PYTHON_BIN" - <<'PY'
from base.config import MACHINE, ProjectPaths
p = ProjectPaths()
print(f"=== machine={MACHINE} 数据路径（来自 base/config.py，不在本脚本里定义）===")
for name in ("project_path", "dtu_train_root", "test_list_file",
             "prior_cache_path", "depth_cache_path", "pred_points_path"):
    v = getattr(p, name)
    print(f"    {name:18s} = {v}{'' if v.exists() else '   <<< 不存在'}")
PY

# --- checkpoint 解析与新鲜度检查（只建 prior 时用不到权重，跳过）--------------
CKPT_ARGS=()
if [[ "$PHASE" == "priors" ]]; then
    echo "=== PHASE=priors：不加载权重，跳过 checkpoint 检查 ==="
elif [[ -n "$CKPT" ]]; then
    [[ -f "$CKPT" ]] || { echo "CKPT 不存在: $CKPT" >&2; exit 1; }
    CKPT_ARGS=(--ckpt "$CKPT")
    CKPT_SHOWN="$CKPT"
else
    CKPT_SHOWN="$CKPT_DIR/best.pth"
    [[ -f "$CKPT_SHOWN" ]] || CKPT_SHOWN="$CKPT_DIR/latest.pth"
    if [[ ! -f "$CKPT_SHOWN" ]]; then
        echo "在 $CKPT_DIR 里找不到 best.pth / latest.pth。" >&2
        echo "从集群拷一份下来（训练已跑完 30k 步）：" >&2
        echo "  mkdir -p $CKPT_DIR && scp login01.dicc.um.edu.my:/scr/user/qinglong/projects/upr-mvs01/log/model/best.pth $CKPT_DIR/" >&2
        exit 1
    fi
    CKPT_ARGS=(--ckpt-dir "$CKPT_DIR")
fi

# 权重里既不存配置也不存代码版本，只能从 key 集合反推：cost_builders.<i> 的个数
# 就是级联级数，和当前代码的 UprMVSNet.fpn_stage_strides 对不上就是旧架构的权重。
if [[ "$PHASE" != "priors" ]]; then
CKPT_INFO=$("$PYTHON_BIN" - "$CKPT_SHOWN" <<'PY'
import sys, torch
from models.network import UprMVSNet
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
keys = list(ck["model"])
stages = len({k.split(".")[1] for k in keys if k.startswith("cost_builders.")})
print(int(ck.get("step", -1)), float(ck.get("best_metric", float("nan"))),
      int(any(k.startswith("spre.") for k in keys)), stages, len(UprMVSNet.fpn_stage_strides))
PY
)
read -r CKPT_STEP CKPT_METRIC CKPT_SPRE CKPT_STAGES CODE_STAGES <<<"$CKPT_INFO"
echo "=== checkpoint: $CKPT_SHOWN  step=$CKPT_STEP  best_metric=$CKPT_METRIC  spre=$CKPT_SPRE  stages=$CKPT_STAGES ==="

if [[ "$CKPT_STAGES" != "$CODE_STAGES" ]]; then
    echo "" >&2
    echo "架构不匹配：这份权重是 $CKPT_STAGES 级级联，当前代码是 $CODE_STAGES 级，load_state_dict 一定会崩。" >&2
    echo "（多半是翻出了 4 级重构之前的旧权重，换成本次训练的。）" >&2
    exit 1
fi

if [[ "$ALLOW_STALE" != "1" && "$CKPT_STEP" -lt "$MIN_STEP" ]]; then
    echo "" >&2
    echo "这份权重只有 $CKPT_STEP 步（< MIN_STEP=$MIN_STEP），是训练中断留下的残骸，不是 30k 的模型。" >&2
    echo "从集群拷正式权重：" >&2
    echo "  scp login01.dicc.um.edu.my:/scr/user/qinglong/projects/upr-mvs01/log/model/best.pth $CKPT_DIR/" >&2
    echo "确认要用这份跑，就加 ALLOW_STALE=1。" >&2
    exit 1
fi
fi  # PHASE != priors

# 两个阶段共用的参数。注意 --num-views 也进阶段 1: prior 是多视角算的
# (VGGT 吃 1 ref + N-1 src)，视角数变了 prior 就得重建。
common_args=(
    --split "$SPLIT"
    --num-views "$NUM_VIEWS"
    --resize-scale "$RESIZE"
    --num-workers "$NUM_WORKERS"
    --max-scans "$MAX_SCANS"
    --max-refs "$MAX_REFS"
)
[[ -n "$OUT" ]] && common_args+=(--out "$OUT")
# 不加引号: 要按空格拆成多个参数喂给 nargs="+"
[[ -n "$SCANS" ]] && common_args+=(--scans $SCANS)

case "$FULL_IMAGE" in
    1|true|TRUE|yes|YES) common_args+=(--full-image) ;;
    0|false|FALSE|no|NO) ;;
    *) echo "FULL_IMAGE 必须是 0/1，收到: $FULL_IMAGE" >&2; exit 2 ;;
esac

infer_args=(--vis "$VIS" "${CKPT_ARGS[@]}")
case "$FUSE" in
    1|true|TRUE|yes|YES)
        infer_args+=(
            --fuse
            --photo-thresh "$PHOTO_THRESH"
            --geo-views "$GEO_VIEWS"
            --geo-pix "$GEO_PIX"
            --geo-rel "$GEO_REL"
        )
        infer_args+=(--fusion "$FUSION")
        if [[ "$FUSION" == "gipuma" ]]; then
            infer_args+=(
                --fusibile-exe "$FUSIBILE_EXE"
                --gipuma-disp-thresh "$GIPUMA_DISP"
                --gipuma-num-consistent "$GIPUMA_NC"
            )
        fi
        [[ -n "$PLY_DIR" ]] && infer_args+=(--ply-dir "$PLY_DIR")
        # 几何一致性只在"已缓存的"视角之间做：截断参考视角会把源视角一起截掉，
        # GEO_VIEWS 个一致视角凑不齐，整个 scan 融出 0 个点。
        if [[ "$MAX_REFS" -gt 0 && "$MAX_REFS" -le 16 ]]; then
            echo "!!! MAX_REFS=$MAX_REFS 且开着融合：源视角不足，点云大概率是空的。" >&2
            echo "!!! 要真的出点云就让 MAX_REFS=0（全部 49 个视角）。" >&2
        fi
        ;;
    0|false|FALSE|no|NO) infer_args+=(--no-fuse) ;;
    *) echo "FUSE 必须是 0/1，收到: $FUSE" >&2; exit 2 ;;
esac

echo "=== phase=$PHASE split=$SPLIT scans=$( [[ $MAX_SCANS -eq 0 ]] && echo all || echo $MAX_SCANS ) \
views=$NUM_VIEWS resize=$RESIZE full_image=$FULL_IMAGE fuse=$FUSE fusion=$FUSION out=$OUT ==="

# ---- PHASE=fuse: 只重新融合 --------------------------------------------------
if [[ "$PHASE" == "fuse" ]]; then
    echo "=== 只重新融合 (backend=$FUSION), 不跑先验也不跑推理 ==="
    echo "+ python test.py ${common_args[*]} --fuse-only ${infer_args[*]}"
    exec "$PYTHON_BIN" test.py "${common_args[@]}" --fuse-only "${infer_args[@]}"
fi

# ---- 阶段 1: 建 prior --------------------------------------------------------
if [[ "$PHASE" == "all" || "$PHASE" == "priors" ]]; then
    echo "=== 阶段 1: prior (mode=$BUILD_PRIORS, prior_resize=$PRIOR_RESIZE) ==="
    echo "+ python test.py ${common_args[*]} --build-priors $BUILD_PRIORS --prior-resize-scale $PRIOR_RESIZE --priors-only --no-fuse"
    "$PYTHON_BIN" test.py "${common_args[@]}" \
        --build-priors "$BUILD_PRIORS" \
        --prior-resize-scale "$PRIOR_RESIZE" \
        --priors-only --no-fuse
fi

if [[ "$PHASE" == "priors" ]]; then
    echo "=== PHASE=priors，到此为止。推理用 PHASE=infer 接着跑 ==="
    exit 0
fi

# ---- 阶段 2: 推理 + 融合 -----------------------------------------------------
# PHASE=all 时阶段 1 刚跑完，缓存必然齐全，用 skip 省掉一次全量 stat；
# PHASE=infer 单独跑时按 BUILD_PRIORS 兜底，缺的就地补，别炸在 worker 里。
[[ "$PHASE" == "all" ]] && PHASE2_PRIORS=skip || PHASE2_PRIORS="$BUILD_PRIORS"
[[ "$PHASE2_PRIORS" == "force" ]] && PHASE2_PRIORS=auto   # 别在推理阶段重建一遍
echo "=== 阶段 2: 推理 + 融合 (priors=$PHASE2_PRIORS) ==="
echo "+ python test.py ${common_args[*]} --build-priors $PHASE2_PRIORS ${infer_args[*]}"
exec "$PYTHON_BIN" test.py "${common_args[@]}" \
    --build-priors "$PHASE2_PRIORS" \
    --prior-resize-scale "$PRIOR_RESIZE" \
    "${infer_args[@]}"
