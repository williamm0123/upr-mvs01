#!/bin/bash -l
#SBATCH --job-name=plyeval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --time=12:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# =============================================================================
# D0 与 R 的点云 eval —— 终审判据, 不是 val abs_err。
#
#     sbatch scripts/eval_pointcloud.sh                 # 全部 22 个测试场景
#     SMOKE=1 sbatch scripts/eval_pointcloud.sh         # 先跑 scan1 验流程 (几分钟)
#     ARMS="R" sbatch scripts/eval_pointcloud.sh        # 只跑一个 arm
#
# 为什么必须跑: visibility 头和 gate 作用在多视角一致性判断与融合/过滤上, 而
# val abs_err 是在 GT 深度图上按像素算的单视角量 —— 融合阶段的收益在那个指标上
# 几乎不显影。拿 val abs_err 单独判这两个模块的取舍, 用的是量不到目标量的尺子。
#
# 两个 arm 在同一个作业里顺序跑, 因此保证同节点、同设置、同一份 prior 缓存 ——
# 它们之间的差值才是干净的。ply 和逐视角缓存按 arm 分目录, 不会互相覆盖。
#
# 打分是完全独立的第三步: 跑完拿 log/pred_points_<arm>/ 去 Fast-DTU-Evaluation,
# 本脚本不掺和。要比的是 Phase-1 的 overall 0.326mm。
#
# 分辨率: 沿用记录在案的 0.8 + 整幅 + 5 视角。A100 80GB 跑 1.0 绰绰有余 (16GB 卡
# 上才会 OOM), 但换了分辨率就跟历史所有数字不在同一个口径里 —— 要换就 D0/R 一起
# 换, 并且明确知道 Phase-1 的 0.326 是在哪个分辨率上量的。
# =============================================================================

set -e

source ~/.bashrc
conda activate uprmvs

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ARMS=${ARMS:-"D0 R"}          # 要评的 run 名, 空格分隔
SMOKE=${SMOKE:-0}             # 1 = 只跑 scan1
RESIZE=${RESIZE:-0.8}         # 见上面关于口径的说明
NUM_VIEWS=${NUM_VIEWS:-5}     # 与训练一致; 降到 3 明显变差
EXP_ROOT=log/experiments
# --- 融合后端 ---
# dedup  = 默认。内置的几何+光度一致性, 再加 fusibile 式的跨视角消费标记: 一个 ref
#          像素被写成点之后, 它在各源视角里对应的像素标记为已消费, 那些视角轮到当
#          ref 时跳过。同一个表面点只输出一次。纯 torch, 无外部依赖。
# geo    = 旧行为, **跨视角不去重** —— 49 视角下一个 scan 出 2500-4500 万点、
#          ply 约 600MB。留着只为对照。
# gipuma = 外部 fusibile 二进制。A100 是 sm_80, CUDA 11 支持, 所以 HPC 上可以编
#          (bash scripts/build_fusibile.sh, 需要 module load cuda/11.x —— CUDA 12
#          移除了它用的 texture reference API)。
FUSION=${FUSION:-dedup}
FUSIBILE_EXE=${FUSIBILE_EXE:-third_party/fusibile/fusibile}
PHOTO_THRESH=${PHOTO_THRESH:-0.3}   # MVSFormer++ 用 0.5; 换阈值要三个 arm 一起换
GEO_VIEWS=${GEO_VIEWS:-3}
TAG=${TAG:-$FUSION}                 # ply 目录后缀, 不同融合/阈值的结果可并存对比
# 打分 (可选): 装了 Fast-DTU-Evaluation 和 DTU GT 才跑, 否则只提示命令。
SCORE=${SCORE:-0}
EVAL_TOOL=${EVAL_TOOL:-$PWD/third_party/Fast-DTU-Evaluation}
EVAL_GT=${EVAL_GT:-/scr/user/qinglong/dataset/DTU/SampleSet/MVS Data}
# REFUSE=1: 跳过推理, 直接拿 log/depth_cache/test_<arm> 里已缓存的深度重新融合。
# 换融合方法或调阈值时用它 —— 重跑推理是每个 arm 1078 个样本的浪费。
REFUSE=${REFUSE:-0}

# --- 打分 (可选) -------------------------------------------------------------
score_all () {
    echo ""
    # 只报 du 是不够的: ply 头里写了顶点数, 文件大小必须等于 头 + 15*顶点数。
    # 对不上 = 写被截断 (配额/磁盘满), 而 [fuse] 那行照样打印了完整点数。
    echo "=== ply 完整性 ==="
    local bad=0
    for arm in $ARMS; do
        d="log/pred_points_${arm}_${TAG}"
        local n; n=$(ls "$d"/*.ply 2>/dev/null | wc -l)
        local msg; msg=$(python3 - "$d" <<'EOF'
import sys, glob, os
tot = short = 0
worst = ""
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.ply"))):
    with open(f, "rb") as fh:
        head = b""
        while b"end_header\n" not in head and len(head) < 4096:
            c = fh.read(1)
            if not c: break
            head += c
    try:
        nv = int([l for l in head.decode("ascii", "replace").splitlines()
                  if l.startswith("element vertex")][0].split()[-1])
    except Exception:
        print("头损坏 " + os.path.basename(f)); sys.exit(0)
    want, got = len(head) + 15 * nv, os.path.getsize(f)
    tot += nv
    if got != want:
        short += 1
        if not worst:
            worst = f"{os.path.basename(f)} {got:,}/{want:,} 字节"
print(f"{tot/1e6:.1f}M 点  " + (f"!! {short} 个被截断: {worst}" if short else "全部完整"))
EOF
)
        echo "    $arm: $(du -sh "$d" 2>/dev/null | cut -f1)  ($n 个)  $msg"
        [[ "$msg" == *"!!"* || "$msg" == *"损坏"* ]] && bad=1
    done
    if [[ $bad == 1 ]]; then
        echo ""
        echo "!! ply 被截断 —— 不要拿去打分。先查配额:  quota -s ; df -h /scr" >&2
        return 1
    fi

    if [[ "$SCORE" != "1" ]]; then
        echo ""
        echo "=== 打分是独立的第三步 (SCORE=1 可让本脚本代跑) ==="
        for arm in $ARMS; do
            echo "  python eval_dtu.py --method mvsnet --save \\"
            echo "      --pred_dir $PWD/log/pred_points_${arm}_${TAG} --gt_dir \"$EVAL_GT\""
        done
        echo "=== overall 与 Phase-1 的 0.326mm 比; arm 之间的差值才是结论 ==="
        return
    fi

    [[ -f "$EVAL_TOOL/eval_dtu.py" ]] || { echo "SCORE=1 但找不到 $EVAL_TOOL/eval_dtu.py" >&2; return 1; }
    [[ -d "$EVAL_GT" ]] || { echo "SCORE=1 但找不到 DTU GT: $EVAL_GT" >&2; return 1; }
    if [[ "$SMOKE" == "1" ]]; then SCAN_IDS=(1); else SCAN_IDS=($(sed 's/scan//' lists/dtu/test.txt)); fi
    for arm in $ARMS; do
        echo ""
        echo "=== 打分 $arm ==="
        ( cd "$EVAL_TOOL" && python eval_dtu.py \
            --scans "${SCAN_IDS[@]}" --method mvsnet --save \
            --pred_dir "$OLDPWD/log/pred_points_${arm}_${TAG}" \
            --gt_dir "$EVAL_GT" \
            --out_dir "$OLDPWD/log/dtu_eval/${arm}_${TAG}" )
    done
    echo ""
    echo "=== 打分结果在 log/dtu_eval/<arm>_${TAG}/ ==="
}

echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) ==="
echo "=== git=$(git rev-parse --short=12 HEAD) ==="
nvidia-smi -L
echo "=== arms=$ARMS resize=$RESIZE views=$NUM_VIEWS smoke=$SMOKE ==="
echo "=== fusion=$FUSION tag=$TAG photo_thresh=$PHOTO_THRESH geo_views=$GEO_VIEWS refuse=$REFUSE score=$SCORE ==="

if [[ "$FUSION" == "gipuma" ]]; then
    _exe="$FUSIBILE_EXE"; [[ "$_exe" = /* ]] || _exe="$PWD/$_exe"
    [[ -x "$_exe" ]] || { echo "找不到可执行的 fusibile: $_exe" >&2
                          echo "先 bash scripts/build_fusibile.sh, 或用 FUSIBILE_EXE=<路径> 指向已有的一份" >&2
                          exit 1; }
    echo "    fusibile -> $_exe"
fi

# --- REFUSE: 只重新融合, 不碰先验和推理 ------------------------------------
if [[ "$REFUSE" == "1" ]]; then
    for arm in $ARMS; do
        echo ""
        echo "=== arm=$arm 重新融合 (backend=$FUSION) ==="
        PHASE=fuse SMOKE="$SMOKE" NUM_VIEWS="$NUM_VIEWS" RESIZE="$RESIZE" FULL_IMAGE=1 FUSE=1 \
            FUSION="$FUSION" FUSIBILE_EXE="$FUSIBILE_EXE" \
        PHOTO_THRESH="$PHOTO_THRESH" GEO_VIEWS="$GEO_VIEWS" \
            PHOTO_THRESH="$PHOTO_THRESH" GEO_VIEWS="$GEO_VIEWS" \
            OUT="log/depth_cache/test_$arm" \
            PLY_DIR="log/pred_points_${arm}_${TAG}" \
            bash scripts/test_dtu.sh
    done
    score_all
    exit 0
fi

# --- 先确认权重都在, 免得建完 prior 才发现少一个 --------------------------
declare -A CKPT
for arm in $ARMS; do
    # HPC 上训练写的是 <arm>/model/best.pth; 从别处拷过来的可能少一层。两种都认。
    for c in "$EXP_ROOT/$arm/model/best.pth" "$EXP_ROOT/$arm/best.pth"; do
        [[ -f "$c" ]] && { CKPT[$arm]=$c; break; }
    done
    [[ -n "${CKPT[$arm]:-}" ]] || {
        echo "找不到 $arm 的权重。找过 $EXP_ROOT/$arm/{model/,}best.pth" >&2
        echo "现有: $(ls "$EXP_ROOT/$arm" 2>/dev/null | tr '\n' ' ')" >&2
        exit 1
    }
    echo "    $arm -> ${CKPT[$arm]}"
done

# --- 阶段 1: 测试集 prior, 只建一次, 两个 arm 共用 ------------------------
# 权重完全不参与 prior, 所以这一步跟 arm 无关。放在循环外是为了让两个 arm 吃到
# 逐字节相同的一份缓存 —— 否则"差值"里会混进 prior 的重建噪声。
echo "=== 阶段 1: 建测试集 prior (两个 arm 共用) ==="
PHASE=priors SMOKE="$SMOKE" BUILD_PRIORS=auto \
    NUM_VIEWS="$NUM_VIEWS" RESIZE="$RESIZE" \
    bash scripts/test_dtu.sh

# --- 阶段 2: 逐 arm 推理 + 融合 -------------------------------------------
for arm in $ARMS; do
    echo ""
    echo "======================================================================"
    echo "=== arm=$arm 推理 + 融合 ==="
    echo "======================================================================"
    PHASE=infer BUILD_PRIORS=skip SMOKE="$SMOKE" \
        NUM_VIEWS="$NUM_VIEWS" RESIZE="$RESIZE" FULL_IMAGE=1 FUSE=1 \
        FUSION="$FUSION" FUSIBILE_EXE="$FUSIBILE_EXE" \
        PHOTO_THRESH="$PHOTO_THRESH" GEO_VIEWS="$GEO_VIEWS" \
        CKPT="${CKPT[$arm]}" \
        OUT="log/depth_cache/test_$arm" \
        PLY_DIR="log/pred_points_${arm}_${TAG}" \
        bash scripts/test_dtu.sh
done

echo ""
echo "======================================================================"
echo "=== 完成。ply 在:"
for arm in $ARMS; do echo "===   $arm -> log/pred_points_${arm}_${TAG}/"; done
echo "======================================================================"
score_all
