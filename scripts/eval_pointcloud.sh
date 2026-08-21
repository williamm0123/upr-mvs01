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
# geo    = 内置的几何+光度一致性。每个 ref 视角各输出一遍存活像素, **跨视角不去重**,
#          49 视角下一个 scan 出 2500-4500 万点、ply 约 600MB。
# gipuma = 调 fusibile 二进制, 把跨视角一致的点聚成一个, 通常降到 1-3M 点。
#          需要先编译: bash scripts/build_fusibile.sh (或指向已有的一份)。
FUSION=${FUSION:-geo}
FUSIBILE_EXE=${FUSIBILE_EXE:-third_party/fusibile/fusibile}
# REFUSE=1: 跳过推理, 直接拿 log/depth_cache/test_<arm> 里已缓存的深度重新融合。
# 换融合方法或调阈值时用它 —— 重跑推理是每个 arm 1078 个样本的浪费。
REFUSE=${REFUSE:-0}

echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) ==="
echo "=== git=$(git rev-parse --short=12 HEAD) ==="
nvidia-smi -L
echo "=== arms=$ARMS resize=$RESIZE views=$NUM_VIEWS smoke=$SMOKE fusion=$FUSION refuse=$REFUSE ==="

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
            OUT="log/depth_cache/test_$arm" \
            PLY_DIR="log/pred_points_${arm}_${FUSION}" \
            bash scripts/test_dtu.sh
    done
    echo ""
    for arm in $ARMS; do echo "===   $arm -> log/pred_points_${arm}_${FUSION}/"; done
    exit 0
fi

# --- 先确认权重都在, 免得建完 prior 才发现少一个 --------------------------
for arm in $ARMS; do
    ck="$EXP_ROOT/$arm/model/best.pth"
    [[ -f "$ck" ]] || { echo "找不到 $ck" >&2; exit 1; }
    echo "    $arm -> $ck"
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
        CKPT="$EXP_ROOT/$arm/model/best.pth" \
        OUT="log/depth_cache/test_$arm" \
        PLY_DIR="log/pred_points_$arm" \
        bash scripts/test_dtu.sh
done

echo ""
echo "======================================================================"
echo "=== 完成。ply 在:"
for arm in $ARMS; do echo "===   $arm -> log/pred_points_$arm/"; done
echo "=== 下一步 (独立第三步): 用 Fast-DTU-Evaluation 对这两个目录打分,"
echo "===   overall 与 Phase-1 的 0.326mm 比。两个 arm 之间的差值才是本次结论。"
echo "======================================================================"
