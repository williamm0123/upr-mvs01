#!/bin/bash -l
#SBATCH --job-name=uprmvs_ply
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --qos=normal
#SBATCH --time=3:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# 工单 v5.3 §7.2 / §7.5 的点云推理 —— **sbatch 提交, 不是 bash**。
#
#   WHICH=w0    sbatch scripts/sbatch_test_dtu.sh     # W0 基线点云 (§7.2)
#   WHICH=calib sbatch scripts/sbatch_test_dtu.sh     # Platt 标定     (§7.4)
#   WHICH=vnext sbatch scripts/sbatch_test_dtu.sh     # vNext 终审点云 (§7.5)
#
# calib 也在这里而不是另起一个脚本: 它同样要 GPU (calibrate_conf.py 是
# `cuda if available else cpu`, 在登录节点上会**静默**退回 CPU 然后跑一整个
# val split), 而且要与 §7.5 同一份 conda / PYTHONPATH / profile。两个脚本各
# 抄一份环境段, 迟早只改一边。
#
# 为什么要这个包装: scripts/test_dtu.sh **没有 #SBATCH 头**, 它是"在已经拿到卡
# 的地方跑"的脚本。直接 `bash scripts/test_dtu.sh` 会跑在登录节点上 (没有 GPU,
# 而且这是几小时的作业); 直接 `sbatch scripts/test_dtu.sh` 则拿不到 --gres=gpu:1。
# 本脚本负责排队与环境, 拿到卡之后再 bash 调它。
#
# 两条协议的差别只有三个量 (checkpoint / 置信度来源 / 输出目录), 其余逐字相同 ——
# 这是配对比较的前提, 所以它们写在同一个 case 里而不是两份脚本。
# scripts/compare_pointclouds.py 会核对两边 run_manifest.json 的这些参数。
#
# 打分是**独立的第三步**, 本脚本不掺和: 跑完拿 PLY_DIR 去 Fast-DTU-Evaluation。
# =============================================================================

set -euo pipefail

WHICH=${WHICH:-w0}
case "$WHICH" in w0|calib|vnext) ;; *)
    echo "WHICH 只能是 w0 / calib / vnext, 收到 '$WHICH'" >&2; exit 2 ;; esac

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

case "$WHICH" in
  w0)
    # 现有 log/pred_points_R_geo_dedup 的谱系不明 (只有 result.txt, 没有
    # manifest, 而 W0 的 ckpt 更晚), 所以基线必须用 W0 自己重跑一遍。
    CKPT_DEF=log/experiments/R32_f10_30k/model/latest.pth
    CONF_SOURCE_DEF=cascade          # W0 没有置信度头
    OUT_DEF=log/depth_cache/W0_final
    PLY_DEF=log/pred_points_W0_final
    ;;
  calib)
    # 标定跑在 **val** split 上 (calibrate_conf.py 的 eval_namespace 默认),
    # 不是 test —— 拿终审集去拟合温度, 那个终审就不独立了。
    CKPT_DEF=log/experiments/UPRMVS_vNext/model/latest.pth
    CALIB_OUT=${CALIB_OUT:-log/experiments/UPRMVS_vNext/model/latest_calibrated.pth}
    CALIB_REPORT=${CALIB_REPORT:-experiments/out/calibrate_vnext.json}
    REQUIRE_STEP=${REQUIRE_STEP:-30000}
    TAU_MM=${TAU_MM:-2.0}                # 必须等于训练的 --conf-tau-mm
    CONF_SOURCE_DEF=learned              # 仅用于回显, 标定本身不走这条路
    OUT_DEF=- ; PLY_DEF=-
    ;;
  vnext)
    # **标定过的**那份。未标定的 logit 读不成概率, 而 --conf-source learned
    # 的整个意义就是让 photo 门有物理含义。
    CKPT_DEF=log/experiments/UPRMVS_vNext/model/latest_calibrated.pth
    CONF_SOURCE_DEF=learned
    OUT_DEF=log/depth_cache/UPRMVS_vNext_final
    PLY_DEF=log/pred_points_UPRMVS_vNext_final
    ;;
esac

# --- 两条协议逐字相同的部分 (v5.3 §7.2/§7.5) ---
CKPT=${CKPT:-$CKPT_DEF}
CONF_SOURCE=${CONF_SOURCE:-$CONF_SOURCE_DEF}
OUT=${OUT:-$OUT_DEF}
PLY_DIR=${PLY_DIR:-$PLY_DEF}
MIN_STEP=${MIN_STEP:-30000}          # 拦下训练中断的残权重
NUM_VIEWS=${NUM_VIEWS:-5}
RESIZE=${RESIZE:-0.8}                # test_dtu.sh 的默认是 1.0, 必须显式给
PRIOR_RESIZE=${PRIOR_RESIZE:-1.0}    # 与 RESIZE 独立, 改它才要 BUILD_PRIORS=force
FULL_IMAGE=${FULL_IMAGE:-1}
MAX_REFS=${MAX_REFS:-0}              # 0 = 全部 49 个参考视角, 融合必须用全部
BUILD_PRIORS=${BUILD_PRIORS:-skip}
PHOTO_KEEP_RATIO=${PHOTO_KEEP_RATIO:-0.60}   # 固定保留率, 取代阈值扫描
FUSION=${FUSION:-dedup}
SPLIT=${SPLIT:-test}
SMOKE=${SMOKE:-0}

if [[ ! -f "$CKPT" ]]; then
    echo "找不到 checkpoint: $CKPT" >&2
    echo "  (WHICH=$WHICH 的默认路径是 $CKPT_DEF; 可以用 CKPT=... 覆盖)" >&2
    exit 1
fi

set +u
source ~/.bashrc
conda activate uprmvs
set -u

mkdir -p logs
export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "======================================================================"
echo " 点云推理  which=$WHICH  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " ckpt=$CKPT"
echo " conf_source=$CONF_SOURCE  photo_keep_ratio=$PHOTO_KEEP_RATIO  fusion=$FUSION"
echo " split=$SPLIT resize=$RESIZE views=$NUM_VIEWS full_image=$FULL_IMAGE max_refs=$MAX_REFS"
echo " out=$OUT  ply=$PLY_DIR"
echo " git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "======================================================================"

if [[ "$WHICH" == "calib" ]]; then
    # 与 §7.5 同一套部署口径 (0.8 整幅 + 5 视角): 标定要在**部署条件**下做,
    # 换了分辨率 logit 的分布就变了, 拟合出来的 T/b 对不上要用它的那次推理。
    python scripts/calibrate_conf.py \
        --ckpt "$CKPT" \
        --require-step "$REQUIRE_STEP" \
        --write-out "$CALIB_OUT" \
        --tau-mm "$TAU_MM" \
        --resize-scale "$RESIZE" --num-views "$NUM_VIEWS" --full-image on \
        --num-workers 8 \
        --out "$CALIB_REPORT"
    echo ""
    echo "=== 完成。标定过的 ckpt -> $CALIB_OUT   报告 -> $CALIB_REPORT"
    echo "=== 下一步:  WHICH=vnext sbatch scripts/sbatch_test_dtu.sh"
    exit 0
fi

# 拿到卡之后才 bash 调它 —— test_dtu.sh 自己不排队。
CKPT="$CKPT" MIN_STEP="$MIN_STEP" NUM_VIEWS="$NUM_VIEWS" RESIZE="$RESIZE" \
PRIOR_RESIZE="$PRIOR_RESIZE" FULL_IMAGE="$FULL_IMAGE" MAX_REFS="$MAX_REFS" \
BUILD_PRIORS="$BUILD_PRIORS" CONF_SOURCE="$CONF_SOURCE" \
PHOTO_KEEP_RATIO="$PHOTO_KEEP_RATIO" FUSION="$FUSION" \
SPLIT="$SPLIT" SMOKE="$SMOKE" OUT="$OUT" PLY_DIR="$PLY_DIR" \
    bash scripts/test_dtu.sh

echo ""
echo "=== 完成。ply -> $PLY_DIR"
echo "=== 打分是独立的第三步 (本脚本不做):"
echo "      cd <Fast-DTU-Evaluation> && python eval_dtu.py --method mvsnet --save \\"
echo "          --pred_dir $PROJECT_DIR/$PLY_DIR --gt_dir <GT>"
echo "=== 两边都打完分之后:"
echo "      python scripts/compare_pointclouds.py \\"
echo "          --base log/pred_points_W0_final --cand log/pred_points_UPRMVS_vNext_final"
