#!/bin/bash -l
#SBATCH --job-name=uprmvs_sfm_da3
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# 全量 SfM 稀疏点云 + DA3 metric 标定 —— **sbatch 提交, 不是 bash**。
#
#   cd /scr/user/qinglong/projects/upr-mvs01
#   git pull
#   sbatch scripts/sbatch_build_sfm_da3_calib.sh
#
# 两个阶段, 按顺序跑:
#   1. sfm    scripts/build_sfm_cache_all.py      -> log/sfm_cache/<scan>/sfm_{view:04d}_3.npz
#             Rectified_raw/ 下全部 scan x 49 视角, 只用 light 3, DTU 相机已知, GPU 做极线引导匹配。
#   2. calib  scripts/calibrate_da3_with_sfm.py   -> log/da3_sfm_cache/<scan>/da3_{view:04d}_{light}.npz
#             log/da3_cache 里**有的**每个 (scan, view, light) 都用该视角的 light-3 点云做深度域
#             a*d+b 标定 (7 种光照共用一份点云, 各自拟合参数), 纯 CPU。
#
# 依赖: 阶段 2 读 log/da3_cache (sbatch_build_da3_cache.sh 的产物)。那边没建完也能跑, 只标定
# 已经存在的文件; 之后 DA3 补齐了再提交一次本脚本即可 (两个阶段都按文件断点续跑, 已完成的跳过,
# 阶段 1 会秒过)。
#
# 本地 5060 Ti 实测 (6 个 scan): SfM 每 scan 15~165s (纹理越多越慢, 瓶颈是 GPU 匹配, 2 进程共卡),
# 标定约 0.7s/文件/进程 (含 GT 上界对照)。外推到 A100 + 16 核: 阶段 1 约 0.5~1 小时, 阶段 2
# (42532 个文件) 约 0.5~1 小时, 8 小时留足余量。显存每个 SfM 进程峰值约 2GiB。
#
# 可调 (sbatch --export=ALL,变量=值 ... 或提交前 export):
#   STAGES="sfm calib"   只跑某一段就写 STAGES=calib
#   SFM_WORKERS=4        阶段 1 并行 scan 数 (每个一份 CUDA context, SIFT 在 CPU)
#   CALIB_WORKERS=15     阶段 2 进程数
#   SFM_ARGS=""          透传给 build_sfm_cache_all.py, 例如 "--scans 1 4 9" / "--force"
#   CALIB_ARGS="--gt-oracle"
#                        透传给 calibrate_da3_with_sfm.py。--gt-oracle 只是在汇总表里多算一列
#                        "DA3 直接对 GT 仿射" 的上界做对照 (约多 15 分钟), 不影响输出文件
#
# 结果怎么看:
#   log/sfm_cache/_sfm_summary_<时间戳>.csv       每视角点数 / 与 GT 的偏差 (GT 只做统计)
#   log/da3_sfm_cache/_calib_summary_<时间戳>.csv 每文件 a,b / extrap_frac / gt_med / oracle_med
#   *_failures_<时间戳>.csv                       失败清单 (没有失败就不留)
#   extrap_frac >= 0.5 的视角 (白桌面整片过曝、没有 SfM 点的 scan, 如 scan77) 标定误差明显变大,
#   下游可以按它或 support_depth 过滤。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

STAGES=${STAGES:-"sfm calib"}
SFM_WORKERS=${SFM_WORKERS:-4}
CALIB_WORKERS=${CALIB_WORKERS:-15}
SFM_ARGS=${SFM_ARGS:-}
CALIB_ARGS=${CALIB_ARGS:-"--gt-oracle"}

# conda 的 activate.d 脚本会读 LD_LIBRARY_PATH 这类未必存在的变量; set -u 下
# 会直接 unbound variable 退出。只在激活期间关掉 nounset。
set +u
source ~/.bashrc
conda activate uprmvs
set -u

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# 并行靠多进程, 每个进程内部的 BLAS/OpenMP 线程不要再各自吃满 16 核
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p logs

echo "=================================================================="
echo " job=${SLURM_JOB_ID:-manual}  host=$(hostname)  stages=[$STAGES]"
echo " git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=================================================================="
nvidia-smi -L || true

for stage in $STAGES; do
    case "$stage" in
        sfm)
            python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
                || { echo "CUDA 不可用 —— SfM 阶段需要 --gres=gpu:1" >&2; exit 1; }
            echo; echo "################ 阶段 1: SfM 稀疏点云 (light 3) ################"
            date
            # shellcheck disable=SC2086
            python scripts/build_sfm_cache_all.py --workers "$SFM_WORKERS" $SFM_ARGS
            ;;
        calib)
            echo; echo "################ 阶段 2: DA3 -> metric 标定 ################"
            date
            echo "---- da3_cache 完整性 (待建 > 0 说明 DA3 缓存还没建完, 本轮只标定已有的):"
            python scripts/build_da3_cache_all.py --dry-run | grep -E "目标组合" || true
            # shellcheck disable=SC2086
            python scripts/calibrate_da3_with_sfm.py --workers "$CALIB_WORKERS" $CALIB_ARGS
            ;;
        *)
            echo "未知阶段: $stage (只认 sfm / calib)" >&2
            exit 1
            ;;
    esac
done
echo; date; echo "全部完成"
