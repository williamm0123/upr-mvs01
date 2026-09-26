#!/bin/bash -l
#SBATCH --job-name=uprmvs_sfm5v
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# 5 视角 SfM 稀疏点云 —— **sbatch 提交, 不是 bash**。
#
#   cd /scr/user/qinglong/projects/upr-mvs01
#   git pull
#   sbatch scripts/sbatch_build_sfm_5view.sh
#
# 每个 (scan, view) 只和 pair.txt 里它自己的前 4 个 src 做极线引导匹配 + 已知相机三角化
# (= 测试时 MVS 网络的 5 视角输入, 不借用其它视角), 输出
#   log/sfm_cache/<scan>/sfm_{view:04d}_3.npz     (Rectified_raw/ 下全部 scan x 49 视角, light 3)
#   log/sfm_cache/_sfm_summary_<时间戳>.csv       每视角点数 / 与 GT 的偏差 (GT 只做统计)
#
# 注意: log/sfm_cache 如果已有旧的 10 邻居对称版缓存, 脚本会报 "拒绝混写" 退出 (params 不同的文件
# 不会被当成已完成)。先挪走旧目录再提交:
#   mv log/sfm_cache log/sfm_cache_nb10
# 或者直接覆盖: sbatch --export=ALL,SFM_ARGS="--overwrite-mismatched" scripts/sbatch_build_sfm_5view.sh
# 下游 log/da3_sfm_cache (calibrate_da3_with_sfm.py) 是用旧点云标定的, 换了点云要重新标定。
#
# 本地 5060 Ti (scan1/13, 2 进程): 每 scan 28~47s, 每视角点数中位 ~5K, 点深度对 GT 偏差中位
# 0.5~0.6mm。124 个 scan 在 A100 上预计 < 1 小时; 按文件断点续跑, 中断后原样重新 sbatch 即可。
#
# 可调 (sbatch --export=ALL,变量=值 ... 或提交前 export):
#   NVIEWS=5          参考视角 + NVIEWS-1 个 src
#   SFM_WORKERS=4     并行 scan 数 (每个一份 CUDA context, 显存峰值约 2GiB; SIFT 在 CPU)
#   SFM_ARGS=""       透传, 例如 "--scans 1 4 9" / "--force" / "--save-ply" / "--dry-run"
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

NVIEWS=${NVIEWS:-5}
SFM_WORKERS=${SFM_WORKERS:-4}
SFM_ARGS=${SFM_ARGS:-}
OUT_DIR="$PROJECT_DIR/log/sfm_cache"

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
echo " job=${SLURM_JOB_ID:-manual}  host=$(hostname)  nviews=$NVIEWS  out=$OUT_DIR"
echo " git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=================================================================="
nvidia-smi -L || true

python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || { echo "CUDA 不可用 —— 需要 --gres=gpu:1" >&2; exit 1; }

date
# shellcheck disable=SC2086
python scripts/build_sfm_cache_all.py --nviews "$NVIEWS" --workers "$SFM_WORKERS" \
    --out "$OUT_DIR" $SFM_ARGS
echo; date; echo "全部完成"
