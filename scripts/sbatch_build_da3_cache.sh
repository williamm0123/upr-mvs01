#!/bin/bash -l
#SBATCH --job-name=uprmvs_da3cache
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# 全量 DA3 单目深度缓存 —— **sbatch 提交, 不是 bash**。
#
#   cd /scr/user/qinglong/projects/upr-mvs01
#   git pull
#   sbatch scripts/sbatch_build_da3_cache.sh
#
# 124 个 scan x 49 视角 x 7 光照 = 42532 个组合, 原生 1200x1600 分辨率 (process_res=1600,
# DA3 内部按 patch14 取整到约 1596x1204 再还原), float16 压缩存盘。本地 5060 Ti 实测单张
# 约 0.6~0.7s (含 IO), 峰值显存约 4.2GiB —— A100 80GB 跑这个绰绰有余, 显存不是瓶颈。
# 按本地速率外推全量约 8~9 小时, 给 2 天 + qos=long 留足余量 (含集群 IO 抖动、断点续跑)。
#
# **可断点续跑**: scripts/build_da3_cache_all.py 按 (scan,view,light) 缺什么建什么,
# 作业超时/中断后原样重新 sbatch 同一个命令即可从断点继续, 不会重算已完成的文件。
#
# 输出: cfg.paths.da3_cache_path = <项目根>/log/da3_cache (umhpc 上即
# /scr/user/qinglong/projects/upr-mvs01/log/da3_cache), 一个 npz 每 (scan,view,light);
# 失败的样本 (读图/推理异常) 记在 log/da3_cache/_failures_<时间戳>.csv, 不会带崩整轮。
#
# 调试/预演 (不占 GPU, 几秒钟看看待办数对不对):
#   python scripts/build_da3_cache_all.py --dry-run
# 只想先试跑几个样本验证环境, 交互式分配里 (不是这个 sbatch 脚本):
#   python scripts/build_da3_cache_all.py --scans 1 --limit 5
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

# conda 的 activate.d 脚本会读 LD_LIBRARY_PATH 这类未必存在的变量; set -u 下
# 会直接 unbound variable 退出。只在激活期间关掉 nounset。
set +u
source ~/.bashrc
conda activate uprmvs
set -u

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p logs

echo "=================================================================="
echo " job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=================================================================="
nvidia-smi -L || true
# 拿不到卡时 build_da3_cache_all.py 会直接报错退出 (不像训练脚本那样可能静默退 CPU),
# 这里提前检查只是为了在日志顶部给出更直白的原因。
python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || { echo "CUDA 不可用 —— 这个作业需要 --gres=gpu:1" >&2; exit 1; }

exec python scripts/build_da3_cache_all.py "$@"
