#!/bin/bash -l
#SBATCH --job-name=uprmvs_prior
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# umhpc 单卡重建 prior。默认做 **scan81-128** —— 本地那台 5060 Ti 做 scan1-80
# (scripts/build_priors_local.sh), 两边按 scan 编号闭区间切, 不重叠不遗漏。
#
#   sbatch scripts/sbatch_build_priors.sh                    # scan81-128
#   SCANS=81-104 sbatch scripts/sbatch_build_priors.sh       # 再切细一点
#   DRY=1 bash scripts/sbatch_build_priors.sh                # 不排队, 直接看缺口
#
# 用编号区间而不是 split 名: 编号不随 listfile 变化, 两台机器各跑一半时不会因为
# 一边的列表改了就重叠或漏掉。scan81-128 = 48 个 scan × 49 视角 = 2352 个样本
# (前提是 umhpc 的 Rectified_raw 里这 48 个都在; 缺的会被跳过并打出名字 ——
# 本地磁盘就缺 scan78/79/80/81)。
#
# 规模与速度: 只建 light 3, 训练时同一个 (scan, view) 的 7 个光照共用这一份 (见
# PriorConfig.shared_light)。实测 (RTX 5060 Ti, 784x588, 5 视角, resize 0.5)
# 8.5s/样本 -> 2352 个约 5.5h, A100 上更快。--time 给了 48h, 够重跑几轮。
#
# 显存: 784x588 / 5 视角峰值实测 8.83 GiB (allocator 保留约 13 GiB)。A100-80G
# 绰绰有余; 但**不要和别的作业共卡**。
#
# 日志: 逐个 view 一行, 写在 logs/uprmvs_prior_<jobid>.out。看进度:
#     tail -f logs/uprmvs_prior_<jobid>.out
# 逐样本 CSV 在 log/rebuild/ 下, 是作业结束后做审计的那一份。
#
# 中断/超时后重跑同一条命令即可续跑 (save_prior 是 tmp+os.replace 原子写, 差集
# 扫描会跳过已完成的)。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

# conda 的 activate.d 会读未必存在的变量, set -u 下会直接 unbound variable 退出
set +u
source ~/.bashrc
conda activate uprmvs
set -u

export UPRMVS_MACHINE=umhpc
# 缓存目录: 换 prior 生成方式/分辨率必须换目录 (文件名两者都不编码)。
# 训练侧读同一个变量, 见 base/config.py。
export PRIOR_CACHE=${PRIOR_CACHE:-$PROJECT_DIR/log/prior_cache_DA_vggt}
export UPRMVS_PRIOR_CACHE="$PRIOR_CACHE"
export PYTHONPATH="$PROJECT_DIR/models:${PYTHONPATH:-}"

SCANS=${SCANS:-81-128}
NUM_VIEWS=${NUM_VIEWS:-5}
TARGET_W=${TARGET_W:-784}
TARGET_H=${TARGET_H:-588}
# 有多张卡时才用: SHARD_N=2 配 --array=0-1
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
SHARD_N=${SHARD_N:-${SLURM_ARRAY_TASK_COUNT:-1}}
TAG="${SLURM_JOB_ID:-nojob}_${SHARD_ID}of${SHARD_N}"

EXTRA=()
[[ -n "${DRY:-}" ]] && EXTRA+=(--dry-run)

mkdir -p logs log/rebuild

echo "=== umhpc prior 重建 ==="
echo "    缓存目录   $UPRMVS_PRIOR_CACHE"
echo "    scans      $SCANS   num_views=$NUM_VIEWS   target=${TARGET_W}x${TARGET_H}"
echo "    分片       $SHARD_ID/$SHARD_N"
echo "    git        $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

exec python -u scripts/build_prior_cache_all.py \
    --scans "$SCANS" \
    --num-views "$NUM_VIEWS" \
    --target-w "$TARGET_W" --target-h "$TARGET_H" \
    --prior-method residual \
    --cache-dir "$UPRMVS_PRIOR_CACHE" \
    --shard "$SHARD_ID/$SHARD_N" \
    --report "log/rebuild/umhpc_${TAG}.csv" \
    --log-every 50 \
    "${EXTRA[@]}"
