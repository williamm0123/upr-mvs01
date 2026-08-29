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
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# =============================================================================
# 用残差场 prior (models/residual_field.py) 重建整个缓存, 支持分片并行。
#
#   sbatch --array=0-1 scripts/sbatch_build_priors.sh          # 2 片 (umhpc 上限)
#   SCANS=val sbatch --array=0-1 scripts/sbatch_build_priors.sh
#   DRY=1 SHARD_N=1 bash scripts/sbatch_build_priors.sh        # 不排队, 只报缺口
#
# 每片是独立进程、独立 GPU, 写的是**不相交**的 (scan, view) 集合, 所以可以随便
# 并行。中断后重跑同一条命令即可续跑 (save_prior 是 tmp+os.replace 原子写, 且
# 差集扫描会跳过已完成的)。
#
# 规模: 只建 light 3 —— 训练时同一个 (scan, view) 的 7 个光照共用这一份, 见
# PriorConfig.shared_light 和 dtu.prior_cache_path_for。于是目标全集是
# (79+18+22)*49 = 5831 个样本, 而不是逐光照的 29057 个。实测 (RTX 5060 Ti,
# 784x588, 5 视角, resize 0.5) 8.5s/样本 = 单卡约 14h; --array=0-1 双卡约 7h,
# A100 上更快。磁盘约 25~30G。
#
# 显存: VGGT+DA3 在 784x588 / 5 视角下峰值实测 8.83 GiB。A100-80G 绰绰有余,
# 16G 卡也够; 但**不要和别的作业共卡**, 峰值撞上就 OOM (脚本里有 nvidia-smi
# 提醒但不拦截)。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

# conda 的 activate.d 会读未必存在的变量, set -u 下会直接退出
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

SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
SHARD_N=${SHARD_N:-${SLURM_ARRAY_TASK_COUNT:-1}}
SCANS=${SCANS:-all}
NUM_VIEWS=${NUM_VIEWS:-5}
TARGET_W=${TARGET_W:-784}
TARGET_H=${TARGET_H:-588}
EXTRA=()
[[ -n "${DRY:-}" ]] && EXTRA+=(--dry-run)

mkdir -p logs log/rebuild

echo "=== prior 重建 ==="
echo "    缓存目录   $UPRMVS_PRIOR_CACHE"
echo "    分片       $SHARD_ID/$SHARD_N"
echo "    scans      $SCANS   num_views=$NUM_VIEWS   target=${TARGET_W}x${TARGET_H}"
echo "    git        $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec python -u scripts/build_prior_cache_all.py \
    --scans "$SCANS" \
    --num-views "$NUM_VIEWS" \
    --target-w "$TARGET_W" --target-h "$TARGET_H" \
    --prior-method residual \
    --cache-dir "$UPRMVS_PRIOR_CACHE" \
    --shard "$SHARD_ID/$SHARD_N" \
    --report "log/rebuild/prior_DA_vggt_shard${SHARD_ID}of${SHARD_N}.csv" \
    "${EXTRA[@]}"
