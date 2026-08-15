#!/usr/bin/env bash
# 用当前 5 视角管线重建 train/val/test 的全部 prior 缓存。
#
# 为什么要全量重建: 现有缓存是 1 ref + 2 source (pipeline_version=0) 生成的,
# 而训练喂 1 ref + 4 source。VGGT 是多视图模型, 视角集变了先验就不是同一个
# 东西; SfM 标尺也因为可三角化的点太少而大量触发 num_pairs<20 的兜底 ——
# 这就是 8.05% 未标尺的来源。实测 5 视角下 num_pairs = 420~2970。
#
#   ./scripts/rebuild_all_priors.sh              # 全部, 可续跑
#   ./scripts/rebuild_all_priors.sh --dry-run    # 只报数量和预估时间
#   SPLITS="test" ./scripts/rebuild_all_priors.sh
#   NO_SLIM=1 ./scripts/rebuild_all_priors.sh    # 保留 norm_depth_fill 和 float32 conf
#
# 中断后重跑同一条命令即可接着做 (按缓存签名跳过已完成的)。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-uprmvs}"
PY="${PY:-$HOME/miniconda3/envs/$CONDA_ENV/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python)"

SPLITS="${SPLITS:-train val test}"
NUM_VIEWS="${NUM_VIEWS:-5}"
SLIM_FLAG="--slim"; [[ -n "${NO_SLIM:-}" ]] && SLIM_FLAG=""
LOG_DIR="log/rebuild"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rebuild_$(date +%Y%m%d_%H%M%S).log"

# GPU 必须是空的: VGGT+DA3 常驻 5.9 GiB, 峰值 7.3 GiB, 别的进程占着就会 OOM
BUSY="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true)"
if [[ -n "$BUSY" ]]; then
  echo "!! GPU 上已有进程:" >&2
  echo "$BUSY" | sed 's/^/     /' >&2
  echo "   prior 重建需要独占 (峰值 7.3 GiB)。先停掉它们, 或设 FORCE=1 强跑。" >&2
  [[ -z "${FORCE:-}" ]] && exit 1
fi

echo "=== 重建 prior: splits='$SPLITS'  num_views=$NUM_VIEWS  slim=${SLIM_FLAG:-off} ==="
echo "=== 日志: $LOG ==="
echo "=== 中断后重跑同一条命令即可续跑 ==="

exec "$PY" -u scripts/rebuild_priors.py \
    --splits $SPLITS \
    --num-views "$NUM_VIEWS" \
    --prior-resize "train:0.5,val:0.5,test:1.0" \
    $SLIM_FLAG "$@" 2>&1 | tee "$LOG"
