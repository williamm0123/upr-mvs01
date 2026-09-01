#!/usr/bin/env bash
# 本地 (RTX 5060 Ti 16G) 重建 prior —— 前台跑, 每建完一个 view 立刻打印一行。
#
#   ./scripts/build_priors_local.sh                  # scan1-80
#   SCANS=1-40 ./scripts/build_priors_local.sh       # 只做一段
#   DRY=1 ./scripts/build_priors_local.sh            # 只报缺口和预估
#
# 前台执行是故意的: 长跑作业最需要的是"它还活着、刚做完哪个"。输出同时 tee 到
# log/rebuild/local_<时间戳>.log, 所以窗口关了日志还在。
#
# **在 tmux / screen 里跑** —— 这一段要 8~9 小时, 终端断了作业就没了:
#     tmux new -s prior
#     ./scripts/build_priors_local.sh
#     (Ctrl-b d 脱离, tmux attach -t prior 回来)
#
# 中断后重跑同一条命令即可续跑 (save_prior 是 tmp+os.replace 原子写, 差集扫描
# 会跳过已完成的)。
#
# 任务划分: 本地 scan1-80, umhpc scan81-128 (scripts/sbatch_build_priors.sh)。
# 用 scan 编号闭区间而不是 split 名, 是因为编号不随 listfile 变化, 两台机器各跑
# 一半时不会重叠或漏掉。
#
# 注意: 本地磁盘上没有 scan78/79/80/81, 所以 1-80 实际只有 77 个 scan
# (3773 个样本)。脚本会把跳过的 scan 名打出来。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-uprmvs}"
PY="${PY:-$HOME/miniconda3/envs/$CONDA_ENV/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python)"

export UPRMVS_PRIOR_CACHE="${PRIOR_CACHE:-$REPO_ROOT/log/prior_cache_DA_vggt}"
export PYTHONPATH="$REPO_ROOT/models:${PYTHONPATH:-}"

SCANS="${SCANS:-1-89}"
NUM_VIEWS="${NUM_VIEWS:-5}"
TARGET_W="${TARGET_W:-784}"
TARGET_H="${TARGET_H:-588}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="log/rebuild"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/local_${STAMP}.log"
REPORT="$LOG_DIR/local_${STAMP}.csv"

EXTRA=()
[[ -n "${DRY:-}" ]] && EXTRA+=(--dry-run)

# 卡必须是空的: VGGT+DA3 在 784x588 / 5 视角下 allocator 保留约 13 GiB, 16G 卡上
# 起第二个进程必 OOM (实测: 第二个在 load VGGT 时就死)。
BUSY="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "$BUSY" && -z "${DRY:-}" ]]; then
  echo "!! GPU 上已有进程 —— 16G 卡装不下第二份 VGGT+DA3:" >&2
  echo "$BUSY" | sed 's/^/     /' >&2
  echo "   先停掉它们, 或设 FORCE=1 强跑。" >&2
  [[ -z "${FORCE:-}" ]] && exit 1
fi

echo "=== 本地 prior 重建 ==="
echo "    缓存目录   $UPRMVS_PRIOR_CACHE"
echo "    scans      $SCANS   num_views=$NUM_VIEWS   target=${TARGET_W}x${TARGET_H}"
echo "    日志       $LOG"
echo "    逐样本 CSV $REPORT"
echo "    git        $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

exec "$PY" -u scripts/build_prior_cache_all.py \
    --scans "$SCANS" \
    --num-views "$NUM_VIEWS" \
    --target-w "$TARGET_W" --target-h "$TARGET_H" \
    --prior-method residual \
    --cache-dir "$UPRMVS_PRIOR_CACHE" \
    --report "$REPORT" \
    --log-every 50 \
    "${EXTRA[@]}" 2>&1 | tee "$LOG"
