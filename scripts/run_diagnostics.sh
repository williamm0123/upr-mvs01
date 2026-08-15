#!/usr/bin/env bash
# 两个诊断任务的驱动脚本。默认只跑只读的部分, 破坏性动作要显式开。
#
#   ./scripts/run_diagnostics.sh coloc      # 实验1: 共位测试 (纯推理, 需 GPU)
#   ./scripts/run_diagnostics.sh audit      # 实验2a: 缓存审计 (只读)
#   ./scripts/run_diagnostics.sh clean      # 实验2b: 审计 + 产出 *_clean.txt
#   ./scripts/run_diagnostics.sh quarantine # 实验2c: 隔离坏文件 (会移动文件)
#   ./scripts/run_diagnostics.sh rebuild    # 实验2d: 重建坏 scan (需 GPU, 慢)
#   ./scripts/run_diagnostics.sh all        # coloc + audit + clean
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-uprmvs}"
PY="${PY:-$HOME/miniconda3/envs/$CONDA_ENV/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python)"

CKPT="${CKPT:-log/model/best.pth}"
SPLIT="${SPLIT:-val}"
TAIL_MM="${TAIL_MM:-20}"
LOG_DIR="experiments/out"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

run_coloc() {
  echo "=== [1] 共位测试: 模型尾巴像素上 prior 的误差 ==="
  [[ -f "$CKPT" ]] || { echo "缺 checkpoint: $CKPT" >&2; exit 1; }
  "$PY" -u experiments/coloc_tail_prior.py \
      --ckpt "$CKPT" --split "$SPLIT" --tail-mm "$TAIL_MM" ${LISTFILE:+--listfile "$LISTFILE"} \
      --out "$LOG_DIR/coloc_${SPLIT}_${STAMP}.npz" \
      2>&1 | tee "$LOG_DIR/coloc_${SPLIT}_${STAMP}.log"
}

run_audit()      { echo "=== [2] prior cache 审计 ===";  "$PY" -u scripts/audit_prior_cache.py 2>&1 | tee "$LOG_DIR/audit_${STAMP}.log"; }
run_clean()      { echo "=== [2] 审计 + clean lists ==="; "$PY" -u scripts/audit_prior_cache.py --clean-lists 2>&1 | tee "$LOG_DIR/audit_clean_${STAMP}.log"; }
run_quarantine() { echo "=== [2] 隔离坏文件 (会移动文件) ==="; "$PY" -u scripts/audit_prior_cache.py --clean-lists --quarantine 2>&1 | tee "$LOG_DIR/audit_quarantine_${STAMP}.log"; }
run_rebuild()    { echo "=== [2] 重建坏 scan (GPU, 慢) ==="; "$PY" -u scripts/audit_prior_cache.py --rebuild all-bad 2>&1 | tee "$LOG_DIR/audit_rebuild_${STAMP}.log"; }

case "${1:-all}" in
  coloc)      run_coloc ;;
  audit)      run_audit ;;
  clean)      run_clean ;;
  quarantine) run_quarantine ;;
  rebuild)    run_rebuild ;;
  all)        run_coloc; run_clean ;;
  *) echo "用法: $0 {coloc|audit|clean|quarantine|rebuild|all}" >&2; exit 2 ;;
esac
echo "日志在 $LOG_DIR/"
