#!/bin/bash -l
#
# P0/P1 — offline geometric-filter attribution. NO model, NO priors, NO dataset,
# NO GPU inference: everything below reads the cached per-view depth maps.
#
# Why this runs before anything else: the cached photometric confidence is
# identically 1.0 (mode_window=2 spanned all 4 final-stage hypotheses), so
# --photo-thresh has never filtered a single point and the fusion has been
# running on geometric consistency alone — with thresholds that were never
# tuned. geo_rel=0.01 admits a 7 mm depth disagreement at DTU's ~700 mm working
# distance, against an Acc target of ~0.3 mm.
#
# Stage 1 (this script, default): response surface, no PLY, no evaluator.
# Stage 2: pick 5-7 retention levels off the surface, fuse only those, score
#          them with Fast-DTU-Evaluation, and compare at matched Comp.
#
# Usage:
#   bash scripts/geo_sweep.sh                       # full grid, all cached scans
#   SCANS="4 9 10" STRIDE=7 bash scripts/geo_sweep.sh   # quick look
#   MODE=fuse PIX=0.5 REL=0.001 VIEWS=3 POOL=4 bash scripts/geo_sweep.sh
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
cd "$PROJECT_DIR"

# The complete 22 x 49 cache. NOT log/depth_cache/test — that one holds only a
# few scans and one of them is short, which would silently change every number.
CACHE=${CACHE:-outputs/test_test}
MODE=${MODE:-sweep}                 # sweep | fuse
SCANS=${SCANS:-}                    # e.g. "4 9 10"; empty = every cached scan
STRIDE=${STRIDE:-1}                 # use every k-th ref view
OUT=${OUT:-}                        # sweep json (default <CACHE>/geo_sweep.json)
PLY_DIR=${PLY_DIR:-log/pred_points}

# sweep grids
SWEEP_PIX=${SWEEP_PIX:-"1.0,0.75,0.5,0.25,0.125"}
SWEEP_REL=${SWEEP_REL:-"0.01,0.005,0.002,0.001,0.0005"}
SWEEP_ABS=${SWEEP_ABS:-"0,0.5,1.0,2.0"}
SWEEP_VIEWS=${SWEEP_VIEWS:-"3,4,5,6"}
SWEEP_POOL=${SWEEP_POOL:-"4,6,10"}

# single fusion point (MODE=fuse)
PIX=${PIX:-1.0}
REL=${REL:-0.01}
ABS=${ABS:-}                        # empty = disabled
VIEWS=${VIEWS:-3}
POOL=${POOL:-10}

args=(--fuse-only --out "$CACHE")
[[ -n "$SCANS" ]] && args+=(--scans $SCANS)

case "$MODE" in
  sweep)
    args+=(--sweep
           --sweep-pix "$SWEEP_PIX" --sweep-rel "$SWEEP_REL" --sweep-abs "$SWEEP_ABS"
           --sweep-views "$SWEEP_VIEWS" --sweep-pool "$SWEEP_POOL"
           --sweep-ref-stride "$STRIDE")
    [[ -n "$OUT" ]] && args+=(--sweep-out "$OUT")
    ;;
  fuse)
    args+=(--ply-dir "$PLY_DIR"
           --geo-pix "$PIX" --geo-rel "$REL" --geo-views "$VIEWS"
           --fusion-src-views "$POOL")
    [[ -n "$ABS" ]] && args+=(--geo-abs-mm "$ABS")
    ;;
  *)
    echo "MODE must be sweep or fuse, got: $MODE" >&2; exit 2 ;;
esac

echo "=== geo_sweep mode=$MODE cache=$CACHE scans=${SCANS:-all} ==="
exec "$PYTHON_BIN" test.py "${args[@]}"
