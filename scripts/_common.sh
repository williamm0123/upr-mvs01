# shellcheck shell=bash
#
# Shared setup for every launch script: project root, interpreter, environment.
# Source it, do not execute it:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"      # normal scripts
#     PROJECT_DIR=/abs/path; source "$PROJECT_DIR/scripts/_common.sh"   # sbatch
#
# Why this file exists: the five launch scripts had grown five different ways of
# finding Python — `conda activate` + bare python, a hard-coded absolute
# interpreter path, `conda run -n`, a candidate-search loop, and plain `python`
# — plus the project root and PYTHONPATH written out by hand in each of them.
# That is five things to update when a cluster path changes, and four of them
# fail silently or late. Everything below is derived once, here.
#
# Nothing here is cluster-specific: the interpreter is SEARCHED for, not
# assumed, so it does not matter whether the conda env lives under $HOME,
# /scr/user/$USER, or a prefix given in ENV_PREFIX.

# ---------------------------------------------------------------- project root
# Preference order: an explicit PROJECT_DIR, then this file's own location.
# The location fallback is what makes the interactive scripts path-free, but it
# cannot be relied on under sbatch: SLURM copies the batch script to a spool
# directory, so ${BASH_SOURCE} inside a *job* points at /var/spool/... and not
# at the checkout. That is why train_umhpc.sh still carries one explicit default.
if [[ -z "${PROJECT_DIR:-}" || ! -f "${PROJECT_DIR}/train.py" ]]; then
    _self_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    PROJECT_DIR=$(dirname "$_self_dir")
    unset _self_dir
fi
if [[ ! -f "$PROJECT_DIR/train.py" ]]; then
    echo "PROJECT_DIR=$PROJECT_DIR does not look like the uprmvs checkout (no train.py)" >&2
    echo "Set it explicitly:  PROJECT_DIR=/path/to/upr-mvs01 bash \$0" >&2
    exit 1
fi
export PROJECT_DIR
cd "$PROJECT_DIR"

CONDA_ENV=${CONDA_ENV:-uprmvs}

# ------------------------------------------------------------------ interpreter
# Search instead of assume. On the cluster conda is usually not on PATH (which
# is exactly why the launch scripts stopped using `conda activate`), and the env
# may sit under $HOME/.conda, under scratch, or behind a prefix — all covered.
uprmvs_resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        if [[ -x "$PYTHON_BIN" ]]; then return 0; fi
        echo "PYTHON_BIN=$PYTHON_BIN is not executable" >&2
        return 1
    fi
    local cand tried=()
    for cand in \
        "${ENV_PREFIX:-}/bin/python" \
        "${CONDA_PREFIX:-}/bin/python" \
        "$HOME/.conda/envs/$CONDA_ENV/bin/python" \
        "$HOME/miniconda3/envs/$CONDA_ENV/bin/python" \
        "$HOME/anaconda3/envs/$CONDA_ENV/bin/python" \
        "/scr/user/$USER/.conda/envs/$CONDA_ENV/bin/python" \
        "/scr/user/$USER/miniconda3/envs/$CONDA_ENV/bin/python" \
        "$(conda info --base 2>/dev/null)/envs/$CONDA_ENV/bin/python"
    do
        [[ -z "$cand" || "$cand" == "/bin/python" ]] && continue
        tried+=("$cand")
        # CONDA_PREFIX only counts when the ACTIVE env is the one we want —
        # otherwise an unrelated activated env silently wins the search.
        if [[ "$cand" == "${CONDA_PREFIX:-}/bin/python" \
              && "$(basename "${CONDA_PREFIX:-}")" != "$CONDA_ENV" ]]; then
            continue
        fi
        if [[ -x "$cand" ]]; then PYTHON_BIN="$cand"; return 0; fi
    done
    echo "找不到 '$CONDA_ENV' 环境的解释器。按顺序试过：" >&2
    printf '  ✗ %s\n' "${tried[@]}" >&2
    echo "显式指定一次即可： PYTHON_BIN=/path/to/envs/$CONDA_ENV/bin/python bash \$0" >&2
    return 1
}

uprmvs_resolve_python || exit 1
export PYTHON_BIN

# ------------------------------------------------------------------ environment
# PYTHONPATH is derived, not written out: `vggt` is a top-level namespace package
# under $PROJECT_DIR/models, and inheriting an outside PYTHONPATH has previously
# imported a different checkout's copy. PYTHONNOUSERSITE keeps ~/.local out too.
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# Same rule base/config.py:_detect_machine() uses, so the shell and the config
# can never disagree about which machine this is.
if [[ -z "${UPRMVS_MACHINE:-}" ]]; then
    if [[ "$PROJECT_DIR" == /scr/* ]]; then UPRMVS_MACHINE=umhpc; else UPRMVS_MACHINE=ubuntu; fi
fi
export UPRMVS_MACHINE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
# "reserved but unallocated" in an OOM message is fragmentation, not a real
# shortage; this is the cheapest mitigation and costs nothing when unneeded.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Print exactly which interpreter and which copy of the deps a job is running.
# A stale checkout or the wrong env is the most common cause of a wasted job and
# the hardest to spot afterwards, so every script logs this before doing work.
uprmvs_env_banner() {
    echo "=== project=$PROJECT_DIR machine=$UPRMVS_MACHINE ==="
    echo "=== python=$PYTHON_BIN ==="
    "$PYTHON_BIN" - <<'PY'
import importlib.util, sys
print("python:", sys.executable)
try:
    import torch; print("torch:", torch.__version__, torch.__file__)
except Exception as e:  # noqa: BLE001
    print("torch: MISSING", e)
spec = importlib.util.find_spec("vggt.models.vggt")
print("vggt:", spec.origin if spec else "MISSING")
PY
}
