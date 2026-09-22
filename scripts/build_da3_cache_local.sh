#!/bin/bash
# =============================================================================
# 本地 (RTX 5060 Ti 16GB) 离线补建 DA3 深度缓存 -> log/da3_cache
#
# umhpc 上的 log/da3_cache 只建了一部分 scan, 本地用 rsync 从 umhpc 同步。这个脚本检查
# scan1..scan128, 把 umhpc 上没有的 scan 在本地建出来, 最后列出这些 scan 并给出上传命令,
# 回头传回 umhpc 让两边一致。
#
#   bash scripts/build_da3_cache_local.sh              # 检查 + 建缺失的 scan + 打印待上传清单
#   bash scripts/build_da3_cache_local.sh --dry-run    # 只检查/统计, 不装模型, 不写任何文件
#   bash scripts/build_da3_cache_local.sh --verify     # 逐个读一遍已有 npz, 查坏文件
#   NPROC=1 bash scripts/build_da3_cache_local.sh --limit 5   # 冒烟: 单进程 5 张
#
# 每个 scanN 归为下面一类 (图片数按 Rectified_raw/ 实际数, 目前每 scan 49 视角 x 7 光照 = 343):
#   完整    目录里 da3_*.npz 数 >= 图片数
#   缺失    本地没有这个目录 (也没有 scanN.zip)。rsync 一开始就把 umhpc 上所有 scan 的目录
#           建好了, 所以没有目录 == umhpc 上没有这个 scan -> 整个在本地建
#   不完整  有目录但文件不够: 多数是 rsync 还没传到 (umhpc 上有), 少数是 umhpc 上本来就只建了
#           一半 (例如 scan110 只有 19/343)。默认只报告不动手 —— "rsync 此刻没在跑" 不等于
#           "rsync 传完了", 中途断开时去本地重算 umhpc 上已有的 scan 是白跑几小时。等 rsync
#           真的传完 (再跑一次 rsync 没有新文件) 再 FILL_PARTIAL=1 跑一次补齐剩下的。
#   不存在  数据集 Rectified_raw/ 里就没有 (scan78-81), 无事可做
#
# 本地建过/补过的 scan 记在 $OUT/_local_built_scans.txt (= 待上传清单)。中断后重跑时, 清单里的
# scan 即使已经有了半截目录也还是按"本地负责"续建, 不会被当成 umhpc 同步来的而跳过。
# 已写完的 npz 都是原子落盘的, Ctrl-C 后原样重跑即可续上。
#
# 参数与 umhpc 那份 cache 一致: process_res=1600, float16, 1200x1600 存盘; 已有文件的
# process_res 和这里不同时 Python 端直接拒绝 (同一目录混两种分辨率会被数据集当成一种读)。
#
# 离线: DA3 权重从 cfg.paths.da3_weights_file 的本地目录加载, 这里再设 HF_HUB_OFFLINE=1,
# 任何联网尝试都会立刻报错而不是卡住。
#
# 速度/显存 (2026-09-23 本机实测, 1600 档): NPROC=1 1.39 张/s ~5 GiB; NPROC=2 1.52 张/s
# 9.7 GiB; NPROC=3 1.68 张/s 14.2 GiB。5060 Ti 上 GPU 已经 100%, 多进程收益很小, 默认 2。
# 每个分片用 --shard i/NPROC, 日志: logs/da3_cache_local_<时间>_s<i>.log
#
# 本地 uprmvs 环境 2026-09-18 起 import torch 失败 (缺 typing_extensions); 修好之前可用
# PYTHON_BIN=/path/to/python 或 EXTRA_PYTHONPATH=/path/to/deps 指定可用的依赖。
# =============================================================================

set -euo pipefail
shopt -s nullglob

PROJECT_DIR=${PROJECT_DIR:-/home/william/project/uprmvs01}
CONDA_ENV=${CONDA_ENV:-uprmvs}
PYTHON_BIN=${PYTHON_BIN:-}                 # 空 = conda run -n $CONDA_ENV python
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-}
GPU_ID=${GPU_ID:-0}
NPROC=${NPROC:-2}
PROCESS_RES=${PROCESS_RES:-1600}
OUT=${OUT:-$PROJECT_DIR/log/da3_cache}
DTU_ROOT=${DTU_ROOT:-/home/william/project/dataset/DTU/dtu_training}
SCAN_MIN=${SCAN_MIN:-1}
SCAN_MAX=${SCAN_MAX:-128}
FILL_PARTIAL=${FILL_PARTIAL:-0}            # 1 = 连不完整的 scan 也在本地补齐 (rsync 确实传完后再用)
UMHPC=${UMHPC:-qinglong@login01.dicc.um.edu.my}
UMHPC_CACHE=${UMHPC_CACHE:-/scr/user/qinglong/projects/upr-mvs01/log/da3_cache}
MANIFEST="$OUT/_local_built_scans.txt"

cd "$PROJECT_DIR"
export UPRMVS_MACHINE=ubuntu
export UPRMVS_PROFILE=local
export PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

dry_run=0 verify=0 user_shard=0
for a in "$@"; do
    case "$a" in
        --dry-run) dry_run=1 ;;
        --verify) verify=1 ;;
        --shard|--shard=*) user_shard=1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 检查 scan1..scan128
# ---------------------------------------------------------------------------
n_npz() { local f=("$OUT/$1"/da3_*.npz); echo ${#f[@]}; }
n_img() { local f=("$DTU_ROOT/Rectified_raw/$1"/rect_[0-9][0-9][0-9]_[0-6]_r5000.png); echo ${#f[@]}; }
in_manifest() { [[ -f "$MANIFEST" ]] && grep -qx "$1" "$MANIFEST"; }
rsync_running() { pgrep -f 'rsync.*da3_cache' >/dev/null; }

classify() {
    complete=() missing=() partial=() local_todo=() absent=()
    partial_desc=() local_desc=()
    local n s have need
    for ((n = SCAN_MIN; n <= SCAN_MAX; n++)); do
        s=scan$n
        if [[ ! -d "$DTU_ROOT/Rectified_raw/$s" ]]; then
            absent+=("$s"); continue
        fi
        have=$(n_npz "$s"); need=$(n_img "$s")
        if ((have >= need)); then
            complete+=("$s")
        elif in_manifest "$s"; then            # 本地负责的, 上次没建完
            local_todo+=("$s"); local_desc+=("$s($have/$need)")
        elif [[ ! -d "$OUT/$s" && ! -e "$OUT/$s.zip" ]]; then
            missing+=("$s")
        else
            partial+=("$s"); partial_desc+=("$s($have/$need)")
        fi
    done
}

head_n() {   # 最多列 $1 个, 其余用 ... 带过 (不完整的 scan 有几十个, 全列出来刷屏)
    local n=$1; shift
    if (($# > n)); then echo "${*:1:$n} ... (共 $#)"; else echo "$*"; fi
}

report() {
    echo "  完整     ${#complete[@]}"
    echo "  缺失     ${#missing[@]}${missing[*]:+: $(head_n 30 "${missing[@]}")}"
    if ((${#local_todo[@]})); then echo "  本地未完 ${#local_todo[@]}: $(head_n 30 "${local_desc[@]}")"; fi
    echo "  不完整   ${#partial[@]}${partial_desc[*]:+: $(head_n 10 "${partial_desc[@]}")}"
    echo "  不存在   ${#absent[@]}${absent[*]:+: ${absent[*]}}  (数据集里没有)"
}

upload_summary() {
    if [[ ! -s "$MANIFEST" ]]; then
        echo "=== 待上传清单为空 (本地没有建过 umhpc 缺的 scan) ==="
        return
    fi
    local s have need list=() bad=()
    while read -r s; do
        [[ -n "$s" ]] || continue
        have=$(n_npz "$s"); need=$(n_img "$s")
        list+=("$s")
        ((have >= need)) || bad+=("$s($have/$need)")
    done <"$MANIFEST"
    echo "=== 需要上传到 umhpc 的 scan (${#list[@]} 个, 清单: $MANIFEST) ==="
    echo "  ${list[*]}"
    if ((${#bad[@]})); then
        echo "  !! 其中还没建完的: ${bad[*]} —— 重跑本脚本补齐后再上传"
    fi
    echo "  上传命令 (--ignore-existing: umhpc 上已有的文件不会重传/覆盖; 会提示输入密码):"
    echo "  rsync -avP -r --ignore-existing --files-from=\"$MANIFEST\" \"$OUT/\" $UMHPC:$UMHPC_CACHE/"
}

if ((verify == 0)); then
    classify
    case "$FILL_PARTIAL" in
        0) fill=0 ;;
        1) fill=1 ;;
        *) echo "FILL_PARTIAL 只能是 0/1, 收到 '$FILL_PARTIAL'" >&2; exit 2 ;;
    esac
    if ((fill)) && rsync_running; then
        echo "!! FILL_PARTIAL=1 但检测到 rsync 正在同步 da3_cache: 两边会抢着建同一批文件。" >&2
        echo "   等 rsync 传完再跑, 或 FILL_PARTIAL=0 只建 umhpc 上没有的 scan。" >&2
        exit 2
    fi
    build=("${missing[@]}" "${local_todo[@]}")
    if ((fill)); then build+=("${partial[@]}"); fi

    echo "=== scan$SCAN_MIN..scan$SCAN_MAX 检查 (out=$OUT) ==="
    report
    if ((${#partial[@]})); then
        if ((fill)); then
            echo "  -> FILL_PARTIAL=1: 不完整的 scan 也在本地补齐, 并加进待上传清单"
        else
            echo "  -> 不完整的 scan 本次不碰 (umhpc 上有, 交给 rsync)。等 rsync 真的传完, 再"
            echo "     FILL_PARTIAL=1 跑一次, 补 umhpc 自己就缺的部分 (例如 scan110 只有 19/343)"
        fi
    fi
    echo "  -> 本次要在本地建/补: ${#build[@]} 个 scan${build[*]:+: $(head_n 30 "${build[@]}")}"

    if ((dry_run)); then
        upload_summary
        exit 0
    fi
    if ((${#build[@]} == 0)); then
        echo "=== 没有要建的 scan ==="
        upload_summary
        exit 0
    fi
    # 先登记再建: 中途中断后, 这些 scan 的半截目录在重跑时仍按"本地负责"续建
    mkdir -p "$OUT"
    touch "$MANIFEST"
    for s in "${build[@]}"; do
        in_manifest "$s" || echo "$s" >>"$MANIFEST"
    done
fi

# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------
if [[ -n "$PYTHON_BIN" ]]; then
    PY=("$PYTHON_BIN")
else
    PY=(conda run -n "$CONDA_ENV" --no-capture-output python)
fi
"${PY[@]}" -c 'import torch, cv2; assert torch.cuda.is_available(), "no CUDA"' 2>/dev/null || {
    echo "Python 环境不可用 (import torch/cv2 失败或没有 CUDA)。" >&2
    echo "  本地 uprmvs 环境 2026-09-18 起缺 typing_extensions 等包; 修好后再跑," >&2
    echo "  或 PYTHON_BIN=/path/to/python / EXTRA_PYTHONPATH=/path/to/deps 指定可用的环境。" >&2
    exit 1
}

cmd=("${PY[@]}" scripts/build_da3_cache_all.py --out "$OUT" --dtu-root "$DTU_ROOT"
     --process-res "$PROCESS_RES" --device cuda)

if ((verify)); then
    exec "${cmd[@]}" "$@"                  # 校验全部已有文件, 不限于本地建的 scan
fi

cmd+=(--scans "${build[@]#scan}")
if ((NPROC <= 1 || user_shard)); then
    rc=0
    "${cmd[@]}" "$@" || rc=$?
else
    mkdir -p logs
    stamp=$(date +%Y%m%d_%H%M%S)
    echo "=== DA3 cache (local): NPROC=$NPROC  process_res=$PROCESS_RES  GPU=$GPU_ID ==="

    # Ctrl-C / kill 时连同所有分片一起停
    trap 'trap - INT TERM; echo "中断, 停止全部分片" >&2; kill 0' INT TERM

    pids=()
    for ((i = 0; i < NPROC; i++)); do
        log="logs/da3_cache_local_${stamp}_s${i}.log"
        "${cmd[@]}" --shard "$i/$NPROC" "$@" 2>&1 | tee "$log" | sed -u "s/^/[s$i] /" &
        pids+=($!)
        echo "分片 $i/$NPROC -> $log"
    done
    rc=0
    for p in "${pids[@]}"; do
        wait "$p" || rc=1
    done
    ((rc == 0)) || echo "!! 有分片非零退出, 见 logs/da3_cache_local_${stamp}_s*.log 与 $OUT/_failures_*.csv" >&2
fi

echo
echo "=== 结束后再检查一次 ==="
classify
report
upload_summary
exit $rc
