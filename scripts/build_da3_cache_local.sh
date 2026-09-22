#!/bin/bash
# =============================================================================
# 本地 (RTX 5060 Ti 16GB) 离线构建全量 DA3 深度缓存 -> log/da3_cache
#
#   bash scripts/build_da3_cache_local.sh                  # 全部没建的 scan, NPROC 个进程并行
#   bash scripts/build_da3_cache_local.sh --dry-run        # 只打印待办数, 不装模型
#   bash scripts/build_da3_cache_local.sh --scans 3 4 5    # 只建这几个 scan
#   NPROC=1 bash scripts/build_da3_cache_local.sh --limit 5   # 冒烟: 单进程 5 张
#   bash scripts/build_da3_cache_local.sh --verify         # 只校验已有文件
#
# 跳过规则 (由 scripts/build_da3_cache_all.py 实现):
#   - log/da3_cache/scanN.zip 存在 (从 umhpc 下载, 还没解压) -> 整个 scan 跳过;
#     每开始一个新 scan 前都重新扫一次, 运行中才下载完的 scan 也会跳过。
#   - 已有的 scan 目录按文件取差集: 建满的 (49 视角 x 7 光照 = 343 个) 不会再算,
#     中断留下的半截目录只补缺的文件。Ctrl-C 后原样重跑即可续上。
#   - 默认 ORDER=lex-desc: 按字典序倒着建 (scan99, scan98, ... scan1), Chrome 从 umhpc
#     按字典序正着下载 (scan1, scan10, scan100, ...), 两边从两头往中间走, 不抢同一个 scan。
#
# 参数与 umhpc 那份 cache 保持一致: process_res=1600, float16, 1200x1600 存盘。如果
# 目录里已有文件的 process_res 和这里不同, Python 端直接拒绝 (同一目录混两种分辨率会被
# 数据集当成一种读)。想建 518 档就换目录: OUT=log/da3_cache_518 ... --process-res 518
#
# 离线: DA3 权重从 cfg.paths.da3_weights_file 的本地目录加载, 这里再设 HF_HUB_OFFLINE=1,
# 任何联网尝试都会立刻报错而不是卡住。
#
# 速度/显存 (2026-09-23 本机实测, 1600 档): NPROC=1 1.39 张/s ~5 GiB; NPROC=2 1.52 张/s
# 9.7 GiB; NPROC=3 1.68 张/s 14.2 GiB。5060 Ti 上 GPU 已经 100%, 多进程收益很小 (A100 上才是
# IO 瓶颈), 默认 2; 37730 张全量约 7 小时。每个分片用 --shard i/NPROC, 日志:
# logs/da3_cache_local_<时间>_s<i>.log
#
# 本地 uprmvs 环境 2026-09-18 起 import torch 失败 (缺 typing_extensions); 修好之前可用
# PYTHON_BIN=/path/to/python 或 EXTRA_PYTHONPATH=/path/to/deps 指定可用的依赖。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/william/project/uprmvs01}
CONDA_ENV=${CONDA_ENV:-uprmvs}
PYTHON_BIN=${PYTHON_BIN:-}                 # 空 = conda run -n $CONDA_ENV python
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-}
GPU_ID=${GPU_ID:-0}
NPROC=${NPROC:-2}
ORDER=${ORDER:-lex-desc}
PROCESS_RES=${PROCESS_RES:-1600}
OUT=${OUT:-$PROJECT_DIR/log/da3_cache}

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

cmd=("${PY[@]}" scripts/build_da3_cache_all.py --out "$OUT" --order "$ORDER"
     --process-res "$PROCESS_RES" --device cuda)

# 只统计/只校验/用户自己指定了分片时不再拆进程
single=0
[[ "$NPROC" -le 1 ]] && single=1
for a in "$@"; do
    case "$a" in --dry-run|--verify|--shard|--shard=*) single=1 ;; esac
done
if [[ $single -eq 1 ]]; then
    exec "${cmd[@]}" "$@"
fi

mkdir -p logs
stamp=$(date +%Y%m%d_%H%M%S)
echo "=== DA3 cache (local): out=$OUT  NPROC=$NPROC  order=$ORDER  process_res=$PROCESS_RES  GPU=$GPU_ID ==="
"${cmd[@]}" --dry-run "$@"

# Ctrl-C / kill 时连同所有分片一起停 (已写完的文件是原子落盘的, 重跑会续上)
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
if [[ $rc -eq 0 ]]; then
    echo "=== 全部分片正常结束; 再跑一次 --dry-run 应显示待建 0 (zip 的 scan 不计入) ==="
else
    echo "=== 有分片非零退出, 见 logs/da3_cache_local_${stamp}_s*.log 与 $OUT/_failures_*.csv ===" >&2
fi
exit $rc
