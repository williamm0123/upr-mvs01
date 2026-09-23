#!/bin/bash -l
# =============================================================================
# MoAMVSNet — 本地 RTX 5060 Ti 16GB 小参数试跑 (拿不到集群资源时先看效果)。
#
#   bash scripts/train_moa_local.sh                       # MoA, 2 epoch
#   SMOKE=1 bash scripts/train_moa_local.sh               # 合成数据 20 步: 形状/反传/显存
#   MOA=off bash scripts/train_moa_local.sh               # 纯 MVS 级联基线
#   MAX_STEPS=3000 bash scripts/train_moa_local.sh        # 只跑固定步数
#   TRAIN_LIST=my_scans.txt VAL_LIST=my_val.txt bash scripts/train_moa_local.sh
#
# 与 umhpc 正式训练的差别 (都可用环境变量改回):
#   batch 1 / 3 视角 / 固定 512x640 (无多尺度) / MoA 宽度 8 (正式 16) / lr 1e-4 /
#   val 只抽 60 个样本 / DA3 缺失的样本自动跳过 (--da3-missing skip), 所以只下载了
#   部分 scan 的 DA3 cache 也能跑 —— 启动日志会打印跳过了多少。
# 实测 (合成数据, v2 的四级全 128): batch 1 x 3 视角 x 512x640 峰值 9.9 GiB, 1.15 s/step;
#   5 视角 448x576 峰值 11.7 GiB。v1 的窄通道用 WARP_CHANNELS=128,64,32,16 (峰值 7.7)。
#
# 本地 uprmvs 环境目前 import torch 失败 (缺 typing_extensions, 2026-09-18 装 pcl
# 时被删)。修好之前可以用 PYTHON_BIN / EXTRA_PYTHONPATH 指向能用的解释器或依赖目录。
# 输出: log/experiments/$RUN_NAME/{model,tensorboard}; RESUME=auto 时接着 latest.pth 训。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/william/project/uprmvs01}
CONDA_ENV=${CONDA_ENV:-uprmvs}
PYTHON_BIN=${PYTHON_BIN:-}                 # 空 = conda run -n $CONDA_ENV python
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-}
GPU_ID=${GPU_ID:-0}

MOA=${MOA:-on}
RUN_NAME=${RUN_NAME:-moa_local_${MOA}}
EPOCHS=${EPOCHS:-2}
MAX_STEPS=${MAX_STEPS:-0}                  # 0 = EPOCHS x steps_per_epoch
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_VIEWS=${NUM_VIEWS:-3}
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-640}
MOA_DIM=${MOA_DIM:-8}
WARP_CHANNELS=${WARP_CHANNELS:-128,128,128,128}
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-500}
NUM_WORKERS=${NUM_WORKERS:-4}
LOG_INTERVAL=${LOG_INTERVAL:-20}
VAL_INTERVAL=${VAL_INTERVAL:-1000}
CKPT_INTERVAL=${CKPT_INTERVAL:-500}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-60}
TRAIN_LIST=${TRAIN_LIST:-}                 # 空 = lists/dtu/train.txt
VAL_LIST=${VAL_LIST:-}                     # 空 = lists/dtu/val.txt
DA3_ROOT=${DA3_ROOT:-}
DA3_MISSING=${DA3_MISSING:-skip}
RESUME=${RESUME:-auto}
SMOKE=${SMOKE:-0}
SMOKE_STEPS=${SMOKE_STEPS:-20}

cd "$PROJECT_DIR"
export UPRMVS_MACHINE=ubuntu
export UPRMVS_PROFILE=local
export PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONUNBUFFERED=1
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

# 同名的本地训练已经在跑 = 两个进程会轮流覆盖同一份 latest.pth
if pgrep -f "train_moa.py.*--name $RUN_NAME( |$)" >/dev/null 2>&1; then
    echo "已经有一个 --name $RUN_NAME 的 train_moa.py 在跑 (pid: $(pgrep -f "train_moa.py.*--name $RUN_NAME" | tr '\n' ' '))。" >&2
    echo "  并行跑就换个名字: RUN_NAME=${RUN_NAME}_v2 bash $0" >&2
    exit 2
fi

args=(
    --profile local
    --device cuda:0
    --name "$RUN_NAME"
    --moa "$MOA"
    --moa-dim "$MOA_DIM"
    --warp-channels "$WARP_CHANNELS"
    --epochs "$EPOCHS"
    --max-steps "$MAX_STEPS"
    --batch-size "$BATCH_SIZE"
    --val-batch-size 1
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --multi-scale off --height "$HEIGHT" --width "$WIDTH"
    --lr "$LR"
    --warmup-steps "$WARMUP_STEPS"
    --amp on --amp-dtype bf16
    --log-interval "$LOG_INTERVAL"
    --val-interval "$VAL_INTERVAL"
    --ckpt-interval "$CKPT_INTERVAL"
    --max-val-samples "$MAX_VAL_SAMPLES"
    --da3-missing "$DA3_MISSING"
)
[[ -n "$TRAIN_LIST" ]] && args+=(--train-list "$TRAIN_LIST")
[[ -n "$VAL_LIST" ]] && args+=(--val-list "$VAL_LIST")
[[ -n "$DA3_ROOT" ]] && args+=(--da3-root "$DA3_ROOT")

echo "=== local MoAMVSNet: moa=$MOA run=$RUN_NAME GPU=$GPU_ID batch=$BATCH_SIZE views=$NUM_VIEWS" \
     "${HEIGHT}x${WIDTH} moa_dim=$MOA_DIM epochs=$EPOCHS max_steps=$MAX_STEPS ==="

case "$SMOKE" in
    1|true|yes)
        exec "${PY[@]}" train_moa.py "${args[@]}" --smoke --smoke-steps "$SMOKE_STEPS" \
            --smoke-hw "$HEIGHT" "$WIDTH" ;;
    0|false|no) ;;
    *) echo "SMOKE 只能是 0/1, 收到 '$SMOKE'" >&2; exit 2 ;;
esac

exec "${PY[@]}" train_moa.py "${args[@]}" --resume "$RESUME"
