#!/bin/bash -l
#SBATCH --job-name=uprmvs_moa
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# MoAMVSNet v2 —— 单卡 A100-80GB, 30k 步 (sbatch 排队, 不是 bash)。
#
#   cd /scr/user/qinglong/projects/upr-mvs01 && git pull
#   sbatch scripts/train_moa_umhpc.sh                    # v2: 30k 步, warp 128/128/64/64
#   MOA=off sbatch scripts/train_moa_umhpc.sh            # 同口径的纯 MVS 基线 (归因必需)
#   STEPS=0 EPOCHS=15 sbatch scripts/train_moa_umhpc.sh  # 回到按 epoch 跑
#
# v1 -> v2 的三处改动:
#   * 30k 步而不是 15 epoch(101.6k)。**退火 horizon 跟随停止步数**, 所以 30k 结束时
#     lr 已经退到底, 这个数才能和 vNext 的 30k 横比。v1 在 22k 时 lr 还有峰值的 88%,
#     它跟 vNext(21.5k 时只剩 20%) 的比较是不成立的。
#   * warp 通道 128/64/32/16 -> 四级全 128。代价几乎全在 warp 张量 [B,C,D,H,W]
#     上 (每个 source 都要留给反向), 本地实测每样本 +60% 显存、每步 +43%。
#   * batch 4 -> 2。v1 的 batch 4 在 80GB 上峰值已经 98%, 加宽之后放不下。
#     lr 随之按 sqrt 缩放 (LR_REF 3e-4 @ 全局 batch 2), batch 2 正好回到 3e-4。
#
# 显存 (本地两点斜率 + 集群 v1 实测 98%@batch4 校准, 最大训练尺度 640x896):
#   全 128:        batch 2 约 63 GiB (79%)   batch 3 约 94 GiB (装不下)
#   128/128/64/64: batch 2 约 49 GiB (61%)   batch 3 约 72 GiB (90%)
#   这是外推, 正式提交前先在 interactive 里实测一次最大尺度:
#     python train_moa.py --profile umhpc --smoke --smoke-steps 5 --batch-size 2 \
#         --num-views 5 --smoke-hw 640 896 --warp-channels 128,128,128,128
# 速度: v1 是 1.89 s/step @ batch 4; batch 2 + 全 128 预计 ~1.3-1.5 s/step,
#   30k 步约 11-13 小时, 一个作业跑得完 (自动续投基本用不上, 但保留)。
#
# 前提: log/da3_cache 已建全 (train+val 全部 scan x 49 视角 x 7 光照)。缺任何一个
# 样本都会在启动时直接报错 (--da3-missing error)。MOA=off 不读 DA3。
#
# 超时自动续投: slurm 在超时前 15 分钟发 USR1, train_moa.py 跑完当前 step 写
# latest.pth 并以 124 退出, 本脚本用 FRESH=0 (--resume auto) 重投自己。
# 手动续训: FRESH=0 sbatch scripts/train_moa_umhpc.sh
#
# 输出: log/experiments/$RUN_NAME/{model,tensorboard,config.json}。FRESH=1 时同名旧
# 目录会被归档 (不删除); 目录最近 60 分钟还有写入则拒绝启动, 防止踩掉在跑的作业。
# 训练完之后: scripts/test_moa_fusibile.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

MOA=${MOA:-on}
case "$MOA" in
    on)  RUN_NAME=${RUN_NAME:-MOA_V2_30K} ;;
    off) RUN_NAME=${RUN_NAME:-MVSBASE_V2_30K} ;;
    *) echo "MOA 只能是 on / off, 收到 '$MOA'" >&2; exit 2 ;;
esac
STEPS=${STEPS:-30000}                   # >0 按步数跑 (退火 horizon 跟随它); 0 = 按 EPOCHS
EPOCHS=${EPOCHS:-15}                    # 只在 STEPS=0 时生效
LR_HORIZON=${LR_HORIZON:-0}             # 0 = 跟随实际停止步数; 只有做"与长跑同轨迹的短筛"才改
WARP_CHANNELS=${WARP_CHANNELS:-128,128,128,128}
PER_GPU_BATCH=${PER_GPU_BATCH:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
NUM_VIEWS=${NUM_VIEWS:-5}
NUM_WORKERS=${NUM_WORKERS:-12}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-2000}      # 另外每个 epoch 末都验证一次
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
SEED=${SEED:-20260526}
DETERMINISTIC=${DETERMINISTIC:-1}       # 与旧 arm 一致; 0 = cudnn benchmark, 更快但不可逐位复现
DA3_ROOT=${DA3_ROOT:-}                  # 空 = cfg.paths.da3_cache_path (log/da3_cache)
FRESH=${FRESH:-1}                       # 1 = 新 run (归档旧目录, --resume off); 0 = 续训
CHAIN=${CHAIN:-0}
MAX_CHAIN=${MAX_CHAIN:-4}
# lr: 3e-4 @ 全局 batch 2 按 sqrt 缩放, 与 scripts/_arm_common.sh 同一规则
LR_REF=${LR_REF:-3e-4}
LR_REF_BATCH=${LR_REF_BATCH:-2}
LR=${LR:-$(awk -v l="$LR_REF" -v g="$PER_GPU_BATCH" -v r="$LR_REF_BATCH" 'BEGIN{printf "%.4g", l*sqrt(g/r)}')}

# conda 的 activate.d 会读未定义变量, set -u 下直接退出; 只在激活期间关掉 nounset。
set +u
source ~/.bashrc
conda activate uprmvs
set -u

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p logs

RUN_DIR="log/experiments/$RUN_NAME"
if [[ "$FRESH" == "1" ]]; then
    if [[ -e "$RUN_DIR" ]]; then
        # 同名目录还在被写 = 多半有另一个作业正在跑它。直接 mv 会把它的
        # model/ 从脚下挪走, 那个作业下一次存档就 FileNotFoundError 死掉,
        # 而它的 tfevents 会继续写进 _archive —— 两个 run 的日志就此对半劈开。
        RECENT=$(find "$RUN_DIR" -type f -mmin -60 -print -quit 2>/dev/null || true)
        if [[ -n "$RECENT" && "${FORCE_ARCHIVE:-0}" != "1" ]]; then
            echo "拒绝归档: $RUN_DIR 最近 60 分钟内仍有写入 ($RECENT)。" >&2
            echo "  另一个作业很可能正在用这个名字。并行跑新版本请换名字:" >&2
            echo "     RUN_NAME=${RUN_NAME}_v2 sbatch $0" >&2
            echo "  确认那个作业已经结束再归档: FORCE_ARCHIVE=1 sbatch $0" >&2
            exit 2
        fi
        ARCHIVE="log/experiments/_archive/${RUN_NAME}_$(date -u +%Y%m%d_%H%M%S)"
        mkdir -p "$(dirname "$ARCHIVE")"
        mv "$RUN_DIR" "$ARCHIVE"
        echo "=== 上一轮已归档 (未删除): $RUN_DIR -> $ARCHIVE ==="
    fi
    RESUME=off
else
    RESUME=auto
fi

GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
echo "=================================================================="
echo " MoAMVSNet  moa=$MOA  run=$RUN_NAME  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=${GIT_SHA:0:12}  chain=$CHAIN/$MAX_CHAIN  fresh=$FRESH  resume=$RESUME"
echo " steps=$STEPS (0=epochs $EPOCHS)  batch=$PER_GPU_BATCH  views=$NUM_VIEWS  warp=$WARP_CHANNELS"
echo " lr=$LR  warmup=$WARMUP_STEPS  lr_horizon=$([[ $LR_HORIZON -gt 0 ]] && echo $LR_HORIZON || echo 跟随停止步数)  seed=$SEED"
echo "=================================================================="
nvidia-smi -L || true
python - "$PER_GPU_BATCH" <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA 不可用 —— 这个作业需要 --gres=gpu:1")
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"=== GPU {torch.cuda.get_device_name(0)}  {gib:.0f} GiB ===")
if gib < 70 and int(sys.argv[1]) >= 2:
    sys.exit(f"只分到 {gib:.0f} GiB 的卡, batch {sys.argv[1]} (warp 128/128/64/64) 需要 80GB A100; "
             f"换 80GB 节点或用 PER_GPU_BATCH=1 重投")
PY

args=(
    --profile umhpc
    --name "$RUN_NAME"
    --moa "$MOA"
    --warp-channels "$WARP_CHANNELS"
    --batch-size "$PER_GPU_BATCH"
    --val-batch-size "$VAL_BATCH_SIZE"
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --warmup-steps "$WARMUP_STEPS"
    --amp on --amp-dtype bf16
    --multi-scale on
    --seed "$SEED"
    --log-interval "$LOG_INTERVAL"
    --val-interval "$VAL_INTERVAL"
    --ckpt-interval "$CKPT_INTERVAL"
    --da3-missing error
    --resume "$RESUME"
)
if [[ "$STEPS" -gt 0 ]]; then
    args+=(--max-steps "$STEPS")
else
    args+=(--epochs "$EPOCHS")
fi
[[ "$LR_HORIZON" -gt 0 ]] && args+=(--lr-schedule-steps "$LR_HORIZON")
[[ "$DETERMINISTIC" == "1" ]] && args+=(--deterministic)
[[ -n "$DA3_ROOT" ]] && args+=(--da3-root "$DA3_ROOT")

# python 放后台 + wait, 这样 USR1 到来时 trap 能立刻执行并转发给它。
python train_moa.py "${args[@]}" &
PID=$!
trap 'echo "=== 收到 USR1 (即将超时), 通知训练进程存档 ==="; kill -USR1 "$PID" 2>/dev/null || true' USR1
set +e
wait "$PID"
RC=$?
if [[ $RC -gt 128 ]]; then
    # wait 被 USR1 打断 (128+10); 再 wait 一次拿 python 的真实退出码。已退出的
    # 子进程 bash 会记住它的状态, 所以这里不会挂住。
    wait "$PID"
    RC=$?
fi
set -e

if [[ $RC -eq 124 ]]; then
    if [[ $CHAIN -ge $MAX_CHAIN ]]; then
        echo "=== 已续投 $CHAIN 次 (MAX_CHAIN=$MAX_CHAIN), 不再自动续投; 手动: FRESH=0 sbatch $0 ===" >&2
        exit 1
    fi
    NEXT=$((CHAIN + 1))
    echo "=== 超时存档完成, 续投第 $NEXT 次 ==="
    sbatch --export=ALL,FRESH=0,CHAIN=$NEXT,MOA=$MOA,RUN_NAME=$RUN_NAME,PER_GPU_BATCH=$PER_GPU_BATCH,LR=$LR,STEPS=$STEPS,EPOCHS=$EPOCHS,LR_HORIZON=$LR_HORIZON,WARP_CHANNELS=$WARP_CHANNELS \
        "$PROJECT_DIR/scripts/train_moa_umhpc.sh"
    exit 0
fi
echo "=== train_moa.py 退出码 $RC ==="
exit $RC
