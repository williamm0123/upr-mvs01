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
# MoAMVSNet — 单卡 A100-80GB 正式训练, 15 个 epoch (sbatch 排队, 不是 bash)。
#
#   cd /scr/user/qinglong/projects/upr-mvs01 && git pull
#   sbatch scripts/train_moa_umhpc.sh                    # MoA, 15 epoch
#   MOA=off sbatch scripts/train_moa_umhpc.sh            # 纯 MVS 级联基线 (MoA.md 阶段 A)
#
# 前提: log/da3_cache 已按 scripts/build_da3_cache_all.py 建全 (train+val 全部
# scan x 49 视角 x 7 光照)。缺任何一个样本都会在启动时直接报错 (--da3-missing error),
# 而不是训练到一半才发现。MOA=off 不读 DA3。
#
# 步数: steps_per_epoch = len(train_loader) 由 train_moa.py 算, 15 epoch 的 cosine
# horizon 也随之确定。78 scan x 49 x 7 = 26754 样本, batch 4 -> 6688 步/epoch,
# 15 epoch 约 10 万步。按旧 SVA 基座 ~2.5 s/step 再加 MoA 的开销估算需要 3.5-4 天,
# 超过 --time 3 天, 所以:
#
#   自动续投: slurm 在超时前 15 分钟给 batch shell 发 USR1 (--signal=B:USR1@900),
#   这里转给 python; train_moa.py 跑完当前 step 写 latest.pth 并以 124 退出, 本脚本
#   随即用 FRESH=0 (--resume auto) 重新 sbatch 自己。CHAIN 计数到 MAX_CHAIN 为止。
#   手动续训 (比如作业崩了): FRESH=0 sbatch scripts/train_moa_umhpc.sh
#
# 显存 (5060 Ti 实测外推): 每样本 ~26 GiB/Mpx, batch 4 在最大训练尺度 640x896 约
# 61 GiB (~77%)。OOM 就 PER_GPU_BATCH=3 (lr 按 sqrt 自动缩放)。
#
# 输出: log/experiments/$RUN_NAME/{model,tensorboard,config.json}。FRESH=1 (默认,
# 首次提交) 时同名旧目录会被**归档**到 log/experiments/_archive/, 不删除。
# 训练完之后: scripts/test_moa_fusibile.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
cd "$PROJECT_DIR"

MOA=${MOA:-on}
case "$MOA" in
    on)  RUN_NAME=${RUN_NAME:-MOA_E15} ;;
    off) RUN_NAME=${RUN_NAME:-MVSBASE_E15} ;;
    *) echo "MOA 只能是 on / off, 收到 '$MOA'" >&2; exit 2 ;;
esac
EPOCHS=${EPOCHS:-15}
PER_GPU_BATCH=${PER_GPU_BATCH:-4}
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
echo " epochs=$EPOCHS  batch=$PER_GPU_BATCH  views=$NUM_VIEWS  lr=$LR  warmup=$WARMUP_STEPS  seed=$SEED"
echo "=================================================================="
nvidia-smi -L || true
python - "$PER_GPU_BATCH" <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA 不可用 —— 这个作业需要 --gres=gpu:1")
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"=== GPU {torch.cuda.get_device_name(0)}  {gib:.0f} GiB ===")
if gib < 70 and int(sys.argv[1]) >= 4:
    sys.exit(f"只分到 {gib:.0f} GiB 的卡, batch {sys.argv[1]} 需要 80GB A100; "
             f"换 80GB 节点或用 PER_GPU_BATCH=2 重投")
PY

args=(
    --profile umhpc
    --name "$RUN_NAME"
    --moa "$MOA"
    --epochs "$EPOCHS"
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
    sbatch --export=ALL,FRESH=0,CHAIN=$NEXT,MOA=$MOA,RUN_NAME=$RUN_NAME,PER_GPU_BATCH=$PER_GPU_BATCH,LR=$LR \
        "$PROJECT_DIR/scripts/train_moa_umhpc.sh"
    exit 0
fi
echo "=== train_moa.py 退出码 $RC ==="
exit $RC
