#!/bin/bash -l

set -euo pipefail


PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
# 直接指定 uprmvs 环境的解释器，不依赖当前 shell 的 conda activate/PATH。
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/uprmvs/bin/python}
TRAIN_PROFILE=${TRAIN_PROFILE:-local}   # local = 单卡档 (max_steps 20000)；要 30000 步就设 umhpc 或 STEPS=30000
RUN_NAME=${RUN_NAME:-uprmvs_1gpu_${SLURM_JOB_ID:-manual}}

# 核心训练参数（命令行会覆盖 TRAIN_PROFILE 中的同名参数）
# !! 末级假设数已从 4 改为 8 (base/config.py num_depths_stage4)。stage 4 是全
#    分辨率级、显存占大头, 这一改把最大的那一项翻倍 (512x640 下 cost volume 体素
#    2.54M -> 3.86M, 约 1.5x)。OOM 的退让顺序:
#      1) BATCH_SIZE 2 -> 1
#      2) UPRMVS_MAX_SCALE=576 (砍掉 augment 里 640x832 / 640x896 两档)
#      3) 把 num_depths_stage4 改回 4 (那就等于放弃 P5 这个实验)
BATCH_SIZE=${BATCH_SIZE:-2}       # 每卡 batch；SPRE 现在对全部 NUM_VIEWS 个视角跑 DINOv3，OOM 就设 1
NUM_VIEWS=${NUM_VIEWS:-5}         # MVS 总视图数：1 个参考视图 + 4 个源视图
NUM_WORKERS=${NUM_WORKERS:-16}    # DataLoader 进程数；32 CPU 下建议 8~16
LEARNING_RATE=${LEARNING_RATE:-3.0e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-500} # 500 步一次 val；val 集 882 样本，约占 8% 训练时间，嫌慢设 1000
AMP=${AMP:-on}                    # on/off；A100 建议 on
STEPS=${STEPS:-0}                 # 0=使用 profile 默认值；测试可设 2
SPRE=${SPRE:-on}                  # on/off：DINOv3 先验可靠度头（SPRE）

# FRESH=1（默认）：训练开始前把 log/model/ 里已有的 .pth 整体挪到
# log/model.bak_<时间戳>/。挪走不是删除，但本次训练绝对读不到任何旧权重。
# 之所以必须物理隔离而不是靠加载报错兜底：末级 D4->D8 不改变 state_dict
# （Conv3d 权重是 [out,in,3,3,3]，与 D 无关），旧 checkpoint 会干净地加载进来
# 然后训练一个不是你要的模型。train.py 的级联签名检查是第二道防线。
FRESH=${FRESH:-1}
# RESUME=auto 在 FRESH=1 之后是安全的：目录已被清空，auto 找不到东西，从 step 0
# 开始；SLURM 重排队时 job id 不变，preflight 认出是同一次运行、跳过归档，auto
# 于是正确续上本次训练自己写的 checkpoint。要完全禁用续跑就设 RESUME=off。
RESUME=${RESUME:-auto}

# 先验与跑通测试
BUILD_PRIORS=${BUILD_PRIORS:-auto}
# BUILD_PRIORS: auto=补齐缺失先验，force=全部重算，skip=要求缓存已存在，
#               only=只构建先验然后退出（换 val 列表后先跑一次这个）
SMOKE=${SMOKE:-0}                 # 1=合成数据跑通测试；0=真实数据训练
SMOKE_STEPS=${SMOKE_STEPS:-2}     # SMOKE=1 时执行的训练步数
# =============================================================================

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    echo "Override it with: PYTHON_BIN=/path/to/uprmvs/bin/python bash $0" >&2
    exit 1
fi

cd "$PROJECT_DIR"

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE="$TRAIN_PROFILE"
# vggt 是 PROJECT_DIR/models 下的顶层 namespace package；不继承外部
# PYTHONPATH，避免误导入 /scr/user/qinglong/projects/vggt。
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTHONUNBUFFERED=1
# 缓解显存碎片（错误信息里 "reserved but unallocated" 就是碎片）。可被外部覆盖。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}


echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) profile=$TRAIN_PROFILE ==="
nvidia-smi -L
echo "=== python=$PYTHON_BIN ==="
"$PYTHON_BIN" -c 'import importlib.util, sys, torch, huggingface_hub; print("python:", sys.executable); print("torch:", torch.__version__, torch.__file__); print("huggingface_hub:", huggingface_hub.__version__); print("vggt:", importlib.util.find_spec("vggt.models.vggt").origin)'

# 确认跑的是改动后的代码，并（FRESH=1 时）把旧 checkpoint 挪开。
# 任一检查失败说明 checkout 是旧的，直接退出而不是白跑一天。
# SMOKE 不归档：冒烟跑合成数据、写 log/model_smoke/，不该动真实训练的目录。
preflight_args=()
if [[ "$FRESH" == "1" && "$SMOKE" != "1" ]]; then
    preflight_args+=(--fresh --run-id "${SLURM_JOB_ID:-manual}")
fi
"$PYTHON_BIN" scripts/preflight.py "${preflight_args[@]}"

echo "=== batch=$BATCH_SIZE views=$NUM_VIEWS workers=$NUM_WORKERS lr=$LEARNING_RATE warmup=$WARMUP_STEPS \
val_interval=$VAL_INTERVAL amp=$AMP steps=$STEPS spre=$SPRE fresh=$FRESH resume=$RESUME \
build_priors=$BUILD_PRIORS smoke=$SMOKE max_scale=${UPRMVS_MAX_SCALE:-none} ==="

train_args=(
    --profile "$TRAIN_PROFILE"
    --gpus 1
    --ddp off
    --batch-size "$BATCH_SIZE"
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --lr "$LEARNING_RATE"
    --warmup-steps "$WARMUP_STEPS"
    --val-interval "$VAL_INTERVAL"
    --amp "$AMP"
    --spre "$SPRE"
    --name "$RUN_NAME"
)

case "$SMOKE" in
    1|true|TRUE|yes|YES)
        train_args+=(
            --smoke
            --smoke-steps "$SMOKE_STEPS"
            --build-priors skip
        )
        ;;
    0|false|FALSE|no|NO)
        train_args+=(
            --steps "$STEPS"
            --build-priors "$BUILD_PRIORS"
            --resume "$RESUME"
        )
        ;;
    *)
        echo "SMOKE must be 0/1, true/false, or yes/no; got: $SMOKE" >&2
        exit 2
        ;;
esac

exec "$PYTHON_BIN" train.py "${train_args[@]}"
