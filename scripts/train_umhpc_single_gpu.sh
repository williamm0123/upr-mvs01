#!/bin/bash -l
#SBATCH --job-name=uprmvs1g
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/scr/user/qinglong/projects/upr-mvs01
#SBATCH --output=/scr/user/qinglong/projects/upr-mvs01/slurm-%x-%j.out
#SBATCH --error=/scr/user/qinglong/projects/upr-mvs01/slurm-%x-%j.err

set -euo pipefail


PROJECT_DIR=${PROJECT_DIR:-/scr/user/qinglong/projects/upr-mvs01}
# 直接指定 uprmvs 环境的解释器，不依赖当前 shell 的 conda activate/PATH。
PYTHON_BIN=${PYTHON_BIN:-/home/user/qinglong/.conda/envs/uprmvs/bin/python}
TRAIN_PROFILE=${TRAIN_PROFILE:-umhpc}  # 单卡也用正式 30000-step profile；--gpus/--ddp 会覆盖其多卡设置
RUN_NAME=${RUN_NAME:-uprmvs_1gpu_${SLURM_JOB_ID:-manual}}

# 核心训练参数（命令行会覆盖 TRAIN_PROFILE 中的同名参数）
BATCH_SIZE=${BATCH_SIZE:-2}       # 每卡 batch；SPRE 现在对全部 NUM_VIEWS 个视角跑 DINOv3，OOM 就设 1
NUM_VIEWS=${NUM_VIEWS:-5}         # MVS 总视图数：1 个参考视图 + 4 个源视图
NUM_WORKERS=${NUM_WORKERS:-16}    # DataLoader 进程数；32 CPU 下建议 8~16
LEARNING_RATE=${LEARNING_RATE:-3.0e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-500} # 500 步一次 val；val 集 882 样本，约占 8% 训练时间，嫌慢设 1000
AMP=${AMP:-on}                    # on/off；A100 建议 on
STEPS=${STEPS:-0}                 # 0=使用 profile 默认值；测试可设 2

# 三个已解耦的消融开关。正式默认 = all-view DINO 注入 FPN + SPRE reliability。
# 例：纯 FPN/cached 消融：
#   sbatch --export=ALL,DINO_MODE=off,FEED_FPN=off,RELIABILITY=cached scripts/train_umhpc_single_gpu.sh
DINO_MODE=${DINO_MODE:-all_view}  # off / all_view / ref_only (ref_only 尚未实现时会明确报错)
FEED_FPN=${FEED_FPN:-on}          # on / off
RELIABILITY=${RELIABILITY:-spre}  # cached / edge / spre
# 2026-08-16: prior 缓存已全部标尺 (29302 个, 未标尺 0, 尺度离群 0),
# *_clean.txt / exclude_*.csv 是当初为了绕开 8.35% 未标尺样本做的剔除表,
# 现在留着只会白少 10 个 train scan (79->69) 和 294 个样本。除非要复现旧
# 实验, 否则保持 off。
CLEAN_LISTS=${CLEAN_LISTS:-off}

# RESUME: auto=从 log/model/latest.pth 续跑（SLURM 重排队要靠它）；off=从头开始。
# ！！4 级级联重构之后（3 级 -> 4 级、假设数 48-16-8-4、DINO 进 FPN），旧
# checkpoint 的 state_dict/配置对不上。本轮是新架构 fresh run，默认 off；
# Slurm 到时限后重提同一配置时再用 RESUME=auto。
RESUME=${RESUME:-off}

# 先验与跑通测试
# 缓存已完整; auto 只是核验一遍 (models/pre_prior.py 提速后约 5 秒,
# 若集群上还是旧版会读满 98GB/9 分钟, 那就用 skip)。
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

# 供下面的 checkpoint fingerprint 自检读取。
export TRAIN_PROFILE BATCH_SIZE NUM_VIEWS NUM_WORKERS LEARNING_RATE WARMUP_STEPS
export VAL_INTERVAL AMP DINO_MODE FEED_FPN RELIABILITY CLEAN_LISTS

case "$DINO_MODE:$FEED_FPN:$RELIABILITY" in
    off:*:spre)
        echo "Invalid config: RELIABILITY=spre requires DINO_MODE=all_view/ref_only" >&2
        exit 2
        ;;
    off:on:*)
        echo "Invalid config: DINO_MODE=off requires FEED_FPN=off" >&2
        exit 2
        ;;
    all_view:off:cached|all_view:off:edge)
        echo "Invalid config: DINO is enabled but neither FPN nor SPRE consumes it" >&2
        exit 2
        ;;
esac

# --spre 仅表示是否实例化可训练的 SPRE head，由 reliability 自动推导，
# 避免 edge/cached 消融在 DDP 中留下 unused parameters。
if [[ "$RELIABILITY" == "spre" ]]; then
    SPRE=on
else
    SPRE=off
fi
export SPRE

# 旧 checkpoint 配 auto 必然崩，且要跑完先验构建才崩，白等很久。
# 新 checkpoint 会保存 fingerprint，在这里与本次 sbatch 配置严格比较。
if [[ "$RESUME" == "auto" && -f "log/model/latest.pth" ]]; then
    "$PYTHON_BIN" - <<'PYCHECK'
import os
import sys
from dataclasses import replace

import torch

from base.config import build_mvs_config
from train import _arch_fingerprint

cfg = build_mvs_config(profile=os.environ["TRAIN_PROFILE"])
cfg = replace(
    cfg,
    train=replace(
        cfg.train,
        batch_size=int(os.environ["BATCH_SIZE"]),
        num_views=int(os.environ["NUM_VIEWS"]),
        num_workers=int(os.environ["NUM_WORKERS"]),
        lr=float(os.environ["LEARNING_RATE"]),
        warmup_steps=int(os.environ["WARMUP_STEPS"]),
        val_interval=int(os.environ["VAL_INTERVAL"]),
        amp=os.environ["AMP"] == "on",
    ),
    dino=replace(
        cfg.dino,
        mode=os.environ["DINO_MODE"],
        feed_fpn=os.environ["FEED_FPN"] == "on",
    ),
    spre=replace(
        cfg.spre,
        enabled=os.environ["SPRE"] == "on",
        reliability_source=os.environ["RELIABILITY"],
    ),
)
ckpt = torch.load("log/model/latest.pth", map_location="cpu", weights_only=False)
saved = ckpt.get("fingerprint")
current = _arch_fingerprint(cfg)
if saved is None:
    sys.exit(
        "RESUME=auto but latest.pth has no architecture fingerprint. "
        "Use RESUME=off for a fresh run or resume it with the matching old code."
    )
if saved != current:
    keys = sorted(set(saved) | set(current))
    diff = "\n".join(
        f"  {k}: checkpoint={saved.get(k)!r} current={current.get(k)!r}"
        for k in keys if saved.get(k) != current.get(k)
    )
    sys.exit("RESUME=auto configuration mismatch:\n" + diff)
print(f"resume check: fingerprint matches (step={ckpt.get('step', '?')})")
PYCHECK
elif [[ "$RESUME" == "off" && -f "log/model/latest.pth" ]]; then
    echo "WARNING: RESUME=off and log/model/latest.pth exists."
    echo "The first checkpoint of this fresh run will overwrite latest.pth/best.pth."
fi

echo "=== job=${SLURM_JOB_ID:-manual} host=$(hostname) profile=$TRAIN_PROFILE ==="
echo "=== submit_dir=${SLURM_SUBMIT_DIR:-manual} project=$PROJECT_DIR ==="
nvidia-smi -L
echo "=== python=$PYTHON_BIN ==="
"$PYTHON_BIN" -c 'import importlib.util, sys, torch, huggingface_hub; print("python:", sys.executable); print("torch:", torch.__version__, torch.__file__); print("huggingface_hub:", huggingface_hub.__version__); print("vggt:", importlib.util.find_spec("vggt.models.vggt").origin)'

# 确认跑的是改动后的代码：LayerScale 必须已接回（否则 DINOv3 特征是常数图），
# CrossViTFusion 必须存在。任一缺失说明 checkout 是旧的，直接退出而不是白跑一天。
"$PYTHON_BIN" - <<'PYCHECK'
import sys
from models.dinov3.vision_transformer import vit_base
from models.spre import SVAFusion, DinoSVA  # noqa: F401
from base.config import build_mvs_config
from models.cost_volume import VisibilityHead
from models.depth_range import apply_branch_prior
from models.network import UprMVSNet
blk = vit_base(patch_size=16, n_storage_tokens=4).blocks[0]
if not hasattr(blk.ls1, "gamma"):
    sys.exit("DINOv3 LayerScale 仍是 nn.Identity —— 这是旧代码，git pull 后再跑")
if UprMVSNet.fpn_stage_strides != (8, 4, 2, 1):
    sys.exit(f"级联仍是 {UprMVSNet.fpn_stage_strides} —— 这是旧代码，git pull 后再跑")
cfg = build_mvs_config(profile="umhpc")
if (cfg.depth_range.num_global, cfg.depth_range.num_local) != (40, 8):
    sys.exit("stage1 candidate split is not 40/8")
if cfg.depth_range.dual_mode_stage2:
    sys.exit("dual_mode_stage2 must remain off until the two-volume implementation is ready")
print("code check: LayerScale/SVAFusion/VisibilityHead/branch prior/40+8 cascade ok")
PYCHECK

echo "=== batch=$BATCH_SIZE views=$NUM_VIEWS workers=$NUM_WORKERS lr=$LEARNING_RATE warmup=$WARMUP_STEPS \
val_interval=$VAL_INTERVAL amp=$AMP steps=$STEPS dino=$DINO_MODE feed_fpn=$FEED_FPN \
reliability=$RELIABILITY clean_lists=$CLEAN_LISTS resume=$RESUME build_priors=$BUILD_PRIORS smoke=$SMOKE ==="

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
    --dino-mode "$DINO_MODE"
    --feed-fpn "$FEED_FPN"
    --reliability "$RELIABILITY"
    --spre "$SPRE"
    --name "$RUN_NAME"
)

case "$CLEAN_LISTS" in
    on|ON|1|true|TRUE|yes|YES) ;;
    off|OFF|0|false|FALSE|no|NO) train_args+=(--no-clean-lists) ;;
    *)
        echo "CLEAN_LISTS must be on/off; got: $CLEAN_LISTS" >&2
        exit 2
        ;;
esac

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
