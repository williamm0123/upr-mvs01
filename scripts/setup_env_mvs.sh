#!/bin/bash -l
# =============================================================================
# upr-mvs01 环境搭建脚本 (umhpc / A100, sm_80)
# 目标：全新 conda 环境 mvs，核心模块 DA3 / VGGT / dinov3 靠 PYTHONPATH 引用源码，
#      只装依赖、不把 DA3 当包 pip install（避开 hatch-vcs + xformers 编译）。
#
# 用法：  bash scripts/setup_env_mvs.sh
#         全程可重复执行（幂等）。
# =============================================================================
set -euo pipefail

ENV_NAME=mvs
PY_VER=3.11
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">>> 项目根目录: ${PROJ_ROOT}"
source ~/.bashrc

# ---------------------------------------------------------------------------
# 1) 创建环境（已存在则跳过）
# ---------------------------------------------------------------------------
if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  conda create -y -n "${ENV_NAME}" python=${PY_VER}
else
  echo ">>> 环境 ${ENV_NAME} 已存在，跳过创建"
fi
conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1          # 隔离 ~/.local，避免串包

python -m pip install -U pip setuptools wheel

# ---------------------------------------------------------------------------
# 2) PyTorch 栈（cu121 wheel，精确配对，全部走预编译）
# ---------------------------------------------------------------------------
pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

# xformers 必须匹配 torch 2.5.1，否则 import DA3 时段错误/符号缺失
pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 3) numpy 先钉死 <2（很多包会顺手把它升到 2.x，先占位）
# ---------------------------------------------------------------------------
pip install "numpy==1.26.4"

# ---------------------------------------------------------------------------
# 4) DA3 依赖（不含 torch/xformers/numpy，已在上面装好）
#    从 requirements.txt 去掉这三项，其余照装
# ---------------------------------------------------------------------------
pip install \
  einops omegaconf safetensors huggingface_hub \
  "opencv-python" imageio "pillow" pillow_heif \
  trimesh plyfile open3d \
  e3nn evo \
  "moviepy==1.0.3" \
  pycolmap \
  scipy matplotlib tqdm requests \
  "typer>=0.9.0" fastapi uvicorn

# ---------------------------------------------------------------------------
# 5) VGGT 依赖
# ---------------------------------------------------------------------------
pip install hydra-core
pip install "lightglue @ git+https://github.com/cvg/LightGlue.git"

# ---------------------------------------------------------------------------
# 6) 收尾：确保没有包偷偷把 numpy 升到 2.x
# ---------------------------------------------------------------------------
pip install "numpy==1.26.4"

# ---------------------------------------------------------------------------
# 7) 自检
# ---------------------------------------------------------------------------
export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/models:${PROJ_ROOT}/models/Depth-Anything-3/src:${PYTHONPATH:-}"

python - <<'PY'
import numpy, torch
print("numpy   :", numpy.__version__, "(需 <2)")
print("torch   :", torch.__version__, "| cuda:", torch.version.cuda, "| avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu     :", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
import xformers; print("xformers:", xformers.__version__)
from depth_anything_3.api import DepthAnything3   # 验证 DA3 靠 PYTHONPATH 可导入
print("DA3 api : OK")
import vggt.models.vggt as _; print("vggt    : OK")
from lightglue import SuperPoint; print("lightglue: OK")
import pycolmap, e3nn, evo, open3d; print("pycolmap/e3nn/evo/open3d: OK")
print("\n>>> 环境自检全部通过 ✅")
PY

echo ">>> 完成。训练前请：conda activate ${ENV_NAME}"
