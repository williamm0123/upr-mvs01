"""Shared configuration defaults for the notebook-derived experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path = REPO_ROOT
    upr_mvs_root: Path = field(
        default_factory=lambda: _env_path("UPR_MVS_PROJECT_ROOT", "/home/william/project/UPR-MVS")
    )
    dtu_train_root: Path = field(
        default_factory=lambda: _env_path("DTU_TRAIN_ROOT", "/home/william/project/dataset/DTU/dtu_training")
    )
    dtu_test_root: Path = field(
        default_factory=lambda: _env_path("DTU_TEST_ROOT", "/home/william/project/dataset/DTU/dtu_test")
    )
    dtu_list_path: Path = field(default_factory=lambda: REPO_ROOT / "lists/dtu/test.txt")
    dinov3_weights_file: Path = field(
        default_factory=lambda: _env_path(
            "DINOV3_WEIGHTS_FILE",
            "/home/william/project/dataset/DINOv3/pre_trained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        )
    )


@dataclass(frozen=True)
class DTUConfig:
    n_views: int = 3
    light_id: int = 3
    split: str = "train"
    image_dir: str = "Rectified_raw"
    depth_dir: str = "Depths_raw"
    n_depths: int = 192


@dataclass(frozen=True)
class CostVolumeConfig:
    scale: float = 0.25
    num_depths: int = 64
    temperature: float = 0.02
    channel_chunk: int = 4


@dataclass(frozen=True)
class DINOConfig:
    patch_size: int = 16
    input_max_side: int = 384
    layers: tuple[int, ...] = tuple(range(12))
    project_channels: int = 32
    random_projection_seed: int = 20260416
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AdapterConfig:
    single_layer_id: int = 6
    layer_ids: tuple[int, ...] = (1, 6, 12)
    hidden_ch: int = 128
    out_ch: int = 64
    train_steps: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ce_weight: float = 1.0
    l1_weight: float = 0.25
    grad_clip: float | None = 1.0
    train_grid_divisor: int = 2
