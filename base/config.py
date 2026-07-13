from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[1]


# Machine (which filesystem layout to use) is resolved at import time because
# ProjectPaths() is constructed ad-hoc in several places (dtu / norm_fill /
# pre_prior / dinov3). Drive it from the environment so a cluster job switches
# *all* of them at once, e.g. the Slurm script does `export UPRMVS_MACHINE=umhpc`.
MACHINE: Literal["ubuntu", "umhpc"] = os.environ.get("UPRMVS_MACHINE", "ubuntu")  # type: ignore[assignment]

# Default training profile follows the machine (still overridable by --profile).
TRAIN_PROFILE: Literal["local", "umhpc"] = os.environ.get(  # type: ignore[assignment]
    "UPRMVS_PROFILE", "umhpc" if MACHINE == "umhpc" else "local"
)


def _default_paths() -> dict[str, Path]:
    if MACHINE == "umhpc":
        project_path = Path("/scr/user/qinglong/projects/upr-mvs01")
        data_path = Path("/scr/user/qinglong/dataset")
    else:
        project_path = Path("/home/william/project/uprmvs01")
        data_path = Path("/home/william/project/dataset")

    return {
            "project_path": project_path,
            "output_root": project_path/ "uprmvs_outputs",
            "dtu_train_root": data_path / "DTU/dtu_training",
            "dtu_test_root": data_path / "DTU/dtu_testing",
            "dtu_list_path": project_path / "lists/dtu/train.txt",
            "sfm_cache_path":project_path / "log/sfm_depth",
            "resnet50_weights_file": data_path / "Resnet50/Model_v2.pth",
            "dinov3_weights_file":
                data_path/"DINOv3/pre_trained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "da3_weights_file": data_path / "DA3/pretrained/DA3MONO-LARGE/",
            "vggt_weights_path": data_path / "VGGT/pretrained/VGGT-1B",
        }



_DEFAULT_PATHS = _default_paths()


@dataclass(frozen=True)
class ProjectPaths:
    project_path: Path = _DEFAULT_PATHS["project_path"]
    dtu_train_root: Path = _DEFAULT_PATHS["dtu_train_root"]
    dtu_test_root: Path = _DEFAULT_PATHS["dtu_test_root"]
    dtu_list_path: Path = _DEFAULT_PATHS["dtu_list_path"]
    sfm_cache_path: Path = _DEFAULT_PATHS["sfm_cache_path"]
    resnet50_weights_file: Path = _DEFAULT_PATHS["resnet50_weights_file"]
    dinov3_weights_file: Path = _DEFAULT_PATHS["dinov3_weights_file"]
    da3_weights_file: Path = _DEFAULT_PATHS["da3_weights_file"]
    vggt_weights_path: Path = _DEFAULT_PATHS["vggt_weights_path"]
    train_list_file: Path = _DEFAULT_PATHS["project_path"] / "lists/dtu/train.txt"
    val_list_file: Path = _DEFAULT_PATHS["project_path"] / "lists/dtu/val.txt"
    test_list_file: Path = _DEFAULT_PATHS["project_path"] / "lists/dtu/test.txt"
    output_root: Path = _DEFAULT_PATHS["output_root"]


@dataclass(frozen=True)
class DataConfig:
    target_h: int = 512
    target_w: int = 640
    nviews: int = 3
    feature_strides: tuple[int, ...] = (1, 2, 4)
    pair_min_overlap: float = 0.30
    pair_min_baseline_deg: float = 5.0
    pair_max_baseline_deg: float = 45.0
    use_pair_filter: bool = True




@dataclass(frozen=True)
class FPNConfig:
    out_channels: int = 128
    base_channel: int = 32



@dataclass(frozen=True)
class DepthRangeConfig:
    sigma_max_ratio: float = 0.15
    k_sigma: float = 3.0
    uncertain_threshold: float = 0.3


@dataclass(frozen=True)
class CostVolumeConfig:
    num_groups: int = 8
    num_depths_stage1: int = 48
    num_depths_stage2: int = 16
    num_depths_stage3: int = 16
    num_depths_uncertain: int = 64
    interval_ratio_stage2: float = 0.25
    interval_ratio_stage3: float = 0.1
    # warp-channel width per cascade stage (must be divisible by num_groups).
    # Shrinks as resolution grows so the full-res stage does not OOM: the warp
    # intermediate is [B, warp_channels, D, H, W].
    warp_channels_stage1: int = 128
    warp_channels_stage2: int = 64
    warp_channels_stage3: int = 32
    # sample/correlate features in fp16 on CUDA (geometry stays fp32) for a
    # further ~2x memory cut over the channel reduction alone.
    warp_use_half: bool = True
    # per-source cost-volume weighting: True uses batch["src_weights"]; False
    # forces uniform averaging regardless of what the batch provides.
    use_src_weights: bool = False


@dataclass(frozen=True)
class PointsAlignmentConfig:
    epipolar_search_radius_px: float = 2.0
    knn_k: int = 5
    knn_max_distance_world: float = 50.0
    filled_confidence: float = 0.2
    enabled: bool = True


@dataclass(frozen=True)
class DecoderConfig:
    unet_base_channels: int = 16
    unet_depth: int = 3
    use_residual_to_vggt: bool = True


@dataclass(frozen=True)
class LossConfig:
    w_depth: float = 1.0      # weight of the per-stage cross-entropy (classification) term
    w_reg: float = 1.0        # weight of the per-stage smooth-L1 (regression) term
    w_grad: float = 0.5
    w_normal: float = 0.5
    w_residual: float = 0.1
    w_ssim: float = 0.1
    w_feat: float = 0.05
    residual_b_scale: float = 0.1
    residual_min_confidence: float = 0.3
    residual_relative: bool = True
    use_cross_entropy: bool = True
    residual_warmup_steps: int = 20000
    ssim_warmup_steps: int = 20000
    feat_warmup_steps: int = 50000


@dataclass(frozen=True)
class StageWeights:
    stage1: float = 0.5
    stage2: float = 1.0
    stage3: float = 2.0


@dataclass(frozen=True)
class TrainConfig:
    profile: str = TRAIN_PROFILE
    batch_size: int = 1
    num_workers: int = 2
    num_views: int = 3
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    max_steps: int = 200000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 20260526
    log_interval: int = 50
    vis_interval: int = 500
    vis_max_views: int = 5
    val_interval: int = 2000
    ckpt_interval: int = 5000
    devices: tuple[int, ...] = (0,)
    distributed: bool = False
    use_anchor_pe: bool = True
    use_geo_fusion: bool = True
    use_points_alignment: bool = True


def _train_local() -> TrainConfig:
    return TrainConfig(
        profile="local",
        batch_size=1,
        num_workers=2,
        num_views=3,
        lr=1.0e-4,
        weight_decay=1.0e-4,
        max_steps=200000,
        warmup_steps=200,
        grad_clip=1.0,
        amp=True,
        seed=20260526,
        log_interval=20,
        vis_interval=100,
        vis_max_views=3,
        val_interval=500,
        ckpt_interval=1000,
        devices=(0,),
        distributed=False,
    )


def _train_umhpc() -> TrainConfig:
    return TrainConfig(
        profile="umhpc",
        batch_size=4,
        num_workers=8,
        num_views=5,
        lr=2.0e-4,
        weight_decay=1.0e-4,
        max_steps=200000,
        warmup_steps=1000,
        grad_clip=1.0,
        amp=True,
        seed=20260526,
        log_interval=50,
        vis_interval=500,
        vis_max_views=5,
        val_interval=2000,
        ckpt_interval=5000,
        devices=(0, 1, 2, 3),
        distributed=True,
    )


def get_train_config(profile: str | None = None) -> TrainConfig:
    profile = profile or TRAIN_PROFILE
    if profile == "local":
        return _train_local()
    if profile == "umhpc":
        return _train_umhpc()
    raise ValueError(f"Unknown train profile: {profile!r}")


@dataclass(frozen=True)
class MVSConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    data: DataConfig = field(default_factory=DataConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    depth_range: DepthRangeConfig = field(default_factory=DepthRangeConfig)
    cost_volume: CostVolumeConfig = field(default_factory=CostVolumeConfig)
    points_alignment: PointsAlignmentConfig = field(default_factory=PointsAlignmentConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    stage_weights: StageWeights = field(default_factory=StageWeights)
    train: TrainConfig = field(default_factory=lambda: get_train_config(None))


def build_mvs_config(profile: str | None = None) -> MVSConfig:
    cfg = MVSConfig()
    if profile is not None and profile != cfg.train.profile:
        cfg = MVSConfig(train=get_train_config(profile))
    return cfg
