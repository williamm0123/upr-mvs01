from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[1]


MACHINE: Literal["ubuntu", "windows", "umhpc"] = "ubuntu"

TRAIN_PROFILE: Literal["local", "umhpc"] = "local"


def _default_paths() -> dict[str, Path]:
    if MACHINE not in {"ubuntu", "windows", "umhpc"}:
        raise ValueError(
            f"Unsupported MACHINE={MACHINE!r}. "
            "Expected one of: 'ubuntu', 'windows', 'umhpc'."
        )

    if MACHINE == "windows":
        project_path = Path(r"E:/documents/Project/point based/uprmvs01")
        return {
            "project_path": project_path,
            "dtu_train_root": Path(r"E:/documents/dataset/DTU/dtu_training"),
            "dtu_test_root": Path(r"E:/documents/dataset/DTU/dtu_testing"),
            "dtu_list_path": project_path / "lists/dtu/test.txt",
            "dinov3_weights_file": Path(
                r"E:/documents/dataset/DINOv3/pre_trained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            ),
            "da3_weights_file": Path("E:/documents/dataset/pretrained/DA3/DA3MONO-LARGE"),
            "vggt_weights_path": Path("E:/documents/dataset/VGGT/pretrained/VGGT-1B"),
            "offline_prior_root": project_path / "outputs/sfm_da3_loggrad_fill_testset_denoised",
        }

    if MACHINE == "umhpc":
        project_path = Path("/scr/user/qinglong/projects/upr-mvs01")
        return {
            "project_path": project_path,
            "dtu_train_root": Path("/scr/user/qinglong/dataset/DTU/dtu_training"),
            "dtu_test_root": Path("/scr/user/qinglong/dataset/DTU/dtu_testing"),
            "dtu_list_path": project_path / "lists/dtu/test.txt",
            "dinov3_weights_file": Path(
                "/scr/user/qinglong/dataset/DINOv3/pre_trained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            ),
            "da3_weights_file": Path("/scr/user/qinglong/dataset/pretrained/DA3/DA3MONO-LARGE"),
            "vggt_weights_path": Path("/scr/user/qinglong/dataset/VGGT/pretrained/VGGT-1B"),
            "offline_prior_root": project_path / "outputs/sfm_da3_loggrad_fill_testset_denoised",
        }

    project_path = Path("/home/william/project/uprmvs01")
    return {
        "project_path": project_path,
        "dtu_train_root": Path("/home/william/project/dataset/DTU/dtu_training"),
        "dtu_test_root": Path("/home/william/project/dataset/DTU/dtu_test"),
        "dtu_list_path": project_path / "lists/dtu/test.txt",
        "dinov3_weights_file": Path(
            "/home/william/project/dataset/DINOv3/pre_trained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ),
        "da3_weights_file": Path("/home/william/project/dataset/DA3/DA3MONO-LARGE"),
        "vggt_weights_path": Path("/home/william/project/dataset/VGGT/pretrained/VGGT-1B"),
        "offline_prior_root": project_path / "outputs/sfm_da3_loggrad_fill_testset_denoised",
    }


_DEFAULT_PATHS = _default_paths()


@dataclass(frozen=True)
class ProjectPaths:
    project_path: Path = _DEFAULT_PATHS["project_path"]
    dtu_train_root: Path = _DEFAULT_PATHS["dtu_train_root"]
    dtu_test_root: Path = _DEFAULT_PATHS["dtu_test_root"]
    dtu_list_path: Path = _DEFAULT_PATHS["dtu_list_path"]
    dinov3_weights_file: Path = _DEFAULT_PATHS["dinov3_weights_file"]
    da3_weights_file: Path = _DEFAULT_PATHS["da3_weights_file"]
    vggt_weights_path: Path = _DEFAULT_PATHS["vggt_weights_path"]
    offline_prior_root: Path = _DEFAULT_PATHS["offline_prior_root"]
    train_list_file: Path = _DEFAULT_PATHS["project_path"] / "lists/dtu/train.txt"
    val_list_file: Path = _DEFAULT_PATHS["project_path"] / "lists/dtu/val.txt"
    test_list_file: Path = _DEFAULT_PATHS["project_path"] / "lists/dtu/test.txt"
    output_root: Path = _DEFAULT_PATHS["project_path"] / "outputs"


@dataclass(frozen=True)
class DataConfig:
    target_h: int = 512
    target_w: int = 640
    nviews: int = 3
    feature_strides: tuple[int, ...] = (4, 8, 16)
    pair_min_overlap: float = 0.30
    pair_min_baseline_deg: float = 5.0
    pair_max_baseline_deg: float = 45.0
    use_pair_filter: bool = True


@dataclass(frozen=True)
class DINOConfig:
    patch_size: int = 16
    input_max_side: int = 512
    layers: tuple[int, ...] = (3, 6, 9, 11)
    project_channels: int = 128
    random_projection_seed: int = 20260416
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FPNConfig:
    backbone: str = "resnet50"
    out_channels: int = 128
    pretrained: bool = True


@dataclass(frozen=True)
class VGGTPriorConfig:
    # offline: use precomputed depth_fill priors from ProjectPaths.offline_prior_root.
    # online: run VGGT inside the training step.
    # auto: use offline when present, otherwise run VGGT.
    # none: train without a geometric prior.
    prior_source: Literal["offline", "online", "auto", "none"] = "offline"
    offline_confidence: float = 0.9
    offline_prior_required: bool = False
    generate_missing_offline: bool = True
    offline_generation_light_idx: int = 3
    offline_generation_include_val: bool = True
    offline_generation_max_groups: int = 0
    offline_sparse_min_confidence: float = 0.5
    offline_sparse_keep_ratio: float = 0.35
    offline_denoise_points: bool = True
    offline_denoise_max_points: int = 50000
    confidence_w_vggt: float = 0.5
    confidence_w_reproj: float = 0.3
    confidence_w_normal: float = 0.2
    sor_k: int = 20
    sor_std_ratio: float = 2.0
    normal_neighbor_k: int = 16
    normal_consistency_deg: float = 30.0
    reproj_threshold_px: float = 4.0
    enabled: bool = True


@dataclass(frozen=True)
class GeoFusionConfig:
    geo_channels: int = 128
    encoder_hidden: int = 64
    init_alpha: float = 0.0
    alpha_warmup_steps: int = 10000
    alpha_release_steps: int = 30000
    alpha_max_during_warmup: float = 0.1


@dataclass(frozen=True)
class DepthRangeConfig:
    sigma_max_ratio: float = 0.15
    k_sigma: float = 3.0
    uncertain_threshold: float = 0.3


@dataclass(frozen=True)
class CostVolumeConfig:
    num_groups: int = 8
    num_depths_stage1: int = 48
    num_depths_stage2: int = 32
    num_depths_stage3: int = 16
    num_depths_uncertain: int = 64
    interval_ratio_stage2: float = 0.5
    interval_ratio_stage3: float = 0.25


@dataclass(frozen=True)
class AnchorPEConfig:
    num_anchors: int = 24
    min_visible_views_ratio: float = 0.5
    min_confidence: float = 0.7
    pe_hidden: int = 64
    pe_out_channels: int = 64
    lambda_warmup_steps: int = 20000
    lambda_release_steps: int = 30000


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
    w_depth: float = 1.0
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
    num_depths_stage1: int = 48
    num_depths_stage2: int = 32
    num_depths_stage3: int = 16
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
    use_vggt_prior: bool = True
    use_anchor_pe: bool = True
    use_geo_fusion: bool = True
    use_points_alignment: bool = True


def _train_local() -> TrainConfig:
    return TrainConfig(
        profile="local",
        batch_size=1,
        num_workers=2,
        num_views=3,
        num_depths_stage1=32,
        num_depths_stage2=24,
        num_depths_stage3=8,
        lr=1.0e-4,
        weight_decay=1.0e-4,
        max_steps=5000,
        warmup_steps=200,
        grad_clip=1.0,
        amp=True,
        seed=20260526,
        log_interval=10,
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
        num_depths_stage1=96,
        num_depths_stage2=48,
        num_depths_stage3=24,
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
    dino: DINOConfig = field(default_factory=DINOConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    vggt_prior: VGGTPriorConfig = field(default_factory=VGGTPriorConfig)
    geo_fusion: GeoFusionConfig = field(default_factory=GeoFusionConfig)
    depth_range: DepthRangeConfig = field(default_factory=DepthRangeConfig)
    cost_volume: CostVolumeConfig = field(default_factory=CostVolumeConfig)
    anchor_pe: AnchorPEConfig = field(default_factory=AnchorPEConfig)
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
