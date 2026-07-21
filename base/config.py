from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Single profile switch.
#
# ``profile`` is the ONLY knob that selects a machine: it drives BOTH the
# filesystem paths (below) and the training hyper-parameters (get_train_config).
# There is intentionally no separate UPRMVS_MACHINE anymore — one name, one
# source of truth.
#
#   * default comes from env ``UPRMVS_PROFILE`` (batch scripts export it);
#   * CLI ``--profile`` overrides it at runtime via ``set_profile`` / the
#     ``build_mvs_config(profile=...)`` entry point.
#
# ProjectPaths reads the *active* profile at construction time (default_factory),
# so every bare ``ProjectPaths()` call across the codebase follows the CLI
# selection without having to thread the profile through.
# ---------------------------------------------------------------------------
Profile = Literal["local", "umhpc"]

_ACTIVE_PROFILE: str = os.environ.get("UPRMVS_PROFILE", "local")


def set_profile(profile: str | None) -> str:
    """Select the active profile. ``None`` keeps the current one (env default)."""
    global _ACTIVE_PROFILE
    if profile is not None:
        if profile not in ("local", "umhpc"):
            raise ValueError(f"Unknown profile: {profile!r} (expected 'local' or 'umhpc')")
        _ACTIVE_PROFILE = profile
    return _ACTIVE_PROFILE


def get_profile() -> str:
    return _ACTIVE_PROFILE


def _paths_for(profile: str) -> dict[str, Path]:
    if profile == "umhpc":
        project_path = Path("/scr/user/qinglong/projects/upr-mvs01")
        data_path = Path("/scr/user/qinglong/dataset")
        eval_tool = Path("/scr/user/qinglong/tools/Fast-DTU-Evaluation")
    else:  # local
        project_path = Path("/home/william/project/uprmvs01")
        data_path = Path("/home/william/project/dataset")
        eval_tool = Path("/home/william/Downloads/Fast-DTU-Evaluation")

    return {
        "project_path": project_path,
        "output_root": project_path / "uprmvs_outputs",
        "dtu_train_root": data_path / "DTU/dtu_training",
        "dtu_test_root": data_path / "DTU/dtu_testing",
        "dtu_list_path": project_path / "lists/dtu/train.txt",
        "sfm_cache_path": project_path / "log/sfm_depth",
        "resnet50_weights_file": data_path / "Resnet50/Model_v2.pth",
        "dinov3_weights_file":
            data_path / "DINOv3/pre_trained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "da3_weights_file": data_path / "DA3/pretrained/DA3MONO-LARGE/",
        "vggt_weights_path": data_path / "VGGT/pretrained/VGGT-1B",
        "train_list_file": project_path / "lists/dtu/train.txt",
        "val_list_file": project_path / "lists/dtu/val.txt",
        "test_list_file": project_path / "lists/dtu/test.txt",
        "eval_tool": eval_tool,
        "eval_gt": data_path / "DTU/SampleSet/MVS Data",
    }


def _path_field(key: str):
    """A frozen-dataclass default that resolves against the ACTIVE profile at
    instance-construction time (not import time)."""
    return field(default_factory=lambda: _paths_for(_ACTIVE_PROFILE)[key])


@dataclass(frozen=True)
class ProjectPaths:
    project_path: Path = _path_field("project_path")
    dtu_train_root: Path = _path_field("dtu_train_root")
    dtu_test_root: Path = _path_field("dtu_test_root")
    dtu_list_path: Path = _path_field("dtu_list_path")
    sfm_cache_path: Path = _path_field("sfm_cache_path")
    resnet50_weights_file: Path = _path_field("resnet50_weights_file")
    dinov3_weights_file: Path = _path_field("dinov3_weights_file")
    da3_weights_file: Path = _path_field("da3_weights_file")
    vggt_weights_path: Path = _path_field("vggt_weights_path")
    train_list_file: Path = _path_field("train_list_file")
    val_list_file: Path = _path_field("val_list_file")
    test_list_file: Path = _path_field("test_list_file")
    output_root: Path = _path_field("output_root")
    eval_tool: Path = _path_field("eval_tool")
    eval_gt: Path = _path_field("eval_gt")


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
class PriorConfig:
    """VGGT/DA3 depth-prior generation.

    ``target_w`` / ``target_h`` is the resolution VGGT + DA3 actually run at, and
    therefore the prior's *true* resolution before ``inverse_transform_map``
    resamples it up to the working image size. Raising it makes the depth prior
    genuinely sharper (instead of an upsampled 518x420) at the cost of VGGT/DA3
    compute+memory (attention is ~O(tokens^2), tokens = (w/14)*(h/14)).

    Both dims MUST be multiples of the backbone patch size (14); the DPT head
    reassembles on ``H//14`` patches and a non-multiple truncates / misaligns.
    Defaults 518=37*14, 420=30*14.
    """
    target_w: int = 518
    target_h: int = 420

    @property
    def target_wh(self) -> tuple[int, int]:
        return (self.target_w, self.target_h)


@dataclass(frozen=True)
class FPNConfig:
    out_channels: int = 128
    base_channel: int = 32



@dataclass(frozen=True)
class DepthRangeConfig:

    num_global: int = 48
    num_local: int = 16
    global_quantile_lo: float = 0.002
    global_quantile_hi: float = 0.998
    global_margin_ratio: float = 0.12
    global_min_span_frac: float = 1.0
    inverse_depth_global: bool = True
    spike_k: float = 4.0
    spike_min_mad_rel: float = 0.002  # MAD floor as a fraction of local depth

    local_half_min_gi: float = 0.75
    local_half_max_gi: float = 2.0

    mode_window: int = 2

    range_k: float = 3.0
    range_entropy_a: float = 1.0
    range_edge_b: float = 1.0
    range_min_gi: float = 1.0
    range_max_gi: float = 8.0
    edge_grad_rel: float = 0.03
    sigma_max_ratio: float = 0.15
    k_sigma: float = 3.0


@dataclass(frozen=True)
class CostVolumeConfig:
    num_groups: int = 8
    num_depths_stage1: int = 64
    num_depths_stage2: int = 24
    num_depths_stage3: int = 16
    stage1_meta_channels: int = 6
    warp_channels_stage1: int = 128
    warp_channels_stage2: int = 48
    warp_channels_stage3: int = 16
    warp_use_half: bool = True
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
    w_ce: float = 1.0         # unified 64-candidate soft-label CE (stage1) / per-stage CE (2,3)
    w_reg: float = 1.0        # interval-normalized Huber on the regressed depth, ALL valid pixels
    w_global_aux: float = 0.5
    w_local_aux: float = 0.25
    edge_reg_boost: float = 2.0
    use_cross_entropy: bool = True


@dataclass(frozen=True)
class StageWeights:
    stage1: float = 0.5
    stage2: float = 1.0
    stage3: float = 2.0


@dataclass(frozen=True)
class TrainConfig:
    profile: str = field(default_factory=get_profile)
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
    vis_interval: int = 100
    vis_max_views: int = 5
    val_interval: int = 2000
    ckpt_interval: int = 5000
    devices: tuple[int, ...] = (0,)
    distributed: bool = False
    use_anchor_pe: bool = True
    use_geo_fusion: bool = True
    use_points_alignment: bool = True
    prior_corruption_prob: float = 0.4


def _train_local() -> TrainConfig:
    return TrainConfig(
        profile="local",
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
        log_interval=20,
        vis_interval=100,
        vis_max_views=5,
        val_interval=500,
        ckpt_interval=1000,
        devices=(0,),
        distributed=False,
    )


def _train_umhpc() -> TrainConfig:
    return TrainConfig(
        profile="umhpc",
        batch_size=3,
        num_workers=8,
        num_views=5,
        lr=2.0e-4,
        weight_decay=1.0e-4,
        max_steps=70000,
        warmup_steps=1000,
        grad_clip=1.0,
        amp=True,
        seed=20260526,
        log_interval=50,
        vis_interval=100,
        vis_max_views=5,
        val_interval=2000,
        ckpt_interval=5000,
        devices=(0, 1, 2, 3),
        distributed=True,
    )


def get_train_config(profile: str | None = None) -> TrainConfig:
    profile = profile or get_profile()
    if profile == "local":
        return _train_local()
    if profile == "umhpc":
        return _train_umhpc()
    raise ValueError(f"Unknown train profile: {profile!r}")


@dataclass(frozen=True)
class MVSConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    data: DataConfig = field(default_factory=DataConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    depth_range: DepthRangeConfig = field(default_factory=DepthRangeConfig)
    cost_volume: CostVolumeConfig = field(default_factory=CostVolumeConfig)
    points_alignment: PointsAlignmentConfig = field(default_factory=PointsAlignmentConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    stage_weights: StageWeights = field(default_factory=StageWeights)
    train: TrainConfig = field(default_factory=lambda: get_train_config(None))


def build_mvs_config(profile: str | None = None) -> MVSConfig:
    # Select the active profile FIRST so that ProjectPaths (and every bare
    # ProjectPaths() elsewhere) resolve to this profile's paths, not just the
    # train hyper-params.
    set_profile(profile)
    return MVSConfig()
