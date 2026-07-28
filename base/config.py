from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[1]

def _detect_machine() -> str:
    """UPRMVS_MACHINE wins; otherwise infer from where this checkout lives.

    Only scripts/train_*.sh export the variable, so anything run bare (test.py,
    eval_ablation.py) used to fall back to "ubuntu" and resolve the laptop's
    paths while sitting on the cluster — a confusing FileNotFoundError on
    lists/dtu/*.txt rather than an obvious misconfiguration.
    """
    env = os.environ.get("UPRMVS_MACHINE")
    if env:
        return env
    return "umhpc" if str(REPO_ROOT).startswith("/scr/") else "ubuntu"


MACHINE: Literal["ubuntu", "umhpc"] = _detect_machine()  # type: ignore[assignment]

TRAIN_PROFILE: Literal["local", "umhpc"] = os.environ.get(  # type: ignore[assignment]
    "UPRMVS_PROFILE", "umhpc" if MACHINE == "umhpc" else "local"
)


def _default_paths() -> dict[str, Path]:
    # project_path is always this checkout: lists/, log/, caches and outputs are
    # inside the repo, so deriving it from REPO_ROOT keeps them correct even when
    # MACHINE is wrong. (Same value as the old hard-coded paths on both boxes.)
    project_path = REPO_ROOT
    if MACHINE == "umhpc":
        data_path = Path("/scr/user/qinglong/dataset")
    else:
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
class DINOConfig:
    """Frozen DINOv3 ViT-B/16 backbone used as the SPRE 'independent witness'.

    ``layers`` picks the intermediate blocks whose patch tokens SPRE consumes
    (shallow=more spatial, deep=more semantic — the DPT recipe). ``max_side``
    is the patch-aligned resize for the ref image before the ViT.
    """
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    patch_size: int = 16
    layers: tuple[int, ...] = (4, 7, 11)
    max_side: int = 512


@dataclass(frozen=True)
class SPREConfig:
    """Semantic Prior Reliability Estimator (learned, DINOv3-witnessed conf).

    When ``enabled`` the network predicts a per-pixel prior reliability r in
    [0, 1] from frozen DINOv3 semantic features + online statistics of the
    (possibly corrupted) depth prior, and r REPLACES the cached ``conf_prior``
    fed to ``build_stage1_hypotheses``. Supervised by the corruption mask and
    the prior's true error (see ``losses/composite.py``). ``enabled=False``
    restores the exact previous behaviour (cached conf, no DINOv3 loaded).
    """
    enabled: bool = False
    proj_dim: int = 64            # fused tokens -> proj_dim, concatenated with the 4 stats
    hidden: int = 64
    # Cross-ViT fusion of the DINOv3 layers, ported from MVSFormer++'s
    # CrossVITDecoder. ``attn_dim`` is the width the blocks run at (768 -> 384
    # first): SPRE emits one scalar per pixel, so full ViT width is 4x the
    # parameters for no extra capacity where it matters.
    attn_dim: int = 384
    num_heads: int = 6            # 64 dims/head, the same ratio MVSFormer++ uses
    aas_init: float = 0.5         # their ``prev_values``: scale on the previous stage
    cross_view: bool = True       # source views as cross-attention keys/values


@dataclass(frozen=True)
class LossConfig:
    w_ce: float = 1.0         # unified 64-candidate soft-label CE (stage1) / per-stage CE (2,3)
    w_reg: float = 1.0        # interval-normalized Huber on the regressed depth, ALL valid pixels
    w_global_aux: float = 0.5
    w_local_aux: float = 0.25
    edge_reg_boost: float = 2.0
    use_cross_entropy: bool = True
    # SPRE supervision (only active when the network emits a 'spre' output)
    w_spre: float = 0.5           # corruption-BCE weight (corrupted prior -> 0, clean -> 1)
    w_spre_soft: float = 0.5      # prior-error soft-target weight
    spre_soft_tau_mm: float = 10.8 # exp(-(|prior-gt|/tau)^2) scale, in the depth unit (mm)


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
        batch_size=2,
        num_workers=8,
        num_views=5,
        lr=2.0e-4,
        weight_decay=1.0e-4,
        max_steps=20000,
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
        batch_size=2,
        num_workers=8,
        num_views=5,
        lr=3.0e-4,
        weight_decay=1.0e-4,
        max_steps=30000,
        warmup_steps=1000,
        grad_clip=1.0,
        amp=True,
        seed=20260526,
        log_interval=50,
        vis_interval=100,
        vis_max_views=5,
        # 500 not 2000: at 2000 a 3k-step probe yields a single val point, which
        # cannot show a trend. Overridable with --val-interval.
        val_interval=500,
        ckpt_interval=2000,
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
    prior: PriorConfig = field(default_factory=PriorConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    depth_range: DepthRangeConfig = field(default_factory=DepthRangeConfig)
    cost_volume: CostVolumeConfig = field(default_factory=CostVolumeConfig)
    points_alignment: PointsAlignmentConfig = field(default_factory=PointsAlignmentConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    spre: SPREConfig = field(default_factory=SPREConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    stage_weights: StageWeights = field(default_factory=StageWeights)
    train: TrainConfig = field(default_factory=lambda: get_train_config(None))


def build_mvs_config(profile: str | None = None) -> MVSConfig:
    cfg = MVSConfig()
    if profile is not None and profile != cfg.train.profile:
        cfg = MVSConfig(train=get_train_config(profile))
    return cfg
