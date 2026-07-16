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
    # ---- stage-1 dual-branch hypotheses: global guard + local prior refinement ----
    # The global branch must stay independent of prior confidence: it is the
    # safety net that catches confidently-wrong priors, so nothing the prior
    # reports about itself may shrink it.
    num_global: int = 48
    num_local: int = 16
    # Per-image global bounds: robust quantiles of the valid prior depths with a
    # margin, clamped to the physical [depth_min, depth_max] and never narrower
    # than global_min_span_frac of the physical span (guards against a globally
    # shifted prior making its own quantiles lie).
    # Gate-tuned 2026-07-16: the cached priors under-cover far backgrounds near
    # edges (edge-bucket coverage 0.981 at q0.5%/5% margin, still 0.987 at
    # q0.2%/12%), so the guard cannot trust prior quantiles on DTU at all:
    # min_span_frac=1.0 forces the full physical [depth_min, depth_max] range,
    # making coverage exact by construction. Quantile tightening stays in the
    # code for datasets whose physical bounds are loose (tune there, re-gate).
    global_quantile_lo: float = 0.002
    global_quantile_hi: float = 0.998
    global_margin_ratio: float = 0.12
    global_min_span_frac: float = 1.0
    # Sample the global branch uniformly in inverse depth (≈ uniform pixel
    # displacement); False = uniform in depth.
    inverse_depth_global: bool = True
    # Local-branch spike rejection: |prior - median_3x3| > spike_k * MAD flags a
    # fly-point; its center falls back to the neighborhood robust median.
    spike_k: float = 4.0
    spike_min_mad_rel: float = 0.002  # MAD floor as a fraction of local depth
    # Local half-width in units of the per-image mean global bin interval:
    # floor keeps a confidently-wrong prior from locking the search; ceiling
    # keeps the dense branch dense (the guard covers the rest).
    local_half_min_gi: float = 0.75
    local_half_max_gi: float = 2.0
    # ---- stage-1 posterior -> depth / next-stage range ----
    # Mode-centered regression window (bins to each side of the argmax); a
    # global soft-argmin over a bimodal posterior lands between the modes.
    mode_window: int = 2
    # Next-stage half-range = range_k * (winning bin's interval)
    #                         * (1 + range_entropy_a*H_norm + range_edge_b*E),
    # clipped to [range_min_gi, range_max_gi] global intervals. A local winner
    # shrinks the search; a global winner keeps a correction-sized range.
    range_k: float = 3.0
    range_entropy_a: float = 1.0
    range_edge_b: float = 1.0
    range_min_gi: float = 1.0
    range_max_gi: float = 8.0
    # Rule-based edge/unreliable map from the (possibly corrupted) prior:
    # relative depth-gradient threshold at which E saturates to 1.
    edge_grad_rel: float = 0.03
    # legacy sigma-based width params (still used by the local branch scaling)
    sigma_max_ratio: float = 0.15
    k_sigma: float = 3.0


@dataclass(frozen=True)
class CostVolumeConfig:
    num_groups: int = 8
    # stage1 = depth_range.num_global + num_local (asserted at model build)
    num_depths_stage1: int = 64
    # stage2 must span >= ~2 global intervals after a global-branch win, so it
    # needs enough bins to keep sub-interval resolution at that width.
    num_depths_stage2: int = 24
    num_depths_stage3: int = 16
    # number of hypothesis metadata channels appended to the stage-1 cost
    # volume (normalized depth/interval/is_local/dist-to-prior/conf/edge) so
    # the 3D regularizer can see the irregular depth axis and branch identity.
    stage1_meta_channels: int = 6
    # warp-channel width per cascade stage (must be divisible by num_groups).
    # Shrinks as resolution grows so the full-res stage does not OOM: the warp
    # intermediate is [B, warp_channels, D, H, W].
    warp_channels_stage1: int = 128
    warp_channels_stage2: int = 48
    warp_channels_stage3: int = 16
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
    w_ce: float = 1.0         # unified 64-candidate soft-label CE (stage1) / per-stage CE (2,3)
    w_reg: float = 1.0        # interval-normalized Huber on the regressed depth, ALL valid pixels
    # Stage-1 auxiliary heads: the global branch is supervised on every valid
    # pixel it covers so it never loses the ability to localize GT on its own,
    # even in the (frequent) regime where the local branch wins the 64-way
    # softmax; the local branch is supervised only where GT falls inside it.
    w_global_aux: float = 0.5
    w_local_aux: float = 0.25
    # Regression weight multiplier inside the edge band (E ~ 1): edge pixels
    # are ~5-10% of the image and get drowned out at uniform weighting.
    edge_reg_boost: float = 2.0
    use_cross_entropy: bool = True


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
    # Fraction of training samples whose prior gets synthetic failure modes
    # (edge ramps, fly-points, drift, wrong-high-confidence, ...). Without this
    # the prior is right most of the time, the network learns the local-branch
    # shortcut, and the global branch's rescue path is never trained.
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
        # Cosine horizon must match what the SLURM walltime can actually deliver
        # (~2.9 s/step -> ~70k steps in <60h of the 3-day limit incl. validation),
        # otherwise the LR never anneals before the job is killed.
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
