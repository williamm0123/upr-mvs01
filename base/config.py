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
            # Every split — train, val AND test — is served from dtu_train_root:
            # data/dtu.py reads Rectified_raw/ (1200x1600 PNG), Cameras/ and
            # Depths_raw/ from it. For the 22 test scans that source is the
            # official eval data: Cameras/ and pair.txt are byte-identical to
            # dtu_testing/scan*/, and Rectified_raw/scan*/rect_*_3_r5000.png is
            # the lossless original of dtu_testing/scan*/images/*.jpg (test modes
            # pin light_idx=3). Reading it here additionally gives GT depth, so
            # the depth metrics work on the test split too.
            "dtu_train_root": data_path / "DTU/dtu_training",
            # The MVSNet-format eval set (scan*/{images,cams}/, pair.txt, no GT
            # depth). NOT currently read by any code path — see above.
            "dtu_test_root": data_path / "DTU/dtu_testing",
            "dtu_list_path": project_path / "lists/dtu/train.txt",
            "sfm_cache_path":project_path / "log/sfm_depth",
            # VGGT/DA3 depth+conf priors, one npz per (scan, ref_view, light).
            # The filename encodes none of the resolutions it was built at, so a
            # changed --prior-resize-scale / --prior-target-* needs a force rebuild.
            "prior_cache_path": project_path / "log/prior_cache",
            # per-view depth/conf/K/E/image that fusion consumes, plus metrics.json,
            # under a <split>/ subdir. Kept after the run on purpose: re-fusing at
            # different --photo-thresh/--geo-* costs nothing while re-running
            # inference does.
            "depth_cache_path": project_path / "log/depth_cache",
            # fused point clouds from test.py, named mvsnet{scan:03d}_l3.ply —
            # feed this directory straight to Fast-DTU-Evaluation --pred_dir.
            "pred_points_path": project_path / "log/pred_points",
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
    prior_cache_path: Path = _DEFAULT_PATHS["prior_cache_path"]
    depth_cache_path: Path = _DEFAULT_PATHS["depth_cache_path"]
    pred_points_path: Path = _DEFAULT_PATHS["pred_points_path"]
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

    # 32/16 -> 40/8: 共位测试 (experiments/out/coloc_val_*.log) 显示 stage1 尾巴上
    # global 分支 100% 含 <20mm 候选、81% 含 <8mm, 而 local 分支只有 47.8%/37.9%。
    # 把预算从命中率低的分支挪走。总数仍须 == num_depths_stage1。
    num_global: int = 40
    num_local: int = 8
    global_quantile_lo: float = 0.002
    global_quantile_hi: float = 0.998
    global_margin_ratio: float = 0.12
    global_min_span_frac: float = 1.0
    # --- local 分支门控 (P2) ---
    gate_local_branch: bool = True   # conf < gate_hard_conf -> 整条 local 分支退化为 guard 网格
    gate_hard_conf: float = 0.25
    branch_prior: bool = True        # stage1 logits 加 log q / log(1-q)
    branch_q_min: float = 0.02       # q 的下限, 防 log(0)
    # --- stage2 双模态 (P4) ---
    # 默认关闭。当前实现把 winner-centered 轴的"低深度一半"当成 winner 侧
    # (winner 在轴中心, 所以上半边全丢), 而且拼出来的是非均匀轴, 而 stage2 的
    # 3D UNet / interval loss / 下一级 range 都假设均匀。正式实现应该建两个
    # 独立的均匀 8-bin cost volume 共享 regularizer, 而不是拼一条轴。
    # 启用前先统计最终尾巴上正确候选的 posterior rank —— 只有它通常是 rank 2
    # 时 top-2 才真的有用。
    dual_mode_stage2: bool = False
    second_mode_guard: float = 2.0   # 屏蔽 winner ± guard 个 bin 后再找次峰
    # 注意: second_mode 返回的是 winner 邻域*之外*的总质量, 不是次峰本身的质量,
    # 平坦分布也容易超过这个阈值。真正实现时应改成次峰邻域质量。
    dual_mode_min_mass: float = 0.10
    inverse_depth_global: bool = True
    spike_k: float = 4.0
    spike_min_mad_rel: float = 0.002  # MAD floor as a fraction of local depth

    local_half_min_gi: float = 0.75
    local_half_max_gi: float = 2.0

    mode_window: int = 2

    # Next-stage window: half = range_k[k] * winner_interval * (1 + a*entropy +
    # b*edge), clamped to [range_min_gi[k], range_max_gi] * gi.
    #
    # These MUST be per-stage. The resulting bin is 2*half/(D-1), so with a
    # single range_k the shrinking hypothesis counts (16/8/4) make every stage
    # COARSER than its parent: at range_k=3.0 and a=b=1.0 the ratios are 1.20x /
    # 2.57x / 6.00x. MVSFormer++'s (2.67, 1.5, 1.0) against the same 16/8/4 are
    # tuned for exactly this constraint (0.356x / 0.429x / 0.667x); the values
    # below are the same idea with room left for the entropy/edge widening, so
    # bin/wi stays under 1 even with both terms saturated:
    #     stage2 0.200-0.400x   stage3 0.257-0.514x   stage4 0.400-0.800x
    range_k: tuple[float, float, float] = (1.5, 0.9, 0.6)
    range_entropy_a: float = 0.5
    range_edge_b: float = 0.5
    # Floor = recovery room, i.e. permission to grow past what the posterior
    # suggests. That belongs at the coarse end; by stage 4 there is no later
    # stage to recover into and only precision matters. stage2's 0.66*gi keeps
    # the old absolute 10.83mm floor now that gi grew with num_global 48 -> 32.
    range_min_gi: tuple[float, float, float] = (0.66, 0.20, 0.05)
    range_max_gi: float = 8.0
    edge_grad_rel: float = 0.03
    sigma_max_ratio: float = 0.15
    k_sigma: float = 3.0


@dataclass(frozen=True)
class CostVolumeConfig:
    """Four cascade stages at strides 8 / 4 / 2 / 1.

    Hypothesis counts (32+16) - 16 - 8 - 4 follow MVSFormer++'s coarse-to-fine
    budget: nearly all candidates live at the cheapest resolution, and the
    full-res stage keeps only four. A fine stage that still needs many planes is
    evidence the previous stage failed to converge its range, not a reason to
    add planes there — and planes cost 64x more at stride 1 than at stride 8.

    Total cost-volume voxels at 512x640 drop 2.8x versus the old 3-stage
    (64-24-16 at strides 4/2/1) layout, and 4x at the full-res stage that
    dominated memory.
    """

    num_groups: int = 8
    num_depths_stage1: int = 48   # = depth_range.num_global + num_local
    num_depths_stage2: int = 16
    num_depths_stage3: int = 8
    num_depths_stage4: int = 4
    stage1_meta_channels: int = 6
    # Warp width shrinks as resolution grows; stage 1 moved to stride 8 so it
    # can afford to stay wide.
    warp_channels_stage1: int = 128
    warp_channels_stage2: int = 64
    warp_channels_stage3: int = 32
    warp_channels_stage4: int = 16
    warp_use_half: bool = True
    # 保持关闭。缓存里的 src_weights 长度 = 建缓存时的 src 视角数, 与当前
    # nviews 不一致 (实测缓存 S=2, 现在需要 S=4), 而且那批权重本身就是按旧
    # 视角集算的, 形状对得上也是错的。逐像素可见性头 (下一行) 完全取代它,
    # 且不依赖任何缓存。要重新启用必须先重建 prior cache。
    use_src_weights: bool = False
    # per-(source, pixel) 可见性权重: stage1 尾巴里 58.9% 是 global 赢了但选错
    # 平面, 而聚合原本是所有 source 等权平均, 表达不了局部遮挡。
    visibility_weighting: bool = True


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
    # off / all_view / ref_only —— 与 SPRE reliability 解耦的独立开关
    mode: str = "all_view"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    patch_size: int = 16
    layers: tuple[int, ...] = (3, 7, 11)
    max_side: int = 512
    # Inject the SVA-fused per-view DINO features into the FPN bottleneck (the
    # 1/8 level), the way MVSFormer++ does ``conv31 = conv31 + vit_feat``. Only
    # takes effect when SPRE is enabled, since that is what loads the backbone.
    feed_fpn: bool = True


@dataclass(frozen=True)
class SPREConfig:
    """Semantic Prior Reliability Estimator (learned, DINOv3-witnessed conf).


    """
    # cached / edge / spre —— prior 可靠度从哪来, 独立于 DINO matching
    reliability_source: str = "spre"
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
    # 分支校准: 监督"哪个分支的 oracle 候选更接近 GT", 而不是 GT 是否在
    # local 跨度内。见 losses/composite.py 的推导。
    w_branch: float = 0.5
    branch_tau_gi: float = 0.5    # 软标签温度, 单位是 global_interval
    branch_margin_gi: float = 0.5 # 两分支 oracle 误差差距小于它就不监督
    # corruption BCE 按 clean/corrupt 两类平衡平均, 而不是按全体像素平均
    spre_balance_corrupt: bool = True
    w_spre: float = 0.5           # corruption-BCE weight (corrupted prior -> 0, clean -> 1)
    w_spre_soft: float = 0.5      # prior-error soft-target weight
    spre_soft_tau_mm: float = 10.8 # exp(-(|prior-gt|/tau)^2) scale, in the depth unit (mm)


@dataclass(frozen=True)
class StageWeights:
    # 0.5 -> 1.5: 100% 的最终尾巴都源自 stage1 的选择失败, 而候选放置是
    # detach 的 (depth_range.py:328), 后续 stage 的 loss 传不回来纠正它。
    stage1: float = 1.5
    stage2: float = 1.0
    stage3: float = 1.5
    stage4: float = 2.0


@dataclass(frozen=True)
class AugmentConfig:
    """Training-time augmentation (train split only; val/test stay deterministic).


    ``resize_range`` is the random over-resize before cropping — at 1.0 the crop
    fills the frame and cannot move, above it the crop has somewhere to land.
    """

    photometric: bool = True
    brightness: float = 0.2
    contrast: float = 0.1
    saturation: float = 0.1
    hue: float = 0.05
    min_gamma: float = 0.9
    max_gamma: float = 1.1

    multi_scale: bool = True
    scales: tuple[tuple[int, int], ...] = (
        (448, 576), (448, 640), (512, 640), (512, 704), (512, 768),
        (576, 704), (576, 768), (576, 832), (640, 832), (640, 896),
    )
    resize_range: tuple[float, float] = (1.0, 1.2)


@dataclass(frozen=True)
class TrainConfig:
    profile: str = TRAIN_PROFILE
    batch_size: int = 1
    num_workers: int = 2
    num_views: int = 3
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    max_steps: int = 30000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    amp: bool = True
    # 出现第一个非有限 loss/梯度就终止, 而不是让 GradScaler 一路跳步空转。
    # 关掉只该用于复现历史事故 —— 正常训练没有理由带着 nan 继续跑。
    nan_watchdog: bool = True
    # 连续多少步梯度非有限才判定为真发散。1 步不算: AMP 的 GradScaler 本来就
    # 靠"故意溢出 -> 跳过这步 -> 把 scale 减半"来找可用的缩放系数, 而
    # scaler.unscale_() 之后 inf/scale 仍然是 inf —— 见 train.py 关口 2。
    # 10 而不是 3: GradScaler 从初始 scale 2^16 起步, 每次溢出减半, 要降到
    # 2^6 就得连续溢出 10 次 —— 这段热身全程梯度都是 inf, 却完全正常。
    # 真发散的特征是 scale 已经被压到很低却仍然救不回来。
    nan_grad_patience: int = 10
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
        lr=2.0e-4,
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
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=lambda: get_train_config(None))


def build_mvs_config(profile: str | None = None) -> MVSConfig:
    cfg = MVSConfig()
    if profile is not None and profile != cfg.train.profile:
        cfg = MVSConfig(train=get_train_config(profile))
    return cfg


def resolve_split(list_file, name: str, use_clean: bool = True) -> tuple[str, str | None]:
    """把 ``lists/dtu/<name>.txt`` 解析成 (实际列表, 排除表)。

    审计 (scripts/audit_prior_cache.py) 发现 8.05% 的 prior 缓存未完成标尺校准。
    重灾区整 scan 由 ``<name>_clean.txt`` 剔除, 幸存 scan 里的零散坏样本由
    ``exclude_<name>.csv`` 逐 (scan, view, light) 剔除。两个文件都是可选的,
    不存在就原样返回。

    ``cfg.paths.*_list_file`` 故意保持指向原始列表 —— 切换发生在这里, 所以
    "跑的是哪个列表" 是一个可以在日志里看到的运行期决定, 而不是藏在配置里。
    所有入口 (train / test / 诊断脚本) 都该走这个函数, 否则训练和评测会悄悄
    跑在不同的 scan 集合上。
    """
    p = Path(list_file)
    if not use_clean:
        return str(p), None
    clean = p.with_name(f"{name}_clean.txt")
    excl = p.with_name(f"exclude_{name}.csv")
    lf = str(clean) if clean.exists() else str(p)
    ef = str(excl) if excl.exists() else None
    if lf != str(p) or ef:
        print(f"[data] {name}: list={Path(lf).name} exclude={Path(ef).name if ef else '-'}")
    return lf, ef
