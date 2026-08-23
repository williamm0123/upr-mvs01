"""Inference / point-cloud driver for UprMVSNet.

Runs the checkpoint over a DTU split and does two things:

1. Depth metrics: masked depth-map errors (same masking as train.py
   validation) with a per-scan breakdown plus median/p90, written to
   ``<out>/metrics.json``. Free — the same forward pass fusion needs.
2. Fusion (default, ``--no-fuse`` to skip): caches per-view depth/conf and
   fuses each scan into a point cloud (photometric + geometric consistency
   filtering), written as ``cfg.paths.pred_points_path/mvsnet{scan:03d}_l3.ply``
   (i.e. ``<project>/log/pred_points``) — the naming and layout
   Fast-DTU-Evaluation expects.

Scoring stops here: point the standalone Fast-DTU-Evaluation at that
directory. ``run_fast_eval`` below still implements the subprocess call and
``--run-eval`` still parses, but main() no longer invokes it — running the
benchmark separately keeps a long fusion job from being held hostage by the
scorer, and lets the scorer be re-run at different thresholds for free.

Priors: the network never computes them inline — it reads a disk cache, which
this script fills first (VGGT + DA3 loaded once, then freed before the MVS
model is built). ``--prior-resize-scale`` decouples the cache resolution from
``--resize-scale``; build once at 1.0 and every inference resolution is served
by a downsample. The cache filename encodes only (scan, ref_view, light), NOT
the resize or ``--prior-target-*``, so changing either needs ``--build-priors
force`` — otherwise the stale cache is silently reused.

Examples
--------
python test.py --split test --build-priors force --priors-only   # phase 1
python test.py --split test --build-priors skip                  # phase 2 -> ply
python test.py --split val --no-fuse                             # depth metrics only
python test.py --split test --max-refs 5 --no-fuse               # quick smoke check
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base.config import ProjectPaths, build_mvs_config, resolve_split
from data.dtu import DTUMVSDataset
from models.network import UprMVSNet
from utils.geometry import unproject_depth


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("UprMVSNet test / DTU evaluation")
    p.add_argument("--profile", choices=["local", "umhpc"], default="umhpc")
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt", default=None, help="explicit checkpoint file; overrides --ckpt-dir")
    p.add_argument("--ckpt-dir", default="log/model_eval",
                   help="dir to load best.pth (else latest.pth) from; default log/model_eval — copy a "
                        "snapshot here so eval never reads the live-updating log/model during training. "
                        "Falls back to log/model if this dir is absent. Relative paths resolve under the project root.")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--list", default=None, help="override the split's scan list file")
    p.add_argument("--num-views", type=int, default=None, help="views fed to the network (default cfg.train.num_views)")
    p.add_argument("--resize-scale", type=float, default=0.5)
    p.add_argument("--full-image", action="store_true",
                   help="reconstruct the whole image (no center crop). Sets the crop window to the "
                        "full resized DTU frame (1200x1600 * resize_scale) so no pixels are dropped.")
    p.add_argument("--prior-resize-scale", type=float, default=1.0,
                   help="resize scale the prior cache is BUILT at, independent of --resize-scale "
                        "(default 1.0 = native 1200x1600). Priors are stored at this resolution and "
                        "resampled to the inference resolution on load, so building at 1.0 once serves "
                        "every inference resolution — downsampling loses nothing, upsampling does. It "
                        "also runs the SfM metric-scale calibration on the sharpest images. Only "
                        "affects runs that actually build (--build-priors auto/force).")
    p.add_argument("--prior-target-w", type=int, default=None,
                   help="VGGT/DA3 prior width (default cfg 518; must be a multiple of 14). Raises the "
                        "true depth-prior resolution. Needs --build-priors force to take effect (else "
                        "the existing cache is reused). VGGT cost/memory grows ~O((w*h/196)^2).")
    p.add_argument("--prior-target-h", type=int, default=None,
                   help="VGGT/DA3 prior height (default cfg 420; must be a multiple of 14)")
    p.add_argument("--scans", type=int, nargs="+", default=None, metavar="ID",
                   help="run only these scan ids (e.g. --scans 1 4 9). Lets N jobs split the split "
                        "across N GPUs: the ply filenames are per-scan so they can share --ply-dir, "
                        "but give each job its own --out (the per-view npz cache and metrics.json).")
    p.add_argument("--max-scans", type=int, default=0)
    p.add_argument("--max-refs", type=int, default=0, help="limit ref views per scan (0 = all 49)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--spre", choices=["auto", "on", "off"], default="auto",
                   help="SPRE DINOv3 prior-reliability head: 'auto' (default) mirrors whatever the "
                        "checkpoint was trained with; 'on'/'off' force it and load non-strictly.")
    p.add_argument("--out", default=None,
                   help="per-view npz cache + metrics.json root (default "
                        "cfg.paths.depth_cache_path/<split>, i.e. <project>/log/depth_cache/<split>). "
                        "NOT where the ply goes — that is --ply-dir. Relative paths resolve under "
                        "the project root.")
    p.add_argument("--vis", type=int, default=0, help="save the first N depth visualizations per scan")
    p.add_argument("--build-priors", choices=["auto", "skip", "force"], default="auto",
                   help="auto = fill in missing priors; force = rebuild all (needed after changing "
                        "--prior-resize-scale / --prior-target-*, since the cache filename encodes "
                        "none of them); skip = require a complete cache")
    p.add_argument("--priors-only", action="store_true",
                   help="exit after the prior phase, without loading the MVS model. Phase 1 of a "
                        "two-process run: VGGT/DA3 are then fully gone before inference starts, and "
                        "an interrupted build resumes with --build-priors auto instead of redoing it.")
    # fusion
    p.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True,
                   help="save per-view outputs and fuse point clouds (default on; --no-fuse "
                        "for depth metrics only)")
    p.add_argument("--ply-dir", default=None,
                   help="where the fused clouds go (default cfg.paths.pred_points_path, i.e. "
                        "<project>/log/pred_points). Relative paths resolve under the project root.")
    p.add_argument("--photo-thresh", type=float, default=0.3,
                   help="cascade mode-probability threshold (was inert when confidence "
                        "came from stage 4 alone — see cascade_confidence)")
    p.add_argument("--no-clean-lists", action="store_true",
                   help="不使用 audit 产出的 *_clean.txt / exclude_*.csv")
    p.add_argument("--conf-window", type=int, default=1,
                   help="+-bins around each stage's argmax for the fusion confidence")
    p.add_argument("--conf-mode", choices=["product", "geomean", "last"], default="product",
                   help="how to combine per-stage mode mass; 'last' reproduces the inert "
                        "pre-fix behaviour for A/B only")
    p.add_argument("--conf-source", choices=["auto", "cascade", "learned"], default="auto",
                   help="W3-C: learned = 用训练出来的融合置信度头 (温度标定过, "
                        "--photo-thresh 才有物理含义); cascade = 旧的手工概率乘积; "
                        "auto = checkpoint 里有头就用头, 没有就退回 cascade")
    p.add_argument("--fuse-only", action="store_true",
                   help="跳过推理, 直接拿 --out 下已缓存的逐视角深度重新融合。换融合方法或"
                        "调阈值时用它 —— 重跑推理是 3 个 arm x 1078 样本的浪费。")
    p.add_argument("--fusion", choices=["geo", "dedup", "gipuma"], default="dedup",
                   help="默认 dedup。geo = 几何+光度一致性, 每个 ref 视角各输出一遍存活像素, "
                        "**跨视角不去重** —— 49 视角下一个 scan 出 2500-4500 万点; "
                        "dedup = geo 再加 fusibile 式的跨视角消费标记, 同一个表面点只由"
                        "最先认领它的 ref 视角输出一次 (纯 torch, 无外部依赖); "
                        "gipuma = 调 fusibile 二进制 (需 CUDA<12 编译, 见 scripts/build_fusibile.sh)。")
    p.add_argument("--fusibile-exe", default="third_party/fusibile/fusibile",
                   help="fusibile 可执行文件 (相对路径按项目根解析)")
    p.add_argument("--gipuma-disp-thresh", type=float, default=0.25,
                   help="fusibile 的视差一致性阈值 (MVSFormer++ 用 0.25)")
    p.add_argument("--gipuma-num-consistent", type=int, default=3,
                   help="fusibile 要求的一致视角数 (MVSFormer++ 用 3)")
    p.add_argument("--keep-gipuma-tmp", action="store_true",
                   help="保留导出的中间目录 (每个 scan 约 1GB), 默认跑完即删")
    p.add_argument("--geo-views", type=int, default=3, help="min consistent source views")
    p.add_argument("--geo-pix", type=float, default=1.0, help="max reprojection error (px)")
    p.add_argument("--geo-rel", type=float, default=0.01, help="max relative depth difference")
    # Fast-DTU-Evaluation — parsed but no longer driven from main(); see the module docstring.
    p.add_argument("--run-eval", action="store_true",
                   help="DEPRECATED / inert: scoring is now a separate Fast-DTU-Evaluation run "
                        "against --ply-dir. Kept so old command lines still parse.")
    p.add_argument("--eval-tool", default="/home/william/Downloads/Fast-DTU-Evaluation")
    p.add_argument("--eval-gt", default="/home/william/project/dataset/DTU/SampleSet/MVS Data")
    p.add_argument("--eval-workers", type=int, default=1)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data / model setup
# --------------------------------------------------------------------------- #
def _collate(samples: list[dict]) -> dict:
    out: dict = {}
    for k in samples[0]:
        v = samples[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        elif isinstance(v, np.ndarray):
            out[k] = torch.stack([torch.from_numpy(s[k]) for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


def build_dataset(cfg, args) -> DTUMVSDataset:
    # 走和 train.py 同一个解析器, 否则训练和评测会悄悄跑在不同的 scan 集合上。
    if args.list:
        listfile, exclude_file = args.list, None
    else:
        _base = cfg.paths.val_list_file if args.split == "val" else cfg.paths.test_list_file
        listfile, exclude_file = resolve_split(_base, args.split, not args.no_clean_lists)
    ds = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=listfile,
        exclude_file=exclude_file,
        nviews=args.num_views or cfg.train.num_views,
        mode=args.split,
        use_src_weights=cfg.cost_volume.use_src_weights,
    )
    # DTUMVSDataset declares resize_scale as a named __init__ arg but reads
    # self.resize_scale from **kwargs, so a keyword arg is silently ignored —
    # set the attribute directly until that is fixed.
    ds.resize_scale = args.resize_scale
    # No-crop mode: set the crop window equal to the full resized DTU frame
    # (all DTU raw frames are 1200x1600). pick_crop_origin then returns (0, 0)
    # and crop_at keeps the whole image with K only scaled, never shifted — so
    # the plane-sweep stays geometrically aligned (see crop_at / homography_warp).
    if args.full_image:
        ds.height = int(round(1200 * args.resize_scale))
        ds.width = int(round(1600 * args.resize_scale))
    # non-train modes emit one meta per ref view at light 3 — group and trim.
    # Skip empty scan names (lists/dtu/test.txt has a blank first line, which
    # otherwise yields phantom metas that crash on a missing image path).
    per_scan: dict[str, list] = defaultdict(list)
    for meta in ds.metas:
        if meta[0]:
            per_scan[meta[0]].append(meta)
    scans = list(per_scan)
    if args.scans:
        want = {f"scan{s}" for s in args.scans}
        missing = want - set(scans)
        if missing:
            raise SystemExit(f"--scans: {sorted(missing)} not in {listfile}")
        scans = [s for s in scans if s in want]
    if args.max_scans > 0:
        scans = scans[: args.max_scans]
    metas = []
    for scan in scans:
        refs = per_scan[scan]
        if args.max_refs > 0:
            refs = refs[: args.max_refs]
        metas.extend(refs)
    ds.metas = metas
    return ds


def eval_namespace(**overrides) -> argparse.Namespace:
    """离线诊断脚本用的 args 命名空间。

    ``build_dataset`` / ``load_model`` 读十来个 ``args.*`` 字段, 每个诊断脚本
    自己拼一份 Namespace 的话, 漏一个字段就是一次 AttributeError, 而且三份会
    慢慢漂移到不同的默认值上 (于是"同一个验证集"其实不是同一个)。这里给一份,
    诊断脚本只覆盖自己关心的。
    """
    base = dict(
        ckpt=None, ckpt_dir="log/model_eval", spre="auto",
        split="val", list=None, no_clean_lists=False,
        num_views=None, resize_scale=0.8, full_image=True,
        scans=None, max_scans=0, max_refs=0,
        num_workers=8, limit=0, cfg_override="auto",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _resolve_ckpt(args) -> Path:
    """Explicit --ckpt wins. Otherwise prefer best.pth then latest.pth inside
    --ckpt-dir (default log/model_eval, a stable snapshot copied aside so eval
    never reads the checkpoint the running trainer is mid-writing); if that dir
    is absent, fall back to the live log/model."""
    if args.ckpt:
        p = Path(args.ckpt)
        if not p.exists():
            raise FileNotFoundError(f"--ckpt {p} not found")
        return p
    root = ProjectPaths().project_path
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    if not ckpt_dir.exists():
        fallback = root / "log" / "model"
        print(f"[test] --ckpt-dir {ckpt_dir} absent; falling back to {fallback}")
        ckpt_dir = fallback
    for name in ("best.pth", "latest.pth"):
        if (ckpt_dir / name).exists():
            return ckpt_dir / name
    raise FileNotFoundError(
        f"no best.pth/latest.pth in {ckpt_dir} — copy a snapshot there "
        f"(e.g. `cp log/model/latest.pth {ckpt_dir}/`) or pass --ckpt"
    )


def _align_cfg_to_ckpt(cfg, state: dict, override: str = "auto", fingerprint=None):
    """Checkpoints store weights only, never the config they were trained with,
    so architecture switches must be recovered from the key set. Currently that
    means the SPRE head (``spre.*``, whose DINOv3/SVA trunk lives under
    ``dino_sva.*``); with ``--spre auto`` the model is rebuilt to match the
    checkpoint."""
    has_spre = any(k.startswith("spre.") for k in state)
    want = has_spre if override == "auto" else (override == "on")
    if want != cfg.spre.enabled:
        cfg = replace(cfg, spre=replace(cfg.spre, enabled=want))
        src = "checkpoint" if override == "auto" else f"--spre {override}"
        print(f"[test] spre.enabled -> {want} (from {src})")
    if not want:
        cfg = replace(cfg, spre=replace(cfg.spre, reliability_source="cached"))
    # 架构指纹优先于从 state_dict 猜: 40/8 vs 32/16、门控、双模态、可见性头
    # 都不在 state_dict 里 (或只体现为几个 vis_head 的 key), 猜不出来。
    if fingerprint:
        print(f"[test] 按 checkpoint fingerprint 建模型: {fingerprint}")
        cfg = replace(cfg,
                      depth_range=replace(cfg.depth_range,
                                          num_global=fingerprint["num_global"],
                                          num_local=fingerprint["num_local"],
                                          gate_local_branch=fingerprint["gate_local_branch"],
                                          branch_prior=fingerprint["branch_prior"],
                                          dual_mode_stage2=fingerprint["dual_mode_stage2"]),
                      cost_volume=replace(cfg.cost_volume,
                                          visibility_weighting=fingerprint["visibility_weighting"],
                                          use_src_weights=fingerprint["use_src_weights"]))
        # W1/W3 的开关。它们里面有三个会改**参数形状** (spre_cascade 给
        # stage2-4 的正则器 +2 通道, geo_valid +1 通道), 对不上就直接 load 失败;
        # 其余的只改前向路径, 对不上则是"跑了另一个模型"而不报错 —— 更危险。
        # 用 .get 带 legacy 默认值, 好让 2026-08-22 之前的 checkpoint 照常读。
        cfg = replace(cfg,
                      depth_range=replace(cfg.depth_range,
                                          axis_space=fingerprint.get("axis_space", "legacy_depth"),
                                          stage4_head=fingerprint.get("stage4_head", "expect"),
                                          spre_cascade=fingerprint.get("spre_cascade", False),
                                          mode_window_stages=(
                                              tuple(fingerprint["mode_window_stages"])
                                              if fingerprint.get("mode_window_stages") else None)),
                      cost_volume=replace(cfg.cost_volume,
                                          geo_valid_aggregation=fingerprint.get(
                                              "geo_valid_aggregation", False),
                                          vis_mode=fingerprint.get("vis_mode", "softmax")),
                      decoder=replace(cfg.decoder,
                                      fusion_conf=fingerprint.get("fusion_conf", False),
                                      fusion_conf_detach=fingerprint.get(
                                          "fusion_conf_detach", True)))
        if fingerprint.get("tau_stages"):
            cfg = replace(cfg, depth_range=replace(
                cfg.depth_range, tau_stages=tuple(fingerprint["tau_stages"])))
        if override == "auto":
            cfg = replace(cfg, spre=replace(
                cfg.spre, enabled=fingerprint["spre_enabled"],
                reliability_source=fingerprint["reliability_source"]),
                dino=replace(cfg.dino, mode=fingerprint["dino_mode"],
                             feed_fpn=fingerprint["feed_fpn"]))
    else:
        print("[test] WARNING: checkpoint 没有 fingerprint (2026-08-14 之前存的)。"
              " 架构开关只能用当前默认值, 与训练时可能不符 —— "
              "log/model/best.pth 是 32/16 且无 VisibilityHead。")
    return cfg, has_spre


def load_model(cfg, args, device: torch.device) -> tuple[UprMVSNet, Path]:
    ckpt_path = _resolve_ckpt(args)
    # weights_only 默认值在 torch 2.6 变成 True, 而 checkpoint 里有配置快照。
    # 见 train.load_checkpoint。
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    cfg, has_spre = _align_cfg_to_ckpt(cfg, state, getattr(args, "spre", "auto"),
                                       fingerprint=ckpt.get("fingerprint"))
    model = UprMVSNet(cfg).to(device)
    if cfg.spre.enabled == has_spre:
        model.load_state_dict(state)
    else:
        # explicit --spre override that contradicts the checkpoint: the SPRE
        # weights are either absent (head stays at init) or unused (cached conf
        # instead). Either way the numbers are NOT the trained model's.
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[test] WARNING: --spre {args.spre} contradicts the checkpoint "
              f"({len(missing)} missing / {len(unexpected)} unexpected keys); "
              f"this is an ablation, not the trained configuration")
    model.eval()
    step = ckpt.get("step", "?")
    print(f"[test] loaded {ckpt_path} (step {step}, best_metric {ckpt.get('best_metric', float('nan')):.4f})")
    return model, ckpt_path


def ensure_priors(ds: DTUMVSDataset, device: torch.device, mode: str,
                  image_target_wh: tuple[int, int], prior_resize: float) -> None:
    """Build the missing (or all, under 'force') priors, then free VGGT/DA3.

    ``build_prior_cache`` drives ``ds.precrop_inputs``, which sizes its images
    from ``ds.resize_scale`` — so the prior is stored at whatever resize the
    dataset carries. Swap in ``prior_resize`` for the build and restore after,
    which is what decouples the cache resolution from the inference resolution:
    ``_match_hw`` resamples on load, and downsampling a 1200x1600 prior costs
    nothing while upsampling a 600x800 one cannot invent detail.
    """
    if mode == "skip":
        return
    from models.pre_prior import build_prior_cache

    saved = ds.resize_scale
    ds.resize_scale = prior_resize
    try:
        build_prior_cache(ds, device, overwrite=(mode == "force"),
                          image_target_wh=image_target_wh)
    finally:
        ds.resize_scale = saved
    torch.cuda.empty_cache() if device.type == "cuda" else None


# --------------------------------------------------------------------------- #
# Depth metrics
# --------------------------------------------------------------------------- #
class ScanMeter:
    """Pixel-weighted sums per scan + a subsampled error pool for quantiles."""

    def __init__(self) -> None:
        self.sums = defaultdict(lambda: np.zeros(6, dtype=np.float64))  # err, n, <1, <2, <4, <8
        self.pool: dict[str, list[np.ndarray]] = defaultdict(list)

    def update(self, scan: str, err: torch.Tensor) -> None:
        e = err.detach().float()
        s = self.sums[scan]
        s[0] += e.sum().item()
        s[1] += e.numel()
        for i, t in enumerate((1.0, 2.0, 4.0, 8.0)):
            s[2 + i] += (e < t).sum().item()
        if e.numel():
            self.pool[scan].append(e[:: max(e.numel() // 4096, 1)].cpu().numpy())

    def scan_metrics(self, scan: str) -> dict[str, float]:
        s = self.sums[scan]
        n = max(s[1], 1.0)
        pool = np.concatenate(self.pool[scan]) if self.pool[scan] else np.zeros(1)
        return {
            "abs_err": s[0] / n,
            "median": float(np.median(pool)),
            "p90": float(np.percentile(pool, 90)),
            "acc_1mm": s[2] / n, "acc_2mm": s[3] / n,
            "acc_4mm": s[4] / n, "acc_8mm": s[5] / n,
            "pixels": int(s[1]),
        }

    def overall(self) -> dict[str, float]:
        tot = np.sum([self.sums[s] for s in self.sums], axis=0)
        n = max(tot[1], 1.0)
        pool = np.concatenate([v for vs in self.pool.values() for v in vs]) if self.pool else np.zeros(1)
        return {
            "abs_err": tot[0] / n,
            "median": float(np.median(pool)),
            "p90": float(np.percentile(pool, 90)),
            "acc_1mm": tot[2] / n, "acc_2mm": tot[3] / n,
            "acc_4mm": tot[4] / n, "acc_8mm": tot[5] / n,
            "pixels": int(tot[1]),
        }


def _stage_mode_mass(stage: dict, window: int, target_hw) -> torch.Tensor:
    """Posterior mass within +-``window`` bins of this stage's argmax, at ``target_hw``."""
    prob = stage["prob"].float()
    D = prob.shape[1]
    w = min(2 * window + 1, D)
    idx = stage.get("mode_idx")
    if idx is None:
        idx = prob.argmax(dim=1, keepdim=True)
    # Slide the window to stay in bounds rather than clamping indices, which
    # would gather the edge bin repeatedly and double-count its mass.
    start = (idx - (w // 2)).clamp(0, D - w)
    offs = torch.arange(w, device=prob.device).view(1, -1, 1, 1)
    mass = prob.gather(1, start + offs).sum(dim=1).clamp(0.0, 1.0)
    if tuple(mass.shape[-2:]) != tuple(target_hw):
        mass = F.interpolate(mass.unsqueeze(1), size=tuple(target_hw),
                             mode="bilinear", align_corners=False).squeeze(1)
    return mass


def cascade_confidence(outputs: dict, window: int = 1, mode: str = "product",
                       stages=("stage1", "stage2", "stage3", "stage4")) -> torch.Tensor:
    """Fusion confidence combined across cascade stages (ported from test_tt.py).

    Why not the final stage alone: it carries ``num_depths_stage4=4`` hypotheses,
    so the old ``mode_window=2`` window spanned the whole axis and the mass was
    identically 1.0 — which made ``--photo-thresh`` an inert gate and fusion ran
    on geometric consistency alone. That is what invalidated the 2026-08-08 DTU
    numbers (0.3944/0.2482/0.3213); see the note in that run's metrics.

    Stage 1 has 48 bins and is the only level where a +-1 window is genuinely
    selective, so a pixel is trusted when *every* stage concentrated its
    posterior, not just the last.

    ``mode``:
      ``product``  — all stages must agree; the sharpest, and the default.
      ``geomean``  — same ordering, rescaled so a threshold tuned on one stage
                     count still means something.
      ``last``     — final stage only. 注意要复现修复前的失效行为需要
                     ``--conf-mode last --conf-window 2`` (window=2 才覆盖 4 元
                     轴的全部); ``last`` 配 window=1 是另一回事。仅用于 A/B。

    阈值不可跨 mode 迁移: 实测 product 的 min 是 0.019、geomean 的 min 是 0.372,
    同一个 ``--photo-thresh 0.3`` 在 geomean 下仍然一个像素都不过滤。换 mode
    必须重扫 ``--photo-thresh``。
    """
    hw = outputs["depth_full"].shape[-2:]
    masses = [_stage_mode_mass(outputs[s], window, hw) for s in stages if s in outputs]
    if not masses:
        raise KeyError(f"none of {stages} present in outputs")
    if mode == "last":
        return masses[-1]
    stacked = torch.stack(masses, dim=0)
    if mode == "product":
        return stacked.prod(dim=0)
    if mode == "geomean":
        return stacked.clamp_min(1e-8).log().mean(dim=0).exp()
    raise ValueError(f"unknown conf mode {mode!r}")


def depth_vis(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    img = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], (1.0, 99.0))
        gray = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        img = cv2.applyColorMap((gray * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[~valid] = 0
    return img


@torch.no_grad()
def run_inference(model, ds, cfg, args, device, out_root: Path) -> dict:
    # [min, max, mean_sum, n] —— 融合置信度的饱和自检, 见下面的 WARNING
    _conf_samples: list = []      # 融合置信度的抽样, 用于下面的阈值自检
    # risk-coverage 的原料。它是连接 val 与点云的那条桥: 按 conf 降序保留 kappa,
    # 报 R(kappa)。相关性要在最终候选上**实测**确认, 不能预设。
    _rc_conf: list = []
    _rc_err: list = []
    _conf_kind = "cascade"
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=_collate, pin_memory=True)
    use_amp = cfg.train.amp and device.type == "cuda"
    meter = ScanMeter()
    vis_count: dict[str, int] = defaultdict(int)
    mw = cfg.depth_range.mode_window

    for i, batch in enumerate(loader):
        scan, light_idx, ref_view, src_views = ds.metas[i]
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
        pred = outputs["depth_full"].float()
        fc = outputs.get("fusion_conf")
        if args.conf_source == "learned" and fc is None:
            raise SystemExit("--conf-source learned 但这个 checkpoint 没有融合置信度头 "
                             "(训练时需要 --conf-head on)")
        if fc is not None and args.conf_source in ("auto", "learned"):
            conf = fc["prob"].float()          # 已过温度标定
            _conf_kind = f"learned(T={fc['T']:.3f})"
        else:
            conf = cascade_confidence(outputs, window=args.conf_window, mode=args.conf_mode)
        # 饱和自检: conf 恒为 1 意味着 --photo-thresh 门控是死的, 融合实际只跑了
        # 几何一致性。2026-08-08 那组 DTU 数就是这么废掉的。
        _conf_samples.append(conf.detach().flatten()[::97].float().cpu())
        if conf.shape[-2:] != pred.shape[-2:]:
            conf = F.interpolate(conf.unsqueeze(1), size=pred.shape[-2:], mode="bilinear",
                                 align_corners=False).squeeze(1)

        gt = batch["depth_gt"].float()
        m = batch["mask"].bool() & (gt > 0)
        dv = batch["depth_values"].float()
        m &= (gt >= dv.amin(dim=1).view(-1, 1, 1)) & (gt <= dv.amax(dim=1).view(-1, 1, 1))
        if m.any():
            err_m = (pred[m] - gt[m]).abs()
            meter.update(scan, err_m)
            _rc_conf.append(conf[m].detach().flatten()[::13].float().cpu())
            _rc_err.append(err_m.detach().flatten()[::13].float().cpu())

        if args.fuse:
            d = out_root / "depth" / scan
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                d / f"{ref_view:08d}.npz",
                depth=pred[0].cpu().numpy().astype(np.float32),
                conf=conf[0].cpu().numpy().astype(np.float16),
                K=batch["intrinsics"][0, 0].float().cpu().numpy(),
                E=batch["extrinsics"][0, 0].float().cpu().numpy(),
                image=batch["images"][0, 0].permute(1, 2, 0).to(torch.uint8).cpu().numpy(),
                src_views=np.asarray(src_views, dtype=np.int64),
            )
        if args.vis and vis_count[scan] < args.vis:
            d = out_root / "vis" / scan
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{ref_view:08d}.png"), depth_vis(pred[0].cpu().numpy()))
            vis_count[scan] += 1
        if (i + 1) % 20 == 0 or i + 1 == len(ds):
            print(f"[test] {i + 1}/{len(ds)} ({scan} ref {ref_view})", flush=True)

    per_scan = {scan: meter.scan_metrics(scan) for scan in meter.sums}
    if _conf_samples:
        cs = torch.cat(_conf_samples).numpy()
        import numpy as _np
        q = _np.percentile(cs, [1, 5, 50, 95, 99])
        keep = float((cs > args.photo_thresh).mean())
        print(f"[test] fusion confidence [{_conf_kind}] "
              f"(window={args.conf_window}, mode={args.conf_mode}): "
              f"p1 {q[0]:.4f}  p5 {q[1]:.4f}  p50 {q[2]:.4f}  p95 {q[3]:.4f}  p99 {q[4]:.4f}")
        print(f"[test] --photo-thresh {args.photo_thresh} 保留 {100*keep:.2f}% 的像素")
        # "置信度不恒定" != "阈值有效"。真正要报警的是阈值一个像素都不筛,
        # 或者把所有像素都筛掉。
        if keep > 0.999 or keep < 0.001:
            print(f"[test] WARNING: --photo-thresh 保留率 {100*keep:.2f}% —— 这个门"
                  f"实际上没有起作用, 融合等于只跑几何一致性 (或全被筛掉)。"
                  f"这正是 2026-08-08 那组 DTU 数作废的原因。请按上面的分位数"
                  f"重新选阈值; 注意 product / geomean 的阈值不可互换。")

    rc = {}
    if _rc_conf:
        from models.fusion_conf import (
            brier_score, expected_calibration_error, risk_at_coverage, risk_coverage,
        )
        c_all = torch.cat(_rc_conf)
        e_all = torch.cat(_rc_err)
        cov, rk, aurc = risk_coverage(c_all, e_all)
        y2 = (e_all < 2.0).float()
        rc = {
            "conf_source": _conf_kind,
            "aurc_mm": aurc,
            "risk_at_0.6_mm": risk_at_coverage(c_all, e_all, 0.6),
            "risk_at_1.0_mm": float(e_all.mean()),
            "aurc_1m_acc2": risk_coverage(c_all, 1.0 - y2)[2],
            "ece": expected_calibration_error(c_all, y2),
            "brier": brier_score(c_all, y2),
            "curve": {f"{float(k):.2f}": float(v) for k, v in
                      zip(cov[::10].tolist(), rk[::10].tolist())},
        }
        print(f"[test] risk-coverage [{_conf_kind}]: AURC={aurc:.4f}mm  "
              f"R(0.6)={rc['risk_at_0.6_mm']:.4f}mm  R(1.0)={rc['risk_at_1.0_mm']:.4f}mm")
        # ECE 说的是"准不准", AURC 说的是"当过滤器好不好用" —— 两者可以一好一坏。
        # AURC 好而 ECE 差 = 只需重新校准, 不必重训。
        print(f"[test] calibration vs 1[|d-gt|<2mm]: ECE={rc['ece']:.4f}  "
              f"Brier={rc['brier']:.4f}")

    return {"overall": meter.overall(), "per_scan": per_scan, "risk_coverage": rc}


# --------------------------------------------------------------------------- #
# Point-cloud fusion
# --------------------------------------------------------------------------- #
def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v = np.empty(len(points), dtype=dtype)
    v["x"], v["y"], v["z"] = points[:, 0], points[:, 1], points[:, 2]
    v["red"], v["green"], v["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    head = header.encode("ascii")
    want = len(head) + v.nbytes
    with path.open("wb") as f:
        f.write(head)
        v.tofile(f)
        # 网络/配额文件系统上, 缓冲写在配额耗尽时不会立刻报错 —— 错误只在 flush
        # 时冒出来, 不 fsync 的话可能连 flush 都不报, 于是留下一个只有文件头的
        # ply, 而调用方还照常打印 "N points"。2026-08-22 那次 dedup 融合就是:
        # 逐 scan 打印 700-1200 万点, du 却只有 45K。
        f.flush()
        os.fsync(f.fileno())
    got = path.stat().st_size
    if got != want:
        raise IOError(f"{path} 写入不完整: {got} 字节, 应为 {want} "
                      f"({len(points):,} 点)。多半是磁盘配额满了 —— 先 df/quota 再重跑。")


@torch.no_grad()
def fuse_scan(scan_dir: Path, args, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """几何+光度一致性融合。``args.fusion == "dedup"`` 时额外做跨视角去重。

    不去重的话, 49 个 ref 视角各自输出一遍自己的存活像素, 同一个物理表面点在重叠
    区被重复写 5-15 次 —— 一个 scan 2500-4500 万点、ply 600MB。

    去重照搬 fusibile 的做法: 一个 ref 像素通过一致性检查、被写成点之后, 把它在
    各个**源视角**里对应的那个像素标记为"已消费"; 那些视角轮到当 ref 时直接跳过。
    于是每个表面点只由最先认领它的视角输出一次。视角按 id 升序处理, 结果确定。
    """
    dedup = getattr(args, "fusion", "geo") == "dedup"
    consumed: dict[int, torch.Tensor] = {}
    views = {}
    for f in sorted(scan_dir.glob("*.npz")):
        z = np.load(f)
        views[int(f.stem)] = {
            "depth": torch.from_numpy(z["depth"]).to(device),
            "conf": torch.from_numpy(z["conf"].astype(np.float32)).to(device),
            "K": torch.from_numpy(z["K"]).to(device),
            "E": torch.from_numpy(z["E"]).to(device),
            "image": z["image"],
            "src_views": [int(s) for s in z["src_views"]],
        }
    if dedup:
        for vid, v in views.items():
            consumed[vid] = torch.zeros_like(v["depth"], dtype=torch.bool)

    all_pts, all_cols = [], []
    for ref_id in sorted(views):
        ref = views[ref_id]
        src_ids = [s for s in ref["src_views"] if s in views]
        srcs = [views[s] for s in src_ids]
        if not srcs:
            continue
        H, W = ref["depth"].shape
        S = len(srcs)
        d_ref = ref["depth"].view(1, H, W)
        K_ref, E_ref = ref["K"].unsqueeze(0), ref["E"].unsqueeze(0)
        K_src = torch.stack([s["K"] for s in srcs])
        E_src = torch.stack([s["E"] for s in srcs])
        d_src = torch.stack([s["depth"] for s in srcs]).unsqueeze(1)  # [S,1,H,W]

        # ref pixels -> world -> each src image plane
        world = unproject_depth(d_ref, torch.inverse(K_ref), torch.inverse(E_ref))  # [1,3,H,W]
        wf = world.view(1, 3, -1).expand(S, -1, -1)
        cam = torch.bmm(E_src[:, :3, :3], wf) + E_src[:, :3, 3:]
        uv_h = torch.bmm(K_src, cam)
        uv = uv_h[:, :2] / uv_h[:, 2:3].clamp_min(1e-6)                            # [S,2,N]

        # sample each src depth at the projected pixel
        gx = uv[:, 0] / (W - 1) * 2.0 - 1.0
        gy = uv[:, 1] / (H - 1) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(S, H, W, 2)
        d_samp = F.grid_sample(d_src, grid, mode="nearest", padding_mode="zeros",
                               align_corners=True).view(S, -1)                     # [S,N]

        # lift the sampled src depth and project back into the ref view
        uv1 = torch.cat([uv, torch.ones_like(uv[:, :1])], dim=1)
        cam_s = torch.bmm(torch.inverse(K_src), uv1) * d_samp.unsqueeze(1)
        world_s = torch.bmm(torch.inverse(E_src)[:, :3, :3], cam_s) + torch.inverse(E_src)[:, :3, 3:]
        cam_b = torch.bmm(E_ref[:, :3, :3].expand(S, -1, -1), world_s) + E_ref[:, :3, 3:]
        z_back = cam_b[:, 2]
        uv_b = torch.bmm(K_ref.expand(S, -1, -1), cam_b)
        uv_b = uv_b[:, :2] / uv_b[:, 2:3].clamp_min(1e-6)

        gridpix = torch.stack(torch.meshgrid(
            torch.arange(W, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32), indexing="xy"), dim=0)
        err_px = (uv_b - gridpix.view(1, 2, -1)).norm(dim=1)                       # [S,N]
        dr = d_ref.view(1, -1)
        consistent = (d_samp > 0) & (err_px < args.geo_pix) & ((z_back - dr).abs() / dr.clamp_min(1e-6) < args.geo_rel)

        n_geo = consistent.sum(dim=0)
        d_avg = (dr.squeeze(0) + (z_back * consistent).sum(dim=0)) / (n_geo + 1).float()
        keep = ((ref["conf"].view(-1) > args.photo_thresh) & (n_geo >= args.geo_views)
                & (dr.squeeze(0) > 0)).view(H, W)
        if dedup:
            # 已经被更早的 ref 视角认领过的像素不再输出
            keep = keep & ~consumed[ref_id]
        if not keep.any():
            continue
        if dedup:
            # 把这些点在各源视角里对应的像素标记为已消费。uv 是 ref 像素投到该
            # 源视角的浮点坐标, 四舍五入取整并夹进画幅; 只标记"一致 且 被输出"的。
            kf = keep.view(-1)
            ui = uv.round().long()                                     # [S,2,N]
            inb = ((ui[:, 0] >= 0) & (ui[:, 0] < W)
                   & (ui[:, 1] >= 0) & (ui[:, 1] < H))                 # [S,N]
            mark = consistent & inb & kf.unsqueeze(0)
            flat = (ui[:, 1].clamp(0, H - 1) * W + ui[:, 0].clamp(0, W - 1))
            for si, sid in enumerate(src_ids):
                idx = flat[si][mark[si]]
                if idx.numel():
                    consumed[sid].view(-1)[idx] = True
        pts = unproject_depth(d_avg.view(1, H, W), torch.inverse(K_ref), torch.inverse(E_ref))
        pts = pts[0].permute(1, 2, 0)[keep]
        all_pts.append(pts.cpu().numpy())
        all_cols.append(ref["image"][keep.cpu().numpy()])
    if not all_pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(all_pts), np.concatenate(all_cols)


def run_fusion(out_root: Path, ply_dir: Path, args, device: torch.device) -> Path:
    ply_dir.mkdir(parents=True, exist_ok=True)
    scan_dirs = sorted((out_root / "depth").iterdir())

    if getattr(args, "fusion", "geo") == "gipuma":
        from utils.fusion_gipuma import fuse_scan_gipuma
        exe = Path(args.fusibile_exe)
        if not exe.is_absolute():
            exe = Path(ProjectPaths().project_path) / exe
        if not exe.exists():
            raise SystemExit(
                f"找不到 fusibile: {exe}\n"
                "先编译: bash scripts/build_fusibile.sh\n"
                "或改用内置融合: --fusion geo (但它不做跨视角去重, 点数会是几千万)")
        print(f"[fuse] backend=gipuma exe={exe} disp={args.gipuma_disp_thresh} "
              f"num_consistent={args.gipuma_num_consistent} photo_thresh={args.photo_thresh}")
        for sd in scan_dirs:
            scan_id = int(sd.name.replace("scan", ""))
            out = ply_dir / f"mvsnet{scan_id:03d}_l3.ply"
            n = fuse_scan_gipuma(sd, out, str(exe), args.photo_thresh,
                                 args.gipuma_disp_thresh, args.gipuma_num_consistent,
                                 keep_tmp=args.keep_gipuma_tmp)
            print(f"[fuse] {sd.name}: {n:,} points -> {out}" if n >= 0
                  else f"[fuse] {sd.name}: -> {out}")
        return ply_dir

    for sd in scan_dirs:
        scan_id = int(sd.name.replace("scan", ""))
        pts, cols = fuse_scan(sd, args, device)
        out = ply_dir / f"mvsnet{scan_id:03d}_l3.ply"
        save_ply(out, pts, cols)
        print(f"[fuse] {sd.name}: {len(pts):,} points -> {out}")
    return ply_dir


# --------------------------------------------------------------------------- #
# Fast-DTU-Evaluation
#
# No longer called from main() — scoring runs as its own Fast-DTU-Evaluation
# invocation against the ply directory. Kept because it encodes the exact
# argument set the tool wants (and the first-run build hints); re-hook it in
# main() if in-process scoring is ever wanted again.
# --------------------------------------------------------------------------- #
def run_fast_eval(ply_dir: Path, scan_ids: list[int], args) -> None:
    tool = Path(args.eval_tool)
    if not (tool / "eval_dtu.py").exists():
        print(f"[eval] Fast-DTU-Evaluation not found at {tool}; skipping")
        return
    cmd = [sys.executable, "eval_dtu.py",
           "--scans", *[str(s) for s in scan_ids],
           "--method", "mvsnet",
           "--pred_dir", str(ply_dir.resolve()),
           "--gt_dir", args.eval_gt,
           "--num_workers", str(args.eval_workers),
           "--save"]
    print("[eval] running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=tool)
    if r.returncode != 0:
        print("[eval] FAILED — if this is the first run, build its CUDA extension and deps:\n"
              f"  cd {tool}/chamfer3D && {sys.executable} setup.py install --user\n"
              f"  {sys.executable} -m pip install open3d plyfile scikit-learn scipy tqdm")


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    cfg = build_mvs_config(profile=args.profile)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 逐视角 npz + metrics.json；和 ply 一样默认落在 config 的 log/ 下面
    out_root = Path(args.out) if args.out else cfg.paths.depth_cache_path / args.split
    if not out_root.is_absolute():
        out_root = cfg.paths.project_path / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    # ply goes to the config path (<project>/log/pred_points) unless overridden,
    # so it lands in the same place on the laptop and on the cluster.
    ply_dir = Path(args.ply_dir) if args.ply_dir else cfg.paths.pred_points_path
    if not ply_dir.is_absolute():
        ply_dir = cfg.paths.project_path / ply_dir

    ds = build_dataset(cfg, args)
    scans = sorted({m[0] for m in ds.metas}, key=lambda s: int(s.replace("scan", "")))
    prior_wh = (args.prior_target_w or cfg.prior.target_w, args.prior_target_h or cfg.prior.target_h)
    print(f"[test] split={args.split} scans={len(scans)} samples={len(ds)} out={out_root} "
          f"resize={args.resize_scale} full_image={args.full_image} prior_target_wh={prior_wh} "
          f"prior_resize={args.prior_resize_scale}")
    if args.fuse and not args.priors_only:
        print(f"[test] fused point clouds -> {ply_dir}")

    if args.fuse_only:
        # 深度缓存已经在 out_root/depth 下, 不需要先验、不需要模型、不需要 GPU 推理
        dcache = out_root / "depth"
        if not dcache.is_dir() or not any(dcache.iterdir()):
            raise SystemExit(f"--fuse-only 需要已有的深度缓存, 但 {dcache} 为空。\n"
                             f"先不带 --fuse-only 跑一次推理。")
        print(f"[test] --fuse-only: 用 {dcache} 下已缓存的深度重新融合")
        run_fusion(out_root, ply_dir, args, device)
        scan_ids = " ".join(str(int(s.replace("scan", ""))) for s in scans)
        print(f"\n[test] {len(scans)} clouds in {ply_dir}")
        return

    ensure_priors(ds, device, args.build_priors, prior_wh, args.prior_resize_scale)
    if args.priors_only:
        print("[test] --priors-only: prior phase done, exiting before the MVS model")
        return
    model, ckpt_path = load_model(cfg, args, device)

    result = run_inference(model, ds, cfg, args, device, out_root)
    o = result["overall"]
    print(f"\n[depth metrics] overall: abs_err={o['abs_err']:.3f}mm median={o['median']:.3f} "
          f"p90={o['p90']:.3f} acc@1/2/4/8mm={o['acc_1mm']:.3f}/{o['acc_2mm']:.3f}/"
          f"{o['acc_4mm']:.3f}/{o['acc_8mm']:.3f}")
    for scan in scans:
        if scan in result["per_scan"]:
            s = result["per_scan"][scan]
            print(f"  {scan:>8s}: abs_err={s['abs_err']:.3f} median={s['median']:.3f} "
                  f"p90={s['p90']:.3f} acc_2mm={s['acc_2mm']:.3f}")

    summary = {"ckpt": str(ckpt_path), "split": args.split, "num_views": args.num_views or cfg.train.num_views,
               "resize_scale": args.resize_scale, **result}
    (out_root / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[test] wrote {out_root / 'metrics.json'}")

    if not args.fuse:
        return

    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    run_fusion(out_root, ply_dir, args, device)

    # Scoring is deliberately detached: run Fast-DTU-Evaluation against ply_dir.
    scan_ids = " ".join(str(int(s.replace("scan", ""))) for s in scans)
    print(f"\n[test] {len(scans)} clouds in {ply_dir}\n"
          f"[test] score them with:\n"
          f"  python eval_dtu.py --scans {scan_ids} --method mvsnet \\\n"
          f"      --pred_dir {ply_dir} --gt_dir <DTU GT root> --save")
    if args.run_eval:
        print("[test] note: --run-eval is inert now; scoring runs as its own job (see above)")


if __name__ == "__main__":
    main()
