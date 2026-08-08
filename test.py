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

Offline fusion (no model, no dataset, no priors — reads the depth cache only):

python test.py --fuse-only --out outputs/test_test --sweep      # response surface
python test.py --fuse-only --out outputs/test_test \\
    --geo-pix 0.5 --geo-rel 0.001 --geo-views 3 --fusion-src-views 4
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base.config import ProjectPaths, build_mvs_config
from data.dtu import DTUMVSDataset
from models.network import UprMVSNet
from utils.geometry import unproject_depth

# Confidence channels cached per view. Convention: every channel stores its
# NATURAL quantity, and this table says which direction means "more confident",
# so a threshold always reads the way it is written
# (``--conf-channel sigma_mm --conf-thresh 1.5`` == keep sigma < 1.5 mm).
CONF_POLARITY: dict[str, int] = {
    "pmax": +1,        # peak posterior mass
    "margin": +1,      # top1 - top2, the "is there a rival mode" signal
    "entropy": -1,     # normalised to [0, 1]
    "sigma_mm": -1,    # full-axis posterior spread, in millimetres
    "mass1mm": +1,     # posterior mass within +-1 mm of the reported depth
    "mass2mm": +1,
    "mass_win": +1,    # legacy mode-window mass — saturates, see posterior_confidence
}
CONF_CHANNELS: tuple[str, ...] = tuple(CONF_POLARITY)


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
    p.add_argument("--fuse-only", action="store_true",
                   help="skip dataset/prior/model entirely and fuse an existing depth cache "
                        "under --out. This is the whole offline-attribution path: no GPU "
                        "inference, so a threshold grid costs minutes instead of hours.")
    p.add_argument("--cache-stage-depth", action=argparse.BooleanOptionalAction, default=True,
                   help="also cache the coarse stages' depth (~0.2 MB/view). Insurance: it keeps "
                        "a later per-stage fusion possible without re-running inference.")
    p.add_argument("--expect-refs", type=int, default=49,
                   help="required ref views per scan; 0 disables the check. A partially built "
                        "cache silently changes every fused number, so this is on by default.")
    p.add_argument("--allow-partial", action="store_true",
                   help="fuse anyway when a scan has the wrong number of refs (warns).")
    p.add_argument("--no-tag-subdir", action="store_true",
                   help="write PLYs straight into --ply-dir instead of a per-parameter subdir. "
                        "Off by default: overwriting one parameter set with another invalidates "
                        "whatever score was measured from it.")
    # --- geometric consistency ---
    p.add_argument("--geo-views", type=int, default=3, help="min consistent source views")
    p.add_argument("--geo-pix", type=float, default=1.0, help="max reprojection error (px)")
    p.add_argument("--geo-rel", type=float, default=0.01, help="max relative depth difference")
    p.add_argument("--geo-abs-mm", type=float, default=None,
                   help="max ABSOLUTE depth difference in mm (<=0 or unset disables). DTU is "
                        "scored in mm, so this is easier to reason about than --geo-rel, which "
                        "means 4.3mm at 425mm and 9.3mm at 934mm.")
    p.add_argument("--fusion-src-views", type=int, default=10,
                   help="how many of the ref's top-N sources may vote. The cache stores the full "
                        "top-10 while the network only saw num_views-1, so the default 10 makes "
                        "--geo-views 3 mean '3 of 10 agree'. Set 4 to match what the network saw.")
    # --- photometric / posterior confidence ---
    p.add_argument("--conf-channel", default="pmax", choices=sorted(CONF_POLARITY),
                   help="which cached posterior channel filters points. pmax/margin/mass* keep "
                        "HIGH values, entropy/sigma_mm keep LOW ones — --conf-thresh reads the "
                        "same way either direction.")
    p.add_argument("--conf-thresh", type=float, default=float("-inf"),
                   help="threshold on --conf-channel. Defaults to -inf (no photometric filter) "
                        "on purpose: the old --photo-thresh was silently inert, so any non-inf "
                        "default here would change results without saying so.")
    p.add_argument("--photo-thresh", type=float, default=None,
                   help="DEPRECATED alias for --conf-thresh. The channel it used to threshold "
                        "was identically 1.0, so this never filtered anything.")
    # --- offline response surface ---
    p.add_argument("--sweep", action="store_true",
                   help="compute the geometry response surface over the grids below and write "
                        "JSON; no PLY, no evaluator. Implies --fuse-only.")
    p.add_argument("--sweep-pix", default="1.0,0.75,0.5,0.25,0.125")
    p.add_argument("--sweep-rel", default="0.01,0.005,0.002,0.001,0.0005")
    p.add_argument("--sweep-abs", default="0,0.5,1.0,2.0", help="0 = disabled")
    p.add_argument("--sweep-views", default="3,4,5,6")
    p.add_argument("--sweep-pool", default="4,6,10")
    p.add_argument("--sweep-ref-stride", type=int, default=1,
                   help="use every k-th ref view (subsampling for a faster first look)")
    p.add_argument("--sweep-out", default=None, help="default <out>/geo_sweep.json")
    # Fast-DTU-Evaluation — parsed but no longer driven from main(); see the module docstring.
    p.add_argument("--run-eval", action="store_true",
                   help="DEPRECATED / inert: scoring is now a separate Fast-DTU-Evaluation run "
                        "against --ply-dir. Kept so old command lines still parse.")
    p.add_argument("--eval-tool", default="/home/william/Downloads/Fast-DTU-Evaluation")
    p.add_argument("--eval-gt", default="/home/william/project/dataset/DTU/SampleSet/MVS Data")
    p.add_argument("--eval-workers", type=int, default=1)
    args = p.parse_args()
    if args.photo_thresh is not None:
        print("[test] --photo-thresh is deprecated (it thresholded a channel that was "
              "identically 1.0); forwarding it to --conf-thresh on --conf-channel "
              f"{args.conf_channel}")
        args.conf_thresh = args.photo_thresh
    if args.sweep:
        args.fuse_only = True
    return args


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
    listfile = args.list or (cfg.paths.val_list_file if args.split == "val" else cfg.paths.test_list_file)
    ds = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=listfile,
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


def _align_cfg_to_ckpt(cfg, state: dict, override: str = "auto"):
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
    return cfg, has_spre


def load_model(cfg, args, device: torch.device) -> tuple[UprMVSNet, Path]:
    ckpt_path = _resolve_ckpt(args)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    cfg, has_spre = _align_cfg_to_ckpt(cfg, state, getattr(args, "spre", "auto"))
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


def posterior_confidence(
    prob: torch.Tensor,
    hypos: torch.Tensor,
    depth: torch.Tensor,
    mode_idx: torch.Tensor,
    window: int,
) -> dict[str, torch.Tensor]:
    """Per-pixel posterior statistics of the final stage, all [B, H, W].

    Replaces the single ``photometric_confidence`` scalar, which was
    IDENTICALLY 1.0 for the entire cached test set: it summed the mass in
    ``+-window`` bins, and with ``window=2`` on a 4-hypothesis final stage the
    window covers the whole axis, so the sum is 1 by construction. Every
    ``--photo-thresh`` below 1.0 was therefore a no-op and the fusion ran on
    geometric consistency alone.

    ``mass_win`` reproduces that legacy quantity so old runs stay comparable;
    it saturates whenever ``2*window+1 >= D`` and must not be used as a filter.
    The channels that do not depend on the hypothesis count — ``sigma_mm``,
    ``mass1mm``/``mass2mm`` (a fixed *physical* radius) — stay meaningful when
    the cascade layout changes, which is why they are preferred.
    """
    D = prob.shape[1]
    p = prob.float().clamp_min(1e-12)
    h = hypos.float()
    d = depth.float().unsqueeze(1)

    top = p.topk(min(2, D), dim=1).values
    pmax = top[:, 0]
    margin = top[:, 0] - top[:, 1] if D >= 2 else torch.ones_like(pmax)
    entropy = -(p * p.log()).sum(dim=1) / float(torch.log(torch.tensor(float(max(D, 2)))))
    sigma = (p * (h - d) ** 2).sum(dim=1).clamp_min(0).sqrt()

    dist = (h - d).abs()
    w = min(2 * window + 1, D)
    start = (mode_idx - (w // 2)).clamp(0, D - w)
    offs = torch.arange(w, device=prob.device).view(1, -1, 1, 1)
    return {
        "pmax": pmax,
        "margin": margin,
        "entropy": entropy.clamp(0.0, 1.0),
        "sigma_mm": sigma,
        "mass1mm": (p * (dist < 1.0)).sum(dim=1),
        "mass2mm": (p * (dist < 2.0)).sum(dim=1),
        "mass_win": p.gather(1, start + offs).sum(dim=1),
    }


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
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=_collate, pin_memory=True)
    use_amp = cfg.train.amp and device.type == "cuda"
    meter = ScanMeter()
    vis_count: dict[str, int] = defaultdict(int)
    mw = cfg.depth_range.mode_window
    num_views = args.num_views or cfg.train.num_views
    stage_names: tuple[str, ...] = ()

    for i, batch in enumerate(loader):
        scan, light_idx, ref_view, src_views = ds.metas[i]
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
        if not stage_names:   # read the cascade depth off the outputs, not a constant
            stage_names = tuple(sorted((k for k in outputs if k.startswith("stage")),
                                       key=lambda s: int(s[5:])))
        pred = outputs["depth_full"].float()
        last = outputs[stage_names[-1]]
        conf = posterior_confidence(last["prob"].float(), last["depth_hypos"].float(),
                                    last["depth"].float(), last["mode_idx"], mw)
        conf = {k: (v if v.shape[-2:] == pred.shape[-2:] else
                    F.interpolate(v.unsqueeze(1), size=pred.shape[-2:], mode="bilinear",
                                  align_corners=False).squeeze(1))
                for k, v in conf.items()}

        gt = batch["depth_gt"].float()
        m = batch["mask"].bool() & (gt > 0)
        dv = batch["depth_values"].float()
        m &= (gt >= dv.amin(dim=1).view(-1, 1, 1)) & (gt <= dv.amax(dim=1).view(-1, 1, 1))
        if m.any():
            meter.update(scan, (pred[m] - gt[m]).abs())

        if args.fuse:
            d = out_root / "depth" / scan
            d.mkdir(parents=True, exist_ok=True)
            payload = {
                "depth": pred[0].cpu().numpy().astype(np.float32),
                # ``conf`` is kept as an alias of the configured default channel
                # so anything reading the old layout still works; the real
                # filters are the conf_* channels.
                "conf": conf[args.conf_channel][0].cpu().numpy().astype(np.float16),
                "K": batch["intrinsics"][0, 0].float().cpu().numpy(),
                "E": batch["extrinsics"][0, 0].float().cpu().numpy(),
                "image": batch["images"][0, 0].permute(1, 2, 0).to(torch.uint8).cpu().numpy(),
                "src_views": np.asarray(src_views, dtype=np.int64),
                # Provenance so fusion can refuse to mix caches (see
                # assert_cache_coherent). Old caches lack it and are tolerated
                # with a warning.
                "cache_ckpt": np.asarray(str(args.ckpt or args.ckpt_dir)),
                "cache_resize": np.asarray(float(args.resize_scale), dtype=np.float32),
                "cache_num_views": np.asarray(int(num_views), dtype=np.int64),
            }
            for name, m in conf.items():
                payload[f"conf_{name}"] = m[0].cpu().numpy().astype(np.float16)
            if args.cache_stage_depth:
                # Costs ~0.2 MB/view (the coarse stages are small) and is the
                # only thing that would let a per-stage fusion happen later
                # without re-running inference over 22 scans. Insurance, not a
                # plan: the per-stage question is answered in pixel space by
                # utils/stage_metrics.
                for sname in stage_names[:-1]:
                    if sname in outputs:
                        payload[f"depth_{sname}"] = (
                            outputs[sname]["depth"][0].float().cpu().numpy().astype(np.float16))
            np.savez_compressed(d / f"{ref_view:08d}.npz", **payload)
        if args.vis and vis_count[scan] < args.vis:
            d = out_root / "vis" / scan
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{ref_view:08d}.png"), depth_vis(pred[0].cpu().numpy()))
            vis_count[scan] += 1
        if (i + 1) % 20 == 0 or i + 1 == len(ds):
            print(f"[test] {i + 1}/{len(ds)} ({scan} ref {ref_view})", flush=True)

    per_scan = {scan: meter.scan_metrics(scan) for scan in meter.sums}
    return {"overall": meter.overall(), "per_scan": per_scan}


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
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        v.tofile(f)


@dataclass(frozen=True)
class FusionParams:
    """One point on the fusion response surface."""

    geo_pix: float
    geo_rel: float
    geo_abs_mm: float | None      # None = disabled
    geo_views: int
    src_pool: int                 # how many of the ref's top-N sources to vote
    conf_channel: str
    conf_thresh: float

    @property
    def conf_active(self) -> bool:
        return math.isfinite(self.conf_thresh)

    @property
    def tag(self) -> str:
        a = "off" if self.geo_abs_mm is None else f"{self.geo_abs_mm:g}"
        c = f"{self.conf_channel}{self.conf_thresh:g}" if self.conf_active else "confoff"
        return (f"pix{self.geo_pix:g}_rel{self.geo_rel:g}_abs{a}"
                f"_v{self.geo_views}_pool{self.src_pool}_{c}")

    def as_dict(self) -> dict:
        return {
            "geo_pix": self.geo_pix, "geo_rel": self.geo_rel,
            "geo_abs_mm": self.geo_abs_mm, "geo_views": self.geo_views,
            "src_pool": self.src_pool,
            # null, not -Infinity: json.dumps emits a bare ``-Infinity`` that
            # strict JSON parsers reject, and the manifest exists to be read back.
            "conf_channel": self.conf_channel if self.conf_active else None,
            "conf_thresh": self.conf_thresh if self.conf_active else None,
        }


def params_from_args(args) -> FusionParams:
    return FusionParams(
        geo_pix=args.geo_pix, geo_rel=args.geo_rel,
        geo_abs_mm=(None if args.geo_abs_mm is None or args.geo_abs_mm <= 0 else args.geo_abs_mm),
        geo_views=args.geo_views, src_pool=args.fusion_src_views,
        conf_channel=args.conf_channel, conf_thresh=args.conf_thresh,
    )


def load_scan_cache(scan_dir: Path, device: torch.device, conf_channel: str | None) -> dict:
    """Read one scan's per-view npz cache onto ``device``.

    ``conf_channel=None`` skips the confidence map entirely — that is what the
    geometry sweep wants, and it is what lets the pre-fix cache (which only has
    the degenerate all-ones ``conf``) still be attributed offline.
    """
    views: dict[int, dict] = {}
    for f in sorted(scan_dir.glob("*.npz")):
        z = np.load(f)
        conf = None
        if conf_channel is not None:
            key = f"conf_{conf_channel}"
            if key not in z:
                raise KeyError(
                    f"{f} has no '{key}'. This cache predates the multi-channel "
                    f"confidence fix; available: {sorted(k for k in z if k.startswith('conf'))}. "
                    "Re-run inference, or leave --conf-thresh at -inf to fuse on geometry alone."
                )
            conf = torch.from_numpy(z[key].astype(np.float32)).to(device)
        views[int(f.stem)] = {
            "depth": torch.from_numpy(z["depth"]).to(device),
            "conf": conf,
            "K": torch.from_numpy(z["K"]).to(device),
            "E": torch.from_numpy(z["E"]).to(device),
            "image": z["image"],
            "src_views": [int(s) for s in z["src_views"]],
            "meta": {
                "ckpt": str(z["cache_ckpt"]) if "cache_ckpt" in z else None,
                "resize": float(z["cache_resize"]) if "cache_resize" in z else None,
                "num_views": int(z["cache_num_views"]) if "cache_num_views" in z else None,
                "hw": tuple(z["depth"].shape),
            },
        }
    return views


def assert_cache_coherent(views: dict, scan: str, expect_refs: int, allow_partial: bool) -> list[str]:
    """Refuse to fuse a cache that cannot produce a comparable number.

    Returns the list of warnings raised (missing provenance on old caches);
    anything that would silently change the result raises instead.
    """
    warns: list[str] = []
    if not views:
        raise RuntimeError(f"{scan}: empty depth cache")
    if expect_refs > 0 and len(views) != expect_refs:
        msg = f"{scan}: {len(views)} refs cached, expected {expect_refs}"
        if not allow_partial:
            raise RuntimeError(msg + " — pass --allow-partial to fuse anyway")
        warns.append(msg)
    metas = [v["meta"] for v in views.values()]
    for field in ("ckpt", "resize", "num_views", "hw"):
        vals = {m[field] for m in metas}
        if None in vals and len(vals) == 1:
            warns.append(f"{scan}: cache has no '{field}' provenance (pre-fix cache)")
            continue
        if len(vals) > 1:
            raise RuntimeError(
                f"{scan}: cache mixes {field}={sorted(map(str, vals))} — "
                "these views came from different runs, refusing to fuse"
            )
    missing = {s for v in views.values() for s in v["src_views"] if s not in views}
    if missing:
        warns.append(f"{scan}: {len(missing)} referenced source views absent from the cache "
                     f"(e.g. {sorted(missing)[:5]}) — they cannot vote")
    return warns


@torch.no_grad()
def compute_pair_geometry(views: dict, ref_id: int, src_pool: int,
                          device: torch.device) -> dict | None:
    """Reproject one ref view through every source ONCE.

    This is the expensive half of fusion (a grid_sample and four bmm per
    source). Splitting it out means a whole threshold grid costs one projection
    pass plus boolean ops, instead of one projection pass per grid point.

    ``src_pool`` caps how many of the ref's sources vote. ``src_views`` is
    pair.txt order, i.e. descending match score, so ``[:src_pool]`` is top-N.
    This matters: the network only ever saw ``num_views-1`` sources, while the
    cache stores the full top-10 and fusion used all of them — so ``geo_views=3``
    meant "3 of 10 agree", not "3 of the 4 the network used".
    """
    ref = views[ref_id]
    src_ids = [s for s in ref["src_views"][:src_pool] if s in views]
    if not src_ids:
        return None
    srcs = [views[s] for s in src_ids]
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
    dr = d_ref.view(-1)
    abs_err = (z_back - dr.unsqueeze(0)).abs()
    return {
        "H": H, "W": W, "S": S, "src_ids": src_ids,
        "d_ref": dr,                                   # [N]
        "z_back": z_back,                              # [S,N]
        "err_px": err_px,                              # [S,N]
        "abs_err": abs_err,                            # [S,N]  millimetres
        "rel_err": abs_err / dr.unsqueeze(0).clamp_min(1e-6),
        "valid": d_samp > 0,                           # [S,N]
        # ``err_px`` divided by the focal length: the same 1 px means a different
        # angular tolerance at a different inference resolution, so this is the
        # only form comparable across --resize-scale settings.
        "focal": float(ref["K"][0, 0]),
        "conf": None if ref["conf"] is None else ref["conf"].view(-1),   # [N]
        "K_ref": K_ref, "E_ref": E_ref,
    }


def apply_fusion_thresholds(geo: dict, p: FusionParams) -> dict:
    """Pure boolean/reduction pass over a precomputed geometry bundle."""
    pass_pix = geo["valid"] & (geo["err_px"] < p.geo_pix)
    pass_rel = geo["valid"] & (geo["rel_err"] < p.geo_rel)
    pass_abs = (geo["valid"] if p.geo_abs_mm is None
                else geo["valid"] & (geo["abs_err"] < p.geo_abs_mm))
    consistent = pass_pix & pass_rel & pass_abs
    n_geo = consistent.sum(dim=0)
    d_ref = geo["d_ref"]
    d_avg = (d_ref + (geo["z_back"] * consistent).sum(dim=0)) / (n_geo + 1).float()
    keep = (n_geo >= p.geo_views) & (d_ref > 0)
    if geo["conf"] is not None and math.isfinite(p.conf_thresh):
        pol = CONF_POLARITY.get(p.conf_channel, 1)
        keep = keep & (pol * geo["conf"] > pol * p.conf_thresh)
    return {
        "keep": keep, "n_geo": n_geo, "d_avg": d_avg,
        "pass_pix": pass_pix, "pass_rel": pass_rel, "consistent": consistent,
    }


@torch.no_grad()
def fuse_scan(scan_dir: Path, p: FusionParams, device: torch.device,
              expect_refs: int = 0, allow_partial: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    # Only demand a confidence channel when one is actually thresholded, so a
    # geometry-only fusion still runs against the pre-fix cache.
    views = load_scan_cache(
        scan_dir, device, p.conf_channel if math.isfinite(p.conf_thresh) else None)
    for w in assert_cache_coherent(views, scan_dir.name, expect_refs, allow_partial):
        print(f"[fuse] WARNING {w}")
    all_pts, all_cols = [], []
    kept = total = 0
    shift = []
    for ref_id in views:
        geo = compute_pair_geometry(views, ref_id, p.src_pool, device)
        if geo is None:
            continue
        r = apply_fusion_thresholds(geo, p)
        keep = r["keep"]
        kept += int(keep.sum())
        total += keep.numel()
        if not keep.any():
            continue
        shift.append(float((r["d_avg"] - geo["d_ref"])[keep].abs().mean()))
        H, W = geo["H"], geo["W"]
        pts = unproject_depth(r["d_avg"].view(1, H, W),
                              torch.inverse(geo["K_ref"]), torch.inverse(geo["E_ref"]))
        keep_hw = keep.view(H, W)
        all_pts.append(pts[0].permute(1, 2, 0)[keep_hw].cpu().numpy())
        all_cols.append(views[ref_id]["image"][keep_hw.cpu().numpy()])
    stats = {"refs": len(views), "keep_ratio": kept / max(total, 1),
             "mean_depth_shift_mm": float(np.mean(shift)) if shift else 0.0}
    if not all_pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), stats
    return np.concatenate(all_pts), np.concatenate(all_cols), stats


def run_fusion(out_root: Path, ply_dir: Path, args, device: torch.device) -> Path:
    """Fuse every cached scan into ``ply_dir/<param tag>/`` + a manifest.

    Each parameter set gets its own directory: overwriting one PLY set with
    another silently invalidates whatever score was measured from it, and the
    whole point of the sweep is comparing sets.
    """
    p = params_from_args(args)
    ply_dir = ply_dir if args.no_tag_subdir else ply_dir / p.tag
    ply_dir.mkdir(parents=True, exist_ok=True)
    scan_dirs = _select_scan_dirs(out_root, args)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "cache_root": str(out_root), "params": p.as_dict(), "scans": {},
    }
    for sd in scan_dirs:
        scan_id = int(sd.name.replace("scan", ""))
        pts, cols, stats = fuse_scan(sd, p, device, args.expect_refs, args.allow_partial)
        out = ply_dir / f"mvsnet{scan_id:03d}_l3.ply"
        save_ply(out, pts, cols)
        manifest["scans"][sd.name] = {"points": int(len(pts)), **stats}
        print(f"[fuse] {sd.name}: {len(pts):,} points "
              f"(keep {stats['keep_ratio']:.1%}, |d_avg-d_ref|={stats['mean_depth_shift_mm']:.3f}mm) -> {out}")
    (ply_dir / "fusion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[fuse] manifest -> {ply_dir / 'fusion_manifest.json'}")
    return ply_dir


def _select_scan_dirs(out_root: Path, args) -> list[Path]:
    dirs = sorted((d for d in (out_root / "depth").iterdir() if d.is_dir()),
                  key=lambda d: int(d.name.replace("scan", "")))
    if args.scans:
        want = {f"scan{s}" for s in args.scans}
        dirs = [d for d in dirs if d.name in want]
        missing = want - {d.name for d in dirs}
        if missing:
            raise RuntimeError(f"--scans requested {sorted(missing)}, not present in {out_root/'depth'}")
    if not dirs:
        raise RuntimeError(f"no scan directories under {out_root / 'depth'}")
    return dirs


def _parse_grid(spec: str, cast) -> list:
    return [cast(x) for x in spec.split(",") if x.strip() != ""]


@torch.no_grad()
def run_sweep(out_root: Path, args, device: torch.device) -> Path:
    """Geometry response surface — no PLY, no evaluator, no inference.

    Answers "how much of the current Acc is filtering?" cheaply enough to scan
    a whole grid: the projection pass runs once per (ref, src) pair and every
    threshold combination is then a boolean reduction over the cached result.
    Pick a handful of retention levels off this surface, and only fuse those.
    """
    pix_grid = _parse_grid(args.sweep_pix, float)
    rel_grid = _parse_grid(args.sweep_rel, float)
    abs_grid = [None if a <= 0 else a for a in _parse_grid(args.sweep_abs, float)]
    view_grid = _parse_grid(args.sweep_views, int)
    pool_grid = _parse_grid(args.sweep_pool, int)
    scan_dirs = _select_scan_dirs(out_root, args)
    rows: list[dict] = []

    for sd in scan_dirs:
        # Geometry only — the sweep never touches confidence, which is also what
        # makes it runnable on the cache that has none.
        views = load_scan_cache(sd, device, None)
        for w in assert_cache_coherent(views, sd.name, args.expect_refs, args.allow_partial):
            print(f"[sweep] WARNING {w}")
        ref_ids = sorted(views)[::max(1, args.sweep_ref_stride)]
        combos = [(pool, pix, rel, amm, nv)
                  for pool in pool_grid for pix in pix_grid for rel in rel_grid
                  for amm in abs_grid for nv in view_grid]
        cindex = {c: i for i, c in enumerate(combos)}
        print(f"[sweep] {sd.name}: {len(ref_ids)} refs x {len(combos)} combos", flush=True)

        # Accumulate on the GPU. A .item() inside the grid would force a sync per
        # combo per ref (here: ~10^5 syncs per scan) and dominate the runtime.
        # slots: keep, px, pair, pass_pix, pass_rel, pass_both, pix_not_rel, rel_not_pix
        acc = torch.zeros(len(combos), 8, dtype=torch.float64, device=device)
        hist = torch.zeros(len(combos), 11, dtype=torch.float64, device=device)
        shift_acc = torch.zeros(len(combos), 3, dtype=torch.float64, device=device)
        shift_n = torch.zeros(len(combos), dtype=torch.float64, device=device)
        qs = torch.tensor([0.5, 0.9, 0.99], device=device)
        max_pool = max(pool_grid)

        for ref_id in ref_ids:
            # One projection pass at the largest pool; smaller pools are a row
            # slice, because src_views is descending match score so the first k
            # rows ARE the top-k pool.
            full = compute_pair_geometry(views, ref_id, max_pool, device)
            if full is None:
                continue
            npx = full["d_ref"].numel()
            d_ref = full["d_ref"]
            pos_ref = d_ref > 0
            for pool in pool_grid:
                s = min(pool, full["S"])
                v = full["valid"][:s]
                e_px, e_rel, e_abs = full["err_px"][:s], full["rel_err"][:s], full["abs_err"][:s]
                z_back = full["z_back"][:s]
                npair = v.numel()
                for pix in pix_grid:
                    p_pix = v & (e_px < pix)
                    n_pix = p_pix.sum()
                    for rel in rel_grid:
                        p_rel = v & (e_rel < rel)
                        n_rel = p_rel.sum()
                        n_pix_not_rel = (p_pix & ~p_rel).sum()
                        n_rel_not_pix = (p_rel & ~p_pix).sum()
                        for amm in abs_grid:
                            cons = p_pix & p_rel if amm is None else (p_pix & p_rel & (e_abs < amm))
                            n_geo = cons.sum(dim=0)
                            n_both = cons.sum()
                            d_avg = (d_ref + (z_back * cons).sum(dim=0)) / (n_geo + 1).float()
                            shift = (d_avg - d_ref).abs()
                            h = torch.bincount(n_geo.clamp(max=10), minlength=11).double()
                            for nv in view_grid:
                                i = cindex[(pool, pix, rel, amm, nv)]
                                keep = (n_geo >= nv) & pos_ref
                                acc[i, 0] += keep.sum()
                                acc[i, 1] += npx
                                acc[i, 2] += npair
                                acc[i, 3] += n_pix
                                acc[i, 4] += n_rel
                                acc[i, 5] += n_both
                                acc[i, 6] += n_pix_not_rel
                                acc[i, 7] += n_rel_not_pix
                                hist[i] += h
                                sk = shift[keep]
                                if sk.numel():
                                    shift_acc[i] += torch.quantile(sk.float(), qs).double()
                                    shift_n[i] += 1
            del full

        a = acc.cpu().numpy()
        hn = (hist / hist.sum(1, keepdim=True).clamp_min(1)).cpu().numpy()
        sq = (shift_acc / shift_n.clamp_min(1).unsqueeze(1)).cpu().numpy()
        focal = _first_focal(views)
        for c, i in cindex.items():
            pool, pix, rel, amm, nv = c
            rows.append({
                "scan": sd.name, "src_pool": pool, "geo_pix": pix, "geo_rel": rel,
                "geo_abs_mm": amm, "geo_views": nv,
                "keep_ratio": float(a[i, 0] / max(a[i, 1], 1)),
                "raw_points": int(a[i, 0]),
                "pass_pix": float(a[i, 3] / max(a[i, 2], 1)),
                "pass_rel": float(a[i, 4] / max(a[i, 2], 1)),
                "pass_both": float(a[i, 5] / max(a[i, 2], 1)),
                "pix_not_rel": float(a[i, 6] / max(a[i, 2], 1)),
                "rel_not_pix": float(a[i, 7] / max(a[i, 2], 1)),
                "n_geo_hist": [float(x) for x in hn[i]],
                "shift_p50": float(sq[i, 0]), "shift_p90": float(sq[i, 1]),
                "shift_p99": float(sq[i, 2]),
                # 1 px means a different angular tolerance at a different
                # --resize-scale; this is the comparable form.
                "err_norm_pix": pix / max(focal, 1e-6),
            })
        del views
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.sweep_out) if args.sweep_out else out_root / "geo_sweep.json"
    if not out.is_absolute():
        out = out_root / out
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _print_sweep_summary(rows)
    print(f"\n[sweep] {len(rows)} rows -> {out}")
    return out


def _first_focal(views: dict) -> float:
    return float(next(iter(views.values()))["K"][0, 0])


def _print_sweep_summary(rows: list[dict]) -> None:
    """Scan-averaged surface, sorted by retention."""
    agg: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        agg[(r["src_pool"], r["geo_pix"], r["geo_rel"], r["geo_abs_mm"], r["geo_views"])].append(r)
    print(f"\n{'pool':>4} {'pix':>6} {'rel':>7} {'abs':>5} {'v':>2} "
          f"{'keep':>7} {'pass_px':>8} {'pass_rel':>9} {'px!rel':>7} {'rel!px':>7} "
          f"{'shift50':>8} {'shift99':>8}")
    for k in sorted(agg, key=lambda k: -np.mean([r["keep_ratio"] for r in agg[k]])):
        rs = agg[k]
        pool, pix, rel, amm, nv = k
        g = lambda f: np.mean([r[f] for r in rs])  # noqa: E731
        print(f"{pool:>4} {pix:>6g} {rel:>7g} {str(amm):>5} {nv:>2} "
              f"{g('keep_ratio'):>7.4f} {g('pass_pix'):>8.4f} {g('pass_rel'):>9.4f} "
              f"{g('pix_not_rel'):>7.4f} {g('rel_not_pix'):>7.4f} "
              f"{g('shift_p50'):>8.3f} {g('shift_p99'):>8.3f}")


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

    # Offline path: everything below this point (dataset -> priors -> model ->
    # inference) is skipped. The depth cache is all the geometry sweep needs,
    # and re-deriving it costs 22 scans of GPU inference.
    if args.fuse_only:
        if not (out_root / "depth").is_dir():
            raise RuntimeError(
                f"--fuse-only needs a depth cache at {out_root / 'depth'}; pass --out "
                "to point at one (e.g. --out outputs/test_test)"
            )
        if args.sweep:
            run_sweep(out_root, args, device)
        else:
            run_fusion(out_root, ply_dir, args, device)
        return

    ds = build_dataset(cfg, args)
    scans = sorted({m[0] for m in ds.metas}, key=lambda s: int(s.replace("scan", "")))
    prior_wh = (args.prior_target_w or cfg.prior.target_w, args.prior_target_h or cfg.prior.target_h)
    print(f"[test] split={args.split} scans={len(scans)} samples={len(ds)} out={out_root} "
          f"resize={args.resize_scale} full_image={args.full_image} prior_target_wh={prior_wh} "
          f"prior_resize={args.prior_resize_scale}")
    if args.fuse and not args.priors_only:
        print(f"[test] fused point clouds -> {ply_dir}")

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
