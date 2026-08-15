"""Tanks-and-Temples inference + point-cloud fusion for UprMVSNet.

Separate from ``test.py``: T&T differs from DTU on every axis that script
assumes — camera file convention, directory layout, image aspect, and the
complete absence of ground truth. Rather than growing branches inside the DTU
driver, this file carries the T&T-specific parts and imports everything that is
genuinely shared (checkpoint loading, geometric fusion, PLY writing).

    # 1) build the prior cache for one scene (loads VGGT + DA3, then frees them)
    python test_tt.py --scene advanced/Temple --priors-only

    # 2) inference + fusion -> log/pred_points_tt/Temple.ply
    python test_tt.py --scene advanced/Temple --build-priors skip

    # quick smoke check on 5 reference views
    python test_tt.py --scene advanced/Temple --max-refs 5

What differs from DTU, and why it matters
-----------------------------------------
**Camera files.** DTU and T&T both put two numbers on line 12 of ``_cam.txt``,
but DTU means ``(depth_min, depth_interval)`` while T&T means
``(depth_min, depth_max)``. ``data/dtu.py``'s reader multiplies field 2 by 1.06
and treats it as an interval, which on T&T turns a 0.875 max into a range
ending near 178 — with no exception raised, just silently wrong depth
hypotheses. :func:`read_tt_cam_file` branches on the field count instead
(3+ fields = the MVSNet ``min interval num`` form, 2 = ``min max``).

**No ground truth.** T&T withholds GT for intermediate/advanced, so there are
no depth metrics here and no ``depth_gt``/``mask`` keys anywhere. F-score comes
from uploading the fused clouds to the official benchmark site.

**Confidence.** ``test.py``'s ``photometric_confidence`` sums the posterior over
``min(2*window+1, D)`` bins; with ``mode_window=2`` and a 4-hypothesis final
stage that is the whole axis, so the value is identically 1.0 and the
photometric filter does nothing. Fusion then rests on geometric consistency
alone. :func:`cascade_confidence` combines the mode neighbourhood across all
four stages (stage 1 has 48 bins, so it actually discriminates); pass
``--conf-mode last`` to reproduce the old inert behaviour for comparison.

**Aspect ratio.** T&T images are 1920x1080 (1.78:1); DTU is 1600x1200 (1.33:1).
The prior backbones run at a fixed patch-aligned size, and stretching 1.78 into
DTU's 518x420 (1.23:1) distorts every image VGGT and DA3 see. The default here
is 518x294 (1.76:1), the closest 14-multiple to the native aspect.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from base.config import ProjectPaths, build_mvs_config
from models import pre_prior

# Shared with the DTU driver — same geometry, same file format, no reason to fork.
from test import _collate, depth_vis, fuse_scan, load_model, save_ply

SPLITS = ("intermediate", "advanced")
# Reference/source counts and the network stride the cascade needs.
STRIDE = 8


# --------------------------------------------------------------------------- #
# Camera files
# --------------------------------------------------------------------------- #
def read_tt_cam_file(filename) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Parse an MVSNet ``_cam.txt``. Returns (K, extrinsic, depth_min, depth_max).

    Line 12 has either 2 or 3+ fields and the meaning of field 2 changes:

        ``425.0 2.5``                  DTU  -> (min, INTERVAL)
        ``0.394995 0.874995``          T&T  -> (min, MAX)
        ``425.0 2.5 192 [935.0]``      -> (min, interval, num_depth[, max])

    Guessing wrong is silent: an interval read as a max (or the reverse) still
    produces finite numbers, just a depth range off by orders of magnitude. So
    branch on the field count and never on the dataset name.
    """
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    extrinsic = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
    intrinsic = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))

    fields = lines[11].split()
    depth_min = float(fields[0])
    if len(fields) >= 4:
        depth_max = float(fields[3])
    elif len(fields) == 3:
        depth_max = depth_min + int(float(fields[2])) * float(fields[1])
    elif len(fields) == 2:
        depth_max = float(fields[1])          # T&T convention
    else:
        raise ValueError(f"{filename}: cannot parse depth range from {lines[11]!r}")

    if not (np.isfinite(depth_min) and np.isfinite(depth_max)) or depth_max <= depth_min:
        raise ValueError(
            f"{filename}: parsed depth range ({depth_min}, {depth_max}) is not increasing/finite "
            f"— line 12 was {lines[11]!r}. A two-field DTU camera (e.g. '425.0 2.5') lands here, "
            f"because its second field is an INTERVAL, not a max: this reader is for T&T only, "
            f"use data/dtu.py's read_camera_file for DTU."
        )
    return intrinsic, extrinsic, depth_min, depth_max


def read_pair_file(filename) -> list[tuple[int, list[int]]]:
    """``pair.txt`` -> [(ref_view, [src_views ordered by score]), ...]."""
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    num_viewpoint = int(lines[0])
    pairs = []
    for i in range(num_viewpoint):
        ref = int(lines[1 + 2 * i])
        fields = lines[2 + 2 * i].split()
        srcs = [int(x) for x in fields[1::2]]
        if srcs:
            pairs.append((ref, srcs))
    return pairs


def align_to_stride(h: int, w: int, stride: int = STRIDE) -> tuple[int, int]:
    """Largest (h, w) <= input that the 4-stage cascade can consume.

    Stage 1 runs at 1/8, so both dimensions must be multiples of 8 or the
    top-down FPN path and the per-stage upsampling disagree by a pixel.
    """
    return (h // stride) * stride, (w // stride) * stride


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TanksAndTemplesDataset(Dataset):
    """One T&T scene in MVSNet layout.

        <root>/<split>/<Scene>/images/{:0>8}.jpg
        <root>/<split>/<Scene>/cams_1/{:0>8}_cam.txt
        <root>/<split>/<Scene>/pair.txt

    Exposes ``precrop_inputs(idx)`` and ``prior_cache_path_for(idx)``, which is
    the entire interface ``models.pre_prior.build_prior_cache`` needs — so the
    existing VGGT + DA3 + SfM prior pipeline runs on T&T unmodified. It consumes
    only images/intrinsics/extrinsics and never touches ground truth, which is
    what makes that reuse possible here.

    Samples carry no ``depth_gt``/``mask``: T&T has none, and emitting zeros
    would let a metric silently compute against nothing.
    """

    def __init__(
        self,
        root,
        split: str,
        scene: str,
        nviews: int = 5,
        resize_scale: float = 0.5,
        prior_resize_scale: float | None = None,
        prior_cache_root=None,
        max_refs: int = 0,
        fusion_src_views: int = 10,
    ) -> None:
        self.scene_dir = Path(root) / split / scene
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"scene not found: {self.scene_dir}")
        self.split, self.scene = split, scene
        self.nviews = int(nviews)
        self.resize_scale = float(resize_scale)
        # The prior cache is built once at its own scale and resampled on load,
        # so a single build serves every inference resolution.
        self.prior_resize_scale = float(
            resize_scale if prior_resize_scale is None else prior_resize_scale)
        self.fusion_src_views = int(fusion_src_views)

        root_cache = Path(prior_cache_root or (ProjectPaths().project_path / "log" / "prior_cache_tt"))
        self.prior_cache_dir = root_cache / split / scene

        pairs = read_pair_file(self.scene_dir / "pair.txt")
        available = {int(p.stem) for p in (self.scene_dir / "images").glob("*.jpg")}
        self.metas = [(ref, [s for s in srcs if s in available])
                      for ref, srcs in pairs if ref in available]
        self.metas = [(r, s) for r, s in self.metas if len(s) >= self.nviews - 1]
        if max_refs:
            self.metas = self.metas[:max_refs]
        if not self.metas:
            raise RuntimeError(
                f"{self.scene_dir}: no usable reference views "
                f"(need >= {self.nviews - 1} sources each)")
        print(f"[tt] {split}/{scene}: {len(self.metas)} refs, "
              f"{self.nviews} views/sample, resize={self.resize_scale}")

    def __len__(self) -> int:
        return len(self.metas)

    # -- paths ------------------------------------------------------------- #
    def image_path(self, view: int) -> Path:
        return self.scene_dir / "images" / f"{view:0>8}.jpg"

    def cam_path(self, view: int) -> Path:
        return self.scene_dir / "cams_1" / f"{view:0>8}_cam.txt"

    def prior_cache_path_for(self, idx: int) -> Path:
        """Cache key is (scene, ref_view) only — as on DTU it encodes neither the
        resize nor the prior target size, so changing either needs an explicit
        rebuild (``--build-priors force``) or the stale file is reused."""
        ref_view, _ = self.metas[idx]
        return self.prior_cache_dir / f"prior_{ref_view:0>4}.npz"

    # -- image / camera loading -------------------------------------------- #
    def _load_view(self, view: int, resize_scale: float):
        img = np.asarray(Image.open(self.image_path(view)).convert("RGB"))
        K, E, d_min, d_max = read_tt_cam_file(self.cam_path(view))
        K = K.copy()

        h0, w0 = img.shape[:2]
        if resize_scale != 1.0:
            w1, h1 = int(round(w0 * resize_scale)), int(round(h0 * resize_scale))
            img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)
            K[0, :] *= w1 / w0
            K[1, :] *= h1 / h0

        # Centre-crop to a stride multiple; the principal point moves with it.
        h1, w1 = img.shape[:2]
        ch, cw = align_to_stride(h1, w1)
        y0, x0 = (h1 - ch) // 2, (w1 - cw) // 2
        if (ch, cw) != (h1, w1):
            img = img[y0:y0 + ch, x0:x0 + cw]
            K[0, 2] -= x0
            K[1, 2] -= y0
        return img, K, E, d_min, d_max

    def _view_ids(self, idx: int) -> list[int]:
        ref_view, src_views = self.metas[idx]
        return [ref_view] + src_views[:self.nviews - 1]

    # -- prior-precompute interface ---------------------------------------- #
    def precrop_inputs(self, idx: int, resize_scale: float | None = None, aug_params=None) -> dict:
        """Multi-view sample for the offline prior builder.

        ``pre_prior.PriorPrecomputer.compute`` documents its requirement as
        images/intrinsics/extrinsics, so the DTU-only GT keys are simply absent.
        ``aug_params`` exists to match the DTU signature and is ignored — prior
        caching must stay deterministic.
        """
        scale = self.prior_resize_scale if resize_scale is None else resize_scale
        views_np, Ks, Es = [], [], []
        d_min = d_max = None
        for i, v in enumerate(self._view_ids(idx)):
            img, K, E, dmn, dmx = self._load_view(v, scale)
            if i == 0:
                d_min, d_max = dmn, dmx
            views_np.append(img)
            Ks.append(np.asarray(K, np.float32))
            Es.append(np.asarray(E, np.float32))
        imgs = torch.from_numpy(np.stack(views_np, 0)).permute(0, 3, 1, 2).float()
        ref_view, _ = self.metas[idx]
        return {
            "images": imgs,                              # [V,3,H,W] 0-255 RGB
            "views_np": views_np,
            "intrinsics": np.stack(Ks, 0),
            "extrinsics": np.stack(Es, 0),
            "depth_values": np.linspace(d_min, d_max, 192, dtype=np.float32),
            "scan": f"{self.split}/{self.scene}", "ref_view": ref_view, "light_idx": 0,
        }

    def _match_hw(self, arr: np.ndarray, hw, is_depth: bool) -> np.ndarray:
        h, w = hw
        if arr.shape[:2] == (h, w):
            return arr
        interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
        return cv2.resize(arr, (w, h), interpolation=interp)

    # -- network input ------------------------------------------------------ #
    def __getitem__(self, idx: int) -> dict:
        views_np, Ks, Es = [], [], []
        d_min = d_max = None
        for i, v in enumerate(self._view_ids(idx)):
            img, K, E, dmn, dmx = self._load_view(v, self.resize_scale)
            if i == 0:
                d_min, d_max = dmn, dmx
            views_np.append(img)
            Ks.append(np.asarray(K, np.float32))
            Es.append(np.asarray(E, np.float32))
        imgs = torch.from_numpy(np.stack(views_np, 0)).permute(0, 3, 1, 2).float()
        h, w = views_np[0].shape[:2]

        prior = pre_prior.load_prior(self.prior_cache_path_for(idx))
        depth_prior = self._match_hw(prior["depth_prior"], (h, w), is_depth=True)
        conf_prior = self._match_hw(prior["conf_prior"], (h, w), is_depth=False)

        return {
            "images": imgs,
            "intrinsics": np.stack(Ks, 0),
            "extrinsics": np.stack(Es, 0),
            "depth_values": np.linspace(d_min, d_max, 192, dtype=np.float32),
            "depth_prior": depth_prior.astype(np.float32),
            "conf_prior": conf_prior.astype(np.float32),
        }


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
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
    """Fusion confidence combined across cascade stages.

    Why not the final stage alone: it carries 4 hypotheses, so any window >= 2
    spans the whole axis and the mass is identically 1. Stage 1 has 48 bins and
    is the only level where a +-1 window is genuinely selective, so a pixel is
    trusted when *every* stage concentrated its posterior, not just the last.

    ``mode``:
      ``product``  — all stages must agree; the sharpest, and the default.
      ``geomean``  — same ordering, rescaled to a comparable magnitude, so a
                     threshold tuned on one stage count still means something.
      ``last``     — final stage only. Reproduces the inert pre-fix behaviour;
                     keep it for A/B, do not fuse with it.
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


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference_tt(model, ds, cfg, args, device, out_root: Path) -> dict:
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=_collate, pin_memory=True)
    use_amp = cfg.train.amp and device.type == "cuda"
    depth_dir = out_root / "depth" / ds.scene
    depth_dir.mkdir(parents=True, exist_ok=True)

    conf_lo = conf_hi = conf_sum = 0.0
    n_done = 0
    for i, batch in enumerate(loader):
        ref_view, src_views = ds.metas[i]
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
        pred = outputs["depth_full"].float()
        conf = cascade_confidence(outputs, window=args.conf_window, mode=args.conf_mode)

        c = conf[0]
        conf_lo = min(conf_lo, float(c.min())) if n_done else float(c.min())
        conf_hi = max(conf_hi, float(c.max())) if n_done else float(c.max())
        conf_sum += float(c.mean())
        n_done += 1

        np.savez_compressed(
            depth_dir / f"{ref_view:08d}.npz",
            depth=pred[0].cpu().numpy().astype(np.float32),
            conf=c.cpu().numpy().astype(np.float16),
            K=batch["intrinsics"][0, 0].float().cpu().numpy(),
            E=batch["extrinsics"][0, 0].float().cpu().numpy(),
            image=batch["images"][0, 0].permute(1, 2, 0).to(torch.uint8).cpu().numpy(),
            # Fusion may consult more sources than the network saw; the pool is
            # pair.txt's ranking, capped by --fusion-src-views.
            src_views=np.asarray(src_views[:ds.fusion_src_views], dtype=np.int64),
        )
        if args.vis and i < args.vis:
            vd = out_root / "vis" / ds.scene
            vd.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vd / f"{ref_view:08d}.png"), depth_vis(pred[0].cpu().numpy()))
        if (i + 1) % 20 == 0 or i + 1 == len(ds):
            print(f"[tt] {i + 1}/{len(ds)} (ref {ref_view})", flush=True)

    stats = {"views": n_done, "conf_min": conf_lo, "conf_max": conf_hi,
             "conf_mean": conf_sum / max(n_done, 1),
             "conf_mode": args.conf_mode, "conf_window": args.conf_window}
    if conf_lo > 0.999:
        print("[tt] WARNING: confidence is saturated at 1.0 — the photometric "
              "filter is inert and fusion rests on geometry alone. Check "
              "--conf-mode/--conf-window.")
    return stats


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("UprMVSNet Tanks-and-Temples inference + fusion")
    p.add_argument("--tt-root", default="/home/william/project/dataset/TankandTemples")
    p.add_argument("--scene", default="advanced/Temple",
                   help="'<split>/<Scene>', e.g. advanced/Temple or intermediate/Family")
    p.add_argument("--profile", choices=["local", "umhpc"], default="local")
    p.add_argument("--device", default=None)

    # checkpoint (names match test.py so load_model can be reused verbatim)
    p.add_argument("--ckpt", default=None, help="explicit checkpoint file")
    p.add_argument("--ckpt-dir", default="log/model_eval")
    p.add_argument("--spre", choices=["auto", "on", "off"], default="auto")

    p.add_argument("--num-views", type=int, default=5, help="views fed to the network (1 ref + N-1 src)")
    p.add_argument("--resize-scale", type=float, default=0.5,
                   help="1920x1080 at 1.0 needs far more than 16GB; 0.5 -> 960x536 after "
                        "the stride-8 centre crop")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-refs", type=int, default=0, help="limit reference views (0 = all)")
    p.add_argument("--vis", type=int, default=0, help="save the first N depth visualisations")

    # priors
    p.add_argument("--build-priors", choices=["auto", "force", "skip"], default="auto")
    p.add_argument("--priors-only", action="store_true", help="build the prior cache then exit")
    p.add_argument("--prior-resize-scale", type=float, default=None,
                   help="resize the prior cache is BUILT at (default: --resize-scale)")
    p.add_argument("--prior-target-w", type=int, default=518,
                   help="VGGT/DA3 input width, multiple of 14")
    p.add_argument("--prior-target-h", type=int, default=294,
                   help="VGGT/DA3 input height, multiple of 14. Default 294 keeps T&T's "
                        "16:9 aspect (518/294 = 1.76); DTU's 420 would stretch it to 1.23")

    # confidence + fusion
    p.add_argument("--conf-mode", choices=["product", "geomean", "last"], default="product")
    p.add_argument("--conf-window", type=int, default=1)
    p.add_argument("--photo-thresh", type=float, default=0.05)
    p.add_argument("--geo-pix", type=float, default=1.0)
    p.add_argument("--geo-rel", type=float, default=0.01)
    p.add_argument("--geo-views", type=int, default=3)
    p.add_argument("--fusion-src-views", type=int, default=10,
                   help="source pool for geometric checking (pair.txt ranking)")
    p.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fuse-only", action="store_true",
                   help="skip inference, fuse the cached depth maps")
    p.add_argument("--out", default=None, help="working dir (default log/outputs_tt/<Scene>)")
    p.add_argument("--ply-dir", default=None, help="default log/pred_points_tt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "/" not in args.scene:
        raise SystemExit(f"--scene must be '<split>/<Scene>', got {args.scene!r}")
    split, scene = args.scene.split("/", 1)
    if split not in SPLITS:
        raise SystemExit(f"split must be one of {SPLITS}, got {split!r}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = build_mvs_config(args.profile)
    project = ProjectPaths().project_path
    out_root = Path(args.out) if args.out else project / "log" / "outputs_tt" / scene
    ply_dir = Path(args.ply_dir) if args.ply_dir else project / "log" / "pred_points_tt"
    out_root.mkdir(parents=True, exist_ok=True)

    ds = TanksAndTemplesDataset(
        root=args.tt_root, split=split, scene=scene,
        nviews=args.num_views, resize_scale=args.resize_scale,
        prior_resize_scale=args.prior_resize_scale,
        max_refs=args.max_refs, fusion_src_views=args.fusion_src_views,
    )

    # ---- phase 1: priors (VGGT + DA3 loaded once, freed before the MVS model) --
    if not args.fuse_only and args.build_priors != "skip":
        target_wh = (args.prior_target_w, args.prior_target_h)
        print(f"[tt] building priors at {target_wh} (prior resize={ds.prior_resize_scale}) ...")
        pre_prior.build_prior_cache(
            ds, device, overwrite=(args.build_priors == "force"), image_target_wh=target_wh)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.priors_only:
        print(f"[tt] prior cache ready -> {ds.prior_cache_dir}")
        return

    # ---- phase 2: inference ------------------------------------------------- #
    stats = {}
    if not args.fuse_only:
        model, ckpt_path = load_model(cfg, args, device)
        h, w = ds._load_view(ds.metas[0][0], ds.resize_scale)[0].shape[:2]
        print(f"[tt] inference at {h}x{w}, {args.num_views} views, "
              f"conf={args.conf_mode}/w{args.conf_window}")
        stats = run_inference_tt(model, ds, cfg, args, device, out_root)
        stats["checkpoint"] = str(ckpt_path)
        stats["resolution"] = [h, w]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- phase 3: fusion ---------------------------------------------------- #
    if args.fuse:
        scan_dir = out_root / "depth" / scene
        if not any(scan_dir.glob("*.npz")):
            raise SystemExit(f"no cached depth maps in {scan_dir} — run without --fuse-only first")
        print(f"[tt] fusing (photo>{args.photo_thresh}, geo_views>={args.geo_views}) ...")
        points, colors = fuse_scan(scan_dir, args, device)
        ply_dir.mkdir(parents=True, exist_ok=True)
        ply_path = ply_dir / f"{scene}.ply"
        save_ply(ply_path, points, colors)
        # Record what produced this cloud. A .ply whose thresholds are unknown
        # cannot be reproduced or compared against another run.
        stats["fusion"] = {
            "points": int(len(points)), "ply": str(ply_path),
            "photo_thresh": args.photo_thresh, "geo_pix": args.geo_pix,
            "geo_rel": args.geo_rel, "geo_views": args.geo_views,
            "fusion_src_views": args.fusion_src_views,
        }
        # Sidecar next to the cloud as well: stats.json keeps only the latest
        # fusion, so a threshold sweep would otherwise leave several .ply files
        # with no way to tell which settings produced which.
        ply_path.with_suffix(".json").write_text(
            json.dumps(stats["fusion"], indent=2), encoding="utf-8")
        print(f"[tt] {len(points):,} points -> {ply_path}")

    # Merge, never overwrite: --fuse-only runs no inference, so a plain write
    # would drop the confidence/resolution/checkpoint record of the pass that
    # actually produced the depth maps being fused.
    stats_path = out_root / "stats.json"
    merged = {}
    if stats_path.exists():
        try:
            merged = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(stats)
    stats_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[tt] stats -> {stats_path}")


if __name__ == "__main__":
    main()
