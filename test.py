"""Test / evaluation driver for UprMVSNet.

Three levels, fastest first:

1. Depth metrics (default): run the checkpoint over a DTU split and report
   masked depth-map errors (same masking as train.py validation) with a
   per-scan breakdown plus median/p90. The quick "did it get better" signal.
2. ``--fuse``: additionally cache per-view depth/conf and fuse each scan into
   a point cloud (photometric + geometric consistency filtering), written as
   ``<out>/ply/mvsnet{scan:03d}_l3.ply`` — the naming Fast-DTU-Evaluation
   expects.
3. ``--run-eval``: invoke the GPU Fast-DTU-Evaluation (accuracy /
   completeness / overall against the official STL points) on the fused
   clouds.

Priors: the network consumes cached depth/conf priors; missing entries for
the requested split are built automatically (VGGT + DA3 loaded once) unless
``--build-priors skip``.

Examples
--------
python test.py --split val                        # depth metrics on val scans
python test.py --split test --max-refs 5          # quick test-split check
python test.py --split test --fuse --run-eval     # full point-cloud benchmark
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("UprMVSNet test / DTU evaluation")
    p.add_argument("--profile", choices=["local", "umhpc"], default=None)
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
    p.add_argument("--prior-target-w", type=int, default=None,
                   help="VGGT/DA3 prior width (default cfg 518; must be a multiple of 14). Raises the "
                        "true depth-prior resolution. Needs --build-priors force to take effect (else "
                        "the existing cache is reused). VGGT cost/memory grows ~O((w*h/196)^2).")
    p.add_argument("--prior-target-h", type=int, default=None,
                   help="VGGT/DA3 prior height (default cfg 420; must be a multiple of 14)")
    p.add_argument("--max-scans", type=int, default=0)
    p.add_argument("--max-refs", type=int, default=0, help="limit ref views per scan (0 = all 49)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default=None, help="output root (default outputs/test_<split>)")
    p.add_argument("--vis", type=int, default=0, help="save the first N depth visualizations per scan")
    p.add_argument("--build-priors", choices=["auto", "skip", "force"], default="auto")
    # fusion
    p.add_argument("--fuse", action="store_true", help="save per-view outputs and fuse point clouds")
    p.add_argument("--photo-thresh", type=float, default=0.3, help="stage-3 mode-probability threshold")
    p.add_argument("--geo-views", type=int, default=3, help="min consistent source views")
    p.add_argument("--geo-pix", type=float, default=1.0, help="max reprojection error (px)")
    p.add_argument("--geo-rel", type=float, default=0.01, help="max relative depth difference")
    # Fast-DTU-Evaluation
    p.add_argument("--run-eval", action="store_true", help="run Fast-DTU-Evaluation on the fused clouds")
    p.add_argument("--eval-tool", default=None,
                   help="Fast-DTU-Evaluation dir (default: profile's cfg.paths.eval_tool)")
    p.add_argument("--eval-gt", default=None,
                   help="DTU GT points dir (default: profile's cfg.paths.eval_gt)")
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
    listfile = args.list or (cfg.paths.val_list_file if args.split == "val" else cfg.paths.test_list_file)
    # val 复用 dtu_training (含 GT); test 图像/相机走 dtu_testing, GT 深度/mask 仍取
    # 自 dtu_training/Depths_raw (gt_datapath)。
    datapath = cfg.paths.dtu_test_root if args.split == "test" else cfg.paths.dtu_train_root
    gt_datapath = cfg.paths.dtu_train_root if args.split == "test" else None
    ds = DTUMVSDataset(
        datapath=datapath,
        listfile=listfile,
        nviews=args.num_views or cfg.train.num_views,
        mode=args.split,
        gt_datapath=gt_datapath,
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


def load_model(cfg, args, device: torch.device) -> tuple[UprMVSNet, Path]:
    ckpt_path = _resolve_ckpt(args)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UprMVSNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"[test] loaded {ckpt_path} (step {step}, best_metric {ckpt.get('best_metric', float('nan')):.4f})")
    return model, ckpt_path


def ensure_priors(ds: DTUMVSDataset, device: torch.device, mode: str,
                  image_target_wh: tuple[int, int]) -> None:
    if mode == "skip":
        return
    from models.pre_prior import build_prior_cache

    build_prior_cache(ds, device, overwrite=(mode == "force"), image_target_wh=image_target_wh)
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
        if not self.sums:  # 测试集无 GT: 没有任何指标累积
            return {"abs_err": 0.0, "median": 0.0, "p90": 0.0,
                    "acc_1mm": 0.0, "acc_2mm": 0.0, "acc_4mm": 0.0, "acc_8mm": 0.0,
                    "pixels": 0}
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


def photometric_confidence(prob: torch.Tensor, mode_idx: torch.Tensor, window: int) -> torch.Tensor:
    """Probability mass in +-window bins around the argmax mode ([B, H, W])."""
    D = prob.shape[1]
    offs = torch.arange(-window, window + 1, device=prob.device).view(1, -1, 1, 1)
    nbr = (mode_idx + offs).clamp(0, D - 1)
    return prob.gather(1, nbr).sum(dim=1)


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

    for i, batch in enumerate(loader):
        scan, light_idx, ref_view, src_views = ds.metas[i]
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
        pred = outputs["depth_full"].float()
        conf = photometric_confidence(outputs["stage3"]["prob"].float(),
                                      outputs["stage3"]["mode_idx"], mw)
        if conf.shape[-2:] != pred.shape[-2:]:
            conf = F.interpolate(conf.unsqueeze(1), size=pred.shape[-2:], mode="bilinear",
                                 align_corners=False).squeeze(1)

        # 测试集无 GT: depth_gt/mask 为 None (collate 成 [None]), 跳过深度指标, 只做融合。
        gt = batch.get("depth_gt")
        if isinstance(gt, torch.Tensor):
            gt = gt.float()
            mask_valid = batch["mask"].bool()
            dv = batch["depth_values"].float()
            lo = dv.amin(dim=1).view(-1, 1, 1)
            hi = dv.amax(dim=1).view(-1, 1, 1)
            m = mask_valid & (gt > 0) & (gt >= lo) & (gt <= hi)
            if i == 0:
                # 一次性诊断: 逐级看哪一步把有效像素过滤空了 (GT 对齐/尺度问题)。
                pos = gt[gt > 0]
                gstat = (float(pos.min()), float(pos.median()), float(pos.max())) if pos.numel() else (0.0, 0.0, 0.0)
                print(f"[diag] gt{tuple(gt.shape)} pred{tuple(pred.shape)} "
                      f"mask>0={int(mask_valid.sum())} gt>0={int((gt > 0).sum())} "
                      f"in_range={int(m.sum())} dv=[{float(lo.min()):.1f},{float(hi.max()):.1f}] "
                      f"gt[min,med,max]=[{gstat[0]:.1f},{gstat[1]:.1f},{gstat[2]:.1f}]", flush=True)
            if m.any():
                meter.update(scan, (pred[m] - gt[m]).abs())
        elif i == 0:
            print(f"[diag] depth_gt is not a tensor: {type(gt).__name__}", flush=True)

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
            cv2.imwrite(str(d / f"{ref_view:08d}.jpg"), depth_vis(pred[0].cpu().numpy()))
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


@torch.no_grad()
def fuse_scan(scan_dir: Path, args, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
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
    all_pts, all_cols = [], []
    for ref_id, ref in views.items():
        srcs = [views[s] for s in ref["src_views"] if s in views]
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
        if not keep.any():
            continue
        pts = unproject_depth(d_avg.view(1, H, W), torch.inverse(K_ref), torch.inverse(E_ref))
        pts = pts[0].permute(1, 2, 0)[keep]
        all_pts.append(pts.cpu().numpy())
        all_cols.append(ref["image"][keep.cpu().numpy()])
    if not all_pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(all_pts), np.concatenate(all_cols)


def run_fusion(out_root: Path, args, device: torch.device) -> Path:
    ply_dir = out_root / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    scan_dirs = sorted((out_root / "depth").iterdir())
    for sd in scan_dirs:
        scan_id = int(sd.name.replace("scan", ""))
        pts, cols = fuse_scan(sd, args, device)
        out = ply_dir / f"mvsnet{scan_id:03d}_l3.ply"
        save_ply(out, pts, cols)
        print(f"[fuse] {sd.name}: {len(pts):,} points -> {out}")
    return ply_dir


# --------------------------------------------------------------------------- #
# Fast-DTU-Evaluation
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
    # Fall back to the active profile's paths when eval dirs are not overridden.
    if args.eval_tool is None:
        args.eval_tool = str(cfg.paths.eval_tool)
    if args.eval_gt is None:
        args.eval_gt = str(cfg.paths.eval_gt)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out) if args.out else Path("outputs") / f"test_{args.split}"
    out_root.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(cfg, args)
    scans = sorted({m[0] for m in ds.metas}, key=lambda s: int(s.replace("scan", "")))
    prior_wh = (args.prior_target_w or cfg.prior.target_w, args.prior_target_h or cfg.prior.target_h)
    print(f"[test] split={args.split} scans={len(scans)} samples={len(ds)} out={out_root} "
          f"resize={args.resize_scale} full_image={args.full_image} prior_target_wh={prior_wh}")

    ensure_priors(ds, device, args.build_priors, prior_wh)
    model, ckpt_path = load_model(cfg, args, device)

    result = run_inference(model, ds, cfg, args, device, out_root)
    o = result["overall"]
    if o["pixels"] > 0:
        print(f"\n[depth metrics] overall: abs_err={o['abs_err']:.3f}mm median={o['median']:.3f} "
              f"p90={o['p90']:.3f} acc@1/2/4/8mm={o['acc_1mm']:.3f}/{o['acc_2mm']:.3f}/"
              f"{o['acc_4mm']:.3f}/{o['acc_8mm']:.3f}")
        for scan in scans:
            if scan in result["per_scan"]:
                s = result["per_scan"][scan]
                print(f"  {scan:>8s}: abs_err={s['abs_err']:.3f} median={s['median']:.3f} "
                      f"p90={s['p90']:.3f} acc_2mm={s['acc_2mm']:.3f}")
    else:
        print("\n[depth metrics] test split has no GT depth — skipping metrics (fusion only).")

    summary = {"ckpt": str(ckpt_path), "split": args.split, "num_views": args.num_views or cfg.train.num_views,
               "resize_scale": args.resize_scale, **result}
    (out_root / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[test] wrote {out_root / 'metrics.json'}")

    if args.fuse:
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        ply_dir = run_fusion(out_root, args, device)
        if args.run_eval:
            run_fast_eval(ply_dir, [int(s.replace("scan", "")) for s in scans], args)


if __name__ == "__main__":
    main()
