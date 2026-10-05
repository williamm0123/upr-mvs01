"""Run PDA v1.1 on DTU with a reproducible 1000-point GT prior.

GT is used as an oracle sparse prior, not as a deployable MVS input.
The upstream PDA checkout is left unchanged; optional xformers is disabled
only in this process to run its existing FP32 attention fallback on SM120.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models/PDA"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from data.io import read_pfm
from prior_depth_anything import PriorDepthAnything
from prior_depth_anything.depth_anything_v2.dinov2_layers import attention, block


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def metrics(pred, gt, mask):
    values = pred[mask].astype(np.float64)
    target = gt[mask].astype(np.float64)
    error = np.abs(values - target)
    ratio = np.maximum(values / target, target / np.maximum(values, 1e-12))
    return {
        "pixels": int(mask.sum()), "mae_mm": float(error.mean()),
        "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "abs_rel": float(np.mean(error / target)),
        "median_ae_mm": float(np.median(error)),
        "pct_below_2mm": float(np.mean(error < 2) * 100),
        "pct_below_4mm": float(np.mean(error < 4) * 100),
        "delta1": float(np.mean((values > 0) & (ratio < 1.25))),
    }


def render(out, rgb, gt, sparse, pred, valid, title):
    low, high = np.percentile(gt[valid], [1, 99])
    error = np.where(valid, np.abs(pred - gt), np.nan)
    error_max = max(1., float(np.nanpercentile(error, 95)))
    masked_gt = np.where(valid, gt, np.nan)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.3), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    y, x = np.nonzero(sparse > 0)
    axes[1].scatter(x, y, c=sparse[y, x], s=2, cmap="Spectral_r", vmin=low, vmax=high)
    axes[1].set_xlim(0, sparse.shape[1])
    axes[1].set_ylim(sparse.shape[0], 0)
    axes[1].set_aspect("equal")
    axes[1].set_title("GT prior: 1000 points (markers enlarged)")
    for ax, image, label in zip(axes[2:4], [masked_gt, pred],
                                ["GT depth (mm)", "PDA v1.1 depth (mm)"]):
        artist = ax.imshow(image, cmap="Spectral_r", vmin=low, vmax=high)
        ax.set_title(label)
    fig.colorbar(artist, ax=axes[1:4], shrink=.65, label="Depth (mm)")
    artist = axes[4].imshow(error, cmap="magma", vmin=0, vmax=error_max)
    axes[4].set_title("Absolute error on valid GT")
    fig.colorbar(artist, ax=axes[4], shrink=.65, label="Error (mm)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title + " | GT sparse prior demo (oracle input)")
    fig.savefig(out / "comparison.png", dpi=130)
    plt.close(fig)
    plt.imsave(out / "prediction_color.png", pred, cmap="Spectral_r", vmin=low, vmax=high)
    plt.imsave(out / "error_color.png", error, cmap="magma", vmin=0, vmax=error_max)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/william/project/dataset/DTU/dtu_training"))
    parser.add_argument("--weights", type=Path, default=Path("/home/william/project/dataset/prior depth anything"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scans", type=int, nargs="+", default=[11, 29, 48, 49, 77, 114])
    parser.add_argument("--refs", type=int, nargs="+", default=[1, 17, 33, 45])
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    attention.XFORMERS_AVAILABLE = False
    block.XFORMERS_AVAILABLE = False
    assert torch.cuda.is_available(), "This demo requires CUDA."
    config = {
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT / "models/PDA"), "rev-parse", "HEAD"], text=True).strip(),
        "data_root": str(args.data_root), "scans": args.scans, "refs": args.refs,
        "ref_indexing": "zero-based; RGB filename index = ref + 1", "light": 3,
        "prior": "uniform GT sparse sampling", "oracle_gt_prior": True,
        "points": args.points, "seed": args.seed, "resolution": [1200, 1600],
        "input_depth_unit": "metres (DTU millimetres / 1000)",
        "output_depth_unit": "millimetres", "model_version": "1.1",
        "frozen_model_size": "vitb", "conditioned_model_size": "vitb",
        "attention": "upstream native attention fallback, FP32; process-local flags only",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "torch_threads": 8,
        "weights": [fingerprint(args.weights / name) for name in
                    ["depth_anything_v2_vitb.pth", "prior_depth_anything_vitb_1_1.pth"]],
    }
    dump_json(args.output / "config.json", config)
    model = PriorDepthAnything(device="cuda:0", version="1.1", mde_dir=str(args.weights), ckpt_dir=str(args.weights)).eval()
    rows = []
    all_results = []
    for scan in args.scans:
        for ref in args.refs:
            seed = args.seed + scan * 100 + ref
            torch.manual_seed(seed)
            np.random.seed(seed)
            image_path = args.data_root / f"Rectified_raw/scan{scan}/rect_{ref + 1:03d}_3_r5000.png"
            gt_path = args.data_root / f"Depths_raw/scan{scan}/depth_map_{ref:04d}.pfm"
            mask_path = args.data_root / f"Depths_raw/scan{scan}/depth_visual_{ref:04d}.png"
            rgb = np.asarray(Image.open(image_path).convert("RGB")).copy()
            gt = read_pfm(str(gt_path)).astype(np.float32)
            valid = (np.asarray(Image.open(mask_path)) > 10) & np.isfinite(gt) & (gt > 0)
            assert rgb.shape[:2] == gt.shape == valid.shape
            gt_prior_m = np.where(valid, gt / 1000., 0).astype(np.float32)
            # Use the author's sampler, then pass only sampled depth into inference.
            sparse, sparse_mask, _ = model.sampler.get_sparse_depth(
                image=rgb, prior=torch.from_numpy(gt_prior_m), pattern=str(args.points))
            assert int(sparse_mask.sum()) == args.points
            sparse_mm = sparse.numpy() * 1000.
            sparse_mask_np = sparse_mask.numpy()
            out = args.output / f"scan{scan}" / f"ref{ref:02d}"
            out.mkdir(parents=True)
            print(f"RUN scan{scan}/ref{ref:02d} shape={gt.shape} prior_points={args.points}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = model.infer_one_sample(image=rgb, prior=sparse.numpy(), visualize=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            pred = prediction.float().cpu().numpy() * 1000.
            assert pred.shape == gt.shape
            assert np.isfinite(pred).all(), "PDA output contains nonfinite values."
            result = {
                "scan": scan, "ref": ref, "seed": seed,
                "image_path": str(image_path), "gt_path": str(gt_path), "mask_path": str(mask_path),
                "prior_points": int(sparse_mask.sum()), "oracle_gt_prior": True,
                "seconds": elapsed, "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                "finite_fraction": float(np.isfinite(pred).mean()), "positive_fraction": float((pred > 0).mean()),
                "all_valid_gt": metrics(pred, gt, valid),
                "heldout_gt": metrics(pred, gt, valid & ~sparse_mask_np),
                "sampled_prior": metrics(pred, gt, sparse_mask_np),
            }
            np.save(out / "prediction_mm.npy", pred)
            np.save(out / "gt_mm.npy", np.where(valid, gt, 0).astype(np.float32))
            np.save(out / "sparse_prior_mm.npy", sparse_mm)
            Image.fromarray(valid.astype(np.uint8) * 255).save(out / "valid_mask.png")
            Image.fromarray(rgb).save(out / "rgb.png")
            render(out, rgb, gt, sparse_mm, pred, valid, f"scan{scan} / ref{ref:02d}")
            dump_json(out / "metrics.json", result)
            all_results.append(result)
            rows.append({"scan": scan, "ref": ref, "seconds": elapsed,
                         "peak_cuda_allocated_gib": result["peak_cuda_allocated_gib"],
                         **result["heldout_gt"]})
            dump_json(args.output / "results.json", all_results)
            with (args.output / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"DONE scan{scan}/ref{ref:02d} heldout_MAE={result['heldout_gt']['mae_mm']:.4f}mm time={elapsed:.2f}s", flush=True)
            del prediction
    aggregate = {key: float(np.mean([r[key] for r in rows])) for key in
                 ["mae_mm", "rmse_mm", "abs_rel", "pct_below_2mm", "pct_below_4mm", "delta1", "seconds"]}
    dump_json(args.output / "summary.json", {"completed": len(rows), "expected": len(args.scans) * len(args.refs),
              "oracle_gt_prior": True, "evaluation": "held-out GT pixels excluding sampled prior points",
              "aggregation": "equal-weight mean across samples", "mean": aggregate})
    print("COMPLETE", json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()
