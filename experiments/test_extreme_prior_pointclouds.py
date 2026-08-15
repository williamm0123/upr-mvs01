#!/usr/bin/env python3
"""Visualize extreme DTU priors against ground-truth depth as colored point clouds.

Each output PLY contains two point sets back-projected with exactly the same
scaled camera intrinsics and world-to-camera extrinsics:

* ground truth: white
* metric prior: blue

The default cases are the three large-error samples found in the training-log
audit.  Additional cases can be supplied as ``--case scan:ref:light``.

Example
-------
python experiments/test_extreme_prior_pointclouds.py
python experiments/test_extreme_prior_pointclouds.py --case scan51:37:0 --stride 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import read_pfm


DEFAULT_DTU_ROOT = Path("/home/william/project/dataset/DTU/dtu_training")
DEFAULT_PRIOR_ROOT = PROJECT_ROOT / "log" / "prior_cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "out" / "extreme_prior_pointclouds"


@dataclass(frozen=True)
class Case:
    scan: str
    ref: int
    light: int

    @property
    def stem(self) -> str:
        return f"{self.scan}_ref{self.ref:04d}_light{self.light}"


DEFAULT_CASES = (
    Case("scan51", 37, 0),
    Case("scan51", 38, 0),
    Case("scan8", 39, 0),
)


def parse_case(text: str) -> Case:
    try:
        scan, ref, light = text.split(":")
        if not scan.startswith("scan"):
            scan = f"scan{int(scan)}"
        return Case(scan=scan, ref=int(ref), light=int(light))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"case must be scan:ref:light, for example scan51:37:0; got {text!r}"
        ) from exc


def read_camera(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    extrinsic = np.fromstring(" ".join(lines[1:5]), sep=" ", dtype=np.float64).reshape(4, 4)
    intrinsic = np.fromstring(" ".join(lines[7:10]), sep=" ", dtype=np.float64).reshape(3, 3)
    depth_tokens = lines[11].split()
    depth_min = float(depth_tokens[0])
    # Match data/dtu.py: DTU's nominal interval is enlarged by 1.06.
    depth_interval = float(depth_tokens[1]) * 1.06
    depth_max = depth_min + depth_interval * 192
    return intrinsic, extrinsic, depth_min, depth_max


def scale_intrinsic(
    intrinsic: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    sx = target_w / float(source_w)
    sy = target_h / float(source_h)
    scaled = intrinsic.copy()
    scaled[0, :] *= sx
    scaled[1, :] *= sy
    return scaled


def backproject_world(
    depth: np.ndarray,
    valid: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_world_to_camera: np.ndarray,
) -> np.ndarray:
    """Back-project z-depth into the world frame using DTU's W2C extrinsic."""
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    keep = valid & np.isfinite(depth) & (depth > 0)
    if not keep.any():
        return np.empty((0, 3), dtype=np.float32)

    z = depth[keep].astype(np.float64)
    x = (xx[keep] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy[keep] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points_camera = np.stack((x, y, z), axis=1)

    camera_to_world = np.linalg.inv(extrinsic_world_to_camera)
    points_h = np.concatenate(
        (points_camera, np.ones((len(points_camera), 1), dtype=np.float64)), axis=1
    )
    return (camera_to_world @ points_h.T).T[:, :3].astype(np.float32)


def reprojection_error_px(
    points_world: np.ndarray,
    pixel_xy: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_world_to_camera: np.ndarray,
) -> float:
    """Maximum round-trip pixel error; catches K/extrinsic convention mistakes."""
    if len(points_world) == 0:
        return 0.0
    sample = np.linspace(0, len(points_world) - 1, min(len(points_world), 4096), dtype=np.int64)
    points = points_world[sample].astype(np.float64)
    points_h = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (extrinsic_world_to_camera @ points_h.T).T[:, :3]
    uvw = (intrinsic @ camera.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return float(np.linalg.norm(uv - pixel_xy[sample], axis=1).max())


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError(f"point/color count mismatch: {len(points)} vs {len(colors)}")

    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)


def process_case(
    case: Case,
    dtu_root: Path,
    prior_root: Path,
    output_dir: Path,
    stride: int,
) -> tuple[Path, dict]:
    prior_path = prior_root / case.scan / f"prior_{case.ref:04d}_{case.light}.npz"
    gt_path = dtu_root / "Depths_raw" / case.scan / f"depth_map_{case.ref:04d}.pfm"
    mask_path = dtu_root / "Depths_raw" / case.scan / f"depth_visual_{case.ref:04d}.png"
    camera_path = dtu_root / "Cameras" / f"{case.ref:08d}_cam.txt"
    for path in (prior_path, gt_path, mask_path, camera_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(prior_path) as cache:
        prior = np.asarray(cache["depth_prior"], dtype=np.float32)
        cache_meta = {
            key: float(np.asarray(cache[key]).reshape(-1)[0])
            for key in ("sfm_scale", "sfm_valid", "pipeline_version", "num_views")
            if key in cache
        }

    gt_original = np.asarray(read_pfm(str(gt_path)), dtype=np.float32)
    mask_original = np.asarray(Image.open(mask_path)) > 10
    target_h, target_w = prior.shape
    gt = cv2.resize(gt_original, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    gt_mask = cv2.resize(
        mask_original.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    intrinsic_original, extrinsic, depth_min, depth_max = read_camera(camera_path)
    intrinsic = scale_intrinsic(intrinsic_original, gt_original.shape, prior.shape)

    # Use the exact same object pixels/rays for both clouds.  Do not constrain
    # prior depth to [depth_min, depth_max]: values outside it are precisely the
    # failure we want to see.  GT follows the same in-scene mask as training.
    gt_valid = (
        gt_mask & np.isfinite(gt) & (gt > 0) & (gt >= depth_min) & (gt <= depth_max)
    )
    prior_valid = gt_valid & np.isfinite(prior) & (prior > 0)
    if stride > 1:
        lattice = np.zeros(prior.shape, dtype=bool)
        lattice[::stride, ::stride] = True
        gt_valid &= lattice
        prior_valid &= lattice

    gt_points = backproject_world(gt, gt_valid, intrinsic, extrinsic)
    prior_points = backproject_world(prior, prior_valid, intrinsic, extrinsic)

    # Verify both point sets round-trip through the same camera. Pixel ordering
    # matches np.indices()[valid], hence the sampled xy arrays line up with points.
    yy, xx = np.indices(prior.shape)
    gt_xy = np.stack((xx[gt_valid], yy[gt_valid]), axis=1).astype(np.float64)
    prior_xy = np.stack((xx[prior_valid], yy[prior_valid]), axis=1).astype(np.float64)
    gt_reproj = reprojection_error_px(gt_points, gt_xy, intrinsic, extrinsic)
    prior_reproj = reprojection_error_px(prior_points, prior_xy, intrinsic, extrinsic)
    if max(gt_reproj, prior_reproj) > 1e-3:
        raise RuntimeError(
            f"projection round-trip failed: GT={gt_reproj:.6g}px prior={prior_reproj:.6g}px"
        )

    points = np.concatenate((gt_points, prior_points), axis=0)
    colors = np.concatenate(
        (
            np.full((len(gt_points), 3), 255, dtype=np.uint8),
            np.tile(np.array([[0, 80, 255]], dtype=np.uint8), (len(prior_points), 1)),
        ),
        axis=0,
    )
    ply_path = output_dir / f"{case.stem}_gt_white_prior_blue.ply"
    write_binary_ply(ply_path, points, colors)

    common = gt_valid & prior_valid
    error = np.abs(prior[common] - gt[common])
    stats = {
        "case": {"scan": case.scan, "ref": case.ref, "light": case.light},
        "coordinate_frame": "DTU world coordinates",
        "depth_unit": "millimetres",
        "colors": {"ground_truth": [255, 255, 255], "prior": [0, 80, 255]},
        "source_hw": list(gt_original.shape),
        "output_hw": list(prior.shape),
        "stride": stride,
        "num_gt_points": int(len(gt_points)),
        "num_prior_points": int(len(prior_points)),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "prior_abs_err_mean": float(error.mean()),
        "prior_abs_err_median": float(np.median(error)),
        "prior_abs_err_p90": float(np.quantile(error, 0.90)),
        "prior_tail_frac_8mm": float((error > 8.0).mean()),
        "gt_depth_median": float(np.median(gt[common])),
        "prior_depth_median": float(np.median(prior[common])),
        "max_reprojection_error_px": max(gt_reproj, prior_reproj),
        "intrinsic_original": intrinsic_original.tolist(),
        "intrinsic_scaled": intrinsic.tolist(),
        "extrinsic_world_to_camera": extrinsic.tolist(),
        "cache": cache_meta,
        "ply": str(ply_path),
    }
    stats_path = ply_path.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    return ply_path, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="scan:ref:light; repeat to process multiple cases (default: three audited outliers)",
    )
    parser.add_argument("--dtu-root", type=Path, default=DEFAULT_DTU_ROOT)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stride", type=int, default=1,
        help="pixel subsampling stride; 1 keeps every valid comparison pixel",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    cases = tuple(args.case) if args.case else DEFAULT_CASES
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        ply_path, stats = process_case(
            case=case,
            dtu_root=args.dtu_root,
            prior_root=args.prior_root,
            output_dir=args.output_dir,
            stride=args.stride,
        )
        print(
            f"[{case.stem}] GT={stats['num_gt_points']:,} prior={stats['num_prior_points']:,} "
            f"mean={stats['prior_abs_err_mean']:.2f}mm "
            f"median={stats['prior_abs_err_median']:.2f}mm "
            f"p90={stats['prior_abs_err_p90']:.2f}mm -> {ply_path}"
        )


if __name__ == "__main__":
    main()
