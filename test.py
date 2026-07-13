from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from base.config import build_mvs_config
from data.dtu import DTUMVSDataset
import models.normal_fill as NF
import models.norm_fill as P


KINDS = (
    "input_images",
    "vggt_depth",
    "pointcloud",
    "denoised_pointcloud",
    "denoised_depth",
    "da3_depth",
    "normals",
    "filled_depth",
    "filled_pointcloud",
    "comparison",
)

COMPARE_PANELS = (
    ("input views", "input_images"),
    ("vggt depth", "vggt_depth"),
    ("denoised depth", "denoised_depth"),
    ("da3 depth", "da3_depth"),
    ("da3 normals", "normals"),
    ("filled depth", "filled_depth"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Test compact VGGT + DA3 depth completion.")
    parser.add_argument("--profile", choices=["local", "umhpc"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default="outputs/vggt_test")
    parser.add_argument("--max-scans", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=5)
    parser.add_argument("--image-mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--conf-percentile", type=float, default=10.0)
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--vggt-weights", default=None)
    parser.add_argument("--da3-weights", default=None)
    return parser.parse_args()


def first_ref_dataset(cfg, nviews: int) -> DTUMVSDataset:
    dataset = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=cfg.paths.test_list_file,
        nviews=nviews,
        ndepths=192,
        mode="test",
        resize_scale=1.0,
    )
    selected = []
    seen = set()
    for scan, light_idx, ref_view, src_views in dataset.metas:
        if scan in seen:
            continue
        seen.add(scan)
        selected.append((scan, light_idx, ref_view, src_views))
    dataset.metas = selected
    return dataset


def make_dirs(root: Path) -> None:
    for kind in KINDS:
        (root / kind).mkdir(parents=True, exist_ok=True)


def save_depth(root: Path, kind: str, key: str, depth: np.ndarray) -> None:
    out_dir = root / kind
    depth = depth.astype(np.float32)
    np.save(out_dir / f"{key}.npy", depth)
    cv2.imwrite(str(out_dir / f"{key}.png"), depth_vis_image(depth))


def depth_vis_image(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    image = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], (1.0, 99.0))
        hi = hi if hi > lo else lo + 1e-6
        gray = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        image = cv2.applyColorMap((gray * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[~valid] = 0
    return image


def save_image_grid(root: Path, key: str, images: np.ndarray, max_cols: int = 5) -> None:
    out_dir = root / "input_images"
    images = images.astype(np.uint8)
    n, h, w, c = images.shape
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    grid = np.full((rows * h, cols * w, c), 255, dtype=np.uint8)
    for i, image in enumerate(images):
        y = (i // cols) * h
        x = (i % cols) * w
        grid[y : y + h, x : x + w] = image
    cv2.imwrite(str(out_dir / f"{key}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    np.save(out_dir / f"{key}.npy", images)


def save_normals(root: Path, key: str, normals: np.ndarray) -> None:
    out_dir = root / "normals"
    normals = normals.astype(np.float32)
    np.save(out_dir / f"{key}.npy", normals)
    image = np.clip((normals * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{key}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def labeled_panel(image: np.ndarray, label: str, size: tuple[int, int] = (360, 300)) -> np.ndarray:
    width, height = size
    label_h = 30
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)

    content_h = height - label_h
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), content_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    x = (width - new_w) // 2
    y = label_h + (content_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def save_comparison_grid(root: Path, key: str) -> None:
    panels = []
    for label, kind in COMPARE_PANELS:
        image = cv2.imread(str(root / kind / f"{key}.png"), cv2.IMREAD_COLOR)
        if image is not None:
            panels.append(labeled_panel(image, label))
    if not panels:
        return

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    blank = np.full_like(panels[0], 255)
    while len(panels) < rows * cols:
        panels.append(blank.copy())
    grid_rows = [cv2.hconcat(panels[i * cols : (i + 1) * cols]) for i in range(rows)]
    cv2.imwrite(str(root / "comparison" / f"{key}.png"), cv2.vconcat(grid_rows))


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.astype(np.float32)
    colors = colors.astype(np.uint8)
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def save_cloud(root: Path, kind: str, key: str, points: np.ndarray, colors: np.ndarray) -> None:
    out_dir = root / kind
    np.save(out_dir / f"{key}.npy", points.astype(np.float32))
    save_ply(out_dir / f"{key}.ply", points, colors)


def save_filled_pointcloud(root: Path, key: str, depth: np.ndarray, image: np.ndarray, intrinsic: np.ndarray) -> int:
    out_dir = root / "filled_pointcloud"
    valid = np.isfinite(depth) & (depth > 0)
    points = (NF.camera_rays(depth.shape, intrinsic) * depth[..., None].astype(np.float32))[valid]
    colors = image[valid]
    save_ply(out_dir / f"{key}.ply", points, colors)
    cv2.imwrite(str(out_dir / f"{key}.png"), depth_vis_image(depth))
    return int(len(points))


def run_one(sample: dict, key: str, vggt_model, da3_model, device: torch.device, args, root: Path) -> dict:
    inputs = P._preprocess_vggt_inputs(sample["images"], sample["intrinsics"], mode=args.image_mode)
    save_image_grid(root, key, inputs.images_uint8)
    pred = P._run_vggt(vggt_model, inputs.images_chw, device)
    pred["images_uint8"] = inputs.images_uint8
    raw_points, raw_colors, point_info = P.vggt_prediction_to_pointcloud(
        pred,
        conf_percentile=args.conf_percentile,
        min_conf=args.min_conf,
    )
    denoised_points, denoised_colors, point_denoise_info = P.denoise_pointcloud_points(raw_points, raw_colors)

    vggt_depth = pred["depth"][0]
    image = inputs.images_uint8[0]
    intrinsic = pred["intrinsic"][0]

    sparse_depth, valid, projection_info = P.project_denoised_pointcloud_to_depth(
        denoised_points,
        pred,
        view_idx=0,
        splat_radius=args.splat_radius,
    )
    da3_depth = P._da3_depth(image, da3_model, target_hw=vggt_depth.shape)

    fill_cfg = NF.DepthFillConfig()
    aligned_da3, align_info = NF.robust_affine_align_depth(
        da3_depth,
        sparse_depth,
        valid,
        trim_mad=fill_cfg.align_trim_mad,
        min_points=fill_cfg.align_min_points,
    )
    normals = NF.depth_to_camera_normals(aligned_da3, intrinsic)
    filled_depth, fill_info = NF.fill_depth_with_normal_constraints(
        sparse_depth,
        aligned_da3,
        normals,
        intrinsic,
        valid,
        fill_cfg,
    )

    save_depth(root, "vggt_depth", key, vggt_depth)
    save_cloud(root, "pointcloud", key, raw_points, raw_colors)
    save_cloud(root, "denoised_pointcloud", key, denoised_points, denoised_colors)
    save_depth(root, "denoised_depth", key, sparse_depth)
    save_depth(root, "da3_depth", key, da3_depth)
    save_normals(root, key, normals)
    save_depth(root, "filled_depth", key, filled_depth)
    filled_pointcloud_points = save_filled_pointcloud(root, key, filled_depth, image, intrinsic)
    save_comparison_grid(root, key)

    return {
        "key": key,
        "raw_points": int(len(raw_points)),
        "denoised_points": int(len(denoised_points)),
        "denoised_depth_pixels": int(valid.sum()),
        "filled_depth_pixels": int(NF.valid_depth_mask(filled_depth).sum()),
        "filled_pointcloud_points": filled_pointcloud_points,
        "point_selection": point_info,
        "pointcloud_denoise": point_denoise_info,
        "projection": projection_info,
        "da3_align": align_info,
        "normal_fill": fill_info,
    }


def main() -> None:
    args = parse_args()
    cfg = build_mvs_config(profile=args.profile)
    root = Path(args.output_root)
    make_dirs(root)

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vggt_weights = Path(args.vggt_weights) if args.vggt_weights else Path(cfg.paths.vggt_weights_path)
    da3_weights = Path(args.da3_weights) if args.da3_weights else Path(cfg.paths.da3_weights_file)

    dataset = first_ref_dataset(cfg, args.num_views)
    if args.max_scans > 0:
        dataset.metas = dataset.metas[: args.max_scans]

    print(f"[test] scans={len(dataset.metas)} output={root} device={device}")
    vggt_model = P.load_vggt_model(vggt_weights, device)
    da3_model = P.load_da3_model(da3_weights, device)

    records = []
    for idx, sample in enumerate(dataset):
        scan, light_idx, ref_view, _src_views = dataset.metas[idx]
        key = f"{scan}_light{light_idx}_ref{ref_view:03d}"
        print(f"[test] {idx + 1}/{len(dataset)} {key}", flush=True)
        records.append(run_one(sample, key, vggt_model, da3_model, device, args, root))

    summary = {
        "output_root": str(root),
        "num_views": args.num_views,
        "image_mode": args.image_mode,
        "splat_radius": args.splat_radius,
        "fill_config": asdict(NF.DepthFillConfig()),
        "records": records,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[test] wrote {root / 'summary.json'}")


if __name__ == "__main__":
    main()
