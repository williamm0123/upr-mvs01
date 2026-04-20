"""Build an unoptimized point cloud from DA3 mono depth over one DTU scan."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from data.dtu import image_scan_dir_name
from data.io import read_camera_file
from experiments.depth_anything_v3 import (
    DA3VisualizationConfig,
    depth_stats,
    load_da3_mono_model,
    predict_da3_depth,
    visualize_depth,
)
from experiments.pointcloud import depth_to_world_points
from upr_mvs.config import ProjectPaths


RECTIFIED_RE = re.compile(r"^rect_(?P<view>\d{3})_(?P<light>\d|max)(?:_r\d+)?\.png$")


@dataclass(frozen=True)
class DA3ScanPointCloudConfig:
    """Configuration for projecting DA3 mono depth from a full DTU scan."""

    da3: DA3VisualizationConfig = DA3VisualizationConfig()
    point_stride: int = 8
    depth_scale: float = 1.0
    depth_scale_mode: str = "constant"
    include_max_images: bool = True
    light_id: int | None = None
    save_depths: bool = False
    preview_max_points: int = 180_000
    random_seed: int = 20260418


def parse_rectified_image_name(path: str | Path) -> tuple[int, str]:
    match = RECTIFIED_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"Unsupported DTU rectified image name: {path}")
    view_id = int(match.group("view")) - 1
    light = match.group("light")
    return view_id, light


def first_scan_name(list_path: str | Path) -> str:
    for line in Path(list_path).read_text().splitlines():
        scan = line.strip()
        if scan:
            return scan
    raise ValueError(f"No scans found in list file: {list_path}")


def collect_scan_images(
    root_dir: str | Path,
    scan_name: str,
    split: str,
    image_dir: str,
    include_max_images: bool,
    light_id: int | None,
) -> list[Path]:
    scan_dir = Path(root_dir) / image_dir / image_scan_dir_name(scan_name, split, image_dir)
    if not scan_dir.is_dir():
        raise FileNotFoundError(f"Missing DTU scan image directory: {scan_dir}")

    images = []
    for path in sorted(scan_dir.glob("rect_*.png")):
        try:
            _, light = parse_rectified_image_name(path)
        except ValueError:
            continue
        if light == "max" and not include_max_images:
            continue
        if light_id is not None and light != str(light_id):
            continue
        images.append(path)
    if not images:
        raise FileNotFoundError(f"No rectified images selected under: {scan_dir}")
    return images


def scale_intrinsics_numpy(intrinsics: np.ndarray, original_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    original_h, original_w = original_hw
    target_h, target_w = target_hw
    scaled = intrinsics.astype(np.float32).copy()
    scaled[0, :] *= float(target_w) / float(original_w)
    scaled[1, :] *= float(target_h) / float(original_h)
    return scaled


def strided_valid_mask(shape: tuple[int, int], stride: int) -> torch.Tensor:
    height, width = shape
    mask = torch.zeros((height, width), dtype=torch.bool)
    mask[:: max(1, stride), :: max(1, stride)] = True
    return mask


def resolve_depth_scale(
    depth_scale_mode: str,
    base_depth_scale: float,
    scaled_intrinsics: np.ndarray,
    model_name: str,
) -> tuple[float, str]:
    """Return the scalar needed before projecting depth into the DTU camera frame."""

    mode = depth_scale_mode
    if mode == "auto":
        mode = "metric_focal_300" if "metric" in model_name.lower() else "constant"

    if mode == "constant":
        return float(base_depth_scale), mode
    if mode == "metric_focal_300":
        focal = float((scaled_intrinsics[0, 0] + scaled_intrinsics[1, 1]) * 0.5)
        return float(base_depth_scale) * focal / 300.0, mode
    raise ValueError(f"Unsupported depth_scale_mode: {depth_scale_mode}")


def save_binary_ply(points: np.ndarray, colors: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same number of rows")

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    with output_path.open("wb") as file:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {points.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        file.write(header.encode("ascii"))
        vertices.tofile(file)
    return output_path


def save_pointcloud_preview(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: str | Path,
    max_points: int,
    random_seed: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points_vis = points[idx]
        colors_vis = colors[idx]
    else:
        points_vis = points
        colors_vis = colors

    colors_vis = np.asarray(colors_vis, dtype=np.float32) / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    pairs = [(0, 1, "X/Y"), (0, 2, "X/Z"), (1, 2, "Y/Z")]
    for ax, (a, b, title) in zip(axes, pairs):
        ax.scatter(points_vis[:, a], points_vis[:, b], c=colors_vis, s=0.12, linewidths=0, alpha=0.55)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.suptitle(f"DA3 raw-depth point cloud preview, sampled {points_vis.shape[0]} / {points.shape[0]} points")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_da3_first_scan_pointcloud(
    paths: ProjectPaths | None = None,
    config: DA3ScanPointCloudConfig | None = None,
    output_root: str | Path = "outputs/da3_first_scan_pointcloud",
    device: str | torch.device | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    config = config or DA3ScanPointCloudConfig()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scan_name = first_scan_name(paths.dtu_list_path)
    image_paths = collect_scan_images(
        paths.dtu_train_root,
        scan_name,
        config.da3.split,
        config.da3.image_dir,
        include_max_images=config.include_max_images,
        light_id=config.light_id,
    )
    model, load_info = load_da3_mono_model(config.da3.model_dir, device=device)

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    rows = []
    depth_dir = output_root / scan_name / "depths"
    preview_dir = output_root / scan_name / "depth_preview"
    if config.save_depths:
        depth_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

    for image_index, image_path in enumerate(image_paths):
        view_id, light = parse_rectified_image_name(image_path)
        camera_path = Path(paths.dtu_train_root) / "Cameras/train" / f"{view_id:08d}_cam.txt"
        intrinsics, extrinsics, _, _ = read_camera_file(str(camera_path))

        rgb, depth, _ = predict_da3_depth(
            model,
            image_path,
            process_res=config.da3.process_res,
            process_res_method=config.da3.process_res_method,
        )

        with Image.open(image_path) as image:
            original_hw = (image.height, image.width)
        target_hw = depth.shape
        scaled_intrinsics = scale_intrinsics_numpy(intrinsics, original_hw, target_hw)
        effective_depth_scale, resolved_scale_mode = resolve_depth_scale(
            config.depth_scale_mode,
            config.depth_scale,
            scaled_intrinsics,
            str(load_info.get("model_name", "")),
        )
        if effective_depth_scale != 1.0:
            depth = depth * effective_depth_scale
        valid_mask = strided_valid_mask(target_hw, config.point_stride)

        points, colors = depth_to_world_points(
            torch.from_numpy(depth),
            torch.from_numpy(scaled_intrinsics),
            torch.from_numpy(extrinsics.astype(np.float32)),
            valid_mask=valid_mask,
            colors=rgb,
        )
        all_points.append(points)
        all_colors.append(colors if colors is not None else np.full((points.shape[0], 3), 255, dtype=np.uint8))

        if config.save_depths:
            depth_path = depth_dir / f"{image_path.stem}_depth.npy"
            depth_vis_path = preview_dir / f"{image_path.stem}_depth.png"
            np.save(depth_path, depth.astype(np.float32))
            imageio.imwrite(depth_vis_path, visualize_depth(depth, cmap="Spectral"))
        else:
            depth_path = ""
            depth_vis_path = ""

        row = {
            "image_index": image_index,
            "scan_name": scan_name,
            "image_name": image_path.name,
            "view_id": view_id,
            "light": light,
            "depth_shape": str(tuple(depth.shape)),
            "original_shape": str(original_hw),
            "point_stride": config.point_stride,
            "num_points": int(points.shape[0]),
            "depth_scale": float(config.depth_scale),
            "depth_scale_mode": resolved_scale_mode,
            "effective_depth_scale": effective_depth_scale,
            "depth_path": str(depth_path),
            "depth_vis_path": str(depth_vis_path),
            **depth_stats(depth),
        }
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    point_cloud_dir = output_root / scan_name / "pointclouds"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)
    points_all = np.concatenate(all_points, axis=0)
    colors_all = np.concatenate(all_colors, axis=0)
    model_slug = str(load_info.get("model_name", "da3")).replace("/", "_").replace("-", "_")
    scale_slug = config.depth_scale_mode.replace("/", "_").replace("-", "_")
    pointcloud_path = (
        point_cloud_dir
        / f"{scan_name}_{model_slug}_{scale_slug}_all_images_stride{config.point_stride}.ply"
    )
    save_binary_ply(points_all, colors_all, pointcloud_path)
    preview_path = (
        point_cloud_dir
        / f"{scan_name}_{model_slug}_{scale_slug}_all_images_stride{config.point_stride}_preview.png"
    )
    save_pointcloud_preview(
        points_all,
        colors_all,
        preview_path,
        max_points=config.preview_max_points,
        random_seed=config.random_seed,
    )

    summary_df = pd.DataFrame(rows)
    summary_csv_path = output_root / scan_name / f"{scan_name}_{model_slug}_{scale_slug}_pointcloud_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    return {
        "scan_name": scan_name,
        "num_images": len(image_paths),
        "num_points": int(points_all.shape[0]),
        "pointcloud_path": pointcloud_path,
        "preview_path": preview_path,
        "summary_csv_path": summary_csv_path,
        "summary_df": summary_df,
        "load_info": load_info,
        "output_root": output_root / scan_name,
    }
