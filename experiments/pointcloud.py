"""Depth-map to point-cloud export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from upr_mvs.external import scale_intrinsics


def ref_image_colors_for_grid(sample: dict, feature_hw: tuple[int, int]) -> np.ndarray:
    """Resize the reference RGB image to the feature grid and return uint8 colors."""
    image = sample["imgs"][0].unsqueeze(0).float()
    resized = F.interpolate(image, size=feature_hw, mode="bilinear", align_corners=False)
    colors = resized[0].permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(colors, 0, 255).astype(np.uint8)


def scaled_ref_camera_for_grid(
    sample: dict,
    feature_hw: tuple[int, int],
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reference-view intrinsics scaled to feature_hw and original extrinsics."""
    feature_h, feature_w = feature_hw
    image_h, image_w = sample["imgs"].shape[-2:]
    intrinsics = sample["intrinsics"][0].to(device=device, dtype=torch.float32)
    extrinsics = sample["extrinsics"][0].to(device=device, dtype=torch.float32)
    scaled_intrinsics = scale_intrinsics(
        intrinsics.unsqueeze(0),
        scale_x=float(feature_w) / float(image_w),
        scale_y=float(feature_h) / float(image_h),
    ).squeeze(0)
    return scaled_intrinsics, extrinsics


def depth_to_world_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    colors: np.ndarray | None = None,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project a reference-view depth map to world-space points.

    DTU camera extrinsics are world-to-camera, so points are transformed by
    inverse(extrinsics) after pixel-to-camera back-projection.
    """
    if depth.ndim == 3:
        depth = depth.squeeze(0)
    depth = depth.detach().cpu().float()
    height, width = depth.shape

    if valid_mask is None:
        valid = torch.ones_like(depth, dtype=torch.bool)
    else:
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.squeeze(0)
        valid = valid_mask.detach().cpu().bool()
    valid = valid & torch.isfinite(depth) & (depth > min_depth)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )

    intrinsics_cpu = intrinsics.detach().cpu().float()
    fx = intrinsics_cpu[0, 0].clamp_min(1e-6)
    fy = intrinsics_cpu[1, 1].clamp_min(1e-6)
    cx = intrinsics_cpu[0, 2]
    cy = intrinsics_cpu[1, 2]

    z = depth[valid]
    x = (x_coords[valid] - cx) * z / fx
    y = (y_coords[valid] - cy) * z / fy
    ones = torch.ones_like(z)
    camera_points = torch.stack([x, y, z, ones], dim=0)

    world_from_camera = torch.linalg.inv(extrinsics.detach().cpu().float())
    world_points = (world_from_camera @ camera_points)[:3].T.contiguous().numpy()

    point_colors = None
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape[:2] != (height, width):
            raise ValueError(f"colors shape {colors.shape[:2]} does not match depth shape {(height, width)}")
        point_colors = colors[valid.numpy()].astype(np.uint8)

    return world_points.astype(np.float32), point_colors


def save_ply(points: np.ndarray, output_path: str | Path, colors: np.ndarray | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)

    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape[0] != points.shape[0]:
            raise ValueError("colors and points must have the same number of rows")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        if colors is not None:
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
        file.write("end_header\n")

        if colors is None:
            for x, y, z in points:
                file.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for (x, y, z), (red, green, blue) in zip(points, colors):
                file.write(f"{x:.6f} {y:.6f} {z:.6f} {int(red)} {int(green)} {int(blue)}\n")

    return output_path


def save_depth_point_cloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    output_path: str | Path,
    valid_mask: torch.Tensor | None = None,
    colors: np.ndarray | None = None,
) -> tuple[Path, int]:
    points, point_colors = depth_to_world_points(
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        valid_mask=valid_mask,
        colors=colors,
    )
    path = save_ply(points, output_path, point_colors)
    return path, int(points.shape[0])


def depth_to_pixel_depth_points(
    depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    colors: np.ndarray | None = None,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a depth map to an image-space point cloud: X=u, Y=v, Z=depth."""
    if depth.ndim == 3:
        depth = depth.squeeze(0)
    depth = depth.detach().cpu().float()
    height, width = depth.shape

    if valid_mask is None:
        valid = torch.ones_like(depth, dtype=torch.bool)
    else:
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.squeeze(0)
        valid = valid_mask.detach().cpu().bool()
    valid = valid & torch.isfinite(depth) & (depth > min_depth)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack([x_coords[valid], y_coords[valid], depth[valid]], dim=1).numpy()

    point_colors = None
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape[:2] != (height, width):
            raise ValueError(f"colors shape {colors.shape[:2]} does not match depth shape {(height, width)}")
        point_colors = colors[valid.numpy()].astype(np.uint8)

    return points.astype(np.float32), point_colors


def save_pixel_depth_point_cloud(
    depth: torch.Tensor,
    output_path: str | Path,
    valid_mask: torch.Tensor | None = None,
    colors: np.ndarray | None = None,
) -> tuple[Path, int]:
    points, point_colors = depth_to_pixel_depth_points(
        depth=depth,
        valid_mask=valid_mask,
        colors=colors,
    )
    path = save_ply(points, output_path, point_colors)
    return path, int(points.shape[0])
