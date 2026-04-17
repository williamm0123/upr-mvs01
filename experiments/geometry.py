"""Geometry helpers shared by RGB and DINO cost-volume experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from upr_mvs.external import intrinsics_to_projection, scale_intrinsics


def build_projection_matrix(intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    projection_matrix = np.eye(4, dtype=np.float32)
    projection_matrix[:3, :4] = intrinsics @ extrinsics[:3, :4]
    return projection_matrix


def camera_center_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    return -rotation.T @ translation


def make_linear_depth_values(depth_min: float, depth_interval: float, num_depths: int) -> np.ndarray:
    return depth_min + np.arange(num_depths, dtype=np.float32) * depth_interval


def make_projection_for_feature_grid(sample: dict, feature_hw: tuple[int, int], device: torch.device | str) -> torch.Tensor:
    feature_h, feature_w = feature_hw
    image_h, image_w = sample["imgs"].shape[-2:]
    intrinsics = sample["intrinsics"].to(device=device, dtype=torch.float32)
    extrinsics = sample["extrinsics"].to(device=device, dtype=torch.float32)
    scaled_intrinsics = scale_intrinsics(
        intrinsics,
        scale_x=float(feature_w) / float(image_w),
        scale_y=float(feature_h) / float(image_h),
    )
    return intrinsics_to_projection(scaled_intrinsics, extrinsics).unsqueeze(0)


def resize_gt_to_feature_grid(sample: dict, feature_hw: tuple[int, int], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    depth_gt = F.interpolate(
        sample["depth_gt"].unsqueeze(0).to(device=device, dtype=torch.float32),
        size=feature_hw,
        mode="nearest",
    ).squeeze(1)
    mask = F.interpolate(
        sample["mask"].unsqueeze(0).to(device=device, dtype=torch.float32),
        size=feature_hw,
        mode="nearest",
    ).squeeze(1).bool()
    return depth_gt, mask
