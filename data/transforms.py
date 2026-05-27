from __future__ import annotations

import numpy as np

from .camera_utils import downsample_depth, downsample_mask


def make_multiscale_depth(
    depth: np.ndarray,
    strides: tuple[int, ...] = (4, 8, 16),
) -> dict[int, np.ndarray]:
    return {s: downsample_depth(depth, s) for s in strides}


def make_multiscale_mask(
    mask: np.ndarray,
    strides: tuple[int, ...] = (4, 8, 16),
) -> dict[int, np.ndarray]:
    return {s: downsample_mask(mask, s) for s in strides}


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    return (image - mean) / std


def to_chw(image: np.ndarray) -> np.ndarray:
    return image.transpose(2, 0, 1).astype(np.float32)
