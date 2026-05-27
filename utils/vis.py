from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from models.general import backproject_depth_to_points, save_binary_ply


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def depth_to_colormap(depth: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    d = depth.copy().astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    if vmin is None:
        vmin = float(np.percentile(d[valid], 2))
    if vmax is None:
        vmax = float(np.percentile(d[valid], 98))
    d = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    d8 = (d * 255).astype(np.uint8)
    color = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def save_depth_vis(depth: torch.Tensor | np.ndarray, path: str | Path, vmin: float | None = None, vmax: float | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    color = depth_to_colormap(_to_numpy(depth), vmin, vmax)
    cv2.imwrite(str(path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    return path


def save_image(image: torch.Tensor | np.ndarray, path: str | Path) -> Path:
    arr = _to_numpy(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return path


def save_feature_vis(features: torch.Tensor | np.ndarray, path: str | Path, num_channels: int = 3) -> Path:
    feat = _to_numpy(features)
    if feat.ndim == 4:
        feat = feat[0]
    C = feat.shape[0]
    take = min(num_channels, C)
    vis = feat[:take]
    vis = (vis - vis.min()) / max(vis.max() - vis.min(), 1e-6)
    vis = (vis * 255).astype(np.uint8)
    if take == 1:
        vis = np.repeat(vis, 3, axis=0)
    elif take == 2:
        vis = np.concatenate([vis, np.zeros_like(vis[:1])], axis=0)
    vis = vis.transpose(1, 2, 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return path


def save_pointcloud_from_depth(
    depth: torch.Tensor | np.ndarray,
    image: torch.Tensor | np.ndarray,
    K: torch.Tensor | np.ndarray,
    extrinsic: torch.Tensor | np.ndarray | None,
    path: str | Path,
    mask: torch.Tensor | np.ndarray | None = None,
) -> Path:
    d = _to_numpy(depth)
    K_np = _to_numpy(K).astype(np.float64)
    img = _to_numpy(image)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = (img.clip(0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    d_in = d.copy()
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        d_in[~m] = np.nan
    ext = None if extrinsic is None else _to_numpy(extrinsic).astype(np.float64)
    points = backproject_depth_to_points(d_in, K_np, ext)
    valid = np.isfinite(d_in) & (d_in > 0)
    colors = img[valid].reshape(-1, 3)
    n = min(len(points), len(colors))
    return save_binary_ply(points[:n], colors[:n], path)
