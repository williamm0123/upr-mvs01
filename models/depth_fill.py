import numpy as np
import torch
import os
import sys
import json
import re
from collections import deque
from pathlib import Path
import argparse
from typing import Any

import numpy as np
import cv2
from scipy import sparse
try:
    from scipy.spatial import cKDTree
except Exception:
    try:
        # fallback to KDTree if cKDTree is not available
        from scipy.spatial import KDTree as cKDTree
    except Exception:
        # last resort: provide a minimal wrapper around sklearn's NearestNeighbors
        try:
            from sklearn.neighbors import NearestNeighbors

            class cKDTree:
                def __init__(self, data):
                    self._nn = NearestNeighbors().fit(data)

                def query(self, x, k=1):
                    d, idx = self._nn.kneighbors(x, n_neighbors=k)
                    return d, idx

        except Exception:
            raise
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import cg, spsolve

from dataclasses import dataclass

from PIL import Image

try:
    from depth_anything_3.api import DepthAnything3
except ModuleNotFoundError:
    DepthAnything3 = None


# import models.sfm as sfm
# from base.config import ProjectPaths
# from data.io import read_pfm, write_pfm


@dataclass
class DepthCompletionConfig:
    output_root: Path = Path("outputs/depth_completion")

    min_depth: float = 1e-6
    max_depth: float = 2000.0

    # point cloud denoise
    point_denoise_neighbors: int = 16
    point_denoise_std_ratio: float = 2.5

    eps: float = 1e-6
    da3_is_inverse: bool = False

    min_depth: float = 1e-6
    max_depth: float = 2000.0

    anchor_scale_percentiles: tuple[float, float] = (0.5, 99.5)

    max_iters: int = 900
    edge_sigma: float = 0.08
    min_edge_weight: float = 0.03

    anchor_lambda: float = 1.5
    grad_lambda: float = 1.5
    smooth_lambda: float = 0.4

    relaxation: float = 0.85
    tolerance: float = 1e-5


@dataclass
class NormalConstraintDepthFillConfig:
    hard_keep_sparse: bool = True
    clamp_output: bool = True
    clamp_percentiles: tuple[float, float] = (0.5, 99.5)
    clamp_margin_ratio: float = 0.15

    normal_edge_weight: float = 1.0
    normal_anchor_weight: float = 100.0
    normal_guide_weight: float = 0.02
    normal_min_denom: float = 1e-4
    normal_ratio_limits: tuple[float, float] = (0.4, 2.5)
    normal_min_similarity: float = -0.2
    normal_similarity_power: float = 2.0
    normal_cg_maxiter: int = 600
    normal_cg_rtol: float = 1e-5
    normal_fallback_spsolve: bool = True

    align_trim_mad: float = 3.5
    align_min_points: int = 100


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


@dataclass
class PointCloudDenoiseConfig:
    knn: int = 40
    std_ratio: float = 2.0
    max_knn_distance_percentile: float = 99.0
    radius_filter: bool = True
    radius: float = 0.0
    radius_scale: float = 2.8
    min_radius_neighbors: int = 8
    component_filter: bool = True
    component_voxel_size: float = 0.0
    min_component_points: int = 1200
    min_component_ratio: float = 0.001


def _config_value(config: Any, name: str):
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


def sample_key_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    match = re.search(r"(scan\d+_ref\d+_light\d+)", stem)
    if not match:
        raise ValueError(f"Could not parse sample key from path: {path}")
    return match.group(1)


def load_threshold_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    threshold_map: dict[str, dict[str, Any]] = {}
    for item in manifest.get("files", []):
        key = sample_key_from_path(item.get("input") or item.get("output"))
        threshold_map[key] = item
    return threshold_map


def build_preserve_map(root: Path | None, pattern: str) -> dict[str, Path]:
    if root is None:
        return {}
    preserve_map: dict[str, Path] = {}
    for path in sorted(root.glob(pattern)):
        preserve_map[sample_key_from_path(path)] = path
    return preserve_map


def read_binary_rgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as file:
        header_lines: list[str] = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"Only binary_little_endian PLY is supported: {path}")

        vertex_count = None
        for line in header_lines:
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                break
        if vertex_count is None:
            raise ValueError(f"Could not find vertex count in PLY header: {path}")

        vertices = np.fromfile(file, dtype=PLY_DTYPE, count=vertex_count)

    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32, copy=False)
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"])).astype(np.uint8, copy=False)
    return points, colors


def query_knn(tree: cKDTree, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        return tree.query(points, k=k, workers=-1)
    except TypeError:
        return tree.query(points, k=k)


def query_radius_counts(tree: cKDTree, points: np.ndarray, radius: float) -> np.ndarray:
    try:
        return tree.query_ball_point(points, r=radius, return_length=True, workers=-1)
    except TypeError:
        neighbors = tree.query_ball_point(points, r=radius)
        return np.fromiter((len(item) for item in neighbors), dtype=np.int32, count=len(neighbors))


def robust_upper_limit(values: np.ndarray, std_ratio: float, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad
    if sigma <= 1e-12:
        sigma = float(np.std(finite))

    limits = [median + std_ratio * sigma]
    if percentile > 0:
        limits.append(float(np.percentile(finite, percentile)))
    return min(limits)


def statistical_outlier_mask(
    points: np.ndarray,
    knn: int,
    std_ratio: float,
    percentile: float,
    fixed_threshold: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    tree = cKDTree(points)
    distances, _ = query_knn(tree, points, k=knn + 1)
    mean_knn_distance = distances[:, 1:].mean(axis=1)
    threshold = (
        float(fixed_threshold)
        if fixed_threshold is not None
        else robust_upper_limit(mean_knn_distance, std_ratio, percentile)
    )
    mask = np.isfinite(mean_knn_distance) & (mean_knn_distance <= threshold)

    nn_distance = distances[:, 1]
    radius_base = float(np.median(nn_distance[np.isfinite(nn_distance)]))
    info = {
        "knn": int(knn),
        "mean_knn_threshold": threshold,
        "median_nn_distance": radius_base,
        "fixed_threshold": fixed_threshold is not None,
        "kept": int(mask.sum()),
        "removed": int((~mask).sum()),
    }
    return mask, info


def radius_outlier_mask(
    points: np.ndarray,
    radius: float,
    min_neighbors: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    tree = cKDTree(points)
    counts = query_radius_counts(tree, points, radius)
    mask = counts >= min_neighbors
    info = {
        "radius": float(radius),
        "min_neighbors": int(min_neighbors),
        "median_neighbors": float(np.median(counts)) if counts.size else 0.0,
        "kept": int(mask.sum()),
        "removed": int((~mask).sum()),
    }
    return mask, info


def voxel_component_mask(
    points: np.ndarray,
    voxel_size: float,
    min_component_points: int,
    min_component_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(points) == 0:
        return np.zeros(0, dtype=bool), {"components": 0, "kept_components": 0}

    origin = points.min(axis=0)
    voxel_coords = np.floor((points - origin) / voxel_size).astype(np.int64)
    unique_coords, inverse, voxel_counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
    coord_to_voxel = {tuple(coord.tolist()): idx for idx, coord in enumerate(unique_coords)}

    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    visited = np.zeros(len(unique_coords), dtype=bool)
    keep_voxel = np.zeros(len(unique_coords), dtype=bool)
    min_points = max(int(min_component_points), int(round(len(points) * min_component_ratio)))
    component_sizes: list[int] = []

    for start in range(len(unique_coords)):
        if visited[start]:
            continue
        queue: deque[int] = deque([start])
        visited[start] = True
        component_voxels: list[int] = []
        point_count = 0

        while queue:
            voxel_idx = queue.popleft()
            component_voxels.append(voxel_idx)
            point_count += int(voxel_counts[voxel_idx])
            coord = unique_coords[voxel_idx]

            for offset in offsets:
                neighbor_key = (int(coord[0] + offset[0]), int(coord[1] + offset[1]), int(coord[2] + offset[2]))
                neighbor_idx = coord_to_voxel.get(neighbor_key)
                if neighbor_idx is not None and not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx)

        component_sizes.append(point_count)
        if point_count >= min_points:
            keep_voxel[component_voxels] = True

    mask = keep_voxel[inverse]
    kept_components = int(sum(size >= min_points for size in component_sizes))
    info = {
        "voxel_size": float(voxel_size),
        "min_component_points": int(min_points),
        "components": len(component_sizes),
        "kept_components": kept_components,
        "largest_component_points": int(max(component_sizes)) if component_sizes else 0,
        "kept": int(mask.sum()),
        "removed": int((~mask).sum()),
    }
    return mask, info


def denoise_points(
    points: np.ndarray,
    colors: np.ndarray,
    config: Any | None = None,
    fixed_thresholds: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if config is None:
        config = PointCloudDenoiseConfig()

    original_count = len(points)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    colors = colors[finite_mask]

    info: dict[str, Any] = {
        "original_points": int(original_count),
        "finite_points": int(len(points)),
        "filters": {},
    }
    if len(points) == 0:
        info["denoised_points"] = 0
        info["kept_ratio"] = 0.0
        return points, colors, info

    threshold_filters = fixed_thresholds.get("filters", {}) if fixed_thresholds else {}
    fixed_sor = threshold_filters.get("statistical_outlier", {})
    knn = int(fixed_sor.get("knn", _config_value(config, "knn")))
    fixed_mean_knn_threshold = fixed_sor.get("mean_knn_threshold")
    sor_mask, sor_info = statistical_outlier_mask(
        points,
        knn,
        _config_value(config, "std_ratio"),
        _config_value(config, "max_knn_distance_percentile"),
        fixed_threshold=fixed_mean_knn_threshold,
    )
    points = points[sor_mask]
    colors = colors[sor_mask]
    info["filters"]["statistical_outlier"] = sor_info

    fixed_radius = threshold_filters.get("radius_outlier", {}).get("radius")
    radius = float(fixed_radius) if fixed_radius is not None else _config_value(config, "radius")
    if radius <= 0:
        radius = max(float(sor_info["median_nn_distance"]) * _config_value(config, "radius_scale"), 1e-8)

    if _config_value(config, "radius_filter") and len(points):
        min_neighbors = int(threshold_filters.get("radius_outlier", {}).get(
            "min_neighbors",
            _config_value(config, "min_radius_neighbors"),
        ))
        radius_mask, radius_info = radius_outlier_mask(points, radius, min_neighbors)
        radius_info["fixed_radius"] = fixed_radius is not None
        points = points[radius_mask]
        colors = colors[radius_mask]
        info["filters"]["radius_outlier"] = radius_info

    if _config_value(config, "component_filter") and len(points):
        fixed_component = threshold_filters.get("component", {})
        fixed_voxel_size = fixed_component.get("voxel_size")
        voxel_size = (
            float(fixed_voxel_size)
            if fixed_voxel_size is not None
            else _config_value(config, "component_voxel_size")
            if _config_value(config, "component_voxel_size") > 0
            else radius
        )
        min_component_points = int(fixed_component.get(
            "min_component_points",
            _config_value(config, "min_component_points"),
        ))
        min_component_ratio = 0.0 if fixed_component else _config_value(config, "min_component_ratio")
        component_mask, component_info = voxel_component_mask(
            points,
            voxel_size=voxel_size,
            min_component_points=min_component_points,
            min_component_ratio=min_component_ratio,
        )
        component_info["fixed_voxel_size"] = fixed_voxel_size is not None
        component_info["fixed_min_component_points"] = bool(fixed_component)
        points = points[component_mask]
        colors = colors[component_mask]
        info["filters"]["component"] = component_info

    info["denoised_points"] = int(len(points))
    info["kept_ratio"] = float(len(points) / original_count) if original_count else 0.0
    if fixed_thresholds:
        info["threshold_source"] = {
            "input": fixed_thresholds.get("input"),
            "output": fixed_thresholds.get("output"),
        }
    return points, colors, info


def denoise_pointcloud_points(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    config: PointCloudDenoiseConfig | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if colors is None:
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        denoised_points, denoised_colors, info = denoise_points(points, colors, config)
        return denoised_points, None, info
    return denoise_points(points, colors, config)



def load_da3_model(model_path, device=None):
    if DepthAnything3 is None:
        raise ModuleNotFoundError(
            "depth_anything_3 is required for DA3 depth generation. "
            "Use the uprmvs environment or install Depth-Anything-3 dependencies."
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DepthAnything3.from_pretrained(str(model_path))
    model = model.to(device).eval()

    return model, device

def generate_da3_depth_maps(
    image,
    da3_model,
):
    """
    使用DTU数据集和DA3模型生成单目深度图
    """
    ref_img = image  # 形状为 (H, W, 3)
            
    ref_img_pil = Image.fromarray(ref_img.astype('uint8'), 'RGB')
    
    with torch.inference_mode():
        prediction = da3_model.inference(
            image=[ref_img_pil],
            export_dir=None,  # 不导出任何内容，只获取预测结果
            export_format="npz"
        )
          
        pred_depth = prediction.depth[0].astype(np.float32)  # 取第一个图像的深度图
        
        # 检查是否所有的数值都是有限的 (既不是 inf，也不是 NaN)
        finite_mask = np.isfinite(pred_depth)

        if not finite_mask.all():
            if finite_mask.any():
                max_finite_depth = np.max(pred_depth[finite_mask])
            else:
                max_finite_depth = 10000.0  # 如果极端情况整张图全崩了，给个保底值
            safe_max_value = max_finite_depth 
            # 将 正无穷(inf) 替换为 safe_max_value
            pred_depth[np.isposinf(pred_depth)] = safe_max_value
            # 将 NaN 替换为 0 (或者深度图的最小值)
            pred_depth[np.isnan(pred_depth)] = 0.0
            # 将 负无穷(-inf) 替换为 0 (深度不可能是负数)
            pred_depth[np.isneginf(pred_depth)] = 0.0
            
    return pred_depth


def valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0)


def resize_dense_depth(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    if depth.shape[:2] == (height, width):
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def robust_affine_align_depth(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    target_valid: np.ndarray,
    trim_mad: float = 3.5,
    min_points: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = target_valid & np.isfinite(source_depth) & (source_depth > 0)
    if int(valid.sum()) < min_points:
        raise ValueError(f"Too few valid anchors for DA3/VGGT affine alignment: {int(valid.sum())}")

    x = source_depth[valid].astype(np.float64)
    y = target_depth[valid].astype(np.float64)
    keep = np.ones_like(x, dtype=bool)
    scale, bias = 1.0, 0.0

    for _ in range(3):
        matrix = np.stack([x[keep], np.ones(int(keep.sum()), dtype=np.float64)], axis=1)
        scale, bias = np.linalg.lstsq(matrix, y[keep], rcond=None)[0]
        residual = y - (scale * x + bias)
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - median))
        sigma = max(1.4826 * mad, 1e-8)
        new_keep = np.abs(residual - median) <= trim_mad * sigma
        if int(new_keep.sum()) < min_points or int(new_keep.sum()) == int(keep.sum()):
            break
        keep = new_keep

    aligned = (float(scale) * source_depth + float(bias)).astype(np.float32)
    return aligned, {
        "scale": float(scale),
        "bias": float(bias),
        "anchors": int(valid.sum()),
        "robust_anchors": int(keep.sum()),
    }


def camera_rays(shape_hw: tuple[int, int], intrinsic: np.ndarray) -> np.ndarray:
    height, width = shape_hw
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    return np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1).astype(np.float32)


def depth_to_camera_normals(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    rays = camera_rays(depth.shape, intrinsic)
    points = rays * depth[..., None].astype(np.float32)
    dpx = np.gradient(points, axis=1)
    dpy = np.gradient(points, axis=0)
    normals = np.cross(dpx, dpy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    invalid = ~np.isfinite(norm).squeeze(-1) | (norm.squeeze(-1) < 1e-8)
    normals = normals / np.maximum(norm, 1e-8)
    normals[invalid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    flip = normals[..., 2] < 0
    normals[flip] *= -1.0
    return normals.astype(np.float32)


def _normal_edge_terms(
    index_a: np.ndarray,
    index_b: np.ndarray,
    rays_a: np.ndarray,
    rays_b: np.ndarray,
    normals_a: np.ndarray,
    normals_b: np.ndarray,
    config: NormalConstraintDepthFillConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal = normals_a + normals_b
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(normal_norm, 1e-8)

    similarity = np.sum(normals_a * normals_b, axis=-1)
    numerator = np.sum(normal * rays_a, axis=-1)
    denominator = np.sum(normal * rays_b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator

    ratio_lo, ratio_hi = config.normal_ratio_limits
    keep = (
        np.isfinite(ratio)
        & np.isfinite(similarity)
        & (np.abs(numerator) >= config.normal_min_denom)
        & (np.abs(denominator) >= config.normal_min_denom)
        & (ratio >= ratio_lo)
        & (ratio <= ratio_hi)
        & (similarity >= config.normal_min_similarity)
    )
    if not keep.any():
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f

    sim01 = np.clip(similarity[keep], 0.0, 1.0)
    weights = config.normal_edge_weight * np.maximum(sim01, 1e-3) ** config.normal_similarity_power
    return index_a[keep], index_b[keep], ratio[keep].astype(np.float32), weights.astype(np.float32)


def _add_sparse_rows(
    row_parts: list[np.ndarray],
    col_parts: list[np.ndarray],
    data_parts: list[np.ndarray],
    rhs_parts: list[np.ndarray],
    row_start: int,
    cols: np.ndarray,
    values: np.ndarray,
    rhs: np.ndarray,
) -> int:
    n_rows = int(rhs.size)
    rows = np.arange(row_start, row_start + n_rows, dtype=np.int64)
    row_parts.append(rows)
    col_parts.append(cols.astype(np.int64))
    data_parts.append(values.astype(np.float32))
    rhs_parts.append(rhs.astype(np.float32))
    return row_start + n_rows


def _add_normal_edges(
    row_parts: list[np.ndarray],
    col_parts: list[np.ndarray],
    data_parts: list[np.ndarray],
    rhs_parts: list[np.ndarray],
    row_start: int,
    index_a: np.ndarray,
    index_b: np.ndarray,
    ratio: np.ndarray,
    weights: np.ndarray,
) -> int:
    n_edges = int(ratio.size)
    if n_edges == 0:
        return row_start
    rows = np.arange(row_start, row_start + n_edges, dtype=np.int64)
    weight = np.sqrt(np.maximum(weights, 1e-8)).astype(np.float32)
    row_parts.append(np.repeat(rows, 2))
    col_parts.append(np.stack((index_a, index_b), axis=1).reshape(-1).astype(np.int64))
    data_parts.append(np.stack((-weight * ratio, weight), axis=1).reshape(-1).astype(np.float32))
    rhs_parts.append(np.zeros(n_edges, dtype=np.float32))
    return row_start + n_edges


def _solve_normal_equations(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    initial: np.ndarray,
    config: NormalConstraintDepthFillConfig,
) -> tuple[np.ndarray, int, str]:
    normal_matrix = matrix.T @ matrix
    normal_rhs = matrix.T @ rhs
    try:
        solution, cg_info = cg(
            normal_matrix,
            normal_rhs,
            x0=initial,
            rtol=config.normal_cg_rtol,
            atol=0.0,
            maxiter=config.normal_cg_maxiter,
        )
    except TypeError:
        solution, cg_info = cg(
            normal_matrix,
            normal_rhs,
            x0=initial,
            tol=config.normal_cg_rtol,
            maxiter=config.normal_cg_maxiter,
        )

    solver = "cg"
    if cg_info != 0 and config.normal_fallback_spsolve:
        solution = spsolve(normal_matrix.tocsc(), normal_rhs)
        solver = "spsolve_after_cg"
    return np.asarray(solution, dtype=np.float32), int(cg_info), solver


def clamp_to_anchor_range(
    depth: np.ndarray,
    anchors: np.ndarray,
    valid: np.ndarray,
    config: NormalConstraintDepthFillConfig,
) -> np.ndarray:
    if not config.clamp_output or not valid.any():
        return depth.astype(np.float32)
    lo, hi = np.percentile(anchors[valid], config.clamp_percentiles)
    margin = config.clamp_margin_ratio * max(float(hi - lo), 1e-6)
    return np.clip(depth, float(lo - margin), float(hi + margin)).astype(np.float32)


def fill_depth_with_normal_constraints(
    sparse_depth: np.ndarray,
    aligned_guide_depth: np.ndarray,
    normals: np.ndarray,
    intrinsic: np.ndarray,
    valid: np.ndarray,
    config: NormalConstraintDepthFillConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config is None:
        config = NormalConstraintDepthFillConfig()

    height, width = sparse_depth.shape
    n_pixels = height * width
    rays = camera_rays((height, width), intrinsic)
    pixel_ids = np.arange(n_pixels, dtype=np.int64).reshape(height, width)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    rhs_parts: list[np.ndarray] = []
    row_count = 0

    guide_valid = np.isfinite(aligned_guide_depth) & (aligned_guide_depth > 0)
    if config.normal_guide_weight > 0 and guide_valid.any():
        guide_ids = pixel_ids[guide_valid].reshape(-1)
        guide_w = np.sqrt(config.normal_guide_weight)
        row_count = _add_sparse_rows(
            row_parts,
            col_parts,
            data_parts,
            rhs_parts,
            row_count,
            guide_ids,
            np.full(guide_ids.shape, guide_w, dtype=np.float32),
            guide_w * aligned_guide_depth[guide_valid].reshape(-1),
        )

    anchor_ids = pixel_ids[valid].reshape(-1)
    anchor_w = np.sqrt(config.normal_anchor_weight)
    row_count = _add_sparse_rows(
        row_parts,
        col_parts,
        data_parts,
        rhs_parts,
        row_count,
        anchor_ids,
        np.full(anchor_ids.shape, anchor_w, dtype=np.float32),
        anchor_w * sparse_depth[valid].reshape(-1),
    )

    horizontal = _normal_edge_terms(
        pixel_ids[:, :-1].reshape(-1),
        pixel_ids[:, 1:].reshape(-1),
        rays[:, :-1].reshape(-1, 3),
        rays[:, 1:].reshape(-1, 3),
        normals[:, :-1].reshape(-1, 3),
        normals[:, 1:].reshape(-1, 3),
        config,
    )
    row_count = _add_normal_edges(row_parts, col_parts, data_parts, rhs_parts, row_count, *horizontal)

    vertical = _normal_edge_terms(
        pixel_ids[:-1, :].reshape(-1),
        pixel_ids[1:, :].reshape(-1),
        rays[:-1, :].reshape(-1, 3),
        rays[1:, :].reshape(-1, 3),
        normals[:-1, :].reshape(-1, 3),
        normals[1:, :].reshape(-1, 3),
        config,
    )
    row_count = _add_normal_edges(row_parts, col_parts, data_parts, rhs_parts, row_count, *vertical)

    matrix = sparse.coo_matrix(
        (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(row_count, n_pixels),
    ).tocsr()
    rhs = np.concatenate(rhs_parts)
    initial = np.where(valid, sparse_depth, aligned_guide_depth).astype(np.float32).reshape(-1)
    solution, cg_info, solver = _solve_normal_equations(matrix, rhs, initial, config)

    filled = solution.reshape(height, width)
    if config.hard_keep_sparse:
        filled[valid] = sparse_depth[valid]
    filled = clamp_to_anchor_range(filled, sparse_depth, valid, config)
    return filled.astype(np.float32), {
        "solver": solver,
        "cg_info": int(cg_info),
        "rows": int(row_count),
        "cols": int(n_pixels),
        "horizontal_edges": int(horizontal[2].size),
        "vertical_edges": int(vertical[2].size),
        "anchor_pixels": int(valid.sum()),
        "guide_pixels": int(guide_valid.sum()),
    }


def fill_vggt_depth_by_da3_normals(
    sparse_depth: np.ndarray,
    da3_depth: np.ndarray,
    intrinsic: np.ndarray,
    valid: np.ndarray | None = None,
    config: NormalConstraintDepthFillConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Complete a sparse/projected VGGT depth map using DA3 normals as constraints."""
    if config is None:
        config = NormalConstraintDepthFillConfig()

    sparse_depth = sparse_depth.astype(np.float32)
    da3_depth = resize_dense_depth(da3_depth, sparse_depth.shape)
    valid = valid_depth_mask(sparse_depth) if valid is None else valid.astype(bool)

    aligned_da3, align_info = robust_affine_align_depth(
        da3_depth,
        sparse_depth,
        valid,
        trim_mad=config.align_trim_mad,
        min_points=config.align_min_points,
    )
    da3_normals = depth_to_camera_normals(aligned_da3, intrinsic)
    filled, fill_info = fill_depth_with_normal_constraints(
        sparse_depth=sparse_depth,
        aligned_guide_depth=aligned_da3,
        normals=da3_normals,
        intrinsic=intrinsic,
        valid=valid,
        config=config,
    )
    info = {
        "da3_affine": align_info,
        "normal_fill": fill_info,
        "input_valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "filled_valid_pixels": int(valid_depth_mask(filled).sum()),
    }
    return filled.astype(np.float32), info
