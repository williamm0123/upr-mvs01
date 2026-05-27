from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

def save_binary_ply(points: np.ndarray, colors: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
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



def depth_stats(depth: np.ndarray) -> dict:
    """
    统计深度图有效区域的平均值、最小值、最大值。
    invalid depth = NaN / Inf
    """
    valid = np.isfinite(depth)

    values = depth[valid]

    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "valid_pixels": int(valid.sum()),
    }


def depth_difference(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    absolute: bool = True,
) -> np.ndarray:

    valid = np.isfinite(depth_a) & np.isfinite(depth_b)

    diff = np.full_like(depth_a, np.nan, dtype=np.float32)

    if absolute:
        diff[valid] = np.abs(depth_a[valid] - depth_b[valid]).astype(np.float32)
    else:
        diff[valid] = (depth_a[valid] - depth_b[valid]).astype(np.float32)

    return diff


def backproject_depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray | None = None,
) -> np.ndarray:

    H, W = depth.shape

    v, u = np.indices((H, W))

    valid = np.isfinite(depth) & (depth > 0)

    z = depth[valid].astype(np.float64)
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=1)

    if extrinsic is None:
        return points_cam.astype(np.float32)

    if extrinsic.shape == (3, 4):
        ext4 = np.eye(4, dtype=np.float64)
        ext4[:3, :4] = extrinsic
        extrinsic = ext4

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    # extrinsic 是 world -> camera:
    # X_cam = R @ X_world + t
    # 所以:
    # X_world = R.T @ (X_cam - t)
    points_world = (points_cam - t[None, :]) @ R

    return points_world.astype(np.float32)


def resize_image_and_intrinsic(image, K, target_h=378, target_w=504):
    image = np.asarray(image)
    K = np.asarray(K, dtype=np.float64).copy()

    src_h, src_w = image.shape[:2]

    scale_x = target_w / src_w
    scale_y = target_h / src_h

    image_resized = cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    K_resized = K.copy()
    K_resized[0, :] *= scale_x
    K_resized[1, :] *= scale_y

    return image_resized, K_resized




def resize_sparse_depth_and_intrinsic(sparse_depth, K, target_h=378, target_w=504):
    depth = np.asarray(sparse_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected sparse_depth shape [H, W], got {depth.shape}")

    K = np.asarray(K, dtype=np.float64).copy()
    src_h, src_w = depth.shape
    scale_x = target_w / src_w
    scale_y = target_h / src_h

    resized = np.full((target_h, target_w), np.inf, dtype=np.float32)
    valid = np.isfinite(depth)
    if valid.any():
        y, x = np.nonzero(valid)
        x_resized = np.rint((x.astype(np.float64) + 0.5) * scale_x - 0.5).astype(np.int32)
        y_resized = np.rint((y.astype(np.float64) + 0.5) * scale_y - 0.5).astype(np.int32)
        x_resized = np.clip(x_resized, 0, target_w - 1)
        y_resized = np.clip(y_resized, 0, target_h - 1)
        np.minimum.at(resized, (y_resized, x_resized), depth[valid])

    resized[~np.isfinite(resized)] = np.nan

    K_resized = K.copy()
    K_resized[0, 0] *= scale_x
    K_resized[0, 1] *= scale_x
    K_resized[1, 0] *= scale_y
    K_resized[1, 1] *= scale_y
    K_resized[0, 2] = (K[0, 2] + 0.5) * scale_x - 0.5
    K_resized[1, 2] = (K[1, 2] + 0.5) * scale_y - 0.5

    return resized, K_resized
