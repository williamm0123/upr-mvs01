from __future__ import annotations

import cv2
import numpy as np


def scale_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 0] *= scale_x
    out[0, 2] = (out[0, 2] + 0.5) * scale_x - 0.5
    out[1, 1] *= scale_y
    out[1, 2] = (out[1, 2] + 0.5) * scale_y - 0.5
    out[0, 1] *= scale_x
    out[1, 0] *= scale_y
    return out


def crop_intrinsics(K: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray:
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 2] -= crop_x
    out[1, 2] -= crop_y
    return out


def resize_and_crop_image(
    image: np.ndarray,
    K: np.ndarray,
    target_h: int,
    target_w: int,
    interp: int = cv2.INTER_AREA,
) -> tuple[np.ndarray, np.ndarray, dict]:
    src_h, src_w = image.shape[:2]
    scale = max(target_h / src_h, target_w / src_w)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    crop_x = (new_w - target_w) // 2
    crop_y = (new_h - target_h) // 2
    out = resized[crop_y : crop_y + target_h, crop_x : crop_x + target_w]

    K_out = scale_intrinsics(K, scale, scale)
    K_out = crop_intrinsics(K_out, crop_x, crop_y)
    info = {"scale": scale, "crop_x": crop_x, "crop_y": crop_y, "resized_hw": (new_h, new_w)}
    return out, K_out.astype(np.float32), info


def resize_and_crop_depth(
    depth: np.ndarray,
    target_h: int,
    target_w: int,
    info: dict,
) -> np.ndarray:
    new_h, new_w = info["resized_hw"]
    resized = cv2.resize(depth.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    crop_x = info["crop_x"]
    crop_y = info["crop_y"]
    return resized[crop_y : crop_y + target_h, crop_x : crop_x + target_w]


def resize_and_crop_mask(
    mask: np.ndarray,
    target_h: int,
    target_w: int,
    info: dict,
) -> np.ndarray:
    new_h, new_w = info["resized_hw"]
    resized = cv2.resize(mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    crop_x = info["crop_x"]
    crop_y = info["crop_y"]
    return resized[crop_y : crop_y + target_h, crop_x : crop_x + target_w]


def build_projection_matrix(K: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    proj = np.eye(4, dtype=np.float32)
    proj[:3, :4] = (K @ extrinsic[:3, :4]).astype(np.float32)
    return proj


def camera_center_world(extrinsic: np.ndarray) -> np.ndarray:
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return (-R.T @ t).astype(np.float32)


def view_angle_deg(
    extrinsic_ref: np.ndarray,
    extrinsic_src: np.ndarray,
    point_world: np.ndarray | None = None,
) -> float:
    c_ref = camera_center_world(extrinsic_ref)
    c_src = camera_center_world(extrinsic_src)
    if point_world is None:
        point_world = (c_ref + c_src) * 0.5
    v_ref = c_ref - point_world
    v_src = c_src - point_world
    cos_t = np.dot(v_ref, v_src) / (np.linalg.norm(v_ref) * np.linalg.norm(v_src) + 1e-9)
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def downsample_mask(mask: np.ndarray, stride: int) -> np.ndarray:
    if stride == 1:
        return mask.astype(np.float32)
    h, w = mask.shape[-2:]
    out = cv2.resize(
        mask.astype(np.float32),
        (w // stride, h // stride),
        interpolation=cv2.INTER_NEAREST,
    )
    return out


def downsample_depth(depth: np.ndarray, stride: int) -> np.ndarray:
    if stride == 1:
        return depth.astype(np.float32)
    h, w = depth.shape[-2:]
    out = cv2.resize(
        depth.astype(np.float32),
        (w // stride, h // stride),
        interpolation=cv2.INTER_NEAREST,
    )
    return out
