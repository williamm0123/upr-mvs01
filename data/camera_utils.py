from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt



def scale_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 0] *= scale_x
    out[0, 2] = (out[0, 2] + 0.5) * scale_x - 0.5
    out[1, 1] *= scale_y
    out[1, 2] = (out[1, 2] + 0.5) * scale_y - 0.5
    out[0, 1] *= scale_x
    out[1, 0] *= scale_y
    return out

def image_resize(img, depth, intrinsic, mask, resize_scale):
    ori_h, ori_w, _ = img.shape
    img = cv2.resize(img, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_AREA)
    h, w, _ = img.shape

    output_intrinsics = intrinsic.copy()
    output_intrinsics[0, :] *= resize_scale
    output_intrinsics[1, :] *= resize_scale

    if depth is not None:
        depth = cv2.resize(depth, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

    if mask is not None:
        mask = cv2.resize(mask, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

    return img, depth, output_intrinsics, mask

def image_crop(img, depth,intrinsic, mask, target_width, target_height):

    h, w = img.shape[:2]
    y0 = (h - target_height) // 2
    x0 = (w - target_width) // 2

    cropped_img = img[y0:y0 + target_height, x0:x0 + target_width]
    cropped_depth = depth[y0:y0 + target_height, x0:x0 + target_width] if depth is not None else None
    cropped_mask = mask[y0:y0 + target_height, x0:x0 + target_width] if mask is not None else None
    # 裁剪只平移主点,焦距不变
    new_intrinsic = intrinsic.copy().astype(np.float32)
    new_intrinsic[0, 2] -= x0   # cx
    new_intrinsic[1, 2] -= y0   # cy

    return cropped_img, cropped_depth, new_intrinsic, cropped_mask

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


def backproject_depth_to_world_points(
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project valid z-depth pixels to camera or world coordinates.

    DTU extrinsics are world-to-camera. When ``extrinsic`` is provided, the
    returned points are transformed back to the world frame.
    """
    height, width = depth.shape
    yy, xx = np.indices((height, width))
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)

    z = depth[valid].astype(np.float64)
    x = (xx[valid].astype(np.float64) - float(K[0, 2])) * z / max(float(K[0, 0]), 1e-12)
    y = (yy[valid].astype(np.float64) - float(K[1, 2])) * z / max(float(K[1, 1]), 1e-12)
    points_cam = np.stack((x, y, z), axis=1)
    if extrinsic is None:
        return points_cam.astype(np.float32)

    ext = np.asarray(extrinsic, dtype=np.float64)
    if ext.shape == (3, 4):
        ext4 = np.eye(4, dtype=np.float64)
        ext4[:3, :4] = ext
        ext = ext4
    points_h = np.concatenate((points_cam, np.ones((len(points_cam), 1), dtype=np.float64)), axis=1)
    points_world = (np.linalg.inv(ext) @ points_h.T).T[:, :3]
    return points_world.astype(np.float32)




def project_world_points_to_depth(
    points_world: np.ndarray, 
    conf: np.ndarray | None,    # (N,) 或 None
    K: np.ndarray,              
    extrinsic: np.ndarray,      
    image_size: tuple,          
) -> tuple[np.ndarray, np.ndarray]: # 明确声明返回值永远是两个数组
    W, H = image_size

    # 世界坐标 -> 相机坐标
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    pts_cam = (R @ points_world.T).T + t   

    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 透视投影
    u = np.round(fx * X / Z + cx).astype(np.int32)
    v = np.round(fy * Y / Z + cy).astype(np.int32)

    # 过滤越界和背面点
    mask = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Z = u[mask], v[mask], Z[mask]
    
    # 【修复 Bug】分别初始化两个矩阵
    depth_map = np.zeros((H, W), dtype=np.float32)
    
    # 如果没有置信度，默认生成全为 1.0 的置信度图（代表 100% 置信）
    if conf is None:
        conf_map = np.ones((H, W), dtype=np.float32) 
    else:
        conf_map = np.zeros((H, W), dtype=np.float32)
        conf = conf[mask]

    # 近点优先
    order = np.argsort(Z)[::-1]
    depth_map[v[order], u[order]] = Z[order]
    
    # 只有传入了 conf 时才去刷 conf_map 的值
    if conf is not None:
        conf_map[v[order], u[order]] = conf[order]

    return depth_map, conf_map  
def save_depth_png(
    depth: np.ndarray,                 # [H, W] float, 单视图深度 (来自 pred["depth"][i])
    path,
    valid: np.ndarray | None = None,   # [H, W] bool, 可选; 无效像素不参与归一化也不上色
    colored: bool = True,              # True: 伪彩色可视化 PNG; False: 16-bit 灰度原值
    depth_scale: float = 1000.0,       # 仅 colored=False 时用: 米 -> 毫米存成 uint16
):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.asarray(depth, dtype=np.float32)

    # 有效区域: 给定 valid 就用它, 否则按 有限且 > 0
    if valid is None:
        valid = np.isfinite(d) & (d > 0)
    else:
        valid = valid.astype(bool) & np.isfinite(d)

    if not colored:
        out = np.zeros_like(d, dtype=np.float32)
        out[valid] = d[valid] * depth_scale
        out = np.clip(out, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(path), out)          # 16-bit 单通道 PNG
        return path

    # 伪彩色: 在有效像素范围内做 min-max 归一化
    vis = np.zeros_like(d, dtype=np.float32)
    if valid.any():
        dmin = float(d[valid].min())
        dmax = float(d[valid].max())
        rng = max(dmax - dmin, 1e-8)
        vis[valid] = (d[valid] - dmin) / rng        # 0~1
    vis_u8 = (vis * 255.0).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)   # [H,W,3] BGR
    color[~valid] = 0                                       # 无效区涂黑
    cv2.imwrite(str(path), color)



"""把若干张测试用的中间结果(法向图 / 差值图 / 深度图 / RGB)拼到一张图保存。"""



def save_multi_images(
    images: Sequence[np.ndarray],
    save_path: str,
    rows: int | None = None,
    cols: int | None = None,
    titles: Sequence[str] | None = None,
    cell_wh: tuple[int, int] = (518, 420),
    pad: int = 8,
    bg: int = 255,
) -> np.ndarray:
    """多图拼成网格保存。

    自动识别每张图的类型并转成可显示的 BGR:
      - uint8                : 原样(灰度自动转 3 通道)
      - float 2D             : 深度图,按 2/98 百分位归一化 + TURBO 上色,无效值置黑
      - float HxWx3 且有负值  : 法向/差值,按最大绝对值对称映射到 [0,255](0 → 灰)
      - float HxWx3 且非负    : 当作 0-1 或 0-255 的 RGB

    rows/cols 任一为 None 时自动按接近正方形排布;cell_wh 为每格 (宽, 高)。
    """
    def to_bgr(img: np.ndarray) -> np.ndarray:
        a = np.asarray(img)
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]

        # uint8:原样
        if a.dtype == np.uint8:
            return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else a[..., :3]

        # float 单通道 -> 深度图
        if a.ndim == 2:
            finite = np.isfinite(a) & (a > 0)
            out = np.zeros((*a.shape, 3), np.uint8)
            if finite.any():
                lo, hi = np.percentile(a[finite], (2.0, 98.0))
                norm = np.zeros_like(a, np.float32)
                norm[finite] = np.clip((a[finite] - lo) / max(float(hi - lo), 1e-6), 0, 1)
                out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                out[~finite] = 0
            return out

        # float 3 通道
        a = np.nan_to_num(a.astype(np.float32))
        if a.min() < -1e-3:  # 法向 / 差值:对称映射,0 居中(灰)
            scale = max(float(np.percentile(np.abs(a), 99.0)), 1e-6)
            vis = np.clip(a / (2 * scale) + 0.5, 0, 1)
        else:                # 普通 RGB
            vis = np.clip(a if a.max() <= 1.0 + 1e-6 else a / 255.0, 0, 1)
        return (vis[..., ::-1] * 255).astype(np.uint8)  # RGB -> BGR

    n = len(images)
    if rows is None or cols is None:
        cols = cols or int(np.ceil(np.sqrt(n)))
        rows = rows or int(np.ceil(n / cols))
    if rows * cols < n:
        raise ValueError(f"网格容量不足: {rows}x{cols} < {n}")

    cw, ch = cell_wh
    bar = 28 if titles is not None else 0
    tiles = []
    for i in range(n):
        t = cv2.resize(to_bgr(images[i]), (cw, ch), interpolation=cv2.INTER_NEAREST)
        if titles is not None:
            head = np.full((bar, cw, 3), 40, np.uint8)
            cv2.putText(head, str(titles[i]), (6, bar - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            t = np.vstack([head, t])
        tiles.append(t)

    th, tw = ch + bar, cw
    canvas = np.full((rows * th + (rows + 1) * pad, cols * tw + (cols + 1) * pad, 3), bg, np.uint8)
    for idx in range(n):
        r, c = divmod(idx, cols)
        y, x = pad + r * (th + pad), pad + c * (tw + pad)
        canvas[y:y + th, x:x + tw] = tiles[idx]

    parent = Path(save_path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, canvas)
    return canvas

def save_pointcloud_ply(
    points: np.ndarray,                # [N, 3] 或 [V, H, W, 3] (来自 pred["world_points"])
    path: str | Path,
    colors: np.ndarray | None = None,  # [N, 3] 或 [V, H, W, 3] uint8, 可选
    conf: np.ndarray | None = None,    # [N] 或 [V, H, W] 置信度, 给了就按 conf_percentile 过滤
    conf_percentile: float = 0.0,      # >0 时丢弃最低的这一百分比 (例如 10 表示丢最低 10%)
) -> Path:
    """把点云存成 PLY (ascii)。points/colors/conf 形状要一致(同为展平或同为网格)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = None
    if colors is not None:
        cols = np.asarray(colors).reshape(-1, 3).astype(np.uint8)

    # 有效性 + 置信度过滤
    keep = np.isfinite(pts).all(axis=1)
    if conf is not None and conf_percentile > 0:
        c = np.asarray(conf, dtype=np.float32).reshape(-1)
        keep &= np.isfinite(c)
        if keep.any():
            thr = float(np.percentile(c[keep], conf_percentile))
            keep &= c >= thr
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]

    n = len(pts)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x", "property float y", "property float z",
    ]
    if cols is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        if cols is None:
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(pts, cols):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"Saved {n} points to {path}")
    return path
def camera_center_world(extrinsic: np.ndarray) -> np.ndarray:
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return (-R.T @ t).astype(np.float32)




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


def visualize_depth_3d(depth, stride=1, invalid_value=0.0, cmap='viridis',
                       point_size=1.0, invert_z=False, elev=60, azim=-90):
  
    depth = np.asarray(depth, dtype=np.float32)
    H, W = depth.shape

    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    zs = depth[0:H:stride, 0:W:stride]
    u, v, z = us.ravel(), vs.ravel(), zs.ravel()

    # 过滤无效点（NaN / inf / invalid_value）
    mask = np.isfinite(z)
    if invalid_value is not None:
        mask &= (z != invalid_value)
    u, v, z = u[mask], v[mask], z[mask]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(u, v, z, c=z, cmap=cmap, s=point_size, marker='.')

    ax.set_xlabel('u (col)')
    ax.set_ylabel('v (row)')
    ax.set_zlabel('depth')
    ax.invert_yaxis()          # 图像坐标 v 向下，保持与图像方向一致
    if invert_z:
        ax.invert_zaxis()
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(p, ax=ax, shrink=0.6, label='depth')
    plt.tight_layout()
    plt.show()
    return fig, ax
