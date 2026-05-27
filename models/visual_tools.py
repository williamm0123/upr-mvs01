from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_depth_map(depth_map, title :str ="depth map visualization" , colormap: str = "jet"):
    depth_map = np.asarray(depth_map, dtype=np.float32)
    if depth_map.ndim != 2:
        print("请输入单张深度图，或使用 plot_depth_maps 函数来处理多张深度图。")
        return
    
    im = plt.imshow(depth_map, cmap=colormap)
    plt.colorbar(im)
    plt.title(title)
    plt.axis("off")
    plt.show()
  
def plot_depth_maps(depth_maps, colormap: str = "jet"):
    depth_maps = np.asarray(depth_maps, dtype=np.float32)
    if depth_maps.ndim != 3:
        print("请输入多张深度图，或使用 plot_depth_map 函数来处理单张深度图。")
        return
    
    num_maps = depth_maps.shape[0]
    cols = min(4, num_maps)
    rows = (num_maps + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))
    for i in range(num_maps):
        plt.subplot(rows, cols, i + 1)
        im = plt.imshow(depth_maps[i], cmap=colormap)
        plt.colorbar(im)
        plt.title(f"Depth Map {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_sparse_depth(
    sparse_depth: np.ndarray,
    title: str = "Sparse Depth",
    point_size: float = 10.0,
    cmap: str = "jet",
    percentile: tuple[float, float] = (2.0, 98.0),
    background: str = "#111111",
    alpha: float = 0.95,
    dpi: int = 200,
) -> Path | None:
    """
    Visualize sparse depth as large colored points on a dark background.

    This is easier to inspect than imshow when most pixels are NaN.
    """
    depth = np.asarray(sparse_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected sparse_depth shape [H, W], got {depth.shape}")

    valid = np.isfinite(depth)
    if not valid.any():
        raise ValueError("No valid finite sparse depth values to visualize.")

    y, x = np.nonzero(valid)
    z = depth[valid]
    vmin, vmax = np.percentile(z, percentile)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    height, width = depth.shape
    fig_width = 8.0
    fig_height = fig_width * float(height) / float(width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=background)
    ax.set_facecolor(background)
    ax.scatter(
        x,
        y,
        c=z,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        marker="o",
        linewidths=0,
    )
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color="white")
    fig.tight_layout(pad=0)



    plt.show()





def save_sparse_depth(
    depth: np.ndarray,
    output_path: str | Path | None = None,
    point_size: float = 10.0,
    cmap: str = "turbo",
    percentile: tuple[float, float] = (2.0, 98.0),
    background: str = "#111111",
    alpha: float = 0.95,
    dpi: int = 200,
    show: bool = False,
) -> Path | None:
    """
    Visualize sparse depth as large colored points on a dark background.

    This is easier to inspect than imshow when most pixels are NaN.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape [H, W], got {depth.shape}")

    valid = np.isfinite(depth)
    if not valid.any():
        raise ValueError("No valid finite depth values to visualize.")

    y, x = np.nonzero(valid)
    z = depth[valid]
    vmin, vmax = np.percentile(z, percentile)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    height, width = depth.shape
    fig_width = 8.0
    fig_height = fig_width * float(height) / float(width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=background)
    ax.set_facecolor(background)
    ax.scatter(
        x,
        y,
        c=z,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        marker="o",
        linewidths=0,
    )
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0, dpi=dpi)

    if show:
        plt.show()

    plt.close(fig)
    return saved_path


# 可视化 DA3 模型的 log-gradient magnitude。
def show_loggrad_magnitude(loggrad_magnitude, percentile: float = 99.0, cmap: str = "coolwarm"):
    grad = np.asarray(loggrad_magnitude, dtype=np.float32)

    valid = np.isfinite(grad)

    if not valid.any():
        raise ValueError("loggrad_magnitude has no valid finite values.")

    vmin = 0.0
    vmax = float(np.percentile(grad[valid], percentile))

    plt.figure(figsize=(8, 6))
    im = plt.imshow(grad, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("DA3 Log-Gradient Magnitude")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# 可视化 DA3 模型的 8 个方向的 log-gradient。
def show_8dir_log_grads(log_grads_8dir, percentile: float = 99.0, cmap: str = "coolwarm"):
    names = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]

    grads = np.asarray(log_grads_8dir, dtype=np.float32)

    if grads.shape[0] != 8:
        raise ValueError(f"Expected shape [8, H, W], got {grads.shape}")

    valid = np.isfinite(grads)

    if not valid.any():
        raise ValueError("log_grads_8dir has no valid finite values.")

    abs_max = float(np.percentile(np.abs(grads[valid]), percentile))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        im = axes[i].imshow(
            grads[i],
            cmap=cmap,
            vmin=-abs_max,
            vmax=abs_max,
        )
        axes[i].set_title(names[i])
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()




def save_depth_png(
    depth: np.ndarray,
    output_path: str | Path,
    cmap: str = "jet",
    percentile: tuple[float, float] = (2.0, 98.0),
    invalid_color: tuple[float, float, float, float] = (0, 0, 0, 1),
) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape [H, W], got {depth.shape}")

    valid = np.isfinite(depth)

    if not valid.any():
        raise ValueError("No valid finite values to visualize.")

    vmin, vmax = np.percentile(depth[valid], percentile)

    if vmax <= vmin:
        vmax = vmin + 1e-6

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(invalid_color)

    depth_masked = np.ma.masked_where(~valid, depth)

    plt.figure(figsize=(8, 6))
    plt.imshow(depth_masked, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()

    return output_path

def save_loggrad_magnitude_png(
    loggrad_magnitude: np.ndarray,
    output_path: str | Path,
    cmap: str = "coolwarm",
    percentile: tuple[float, float] = (50.0, 99.5),
) -> Path:
    """
    Save DA3 log-gradient magnitude as PNG.

    Args:
        loggrad_magnitude: [H, W]
    """
    return save_depth_png(
        depth=loggrad_magnitude,
        output_path=output_path,
        cmap=cmap,
        percentile=percentile,
        invalid_color=(0, 0, 0, 1),
    )
def save_signed_scalar_png(
    values: np.ndarray,
    output_path: str | Path,
    cmap: str = "coolwarm",
    percentile: float = 99.0,
    invalid_color: tuple[float, float, float, float] = (0, 0, 0, 1),
) -> Path:
    """
    Save signed scalar map as PNG.

    Suitable for log-gradient maps that contain positive and negative values.

    Args:
        values: [H, W]
        percentile: use symmetric range [-p, p]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError(f"Expected values shape [H, W], got {values.shape}")

    valid = np.isfinite(values)

    if not valid.any():
        raise ValueError("No valid finite values to visualize.")

    abs_max = float(np.percentile(np.abs(values[valid]), percentile))

    if abs_max <= 0:
        abs_max = 1e-6

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(invalid_color)

    values_masked = np.ma.masked_where(~valid, values)

    plt.figure(figsize=(8, 6))
    plt.imshow(values_masked, cmap=cmap_obj, vmin=-abs_max, vmax=abs_max)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()

    return output_path

def save_8dir_loggrad_pngs(
    log_gradients_8dir: np.ndarray,
    output_dir: str | Path,
    prefix: str = "da3",
    cmap: str = "coolwarm",
    percentile: float = 99.0,
) -> list[Path]:
    """
    Save each 8-direction log-gradient map as separate PNG.

    Args:
        log_gradients_8dir: [8, H, W]
        output_dir: output folder

    Returns:
        list of saved paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_gradients_8dir = np.asarray(log_gradients_8dir, dtype=np.float32)

    if log_gradients_8dir.ndim != 3 or log_gradients_8dir.shape[0] != 8:
        raise ValueError(
            f"Expected log_gradients_8dir shape [8, H, W], got {log_gradients_8dir.shape}"
        )

    direction_names = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]

    saved_paths = []

    for i, name in enumerate(direction_names):
        output_path = output_dir / f"{prefix}_loggrad_{name}.png"

        save_signed_scalar_png(
            values=log_gradients_8dir[i],
            output_path=output_path,
            cmap=cmap,
            percentile=percentile,
        )

        saved_paths.append(output_path)

    return saved_paths
def save_8dir_loggrad_overview_png(
    log_gradients_8dir: np.ndarray,
    output_path: str | Path,
    cmap: str = "coolwarm",
    percentile: float = 99.0,
) -> Path:
    """
    Save 8-direction log-gradient maps as one overview PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_gradients_8dir = np.asarray(log_gradients_8dir, dtype=np.float32)

    if log_gradients_8dir.ndim != 3 or log_gradients_8dir.shape[0] != 8:
        raise ValueError(
            f"Expected log_gradients_8dir shape [8, H, W], got {log_gradients_8dir.shape}"
        )

    valid = np.isfinite(log_gradients_8dir)

    if not valid.any():
        raise ValueError("No valid finite log-gradient values to visualize.")

    abs_max = float(np.percentile(np.abs(log_gradients_8dir[valid]), percentile))

    if abs_max <= 0:
        abs_max = 1e-6

    direction_names = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad((0, 0, 0, 1))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        grad = log_gradients_8dir[i]
        grad_masked = np.ma.masked_where(~np.isfinite(grad), grad)

        im = axes[i].imshow(
            grad_masked,
            cmap=cmap_obj,
            vmin=-abs_max,
            vmax=abs_max,
        )
        axes[i].set_title(direction_names[i])
        axes[i].axis("off")

    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close()

    return output_path

import numpy as np




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
    """
    计算两个深度图的差值。

    absolute=True:
        diff = abs(depth_a - depth_b)

    absolute=False:
        diff = depth_a - depth_b

    两张图任意一方无效的位置，输出 NaN。
    """
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
    """
    将深度图反投影成点云。

    Args:
        depth:
            [H, W] depth map, depth 是 camera z-depth。

        K:
            [3, 3] camera intrinsic.

        extrinsic:
            [4, 4] or [3, 4] world-to-camera matrix.
            如果提供，则输出 world coordinates。
            如果为 None，则输出 camera coordinates。

    Returns:
        points:
            [N, 3] point cloud.
    """
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


def plot_depth_pointcloud_3d(
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray | None = None,
    title: str = "Depth Point Cloud",
    point_size: float = 1.0,
    cmap: str = "turbo",
    max_points: int = 200_000,
    stride: int = 1,
    percentile: tuple[float, float] = (2.0, 98.0),
    elev: float = -70.0,
    azim: float = -90.0,
    show: bool = True,
):
    """
    Plot a depth map as an interactive 3D point cloud.

    Matplotlib's 3D window supports mouse drag rotation when an interactive
    backend is available.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape [H, W], got {depth.shape}")

    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected K shape [3, 3], got {K.shape}")

    if stride > 1:
        depth_for_plot = depth[::stride, ::stride]
        K_for_plot = K.copy()
        K_for_plot[0, :] /= float(stride)
        K_for_plot[1, :] /= float(stride)
    else:
        depth_for_plot = depth
        K_for_plot = K

    valid = np.isfinite(depth_for_plot) & (depth_for_plot > 0)
    if not valid.any():
        raise ValueError("No valid finite depth values to plot.")

    points = backproject_depth_to_points(depth_for_plot, K_for_plot, extrinsic)
    colors = depth_for_plot[valid].astype(np.float32)

    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points.shape[0], size=int(max_points), replace=False)
        points = points[keep]
        colors = colors[keep]

    vmin, vmax = np.percentile(colors[np.isfinite(colors)], percentile)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.02, label="depth")

    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    centers = (mins + maxs) * 0.5
    radius = float(np.nanmax(maxs - mins) * 0.5)
    if radius <= 0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


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




def o3d_depth_visualization(depth):

    if o3d is None:
        raise ModuleNotFoundError(
            "open3d is required for o3d_depth_visualization(). "
            "Install it with `pip install open3d` if you need this viewer."
        )

    H, W = depth.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # 假设 depth 是你加载的深度图矩阵 (比如 1200 x 1600 的 numpy 数组)
    # depth = np.load("rect_001_3_r5000_depth.npy")

    H, W = depth.shape

    # 生成每个像素的 u (列号) 和 v (行号) 坐标
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 将矩阵展平，方便组合成点云
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()

    # 过滤掉无效的深度值 (比如 inf 或 nan 或 0)
    valid = np.isfinite(z) & (z > 0)
    u, v, z = u[valid], v[valid], z[valid]

    # 组合成 (N, 3) 的点云坐标
    # 注意：图像的 v 轴朝下，为了 3D 显示正常，通常会把 y 和 z 轴反过来
    points = np.vstack((u, -v, -z)).T

    # 使用 Open3D 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print("生成点云，准备显示...")
    o3d.visualization.draw_geometries([pcd])
    
def visualize_depth_uv_pointcloud(
    depth,
    output_path: str | Path,
    title="Depth UV Point Cloud",
    stride=1,
    max_points=200000,
    z_scale=1.0,
    invert_y=True,
    invert_z=False,
    marker_size=2.0,
    colorscale="Turbo",
):
    import numpy as np
    import plotly.graph_objects as go

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".html":
        safe_title = "".join(
            char if char.isalnum() or char in ("-", "_", ".") else "_"
            for char in str(title)
        ).strip("_")
        output_path = output_path / f"{safe_title or 'depth_uv_pointcloud'}.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth = np.asarray(depth, dtype=np.float32)

    depth_s = depth[::stride, ::stride]
    H, W = depth_s.shape

    v, u = np.indices((H, W))

    u = u.astype(np.float32) * stride
    v = v.astype(np.float32) * stride
    z = depth_s.astype(np.float32)

    valid = np.isfinite(z) & (z > 0)

    u = u[valid]
    v = v[valid]
    z = z[valid]

    if invert_y:
        v = -v

    z_plot = z * z_scale

    if invert_z:
        z_plot = -z_plot

    if z_plot.shape[0] == 0:
        raise ValueError("No valid depth pixels to visualize.")

    if max_points is not None and z_plot.shape[0] > max_points:
        idx = np.random.choice(z_plot.shape[0], max_points, replace=False)
        u = u[idx]
        v = v[idx]
        z = z[idx]
        z_plot = z_plot[idx]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=u,
                y=v,
                z=z_plot,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=z,
                    colorscale=colorscale,
                    colorbar=dict(title="Depth"),
                    opacity=0.9,
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="u",
            yaxis_title="-v" if invert_y else "v",
            zaxis_title="depth",
            aspectmode="data",
        ),
        width=900,
        height=700,
    )
    fig.write_html(str(output_path))
    return output_path
