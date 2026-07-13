
from __future__ import annotations

import os
import sys
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from scipy.spatial import KDTree as cKDTree
from scipy import sparse, ndimage
from scipy.sparse.linalg import cg, spsolve



from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
from depth_anything_3.api import DepthAnything3
 
from base.config import ProjectPaths
from data.camera_utils import  project_world_points_to_depth, backproject_depth_to_world_points

from models.conf import compute_confidence,camera_rays
import data.camera_utils as C
import models.sfm as S





@dataclass
class VoxelDedupConfig:     
    auto_scale: float = 1.3            # 自动估时: voxel_size = scale * 中位近邻距离
    auto_sample: int = 20000            # 自动估时用于测近邻距离的随机采样点数 (控开销)


@dataclass(frozen=True)
class DepthFillConfig:
    hard_keep_sparse: bool = True
    clamp_output: bool = True
    clamp_percentiles: tuple[float, float] = (0.5, 99.5)
    clamp_margin_ratio: float = 0.15
    anchor_weight: float = 100.0
    guide_weight: float = 0.2
    edge_weight: float = 1.0
    edge_min_denom: float = 1e-4
    edge_ratio_limits: tuple[float, float] = (0.4, 2.5)
    edge_min_similarity: float = -0.2
    edge_similarity_power: float = 2.0
    cg_maxiter: int = 600
    cg_rtol: float = 1e-5
    fallback_spsolve: bool = True
    align_trim_mad: float = 3.5
    align_min_points: int = 100





# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_vggt(weights_path: Path, device: torch.device) -> nn.Module:
    if not weights_path.is_dir():
        raise FileNotFoundError(f"VGGT weights path must be a directory: {weights_path}")

    st_files = sorted(weights_path.glob("*.safetensors"))
    weights_file = st_files[0] if st_files else weights_path / "model.pt"
    if not weights_file.is_file():
        raise FileNotFoundError(f"VGGT weights not found under {weights_path}: expected *.safetensors or model.pt")

    model = VGGT()
    if weights_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(weights_file))
    elif weights_file.name == "model.pt":
        ckpt = torch.load(str(weights_file), map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    else:
        raise ValueError(f"Unsupported VGGT weights file: {weights_file}; expected *.safetensors or model.pt")
    model.load_state_dict(state, strict=False)

    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()

def load_vggt_model(weights_path: str | Path, device: torch.device) -> nn.Module:
    return _load_vggt(Path(weights_path), device)


def load_da3_model(weights_path: str | Path, device: torch.device) -> Any:
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained(str(weights_path))
    model = model.to(device=device).eval()

    return model


# ---------------------------------------------------------------------------
# Input resize
# ---------------------------------------------------------------------------




@dataclass
class ResizeTransform:
    """orig -> model 的仿射 (sx, sy, tx, ty) 及正反向尺寸。"""
    sx: float
    sy: float
    tx: float
    ty: float
    src_w: int   # 原图宽 (orig), 例如 640
    src_h: int   # 原图高 (orig), 例如 512
    dst_w: int   # 模型输入宽 (model), 例如 518
    dst_h: int   # 模型输入高 (model), 例如 392
    mode: str    # "resize" | "pad" | "crop"
 
    def forward_pix(self, u, v):
        """orig 像素 -> model 像素"""
        return self.sx * u + self.tx, self.sy * v + self.ty
 
    def inverse_pix(self, u, v):
        """model 像素 -> orig 像素"""
        return (u - self.tx) / self.sx, (v - self.ty) / self.sy
 
 
# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #

def _build_affine(src_w, src_h, dst_w, dst_h, mode):
    if mode == "resize":                 # 各向异性拉伸, 不保持长宽比
        sx = dst_w / src_w
        sy = dst_h / src_h
        tx = 0.5 * sx - 0.5
        ty = 0.5 * sy - 0.5
    elif mode in ("pad", "crop"):        # 等比缩放, 保持长宽比
        if mode == "pad":                # letterbox: 取小比例, 四周补边
            s = min(dst_w / src_w, dst_h / src_h)
        else:                            # crop: 取大比例, 填满后中心裁切
            s = max(dst_w / src_w, dst_h / src_h)
        sx = sy = s
        off_x = (dst_w - s * src_w) / 2.0   # pad 时 >0(补边), crop 时 <0(裁切)
        off_y = (dst_h - s * src_h) / 2.0
        tx = (0.5 * s - 0.5) + off_x
        ty = (0.5 * s - 0.5) + off_y
    else:
        raise ValueError(f"unknown mode: {mode}")
    return float(sx), float(sy), float(tx), float(ty)
 
 
def _resample(src, out_h, out_w, coord_fn, interp="bicubic", padding_mode="zeros"):

    B, C, Hs, Ws = src.shape
    dev, dt = src.device, src.dtype
    vs = torch.arange(out_h, device=dev, dtype=dt)
    us = torch.arange(out_w, device=dev, dtype=dt)
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")      # [out_h, out_w]
    u_src, v_src = coord_fn(grid_u, grid_v)
    # 像素坐标 -> grid_sample 归一化坐标 (align_corners=False)
    x = (u_src + 0.5) / Ws * 2.0 - 1.0
    y = (v_src + 0.5) / Hs * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(src, grid, mode=interp,
                         padding_mode=padding_mode, align_corners=False)
 
 
# --------------------------------------------------------------------------- #
# 正向: 原图 -> 模型输入图 (并同步内参)
# --------------------------------------------------------------------------- #
 
def resize_image_with_transform(
    image,              # [C, H, W] 或 [B, C, H, W], float
    K,                # [3, 3] 原图内参
    target_size: Tuple[int, int],     # (W_target, H_target), 例如 (518, 392)
    mode: str = "resize",             # "resize" | "pad" | "crop"
):

    squeeze = (image.dim() == 3)
    if squeeze:
        image = image.unsqueeze(0)
    B, C, src_h, src_w = image.shape
    dst_w, dst_h = target_size
 
    sx, sy, tx, ty = _build_affine(src_w, src_h, dst_w, dst_h, mode)
    transform = ResizeTransform(sx, sy, tx, ty, src_w, src_h, dst_w, dst_h, mode)
 
    # 生成 model 图: 对每个 model 像素, 取 orig 像素 = inverse(model)
    images_resized = _resample(
        image, dst_h, dst_w,
        coord_fn=transform.inverse_pix,   #在目标图中，对于每个像素，求它在原图中的位置
        interp="bicubic",
        padding_mode="zeros",          # pad 区域填 0
    )
 
    # 内参变换: u_model = sx*u + tx  =>  fx'=sx*fx, cx'=sx*cx+tx (v 同理)
    K_np = K.astype(np.float64) if isinstance(K, np.ndarray) else K.detach().cpu().numpy().astype(np.float64)
    K_new = K_np.copy()
    K_new[0, 0] = K_np[0, 0] * sx
    K_new[1, 1] = K_np[1, 1] * sy
    K_new[0, 2] = K_np[0, 2] * sx + tx
    K_new[1, 2] = K_np[1, 2] * sy + ty
 
    if squeeze:
        images_resized = images_resized.squeeze(0)
    return images_resized, K_new, transform
 
 
# --------------------------------------------------------------------------- #
# 逆向: 模型输出(depth/conf/edge/mask) -> 原分辨率
# --------------------------------------------------------------------------- #
def _inverse_intrinsic(K, transform: ResizeTransform):
    """把 model 尺寸 (dst) 下的内参还原到原图 (src) 尺寸。

    正向 (见 resize_image_with_transform): fx'=sx*fx, cx'=sx*cx+tx (fy/cy 同理),
    因此逆向: fx=fx'/sx, cx=(cx'-tx)/sx。支持单个 [3,3] 或批量 [...,3,3]。
    """
    sx, sy, tx, ty = transform.sx, transform.sy, transform.tx, transform.ty
    K_arr = K if isinstance(K, np.ndarray) else K.detach().cpu().numpy()
    K_new = K_arr.astype(np.float64).copy()
    K_new[..., 0, 0] /= sx
    K_new[..., 1, 1] /= sy
    K_new[..., 0, 2] = (K_new[..., 0, 2] - tx) / sx
    K_new[..., 1, 2] = (K_new[..., 1, 2] - ty) / sy
    return K_new.astype(np.float32)


def inverse_transform_map(
    x_model,            # [H,W] / [1,H,W] / [B,1,H,W] / [B,C,H,W]
    transform: ResizeTransform,
    K=None,  
    interp: str = "nearest",          # depth/mask 建议 nearest; conf/edge 可用 bilinear
                          # 可选: model 尺寸下的内参 [3,3] 或 [...,3,3]
):
    """
    用记录的 transform 把模型坐标系下的图(在 dst_h x dst_w 上)重采样回原图
    (src_h x src_w)。depth 数值本身不缩放, 只做空间重采样。

    K 为 None 时只返回还原后的图 (向后兼容); 传入 K 时同步把内参从 model
    尺寸还原到原图尺寸, 返回 (out, K_src)。
    """
    is_numpy = isinstance(x_model, np.ndarray)
    if is_numpy:
        x_model = torch.from_numpy(np.ascontiguousarray(x_model)).float()

    nd = x_model.ndim
    if nd == 2:
        x_model = x_model[None, None]
    elif nd == 3:
        x_model = x_model.unsqueeze(0)

    assert x_model.shape[-1] == transform.dst_w and x_model.shape[-2] == transform.dst_h, \
        f"输入应为 model 尺寸 ({transform.dst_h},{transform.dst_w}), 实际 {tuple(x_model.shape[-2:])}"

    sx, sy, tx, ty = transform.sx, transform.sy, transform.tx, transform.ty
    out = _resample(
        x_model, transform.src_h, transform.src_w,
        coord_fn=transform.forward_pix,    #在原图尺寸（640*512）中，对于每个像素，求它在目标图（518*420）中的位置
        interp=interp,
        padding_mode="border",         # crop 模式下越界处取边界值; resize/pad 不会越界
    )

    if nd == 2:
        out = out[0, 0]
    elif nd == 3:
        out = out.squeeze(0)

    if is_numpy:
        out = out.cpu().numpy()
    out = np.asarray(out)

    if K is None:
        return out
    return out, _inverse_intrinsic(K, transform)
 
 

@torch.no_grad()
def _run_vggt(vggt_model: nn.Module, images_tensor: torch.Tensor, device: torch.device) -> dict[str, np.ndarray]:
    image_tensor = images_tensor.to(device)
    preds = vggt_model(image_tensor)
    _, _, h, w = images_tensor.shape
    extrinsics, intrinsics = pose_encoding_to_extri_intri(preds["pose_enc"], (h,w))
    extrinsics = extrinsics[0].float().cpu().numpy()
    intrinsics = intrinsics[0].float().cpu().numpy()

    return {
        "depth": preds["depth"][0, ..., 0].float().cpu().numpy(),
        "world_points": preds["world_points"][0].float().cpu().numpy(),
        "world_points_conf": preds["world_points_conf"][0].float().cpu().numpy(),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }



def _get_depth_da3(
    image,
    da3_model: Any,
    process_res,
) -> np.ndarray:
    # 接受 tensor 或 ndarray，统一转成 PIL 需要的 HWC uint8 numpy
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.dim() == 3 and t.shape[0] in (1, 3):   # CHW -> HWC
            t = t.permute(1, 2, 0)
        if t.is_floating_point():                    # float 0~255 -> uint8
            t = t.clamp(0, 255).round()
        image_np = t.to(torch.uint8).contiguous().numpy()
    else:
        image_np = image.astype(np.uint8)

    with torch.inference_mode():
        pred = da3_model.inference(
            image=[Image.fromarray(image_np, "RGB")],
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
            export_format="npz",
        )
    depth = pred.depth[0].astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


# ---------------------------------------------------------------------------
# VGGT point cloud denoise
# ---------------------------------------------------------------------------


def _tensor_to_uint8_hwc(images_resized):
    # [V,3,H,W] float(约0~255, bicubic 可能轻微越界) -> [V,H,W,3] uint8
    imgs = images_resized.detach().float().clamp(0, 255).round().to(torch.uint8)
    return imgs.permute(0,2,3,1).contiguous().cpu().numpy()


def _estimate_voxel_size(points: np.ndarray, scale: float, sample: int) -> float:
    """用随机采样点的最近邻距离中位数估一个体素尺度, 避免写死绝对值。"""

    n = len(points)
    if n <= 1:
        return 1.0
    idx = np.arange(n) if n <= sample else np.random.default_rng(0).choice(n, sample, replace=False)
    tree = cKDTree(points)
    # k=2: 第 0 个是自己, 第 1 个是最近邻
    dists, _ = tree.query(points[idx], k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    med = float(np.median(nn)) if nn.size else 1.0
    return max(med * scale, 1e-8)


def voxel_dedup_pointcloud(
    points,                 # [N, 3] 
    conf,     # [N] 或 [V, H, W]; 每个体素保留 conf 最大的点
    voxel_size ,
):
    n_total = len(points)
    # 只在有限点上操作

    # 体素坐标 (整数)
    origin = points.min(axis=0)
    vox = np.floor((points - origin) / voxel_size).astype(np.int64)   # [Nf, 3]

    # 给每个体素一个唯一 key, 求每个点属于哪个体素组
    # unique 的 inverse 把每个点映射到它所在体素的组号
    _, inverse, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    n_vox = len(counts)

    # 在每个体素组内挑代表点:
    #   有 conf -> 选组内 conf 最大的点; 无 conf -> 选组内第一个点
    
    score = conf
    
    # 对每个体素组求 "组内分数最大的点的全局下标"
    # 做法: 按 (组号, 分数) 排序, 每组最后一个即分数最大者
    order = np.lexsort((score, inverse))      # 先按 inverse 分组, 组内按 score 升序
    inv_sorted = inverse[order]
    # 每组的最后一个位置 = 该组分数最大的点
    last_in_group = np.ones(len(order), dtype=bool)
    last_in_group[:-1] = inv_sorted[1:] != inv_sorted[:-1]
    rep_local = order[last_in_group]          # 代表点在 pts_f 中的下标, 共 n_vox 个
         # 映射回展平原数组的下标
    points_kept = points[rep_local].astype(np.float32)
    conf_kept =  conf[rep_local]

    info = {
        "input_points": int(n_total),
        "voxel_size": float(voxel_size),
        "voxels": int(n_vox),
        "kept_points": int(len(points_kept)),
        "dedup_ratio": float(len(points_kept) / max(len(points), 1)),
        "conf_used": bool(conf is not None),
    }
    return points_kept,conf_kept, info

#  ---------------------------------------------------------------------------
#norm_fill
#----------------------------------------------------------------------------

"""DA3-normal-constrained depth completion."""




def depth_to_camera_normals(depth, intrinsic):
    rays = camera_rays(depth.shape, intrinsic)           # 复用这一份
    pts = rays * depth[..., None].astype(np.float32)
    dx = np.gradient(pts, axis=1)
    dy = np.gradient(pts, axis=0)
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    bad = (~np.isfinite(norm).squeeze(-1)) | (norm.squeeze(-1) < 1e-8)
    normals = normals / np.maximum(norm, 1e-8)

    # —— 关键改动：按视线方向统一朝向相机，而不是按 z 轴 ——
    view = np.sum(normals * rays, axis=-1)               # n·ray，逐像素
    normals[view > 0] *= -1.0                            # 统一到 n·ray < 0（朝相机）

    normals[bad] = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # 兜底也对齐到朝相机
    return normals.astype(np.float32)





def robust_affine_align_depth(
    source: np.ndarray,
    target: np.ndarray,
    target_valid: np.ndarray,
    trim_mad: float,
    min_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit d_aligned = scale * source + bias by iteratively trimmed least squares."""
    valid = target_valid & np.isfinite(source) & (source > 0)
    if int(valid.sum()) < min_points:
        raise ValueError(f"Too few anchors for DA3 alignment: {int(valid.sum())} < {min_points}")
    x = source[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    keep = np.ones_like(x, dtype=bool)
    scale, bias = 1.0, 0.0
    for _ in range(3):
        A = np.stack([x[keep], np.ones(int(keep.sum()))], axis=1)
        scale, bias = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = y - (scale * x + bias)
        med = float(np.median(r[keep]))
        mad = float(np.median(np.abs(r[keep] - med)))
        sigma = max(1.4826 * mad, 1e-8)
        new_keep = np.abs(r - med) <= trim_mad * sigma
        if int(new_keep.sum()) < min_points or int(new_keep.sum()) == int(keep.sum()):
            break
        keep = new_keep
    aligned = (float(scale) * source + float(bias)).astype(np.float32)
    return aligned, {
        "scale": float(scale),
        "bias": float(bias),
        "anchors": int(valid.sum()),
        "robust_anchors": int(keep.sum()),
    }


def _robust_affine_align(
    source: np.ndarray,
    target: np.ndarray,
    target_valid: np.ndarray,
    trim_mad: float,
    min_points: int,
) -> np.ndarray:
    aligned, _info = robust_affine_align_depth(source, target, target_valid, trim_mad, min_points)
    return aligned



# ---------------------------------------------------------------------------
# Normal-constrained Poisson fill
# ---------------------------------------------------------------------------
def _normal_edge_rows(
    a_idx: np.ndarray, b_idx: np.ndarray,
    ray_a: np.ndarray, ray_b: np.ndarray,
    n_a: np.ndarray, n_b: np.ndarray,
    cfg: DepthFillConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = n_a + n_b
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    sim = np.sum(n_a * n_b, axis=-1)
    num = np.sum(n * ray_a, axis=-1)
    den = np.sum(n * ray_b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
    lo, hi = cfg.edge_ratio_limits
    keep = (
        np.isfinite(ratio) & np.isfinite(sim)
        & (np.abs(num) >= cfg.edge_min_denom)
        & (np.abs(den) >= cfg.edge_min_denom)
        & (ratio >= lo) & (ratio <= hi)
        & (sim >= cfg.edge_min_similarity)
    )
    if not keep.any():
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f
    sim01 = np.clip(sim[keep], 0.0, 1.0)
    w = cfg.edge_weight * np.maximum(sim01, 1e-3) ** cfg.edge_similarity_power
    return a_idx[keep], b_idx[keep], ratio[keep].astype(np.float32), w.astype(np.float32)


def _fill_with_normals(
    sparse_depth: np.ndarray,
    guide_depth: np.ndarray,
    normals: np.ndarray,
    K: np.ndarray,
    valid: np.ndarray,
    cfg: DepthFillConfig,
) -> np.ndarray:

    h, w = sparse_depth.shape
    n_pix = h * w
    rays = camera_rays((h, w), K)
    pix = np.arange(n_pix, dtype=np.int64).reshape(h, w)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    row = 0

    g_valid = np.isfinite(guide_depth) & (guide_depth > 0)
    if cfg.guide_weight > 0 and g_valid.any():
        ids = pix[g_valid].reshape(-1)
        w_g = float(np.sqrt(cfg.guide_weight))
        rows.append(np.arange(row, row + len(ids), dtype=np.int64))
        cols.append(ids)
        data.append(np.full(len(ids), w_g, dtype=np.float32))
        rhs.append(w_g * guide_depth[g_valid].reshape(-1).astype(np.float32))
        row += len(ids)

    a_ids = pix[valid].reshape(-1)
    w_a = float(np.sqrt(cfg.anchor_weight))
    rows.append(np.arange(row, row + len(a_ids), dtype=np.int64))
    cols.append(a_ids)
    data.append(np.full(len(a_ids), w_a, dtype=np.float32))
    rhs.append(w_a * sparse_depth[valid].reshape(-1).astype(np.float32))
    row += len(a_ids)

    edge_axes = (
        (pix[:, :-1].reshape(-1), pix[:, 1:].reshape(-1),
         rays[:, :-1].reshape(-1, 3), rays[:, 1:].reshape(-1, 3),
         normals[:, :-1].reshape(-1, 3), normals[:, 1:].reshape(-1, 3)),
        (pix[:-1, :].reshape(-1), pix[1:, :].reshape(-1),
         rays[:-1, :].reshape(-1, 3), rays[1:, :].reshape(-1, 3),
         normals[:-1, :].reshape(-1, 3), normals[1:, :].reshape(-1, 3)),
    )
    for ia, ib, ra, rb, na, nb in edge_axes:
        idx_a, idx_b, ratio, weights = _normal_edge_rows(ia, ib, ra, rb, na, nb, cfg)
        if ratio.size == 0:
            continue
        sw = np.sqrt(np.maximum(weights, 1e-8)).astype(np.float32)
        rows.append(np.repeat(np.arange(row, row + len(ratio), dtype=np.int64), 2))
        cols.append(np.stack((idx_a, idx_b), axis=1).reshape(-1))
        data.append(np.stack((-sw * ratio, sw), axis=1).reshape(-1))
        rhs.append(np.zeros(len(ratio), dtype=np.float32))
        row += len(ratio)

    A = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(row, n_pix),
    ).tocsr()
    b = np.concatenate(rhs)
    init = np.where(valid, sparse_depth, guide_depth).astype(np.float32).reshape(-1)
    ATA = A.T @ A
    ATb = A.T @ b
    try:
        sol, info = cg(ATA, ATb, x0=init, rtol=cfg.cg_rtol, atol=0.0, maxiter=cfg.cg_maxiter)
    except TypeError:
        sol, info = cg(ATA, ATb, x0=init, tol=cfg.cg_rtol, maxiter=cfg.cg_maxiter)
    if info != 0 and cfg.fallback_spsolve:
        sol = spsolve(ATA.tocsc(), ATb)
    filled = np.asarray(sol, dtype=np.float32).reshape(h, w)

    if cfg.hard_keep_sparse:
        filled[valid] = sparse_depth[valid]
    if cfg.clamp_output and valid.any():
        lo, hi = np.percentile(sparse_depth[valid], cfg.clamp_percentiles)
        margin = cfg.clamp_margin_ratio * max(float(hi - lo), 1e-6)
        filled = np.clip(filled, float(lo - margin), float(hi + margin))
    return filled.astype(np.float32)


def fill_depth_with_normal_constraints(
    sparse_depth: np.ndarray,
    aligned_guide_depth: np.ndarray,
    normals: np.ndarray,
    intrinsic: np.ndarray,
    valid: np.ndarray,
    config: DepthFillConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config is None:
        config = DepthFillConfig()
    filled = _fill_with_normals(
        sparse_depth=sparse_depth,
        guide_depth=aligned_guide_depth,
        normals=normals,
        K=intrinsic,
        valid=valid,
        cfg=config,
    )
    guide_valid = np.isfinite(aligned_guide_depth) & (aligned_guide_depth > 0)
    return filled.astype(np.float32), {
        "anchor_pixels": int(valid.sum()),
        "guide_pixels": int(guide_valid.sum()),
        "total_pixels": int(valid.size),
    }


def fill_vggt_depth_by_da3_normals(
    sparse_depth: np.ndarray,
    da3_depth: np.ndarray,
    intrinsic: np.ndarray,
    valid: np.ndarray | None = None,
    config: DepthFillConfig | None = None,
):
    """Complete projected VGGT depth with DA3 normals using the original depth_fill path."""
    if config is None:
        config = DepthFillConfig()
    sparse_depth = sparse_depth.astype(np.float32)

    valid = np.isfinite(sparse_depth) & (sparse_depth> 0) if valid is None else valid.astype(bool)

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

    return filled.astype(np.float32), da3_normals



# # ---------------------------------------------------------------------------
# Per-sample entry point
# ---------------------------------------------------------------------------
def generate_priors_from_sample(
    sample,
    device,
    image_mode: str = "resize",
    fill_config: DepthFillConfig | None = None,
    conf_percentile: float = 10.0,
    image_target_wh: tuple[int, int] = (518, 420),
    sfm_config: "S.SfMConfig | None" = None,
    vggt_model=None,
    da3_model=None,
):
    paths = ProjectPaths()
    # Preloaded models can be passed in (offline precompute loads them once);
    # otherwise fall back to loading per call as before.
    if vggt_model is None:
        vggt_model = load_vggt_model(paths.vggt_weights_path, device)
    if da3_model is None:
        da3_model = load_da3_model(paths.da3_weights_file, device)
    images_orig = _tensor_to_uint8_hwc(sample["images"])
    images_resized, K_resized, ref_transform = resize_image_with_transform(
        sample["images"], sample["intrinsics"], image_target_wh, image_mode)

    images_uint8 = _tensor_to_uint8_hwc(images_resized)

    pred = _run_vggt(vggt_model, images_resized.clamp(0,255)/255.0, device)
    pred["images_uint8"] = images_uint8
    
    ## conf filtering
    points = np.asarray(pred["world_points"])
    conf = np.asarray(pred["world_points_conf"])
    colors = np.asarray(pred.get("images_uint8", np.zeros(points.shape[:-1] + (3,), dtype=np.uint8)))
    ref_points = points[0]

    flat_ref_points = ref_points.reshape(-1, 3)
    flat_ref_conf = conf[0].reshape(-1)
   
    threshold_ref    = float(np.percentile(flat_ref_conf, conf_percentile))
    keep_ref = flat_ref_conf >= threshold_ref
  
    points_ref_kept = flat_ref_points[keep_ref].astype(np.float32)
    
    voxel_size = _estimate_voxel_size(points_ref_kept,VoxelDedupConfig.auto_scale,VoxelDedupConfig.auto_sample)
    points_ref_denoised,conf_ref_denoised,info = voxel_dedup_pointcloud(points_ref_kept, 
        flat_ref_conf[keep_ref],  voxel_size)

    # depth_denoised
    depth_ref_denoised , conf_ref_denoised = project_world_points_to_depth(
        points_ref_denoised,conf_ref_denoised, pred["intrinsics"][0] ,pred["extrinsics"][0], image_target_wh)
    #depth_da3

    depth_da3 = _get_depth_da3(images_resized[0], da3_model, max(image_target_wh))

    depth_filled,norm_da3 = fill_vggt_depth_by_da3_normals(depth_ref_denoised, depth_da3, pred["intrinsics"][0])
    norm_filled = depth_to_camera_normals(depth_filled, pred["intrinsics"][0])
    # points_filled = backproject_depth_to_world_points(depth_filled, pred["intrinsics"][0], pred["extrinsics"][0])
    
    conf_map,conf_info = compute_confidence(
        depth_v=depth_ref_denoised,
        conf_v=conf_ref_denoised,
        depth_f=depth_filled,
        normal_a= norm_da3,
        normal_f=norm_filled,
        intrinsic=pred["intrinsics"][0],
        rgb=images_uint8[0],
    )
    
  
  
    depth_filled,pred_ref_k = inverse_transform_map(depth_filled, ref_transform,pred["intrinsics"][0])

    conf_map = inverse_transform_map(conf_map,ref_transform)


    # 用 SfM 公制稀疏深度给 (归一化尺度的) depth_filled 标定绝对尺度.
    # 此处 depth_filled 已还原到 sample 原分辨率, 与 SfM sparse_depth 同尺寸.
    depth_filled, sfm_scale, sfm_out = S.calibrate_depth_to_metric(
        sample, depth_filled, ref_idx=0, config=sfm_config)
    # print("sfm metric scale", sfm_scale, sfm_out["info"]["scale"])

    return {
        "images_uint8": images_orig,
        "extrinsics": pred["extrinsics"][0],
        "intrinsics": pred_ref_k,
        "normal": norm_filled,
        "norm_da3": norm_da3,
        "depth_filled": depth_filled,
        "conf_map": conf_map,
        # "sfm_scale": sfm_scale,
        # "sparse_depth": sfm_out["sparse_depth"],
        # "sparse_valid": sfm_out["valid_mask"],
    }
