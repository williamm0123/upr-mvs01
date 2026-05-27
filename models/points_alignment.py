from __future__ import annotations

import numpy as np
import torch

from base.config import PointsAlignmentConfig


def _knn_fill(
    sparse_depth: np.ndarray,
    valid_mask: np.ndarray,
    target_mask: np.ndarray,
    intrinsic: np.ndarray,
    k: int = 5,
    max_dist: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = sparse_depth.shape
    if not target_mask.any():
        return sparse_depth, np.zeros_like(sparse_depth, dtype=np.float32)

    v_src, u_src = np.nonzero(valid_mask)
    if len(v_src) == 0:
        return sparse_depth, np.zeros_like(sparse_depth, dtype=np.float32)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    z_src = sparse_depth[v_src, u_src]
    x_src = (u_src - cx) * z_src / max(fx, 1e-6)
    y_src = (v_src - cy) * z_src / max(fy, 1e-6)
    src_xyz = np.stack([x_src, y_src, z_src], axis=1)

    v_tgt, u_tgt = np.nonzero(target_mask)
    out_depth = sparse_depth.copy()
    out_conf = np.zeros_like(sparse_depth, dtype=np.float32)

    chunk = 4096
    for start in range(0, len(v_tgt), chunk):
        end = min(start + chunk, len(v_tgt))
        vt = v_tgt[start:end]
        ut = u_tgt[start:end]
        depth_guess = np.median(z_src) if len(z_src) > 0 else 1.0
        xt = (ut - cx) * depth_guess / max(fx, 1e-6)
        yt = (vt - cy) * depth_guess / max(fy, 1e-6)
        zt = np.full_like(xt, depth_guess)
        tgt_xyz = np.stack([xt, yt, zt], axis=1)
        d = np.linalg.norm(tgt_xyz[:, None, :] - src_xyz[None, :, :], axis=-1)
        kk = min(k, d.shape[1])
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        rows = np.arange(d.shape[0])[:, None]
        nn_d = d[rows, idx]
        nn_z = z_src[idx]
        w = 1.0 / (nn_d + 1e-3)
        nn_d_min = nn_d.min(axis=1)
        valid = nn_d_min < max_dist
        z_fill = (w * nn_z).sum(axis=1) / w.sum(axis=1).clip(min=1e-6)
        out_depth[vt[valid], ut[valid]] = z_fill[valid]
        out_conf[vt[valid], ut[valid]] = 0.1 + 0.2 * (1.0 - (nn_d_min[valid] / max_dist).clip(0, 1))
    return out_depth, out_conf


def fill_sparse_depth(
    sparse_depth: torch.Tensor,
    confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    config: PointsAlignmentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not config.enabled:
        return sparse_depth, confidence
    B, V, H, W = sparse_depth.shape
    sd_np = sparse_depth.detach().cpu().numpy()
    cf_np = confidence.detach().cpu().numpy()
    K_np = intrinsics.detach().cpu().numpy()
    out_sd = sd_np.copy()
    out_cf = cf_np.copy()
    for b in range(B):
        for v in range(V):
            valid = (cf_np[b, v] > 0.2) & np.isfinite(sd_np[b, v]) & (sd_np[b, v] > 0)
            target = (cf_np[b, v] <= 0.05) | (~np.isfinite(sd_np[b, v]))
            filled, conf_new = _knn_fill(
                sd_np[b, v],
                valid,
                target,
                K_np[b, v],
                k=config.knn_k,
                max_dist=config.knn_max_distance_world,
            )
            out_sd[b, v] = filled
            new_mask = (cf_np[b, v] <= 0.05) & (conf_new > 0)
            out_cf[b, v] = np.where(new_mask, conf_new, cf_np[b, v])
    return (
        torch.from_numpy(out_sd).to(sparse_depth.device),
        torch.from_numpy(out_cf).to(confidence.device),
    )
