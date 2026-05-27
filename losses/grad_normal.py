from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.geometry import depth_to_normal


def _spatial_grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[:, :, 1:] - x[:, :, :-1]
    dy = x[:, 1:, :] - x[:, :-1, :]
    return dx, dy


def depth_gradient_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.bool() & (depth_gt > 0)
    if not m.any():
        return depth_pred.new_zeros(())
    pdx, pdy = _spatial_grad(depth_pred)
    gdx, gdy = _spatial_grad(depth_gt)
    mx = m[:, :, 1:] & m[:, :, :-1]
    my = m[:, 1:, :] & m[:, :-1, :]
    lx = (pdx - gdx).abs()[mx].mean() if mx.any() else depth_pred.new_zeros(())
    ly = (pdy - gdy).abs()[my].mean() if my.any() else depth_pred.new_zeros(())
    return 0.5 * (lx + ly)


def normal_consistency_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    K: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.bool() & (depth_gt > 0)
    if not m.any():
        return depth_pred.new_zeros(())
    n_pred = depth_to_normal(depth_pred, K)
    n_gt = depth_to_normal(depth_gt, K)
    cos = (n_pred * n_gt).sum(dim=1).clamp(-1, 1)
    return (1.0 - cos)[m].mean()
