from __future__ import annotations

import torch
import torch.nn.functional as F


def depth_l1_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.bool() & (depth_gt > 0)
    if not m.any():
        return depth_pred.new_zeros(())
    diff = (depth_pred - depth_gt).abs()
    return diff[m].mean()


def depth_cross_entropy_loss(
    prob_volume: torch.Tensor,
    depth_hypos: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    B, D, H, W = prob_volume.shape
    if depth_gt.shape[-2:] != (H, W):
        depth_gt = F.interpolate(depth_gt.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1)
    m = mask.bool() & (depth_gt > 0)
    if not m.any():
        return prob_volume.new_zeros(())
    diff = (depth_hypos - depth_gt.unsqueeze(1)).abs()
    target_idx = diff.argmin(dim=1)
    log_prob = torch.log(prob_volume.clamp(min=1e-8))
    ll = log_prob.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    return -ll[m].mean()
