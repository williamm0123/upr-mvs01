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


def depth_smooth_l1_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    interval: torch.Tensor,
    err_clamp: float = 3.0,
) -> torch.Tensor:
    """Interval-normalized masked smooth-L1 on the soft-argmin depth.

    The error is measured in units of the per-pixel hypothesis interval, so the
    term stays O(1) at every cascade stage and on every scene scale instead of
    inheriting the dataset's metric units. Errors are clamped at ``err_clamp``
    bins: beyond that a pixel contributes a constant (zero gradient) and pulling
    it back is left to the cross-entropy term, which keeps single hard pixels
    from spiking the batch loss.

    ``mask`` must already encode validity and GT-in-range at the stage
    resolution (see MVSLoss); all inputs share the stage's [B, H, W] shape.
    """
    m = mask.bool()
    if not m.any():
        return depth_pred.new_zeros(())
    err = (depth_pred - depth_gt).abs() / interval.clamp(min=1e-6)
    err = err.clamp(max=err_clamp)
    return F.smooth_l1_loss(err[m], torch.zeros_like(err[m]), beta=1.0)


def depth_cross_entropy_loss(
    logits: torch.Tensor,
    depth_hypos: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked cross-entropy on the logits volume against the nearest bin.

    Works on logits with ``log_softmax`` (numerically stable under AMP) rather
    than ``log(clamp(softmax))``. ``mask`` must already encode validity and
    GT-in-range at the stage resolution (see MVSLoss).
    """
    m = mask.bool()
    if not m.any():
        return logits.new_zeros(())
    log_prob = F.log_softmax(logits.float(), dim=1)
    target_idx = (depth_hypos - depth_gt.unsqueeze(1)).abs().argmin(dim=1)
    ll = log_prob.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    return -ll[m].mean()
