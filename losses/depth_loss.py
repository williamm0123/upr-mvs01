from __future__ import annotations

import torch
import torch.nn.functional as F




def normalized_huber_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale-normalized smooth-L1 on the regressed depth over ALL masked pixels.

    No hard error clamp: the smooth-L1 linear tail already bounds the gradient,
    while a clamp before it zeroes the gradient of exactly the far-off pixels
    that most need pulling back (the old supervision blind spot). ``scale`` is
    the per-pixel (or broadcastable) normalizer in depth units, detached by the
    caller. ``weight`` optionally re-weights pixels (e.g. edge-band boost).
    """
    m = mask.bool()
    if not m.any():
        return depth_pred.new_zeros(())
    err = (depth_pred - depth_gt) / scale.clamp(min=1e-6)
    per_px = F.smooth_l1_loss(err, torch.zeros_like(err), beta=1.0, reduction="none")
    if weight is None:
        return per_px[m].mean()
    w = weight[m]
    return (per_px[m] * w).sum() / w.sum().clamp(min=1e-6)


@torch.no_grad()
def huber_contributions(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    split: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Split the reported regression loss into two ADDITIVE shares.

    Returns ``(contrib_where_split, contrib_where_not, frac_split)``. Both
    contributions use the same denominator as ``normalized_huber_loss``, so they
    sum to it exactly — which is what "how much of stage 4's loss is
    out-of-window pixels" actually asks. Reporting two separately-normalised
    means instead would answer a different question and could not be compared
    against the logged total.

    Diagnostic only: no_grad, and it recomputes the per-pixel term rather than
    threading extra returns through the loss path.
    """
    m = mask.bool()
    if not m.any():
        return 0.0, 0.0, 0.0
    err = (depth_pred - depth_gt) / scale.clamp(min=1e-6)
    per_px = F.smooth_l1_loss(err, torch.zeros_like(err), beta=1.0, reduction="none")
    w = torch.ones_like(per_px) if weight is None else weight
    denom = w[m].sum().clamp(min=1e-6)
    s = m & split.bool()
    n = m & ~split.bool()
    return (
        float((per_px[s] * w[s]).sum() / denom) if s.any() else 0.0,
        float((per_px[n] * w[n]).sum() / denom) if n.any() else 0.0,
        float(s.sum() / m.sum().clamp(min=1)),
    )


def soft_label_cross_entropy(
    logits: torch.Tensor,
    depth_hypos: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy against a two-bin linear soft label on a (possibly
    irregular) sorted hypothesis axis.

    The GT depth is assigned to the two straddling bins with linear
    interpolation weights, so a coarse guard bin and a dense local bin next to
    the same GT are supervised fairly instead of a one-hot nearest-neighbour
    picking one of several near-correct candidates at random. Softmax is taken
    over whatever candidate set ``logits`` carries (full 64-axis or a gathered
    branch), which is what makes the branch-auxiliary losses possible.

    ``mask`` must already encode validity and GT-within-range for this
    candidate set; all tensors share [B, D, H, W] / [B, H, W] shapes and
    ``depth_hypos`` is ascending along dim 1.
    """
    m = mask.bool()
    if not m.any():
        return logits.new_zeros(())
    D = depth_hypos.shape[1]
    log_prob = F.log_softmax(logits.float(), dim=1)

    hs = depth_hypos.permute(0, 2, 3, 1).contiguous()            # [B, H, W, D]
    gt = depth_gt.unsqueeze(-1)                                   # [B, H, W, 1]
    gt_c = gt.clamp(min=hs[..., :1], max=hs[..., -1:])
    r = torch.searchsorted(hs, gt_c.contiguous()).clamp(1, D - 1)  # right bin
    left = r - 1
    h_l = hs.gather(-1, left)
    h_r = hs.gather(-1, r)
    denom = (h_r - h_l).clamp(min=1e-6)
    q_l = ((h_r - gt_c) / denom).clamp(0.0, 1.0)
    q_r = 1.0 - q_l

    lp = log_prob.permute(0, 2, 3, 1)
    ce = -(q_l * lp.gather(-1, left) + q_r * lp.gather(-1, r)).squeeze(-1)  # [B, H, W]
    if weight is None:
        return ce[m].mean()
    w = weight[m]
    return (ce[m] * w).sum() / w.sum().clamp(min=1e-6)


# Backwards-compatible aliases (older scripts import these names).


