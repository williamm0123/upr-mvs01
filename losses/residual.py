from __future__ import annotations

import torch
import torch.nn.functional as F


def residual_laplacian_loss(
    depth_pred: torch.Tensor,
    depth_prior: torch.Tensor,
    mask: torch.Tensor,
    confidence: torch.Tensor | None = None,
    b_scale: float = 0.1,
    min_confidence: float = 0.3,
    relative: bool = True,
) -> torch.Tensor:
    if depth_prior.shape != depth_pred.shape:
        depth_prior = F.interpolate(
            depth_prior.unsqueeze(1), size=depth_pred.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
    if confidence is not None and confidence.shape != depth_pred.shape:
        confidence = F.interpolate(
            confidence.unsqueeze(1).float(), size=depth_pred.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
    m = mask.bool() & torch.isfinite(depth_prior) & (depth_prior > 0)
    if confidence is not None:
        m = m & torch.isfinite(confidence) & (confidence >= min_confidence)
    if not m.any():
        return depth_pred.new_zeros(())
    r = (depth_pred - depth_prior).abs()
    if relative:
        r = r / depth_prior.abs().clamp(min=1e-6)
    b = torch.as_tensor(b_scale, dtype=r.dtype, device=r.device).clamp(min=1e-6)
    return (r / b + torch.log(2.0 * b))[m].mean()
