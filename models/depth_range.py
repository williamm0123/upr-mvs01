from __future__ import annotations

import torch
import torch.nn.functional as F

from base.config import DepthRangeConfig
from utils.geometry import make_depth_hypotheses, make_depth_hypotheses_global


def initial_range_from_prior(
    depth_prior: torch.Tensor,
    confidence: torch.Tensor,
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    config: DepthRangeConfig,
    num_depths: int,
    target_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, W = depth_prior.shape
    Ht, Wt = target_hw
    if (H, W) != (Ht, Wt):
        depth_prior = F.interpolate(depth_prior.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False).squeeze(1)
        confidence = F.interpolate(confidence.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False).squeeze(1)

    span = (depth_max - depth_min).view(B, 1, 1)
    valid = torch.isfinite(depth_prior) & (depth_prior > 0) & torch.isfinite(confidence) & (confidence > 0)
    confidence = torch.where(valid, confidence.clamp(0.0, 1.0), torch.zeros_like(confidence))
    sigma_max = config.sigma_max_ratio * span
    sigma = sigma_max * (1.0 - confidence)
    half_range = config.k_sigma * sigma
    half_range = torch.minimum(half_range, span * 0.5)
    global_center = 0.5 * (depth_min + depth_max).view(B, 1, 1)
    depth_center = torch.where(valid, depth_prior, global_center)
    half_range = torch.where(valid, half_range, span * 0.5)
    depth_center = depth_center.clamp(min=depth_min.view(B, 1, 1), max=depth_max.view(B, 1, 1))
    hypos = make_depth_hypotheses(depth_center, half_range, num_depths)
    hypos = hypos.clamp(min=depth_min.view(B, 1, 1, 1), max=depth_max.view(B, 1, 1, 1))
    return hypos, half_range


def refine_range_from_prob(
    prob_volume: torch.Tensor,
    depth_hypos_prev: torch.Tensor,
    depth_pred_prev: torch.Tensor,
    config: DepthRangeConfig,
    num_depths: int,
    interval_ratio: float,
) -> torch.Tensor:
    B, D, H, W = prob_volume.shape
    p_max = prob_volume.max(dim=1).values
    uncertain = p_max < config.uncertain_threshold

    d_min = depth_hypos_prev.min(dim=1).values
    d_max = depth_hypos_prev.max(dim=1).values
    span_prev = d_max - d_min
    half_range_new = 0.5 * span_prev * interval_ratio

    half_range_used = torch.where(uncertain, 0.5 * span_prev, half_range_new)
    hypos = make_depth_hypotheses(depth_pred_prev, half_range_used, num_depths)
    return hypos


def fallback_global_range(
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    num_depths: int,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    return make_depth_hypotheses_global(depth_min, depth_max, num_depths, target_hw[0], target_hw[1])
