from __future__ import annotations

import torch
import torch.nn.functional as F

from base.config import DepthRangeConfig
from utils.geometry import make_depth_hypotheses_global


LOCAL_HYPOTHESIS_CONFIDENCE = 0.3
LOCAL_HYPOTHESIS_COUNT = 16
LOCAL_HYPOTHESIS_KERNEL = 3


def _dynamic_hypothesis_steps(
    confidence: torch.Tensor,
    valid: torch.Tensor,
    num_depths: int,
) -> torch.Tensor:
    bins = torch.floor((1.0 - confidence) * 10.0).to(torch.long) + 1
    bins = bins.clamp(1, 10)
    keep_counts = torch.div(num_depths * bins + 9, 10, rounding_mode="floor").clamp(1, num_depths)#
    keep_counts = torch.where(valid, keep_counts, torch.full_like(keep_counts, num_depths))# shape: [B, H, W]
    #将depth进行分组，后面
    idx = torch.arange(num_depths, device=confidence.device, dtype=confidence.dtype).view(1, num_depths, 1, 1) #  [1, D, 1, 1]
    keep_counts_f = keep_counts.unsqueeze(1).to(confidence.dtype) #  [B, 1, H, W]
    quantized = torch.floor(idx * keep_counts_f / float(num_depths))
    quantized = torch.minimum(quantized, keep_counts_f - 1.0)

    denom = (keep_counts - 1).clamp_min(1).unsqueeze(1).to(confidence.dtype)
    return torch.where(
        keep_counts.unsqueeze(1) > 1,
        -1.0 + 2.0 * quantized / denom,
        torch.zeros_like(quantized),
    )


def _local_neighbor_hypotheses(
    depth_center: torch.Tensor,
    depth_prior: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    half_range: torch.Tensor,
) -> torch.Tensor:
    B, H, W = depth_prior.shape
    kernel = LOCAL_HYPOTHESIS_KERNEL
    padding = kernel // 2
    patches = F.unfold(depth_prior.unsqueeze(1), kernel_size=kernel, padding=padding).view(B, kernel * kernel, H, W)
    valid_patches = F.unfold(valid.float().unsqueeze(1), kernel_size=kernel, padding=padding).view(B, kernel * kernel, H, W) > 0.5

    neighbor_mask = torch.ones((1, kernel * kernel, 1, 1), device=depth_prior.device, dtype=torch.bool)
    neighbor_mask[:, kernel * kernel // 2] = False
    neighbor_valid = valid_patches & neighbor_mask

    diff = patches - depth_center.unsqueeze(1)
    large = torch.finfo(depth_prior.dtype).max
    min_diff = diff.masked_fill(~neighbor_valid, large).min(dim=1).values
    max_diff = diff.masked_fill(~neighbor_valid, -large).max(dim=1).values
    has_neighbor = neighbor_valid.any(dim=1)
    min_diff = torch.where(has_neighbor, min_diff, torch.zeros_like(min_diff))
    max_diff = torch.where(has_neighbor, max_diff, torch.zeros_like(max_diff))

    local_min = depth_center + min_diff
    local_max = depth_center + max_diff
    depth_min = depth_min.view(B, 1, 1)
    depth_max = depth_max.view(B, 1, 1)
    local_min = local_min.clamp(min=depth_min, max=depth_max)
    local_max = local_max.clamp(min=depth_min, max=depth_max)

    local_steps = torch.linspace(
        0.0,
        1.0,
        LOCAL_HYPOTHESIS_COUNT,
        device=depth_prior.device,
        dtype=depth_prior.dtype,
    ).view(1, LOCAL_HYPOTHESIS_COUNT, 1, 1)
    local_hypos = local_min.unsqueeze(1) + (local_max - local_min).unsqueeze(1) * local_steps

    fallback_steps = torch.linspace(
        -1.0,
        1.0,
        LOCAL_HYPOTHESIS_COUNT,
        device=depth_prior.device,
        dtype=depth_prior.dtype,
    ).view(1, LOCAL_HYPOTHESIS_COUNT, 1, 1)
    fallback_hypos = depth_center.unsqueeze(1) + half_range.unsqueeze(1) * fallback_steps
    use_local = valid & has_neighbor & (confidence >= LOCAL_HYPOTHESIS_CONFIDENCE)
    return torch.where(use_local.unsqueeze(1), local_hypos, fallback_hypos)


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
    valid = torch.isfinite(depth_prior) & (depth_prior > 0) & torch.isfinite(confidence) & (confidence >= 0)
    confidence = torch.where(valid, confidence.clamp(0.0, 1.0), torch.zeros_like(confidence))
    sigma_max = config.sigma_max_ratio * span
    sigma = sigma_max * (1.0 - confidence)
    half_range = config.k_sigma * sigma
    half_range = torch.minimum(half_range, span * 0.5)
    global_center = 0.5 * (depth_min + depth_max).view(B, 1, 1)
    depth_center = torch.where(valid, depth_prior, global_center)
    half_range = torch.where(valid, half_range, span * 0.5)
    depth_center = depth_center.clamp(min=depth_min.view(B, 1, 1), max=depth_max.view(B, 1, 1))

    steps = _dynamic_hypothesis_steps(confidence, valid, num_depths)
    hypos = depth_center.unsqueeze(1) + half_range.unsqueeze(1) * steps
    local_hypos = _local_neighbor_hypotheses(
        depth_center,
        depth_prior,
        confidence,
        valid,
        depth_min,
        depth_max,
        half_range,
    )
    hypos = torch.cat([hypos, local_hypos], dim=1)
    hypos = hypos.clamp(min=depth_min.view(B, 1, 1, 1), max=depth_max.view(B, 1, 1, 1))
    hypos = hypos.sort(dim=1).values
    return hypos, half_range


def refine_range_from_prob(
    depth_hypos_prev: torch.Tensor,
    depth_pred_prev: torch.Tensor,
    sigma_prev: torch.Tensor,
    config: DepthRangeConfig,
    num_depths: int,
    interval_ratio: float,
    adaptive: bool = True,
) -> torch.Tensor:

    d_min = depth_hypos_prev.min(dim=1).values
    d_max = depth_hypos_prev.max(dim=1).values
    span_prev = d_max - d_min
    half_floor = 0.5 * span_prev * interval_ratio
    if adaptive:
        half = torch.minimum(
            torch.maximum(config.k_sigma * sigma_prev, half_floor),
            0.5 * span_prev,
        )
    else:
        half = half_floor

    steps = torch.linspace(
        -1.0, 1.0, num_depths, device=depth_pred_prev.device, dtype=depth_pred_prev.dtype
    )
    return depth_pred_prev.unsqueeze(1) + half.unsqueeze(1) * steps.view(1, num_depths, 1, 1)


def fallback_global_range(
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    num_depths: int,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    return make_depth_hypotheses_global(depth_min, depth_max, num_depths, target_hw[0], target_hw[1])
