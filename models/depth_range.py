from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from base.config import DepthRangeConfig
from utils.geometry import make_depth_hypotheses_global


@dataclass
class Stage1Hypotheses:
    """Sorted stage-1 hypothesis axis with branch bookkeeping.

    hypos       [B, D, H, W]  merged sorted depth axis (D = num_global + num_local)
    is_local    [B, D, H, W]  1.0 where the bin came from the local branch
    interval    [B, D, H, W]  per-bin physical spacing (h_{i+1}-h_{i-1})/2
    global_idx  [B, Dg, H, W] positions of the global bins on the sorted axis,
                              ascending in depth (gathers a monotone sub-axis)
    local_idx   [B, Dl, H, W] same for the local bins
    global_lo/hi [B]          per-image global-branch bounds (the guard range)
    local_lo/hi [B, H, W]     per-pixel local-branch bounds
    prior/conf  [B, H, W]     prior depth & confidence resampled to stage res
    edge        [B, H, W]     rule-based edge/unreliable map in [0, 1]
    """

    hypos: torch.Tensor
    is_local: torch.Tensor
    interval: torch.Tensor
    global_idx: torch.Tensor
    local_idx: torch.Tensor
    global_lo: torch.Tensor
    global_hi: torch.Tensor
    local_lo: torch.Tensor
    local_hi: torch.Tensor
    prior: torch.Tensor
    conf: torch.Tensor
    edge: torch.Tensor


def _resize_map(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    return F.interpolate(x.unsqueeze(1), size=hw, mode="bilinear", align_corners=False).squeeze(1)


def edge_map_from_prior(prior: torch.Tensor, valid: torch.Tensor, edge_grad_rel: float) -> torch.Tensor:
    """Relative depth-gradient edge/unreliable map in [0, 1], band-widened 3x3.

    Depth (not RGB) gradients: texture edges are not depth edges. Invalid prior
    pixels count as fully unreliable.
    """
    d = torch.where(valid, prior, torch.zeros_like(prior))
    pad = F.pad(d.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    gx = (pad[:, :, 1:-1, 2:] - pad[:, :, 1:-1, :-2]).abs() * 0.5
    gy = (pad[:, :, 2:, 1:-1] - pad[:, :, :-2, 1:-1]).abs() * 0.5
    grad = torch.maximum(gx, gy).squeeze(1)
    rel = grad / (prior.abs() + 1.0)
    e = (rel / max(edge_grad_rel, 1e-6)).clamp(0.0, 1.0)
    e = torch.where(valid, e, torch.ones_like(e))
    return F.max_pool2d(e.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)


def _robust_global_bounds(
    prior: torch.Tensor,
    valid: torch.Tensor,
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    cfg: DepthRangeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image guard bounds: prior quantiles + margin, clamped to the physical
    range and never narrower than global_min_span_frac of it."""
    B = prior.shape[0]
    flat = torch.where(valid, prior, torch.full_like(prior, float("nan"))).view(B, -1)
    q = torch.nanquantile(
        flat.float(),
        torch.tensor([cfg.global_quantile_lo, cfg.global_quantile_hi], device=prior.device, dtype=torch.float32),
        dim=1,
    )  # [2, B]
    q_lo, q_hi = q[0], q[1]
    bad = ~(torch.isfinite(q_lo) & torch.isfinite(q_hi) & (q_hi > q_lo))
    q_lo = torch.where(bad, depth_min, q_lo)
    q_hi = torch.where(bad, depth_max, q_hi)

    margin = cfg.global_margin_ratio * (q_hi - q_lo)
    lo = q_lo - margin
    hi = q_hi + margin

    # Fixed-width window placement: widen to at least min_span, cap at the
    # physical span, then slide the whole window inside [depth_min, depth_max].
    # (A naive expand-then-clamp silently eats the expansion on the clamped
    # side without compensating on the other, so an offset quantile range
    # never reaches the far bound even at min_span_frac=1.0.)
    span_phys = (depth_max - depth_min).clamp_min(1e-3)
    width = torch.maximum(hi - lo, cfg.global_min_span_frac * span_phys)
    width = torch.minimum(width, span_phys)
    center = 0.5 * (lo + hi)
    center = torch.maximum(center, depth_min + 0.5 * width)
    center = torch.minimum(center, depth_max - 0.5 * width)
    lo = (center - 0.5 * width).clamp_min(1e-3)
    hi = center + 0.5 * width
    # degenerate (physical range itself tiny/inverted) -> physical
    degen = hi - lo < 1e-3
    lo = torch.where(degen, depth_min, lo)
    hi = torch.where(degen, depth_max, hi)
    return lo, hi


def _global_branch(
    lo: torch.Tensor,
    hi: torch.Tensor,
    num: int,
    hw: tuple[int, int],
    inverse_depth: bool,
) -> torch.Tensor:
    """[B, num, H, W] confidence-independent guard bins, unique and monotone."""
    B = lo.shape[0]
    t = torch.linspace(0.0, 1.0, num, device=lo.device, dtype=torch.float32).view(1, num)
    if inverse_depth:
        inv = (1.0 / hi).view(B, 1) + ((1.0 / lo) - (1.0 / hi)).view(B, 1) * t
        bins = (1.0 / inv).flip(dims=[1])  # ascending depth
    else:
        bins = lo.view(B, 1) + (hi - lo).view(B, 1) * t
    return bins.view(B, num, 1, 1).expand(B, num, hw[0], hw[1])


def _neighborhood_median_mad(prior: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3x3 robust stats over valid neighbors: (median, MAD, has_any_valid)."""
    B, H, W = prior.shape
    nan = float("nan")
    d = torch.where(valid, prior, torch.full_like(prior, nan))
    patches = F.unfold(d.unsqueeze(1), kernel_size=3, padding=1).view(B, 9, H, W)
    med = patches.nanmedian(dim=1).values
    mad = (patches - med.unsqueeze(1)).abs().nanmedian(dim=1).values
    has_valid = torch.isfinite(med)
    med = torch.nan_to_num(med, nan=0.0)
    mad = torch.nan_to_num(mad, nan=0.0)
    return med, mad, has_valid


def _local_branch(
    prior: torch.Tensor,
    conf: torch.Tensor,
    valid: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    global_interval: torch.Tensor,
    num: int,
    cfg: DepthRangeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[B, num, H, W] dense bins around a spike-robust prior center.

    Width scales with (1 - conf) between a floor and a ceiling measured in
    global-interval units: the floor keeps a confidently-wrong prior from
    locking the search to a point; the ceiling keeps this branch dense (wide
    coverage is the global branch's job). Pixels with no usable prior fall back
    to a half-offset uniform grid over the guard range (raising its effective
    resolution instead of duplicating it).
    """
    B, H, W = prior.shape
    med, mad, has_nbr = _neighborhood_median_mad(prior, valid)

    mad_floor = cfg.spike_min_mad_rel * med.abs().clamp_min(1.0)
    is_spike = valid & has_nbr & ((prior - med).abs() > cfg.spike_k * torch.maximum(mad, mad_floor))
    center = torch.where(is_spike, med, prior)

    gi = global_interval.view(B, 1, 1)
    half_min = cfg.local_half_min_gi * gi
    half_max = cfg.local_half_max_gi * gi
    c = conf.clamp(0.0, 1.0)
    half = half_min + (1.0 - c) * (half_max - half_min)
    half = torch.where(is_spike, half_max.expand_as(half), half)

    lo_b = lo.view(B, 1, 1)
    hi_b = hi.view(B, 1, 1)
    # keep the whole window inside the guard range so clamping never collapses
    # several bins onto the same boundary value
    half = torch.minimum(half, 0.5 * (hi_b - lo_b) - 1e-4)
    center = center.clamp(min=lo_b + half + 1e-4, max=hi_b - half - 1e-4)

    steps = torch.linspace(-1.0, 1.0, num, device=prior.device, dtype=prior.dtype).view(1, num, 1, 1)
    local = center.unsqueeze(1) + half.unsqueeze(1) * steps

    # no-prior fallback: half-offset uniform grid across the guard range
    t = (torch.arange(num, device=prior.device, dtype=prior.dtype) + 0.5) / num
    fb = lo_b.unsqueeze(1) + (hi_b - lo_b).unsqueeze(1) * t.view(1, num, 1, 1)
    usable = valid | has_nbr
    local = torch.where(usable.unsqueeze(1), local, fb.expand_as(local))
    l_lo = local.amin(dim=1)
    l_hi = local.amax(dim=1)
    return local, l_lo, l_hi


def build_stage1_hypotheses(
    depth_prior: torch.Tensor,
    confidence: torch.Tensor,
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    config: DepthRangeConfig,
    target_hw: tuple[int, int],
) -> Stage1Hypotheses:
    """Dual-branch stage-1 axis: prior-independent global guard + prior-guided
    dense local bins, merged and sorted, with branch identity preserved."""
    with torch.no_grad():
        depth_min = depth_min.float()
        depth_max = depth_max.float()
        prior = _resize_map(depth_prior.float(), target_hw)
        conf = _resize_map(confidence.float(), target_hw)

        valid = torch.isfinite(prior) & (prior > 0) & torch.isfinite(conf) & (conf >= 0)
        conf = torch.where(valid, conf.clamp(0.0, 1.0), torch.zeros_like(conf))

        lo, hi = _robust_global_bounds(prior, valid, depth_min, depth_max, config)
        Dg, Dl = config.num_global, config.num_local
        global_interval = (hi - lo) / max(Dg - 1, 1)  # [B]

        g_bins = _global_branch(lo, hi, Dg, target_hw, config.inverse_depth_global)
        l_bins, l_lo, l_hi = _local_branch(prior, conf, valid, lo, hi, global_interval, Dl, config)

        B, _, H, W = g_bins.shape
        hypos = torch.cat([g_bins, l_bins], dim=1)
        branch = torch.cat(
            [hypos.new_zeros(B, Dg, H, W), hypos.new_ones(B, Dl, H, W)], dim=1
        )
        hypos, order = hypos.sort(dim=1)
        is_local = branch.gather(1, order)

        # positions of each branch on the sorted axis (stable sort keeps them in
        # ascending depth order because the merged axis is already sorted)
        _, pos = is_local.to(torch.int8).sort(dim=1, stable=True)
        global_idx = pos[:, :Dg].contiguous()
        local_idx = pos[:, Dg:].contiguous()

        # per-bin spacing (h_{i+1} - h_{i-1}) / 2 with edge replication
        d_next = torch.cat([hypos[:, 1:], hypos[:, -1:]], dim=1)
        d_prev = torch.cat([hypos[:, :1], hypos[:, :-1]], dim=1)
        interval = 0.5 * (d_next - d_prev)
        interval[:, 0] = hypos[:, 1] - hypos[:, 0]
        interval[:, -1] = hypos[:, -1] - hypos[:, -2]
        interval = interval.clamp_min(1e-4)

        edge = edge_map_from_prior(prior, valid, config.edge_grad_rel)

    return Stage1Hypotheses(
        hypos=hypos,
        is_local=is_local,
        interval=interval,
        global_idx=global_idx,
        local_idx=local_idx,
        global_lo=lo,
        global_hi=hi,
        local_lo=l_lo,
        local_hi=l_hi,
        prior=prior,
        conf=conf,
        edge=edge,
    )


def mode_centered_regression(
    prob: torch.Tensor,
    depth_hypos: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expectation restricted to +-window bins around the argmax mode.

    A global soft-argmin over a bimodal posterior (wrong local peak + correct
    global peak) lands between the peaks, on no surface at all; restricting the
    expectation to the winning mode keeps the estimate on a real candidate.
    Returns (depth, sigma_within_mode, argmax_idx [B,1,H,W]).
    """
    D = prob.shape[1]
    idx = prob.argmax(dim=1, keepdim=True)
    offs = torch.arange(-window, window + 1, device=prob.device).view(1, -1, 1, 1)
    nbr = (idx + offs).clamp(0, D - 1)
    p = prob.gather(1, nbr)
    h = depth_hypos.gather(1, nbr)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    depth = (p * h).sum(dim=1)
    var = (p * (h - depth.unsqueeze(1)) ** 2).sum(dim=1)
    sigma = var.clamp_min(1e-12).sqrt()
    return depth, sigma, idx


def refine_range_from_posterior(
    center: torch.Tensor,
    winner_interval: torch.Tensor,
    prob: torch.Tensor,
    edge: torch.Tensor,
    config: DepthRangeConfig,
    num_depths: int,
    global_interval: torch.Tensor,
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
) -> torch.Tensor:
    """Next-stage hypotheses sized by the *winning candidate's* sampling
    precision: a local winner shrinks the search, a global winner keeps a
    correction-sized range, and entropy / edge uncertainty widen it.

    All geometry is detached: hypothesis placement carries no gradient, each
    stage is trained by its own losses.
    """
    with torch.no_grad():
        D = prob.shape[1]
        p = prob.float().clamp_min(1e-8)
        entropy = -(p * p.log()).sum(dim=1) / float(torch.log(torch.tensor(float(D))))
        half = config.range_k * winner_interval * (
            1.0 + config.range_entropy_a * entropy + config.range_edge_b * edge
        )
        gi = global_interval.view(-1, 1, 1)
        half = torch.maximum(half, config.range_min_gi * gi)
        half = torch.minimum(half, config.range_max_gi * gi)
        steps = torch.linspace(-1.0, 1.0, num_depths, device=center.device, dtype=center.dtype)
        hypos = center.detach().unsqueeze(1) + half.unsqueeze(1) * steps.view(1, num_depths, 1, 1)
        hypos = hypos.clamp(
            min=depth_min.view(-1, 1, 1, 1),
            max=depth_max.view(-1, 1, 1, 1),
        )
    return hypos


def fallback_global_range(
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    num_depths: int,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    return make_depth_hypotheses_global(depth_min, depth_max, num_depths, target_hw[0], target_hw[1])
