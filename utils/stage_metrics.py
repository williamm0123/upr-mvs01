"""Per-stage cascade diagnostics — the cheap replacement for per-stage fusion.

Running a point-cloud evaluation per cascade stage would answer "is the cascade
refining?" at the cost of one fusion + one DTU eval per stage per checkpoint.
It also answers it badly: point clouds from different stages have different
point densities, so Acc/Comp mixes resolution with refinement, and a single
Acc number cannot distinguish the two failure modes that need opposite fixes.

Everything here is computed from tensors the forward pass already produced, so
the marginal cost is a handful of reductions. The decomposition that matters:

    oracle_err    = min_i |h_i - gt|      the best this hypothesis axis allows
    selection_err = |d_hat - gt| - oracle_err

  * oracle_err large  -> the range policy put no candidate near GT.
    Fix the window (lambda / floor / sampling), not the matching.
  * selection_err large -> a good candidate existed and the posterior missed it.
    Fix the features / 3D regularizer / mode_window, not the window.

``selection_err`` is signed on purpose: sub-bin regression legitimately beats
the nearest hypothesis, so a small negative value is a healthy sign, not a bug.

Every stage is measured twice:

  * ``native`` — at the stage's own resolution, against nearest-downsampled GT.
    This is the honest view of what the hypothesis axis did.
  * ``full``   — bilinearly upsampled to the GT resolution. Only this view is
    comparable across stages, because the native ones differ in resolution.

The risk-coverage block answers the fusion question without fusing: point-cloud
Acc is the mean error of the *kept* points, so sorting pixels by a candidate
confidence channel and reporting the mean error of the top-k% traces out the
Acc/Comp frontier in pixel space. ``ause_*`` is the gap to the oracle ordering
(sort by true error) — the standard sparsification measure of whether a
confidence channel predicts error at all. A channel whose AUSE is ~0 relative
to random is useless for fusion no matter what threshold is picked.

All values are returned as ``{name: (weighted_sum, weight)}`` so the caller can
aggregate them pixel-weighted across a logging window and across DDP ranks
instead of averaging per-batch means.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

Accum = dict[str, tuple[float, float]]

# Sub-millimetre thresholds are deliberate: DTU Acc lives at ~0.3 mm, so
# acc_2mm saturates long before the metric we actually care about moves.
ACC_THRESH_MM: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
RETENTION: tuple[float, ...] = (0.8, 0.6, 0.4, 0.2)
# A hypothesis axis with duplicate candidates has zero-width bins there. Well
# below any real DTU bin (the finest is ~0.27 mm), so this only catches
# degeneracy, never a legitimately fine axis.
DEGENERATE_BIN_MM: float = 1e-3


def _put(out: Accum, name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
    """Accumulate a pixel-weighted (sum, count) pair over ``mask``."""
    n = int(mask.sum())
    if n == 0:
        return
    out[name] = (float(values[mask].sum()), float(n))


def _put_scalar(out: Accum, name: str, value: float, weight: float = 1.0) -> None:
    out[name] = (float(value) * weight, float(weight))


def _to_res(x: torch.Tensor, hw: tuple[int, int], mode: str = "nearest") -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(x.unsqueeze(1).float(), size=hw, mode=mode, **kw).squeeze(1)


def hypothesis_intervals(hypos: torch.Tensor) -> torch.Tensor:
    """[B, D, H, W] Voronoi width of every hypothesis, (h_{i+1}-h_{i-1})/2.

    Works on the irregular stage-1 axis as well as the uniform ones, which is
    why the diagnostics never divide by ``(hi-lo)/(D-1)``: on a non-uniform axis
    that average says nothing about the bin the winner actually sits in.
    """
    if hypos.shape[1] < 2:
        return torch.full_like(hypos, 1e-6)
    d_next = torch.cat([hypos[:, 1:], hypos[:, -1:]], dim=1)
    d_prev = torch.cat([hypos[:, :1], hypos[:, :-1]], dim=1)
    itv = 0.5 * (d_next - d_prev)
    itv[:, 0] = hypos[:, 1] - hypos[:, 0]
    itv[:, -1] = hypos[:, -1] - hypos[:, -2]
    return itv.abs().clamp_min(1e-6)


def _risk_coverage(
    err: torch.Tensor,
    conf: torch.Tensor,
    fracs: Sequence[float],
) -> dict[float, float]:
    """Mean error of the top-``f`` fraction of pixels ranked by ``conf``."""
    n = err.numel()
    if n < 256:
        return {}
    order = torch.argsort(conf, descending=True)
    csum = err[order].cumsum(0)
    out: dict[float, float] = {}
    for f in fracs:
        k = max(1, int(n * f))
        out[f] = float(csum[k - 1] / k)
    return out


@torch.no_grad()
def stage_diagnostics(
    outputs: dict,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    stage_names: Sequence[str],
    corrupt_mask: torch.Tensor | None = None,
    retention: Sequence[float] = RETENTION,
    acc_thresh: Sequence[float] = ACC_THRESH_MM,
    risk_coverage: bool = True,
) -> Accum:
    """Diagnostics for every cascade stage present in ``outputs``.

    ``depth_gt`` / ``mask`` are at full image resolution; ``mask`` must already
    exclude GT outside the scene's physical depth range (the loss and the
    pixel metrics both do, so including it here would make the numbers
    incomparable with them).
    """
    out: Accum = {}
    gt_full = depth_gt.float()
    valid_full = mask.bool() & (gt_full > 0)
    if not valid_full.any():
        return out
    full_hw = tuple(gt_full.shape[-2:])

    prev_err_full: torch.Tensor | None = None
    for si, name in enumerate(stage_names):
        stage = outputs.get(name)
        if stage is None:
            continue
        hypos = stage["depth_hypos"].float()
        depth = stage["depth"].float()
        prob = stage["prob"].float()
        hw = tuple(depth.shape[-2:])
        D = hypos.shape[1]

        # ------------------------------ native ------------------------------ #
        gt_n = _to_res(gt_full, hw)
        valid_n = _to_res(mask.float(), hw).bool() & (gt_n > 0)
        if not valid_n.any():
            continue
        err_n = (depth - gt_n).abs()

        oracle = (hypos - gt_n.unsqueeze(1)).abs().amin(dim=1)
        _put(out, f"{name}/mae_native", err_n, valid_n)
        _put(out, f"{name}/oracle_err", oracle, valid_n)
        _put(out, f"{name}/selection_err", err_n - oracle, valid_n)

        # window coverage: which side did we lose the pixel on, and by how much
        lo = hypos.amin(dim=1)
        hi = hypos.amax(dim=1)
        oor_lo = gt_n < lo
        oor_hi = gt_n > hi
        in_range = ~(oor_lo | oor_hi)
        _put(out, f"{name}/in_range", in_range.float(), valid_n)
        _put(out, f"{name}/oor_lo", oor_lo.float(), valid_n)
        _put(out, f"{name}/oor_hi", oor_hi.float(), valid_n)
        oor_dist = torch.where(oor_lo, lo - gt_n, torch.zeros_like(gt_n))
        oor_dist = torch.where(oor_hi, gt_n - hi, oor_dist)
        oor_any = valid_n & ~in_range
        _put(out, f"{name}/oor_dist_mm", oor_dist, oor_any)
        _put(out, f"{name}/window_mm", hi - lo, valid_n)

        # bin width AT THE WINNER, not the axis average: on the irregular
        # stage-1 axis the mean bin is dominated by the coarse guard bins and
        # says nothing about the resolution the winning pixel actually got.
        intervals = hypothesis_intervals(hypos)
        mode_idx = stage.get("mode_idx")
        if mode_idx is None:
            mode_idx = prob.argmax(dim=1, keepdim=True)
        bin_mode = intervals.gather(1, mode_idx).squeeze(1)
        _put(out, f"{name}/bin_mode_mm", bin_mode, valid_n)

        # err / bin separates "quantisation-limited" from "matching-limited",
        # but the plain mean of the per-pixel RATIO is not robust: a per-plane
        # clamp at the scene boundary produces duplicate hypotheses and hence
        # near-zero Voronoi bins, and a handful of those pixels dominate. On the
        # 1k-step run this read 3268 at stage 2 while the ratio of means was 3.5.
        # So report five things and let them disagree visibly:
        #   * _rom     — exact ratio of means (sum err / sum bin), aggregation-safe
        #   * plain    — mean of the ratio, EXCLUDING degenerate bins
        #   * median/p90 — order statistics, which the ratio-of-means hides when
        #                  high error correlates with small bins
        #   * degenerate_bin_rate — the thing that broke it; must be ~0 once the
        #                  window is placed by sliding instead of clamping
        degen = bin_mode < DEGENERATE_BIN_MM
        ok = valid_n & ~degen
        _put(out, f"{name}/degenerate_bin_rate", degen.float(), valid_n)
        # (sum, count) = (sum err, sum bin) makes the window/rank average an
        # exact ratio of means rather than an average of per-batch ratios.
        if valid_n.any():
            out[f"{name}/err_over_bin_rom"] = (
                float(err_n[valid_n].sum()), float(bin_mode[valid_n].sum().clamp_min(1e-6)))
        if ok.any():
            ratio = (err_n / bin_mode.clamp_min(1e-6))[ok]
            _put(out, f"{name}/err_over_bin", err_n / bin_mode.clamp_min(1e-6), ok)
            _put_scalar(out, f"{name}/err_over_bin_median", float(ratio.median()))
            _put_scalar(out, f"{name}/err_over_bin_p90", float(torch.quantile(
                ratio.float() if ratio.numel() < 8_000_000 else ratio[::4].float(), 0.9)))

        # posterior shape. sigma_full is the full-axis spread about the reported
        # depth — this is the quantity a sigma-driven range policy would consume,
        # so measuring it now says in advance whether it carries any signal.
        p = prob.clamp_min(1e-12)
        sigma_full = ((p * (hypos - depth.unsqueeze(1)) ** 2).sum(dim=1)).clamp_min(0).sqrt()
        pmax = prob.amax(dim=1)
        ent = -(p * p.log()).sum(dim=1) / float(torch.log(torch.tensor(float(max(D, 2)))))
        top2 = prob.topk(min(2, D), dim=1).values
        margin = top2[:, 0] - top2[:, 1] if D >= 2 else torch.ones_like(pmax)
        _put(out, f"{name}/sigma_full_mm", sigma_full, valid_n)
        _put(out, f"{name}/pmax", pmax, valid_n)
        _put(out, f"{name}/entropy", ent, valid_n)
        _put(out, f"{name}/margin", margin, valid_n)
        if "sigma" in stage:
            _put(out, f"{name}/sigma_mode_mm", stage["sigma"].float(), valid_n)

        # ------------------------------- full ------------------------------- #
        depth_up = (depth if hw == full_hw else F.interpolate(
            depth.unsqueeze(1), size=full_hw, mode="bilinear", align_corners=False).squeeze(1))
        err_f = (depth_up - gt_full).abs()
        _put(out, f"{name}/mae", err_f, valid_full)
        # clipped MAE: the plain mean is dominated by a few far-off pixels, so a
        # real refinement gain is invisible in it until the tail is fixed.
        _put(out, f"{name}/mae_clip2", err_f.clamp(max=2.0), valid_full)
        for t in acc_thresh:
            _put(out, f"{name}/acc_{t:g}mm", (err_f < t).float(), valid_full)

        e_valid = err_f[valid_full]
        _put_scalar(out, f"{name}/median", float(e_valid.median()))
        _put_scalar(out, f"{name}/p90", float(torch.quantile(
            e_valid.float() if e_valid.numel() < 8_000_000 else e_valid[::4].float(), 0.9)))

        if prev_err_full is not None:
            improved = (err_f < prev_err_full).float()
            _put(out, f"{name}/improve_ratio", improved, valid_full)
        prev_err_full = err_f

        # ------------------------- risk-coverage ---------------------------- #
        # Only on the last stage: this is the channel that fusion will threshold,
        # and one argsort per extra stage buys nothing.
        if risk_coverage and si == len(stage_names) - 1:
            conf_up = {
                "pmax": _to_res(pmax, full_hw, "bilinear"),
                "margin": _to_res(margin, full_hw, "bilinear"),
                "negsigma": -_to_res(sigma_full, full_hw, "bilinear"),
                "negentropy": -_to_res(ent, full_hw, "bilinear"),
            }
            e = err_f[valid_full]
            oracle_rc = _risk_coverage(e, -e, retention)
            for f, v in oracle_rc.items():
                _put_scalar(out, f"rc/oracle@{f:g}", v)
            for ch, cmap in conf_up.items():
                rc = _risk_coverage(e, cmap[valid_full], retention)
                if not rc:
                    continue
                for f, v in rc.items():
                    _put_scalar(out, f"rc/{ch}@{f:g}", v)
                ause = sum(rc[f] - oracle_rc[f] for f in rc) / max(len(rc), 1)
                _put_scalar(out, f"rc/ause_{ch}", ause)

        # --------------------------- partitions ----------------------------- #
        if si == len(stage_names) - 1:
            edge = stage.get("edge")
            if edge is not None:
                e_up = _to_res(edge.float(), full_hw, "bilinear") > 0.5
                _put(out, f"{name}/mae_edge", err_f, valid_full & e_up)
                _put(out, f"{name}/mae_flat", err_f, valid_full & ~e_up)
            if corrupt_mask is not None:
                cm = _to_res(corrupt_mask.float(), full_hw).bool()
                _put(out, f"{name}/mae_prior_corrupt", err_f, valid_full & cm)
                _put(out, f"{name}/mae_prior_clean", err_f, valid_full & ~cm)

    _put_scalar(out, "valid_px", float(valid_full.sum()))
    return out
