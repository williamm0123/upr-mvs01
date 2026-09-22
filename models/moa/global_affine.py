"""Per-sample robust global affine ``a * Z_mono + b ~ Z_mvs`` (MoA.md §6).

The DA3 cache holds raw relative depth (no SfM pre-scaling), so this fit is the
only thing that puts the monocular map into metric depth. The spec's
fallback (a=1, b=0) presumes an SfM-metric input; on raw DA3 it would inject
an arbitrary scale. A failed fit therefore returns ``ok=False`` and the caller
disables every monocular expert for that sample (mixture -> MVS).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def robust_global_affine(z_mono: torch.Tensor, z_mvs: torch.Tensor, weight: torch.Tensor,
                         n_iter: int = 3, huber_k: float = 1.345, min_eff: float = 64.0,
                         max_points: int = 200_000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z_mono, z_mvs, weight: [B,1,H,W] (weight in [0,1], already 0 on invalid pixels).

    ``min_eff`` is a reliability-weighted anchor count (sum of weights): a map
    of uniformly unreliable MVS must not pass as "plenty of anchors" just
    because the weights are uniform. Each sample is solved independently in
    FP32 with Huber/MAD IRLS; nothing is shared across the batch.
    Returns (a [B], b [B], ok [B] bool).
    """
    B = z_mono.shape[0]
    dev = z_mono.device
    a_out = torch.ones(B, device=dev)
    b_out = torch.zeros(B, device=dev)
    ok_out = torch.zeros(B, dtype=torch.bool, device=dev)
    for i in range(B):
        x = z_mono[i].reshape(-1).float()
        y = z_mvs[i].reshape(-1).float()
        w = weight[i].reshape(-1).float()
        m = (w > 0) & torch.isfinite(x) & torch.isfinite(y) & (x > 0) & (y > 0)
        if int(m.sum()) < 3:
            continue
        x, y, w = x[m], y[m], w[m]
        if float(w.sum()) < min_eff:
            continue
        if x.numel() > max_points:
            idx = torch.linspace(0, x.numel() - 1, max_points, device=dev).long()
            x, y, w = x[idx], y[idx], w[idx]
        w = w / w.max()
        h = torch.ones_like(w)
        a = b = None
        for _ in range(n_iter + 1):
            ww = w * h
            sw = ww.sum().clamp_min(1e-12)
            mx = (ww * x).sum() / sw
            my = (ww * y).sum() / sw
            dx = x - mx
            vxx = (ww * dx * dx).sum()
            if float(vxx) <= 1e-12 * float(sw):
                a = None
                break
            a = (ww * dx * (y - my)).sum() / vxx
            b = my - a * mx
            r = y - (a * x + b)
            scale = 1.4826 * r.abs().median() + 1e-6
            h = (huber_k * scale / r.abs().clamp_min(1e-12)).clamp(max=1.0)
        if a is None or not (torch.isfinite(a) and torch.isfinite(b)) or float(a) <= 0.0:
            continue
        a_out[i], b_out[i], ok_out[i] = a, b, True
    return a_out, b_out, ok_out
