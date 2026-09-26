"""Synthetic unit tests for the MoA module (MoA.md §14).

    pytest tests/test_moa_unit.py -q

The network-level test needs CUDA and the DINOv3 weights; it is skipped otherwise.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from base.config_moa import MoAConfig
from models.moa.edge import NeighborhoodBarrier, log_depth_edge, shifted_views, supercover_path
from models.moa.geometry import (
    depth_to_u, interp_along_axis, u_to_depth, window_u_hypotheses,
)
from models.moa.global_affine import ransac_tukey_global_affine, robust_global_affine
from models.moa.local_affine import MultiScaleLocalAffine, spec_closed_form
from models.moa.mixture import apply_mvs_override, mono_proposal, scale_mono_weights
from models.moa.moa import MoACascade

VMIN = torch.tensor(1.0 / 935.0).view(1, 1, 1, 1)
VMAX = torch.tensor(1.0 / 425.0).view(1, 1, 1, 1)


def _solver(**kw) -> tuple[NeighborhoodBarrier, MultiScaleLocalAffine]:
    bar = NeighborhoodBarrier(3)
    return bar, MultiScaleLocalAffine(bar.offset_list, radius=3, **kw)


def _ramp(H=24, W=32, lo=0.3, hi=0.7) -> torch.Tensor:
    xs = torch.linspace(lo, hi, W).view(1, 1, 1, W).expand(1, 1, H, W)
    ys = torch.linspace(0.0, 0.05, H).view(1, 1, H, 1)
    return (xs + ys).contiguous()


def _fit(x, y, *, edge=None, log_z=None, r=None, tau_edge=0.03, tau_jump=0.1, x_valid=None):
    bar, la = _solver()
    B, _, H, W = x.shape
    ones = torch.ones_like(x)
    edge = torch.zeros_like(x) if edge is None else edge
    log_z = torch.zeros_like(x) if log_z is None else log_z
    xv = ones if x_valid is None else x_valid
    p = bar(edge, log_z, xv, tau_edge, tau_jump)["pass"]
    emb = torch.ones(B, 4, H, W) / 2.0
    return la(x, y, xv, ones, ones if r is None else r, emb, p)


# ---------------------------------------------------------------- local WLS
def test_identity():
    x = _ramp()
    res = _fit(x, x.clone())
    assert torch.allclose(res.a, torch.ones_like(res.a), atol=1e-3)
    assert torch.allclose(res.b, torch.zeros_like(res.b), atol=1e-3)
    assert res.valid.min() == 1.0


def test_known_affine():
    x = _ramp()
    res = _fit(x, 1.2 * x + 0.1)
    inner = (slice(None), slice(None), slice(3, -3), slice(3, -3))
    assert torch.allclose(res.a[inner], torch.full_like(res.a[inner], 1.2), atol=1e-2)
    assert torch.allclose(res.b[inner], torch.full_like(res.b[inner], 0.1), atol=1e-2)


def _step_scene(H=24, W=32, c=16):
    """Two surfaces with different affine maps, a DA3 depth jump between columns c-1 | c."""
    col = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W).expand(1, 1, H, W)
    z = torch.where(col < c, 500.0 + 2.0 * col, 800.0 + 2.0 * col)
    x = depth_to_u(z, VMIN, VMAX)
    y = torch.where(col < c, 1.3 * x + 0.05, 0.7 * x - 0.05)
    valid = torch.ones_like(z)
    return z, x, y, log_depth_edge(z, valid), col


def test_piecewise_affine_with_edge():
    z, x, y, edge, col = _step_scene()
    res = _fit(x, y, edge=edge, log_z=z.log())
    left = (col >= 3) & (col <= 14)            # col 15 is the edge cell itself
    right = (col >= 16) & (col <= 28)
    for s in range(3):
        a = res.a[:, s:s + 1]
        fit = res.x_fit[:, s:s + 1]
        # the ridge may shrink the slope toward 1 on shallow windows; the fitted
        # value at p (what the mixture consumes) must still be the own side's
        assert (fit - y)[left | right].abs().max() < 2e-3, f"scale {s}"
        assert (a[left] - 1.3).abs().max() < 0.06, f"scale {s} left"
        assert (a[right] - 0.7).abs().max() < 0.1, f"scale {s} right"


def test_fg_bg_step_zero_cross_weight():
    z, x, _, edge, col = _step_scene()
    bar = NeighborhoodBarrier(3)
    p = bar(edge, z.log(), torch.ones_like(z), 0.03, 0.1)["pass"]
    for k, (dy, dx) in enumerate(bar.offset_list):
        src = col[0, 0]                          # [H, W] column of p
        q_col = src + dx
        crosses = ((src <= 15) & (q_col >= 16)) | ((src >= 16) & (q_col <= 15)) | \
                  ((src == 15) | (q_col == 15))   # any path touching the edge cell
        assert p[0, k][crosses & (q_col >= 0) & (q_col < 32)].abs().sum() == 0, (dy, dx)


def test_no_edge_control_contaminates():
    z, x, y, edge, col = _step_scene()
    ctrl = _fit(x, y, edge=edge, log_z=z.log(), tau_edge=float("inf"), tau_jump=float("inf"))
    guarded = _fit(x, y, edge=edge, log_z=z.log())
    near = col == 14
    # both clusters are nearly points, so the mixed line still passes close to
    # p's own value; the contamination shows up in the slope
    assert (ctrl.a[:, 2:3][near] - 1.3).abs().min() > 0.15, "no barrier: 7x7 should mix both sides"
    assert (guarded.a[:, 2:3][near] - 1.3).abs().max() < 0.06


def test_centered_solve_matches_spec_formula():
    g = torch.Generator().manual_seed(0)
    f64 = torch.float64
    x = 0.5 + 0.01 * torch.randn(30, generator=g, dtype=f64)
    y = 1.1 * x - 0.03 + 0.002 * torch.randn(30, generator=g, dtype=f64)
    w = torch.rand(30, generator=g, dtype=f64)
    la, lb = 1e-4, 1e-3
    a_ref, b_ref = spec_closed_form(w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(),
                                    (w * x * y).sum(), la, lb)
    _, solver = _solver(lambda_a=la, lambda_b=lb, tau_var=0.0, b_max=10.0, a_range=(-10.0, 10.0))
    t = x[3]
    xt, d = x - t, y - x
    v = lambda z: z.reshape(1, 1, 1, 1)
    a, b, off, ok, _, _ = solver._solve(v(w.sum()), v((w * xt).sum()), v((w * xt * xt).sum()),
                                        v((w * d).sum()), v((w * xt * d).sum()), v((w * w).sum()),
                                        v(t), 1.0, torch.ones(1, 1, 1, 1, dtype=f64))
    assert bool(ok)
    assert math.isclose(float(a), float(a_ref), rel_tol=1e-9)
    assert math.isclose(float(b), float(b_ref), rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(float(t + off), float(a_ref * t + b_ref), rel_tol=1e-9)


def test_insufficient_anchors():
    x = _ramp()
    res = _fit(x, 1.2 * x, r=torch.zeros_like(x))
    assert res.valid.sum() == 0
    assert torch.equal(res.a, torch.ones_like(res.a)) and torch.equal(res.b, torch.zeros_like(res.b))
    assert torch.isfinite(res.x_fit).all()
    a, b, ok = robust_global_affine(torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    assert not bool(ok[0]) and float(a[0]) == 1.0 and float(b[0]) == 0.0


def test_flat_mono_offset_only():
    x = torch.full((1, 1, 16, 16), 0.5)
    res = _fit(x, x + 0.03)
    assert res.offset_only.min() == 1.0
    assert torch.allclose(res.a, torch.ones_like(res.a))
    assert torch.allclose(res.b, torch.full_like(res.b, 0.03), atol=1e-5)


def test_negative_affine_rejected():
    x = _ramp()
    res = _fit(x, -1.0 * x + 1.2)
    assert res.valid.sum() == 0
    assert torch.equal(res.a, torch.ones_like(res.a))
    zm = torch.rand(1, 1, 32, 32) + 1.0
    _, _, ok = robust_global_affine(zm, 3.0 - zm, torch.ones_like(zm))
    assert not bool(ok[0])


def test_border_no_wraparound():
    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 4, 4] = 7.0                          # bottom-right corner
    views = shifted_views(x, [(-1, -1), (1, 1)], 3, fill=0.0)
    assert views[0][0, 0, 0, 0] == 0.0           # top-left looking up-left sees padding
    assert views[1][0, 0, 3, 3] == 7.0
    bar = NeighborhoodBarrier(3)
    out = bar(torch.zeros(1, 1, 5, 5), torch.zeros(1, 1, 5, 5), torch.ones(1, 1, 5, 5), 0.03, 0.1)
    k = bar.offset_list.index((-1, -1))
    assert out["pass"][0, k, 0, 0] == 0 and out["pair_valid"][0, k, 0, 0] == 0


def test_supercover_diagonal_blocks_corner_slip():
    assert supercover_path(1, 1) == [(0, 0), (0, 1), (1, 0), (1, 1)] or \
        set(supercover_path(1, 1)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert supercover_path(0, 3) == [(0, 0), (0, 1), (0, 2), (0, 3)]


# ---------------------------------------------------------------- geometry
def test_nonuniform_axis_interpolation():
    u = torch.tensor([0.9, 0.8, 0.5, 0.1]).view(1, 4, 1, 1)       # descending, non-uniform
    pv = torch.tensor([0.1, 0.4, 0.3, 0.2]).view(1, 4, 1, 1)
    for q, expect in [(0.85, 0.25), (0.65, 0.35), (0.3, 0.25), (0.5, 0.3)]:
        v, inside = interp_along_axis(pv, u, torch.tensor(q).view(1, 1, 1, 1))
        assert bool(inside) and math.isclose(float(v), expect, abs_tol=1e-6), (q, float(v))
    v, inside = interp_along_axis(pv, u, torch.tensor(0.95).view(1, 1, 1, 1))
    assert not bool(inside) and float(v) == 0.0


def test_window_stays_in_range_and_keeps_order():
    c = torch.tensor([0.02, 0.5, 0.99]).view(1, 1, 1, 3)
    h = torch.full_like(c, 0.05)
    u = window_u_hypotheses(c, h, 8)
    assert u.min() >= 0.0 and u.max() <= 1.0
    assert (u[:, :-1] > u[:, 1:]).all()                           # descending u = ascending depth
    d = u_to_depth(u, VMIN, VMAX)
    assert (d[:, :-1] < d[:, 1:]).all()


def test_global_affine_recovers_per_sample():
    zm = torch.rand(2, 1, 32, 32) + 0.5
    zt = torch.stack([300.0 * zm[0] + 200.0, 150.0 * zm[1] + 450.0])
    zt[0, 0, :4, :4] += 400.0                                     # outliers
    a, b, ok = robust_global_affine(zm, zt, torch.ones_like(zm))
    assert ok.all()
    assert math.isclose(float(a[0]), 300.0, rel_tol=1e-3) and math.isclose(float(b[0]), 200.0, rel_tol=1e-3)
    assert math.isclose(float(a[1]), 150.0, rel_tol=1e-3) and math.isclose(float(b[1]), 450.0, rel_tol=1e-3)


def _leverage_scene(seed=0):
    """Object anchors on z = 300 zm + 200, plus 1/3 background anchors in DA3's far
    tail (large zm) whose MVS depth is unrelated to the object's affine."""
    g = torch.Generator().manual_seed(seed)
    zm = torch.rand(1, 1, 48, 48, generator=g) + 0.5
    zt = 300.0 * zm + 200.0
    bg = torch.zeros_like(zm, dtype=torch.bool)
    bg[..., :16, :] = True
    zm[bg] = 2.5 + torch.rand(int(bg.sum()), generator=g)
    zt[bg] = 450.0 + 60.0 * torch.rand(int(bg.sum()), generator=g)
    inv_bin = torch.tensor([1.0 / ((1.0 / 300.0 - 1.0 / 900.0) / 47.0)])
    return zm, zt, inv_bin


def test_ransac_global_affine_resists_leverage_outliers():
    zm, zt, inv_bin = _leverage_scene()
    a_h, _, _ = robust_global_affine(zm, zt, torch.ones_like(zm))
    a, b, ok = ransac_tukey_global_affine(zm, zt, torch.ones_like(zm), inv_bin)
    assert bool(ok[0])
    assert math.isclose(float(a[0]), 300.0, rel_tol=1e-3) and math.isclose(float(b[0]), 200.0, rel_tol=1e-3)
    assert abs(float(a_h[0]) - 300.0) > 30.0          # the Huber fit this replaces is dragged off
    *_, wmap = ransac_tukey_global_affine(zm, zt, torch.ones_like(zm), inv_bin, return_weights=True)
    assert wmap.shape == zm.shape
    assert float(wmap[..., :16, :].max()) == 0.0 and float(wmap[..., 16:, :].min()) > 0.9


def test_ransac_global_affine_deterministic_and_fails_cleanly():
    zm, zt, inv_bin = _leverage_scene()
    r1 = ransac_tukey_global_affine(zm, zt, torch.ones_like(zm), inv_bin)
    r2 = ransac_tukey_global_affine(zm, zt, torch.ones_like(zm), inv_bin)
    assert all(torch.equal(x, y) for x, y in zip(r1, r2))
    a, b, ok = ransac_tukey_global_affine(zm, zt, torch.zeros_like(zm), inv_bin)
    assert not bool(ok[0]) and float(a[0]) == 1.0 and float(b[0]) == 0.0
    _, _, ok = ransac_tukey_global_affine(zm, 900.0 - 100.0 * zm, torch.ones_like(zm), inv_bin)
    assert not bool(ok[0])


# ---------------------------------------------------------------- module level
def _moa_inputs(B=1, H=16, W=20, D=8, du=0.004, seed=0, mono_shape="plane"):
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, H).view(1, 1, H, 1)
    xx = torch.linspace(0, 1, W).view(1, 1, 1, W)
    u_true = (0.4 + 0.1 * xx + 0.05 * yy).expand(B, 1, H, W).contiguous()
    z_prev = u_to_depth(u_true, VMIN, VMAX)
    u_hyp = window_u_hypotheses(u_true, torch.full_like(u_true, du * (D - 1) / 2), D)
    dist = (u_hyp - u_true).abs() / du
    prob = torch.softmax(-dist, dim=1)
    if mono_shape == "plane":
        mono = 0.002 * z_prev + 0.3
    elif mono_shape == "step":                                     # a depth discontinuity
        mono = 0.002 * z_prev + 0.3 + 0.5 * (xx >= 0.5).float().expand(B, 1, H, W)
    else:                                                          # disagrees with MVS
        mono = 0.002 * z_prev + 0.3 + 0.2 * torch.sin(8 * xx).expand(B, 1, H, W)
    return dict(
        mono_depth=mono, mono_valid=torch.ones_like(mono), mono_edge=log_depth_edge(mono, torch.ones_like(mono)),
        z_prev=z_prev, prob=prob, u_hyp=u_hyp,
        cv=torch.randn(B, 8, D, H, W, generator=g), n_valid=torch.full((B, D, H, W), 4.0),
        src_std=torch.rand(B, D, H, W, generator=g) * 0.1, num_src=4,
        ref_feat=torch.randn(B, 128, H, W, generator=g),
        vmin=VMIN.expand(B, 1, 1, 1), vmax=VMAX.expand(B, 1, 1, 1))


def _moa(**cfg_kw) -> MoACascade:
    torch.manual_seed(0)
    cfg = MoAConfig(global_min_eff=16.0, **cfg_kw)
    return MoACascade(cfg, fpn_channels=128, num_groups=8, num_depths_stage1=48).eval()


def _force_conf(m: MoACascade, logit: float) -> None:
    for ad in m.adapters:
        last = ad.conf.net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.constant_(last.bias, logit)


def test_high_confidence_conflict_returns_mvs_exactly():
    alpha = torch.ones(1, 1, 2, 2)
    pi = torch.softmax(torch.randn(1, 5, 2, 2), dim=1)
    pt = apply_mvs_override(pi, alpha)
    assert torch.equal(pt[:, 0], torch.ones_like(pt[:, 0])) and pt[:, 1:].abs().max() == 0

    m = _moa()
    _force_conf(m, 40.0)                                           # r_mvs == 1.0 in fp32
    with torch.no_grad():
        out = m(0, **_moa_inputs(mono_shape="wavy"))
    hit = out.alpha == 1.0
    assert hit.float().mean() > 0.3, "fixture should produce many full-confidence conflicts"
    assert torch.equal(out.center_u[hit], out.mvs_u[hit])


def test_low_confidence_conflict_allows_affine():
    """Reliable MVS on the left half (global anchors, mono agrees), unreliable on the
    right half where mono disagrees: there the monocular proposal must win."""
    m = _moa()
    inp = _moa_inputs(mono_shape="plane")
    H, W = inp["z_prev"].shape[-2:]
    right = torch.arange(W).view(1, 1, 1, W).expand(1, 1, H, W) >= W // 2
    xx = torch.linspace(0, 1, W).view(1, 1, 1, W)
    inp["mono_depth"] = inp["mono_depth"] + 0.2 * torch.sin(8 * xx) * right
    inp["mono_edge"] = log_depth_edge(inp["mono_depth"], inp["mono_valid"])
    logit = torch.where(right, torch.tensor(-40.0), torch.tensor(40.0)).float()
    for ad in m.adapters:
        ad.conf.forward = lambda F_mvs, stats, _l=logit: (_l, torch.sigmoid(_l))
        last = ad.mixture.net[-1]
        torch.nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([-5.0, 0.0, 5.0, 0.0, 0.0]))
    with torch.no_grad():
        out = m(0, **inp)
    assert bool(out.global_ok.all())
    assert out.alpha[right].max() < 1e-6
    assert out.mixture_weights[:, 2:3][right].mean() > 0.5
    moved = (out.center_u - out.mvs_u).abs()[right & (out.conflict > 0.5)]
    assert moved.numel() > 0 and moved.mean() > 1e-3


def test_stage4_gain_pulls_centre_back_to_mvs():
    pi = torch.softmax(torch.randn(2, 5, 3, 3), dim=1)
    assert torch.equal(scale_mono_weights(pi, 1.0), pi)
    g = scale_mono_weights(pi, 0.3)
    assert torch.allclose(g.sum(1), torch.ones_like(g[:, 0]), atol=1e-6)
    assert torch.allclose(g[:, 1:], pi[:, 1:] * 0.3)
    assert (g[:, 0] >= pi[:, 0]).all()

    # identical weights and fixture, only the gain differs: the stage-4 centre
    # must move less, and the conflict it detected must be unchanged
    inp = _moa_inputs(mono_shape="wavy")
    outs = {}
    for gain in (1.0, 0.3):
        # edge snap off: it is a hard selection, which would break the exact
        # proportionality the gain is being checked for
        # Huber: the wavy mono has no consistent affine, so RANSAC (correctly) fails the
        # global fit and disables the mono experts; this test needs them active
        m = _moa(moa_gain=(1.0, 1.0, gain), edge_snap=(False, False, False), global_solver=("huber",) * 3)
        _force_conf(m, 0.0)          # r = 0.5: anchors exist, override only partial
        with torch.no_grad():
            outs[gain] = m(2, **inp)
    move = {g: (o.center_u - o.mvs_u).abs().mean() for g, o in outs.items()}
    assert float(move[1.0]) > 1e-4, "fixture should move the centre at gain 1"
    assert torch.allclose(move[0.3], 0.3 * move[1.0], rtol=1e-3), move
    assert torch.allclose(outs[1.0].conflict, outs[0.3].conflict, atol=1e-6)
    assert torch.allclose(outs[1.0].mixture_weights[:, 1:] * 0.3, outs[0.3].mixture_weights[:, 1:], atol=1e-6)


def test_edge_snap_picks_a_surface():
    """At a DA3 depth edge the centre must be exactly the MVS centre or exactly the
    monocular surface — never a blend of the two (MonoMVSNet-style hard choice)."""
    inp = _moa_inputs(mono_shape="step")
    out = {}
    for on in (True, False):
        m = _moa(edge_snap=(on, on, on))
        _force_conf(m, 0.0)
        with torch.no_grad():
            out[on] = m(0, **inp)
    o = out[True]
    assert o.edge.sum() > 0, "fixture should produce depth edges"
    u_mix = (o.mixture_weights * o.experts_u).sum(dim=1, keepdim=True)
    x_prop = mono_proposal(o.mixture_weights_raw, o.experts_u, o.mvs_u)
    at_edge = o.edge > 0.5
    picked_y = o.center_u[at_edge] == o.mvs_u[at_edge]
    picked_x = o.center_u[at_edge] == x_prop[at_edge]
    assert bool((picked_y | picked_x).all()), "edge centre is neither surface"
    assert torch.equal(o.center_u[~at_edge], u_mix[~at_edge])
    snapped_mono = o.edge_snap_mono[at_edge] > 0.5
    assert bool((picked_x | ~snapped_mono).all()) and bool((picked_y | snapped_mono).all())
    # switch off -> plain mixture everywhere
    off = out[False]
    assert torch.equal(off.center_u, (off.mixture_weights * off.experts_u).sum(dim=1, keepdim=True))
    assert float(off.edge_snap_mono.sum()) == 0.0


def test_batch_independence():
    m = _moa()
    a = _moa_inputs(B=1, seed=1)
    b = _moa_inputs(B=1, seed=2, mono_shape="wavy")
    both = {k: (torch.cat([a[k], b[k]]) if torch.is_tensor(a[k]) else a[k]) for k in a}
    with torch.no_grad():
        o_ab = m(1, **both)
        o_a = m(1, **a)
        o_b = m(1, **b)
    assert torch.allclose(o_ab.center_u[:1], o_a.center_u, atol=1e-6)
    assert torch.allclose(o_ab.center_u[1:], o_b.center_u, atol=1e-6)
    assert torch.allclose(o_ab.global_a[1:], o_b.global_a)


def test_gradients_only_reach_moa():
    m = _moa(global_solver=("huber",) * 3).train()   # needs active mono experts on the wavy fixture
    inp = _moa_inputs(mono_shape="wavy")
    for k in ("z_prev", "prob", "cv", "ref_feat", "mono_depth"):
        inp[k] = inp[k].clone().requires_grad_(True)
    out = m(0, **inp)
    loss = out.center_u.sum() + torch.nn.functional.binary_cross_entropy_with_logits(
        out.mvs_conf_logit, torch.full_like(out.mvs_conf_logit, 0.5))
    loss.backward()
    for k in ("z_prev", "prob", "cv", "ref_feat", "mono_depth"):
        assert inp[k].grad is None, f"MoA leaked a gradient into {k}"
    ad = m.adapters[0]
    for name, mod in (("shape", m.shape), ("mixture", ad.mixture), ("conf", ad.conf),
                      ("feat_proj", ad.feat_proj)):
        gs = [p.grad for p in mod.parameters() if p.grad is not None]
        assert gs, f"{name} got no gradient"
        assert all(torch.isfinite(g).all() for g in gs), name
        assert sum(float(g.abs().sum()) for g in gs) > 0, f"{name} gradient is all zero"
    assert m.local.log_tau.grad is not None and torch.isfinite(m.local.log_tau.grad).all()


def test_bf16_autocast_keeps_fp32_statistics():
    m = _moa()
    inp = _moa_inputs(mono_shape="wavy")
    for k in ("prob", "cv", "ref_feat", "n_valid", "src_std"):
        inp[k] = inp[k].to(torch.bfloat16)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(0, **inp)
    for name in ("center_u", "center_depth", "affine_a", "affine_b", "mixture_weights", "mvs_confidence"):
        t = getattr(out, name)
        assert t.dtype == torch.float32, name
        assert torch.isfinite(t).all(), name
    assert torch.allclose(out.mixture_weights.sum(1), torch.ones_like(out.center_u[:, 0]), atol=1e-5)


# ---------------------------------------------------------------- network level
def _dino_weights() -> Path | None:
    try:
        from base.config import ProjectPaths
        p = Path(ProjectPaths().dinov3_weights_file)
        return p if p.is_file() else None
    except Exception:
        return None


@pytest.mark.skipif(not torch.cuda.is_available() or _dino_weights() is None,
                    reason="needs CUDA and the DINOv3 weights")
def test_network_moa_loss_does_not_touch_backbone():
    import dataclasses

    from base.config_moa import MoALossConfig, build_moa_config
    from losses.moa_loss import MoALoss
    from models.network_moa import MoAMVSNet
    from train_moa import synthetic_batch

    cfg = build_moa_config("local")
    cfg = dataclasses.replace(cfg, moa=dataclasses.replace(cfg.moa, global_min_eff=16.0))
    dev = torch.device("cuda")
    net = MoAMVSNet(cfg).to(dev).train()
    batch = synthetic_batch(cfg, dev, 1, (128, 160))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = net(batch)
    assert out["depth_full"].shape == (1, 128, 160)
    for s in (2, 3, 4):
        assert out[f"moa{s}"].center_u.shape[:2] == (1, 1)
    loss_fn = MoALoss(dataclasses.replace(MoALossConfig(), stage_weights=(0.0, 0.0, 0.0, 0.0)), 48)
    loss, _ = loss_fn(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    for mod in (net.fpn, net.sva_pathway, net.cost_volumes, net.decoders, net.dino_sva):
        for p in mod.parameters():
            assert p.grad is None or float(p.grad.abs().max()) == 0.0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.moa.parameters())
