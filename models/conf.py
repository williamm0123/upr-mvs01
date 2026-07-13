"""Per-pixel prior confidence for cost-volume depth-range allocation.

Implements ``depth_prior_confidence.md``.

Core principle: we only trust DA3's **normals** (``n_a``), never its absolute
depth scale.  Therefore *no DA3 absolute depth* enters any formula here; every
piece of evidence is built on whether normals / geometry are mutually
consistent.  The confidence is assembled as

    c_0 (base band)  ->  f_*  (reliability factors)  ->  O (outlier gate)  ->  conf

and ``conf`` is finally mapped to an inverse-depth search bracket
``[d_min, d_max]`` per pixel.

Inputs (all numpy, shape (H, W) unless noted), obtainable in
``data/norm_fill.py`` / ``test01.py``:

    depth_v      d_v   : denoised VGGT projected depth      (depth_ref_denoised)
    conf_v       c_v   : per-pixel VGGT conf, same z-buffer  (conf_denoised)
    depth_f      d_f   : normal-filled dense depth           (filled_out[0])
    normal_a     n_a   : DA3 normals (H, W, 3)               (filled_out[1]["normal_da3"])
    normal_f     n_f   : normals of the filled depth (H,W,3) (norm_filled)
    intrinsic          : 3x3 camera matrix                   (pred["intrinsics"][0])
    rgb (optional)     : H,W,3 uint8 image for RGB edges     (images_uint8[0])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Configuration (see the parameter table, section 7 of the spec)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfidenceConfig:
    # Step 1 - base band
    theta_min_deg: float = 3.0       # fill-region normal angle -> conf mapping
    theta_max_deg: float = 45.0
    conf_norm_pct: tuple[float, float] = (5.0, 95.0)  # robust min-max for c_v

    # Step 2 - reliability factors
    tau_xc_deg: float = 20.0         # anchor normal cross-consistency tolerance
    tau_loc: float = 2.5             # local residual tolerance (MAD units)
    tau_e: float = 0.3               # normal-edge instability scale
    lam: float = 4.0                 # geodesic distance decay scale (pixels)

    # windows
    pca_window: int = 7              # n_v local plane fit window
    pca_min_pts: int = 6             # min anchors in window, else f_xc = 1
    loc_window: int = 5              # local robust fit / MAD window
    iso_window: int = 7              # isolation window
    iso_frac: float = 0.15           # fraction of window that must be "outlier-like"
    iso_resid_k: float = 2.0         # residual threshold (MAD units) for isolation

    # Step 3 - outlier gate
    k: float = 4.0                   # outlier residual threshold (MAD units)
    tau_O: float = 2.0               # outlier gate softness

    # Step 5 - range allocation
    kappa: float = 0.2               # normal-inconsistency -> depth uncertainty
    sigma_conf_scale: float = 0.05   # VGGT-conf -> relative depth uncertainty
    m: float = 2.5                   # metric floor multiplier
    gamma: float = 1.75              # range vs (1-conf) curve
    rho_min: float = 0.02            # inverse-depth relative half-width bounds
    rho_max: float = 0.30
    rho_eff_cap: float = 0.90        # keep u_i > 0


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _robust_norm(x, p_lo, p_hi, mask=None):
    """N(x; p_lo, p_hi) - percentile-based robust min-max into [0, 1]."""
    vals = x[mask] if mask is not None else x.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    q_lo, q_hi = np.percentile(vals, [p_lo, p_hi])
    out = (x - q_lo) / max(float(q_hi - q_lo), 1e-8)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _boxsum(x, k):
    """Unnormalized box filter (sliding-window sum), float64, reflect border."""
    return cv2.boxFilter(
        x.astype(np.float64), ddepth=-1, ksize=(k, k),
        normalize=False, borderType=cv2.BORDER_REFLECT,
    )

def camera_rays(shape_hw: tuple[int, int], intrinsic: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    return np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1).astype(np.float32)


def _angle_deg(n1, n2):
    """Angle (degrees) between two unit-normal maps; no absolute value."""
    dot = np.clip(np.sum(n1 * n2, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot)).astype(np.float32)


# ---------------------------------------------------------------------------
# n_v : local plane normal from the (sparse) VGGT depth, via windowed PCA
# ---------------------------------------------------------------------------
def _windowed_pca_normals(points, valid, k, min_pts):
    """Per-pixel smallest-eigenvector of the local anchor-point covariance.

    Returns (normals[H,W,3], ok[H,W]) where ok marks pixels whose window holds
    >= ``min_pts`` valid anchors (elsewhere the normal is unreliable).
    """
    h, w = valid.shape
    V = valid.astype(np.float64)
    cnt = _boxsum(V, k)
    cnt_safe = np.maximum(cnt, 1.0)

    P = points * V[..., None]
    S = _boxsum(P, k)                      # (H,W,3) summed coords
    mean = S / cnt_safe[..., None]

    px, py, pz = points[..., 0], points[..., 1], points[..., 2]
    sxx = _boxsum(V * px * px, k) / cnt_safe - mean[..., 0] * mean[..., 0]
    syy = _boxsum(V * py * py, k) / cnt_safe - mean[..., 1] * mean[..., 1]
    szz = _boxsum(V * pz * pz, k) / cnt_safe - mean[..., 2] * mean[..., 2]
    sxy = _boxsum(V * px * py, k) / cnt_safe - mean[..., 0] * mean[..., 1]
    sxz = _boxsum(V * px * pz, k) / cnt_safe - mean[..., 0] * mean[..., 2]
    syz = _boxsum(V * py * pz, k) / cnt_safe - mean[..., 1] * mean[..., 2]

    cov = np.empty((h, w, 3, 3), dtype=np.float64)
    cov[..., 0, 0] = sxx; cov[..., 1, 1] = syy; cov[..., 2, 2] = szz
    cov[..., 0, 1] = cov[..., 1, 0] = sxy
    cov[..., 0, 2] = cov[..., 2, 0] = sxz
    cov[..., 1, 2] = cov[..., 2, 1] = syz

    _, vecs = np.linalg.eigh(cov.reshape(-1, 3, 3))    # ascending eigenvalues
    normals = vecs[:, :, 0].reshape(h, w, 3).astype(np.float32)
    flip = normals[..., 2] < 0
    normals[flip] *= -1.0
    return normals, cnt >= min_pts


# ---------------------------------------------------------------------------
# Edge / structure evidence (all scale-irrelevant)
# ---------------------------------------------------------------------------
def _normal_edge_strength(n_a):
    """E_normal: max DA3-normal angle (deg) to the 4-neighbours."""
    deg = np.zeros(n_a.shape[:2], dtype=np.float32)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = np.roll(n_a, shift=(dy, dx), axis=(0, 1))
        deg = np.maximum(deg, _angle_deg(n_a, shifted))
    return deg


def _rgb_edge_strength(rgb):
    """E_rgb: Scharr gradient magnitude on the grayscale image."""
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _geodesic_to_anchor(anchor, barrier, lam):
    """Edge-aware geodesic distance g(p) to the nearest anchor.

    Cost per pixel is ``1 + barrier`` (barrier in [0,1] raises the cost of
    crossing strong normal/RGB edges).  Solved with a single-source Dijkstra
    over the 4-connected grid (virtual source linked to every anchor at cost
    0).  Falls back to a plain Euclidean distance transform if anything fails.
    """
    h, w = anchor.shape
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        cost = (1.0 + barrier.astype(np.float64)).reshape(-1)
        idx = np.arange(h * w).reshape(h, w)
        rows, cols, wts = [], [], []

        def add(a, b):
            a = a.reshape(-1); b = b.reshape(-1)
            wt = 0.5 * (cost[a] + cost[b])
            rows.extend((a, b)); cols.extend((b, a)); wts.extend((wt, wt))

        add(idx[:, :-1], idx[:, 1:])   # horizontal neighbours
        add(idx[:-1, :], idx[1:, :])   # vertical neighbours

        src = h * w
        a_nodes = idx[anchor].reshape(-1)
        rows.append(np.full(a_nodes.shape, src)); cols.append(a_nodes)
        wts.append(np.zeros(a_nodes.shape, dtype=np.float64))

        r = np.concatenate(rows); c = np.concatenate(cols); d = np.concatenate(wts)
        n = h * w + 1
        graph = csr_matrix((d, (r, c)), shape=(n, n))
        dist = dijkstra(graph, directed=True, indices=src)[: h * w]
        g = dist.reshape(h, w)
        g[~np.isfinite(g)] = g[np.isfinite(g)].max() if np.isfinite(g).any() else 0.0
        return g.astype(np.float32)
    except Exception:
        return ndimage.distance_transform_edt(~anchor).astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_confidence(
    depth_v: np.ndarray,
    conf_v: np.ndarray,
    depth_f: np.ndarray,
    normal_a: np.ndarray,
    normal_f: np.ndarray,
    intrinsic: np.ndarray,
    rgb: np.ndarray | None = None,
    config: ConfidenceConfig | None = None,
    n_hypotheses: int | None = None,
):
    """Compute the per-pixel prior confidence and depth search bracket.

    Returns a dict with ``conf`` in [0, 1], ``d_min`` / ``d_max`` brackets, the
    inverse-depth half-width ``rho_eff``, optional hypothesis volume
    ``hypotheses`` (when ``n_hypotheses`` given) and the intermediate factors.
    """
    cfg = config or ConfidenceConfig()
    depth_v = np.asarray(depth_v, dtype=np.float32)
    conf_v = np.asarray(conf_v, dtype=np.float32)
    depth_f = np.asarray(depth_f, dtype=np.float32)
    n_a = np.asarray(normal_a, dtype=np.float32)
    n_f = np.asarray(normal_f, dtype=np.float32)
    h, w = depth_v.shape

    # masks: M anchor, F fill (valid filled & not anchor)
    M = depth_v > 0.0
    F = (np.isfinite(depth_f) & (depth_f > 0.0)) & (~M)

    # Dense depth field for local statistics (filled depth == d_v on anchors).
    D = np.where(np.isfinite(depth_f), depth_f, 0.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Step 1 - base band c_0
    # ------------------------------------------------------------------ #
    t_v = _robust_norm(conf_v, *cfg.conf_norm_pct, mask=M)        # [0,1]
    theta_fa = _angle_deg(n_f, n_a)                              # n_f vs n_a (deg)
    t_theta = np.clip(
        (cfg.theta_max_deg - theta_fa) / (cfg.theta_max_deg - cfg.theta_min_deg),
        0.0, 1.0,
    ).astype(np.float32)

    c0 = np.zeros((h, w), dtype=np.float32)
    c0[M] = 0.5 + 0.5 * t_v[M]          # anchor band [0.5, 1]
    c0[F] = 0.5 * t_theta[F]            # fill   band [0, 0.5]

    # ------------------------------------------------------------------ #
    # Step 2 - reliability factors
    # ------------------------------------------------------------------ #
    # 2.1 f_xc : anchor normal cross-consistency (n_v from VGGT vs n_a from DA3)
    cam_pts = camera_rays((h, w), intrinsic) * depth_v[..., None]
    n_v, nv_ok = _windowed_pca_normals(cam_pts, M, cfg.pca_window, cfg.pca_min_pts)
    theta_v = _angle_deg(n_v, n_a)
    f_xc = np.exp(-((theta_v / cfg.tau_xc_deg) ** 2)).astype(np.float32)
    f_xc[~nv_ok] = 1.0                  # sampling-insufficiency protection
    f_xc[~M] = 1.0                      # only defined on anchors

    # local robust fit + MAD on the dense field -> r_loc
    d_fit = ndimage.median_filter(D, size=cfg.loc_window, mode="reflect")
    abs_dev = np.abs(D - d_fit)
    mad = ndimage.median_filter(abs_dev, size=cfg.loc_window, mode="reflect")
    r_loc = abs_dev / np.maximum(1.4826 * mad, 1e-6)

    # edge evidence B = max(N(E_normal), N(E_rgb))
    e_normal = _normal_edge_strength(n_a)
    b_normal = _robust_norm(e_normal, 50.0, 95.0)
    if rgb is not None:
        e_rgb = _rgb_edge_strength(np.asarray(rgb))
        b_rgb = _robust_norm(e_rgb, 50.0, 95.0)
        B = np.maximum(b_normal, b_rgb)
    else:
        B = b_normal

    # isolation C_iso: high-residual pixel with few high-residual neighbours
    hires = (r_loc > cfg.iso_resid_k).astype(np.float64)
    cnt_hi = _boxsum(hires, cfg.iso_window)
    thr = cfg.iso_frac * (cfg.iso_window ** 2)
    c_iso = (hires * np.clip((thr - cnt_hi) / max(thr, 1e-6), 0.0, 1.0)).astype(np.float32)

    # 3.2 f_loc gating (used in both regions)
    f_loc = np.exp(-((r_loc * (1.0 - B) * c_iso) / cfg.tau_loc) ** 2).astype(np.float32)

    # 3.3 / 3.4 fill-region factors
    g = _geodesic_to_anchor(M, B, cfg.lam)
    f_dist = (1.0 / (1.0 + g / cfg.lam)).astype(np.float32)
    e_edge = _robust_norm(e_normal, 50.0, 95.0)        # scale-irrelevant normal edge
    f_edge = np.exp(-e_edge / cfg.tau_e).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Step 3 - outlier gate O (may punch through the base band)
    # ------------------------------------------------------------------ #
    O = np.exp(-((np.maximum(r_loc - cfg.k, 0.0) * (1.0 - B)) / cfg.tau_O) ** 2).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Step 4 - fuse (region-internal geometric mean), keep [0, 1]
    # ------------------------------------------------------------------ #
    c = np.zeros((h, w), dtype=np.float32)
    c[M] = c0[M] * np.sqrt(np.clip(f_xc[M] * f_loc[M], 0.0, 1.0)) * O[M]
    c[F] = c0[F] * np.cbrt(np.clip(f_dist[F] * f_edge[F] * f_loc[F], 0.0, 1.0)) * O[F]
    conf = np.clip(c, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Step 5 - confidence -> inverse-depth search bracket
    # ------------------------------------------------------------------ #
    rho_base = cfg.rho_min + (cfg.rho_max - cfg.rho_min) * (1.0 - conf) ** cfg.gamma

    theta_rad = np.radians(theta_fa)
    sigma = np.zeros((h, w), dtype=np.float32)
    # anchor metric floor (no d_a): VGGT-conf term and normal-inconsistency term
    sig_anchor = np.maximum(
        depth_v * (1.0 - t_v) * cfg.sigma_conf_scale,
        depth_v * (1.0 - f_xc) * cfg.kappa,
    )
    # fill metric floor: path length x normal-angle error x depth
    sig_fill = (g / cfg.lam) * np.sin(theta_rad) * depth_f
    sigma[M] = sig_anchor[M]
    sigma[F] = sig_fill[F]

    df_safe = np.where(depth_f > 0.0, depth_f, np.where(depth_v > 0.0, depth_v, 1.0)).astype(np.float32)
    rho_floor = cfg.m * sigma / df_safe
    rho_eff = np.clip(np.maximum(rho_base, rho_floor), cfg.rho_min, cfg.rho_eff_cap).astype(np.float32)

    d_min = (df_safe / (1.0 + rho_eff)).astype(np.float32)
    d_max = (df_safe / (1.0 - rho_eff)).astype(np.float32)

    out = {
        "d_min": d_min,
        "d_max": d_max,
        "rho_eff": rho_eff,
        "mask_anchor": M,
        "mask_fill": F,
        "factors": {
            "c0": c0, "t_v": t_v, "t_theta": t_theta,
            "f_xc": f_xc, "f_loc": f_loc, "f_dist": f_dist, "f_edge": f_edge,
            "O": O, "B": B, "C_iso": c_iso, "r_loc": r_loc.astype(np.float32),
            "g": g, "n_v": n_v, "nv_ok": nv_ok,
        },
    }

    # ------------------------------------------------------------------ #
    # Step 5.3 - optional inverse-depth-uniform hypothesis volume
    # ------------------------------------------------------------------ #
    if n_hypotheses is not None and n_hypotheses > 1:
        N = int(n_hypotheses)
        u = 1.0 / df_safe
        i = np.arange(N, dtype=np.float32).reshape(N, 1, 1)
        u_i = u[None] * (1.0 - rho_eff[None] + 2.0 * rho_eff[None] * i / (N - 1))
        out["hypotheses"] = (1.0 / u_i).astype(np.float32)   # (N, H, W)

    return conf, out
