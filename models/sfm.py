"""Lightweight two-view SfM for DTU samples with known camera poses.

DTU provides metric (millimetre) intrinsics/extrinsics, so we do not need to
solve for camera poses. We only need correspondences: detect + match features
between the reference view and each source view, reject outliers with RANSAC,
then triangulate the surviving matches with the *known* projection matrices.
The triangulated world points are projected into the reference camera to form a
sparse, metric-scale depth map that can later anchor the (scale-free) VGGT
prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import data.camera_utils as C


@dataclass
class SfMConfig:
    max_features: int = 8000
    ratio_test: float = 0.75
    max_reproj_error: float = 2.0
    min_depth: float = 1e-3
    max_depth: float = 2000.0
    # per-point confidence (conf = f_reproj * f_angle * w_pair, all in [0,1])
    conf_tau_e: float = 1.0          # reprojection-error decay scale (px)
    conf_theta_sat_deg: float = 10.0 # triangulation-angle saturation (deg)
    conf_pair_n0: float = 100.0      # pair-inlier soft-saturation constant


def _to_uint8_rgb(image) -> np.ndarray:
    """sample["images"][i] is a [C, H, W] float tensor in 0-255 RGB."""
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _build_detector(max_features: int):
    """Prefer SIFT (metric-friendly float descriptors); fall back to ORB."""
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=int(max_features)), "SIFT", cv2.NORM_L2
    return cv2.ORB_create(nfeatures=int(max_features)), "ORB", cv2.NORM_HAMMING


def _projection_matrix(K: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    return (np.asarray(K, np.float64) @ np.asarray(extrinsic, np.float64)[:3, :4])


def _camera_depth(extrinsic: np.ndarray, points_h: np.ndarray) -> np.ndarray:
    """Z of homogeneous world points in the given camera frame."""
    return (points_h @ np.asarray(extrinsic, np.float64)[:3, :4].T)[:, 2]


def _reproj_error(P: np.ndarray, points_h: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    proj = points_h @ P.T
    uv = proj[:, :2] / np.clip(proj[:, 2:3], 1e-12, None)
    return np.linalg.norm(uv - pixels, axis=1)


def _camera_center(extrinsic: np.ndarray) -> np.ndarray:
    """World-frame camera centre from a world->camera extrinsic: C = -R^T t."""
    R = np.asarray(extrinsic, np.float64)[:3, :3]
    t = np.asarray(extrinsic, np.float64)[:3, 3]
    return -R.T @ t


def _point_confidence(points, err_ref, err_src, E_ref, E_src, n_pair, cfg) -> np.ndarray:
    """Per-point triangulation confidence in [0, 1].

        conf = f_reproj * f_angle * w_pair
          f_reproj : reprojection consistency, taken over the *worse* of the two
                     views  -> exp(-(max(err_ref, err_src) / tau_e)^2)
          f_angle  : parallax angle at the 3D point between the rays to each
                     camera; small angle -> uncertain depth. conf ~ sin(theta),
                     saturating at theta_sat.
          w_pair   : pair-level reliability from the RANSAC inlier count
                     (soft-saturating: N / (N + N0)).
    """
    if len(points) == 0:
        return np.empty((0,), np.float32)

    e = np.maximum(err_ref, err_src)
    f_reproj = np.exp(-((e / max(cfg.conf_tau_e, 1e-6)) ** 2))

    C_ref = _camera_center(E_ref)
    C_src = _camera_center(E_src)
    v_ref = C_ref[None, :] - points
    v_src = C_src[None, :] - points
    denom = np.clip(np.linalg.norm(v_ref, axis=1) * np.linalg.norm(v_src, axis=1), 1e-12, None)
    cos_theta = np.clip(np.sum(v_ref * v_src, axis=1) / denom, -1.0, 1.0)
    sin_theta = np.sin(np.arccos(cos_theta))
    sin_sat = max(np.sin(np.radians(cfg.conf_theta_sat_deg)), 1e-6)
    f_angle = np.clip(sin_theta / sin_sat, 0.0, 1.0)

    w_pair = float(n_pair) / (float(n_pair) + max(cfg.conf_pair_n0, 1e-6))

    return (f_reproj * f_angle * w_pair).astype(np.float32)


def _triangulate_pair(gray_ref, gray_src, K_ref, E_ref, K_src, E_src, detector, norm_type, cfg):
    """Return (world_points [M,3], ref_pixels [M,2], stats dict) for one pair."""
    stats = {"matches": 0, "ransac_matches": 0, "triangulated_points": 0}

    kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
    kp_src, des_src = detector.detectAndCompute(gray_src, None)
    if des_ref is None or des_src is None or len(kp_ref) < 2 or len(kp_src) < 2:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32), stats

    matcher = cv2.BFMatcher(norm_type)
    knn = matcher.knnMatch(des_ref, des_src, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < cfg.ratio_test * n.distance]
    stats["matches"] = len(good)
    if len(good) < 8:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32), stats

    pts_ref = np.float64([kp_ref[m.queryIdx].pt for m in good])
    pts_src = np.float64([kp_src[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(
        pts_ref, pts_src, cv2.FM_RANSAC, cfg.max_reproj_error, 0.99
    )
    if F is None or mask is None:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32), stats
    inliers = mask.ravel().astype(bool)
    pts_ref, pts_src = pts_ref[inliers], pts_src[inliers]
    stats["ransac_matches"] = int(inliers.sum())
    if len(pts_ref) < 1:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32), stats

    P_ref = _projection_matrix(K_ref, E_ref)
    P_src = _projection_matrix(K_src, E_src)
    pts4d = cv2.triangulatePoints(P_ref, P_src, pts_ref.T, pts_src.T)
    w = pts4d[3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)  # guard zero only; keep sign of w
    points = (pts4d[:3] / w).T  # [M, 3] world
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)

    z_ref = _camera_depth(E_ref, points_h)
    z_src = _camera_depth(E_src, points_h)
    err_ref = _reproj_error(P_ref, points_h, pts_ref)
    err_src = _reproj_error(P_src, points_h, pts_src)
    keep = (
        (z_ref > cfg.min_depth) & (z_ref < cfg.max_depth)
        & (z_src > cfg.min_depth)
        & (err_ref <= cfg.max_reproj_error) & (err_src <= cfg.max_reproj_error)
    )
    stats["triangulated_points"] = int(keep.sum())
    conf = _point_confidence(
        points[keep], err_ref[keep], err_src[keep], E_ref, E_src, stats["ransac_matches"], cfg
    )
    return points[keep].astype(np.float32), pts_ref[keep].astype(np.float32), conf, stats


def generate_sparse_depth_from_sample(sample, ref_idx: int = 0, config: SfMConfig | None = None):
    """Build a metric sparse depth map for the reference view via two-view SfM.

    Returns a dict with ``sparse_depth`` [H, W], ``sparse_conf`` [H, W] in [0,1]
    (per-pixel confidence of the point that won each pixel), ``valid_mask``
    [H, W] bool, ``points_world`` [N, 3], ``points_conf`` [N] in [0,1],
    ``source_weights`` [num_views-1] in [0,1] (one per-view cost-volume weight,
    aligned to the non-ref source order), ``points_color`` [N, 3] uint8 and an
    ``info`` dict. ``sparse_depth`` is identical to before (nearest-point
    z-buffer); confidence is additive only.
    """
    cfg = config or SfMConfig()
    images = sample["images"]
    intrinsics = np.asarray(sample["intrinsics"], np.float64)
    extrinsics = np.asarray(sample["extrinsics"], np.float64)
    num_views = len(images)

    rgb = [_to_uint8_rgb(images[i]) for i in range(num_views)]
    gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in rgb]
    H, W = gray[ref_idx].shape[:2]

    detector, feature_type, norm_type = _build_detector(cfg.max_features)
    K_ref, E_ref = intrinsics[ref_idx], extrinsics[ref_idx]

    all_points, all_pixels, all_conf, source_weights, pairs = [], [], [], [], []
    for src_idx in range(num_views):
        if src_idx == ref_idx:
            continue
        points, pixels, conf, stats = _triangulate_pair(
            gray[ref_idx], gray[src_idx], K_ref, E_ref,
            intrinsics[src_idx], extrinsics[src_idx], detector, norm_type, cfg,
        )
        all_points.append(points)
        all_pixels.append(pixels)
        all_conf.append(conf)
        # per-view weight for cost-volume fusion: median of this source's point
        # confidences (0.0 if the pair produced no triangulated points).
        w = float(np.median(conf)) if len(conf) else 0.0
        source_weights.append(w)
        pairs.append({"src_idx": src_idx, "weight": w, **stats})

    points_world = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), np.float32)
    pixels = np.concatenate(all_pixels, axis=0) if all_pixels else np.empty((0, 2), np.float32)
    points_conf = np.concatenate(all_conf, axis=0) if all_conf else np.empty((0,), np.float32)
    # aligned to the non-ref source order (src_idx ascending, ref skipped)
    source_weights = np.asarray(source_weights, np.float32)

    # Sample reference-image colours at the matched keypoints for the PLY.
    if len(pixels):
        u = np.clip(np.rint(pixels[:, 0]).astype(np.int32), 0, W - 1)
        v = np.clip(np.rint(pixels[:, 1]).astype(np.int32), 0, H - 1)
        points_color = rgb[ref_idx][v, u]
    else:
        points_color = np.empty((0, 3), np.uint8)

    # conf is per-point; the z-buffer inside keeps nearest-point-wins, so
    # sparse_depth is unchanged vs passing None -- we only additionally get the
    # per-pixel confidence of whichever point won each pixel.
    sparse_depth, sparse_conf = C.project_world_points_to_depth(
        points_world, points_conf, K_ref, E_ref, (W, H)
    )
    valid_mask = sparse_depth > 0

    info = {
        "feature_type": feature_type,
        "num_points_world": int(len(points_world)),
        "pairs": pairs,
    }
    return {
        "sparse_depth": sparse_depth,
        "sparse_conf": sparse_conf,
        "valid_mask": valid_mask,
        "points_world": points_world,
        "points_conf": points_conf,
        "source_weights": source_weights,
        "points_color": points_color,
        "info": info,
    }


def load_or_compute_sparse_depth(
    images,
    intrinsics,
    extrinsics,
    cache_path,
    ref_idx: int = 0,
    config: SfMConfig | None = None,
    save_vis: bool = True,
):
    """Return the ref-view SfM sparse depth at the input image resolution.

    Loads ``cache_path`` (an ``.npy``) if it exists, otherwise runs two-view SfM
    on the given (multi-view) arrays, caches the result and an optional ``.png``
    visualisation next to it. ``images`` may be ``[V, H, W, 3]`` uint8 or
    ``[V, C, H, W]``; ``intrinsics`` ``[V, 3, 3]``; ``extrinsics`` ``[V, 4, 4]``.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    sfm_sample = {"images": images, "intrinsics": intrinsics, "extrinsics": extrinsics}
    out = generate_sparse_depth_from_sample(sfm_sample, ref_idx=ref_idx, config=config)
    sparse_depth = out["sparse_depth"].astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, sparse_depth)
    if save_vis:
        C.save_depth_png(sparse_depth, cache_path.with_suffix(".png"), valid=out["valid_mask"])
    return sparse_depth


def metric_scale_from_sparse(depth, sparse_depth, sparse_valid=None, min_pairs: int = 20):
    """Global VGGT->metric scale s.t. ``depth * scale`` matches ``sparse_depth``.

    VGGT depth is metric-consistent up to a single global scale, so the scale is
    the median of the per-pixel ratio ``sparse_depth / depth`` over the pixels
    where both are valid. Using the (零稀释的) mean of ``sparse_depth`` is wrong.
    Returns ``(scale, info)``; ``scale`` falls back to 1.0 when too few overlaps.
    """
    depth = np.asarray(depth, np.float32)
    sparse_depth = np.asarray(sparse_depth, np.float32)
    if sparse_valid is None:
        sparse_valid = sparse_depth > 0
    mask = (
        np.asarray(sparse_valid, bool)
        & np.isfinite(depth) & (depth > 0)
        & np.isfinite(sparse_depth) & (sparse_depth > 0)
    )
    num_pairs = int(mask.sum())
    if num_pairs < min_pairs:
        return 1.0, {"num_pairs": num_pairs, "valid": False, "scale": 1.0}
    ratio = sparse_depth[mask] / depth[mask]
    scale = float(np.median(ratio))
    return scale, {"num_pairs": num_pairs, "valid": True, "scale": scale}


def calibrate_depth_to_metric(sample, depth, ref_idx: int = 0, config: SfMConfig | None = None):
    """Rescale ``depth`` to metric using the sample's SfM sparse depth.

    Prefers ``sample["sfm_depth"]`` (precomputed/cropped by the dataset) and only
    falls back to running SfM when it is absent. ``depth`` must already be at the
    sample's reference-view resolution. Returns ``(depth_metric, scale, sfm_out)``.
    """
    cached = sample.get("sfm_depth") if hasattr(sample, "get") else None
    if cached is not None:
        sparse_depth = np.asarray(cached, np.float32)
        valid_mask = sparse_depth > 0
        sfm_out = {"sparse_depth": sparse_depth, "valid_mask": valid_mask, "info": {"source": "sample"}}
    else:
        sfm_out = generate_sparse_depth_from_sample(sample, ref_idx=ref_idx, config=config)
        sfm_out["info"]["source"] = "computed"

    scale, scale_info = metric_scale_from_sparse(depth, sfm_out["sparse_depth"], sfm_out["valid_mask"])
    sfm_out["info"]["scale"] = scale_info
    return (np.asarray(depth, np.float32) * scale).astype(np.float32), scale, sfm_out
