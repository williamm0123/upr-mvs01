"""Offline VGGT + DA3 prior cache generation for DTU training."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from base.config import MVSConfig
from data.camera_utils import (
    backproject_depth_to_world_points,
    project_world_points_to_depth,
    resize_and_crop_image,
)
from data.dtu import DTUMVSDataset, _read_dtu_cam_file
from models.vggt_prior import VGGTPrior

from models.depth_fill import (
    NormalConstraintDepthFillConfig,
    PointCloudDenoiseConfig,
    denoise_pointcloud_points,
    fill_vggt_depth_by_da3_normals,
    generate_da3_depth_maps,
    load_da3_model,
)


@dataclass(frozen=True)
class PriorGroup:
    scan: str
    light_idx: int
    ref_view: int
    view_ids: tuple[int, ...]
    ndepths: int


@dataclass
class LoadedView:
    view_id: int
    image: np.ndarray
    K: np.ndarray
    E: np.ndarray
    depth_min: float
    depth_interval: float


def expected_prior_path(
    prior_root: str | Path,
    scan: str,
    light_idx: int,
    ref_view: int,
    view_id: int,
) -> Path:
    rect = f"rect_{view_id + 1:03d}_{light_idx}_r5000"
    return Path(prior_root) / f"{scan}_light{light_idx}_ref{ref_view:03d}" / "npy" / f"{rect}_normal_constraint_filled_depth.npy"


def _prior_candidates(
    prior_root: str | Path,
    scan: str,
    light_idx: int,
    ref_view: int,
    view_id: int,
) -> list[Path]:
    root = Path(prior_root)
    rect = f"rect_{view_id + 1:03d}_{light_idx}_r5000"
    scene_dirs = [
        root / f"{scan}_light{light_idx}_ref{ref_view:03d}",
        root / f"{scan}_ref{ref_view:03d}_light{light_idx}",
        root / scan,
        root,
    ]
    exact_names = [
        f"{rect}_da3_loggrad_filled_depth.npy",
        f"{rect}_da3_local_affine_filled_depth.npy",
        f"{rect}_normal_constraint_filled_depth.npy",
        f"{rect}_filled_depth.npy",
        f"{rect}_depth.npy",
        f"{rect}.npy",
        f"{rect}.npz",
        f"{rect}.pfm",
    ]
    out: list[Path] = []
    for scene_dir in scene_dirs:
        for subdir in (scene_dir / "npy", scene_dir / "depths", scene_dir):
            for name in exact_names:
                out.append(subdir / name)
            out.extend(sorted(subdir.glob(f"{rect}*filled_depth.npy")))
            out.extend(sorted(subdir.glob(f"{rect}*filled_depth.npz")))
            out.extend(sorted(subdir.glob(f"{rect}*filled_depth.pfm")))
    return out


def find_existing_prior_file(
    prior_root: str | Path,
    scan: str,
    light_idx: int,
    ref_view: int,
    view_id: int,
) -> Path | None:
    for path in _prior_candidates(prior_root, scan, light_idx, ref_view, view_id):
        if path.is_file():
            return path
    return None


def _log(logger: Any | None, message: str) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def _make_meta_dataset(cfg: MVSConfig, mode: str, listfile: Path) -> DTUMVSDataset:
    return DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=listfile,
        nviews=cfg.train.num_views,
        target_h=cfg.data.target_h,
        target_w=cfg.data.target_w,
        feature_strides=cfg.data.feature_strides,
        mode=mode,
        use_pair_filter=cfg.data.use_pair_filter,
        pair_min_baseline_deg=cfg.data.pair_min_baseline_deg,
        pair_max_baseline_deg=cfg.data.pair_max_baseline_deg,
        prior_root=None,
    )


def collect_prior_groups(cfg: MVSConfig) -> list[PriorGroup]:
    light_idx = int(cfg.vggt_prior.offline_generation_light_idx)
    splits: list[tuple[str, Path]] = [("train", Path(cfg.paths.train_list_file))]
    if cfg.vggt_prior.offline_generation_include_val:
        splits.append(("val", Path(cfg.paths.val_list_file)))

    groups: list[PriorGroup] = []
    seen: set[tuple[str, int, tuple[int, ...]]] = set()
    for mode, listfile in splits:
        if not listfile.is_file():
            continue
        dataset = _make_meta_dataset(cfg, mode=mode, listfile=listfile)
        for scan, _sample_light, ref_view, src_views in dataset.metas:
            view_ids = tuple([ref_view] + list(src_views[: cfg.train.num_views - 1]))
            key = (scan, ref_view, view_ids)
            if key in seen:
                continue
            seen.add(key)
            groups.append(
                PriorGroup(
                    scan=scan,
                    light_idx=light_idx,
                    ref_view=ref_view,
                    view_ids=view_ids,
                    ndepths=dataset.ndepths,
                )
            )
    return groups


def find_missing_offline_priors(cfg: MVSConfig) -> tuple[list[tuple[PriorGroup, tuple[int, ...]]], dict[str, int]]:
    prior_root = cfg.paths.offline_prior_root
    groups = collect_prior_groups(cfg)
    missing_groups: list[tuple[PriorGroup, tuple[int, ...]]] = []
    existing_files = 0
    total_files = 0
    for group in groups:
        missing: list[int] = []
        for view_id in group.view_ids:
            total_files += 1
            if find_existing_prior_file(prior_root, group.scan, group.light_idx, group.ref_view, view_id) is None:
                missing.append(view_id)
            else:
                existing_files += 1
        if missing:
            missing_groups.append((group, tuple(missing)))

    max_groups = int(cfg.vggt_prior.offline_generation_max_groups)
    if max_groups > 0:
        missing_groups = missing_groups[:max_groups]

    stats = {
        "groups": len(groups),
        "total_files": total_files,
        "existing_files": existing_files,
        "missing_files": total_files - existing_files,
        "missing_groups": len(missing_groups),
    }
    return missing_groups, stats


def _load_group_views(cfg: MVSConfig, group: PriorGroup) -> list[LoadedView]:
    views: list[LoadedView] = []
    for view_id in group.view_ids:
        img_path = (
            Path(cfg.paths.dtu_train_root)
            / "Rectified_raw"
            / group.scan
            / f"rect_{view_id + 1:03d}_{group.light_idx}_r5000.png"
        )
        cam_path = Path(cfg.paths.dtu_train_root) / "Cameras" / f"{view_id:08d}_cam.txt"
        img = np.asarray(Image.open(img_path).convert("RGB"))
        K, E, depth_min, depth_interval = _read_dtu_cam_file(str(cam_path))
        image_resized, K_resized, _ = resize_and_crop_image(
            img,
            K,
            cfg.data.target_h,
            cfg.data.target_w,
            interp=cv2.INTER_AREA,
        )
        views.append(
            LoadedView(
                view_id=view_id,
                image=image_resized,
                K=K_resized.astype(np.float32),
                E=E.astype(np.float32),
                depth_min=float(depth_min),
                depth_interval=float(depth_interval),
            )
        )
    return views


def _select_confident_mask(depth: np.ndarray, confidence: np.ndarray, cfg: MVSConfig) -> np.ndarray:
    finite = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if not finite.any():
        return np.zeros_like(depth, dtype=bool)

    keep_ratio = float(cfg.vggt_prior.offline_sparse_keep_ratio)
    min_conf = float(cfg.vggt_prior.offline_sparse_min_confidence)
    conf_valid = confidence[finite]
    if 0.0 < keep_ratio < 1.0 and conf_valid.size > 0:
        ratio_threshold = float(np.quantile(conf_valid, 1.0 - keep_ratio))
        min_conf = max(min_conf, ratio_threshold)
    mask = finite & (confidence >= min_conf)
    if int(mask.sum()) < NormalConstraintDepthFillConfig().align_min_points:
        mask = finite & (confidence >= min(0.2, float(np.median(conf_valid))))
    if int(mask.sum()) < NormalConstraintDepthFillConfig().align_min_points:
        mask = finite
    return mask


def _limit_points(points: np.ndarray, confidences: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, confidences
    keep_idx = np.argpartition(confidences, -max_points)[-max_points:]
    return points[keep_idx], confidences[keep_idx]


def _build_denoised_world_points(
    depths: np.ndarray,
    confidences: np.ndarray,
    views: list[LoadedView],
    cfg: MVSConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    points_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []
    selected_pixels = 0
    for i, view in enumerate(views):
        valid = _select_confident_mask(depths[i], confidences[i], cfg)
        selected_pixels += int(valid.sum())
        sparse_depth = np.where(valid, depths[i], 0.0).astype(np.float32)
        points = backproject_depth_to_world_points(sparse_depth, view.K, view.E)
        point_conf = confidences[i][valid].astype(np.float32)
        if len(points):
            points_all.append(points)
            conf_all.append(point_conf)

    if not points_all:
        return np.empty((0, 3), dtype=np.float32), {
            "selected_pixels": selected_pixels,
            "points_before_limit": 0,
            "points_after_limit": 0,
            "points_after_denoise": 0,
        }

    points_world = np.concatenate(points_all, axis=0).astype(np.float32)
    point_conf = np.concatenate(conf_all, axis=0).astype(np.float32)
    before_limit = int(len(points_world))
    points_world, point_conf = _limit_points(points_world, point_conf, int(cfg.vggt_prior.offline_denoise_max_points))

    info: dict[str, Any] = {
        "selected_pixels": selected_pixels,
        "points_before_limit": before_limit,
        "points_after_limit": int(len(points_world)),
    }
    if cfg.vggt_prior.offline_denoise_points and len(points_world) > 8:
        denoise_cfg = PointCloudDenoiseConfig(knn=min(40, max(2, len(points_world) - 1)))
        denoised, _, denoise_info = denoise_pointcloud_points(points_world, config=denoise_cfg)
        if len(denoised) >= NormalConstraintDepthFillConfig().align_min_points:
            points_world = denoised.astype(np.float32)
        info["denoise"] = denoise_info
    info["points_after_denoise"] = int(len(points_world))
    return points_world.astype(np.float32), info


def _fallback_sparse_from_vggt(
    depth: np.ndarray,
    confidence: np.ndarray,
    cfg: MVSConfig,
) -> tuple[np.ndarray, np.ndarray]:
    valid = _select_confident_mask(depth, confidence, cfg)
    sparse = np.where(valid, depth, 0.0).astype(np.float32)
    return sparse, valid


def _cached_da3_depth(
    cache: OrderedDict[tuple[str, int, int], np.ndarray],
    key: tuple[str, int, int],
    image: np.ndarray,
    da3_model: Any,
    max_items: int = 8,
) -> np.ndarray:
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    depth = generate_da3_depth_maps(image, da3_model)
    cache[key] = depth
    if len(cache) > max_items:
        cache.popitem(last=False)
    return depth


def _generate_group_priors(
    cfg: MVSConfig,
    group: PriorGroup,
    missing_view_ids: tuple[int, ...],
    vggt_prior: VGGTPrior,
    da3_model: Any,
    device: torch.device,
    da3_cache: OrderedDict[tuple[str, int, int], np.ndarray],
    logger: Any | None = None,
) -> dict[str, int]:
    views = _load_group_views(cfg, group)
    images = np.stack([(view.image.astype(np.float32) / 255.0).transpose(2, 0, 1) for view in views], axis=0)
    intrinsics = np.stack([view.K for view in views], axis=0)
    extrinsics = np.stack([view.E for view in views], axis=0)
    ref = views[0]
    depth_min = ref.depth_min
    depth_max = ref.depth_min + ref.depth_interval * group.ndepths

    with torch.inference_mode():
        prior = vggt_prior(
            torch.from_numpy(images).unsqueeze(0).to(device),
            torch.from_numpy(intrinsics).unsqueeze(0).to(device),
            torch.from_numpy(extrinsics).unsqueeze(0).to(device),
            depth_min=torch.tensor([depth_min], dtype=torch.float32, device=device),
            depth_max=torch.tensor([depth_max], dtype=torch.float32, device=device),
        )
    depths = prior["depth_sparse"][0].detach().cpu().numpy().astype(np.float32)
    confidences = prior["confidence"][0].detach().cpu().numpy().astype(np.float32)
    world_points, pc_info = _build_denoised_world_points(depths, confidences, views, cfg)

    generated = 0
    fallback = 0
    fill_cfg = NormalConstraintDepthFillConfig()
    for i, view in enumerate(views):
        if view.view_id not in missing_view_ids:
            continue
        out_path = expected_prior_path(cfg.paths.offline_prior_root, group.scan, group.light_idx, group.ref_view, view.view_id)
        if out_path.is_file():
            continue

        sparse_depth, valid = project_world_points_to_depth(world_points, view.K, view.E, depths[i].shape)
        if int(valid.sum()) < fill_cfg.align_min_points:
            sparse_depth, valid = _fallback_sparse_from_vggt(depths[i], confidences[i], cfg)
            fallback += 1

        da3_depth = _cached_da3_depth(
            da3_cache,
            (group.scan, group.light_idx, view.view_id),
            view.image,
            da3_model,
        )
        try:
            filled, _info = fill_vggt_depth_by_da3_normals(
                sparse_depth=sparse_depth,
                da3_depth=da3_depth,
                intrinsic=view.K,
                valid=valid,
                config=fill_cfg,
            )
        except Exception as exc:
            fallback += 1
            filled = np.where(np.isfinite(depths[i]) & (depths[i] > 0), depths[i], 0.0).astype(np.float32)
            _log(
                logger,
                f"[offline-prior] fill fallback {group.scan} ref={group.ref_view} view={view.view_id}: {exc}",
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, filled.astype(np.float32))
        generated += 1

    if generated:
        _log(
            logger,
            (
                f"[offline-prior] generated {generated} files for {group.scan} "
                f"ref={group.ref_view:03d} points={pc_info.get('points_after_denoise', 0)}"
            ),
        )
    return {"generated": generated, "fallback": fallback}


def ensure_offline_priors(
    cfg: MVSConfig,
    device: torch.device | str | None = None,
    logger: Any | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    if cfg.vggt_prior.prior_source not in ("offline", "auto"):
        return {"enabled": 0, "generated": 0, "missing_files": 0}
    if not cfg.vggt_prior.generate_missing_offline:
        return {"enabled": 0, "generated": 0, "missing_files": 0}

    missing_groups, stats = find_missing_offline_priors(cfg)
    _log(
        logger,
        (
            "[offline-prior] cache scan: "
            f"groups={stats['groups']} files={stats['total_files']} "
            f"existing={stats['existing_files']} missing={stats['missing_files']}"
        ),
    )
    if not missing_groups or stats["missing_files"] == 0:
        return {"enabled": 1, "generated": 0, **stats}
    if dry_run:
        return {"enabled": 1, "generated": 0, **stats}

    device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Path(cfg.paths.offline_prior_root).mkdir(parents=True, exist_ok=True)
    _log(logger, "[offline-prior] source: VGGT depth -> denoised point cloud -> DA3 normal-constraint fill")
    _log(logger, f"[offline-prior] generating missing priors under {cfg.paths.offline_prior_root}")
    vggt_prior = VGGTPrior(cfg.vggt_prior, weights_path=cfg.paths.vggt_weights_path, device=device).eval()
    da3_model, _da3_device = load_da3_model(cfg.paths.da3_weights_file, device=device)
    da3_cache: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()

    generated = 0
    fallback = 0
    for group, missing_view_ids in missing_groups:
        result = _generate_group_priors(
            cfg=cfg,
            group=group,
            missing_view_ids=missing_view_ids,
            vggt_prior=vggt_prior,
            da3_model=da3_model,
            device=device,
            da3_cache=da3_cache,
            logger=logger,
        )
        generated += int(result["generated"])
        fallback += int(result["fallback"])

    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log(logger, f"[offline-prior] done generated={generated} fallback={fallback}")
    return {"enabled": 1, "generated": generated, "fallback": fallback, **stats}
