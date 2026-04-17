"""Cost-volume construction, depth regression, and metric helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import make_projection_for_feature_grid, resize_gt_to_feature_grid
from upr_mvs.external import homo_warping, intrinsics_to_projection, sample_depth_planes, scale_intrinsics


def prepare_cost_volume_inputs(sample: dict, scale: float, num_depths: int, device: torch.device | str) -> dict:
    imgs = sample["imgs"].unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    intrinsics = sample["intrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32)
    extrinsics = sample["extrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32)
    depth_range = sample["depth_range"].view(1, 2).to(device=device, dtype=torch.float32)

    batch_size, num_views, channels, image_h, image_w = imgs.shape
    feature_h = max(1, int(round(image_h * scale)))
    feature_w = max(1, int(round(image_w * scale)))

    features = F.interpolate(
        imgs.view(batch_size * num_views, channels, image_h, image_w),
        size=(feature_h, feature_w),
        mode="bilinear",
        align_corners=False,
    ).view(batch_size, num_views, channels, feature_h, feature_w)

    scaled_intrinsics = scale_intrinsics(
        intrinsics.view(batch_size * num_views, 3, 3),
        scale_x=float(feature_w) / float(image_w),
        scale_y=float(feature_h) / float(image_h),
    ).view(batch_size, num_views, 3, 3)

    projection_matrices = intrinsics_to_projection(
        scaled_intrinsics.view(batch_size * num_views, 3, 3),
        extrinsics.view(batch_size * num_views, 4, 4),
    ).view(batch_size, num_views, 4, 4)

    depth_values = sample_depth_planes(depth_range, num_depths)
    return {
        "features": features,
        "projection_matrices": projection_matrices,
        "depth_values": depth_values,
        "image_hw": (image_h, image_w),
        "feature_hw": (feature_h, feature_w),
    }


def build_variance_cost_volume(features: torch.Tensor, projection_matrices: torch.Tensor, depth_values: torch.Tensor) -> dict:
    batch_size, num_views, channels, feature_h, feature_w = features.shape
    ref_features = features[:, 0]
    ref_projection = projection_matrices[:, 0]

    ref_volume = ref_features.unsqueeze(2).expand(-1, -1, depth_values.shape[1], -1, -1)
    volume_sum = ref_volume.clone()
    volume_sq_sum = ref_volume.square()

    valid_sum = torch.ones(
        batch_size,
        1,
        depth_values.shape[1],
        feature_h,
        feature_w,
        device=features.device,
        dtype=features.dtype,
    )
    warped_source_volumes = []

    for src_slot in range(1, num_views):
        warped = homo_warping(features[:, src_slot], projection_matrices[:, src_slot], ref_projection, depth_values)
        warped_valid = homo_warping(
            torch.ones_like(features[:, src_slot, :1]),
            projection_matrices[:, src_slot],
            ref_projection,
            depth_values,
        ).clamp(0.0, 1.0)

        volume_sum = volume_sum + warped
        volume_sq_sum = volume_sq_sum + warped.square()
        valid_sum = valid_sum + (warped_valid > 0.5).to(features.dtype)
        warped_source_volumes.append(warped)

    mean_volume = volume_sum / float(num_views)
    variance_volume = volume_sq_sum / float(num_views) - mean_volume.square()
    cost_volume = variance_volume.mean(dim=1)
    all_views_valid = valid_sum.eq(float(num_views)).squeeze(1)
    return {
        "cost_volume": cost_volume,
        "all_views_valid": all_views_valid,
        "warped_source_volumes": warped_source_volumes,
    }


def compute_all_views_valid_for_grid(
    projection_matrices: torch.Tensor,
    depth_values: torch.Tensor,
    feature_hw: tuple[int, int],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    batch_size, num_views = projection_matrices.shape[:2]
    feature_h, feature_w = feature_hw
    valid_sum = torch.ones(batch_size, 1, depth_values.shape[1], feature_h, feature_w, device=device, dtype=dtype)
    ones = torch.ones(batch_size, 1, feature_h, feature_w, device=device, dtype=dtype)
    ref_projection = projection_matrices[:, 0]

    for src_slot in range(1, num_views):
        warped_valid = homo_warping(
            ones,
            projection_matrices[:, src_slot],
            ref_projection,
            depth_values,
        ).clamp(0.0, 1.0)
        valid_sum = valid_sum + (warped_valid > 0.5).to(dtype)

    return valid_sum.eq(float(num_views)).squeeze(1)


def build_variance_cost_volume_chunked(
    features: torch.Tensor,
    projection_matrices: torch.Tensor,
    depth_values: torch.Tensor,
    channel_chunk: int = 4,
    all_views_valid: torch.Tensor | None = None,
) -> dict:
    batch_size, num_views, channels, feature_h, feature_w = features.shape
    ref_projection = projection_matrices[:, 0]
    if all_views_valid is None:
        all_views_valid = compute_all_views_valid_for_grid(
            projection_matrices=projection_matrices,
            depth_values=depth_values,
            feature_hw=(feature_h, feature_w),
            device=features.device,
            dtype=features.dtype,
        )
    cost_sum = torch.zeros(batch_size, depth_values.shape[1], feature_h, feature_w, device=features.device, dtype=features.dtype)

    for start in range(0, channels, channel_chunk):
        end = min(start + channel_chunk, channels)
        ref_chunk = features[:, 0, start:end]
        ref_volume = ref_chunk.unsqueeze(2).expand(-1, -1, depth_values.shape[1], -1, -1)
        volume_sum = ref_volume.clone()
        volume_sq_sum = ref_volume.square()

        for src_slot in range(1, num_views):
            warped = homo_warping(
                features[:, src_slot, start:end],
                projection_matrices[:, src_slot],
                ref_projection,
                depth_values,
            )
            volume_sum = volume_sum + warped
            volume_sq_sum = volume_sq_sum + warped.square()
            del warped

        mean_volume = volume_sum / float(num_views)
        variance_volume = volume_sq_sum / float(num_views) - mean_volume.square()
        cost_sum = cost_sum + variance_volume.sum(dim=1)

        del ref_chunk, ref_volume, volume_sum, volume_sq_sum, mean_volume, variance_volume
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"cost_volume": cost_sum / float(channels), "all_views_valid": all_views_valid}


def depth_regression_from_cost(
    cost_volume: torch.Tensor,
    depth_values: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    temperature: float = 0.02,
) -> dict:
    if candidate_mask is None:
        candidate_mask = torch.ones_like(cost_volume, dtype=torch.bool)
    else:
        candidate_mask = candidate_mask.bool()

    finite_cost = torch.isfinite(cost_volume)
    candidate_mask = candidate_mask & finite_cost
    has_candidate = candidate_mask.any(dim=1)

    masked_cost = cost_volume.masked_fill(~candidate_mask, float("inf"))
    min_cost = masked_cost.amin(dim=1, keepdim=True)
    stable_cost = masked_cost - min_cost
    stable_cost = torch.where(torch.isfinite(stable_cost), stable_cost, torch.zeros_like(stable_cost))

    logits = -stable_cost / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~candidate_mask, -1.0e4)

    prob_volume = torch.softmax(logits, dim=1)
    prob_volume = torch.where(has_candidate.unsqueeze(1), prob_volume, torch.zeros_like(prob_volume))

    depth_volume = depth_values.view(depth_values.shape[0], depth_values.shape[1], 1, 1)
    depth = (prob_volume * depth_volume).sum(dim=1)
    confidence = prob_volume.max(dim=1).values
    entropy = -(prob_volume * torch.log(prob_volume.clamp_min(1e-8))).sum(dim=1)

    depth = torch.where(has_candidate, depth, torch.full_like(depth, float("nan")))
    confidence = torch.where(has_candidate, confidence, torch.zeros_like(confidence))
    entropy = torch.where(has_candidate, entropy, torch.full_like(entropy, float("nan")))
    return {
        "depth": depth,
        "prob_volume": prob_volume,
        "confidence": confidence,
        "entropy": entropy,
        "has_candidate": has_candidate,
    }


def compute_depth_error_summary(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor) -> dict:
    valid = valid_mask & torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
    errors = torch.abs(pred_depth - gt_depth)[valid]
    if errors.numel() == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "within_10mm": float("nan"),
            "within_25mm": float("nan"),
            "within_50mm": float("nan"),
            "num_valid": 0,
        }
    return {
        "mean": float(errors.mean()),
        "median": float(errors.median()),
        "p90": float(torch.quantile(errors, 0.90)),
        "within_10mm": float((errors < 10.0).float().mean()),
        "within_25mm": float((errors < 25.0).float().mean()),
        "within_50mm": float((errors < 50.0).float().mean()),
        "num_valid": int(errors.numel()),
    }


def evaluate_feature_cost_volume_highres(
    feature_name: str,
    features_for_cost: torch.Tensor,
    projection_matrices: torch.Tensor,
    depth_values: torch.Tensor,
    depth_gt_target: torch.Tensor,
    mask_target: torch.Tensor,
    temperature: float,
    channel_chunk: int = 4,
    all_views_valid: torch.Tensor | None = None,
) -> tuple[dict, dict]:
    with torch.no_grad():
        cost_outputs = build_variance_cost_volume_chunked(
            features=features_for_cost,
            projection_matrices=projection_matrices,
            depth_values=depth_values,
            channel_chunk=channel_chunk,
            all_views_valid=all_views_valid,
        )
        layer_cost_volume = cost_outputs["cost_volume"]
        layer_all_views_valid = cost_outputs["all_views_valid"]

        masked_cost = layer_cost_volume.masked_fill(~layer_all_views_valid, float("inf"))
        has_valid_depth = torch.isfinite(masked_cost).any(dim=1)
        best_depth_idx = masked_cost.argmin(dim=1)
        depth_values_volume = depth_values.view(depth_values.shape[0], depth_values.shape[1], 1, 1).expand_as(layer_cost_volume)
        argmin_depth = torch.gather(depth_values_volume, dim=1, index=best_depth_idx.unsqueeze(1)).squeeze(1)
        argmin_depth = torch.where(has_valid_depth, argmin_depth, torch.full_like(argmin_depth, float("nan")))

        soft_outputs = depth_regression_from_cost(
            cost_volume=layer_cost_volume,
            depth_values=depth_values,
            candidate_mask=layer_all_views_valid,
            temperature=temperature,
        )
        soft_depth = soft_outputs["depth"]
        confidence = soft_outputs["confidence"]
        valid_soft = mask_target & soft_outputs["has_candidate"] & torch.isfinite(soft_depth)
        valid_argmin = mask_target & has_valid_depth & torch.isfinite(argmin_depth)

        argmin_summary = compute_depth_error_summary(argmin_depth, depth_gt_target, valid_argmin)
        soft_summary = compute_depth_error_summary(soft_depth, depth_gt_target, valid_soft)

        valid_confidence = confidence[soft_outputs["has_candidate"]]
        confidence_mean = float(valid_confidence.mean()) if valid_confidence.numel() > 0 else float("nan")
        confidence_median = float(valid_confidence.median()) if valid_confidence.numel() > 0 else float("nan")

    row = {
        "feature": feature_name,
        "channels": int(features_for_cost.shape[2]),
        "height": int(features_for_cost.shape[-2]),
        "width": int(features_for_cost.shape[-1]),
        "valid_ratio": float(has_valid_depth.float().mean()),
        "argmin_mean": argmin_summary["mean"],
        "argmin_median": argmin_summary["median"],
        "argmin_p90": argmin_summary["p90"],
        "soft_mean": soft_summary["mean"],
        "soft_median": soft_summary["median"],
        "soft_p90": soft_summary["p90"],
        "soft_<10mm": soft_summary["within_10mm"],
        "soft_<25mm": soft_summary["within_25mm"],
        "soft_<50mm": soft_summary["within_50mm"],
        "confidence_mean": confidence_mean,
        "confidence_median": confidence_median,
        "num_eval_pixels": soft_summary["num_valid"],
    }
    maps_for_save = {
        "argmin_depth": argmin_depth.detach().cpu(),
        "soft_depth": soft_depth.detach().cpu(),
        "confidence": confidence.detach().cpu(),
        "abs_error": torch.abs(soft_depth - depth_gt_target).detach().cpu(),
        "valid_mask": valid_soft.detach().cpu(),
    }

    del layer_cost_volume, layer_all_views_valid, masked_cost, soft_outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, maps_for_save


def prepare_rgb_feature_grid_baseline(
    sample: dict,
    feature_hw: tuple[int, int],
    num_depths: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs = sample["imgs"].unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    batch_size, num_views, channels, image_h, image_w = imgs.shape
    feature_h, feature_w = feature_hw
    rgb_features = F.interpolate(
        imgs.view(batch_size * num_views, channels, image_h, image_w),
        size=feature_hw,
        mode="bilinear",
        align_corners=False,
    ).view(batch_size, num_views, channels, feature_h, feature_w)
    projection_matrices = make_projection_for_feature_grid(sample, feature_hw, device)
    depth_range = sample["depth_range"].view(1, 2).to(device=device, dtype=torch.float32)
    depth_values = sample_depth_planes(depth_range, num_depths)
    depth_gt_target, mask_target = resize_gt_to_feature_grid(sample, feature_hw, device)
    return rgb_features, projection_matrices, depth_values, depth_gt_target, mask_target
