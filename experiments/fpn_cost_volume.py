"""Build a cost volume from FPN-only matching features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dtu import build_dtu_dataset
from experiments.cost_volume import (
    build_variance_cost_volume_chunked,
    compute_depth_error_summary,
    depth_regression_from_cost,
)
from experiments.fpn import ConvFPNVisualizationNet
from experiments.geometry import make_projection_for_feature_grid, resize_gt_to_feature_grid
from experiments.visualization import save_depth_result_image
from upr_mvs.config import DTUConfig, ProjectPaths
from upr_mvs.external import sample_depth_planes


FPN_LEVELS = ("P2", "P3", "P4", "P5")


@dataclass(frozen=True)
class FPNCostVolumeConfig:
    """Configuration for the FPN-only cost-volume test."""

    pyramid_level: int = 2
    max_side: int = 768
    fpn_channels: int = 16
    matching_channels: int = 16
    num_depths: int = 64
    temperature: float = 0.02
    channel_chunk: int = 4
    regularization: str = "avg3d"
    regularization_blend: float = 0.5
    regularization_kernel: tuple[int, int, int] = (3, 3, 3)


def maybe_resize_images(images: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return images
    height, width = images.shape[-2:]
    scale = float(max_side) / float(max(height, width))
    if scale >= 1.0:
        return images
    target_hw = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    return F.interpolate(images, size=target_hw, mode="bilinear", align_corners=False)


def pyramid_level_name(level: int) -> str:
    level_name = f"P{level}"
    if level_name not in FPN_LEVELS:
        raise ValueError(f"Supported FPN levels are {FPN_LEVELS}, got l={level}")
    return level_name


def _init_matching_head(conv1: nn.Conv2d, conv3: nn.Conv2d) -> None:
    with torch.no_grad():
        conv1.weight.zero_()
        if conv1.bias is not None:
            conv1.bias.zero_()
        for channel in range(min(conv1.in_channels, conv1.out_channels)):
            conv1.weight[channel, channel, 0, 0] = 1.0

        blur = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=conv3.weight.dtype,
            device=conv3.weight.device,
        ) / 16.0
        conv3.weight.zero_()
        if conv3.bias is not None:
            conv3.bias.zero_()
        for channel in range(min(conv3.in_channels, conv3.out_channels)):
            conv3.weight[channel, channel] = blur


class FPNMatchingHead(nn.Module):
    """Lightweight matching head: G_l = Conv3x3(ReLU(Conv1x1(P_l)))."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        _init_matching_head(self.conv1, self.conv3)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.conv3(F.relu(self.conv1(feature), inplace=False))
        return F.normalize(feature, p=2, dim=1)


def regularize_cost_volume(
    cost_volume: torch.Tensor,
    mode: str = "avg3d",
    blend: float = 0.5,
    kernel_size: tuple[int, int, int] = (3, 3, 3),
) -> torch.Tensor:
    if mode == "none" or blend <= 0.0:
        return cost_volume
    if mode != "avg3d":
        raise ValueError(f"Unsupported cost regularization mode: {mode}")
    depth_kernel, height_kernel, width_kernel = kernel_size
    smoothed = F.avg_pool3d(
        cost_volume.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=(depth_kernel // 2, height_kernel // 2, width_kernel // 2),
    ).squeeze(1)
    blend = float(np.clip(blend, 0.0, 1.0))
    return (1.0 - blend) * cost_volume + blend * smoothed


def extract_fpn_matching_features(
    sample: dict,
    config: FPNCostVolumeConfig,
    device: torch.device,
) -> dict:
    images = sample["imgs"].to(device=device, dtype=torch.float32) / 255.0
    images = maybe_resize_images(images, config.max_side)
    level_name = pyramid_level_name(config.pyramid_level)

    fpn_model = ConvFPNVisualizationNet(
        c2_channels=config.fpn_channels,
        c3_channels=config.fpn_channels,
        c4_channels=config.fpn_channels,
        c5_channels=config.fpn_channels,
        out_channels=config.fpn_channels,
    ).to(device).eval()
    matching_head = FPNMatchingHead(
        in_channels=config.fpn_channels,
        out_channels=config.matching_channels,
    ).to(device).eval()

    with torch.inference_mode():
        fpn_features_all = fpn_model(images)
        fpn_features = {level: fpn_features_all[level] for level in FPN_LEVELS}
        matching_feature = matching_head(fpn_features[level_name])

    return {
        "images": images,
        "level": level_name,
        "fpn_features": fpn_features,
        "matching_feature": matching_feature,
    }


def _safe_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _normalize_for_display(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.nanpercentile(array[finite], [1, 99])
    if high <= low:
        high = low + 1e-6
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _feature_energy(feature: torch.Tensor) -> np.ndarray:
    return _safe_numpy(feature[0].abs().mean(dim=0))


def save_fpn_matching_overview(
    fpn_features: dict[str, torch.Tensor],
    matching_feature: torch.Tensor,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.8))
    for ax, level in zip(axes[:4], FPN_LEVELS):
        ax.imshow(_normalize_for_display(_feature_energy(fpn_features[level])), cmap="viridis")
        ax.set_title(f"FPN {level}\n{tuple(fpn_features[level].shape)}")
        ax.axis("off")
    axes[4].imshow(_normalize_for_display(_feature_energy(matching_feature)), cmap="magma")
    axes[4].set_title(f"matching G\n{tuple(matching_feature.shape)}")
    axes[4].axis("off")

    fig.suptitle("FPN-only matching features: G_l = Conv3x3(ReLU(Conv1x1(P_l)))", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_cost_volume_diagnostics(
    cost_volume: torch.Tensor,
    probability_volume: torch.Tensor,
    depth_values: torch.Tensor,
    all_views_valid: torch.Tensor,
    argmin_depth: torch.Tensor,
    soft_depth: torch.Tensor,
    confidence: torch.Tensor,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cost = cost_volume[0]
    prob = probability_volume[0]
    valid = all_views_valid[0]
    num_depths = int(cost.shape[0])
    slice_ids = [0, num_depths // 2, num_depths - 1]
    depth_list = _safe_numpy(depth_values[0])

    masked_cost = cost.masked_fill(~valid, float("nan"))
    min_cost = masked_cost.nan_to_num(float("inf")).amin(dim=0)
    valid_ratio = valid.float().mean(dim=0)
    entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=0)

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    panels = [
        (axes[0, 0], "min variance cost", _safe_numpy(min_cost), "magma"),
        (axes[0, 1], "argmin depth", _safe_numpy(argmin_depth[0]), "turbo"),
        (axes[0, 2], "soft-argmin depth", _safe_numpy(soft_depth[0]), "turbo"),
        (axes[0, 3], "confidence max prob", _safe_numpy(confidence[0]), "viridis"),
        (axes[1, 0], "valid depth candidate ratio", _safe_numpy(valid_ratio), "gray"),
        (axes[1, 1], "probability entropy", _safe_numpy(entropy), "inferno"),
    ]
    for ax, title, image, cmap in panels:
        ax.imshow(_normalize_for_display(image), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    for ax, depth_index in zip([axes[1, 2], axes[1, 3], axes[2, 0]], slice_ids):
        ax.imshow(_normalize_for_display(_safe_numpy(masked_cost[depth_index])), cmap="magma")
        ax.set_title(f"cost slice d={depth_list[depth_index]:.1f}")
        ax.axis("off")

    for ax, depth_index in zip([axes[2, 1], axes[2, 2], axes[2, 3]], slice_ids):
        ax.imshow(_normalize_for_display(_safe_numpy(prob[depth_index])), cmap="viridis")
        ax.set_title(f"prob slice d={depth_list[depth_index]:.1f}")
        ax.axis("off")

    fig.suptitle("FPN-only variance cost volume diagnostics", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def regress_depth_from_cost_stage(
    stage: str,
    cost_volume: torch.Tensor,
    depth_values: torch.Tensor,
    all_views_valid: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    feature_name: str,
    level_name: str,
    feature_shape: tuple[int, ...],
) -> tuple[dict, dict, torch.Tensor]:
    masked_cost = cost_volume.masked_fill(~all_views_valid, float("inf"))
    has_valid_depth = torch.isfinite(masked_cost).any(dim=1)
    best_depth_idx = masked_cost.argmin(dim=1)
    depth_volume = depth_values.view(depth_values.shape[0], depth_values.shape[1], 1, 1).expand_as(cost_volume)
    argmin_depth = torch.gather(depth_volume, dim=1, index=best_depth_idx.unsqueeze(1)).squeeze(1)
    argmin_depth = torch.where(has_valid_depth, argmin_depth, torch.full_like(argmin_depth, float("nan")))

    soft_outputs = depth_regression_from_cost(
        cost_volume=cost_volume,
        depth_values=depth_values,
        candidate_mask=all_views_valid,
        temperature=temperature,
    )
    soft_depth = soft_outputs["depth"]
    confidence = soft_outputs["confidence"]
    probability_volume = soft_outputs["prob_volume"]

    valid_soft = mask & soft_outputs["has_candidate"] & torch.isfinite(soft_depth)
    valid_argmin = mask & has_valid_depth & torch.isfinite(argmin_depth)
    argmin_summary = compute_depth_error_summary(argmin_depth, depth_gt, valid_argmin)
    soft_summary = compute_depth_error_summary(soft_depth, depth_gt, valid_soft)
    valid_confidence = confidence[soft_outputs["has_candidate"]]

    row = {
        "feature": feature_name,
        "stage": stage,
        "level": level_name,
        "channels": int(feature_shape[2]),
        "height": int(feature_shape[-2]),
        "width": int(feature_shape[-1]),
        "num_depths": int(depth_values.shape[1]),
        "temperature": float(temperature),
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
        "confidence_mean": float(valid_confidence.mean()) if valid_confidence.numel() else float("nan"),
        "confidence_median": float(valid_confidence.median()) if valid_confidence.numel() else float("nan"),
        "num_eval_pixels": soft_summary["num_valid"],
    }
    maps = {
        "argmin_depth": argmin_depth.detach().cpu(),
        "soft_depth": soft_depth.detach().cpu(),
        "confidence": confidence.detach().cpu(),
        "abs_error": torch.abs(soft_depth - depth_gt).detach().cpu(),
        "valid_mask": valid_soft.detach().cpu(),
    }
    return row, maps, probability_volume


def build_depth_from_fpn_cost_volume(
    sample: dict,
    features_for_cost: torch.Tensor,
    config: FPNCostVolumeConfig,
    device: torch.device,
) -> tuple[dict[str, dict], dict[str, dict], dict, torch.Tensor, torch.Tensor]:
    feature_hw = tuple(features_for_cost.shape[-2:])
    projection_matrices = make_projection_for_feature_grid(sample, feature_hw, device)
    depth_range = sample["depth_range"].view(1, 2).to(device=device, dtype=torch.float32)
    depth_values = sample_depth_planes(depth_range, config.num_depths)
    depth_gt, mask = resize_gt_to_feature_grid(sample, feature_hw, device)
    level_name = f"P{config.pyramid_level}"

    with torch.no_grad():
        cost_outputs = build_variance_cost_volume_chunked(
            features=features_for_cost,
            projection_matrices=projection_matrices,
            depth_values=depth_values,
            channel_chunk=config.channel_chunk,
        )
        raw_cost_volume = cost_outputs["cost_volume"]
        all_views_valid = cost_outputs["all_views_valid"]
        regularized_cost_volume = regularize_cost_volume(
            raw_cost_volume,
            mode=config.regularization,
            blend=config.regularization_blend,
            kernel_size=config.regularization_kernel,
        )

        raw_row, raw_maps, raw_probability_volume = regress_depth_from_cost_stage(
            stage="raw_cost",
            cost_volume=raw_cost_volume,
            depth_values=depth_values,
            all_views_valid=all_views_valid,
            depth_gt=depth_gt,
            mask=mask,
            temperature=config.temperature,
            feature_name="FPN matching feature",
            level_name=level_name,
            feature_shape=tuple(features_for_cost.shape),
        )
        regularized_row, regularized_maps, regularized_probability_volume = regress_depth_from_cost_stage(
            stage="regularized_cost",
            cost_volume=regularized_cost_volume,
            depth_values=depth_values,
            all_views_valid=all_views_valid,
            depth_gt=depth_gt,
            mask=mask,
            temperature=config.temperature,
            feature_name="FPN matching feature",
            level_name=level_name,
            feature_shape=tuple(features_for_cost.shape),
        )

    for row in [raw_row, regularized_row]:
        row["regularization"] = config.regularization
        row["regularization_blend"] = float(config.regularization_blend)

    diagnostics = {
        "raw_cost_volume": raw_cost_volume.detach().cpu(),
        "regularized_cost_volume": regularized_cost_volume.detach().cpu(),
        "raw_probability_volume": raw_probability_volume.detach().cpu(),
        "regularized_probability_volume": regularized_probability_volume.detach().cpu(),
        "all_views_valid": all_views_valid.detach().cpu(),
        "depth_values": depth_values.detach().cpu(),
    }
    rows = {"raw_cost": raw_row, "regularized_cost": regularized_row}
    maps_by_stage = {"raw_cost": raw_maps, "regularized_cost": regularized_maps}
    return rows, maps_by_stage, diagnostics, depth_gt.detach().cpu(), projection_matrices.detach().cpu()


def run_fpn_p2_cost_volume_test(
    sample_index: int = 0,
    paths: ProjectPaths | None = None,
    dtu_config: DTUConfig | None = None,
    config: FPNCostVolumeConfig | None = None,
    output_root: str | Path = "outputs/fpn_p2_cost_volume",
    device: str | torch.device | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    dtu_config = dtu_config or DTUConfig()
    config = config or FPNCostVolumeConfig()
    device_t = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    level_name = pyramid_level_name(config.pyramid_level)

    dataset = build_dtu_dataset(paths=paths, config=dtu_config)
    sample = dataset[sample_index]
    output_dir = Path(output_root) / sample["sample_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_outputs = extract_fpn_matching_features(sample, config, device_t)
    selected_features = feature_outputs["matching_feature"].unsqueeze(0).contiguous()

    rows, maps_by_stage, diagnostics, depth_gt, projection_matrices = build_depth_from_fpn_cost_volume(
        sample=sample,
        features_for_cost=selected_features,
        config=config,
        device=device_t,
    )

    feature_overview_path = save_fpn_matching_overview(
        {level: feature.detach().cpu() for level, feature in feature_outputs["fpn_features"].items()},
        feature_outputs["matching_feature"].detach().cpu(),
        output_dir / f"fpn_{level_name.lower()}_matching_feature.png",
    )
    raw_result_path = save_depth_result_image(
        f"FPN matching {level_name} raw cost",
        rows["raw_cost"],
        maps_by_stage["raw_cost"],
        depth_gt,
        output_dir / f"fpn_{level_name.lower()}_raw_depth_result.png",
    )
    regularized_result_path = save_depth_result_image(
        f"FPN matching {level_name} regularized cost",
        rows["regularized_cost"],
        maps_by_stage["regularized_cost"],
        depth_gt,
        output_dir / f"fpn_{level_name.lower()}_regularized_depth_result.png",
    )
    raw_diagnostics_path = save_cost_volume_diagnostics(
        cost_volume=diagnostics["raw_cost_volume"],
        probability_volume=diagnostics["raw_probability_volume"],
        depth_values=diagnostics["depth_values"],
        all_views_valid=diagnostics["all_views_valid"],
        argmin_depth=maps_by_stage["raw_cost"]["argmin_depth"],
        soft_depth=maps_by_stage["raw_cost"]["soft_depth"],
        confidence=maps_by_stage["raw_cost"]["confidence"],
        output_path=output_dir / f"fpn_{level_name.lower()}_raw_cost_volume_diagnostics.png",
    )
    regularized_diagnostics_path = save_cost_volume_diagnostics(
        cost_volume=diagnostics["regularized_cost_volume"],
        probability_volume=diagnostics["regularized_probability_volume"],
        depth_values=diagnostics["depth_values"],
        all_views_valid=diagnostics["all_views_valid"],
        argmin_depth=maps_by_stage["regularized_cost"]["argmin_depth"],
        soft_depth=maps_by_stage["regularized_cost"]["soft_depth"],
        confidence=maps_by_stage["regularized_cost"]["confidence"],
        output_path=output_dir / f"fpn_{level_name.lower()}_regularized_cost_volume_diagnostics.png",
    )

    stage_paths = {
        "raw_cost": {
            "depth_visualization": raw_result_path,
            "cost_volume_diagnostics": raw_diagnostics_path,
        },
        "regularized_cost": {
            "depth_visualization": regularized_result_path,
            "cost_volume_diagnostics": regularized_diagnostics_path,
        },
    }

    for stage, row in rows.items():
        row["feature_overview"] = str(feature_overview_path)
        row["depth_visualization"] = str(stage_paths[stage]["depth_visualization"])
        row["cost_volume_diagnostics"] = str(stage_paths[stage]["cost_volume_diagnostics"])
        row["sample_name"] = sample["sample_name"]
        row["scan_name"] = sample["scan_name"]
        row["ref_view"] = int(sample["ref_view"])
        row["view_ids"] = str([int(v) for v in sample["view_ids"]])
        row["image_shape"] = str(tuple(feature_outputs["images"].shape))
        row["feature_shape"] = str(tuple(selected_features.shape))
        row["projection_shape"] = str(tuple(projection_matrices.shape))

    metrics_df = pd.DataFrame([rows["raw_cost"], rows["regularized_cost"]])
    metrics_csv_path = output_dir / f"fpn_{level_name.lower()}_cost_volume_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return {
        "sample": sample,
        "level": level_name,
        "image_shape": tuple(feature_outputs["images"].shape),
        "features_shape": tuple(selected_features.shape),
        "projection_shape": tuple(projection_matrices.shape),
        "depth_values_shape": tuple(diagnostics["depth_values"].shape),
        "cost_volume_shape": tuple(diagnostics["regularized_cost_volume"].shape),
        "probability_volume_shape": tuple(diagnostics["regularized_probability_volume"].shape),
        "metrics_df": metrics_df,
        "metrics_csv_path": metrics_csv_path,
        "feature_overview_path": feature_overview_path,
        "raw_depth_visualization_path": raw_result_path,
        "raw_cost_volume_diagnostics_path": raw_diagnostics_path,
        "regularized_depth_visualization_path": regularized_result_path,
        "regularized_cost_volume_diagnostics_path": regularized_diagnostics_path,
        "output_dir": output_dir,
    }
