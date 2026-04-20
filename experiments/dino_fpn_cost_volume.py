"""Build a cost volume from fused DINOv3+FPN pyramid features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.dtu import build_dtu_dataset
from experiments.cost_volume import (
    build_variance_cost_volume_chunked,
    compute_depth_error_summary,
    depth_regression_from_cost,
)
from experiments.dino_fpn_fusion import (
    DinoConcatFusion,
    FPNDinoPyramidFusion,
    FPN_LEVELS,
    maybe_resize_image,
)
from experiments.fpn import ConvFPNVisualizationNet
from experiments.geometry import make_projection_for_feature_grid, resize_gt_to_feature_grid
from experiments.visualization import save_depth_result_image
from models.dinov3.extractor import extract_dinov3_native_features, load_dinov3_vit_base
from upr_mvs.config import DTUConfig, ProjectPaths
from upr_mvs.external import sample_depth_planes


@dataclass(frozen=True)
class DinoFPNCostVolumeConfig:
    """Configuration for the l=2 fused-feature cost-volume test."""

    pyramid_level: int = 2
    dino_layer_numbers: tuple[int, int, int] = (3, 7, 11)
    max_side: int = 768
    dino_input_max_side: int = 0
    fpn_channels: int = 16
    dino_fused_channels: int = 16
    fused_channels: int = 16
    num_depths: int = 64
    temperature: float = 0.02
    channel_chunk: int = 4
    patch_size: int = 16
    regularization: str = "avg3d"
    regularization_blend: float = 0.5
    regularization_kernel: tuple[int, int, int] = (3, 3, 3)


def _layer_numbers_to_indices(layer_numbers: tuple[int, ...]) -> tuple[int, ...]:
    if len(layer_numbers) != 3:
        raise ValueError(f"Expected exactly three DINO layers, got {layer_numbers}")
    layer_indices = tuple(layer_number - 1 for layer_number in layer_numbers)
    if any(layer_index < 0 or layer_index >= 12 for layer_index in layer_indices):
        raise ValueError(f"DINO layer numbers must be in [1, 12], got {layer_numbers}")
    return layer_indices


def _pyramid_level_name(level: int) -> str:
    level_name = f"P{level}"
    if level_name not in FPN_LEVELS:
        raise ValueError(f"Supported FPN levels are {FPN_LEVELS}, got l={level}")
    return level_name


def regularize_cost_volume(
    cost_volume: torch.Tensor,
    mode: str = "avg3d",
    blend: float = 0.5,
    kernel_size: tuple[int, int, int] = (3, 3, 3),
) -> torch.Tensor:
    """Apply a deterministic 3D smoothing regularizer for module testing.

    This is not a trained MVSNet regularizer. It keeps the test lightweight while
    still exercising the cost-volume regularization stage in the pipeline.
    """

    if mode == "none" or blend <= 0.0:
        return cost_volume
    if mode != "avg3d":
        raise ValueError(f"Unsupported cost regularization mode: {mode}")

    depth_kernel, height_kernel, width_kernel = kernel_size
    padding = (depth_kernel // 2, height_kernel // 2, width_kernel // 2)
    smoothed = F.avg_pool3d(
        cost_volume.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    ).squeeze(1)
    blend = float(np.clip(blend, 0.0, 1.0))
    return (1.0 - blend) * cost_volume + blend * smoothed


def extract_multiview_fused_pyramid(
    sample: dict,
    dino_model: torch.nn.Module,
    config: DinoFPNCostVolumeConfig,
    device: torch.device,
) -> dict:
    images = sample["imgs"].to(device=device, dtype=torch.float32) / 255.0
    images = maybe_resize_image(images, config.max_side)

    fpn_model = ConvFPNVisualizationNet(
        c2_channels=config.fpn_channels,
        c3_channels=config.fpn_channels,
        c4_channels=config.fpn_channels,
        c5_channels=config.fpn_channels,
        out_channels=config.fpn_channels,
    ).to(device).eval()

    dino_input_max_side = config.dino_input_max_side
    if dino_input_max_side <= 0:
        dino_input_max_side = int(max(images.shape[-2:]))

    dino_sample = dict(sample)
    dino_sample["imgs"] = (images.detach().cpu() * 255.0).float()

    with torch.inference_mode():
        fpn_features_all = fpn_model(images)
        fpn_features = {level: fpn_features_all[level] for level in FPN_LEVELS}

        dino_output = extract_dinov3_native_features(
            sample=dino_sample,
            model=dino_model,
            device=device,
            max_side=dino_input_max_side,
            patch_size=config.patch_size,
            layers=_layer_numbers_to_indices(config.dino_layer_numbers),
        )

        dino_layer_features = list(dino_output["layer_features"])
        dino_fuser = DinoConcatFusion(
            in_channels=int(dino_layer_features[0].shape[1]),
            num_layers=len(dino_layer_features),
            out_channels=config.dino_fused_channels,
        ).to(device).eval()
        dino_fused = dino_fuser(dino_layer_features)

        pyramid_fuser = FPNDinoPyramidFusion(
            fpn_channels=config.fpn_channels,
            dino_channels=config.dino_fused_channels,
            out_channels=config.fused_channels,
        ).to(device).eval()
        fused_pyramid = pyramid_fuser(fpn_features, dino_fused)

    return {
        "images": images,
        "fpn_features": fpn_features,
        "dino_layers": dino_layer_features,
        "dino_fused": dino_fused,
        "fused_pyramid": fused_pyramid,
        "dino_input_hw": dino_output["input_hw"],
        "dino_native_feature_hw": dino_output["native_feature_hw"],
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

    cost_axes = [axes[1, 2], axes[1, 3], axes[2, 0]]
    for ax, depth_index in zip(cost_axes, slice_ids):
        image = _safe_numpy(masked_cost[depth_index])
        ax.imshow(_normalize_for_display(image), cmap="magma")
        ax.set_title(f"cost slice d={depth_list[depth_index]:.1f}")
        ax.axis("off")

    prob_axes = [axes[2, 1], axes[2, 2], axes[2, 3]]
    for ax, depth_index in zip(prob_axes, slice_ids):
        image = _safe_numpy(prob[depth_index])
        ax.imshow(_normalize_for_display(image), cmap="viridis")
        ax.set_title(f"prob slice d={depth_list[depth_index]:.1f}")
        ax.axis("off")

    fig.suptitle("P2 fused-feature variance cost volume diagnostics", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_depth_from_fused_p2_cost_volume(
    sample: dict,
    features_for_cost: torch.Tensor,
    config: DinoFPNCostVolumeConfig,
    device: torch.device,
) -> tuple[dict, dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_hw = tuple(features_for_cost.shape[-2:])
    projection_matrices = make_projection_for_feature_grid(sample, feature_hw, device)
    depth_range = sample["depth_range"].view(1, 2).to(device=device, dtype=torch.float32)
    depth_values = sample_depth_planes(depth_range, config.num_depths)
    depth_gt, mask = resize_gt_to_feature_grid(sample, feature_hw, device)

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

        masked_cost = regularized_cost_volume.masked_fill(~all_views_valid, float("inf"))
        has_valid_depth = torch.isfinite(masked_cost).any(dim=1)
        best_depth_idx = masked_cost.argmin(dim=1)
        depth_volume = depth_values.view(depth_values.shape[0], depth_values.shape[1], 1, 1).expand_as(regularized_cost_volume)
        argmin_depth = torch.gather(depth_volume, dim=1, index=best_depth_idx.unsqueeze(1)).squeeze(1)
        argmin_depth = torch.where(has_valid_depth, argmin_depth, torch.full_like(argmin_depth, float("nan")))

        soft_outputs = depth_regression_from_cost(
            cost_volume=regularized_cost_volume,
            depth_values=depth_values,
            candidate_mask=all_views_valid,
            temperature=config.temperature,
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
        "feature": "FPN+DINO fused feature",
        "level": f"P{config.pyramid_level}",
        "channels": int(features_for_cost.shape[2]),
        "height": int(features_for_cost.shape[-2]),
        "width": int(features_for_cost.shape[-1]),
        "num_depths": int(config.num_depths),
        "temperature": float(config.temperature),
        "regularization": config.regularization,
        "regularization_blend": float(config.regularization_blend),
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
    diagnostics = {
        "raw_cost_volume": raw_cost_volume.detach().cpu(),
        "regularized_cost_volume": regularized_cost_volume.detach().cpu(),
        "probability_volume": probability_volume.detach().cpu(),
        "all_views_valid": all_views_valid.detach().cpu(),
        "depth_values": depth_values.detach().cpu(),
    }
    return row, maps, diagnostics, depth_gt.detach().cpu(), projection_matrices.detach().cpu()


def run_dino_fpn_p2_cost_volume_test(
    sample_index: int = 0,
    paths: ProjectPaths | None = None,
    dtu_config: DTUConfig | None = None,
    cost_config: DinoFPNCostVolumeConfig | None = None,
    output_root: str | Path = "outputs/dino_fpn_p2_cost_volume",
    device: str | torch.device | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    dtu_config = dtu_config or DTUConfig()
    cost_config = cost_config or DinoFPNCostVolumeConfig()
    device_t = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    level_name = _pyramid_level_name(cost_config.pyramid_level)

    dataset = build_dtu_dataset(paths=paths, config=dtu_config)
    sample = dataset[sample_index]
    output_dir = Path(output_root) / sample["sample_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    dino_model = load_dinov3_vit_base(
        device=device_t,
        weights_file=paths.dinov3_weights_file,
        patch_size=cost_config.patch_size,
    )
    feature_outputs = extract_multiview_fused_pyramid(sample, dino_model, cost_config, device_t)
    selected_features = feature_outputs["fused_pyramid"][level_name].unsqueeze(0).contiguous()

    row, maps, diagnostics, depth_gt, projection_matrices = build_depth_from_fused_p2_cost_volume(
        sample=sample,
        features_for_cost=selected_features,
        config=cost_config,
        device=device_t,
    )

    result_path = save_depth_result_image(
        f"FPN+DINO fused {level_name}",
        row,
        maps,
        depth_gt,
        output_dir / f"fused_{level_name.lower()}_depth_result.png",
    )
    diagnostics_path = save_cost_volume_diagnostics(
        cost_volume=diagnostics["regularized_cost_volume"],
        probability_volume=diagnostics["probability_volume"],
        depth_values=diagnostics["depth_values"],
        all_views_valid=diagnostics["all_views_valid"],
        argmin_depth=maps["argmin_depth"],
        soft_depth=maps["soft_depth"],
        confidence=maps["confidence"],
        output_path=output_dir / f"fused_{level_name.lower()}_cost_volume_diagnostics.png",
    )

    row["depth_visualization"] = str(result_path)
    row["cost_volume_diagnostics"] = str(diagnostics_path)
    row["sample_name"] = sample["sample_name"]
    row["scan_name"] = sample["scan_name"]
    row["ref_view"] = int(sample["ref_view"])
    row["view_ids"] = str([int(v) for v in sample["view_ids"]])
    row["dino_layers"] = str(cost_config.dino_layer_numbers)
    row["dino_input_hw"] = str(feature_outputs["dino_input_hw"])
    row["dino_native_feature_hw"] = str(feature_outputs["dino_native_feature_hw"])
    row["feature_shape"] = str(tuple(selected_features.shape))
    row["projection_shape"] = str(tuple(projection_matrices.shape))

    metrics_df = pd.DataFrame([row])
    metrics_csv_path = output_dir / f"fused_{level_name.lower()}_cost_volume_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return {
        "sample": sample,
        "level": level_name,
        "features_shape": tuple(selected_features.shape),
        "projection_shape": tuple(projection_matrices.shape),
        "depth_values_shape": tuple(diagnostics["depth_values"].shape),
        "cost_volume_shape": tuple(diagnostics["regularized_cost_volume"].shape),
        "probability_volume_shape": tuple(diagnostics["probability_volume"].shape),
        "metrics_df": metrics_df,
        "metrics_csv_path": metrics_csv_path,
        "depth_visualization_path": result_path,
        "cost_volume_diagnostics_path": diagnostics_path,
        "output_dir": output_dir,
        "dino_input_hw": feature_outputs["dino_input_hw"],
        "dino_native_feature_hw": feature_outputs["dino_native_feature_hw"],
    }
