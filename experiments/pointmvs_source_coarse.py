"""PointMVSNet source coarse network visualization test."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.dtu import build_dtu_dataset
from experiments.cost_volume import compute_depth_error_summary
from experiments.geometry import resize_gt_to_feature_grid
from experiments.pointmvs_coarse import (
    build_point_variance_volume,
    normalize_for_display,
    regress_depth,
    save_coarse_depth_visualization,
    save_cost_volume_overview,
)
from upr_mvs.config import DTUConfig, ProjectPaths
from upr_mvs.external import sample_depth_planes, scale_intrinsics


POINTMVS_ROOT = Path(__file__).resolve().parents[1] / "models/PointMVSNet"
if str(POINTMVS_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTMVS_ROOT))

from pointmvsnet.networks import ImageConv, VolumeConv  # noqa: E402


@dataclass(frozen=True)
class PointMVSSourceCoarseConfig:
    """Configuration for the original PointMVSNet coarse network test."""

    max_side: int = 768
    img_base_channels: int = 8
    volume_base_channels: int = 8
    num_depths: int = 48
    temperature: float = 1.0
    point_chunk_size: int = 200_000
    rgb_to_bgr: bool = True
    normalize_images: bool = True
    load_weights: bool = True
    pointmvs_checkpoint: Path | None = Path("models/PointMVSNet/outputs/dtu_wde3/model_pretrained.pth")


def maybe_resize_images(images: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return images
    height, width = images.shape[-2:]
    scale = float(max_side) / float(max(height, width))
    if scale >= 1.0:
        return images
    target_hw = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    return F.interpolate(images, size=target_hw, mode="bilinear", align_corners=False)


def pointmvs_preprocess_images(images_rgb_255: torch.Tensor, config: PointMVSSourceCoarseConfig) -> torch.Tensor:
    images = images_rgb_255
    if config.rgb_to_bgr:
        images = images[:, [2, 1, 0]]
    images = images.float()
    if config.normalize_images:
        mean = images.mean(dim=(2, 3), keepdim=True)
        var = images.var(dim=(2, 3), keepdim=True, unbiased=False)
        images = (images - mean) / (torch.sqrt(var) + 1e-7)
    return images


def load_source_coarse_weights(
    image_model: ImageConv,
    volume_model: VolumeConv,
    checkpoint_path: str | Path | None,
) -> dict:
    if checkpoint_path is None:
        return {"loaded": False, "reason": "checkpoint disabled"}
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        return {"loaded": False, "reason": f"missing checkpoint: {checkpoint_path}"}

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    def strip_prefix(prefix: str) -> dict:
        return {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

    image_result = image_model.load_state_dict(strip_prefix("module.coarse_img_conv."), strict=False)
    volume_result = volume_model.load_state_dict(strip_prefix("module.coarse_vol_conv."), strict=False)
    return {
        "loaded": True,
        "checkpoint": str(checkpoint_path),
        "image_missing": list(image_result.missing_keys),
        "image_unexpected": list(image_result.unexpected_keys),
        "volume_missing": list(volume_result.missing_keys),
        "volume_unexpected": list(volume_result.unexpected_keys),
    }


def feature_energy(feature: torch.Tensor) -> np.ndarray:
    return feature[0].detach().abs().mean(dim=0).cpu().numpy()


def save_source_feature_overview(features: dict[str, torch.Tensor], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    levels = ["conv0", "conv1", "conv2", "conv3"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, level in zip(axes, levels):
        image = normalize_for_display(feature_energy(features[level]))
        title = f"{level} {tuple(features[level].shape)}"
        if level == "conv3":
            title += "\nselected coarse feature"
        ax.imshow(image, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("PointMVSNet source ImageConv features", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_pointmvs_source_coarse_depth_test(
    sample_index: int = 0,
    paths: ProjectPaths | None = None,
    dtu_config: DTUConfig | None = None,
    config: PointMVSSourceCoarseConfig | None = None,
    output_root: str | Path = "outputs/pointmvs_source_coarse",
    device: str | torch.device | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    dtu_config = dtu_config or DTUConfig()
    config = config or PointMVSSourceCoarseConfig()
    device_t = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dtu_dataset(paths=paths, config=dtu_config)
    sample = dataset[sample_index]
    output_dir = Path(output_root) / sample["sample_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    image_model = ImageConv(config.img_base_channels).to(device_t).eval()
    volume_model = VolumeConv(image_model.out_channels, config.volume_base_channels).to(device_t).eval()
    load_info = {"loaded": False, "reason": "disabled"}
    if config.load_weights:
        checkpoint_path = config.pointmvs_checkpoint
        if checkpoint_path is not None and not checkpoint_path.is_absolute():
            checkpoint_path = paths.repo_root / checkpoint_path
        load_info = load_source_coarse_weights(image_model, volume_model, checkpoint_path)

    images_raw = sample["imgs"].to(device=device_t, dtype=torch.float32)
    images_raw = maybe_resize_images(images_raw, config.max_side)
    images = pointmvs_preprocess_images(images_raw, config)

    with torch.inference_mode():
        feature_maps = []
        ref_feature_pyramid = None
        for view_index in range(images.shape[0]):
            pyramid = image_model(images[view_index : view_index + 1])
            if view_index == 0:
                ref_feature_pyramid = {key: value.detach().cpu() for key, value in pyramid.items()}
            feature_maps.append(pyramid["conv3"])
        feature_tensor = torch.stack(feature_maps, dim=1)
        feature_h, feature_w = feature_tensor.shape[-2:]

        intrinsics = sample["intrinsics"].to(device=device_t, dtype=torch.float32)
        extrinsics = sample["extrinsics"].to(device=device_t, dtype=torch.float32)
        original_h, original_w = sample["imgs"].shape[-2:]
        scaled_intrinsics = scale_intrinsics(
            intrinsics,
            scale_x=float(feature_w) / float(original_w),
            scale_y=float(feature_h) / float(original_h),
        ).unsqueeze(0)
        extrinsics_b = extrinsics.unsqueeze(0)
        depth_range = sample["depth_range"].view(1, 2).to(device=device_t, dtype=torch.float32)
        depth_values = sample_depth_planes(depth_range, config.num_depths)

        variance_volume, world_points = build_point_variance_volume(
            feature_tensor,
            scaled_intrinsics,
            extrinsics_b,
            depth_values,
            point_chunk_size=config.point_chunk_size,
        )
        raw_scalar_cost = variance_volume.mean(dim=1)
        raw_outputs = regress_depth(raw_scalar_cost, depth_values, temperature=config.temperature)
        filtered_cost = volume_model(variance_volume).squeeze(1)
        filtered_outputs = regress_depth(filtered_cost, depth_values, temperature=config.temperature)

    depth_gt, valid_mask = resize_gt_to_feature_grid(sample, (feature_h, feature_w), device_t)
    raw_valid = valid_mask & torch.isfinite(raw_outputs["soft_depth"])
    filtered_valid = valid_mask & torch.isfinite(filtered_outputs["soft_depth"])
    raw_summary = compute_depth_error_summary(raw_outputs["soft_depth"], depth_gt, raw_valid)
    filtered_summary = compute_depth_error_summary(filtered_outputs["soft_depth"], depth_gt, filtered_valid)

    rows = [
        {
            "stage": "raw_variance",
            "feature": "PointMVSNet ImageConv conv3",
            "channels": int(feature_tensor.shape[2]),
            "height": int(feature_h),
            "width": int(feature_w),
            "num_depths": int(config.num_depths),
            "soft_mean": raw_summary["mean"],
            "soft_median": raw_summary["median"],
            "soft_p90": raw_summary["p90"],
            "confidence_mean": float(raw_outputs["confidence"][raw_valid].mean()) if raw_valid.any() else float("nan"),
            "confidence_median": float(raw_outputs["confidence"][raw_valid].median()) if raw_valid.any() else float("nan"),
            "num_eval_pixels": raw_summary["num_valid"],
        },
        {
            "stage": "volume_conv",
            "feature": "PointMVSNet ImageConv conv3",
            "channels": int(feature_tensor.shape[2]),
            "height": int(feature_h),
            "width": int(feature_w),
            "num_depths": int(config.num_depths),
            "soft_mean": filtered_summary["mean"],
            "soft_median": filtered_summary["median"],
            "soft_p90": filtered_summary["p90"],
            "confidence_mean": float(filtered_outputs["confidence"][filtered_valid].mean()) if filtered_valid.any() else float("nan"),
            "confidence_median": float(filtered_outputs["confidence"][filtered_valid].median()) if filtered_valid.any() else float("nan"),
            "prob_map_mean": float(filtered_outputs["prob_map"][filtered_valid].mean()) if filtered_valid.any() else float("nan"),
            "prob_map_median": float(filtered_outputs["prob_map"][filtered_valid].median()) if filtered_valid.any() else float("nan"),
            "num_eval_pixels": filtered_summary["num_valid"],
        },
    ]

    feature_path = save_source_feature_overview(ref_feature_pyramid, output_dir / "source_imageconv_feature_overview.png")
    depth_path = save_coarse_depth_visualization(
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in raw_outputs.items()},
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in filtered_outputs.items()},
        depth_gt.detach().cpu(),
        valid_mask.detach().cpu(),
        output_dir / "source_coarse_depth_result.png",
    )
    cost_path = save_cost_volume_overview(
        raw_scalar_cost.detach().cpu(),
        filtered_cost.detach().cpu(),
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in raw_outputs.items()},
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in filtered_outputs.items()},
        depth_values.detach().cpu(),
        output_dir / "source_coarse_cost_volume_overview.png",
    )

    for row in rows:
        row["sample_name"] = sample["sample_name"]
        row["scan_name"] = sample["scan_name"]
        row["ref_view"] = int(sample["ref_view"])
        row["view_ids"] = str([int(v) for v in sample["view_ids"]])
        row["image_shape"] = str(tuple(images.shape))
        row["feature_shape"] = str(tuple(feature_tensor.shape))
        row["depth_range"] = str((float(depth_values[0, 0]), float(depth_values[0, -1])))
        row["weights_loaded"] = bool(load_info.get("loaded", False))

    metrics_df = pd.DataFrame(rows)
    metrics_csv_path = output_dir / "pointmvs_source_coarse_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return {
        "sample": sample,
        "image_shape": tuple(images.shape),
        "feature_shape": tuple(feature_tensor.shape),
        "variance_volume_shape": tuple(variance_volume.shape),
        "filtered_cost_shape": tuple(filtered_cost.shape),
        "world_points_shape": tuple(world_points.shape),
        "depth_values_shape": tuple(depth_values.shape),
        "load_info": load_info,
        "metrics_df": metrics_df,
        "metrics_csv_path": metrics_csv_path,
        "feature_visualization_path": feature_path,
        "depth_visualization_path": depth_path,
        "cost_visualization_path": cost_path,
        "output_dir": output_dir,
    }
