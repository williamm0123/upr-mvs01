"""PointMVSNet-style coarse depth prediction using local FPN features."""

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
from experiments.cost_volume import compute_depth_error_summary
from experiments.fpn import ConvFPNVisualizationNet
from experiments.geometry import resize_gt_to_feature_grid
from upr_mvs.config import DTUConfig, ProjectPaths
from upr_mvs.external import sample_depth_planes, scale_intrinsics


FPN_LEVELS = ("P2", "P3", "P4", "P5")


@dataclass(frozen=True)
class PointMVSCoarseConfig:
    """Configuration for the PointMVSNet-style coarse module test."""

    pyramid_level: int = 4
    max_side: int = 768
    fpn_channels: int = 64
    num_depths: int = 48
    volume_base_channels: int = 8
    temperature: float = 1.0
    point_chunk_size: int = 200_000
    load_volume_weights: bool = True
    pointmvs_checkpoint: Path | None = Path("models/PointMVSNet/outputs/dtu_wde3/model_pretrained.pth")


def pyramid_level_name(level: int) -> str:
    name = f"P{level}"
    if name not in FPN_LEVELS:
        raise ValueError(f"Supported levels are {FPN_LEVELS}, got l={level}")
    return name


def maybe_resize_images(images: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return images
    height, width = images.shape[-2:]
    scale = float(max_side) / float(max(height, width))
    if scale >= 1.0:
        return images
    target_hw = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    return F.interpolate(images, size=target_hw, mode="bilinear", align_corners=False)


class ConvBnReLU3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DeconvBnReLU3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        output_padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


def _match_3d_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if x.shape[-3:] == target.shape[-3:]:
        return x
    return F.interpolate(x, size=target.shape[-3:], mode="trilinear", align_corners=False)


class PointMVSVolumeConv(nn.Module):
    """VolumeConv from PointMVSNet, reproduced without importing its package."""

    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.conv1_0 = ConvBnReLU3d(in_channels, base_channels * 2, 3, stride=2, padding=1)
        self.conv2_0 = ConvBnReLU3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv3_0 = ConvBnReLU3d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        self.conv0_1 = ConvBnReLU3d(in_channels, base_channels, 3, stride=1, padding=1)
        self.conv1_1 = ConvBnReLU3d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv2_1 = ConvBnReLU3d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = ConvBnReLU3d(base_channels * 8, base_channels * 8, 3, stride=1, padding=1)

        self.conv4_0 = DeconvBnReLU3d(base_channels * 8, base_channels * 4, 3, 2, padding=1, output_padding=1)
        self.conv5_0 = DeconvBnReLU3d(base_channels * 4, base_channels * 2, 3, 2, padding=1, output_padding=1)
        self.conv6_0 = DeconvBnReLU3d(base_channels * 2, base_channels, 3, 2, padding=1, output_padding=1)
        self.conv6_2 = nn.Conv3d(base_channels, 1, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv0_1 = self.conv0_1(x)
        conv1_0 = self.conv1_0(x)
        conv2_0 = self.conv2_0(conv1_0)
        conv3_0 = self.conv3_0(conv2_0)

        conv1_1 = self.conv1_1(conv1_0)
        conv2_1 = self.conv2_1(conv2_0)
        conv3_1 = self.conv3_1(conv3_0)

        conv4_0 = _match_3d_size(self.conv4_0(conv3_1), conv2_1)
        conv5_0 = _match_3d_size(self.conv5_0(conv4_0 + conv2_1), conv1_1)
        conv6_0 = _match_3d_size(self.conv6_0(conv5_0 + conv1_1), conv0_1)
        return self.conv6_2(conv6_0 + conv0_1)


def load_pointmvs_volume_weights(model: PointMVSVolumeConv, checkpoint_path: str | Path | None) -> dict:
    if checkpoint_path is None:
        return {"loaded": False, "missing": [], "unexpected": [], "reason": "checkpoint disabled"}
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        return {"loaded": False, "missing": [], "unexpected": [], "reason": f"missing checkpoint: {checkpoint_path}"}

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    volume_state = {}
    prefix = "module.coarse_vol_conv."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            volume_state[key[len(prefix) :]] = value

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in volume_state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    incompatible = sorted(set(volume_state) - set(compatible))
    load_result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": bool(compatible),
        "missing": list(load_result.missing_keys),
        "unexpected": list(load_result.unexpected_keys),
        "incompatible": incompatible,
        "num_loaded": len(compatible),
        "checkpoint": str(checkpoint_path),
    }


def get_pixel_grids(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.linspace(0.5, width - 0.5, width, device=device, dtype=dtype).view(1, width).expand(height, width)
    y = torch.linspace(0.5, height - 0.5, height, device=device, dtype=dtype).view(height, 1).expand(height, width)
    ones = torch.ones(height, width, device=device, dtype=dtype)
    return torch.stack([x.reshape(-1), y.reshape(-1), ones.reshape(-1)], dim=0)


def make_world_points_from_ref_grid(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    depth_values: torch.Tensor,
    feature_hw: tuple[int, int],
) -> torch.Tensor:
    batch_size = depth_values.shape[0]
    feature_h, feature_w = feature_hw
    grid = get_pixel_grids(feature_h, feature_w, intrinsics.device, intrinsics.dtype)
    grid = grid.view(1, 3, -1).expand(batch_size, 3, -1)

    ref_intrinsics = intrinsics[:, 0]
    ref_extrinsics = extrinsics[:, 0]
    rays = torch.matmul(torch.inverse(ref_intrinsics), grid)
    cam_points = rays.unsqueeze(2) * depth_values.view(batch_size, 1, -1, 1)
    cam_points = cam_points.reshape(batch_size, 3, -1)

    rotation = ref_extrinsics[:, :3, :3]
    translation = ref_extrinsics[:, :3, 3:4]
    return torch.matmul(rotation.transpose(1, 2), cam_points - translation)


def fetch_point_features(
    feature_maps: torch.Tensor,
    world_points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_views, channels, feature_h, feature_w = feature_maps.shape
    num_points = world_points.shape[-1]
    feature_maps_flat = feature_maps.reshape(batch_size * num_views, channels, feature_h, feature_w)

    points = world_points.unsqueeze(1).expand(batch_size, num_views, 3, num_points)
    points = points.reshape(batch_size * num_views, 3, num_points)
    rotations = extrinsics[:, :, :3, :3].reshape(batch_size * num_views, 3, 3)
    translations = extrinsics[:, :, :3, 3:4].reshape(batch_size * num_views, 3, 1)
    camera_points = torch.bmm(rotations, points) + translations

    z = camera_points[:, 2]
    z = torch.where(z.abs() > 1e-6, z, torch.full_like(z, 1e-6))
    normalized = torch.stack(
        [
            camera_points[:, 0] / z,
            camera_points[:, 1] / z,
            torch.ones_like(z),
        ],
        dim=1,
    )
    uv = torch.bmm(intrinsics.reshape(batch_size * num_views, 3, 3), normalized)[:, :2]
    grid = uv.transpose(1, 2).reshape(batch_size * num_views, num_points, 1, 2)
    grid[..., 0] = ((grid[..., 0] - 0.5) / float(max(feature_w - 1, 1))) * 2.0 - 1.0
    grid[..., 1] = ((grid[..., 1] - 0.5) / float(max(feature_h - 1, 1))) * 2.0 - 1.0

    sampled = F.grid_sample(
        feature_maps_flat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(3).view(batch_size, num_views, channels, num_points)


def build_point_variance_volume(
    feature_maps: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    depth_values: torch.Tensor,
    point_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_views, channels, feature_h, feature_w = feature_maps.shape
    world_points = make_world_points_from_ref_grid(intrinsics, extrinsics, depth_values, (feature_h, feature_w))
    num_depths = depth_values.shape[1]
    num_points = world_points.shape[-1]
    ref_features = feature_maps[:, 0].unsqueeze(2).expand(batch_size, channels, num_depths, feature_h, feature_w)
    ref_features = ref_features.contiguous().view(batch_size, channels, num_points)

    variance_chunks = []
    chunk_size = max(1, int(point_chunk_size))
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        point_features = fetch_point_features(
            feature_maps,
            world_points[:, :, start:end],
            intrinsics,
            extrinsics,
        )
        point_features[:, 0] = ref_features[:, :, start:end]
        mean = point_features.mean(dim=1)
        mean_square = point_features.square().mean(dim=1)
        variance_chunks.append(mean_square - mean.square())

    variance = torch.cat(variance_chunks, dim=-1)
    cost_volume = variance.view(batch_size, channels, num_depths, feature_h, feature_w)
    return cost_volume, world_points


def probability_map_from_depth(
    probability_volume: torch.Tensor,
    depth_map: torch.Tensor,
    depth_values: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, height, width = depth_map.shape
    num_depths = probability_volume.shape[1]
    interval = (depth_values[:, 1] - depth_values[:, 0]).view(batch_size, 1, 1, 1).clamp_min(1e-6)
    depth_start = depth_values[:, 0].view(batch_size, 1, 1, 1)
    coords = ((depth_map - depth_start) / interval).view(batch_size, height, width)
    left = coords.floor().clamp(0, num_depths - 1).long()
    right = coords.ceil().clamp(0, num_depths - 1).long()
    b = torch.arange(batch_size, device=depth_map.device).view(batch_size, 1, 1).expand(batch_size, height, width)
    y = torch.arange(height, device=depth_map.device).view(1, height, 1).expand(batch_size, height, width)
    x = torch.arange(width, device=depth_map.device).view(1, 1, width).expand(batch_size, height, width)
    prob = probability_volume[b, left, y, x] + probability_volume[b, right, y, x]
    return prob.unsqueeze(1)


def regress_depth(cost_volume: torch.Tensor, depth_values: torch.Tensor, temperature: float) -> dict:
    scaled_cost = cost_volume / max(float(temperature), 1e-6)
    probability = F.softmax(-scaled_cost, dim=1)
    depth = (probability * depth_values.view(depth_values.shape[0], depth_values.shape[1], 1, 1)).sum(dim=1, keepdim=True)
    argmin_idx = cost_volume.argmin(dim=1, keepdim=True)
    depth_candidates = depth_values.view(depth_values.shape[0], depth_values.shape[1], 1, 1).expand_as(cost_volume)
    argmin_depth = torch.gather(depth_candidates, dim=1, index=argmin_idx).squeeze(1)
    prob_map = probability_map_from_depth(probability, depth, depth_values)
    confidence = probability.max(dim=1, keepdim=True).values
    return {
        "probability_volume": probability,
        "soft_depth": depth.squeeze(1),
        "argmin_depth": argmin_depth,
        "prob_map": prob_map.squeeze(1),
        "confidence": confidence.squeeze(1),
    }


def normalize_for_display(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.nanpercentile(array[finite], [1, 99])
    if high <= low:
        high = low + 1e-6
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def feature_energy(feature: torch.Tensor) -> np.ndarray:
    return feature[0].detach().abs().mean(dim=0).cpu().numpy()


def save_fpn_feature_overview(
    fpn_features: dict[str, torch.Tensor],
    selected_level: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, level in zip(axes, FPN_LEVELS):
        image = normalize_for_display(feature_energy(fpn_features[level]))
        ax.imshow(image, cmap="viridis")
        title = f"{level} {tuple(fpn_features[level].shape)}"
        if level == selected_level:
            title += "\nselected coarse feature"
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("FPN feature pyramid for PointMVS-style coarse depth", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_coarse_depth_visualization(
    raw_outputs: dict,
    filtered_outputs: dict,
    depth_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_depth = raw_outputs["soft_depth"][0].detach().cpu().numpy()
    filtered_depth = filtered_outputs["soft_depth"][0].detach().cpu().numpy()
    gt = depth_gt[0].detach().cpu().numpy()
    prob_map = filtered_outputs["prob_map"][0].detach().cpu().numpy()
    confidence = filtered_outputs["confidence"][0].detach().cpu().numpy()
    abs_error = np.abs(filtered_depth - gt)
    abs_error[~valid_mask[0].detach().cpu().numpy()] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        ("raw variance soft depth", raw_depth, "turbo"),
        ("VolumeConv soft depth", filtered_depth, "turbo"),
        ("GT depth", gt, "turbo"),
        ("coarse prob map", prob_map, "viridis"),
        ("max probability", confidence, "viridis"),
        ("VolumeConv abs error", abs_error, "inferno"),
    ]
    for ax, (title, image, cmap) in zip(axes.flatten(), panels):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_cost_volume_overview(
    raw_scalar_cost: torch.Tensor,
    filtered_cost: torch.Tensor,
    raw_outputs: dict,
    filtered_outputs: dict,
    depth_values: torch.Tensor,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_list = depth_values[0].detach().cpu().numpy()
    slice_ids = [0, raw_scalar_cost.shape[1] // 2, raw_scalar_cost.shape[1] - 1]
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    panels = [
        (axes[0, 0], "raw min cost", raw_scalar_cost[0].amin(dim=0).detach().cpu().numpy(), "magma"),
        (axes[0, 1], "filtered min cost", filtered_cost[0].amin(dim=0).detach().cpu().numpy(), "magma"),
        (axes[0, 2], "raw confidence", raw_outputs["confidence"][0].detach().cpu().numpy(), "viridis"),
        (axes[0, 3], "filtered confidence", filtered_outputs["confidence"][0].detach().cpu().numpy(), "viridis"),
    ]
    for ax, title, image, cmap in panels:
        ax.imshow(normalize_for_display(image), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    for ax, depth_idx in zip(axes[1], slice_ids + [slice_ids[-1]]):
        ax.imshow(normalize_for_display(raw_scalar_cost[0, depth_idx].detach().cpu().numpy()), cmap="magma")
        ax.set_title(f"raw cost d={depth_list[depth_idx]:.1f}")
        ax.axis("off")

    for ax, depth_idx in zip(axes[2], slice_ids + [slice_ids[-1]]):
        ax.imshow(normalize_for_display(filtered_cost[0, depth_idx].detach().cpu().numpy()), cmap="magma")
        ax.set_title(f"filtered cost d={depth_list[depth_idx]:.1f}")
        ax.axis("off")

    fig.suptitle("PointMVS-style coarse cost volume diagnostics", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_pointmvs_coarse_depth_test(
    sample_index: int = 0,
    paths: ProjectPaths | None = None,
    dtu_config: DTUConfig | None = None,
    config: PointMVSCoarseConfig | None = None,
    output_root: str | Path = "outputs/pointmvs_coarse",
    device: str | torch.device | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    dtu_config = dtu_config or DTUConfig()
    config = config or PointMVSCoarseConfig()
    device_t = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    level_name = pyramid_level_name(config.pyramid_level)

    dataset = build_dtu_dataset(paths=paths, config=dtu_config)
    sample = dataset[sample_index]
    output_dir = Path(output_root) / sample["sample_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sample["imgs"].to(device=device_t, dtype=torch.float32) / 255.0
    images = maybe_resize_images(images, config.max_side)
    fpn_model = ConvFPNVisualizationNet(
        c2_channels=config.fpn_channels,
        c3_channels=config.fpn_channels,
        c4_channels=config.fpn_channels,
        c5_channels=config.fpn_channels,
        out_channels=config.fpn_channels,
    ).to(device_t).eval()
    volume_model = PointMVSVolumeConv(config.fpn_channels, config.volume_base_channels).to(device_t).eval()
    load_info = {"loaded": False, "reason": "disabled"}
    if config.load_volume_weights:
        checkpoint_path = config.pointmvs_checkpoint
        if checkpoint_path is not None and not checkpoint_path.is_absolute():
            checkpoint_path = paths.repo_root / checkpoint_path
        load_info = load_pointmvs_volume_weights(volume_model, checkpoint_path)

    with torch.inference_mode():
        fpn_features_all = fpn_model(images)
        fpn_features = {level: fpn_features_all[level] for level in FPN_LEVELS}
        selected_features = fpn_features[level_name].unsqueeze(0).contiguous()
        feature_h, feature_w = selected_features.shape[-2:]

        intrinsics = sample["intrinsics"].to(device=device_t, dtype=torch.float32)
        extrinsics = sample["extrinsics"].to(device=device_t, dtype=torch.float32)
        image_h, image_w = sample["imgs"].shape[-2:]
        scaled_intrinsics = scale_intrinsics(
            intrinsics,
            scale_x=float(feature_w) / float(image_w),
            scale_y=float(feature_h) / float(image_h),
        ).unsqueeze(0)
        extrinsics_b = extrinsics.unsqueeze(0)

        depth_range = sample["depth_range"].view(1, 2).to(device=device_t, dtype=torch.float32)
        depth_values = sample_depth_planes(depth_range, config.num_depths)
        variance_volume, world_points = build_point_variance_volume(
            selected_features,
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
            "level": level_name,
            "channels": int(selected_features.shape[2]),
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
            "level": level_name,
            "channels": int(selected_features.shape[2]),
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

    fpn_path = save_fpn_feature_overview(
        {level: feature.detach().cpu() for level, feature in fpn_features.items()},
        level_name,
        output_dir / "fpn_feature_overview.png",
    )
    depth_path = save_coarse_depth_visualization(
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in raw_outputs.items()},
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in filtered_outputs.items()},
        depth_gt.detach().cpu(),
        valid_mask.detach().cpu(),
        output_dir / "coarse_depth_result.png",
    )
    cost_path = save_cost_volume_overview(
        raw_scalar_cost.detach().cpu(),
        filtered_cost.detach().cpu(),
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in raw_outputs.items()},
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in filtered_outputs.items()},
        depth_values.detach().cpu(),
        output_dir / "coarse_cost_volume_overview.png",
    )

    for row in rows:
        row["sample_name"] = sample["sample_name"]
        row["scan_name"] = sample["scan_name"]
        row["ref_view"] = int(sample["ref_view"])
        row["view_ids"] = str([int(v) for v in sample["view_ids"]])
        row["image_shape"] = str(tuple(images.shape))
        row["feature_shape"] = str(tuple(selected_features.shape))
        row["depth_range"] = str((float(depth_values[0, 0]), float(depth_values[0, -1])))
        row["volume_weights_loaded"] = bool(load_info.get("loaded", False))

    metrics_df = pd.DataFrame(rows)
    metrics_csv_path = output_dir / "pointmvs_coarse_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return {
        "sample": sample,
        "level": level_name,
        "image_shape": tuple(images.shape),
        "feature_shape": tuple(selected_features.shape),
        "variance_volume_shape": tuple(variance_volume.shape),
        "filtered_cost_shape": tuple(filtered_cost.shape),
        "world_points_shape": tuple(world_points.shape),
        "depth_values_shape": tuple(depth_values.shape),
        "load_info": load_info,
        "metrics_df": metrics_df,
        "metrics_csv_path": metrics_csv_path,
        "fpn_feature_path": fpn_path,
        "depth_visualization_path": depth_path,
        "cost_visualization_path": cost_path,
        "output_dir": output_dir,
    }
