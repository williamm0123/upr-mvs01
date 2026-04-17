"""Geometry adapter module and single-sample training test."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cost_volume import (
    build_variance_cost_volume_chunked,
    compute_all_views_valid_for_grid,
    depth_regression_from_cost,
)
from .geometry import make_projection_for_feature_grid, resize_gt_to_feature_grid
from models.dinov3.extractor import get_random_projection_matrix
from upr_mvs.config import AdapterConfig
from upr_mvs.external import sample_depth_planes


class Conv1x1Adapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


class Conv1x1Conv3x3Adapter(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 128, out_ch: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(8, hidden_ch)
        self.conv = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.act(self.norm1(x))
        x = self.conv(x)
        return F.normalize(x, p=2, dim=1)


class GeometryAdapter(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 128, out_ch: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(8, hidden_ch)
        self.conv1 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(8, hidden_ch)
        self.out = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.act(self.norm1(x))

        residual = x
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        x = self.norm3(x)
        x = self.act(x + residual)

        x = self.out(x)
        return F.normalize(x, p=2, dim=1)


def make_adapter_native_input(dinov3_native: dict, layer_ids: tuple[int, ...]) -> torch.Tensor:
    native_layers = dinov3_native.get("layers")
    if native_layers is None:
        selected = [dinov3_native["layer_features"][layer_id - 1] for layer_id in layer_ids]
    else:
        by_one_based_layer = {
            int(layer_index) + 1: features
            for layer_index, features in zip(native_layers, dinov3_native["layer_features"])
        }
        missing = [layer_id for layer_id in layer_ids if layer_id not in by_one_based_layer]
        if missing:
            raise ValueError(f"Adapter layers were not extracted by DINOv3: {missing}")
        selected = [by_one_based_layer[layer_id] for layer_id in layer_ids]
    return torch.cat(selected, dim=1).detach()


def adapter_features_for_grid(adapter: nn.Module, native_input: torch.Tensor, target_feature_hw: tuple[int, int]) -> torch.Tensor:
    num_views = native_input.shape[0]
    adapted_native = adapter(native_input)
    out_ch = adapted_native.shape[1]
    adapted_grid = F.interpolate(
        adapted_native,
        size=target_feature_hw,
        mode="bilinear",
        align_corners=False,
    ).view(1, num_views, out_ch, target_feature_hw[0], target_feature_hw[1])
    return F.normalize(adapted_grid.contiguous(), p=2, dim=2)


def raw_selected_dino_features_for_grid(
    native_input: torch.Tensor,
    target_feature_hw: tuple[int, int],
    out_channels: int,
    device: torch.device | str,
    seed: int = 20260416,
) -> torch.Tensor:
    num_views, channels, native_h, native_w = native_input.shape
    features = native_input.to(device=device, dtype=torch.float32)
    projection = get_random_projection_matrix(channels, out_channels, device=device, dtype=features.dtype, seed=seed)
    if projection is not None:
        features = torch.einsum("vchw,ck->vkhw", features, projection)
    features = F.normalize(features, p=2, dim=1)
    features = F.interpolate(
        features,
        size=target_feature_hw,
        mode="bilinear",
        align_corners=False,
    ).view(1, num_views, -1, target_feature_hw[0], target_feature_hw[1])
    return F.normalize(features.contiguous(), p=2, dim=2)


def make_depth_training_targets(
    sample: dict,
    feature_hw: tuple[int, int],
    device: torch.device | str,
    num_depths: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    projection_matrices = make_projection_for_feature_grid(sample, feature_hw, device)
    depth_range = sample["depth_range"].view(1, 2).to(device=device, dtype=torch.float32)
    depth_values = sample_depth_planes(depth_range, num_depths)
    depth_gt, mask = resize_gt_to_feature_grid(sample, feature_hw, device)
    all_views_valid = compute_all_views_valid_for_grid(
        projection_matrices=projection_matrices,
        depth_values=depth_values,
        feature_hw=feature_hw,
        device=device,
        dtype=torch.float32,
    )
    return projection_matrices, depth_values, depth_gt, mask, all_views_valid


def supervised_cost_volume_loss(
    cost_volume: torch.Tensor,
    all_views_valid: torch.Tensor,
    depth_values: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    ce_weight: float = 1.0,
    l1_weight: float = 0.25,
) -> tuple[torch.Tensor, dict]:
    finite_cost = torch.isfinite(cost_volume)
    candidate_mask = all_views_valid & finite_cost
    has_candidate = candidate_mask.any(dim=1)

    depth_axis = depth_values.view(depth_values.shape[0], depth_values.shape[1], 1, 1)
    target_depth_idx = torch.argmin(torch.abs(depth_axis - depth_gt.unsqueeze(1)), dim=1)
    in_depth_range = (depth_gt >= depth_values[:, :1, None]) & (depth_gt <= depth_values[:, -1:, None])
    valid_pixels = mask & has_candidate & in_depth_range.squeeze(1)

    if not valid_pixels.any():
        raise RuntimeError("Adapter training has no valid pixels")

    masked_cost = cost_volume.masked_fill(~candidate_mask, float("inf"))
    min_cost = masked_cost.amin(dim=1, keepdim=True)
    stable_cost = masked_cost - min_cost
    stable_cost = torch.where(torch.isfinite(stable_cost), stable_cost, torch.zeros_like(stable_cost))
    logits = -stable_cost / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~candidate_mask, -1.0e4)

    logits_valid = logits.permute(0, 2, 3, 1)[valid_pixels]
    target_valid = target_depth_idx[valid_pixels]
    ce_loss = F.cross_entropy(logits_valid, target_valid)

    soft_outputs = depth_regression_from_cost(
        cost_volume=cost_volume,
        depth_values=depth_values,
        candidate_mask=all_views_valid,
        temperature=temperature,
    )
    soft_depth = soft_outputs["depth"]
    depth_span = (depth_values[:, -1] - depth_values[:, 0]).view(-1, 1, 1).clamp_min(1e-6)
    l1_loss = (torch.abs(soft_depth - depth_gt) / depth_span)[valid_pixels].mean()

    loss = ce_weight * ce_loss + l1_weight * l1_loss
    with torch.no_grad():
        abs_error = torch.abs(soft_depth - depth_gt)[valid_pixels]
        confidence = soft_outputs["confidence"][soft_outputs["has_candidate"]]
        metrics = {
            "loss": float(loss.detach()),
            "ce": float(ce_loss.detach()),
            "l1_norm": float(l1_loss.detach()),
            "mean_abs_error": float(abs_error.mean()),
            "median_abs_error": float(abs_error.median()),
            "confidence_median": float(confidence.median()) if confidence.numel() > 0 else float("nan"),
        }
    return loss, metrics


def train_geometry_adapter(
    adapter: nn.Module,
    native_input: torch.Tensor,
    sample: dict,
    train_feature_hw: tuple[int, int],
    num_depths: int,
    temperature: float,
    config: AdapterConfig,
    device: torch.device | str,
    channel_chunk: int = 4,
) -> pd.DataFrame:
    projection_train, depth_values_train, depth_gt_train, mask_train, all_views_valid_train = make_depth_training_targets(
        sample=sample,
        feature_hw=train_feature_hw,
        device=device,
        num_depths=num_depths,
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = []
    adapter.train()

    for step in range(1, config.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        features_train = adapter_features_for_grid(adapter, native_input, train_feature_hw)
        cost_outputs = build_variance_cost_volume_chunked(
            features=features_train,
            projection_matrices=projection_train,
            depth_values=depth_values_train,
            channel_chunk=channel_chunk,
            all_views_valid=all_views_valid_train,
        )
        loss, metrics = supervised_cost_volume_loss(
            cost_volume=cost_outputs["cost_volume"],
            all_views_valid=cost_outputs["all_views_valid"],
            depth_values=depth_values_train,
            depth_gt=depth_gt_train,
            mask=mask_train,
            temperature=temperature,
            ce_weight=config.ce_weight,
            l1_weight=config.l1_weight,
        )
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip)
        optimizer.step()

        metrics["step"] = step
        history.append(metrics)
        if step == 1 or step == config.train_steps or step % max(1, config.train_steps // 5) == 0:
            print(
                f"step {step:03d}/{config.train_steps}: "
                f"loss={metrics['loss']:.4f}, "
                f"med_err={metrics['median_abs_error']:.3f}mm, "
                f"mean_err={metrics['mean_abs_error']:.3f}mm, "
                f"conf_med={metrics['confidence_median']:.4f}"
            )

        del features_train, cost_outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(history)
