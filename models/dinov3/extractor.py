"""DINOv3 feature extraction and projection helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from base.config import DINOConfig, ProjectPaths

from .vision_transformer import vit_base


def load_dinov3_vit_base(
    device: torch.device | str,
    weights_file: str | Path | None = None,
    patch_size: int = 16,
) -> torch.nn.Module:
    paths = ProjectPaths()
    weights_file = Path(weights_file) if weights_file is not None else paths.dinov3_weights_file

    if not weights_file.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_file}")

    model = vit_base(patch_size=patch_size, n_storage_tokens=0)
    checkpoint = torch.load(str(weights_file), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    load_result = model.load_state_dict(state_dict, strict=False)
    print("DINOv3 load missing/unexpected:", len(load_result.missing_keys), len(load_result.unexpected_keys))
    return model.to(device).eval()


def compute_patch_aligned_size(image_h: int, image_w: int, max_side: int, patch_size: int) -> tuple[int, int]:
    scale = float(max_side) / float(max(image_h, image_w))
    target_h = max(patch_size, int(round(image_h * scale / patch_size)) * patch_size)
    target_w = max(patch_size, int(round(image_w * scale / patch_size)) * patch_size)
    return target_h, target_w


def dino_normalization_tensors(device: torch.device | str, config: DINOConfig | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    config = config or DINOConfig()
    mean = torch.tensor(config.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(config.std, device=device).view(1, 3, 1, 1)
    return mean, std


def extract_dinov3_native_features(
    sample: dict,
    model: torch.nn.Module,
    device: torch.device | str,
    max_side: int = 384,
    patch_size: int = 16,
    layers: tuple[int, ...] = tuple(range(12)),
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> dict:
    imgs = sample["imgs"].to(device=device, dtype=torch.float32) / 255.0
    _, _, image_h, image_w = imgs.shape
    input_h, input_w = compute_patch_aligned_size(image_h, image_w, max_side=max_side, patch_size=patch_size)

    dino_input = F.interpolate(imgs, size=(input_h, input_w), mode="bilinear", align_corners=False)
    if mean is None or std is None:
        mean, std = dino_normalization_tensors(device)
    dino_input = (dino_input - mean) / std

    with torch.inference_mode():
        layer_features = model.get_intermediate_layers(
            dino_input,
            n=layers,
            reshape=True,
            norm=True,
        )

    layer_features = [F.normalize(features.float(), p=2, dim=1) for features in layer_features]
    native_feature_hw = tuple(layer_features[0].shape[-2:])
    return {
        "input_hw": (input_h, input_w),
        "native_feature_hw": native_feature_hw,
        "layers": layers,
        "layer_features": layer_features,
    }


_DINO_PROJECTION_CACHE: dict[tuple, torch.Tensor] = {}


def get_random_projection_matrix(
    in_channels: int,
    out_channels: int | None,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    seed: int = 20260416,
) -> torch.Tensor | None:
    if out_channels is None or out_channels >= in_channels:
        return None
    key = (in_channels, out_channels, str(device), str(dtype), seed)
    if key not in _DINO_PROJECTION_CACHE:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(in_channels, out_channels, generator=generator, dtype=torch.float32)
        projection = F.normalize(projection, p=2, dim=0)
        _DINO_PROJECTION_CACHE[key] = projection.to(device=device, dtype=dtype)
    return _DINO_PROJECTION_CACHE[key]


def project_and_resize_dino_layer(
    layer_features: torch.Tensor,
    target_feature_hw: tuple[int, int],
    out_channels: int | None,
    device: torch.device | str,
    seed: int = 20260416,
) -> torch.Tensor:
    features = layer_features.unsqueeze(0).to(device=device, dtype=torch.float32)
    batch_size, num_views, channels, native_h, native_w = features.shape
    projection = get_random_projection_matrix(channels, out_channels, device=device, dtype=features.dtype, seed=seed)
    if projection is not None:
        features = torch.einsum("bvchw,ck->bvkhw", features, projection)
    features = F.normalize(features, p=2, dim=2)

    projected_channels = features.shape[2]
    target_h, target_w = target_feature_hw
    features = F.interpolate(
        features.view(batch_size * num_views, projected_channels, native_h, native_w),
        size=target_feature_hw,
        mode="bilinear",
        align_corners=False,
    ).view(batch_size, num_views, projected_channels, target_h, target_w)
    return F.normalize(features.contiguous(), p=2, dim=2)
