"""DINOv3 multi-layer fusion and FPN fusion visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dtu import build_dtu_dataset
from experiments.fpn import ConvFPNVisualizationNet
from models.dinov3.extractor import extract_dinov3_native_features, load_dinov3_vit_base
from upr_mvs.config import DTUConfig, ProjectPaths


FPN_LEVELS = ("P2", "P3", "P4", "P5")


@dataclass(frozen=True)
class DinoFPNFusionConfig:
    """Configuration for the DINO/FPN fusion visualization experiment."""

    dino_layer_numbers: tuple[int, int, int] = (3, 7, 11)
    max_side: int = 768
    dino_input_max_side: int = 0
    fpn_channels: int = 16
    dino_fused_channels: int = 16
    fused_channels: int = 16
    patch_size: int = 16


def maybe_resize_image(image: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return image
    height, width = image.shape[-2:]
    scale = float(max_side) / float(max(height, width))
    if scale >= 1.0:
        return image
    target_hw = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    return F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)


def _layer_numbers_to_indices(layer_numbers: tuple[int, ...]) -> tuple[int, ...]:
    if not layer_numbers:
        raise ValueError("At least one DINO layer number is required")
    layer_indices = tuple(layer_number - 1 for layer_number in layer_numbers)
    if any(layer_index < 0 or layer_index >= 12 for layer_index in layer_indices):
        raise ValueError(f"DINO layer numbers must be in [1, 12], got {layer_numbers}")
    return layer_indices


def _init_dino_concat_compression(conv: nn.Conv2d, num_layers: int) -> None:
    if conv.in_channels % num_layers != 0:
        raise ValueError("DINO concat channels must be divisible by the number of layers")

    channels_per_layer = conv.in_channels // num_layers
    with torch.no_grad():
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()

        for out_channel in range(conv.out_channels):
            start = int(round(out_channel * channels_per_layer / conv.out_channels))
            end = int(round((out_channel + 1) * channels_per_layer / conv.out_channels))
            end = max(end, start + 1)
            weight = 1.0 / float(num_layers * (end - start))
            for layer_index in range(num_layers):
                layer_offset = layer_index * channels_per_layer
                conv.weight[out_channel, layer_offset + start : layer_offset + end, 0, 0] = weight


def _init_two_branch_compression(conv: nn.Conv2d, fpn_channels: int, dino_channels: int) -> None:
    if conv.in_channels != fpn_channels + dino_channels:
        raise ValueError("Fusion conv input channels do not match FPN + DINO channels")

    with torch.no_grad():
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()

        for out_channel in range(conv.out_channels):
            fpn_start = int(round(out_channel * fpn_channels / conv.out_channels))
            fpn_end = int(round((out_channel + 1) * fpn_channels / conv.out_channels))
            fpn_end = max(fpn_end, fpn_start + 1)

            dino_start = int(round(out_channel * dino_channels / conv.out_channels))
            dino_end = int(round((out_channel + 1) * dino_channels / conv.out_channels))
            dino_end = max(dino_end, dino_start + 1)

            conv.weight[out_channel, fpn_start:fpn_end, 0, 0] = 0.5 / float(fpn_end - fpn_start)
            dino_offset = fpn_channels
            conv.weight[out_channel, dino_offset + dino_start : dino_offset + dino_end, 0, 0] = (
                0.5 / float(dino_end - dino_start)
            )


class DinoConcatFusion(nn.Module):
    """Fuse selected DINO layers by concat followed by 1x1 compression."""

    def __init__(self, in_channels: int = 768, num_layers: int = 3, out_channels: int = 16):
        super().__init__()
        self.compress = nn.Conv2d(in_channels * num_layers, out_channels, kernel_size=1, bias=False)
        _init_dino_concat_compression(self.compress, num_layers=num_layers)

    def forward(self, layer_features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(layer_features) == 0:
            raise ValueError("layer_features cannot be empty")
        fused = self.compress(torch.cat(list(layer_features), dim=1))
        return F.normalize(fused, p=2, dim=1)


class FPNDinoPyramidFusion(nn.Module):
    """Fuse each FPN level with a resized DINO fused feature."""

    def __init__(
        self,
        levels: tuple[str, ...] = FPN_LEVELS,
        fpn_channels: int = 16,
        dino_channels: int = 16,
        out_channels: int = 16,
    ):
        super().__init__()
        self.levels = levels
        self.fusers = nn.ModuleDict(
            {
                level: nn.Conv2d(fpn_channels + dino_channels, out_channels, kernel_size=1, bias=False)
                for level in levels
            }
        )
        for fuser in self.fusers.values():
            _init_two_branch_compression(fuser, fpn_channels=fpn_channels, dino_channels=dino_channels)

    def forward(self, fpn_features: dict[str, torch.Tensor], dino_fused: torch.Tensor) -> dict[str, torch.Tensor]:
        fused_features: dict[str, torch.Tensor] = {}
        for level in self.levels:
            fpn_feature = F.normalize(fpn_features[level], p=2, dim=1)
            dino_resized = F.interpolate(dino_fused, size=fpn_feature.shape[-2:], mode="bilinear", align_corners=False)
            dino_resized = F.normalize(dino_resized, p=2, dim=1)
            fused_feature = self.fusers[level](torch.cat([fpn_feature, dino_resized], dim=1))
            fused_features[level] = F.normalize(fused_feature, p=2, dim=1)
        return fused_features


def feature_energy(feature: torch.Tensor) -> np.ndarray:
    return feature[0].detach().abs().mean(dim=0).cpu().numpy()


def normalize_for_display(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(array[finite], [1, 99])
    if high <= low:
        high = low + 1e-6
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def save_reference_image(image: torch.Tensor, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
    plt.imsave(output_path, np.clip(image_np, 0.0, 1.0))
    return output_path


def save_feature_strip(
    title: str,
    features: dict[str, torch.Tensor],
    output_path: Path,
    cmap: str = "magma",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4.5))
    if len(features) == 1:
        axes = [axes]

    for ax, (name, feature) in zip(axes, features.items()):
        energy = normalize_for_display(feature_energy(feature))
        ax.imshow(energy, cmap=cmap)
        ax.set_title(f"{name}\n{tuple(feature.shape)}")
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_fusion_overview(
    image: torch.Tensor,
    dino_layers: dict[str, torch.Tensor],
    dino_fused: torch.Tensor,
    fpn_features: dict[str, torch.Tensor],
    fused_features: dict[str, torch.Tensor],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))

    dino_row_features = {**dino_layers, "DINO concat+1x1": dino_fused}
    for column, (name, feature) in enumerate(dino_row_features.items()):
        axes[0, column].imshow(normalize_for_display(feature_energy(feature)), cmap="magma")
        axes[0, column].set_title(f"{name}\n{tuple(feature.shape)}")
        axes[0, column].axis("off")

    for column, level in enumerate(FPN_LEVELS):
        axes[1, column].imshow(normalize_for_display(feature_energy(fpn_features[level])), cmap="viridis")
        axes[1, column].set_title(f"FPN {level}\n{tuple(fpn_features[level].shape)}")
        axes[1, column].axis("off")

    for column, level in enumerate(FPN_LEVELS):
        axes[2, column].imshow(normalize_for_display(feature_energy(fused_features[level])), cmap="plasma")
        axes[2, column].set_title(f"FPN+DINO {level}\n{tuple(fused_features[level].shape)}")
        axes[2, column].axis("off")

    fig.suptitle(
        f"DINOv3 multi-layer fusion + four-level FPN fusion | reference image {tuple(image.shape)}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_dino_fpn_fusion_visualization(
    sample_index: int = 0,
    paths: ProjectPaths | None = None,
    dtu_config: DTUConfig | None = None,
    fusion_config: DinoFPNFusionConfig | None = None,
    output_root: str | Path = "outputs/dino_fpn_fusion",
    device: str | torch.device | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    dtu_config = dtu_config or DTUConfig()
    fusion_config = fusion_config or DinoFPNFusionConfig()
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dtu_dataset(paths=paths, config=dtu_config)
    sample = dataset[sample_index]
    image = sample["imgs"][0].unsqueeze(0).float() / 255.0
    image = maybe_resize_image(image, fusion_config.max_side).to(device)

    fpn_model = ConvFPNVisualizationNet(
        c2_channels=fusion_config.fpn_channels,
        c3_channels=fusion_config.fpn_channels,
        c4_channels=fusion_config.fpn_channels,
        c5_channels=fusion_config.fpn_channels,
        out_channels=fusion_config.fpn_channels,
    ).to(device).eval()
    dino_model = load_dinov3_vit_base(device=device, weights_file=paths.dinov3_weights_file, patch_size=fusion_config.patch_size)

    dino_layer_indices = _layer_numbers_to_indices(fusion_config.dino_layer_numbers)
    dino_input_max_side = fusion_config.dino_input_max_side
    if dino_input_max_side <= 0:
        dino_input_max_side = int(max(image.shape[-2:]))

    dino_sample = dict(sample)
    dino_sample["imgs"] = (image.detach().cpu()[0:1] * 255.0).float()

    with torch.inference_mode():
        fpn_features_all = fpn_model(image)
        fpn_features = {level: fpn_features_all[level] for level in FPN_LEVELS}

        dino_output = extract_dinov3_native_features(
            dino_sample,
            dino_model,
            device=device,
            max_side=dino_input_max_side,
            patch_size=fusion_config.patch_size,
            layers=dino_layer_indices,
        )
        dino_layer_features = [feature[0:1] for feature in dino_output["layer_features"]]

        dino_fuser = DinoConcatFusion(
            in_channels=dino_layer_features[0].shape[1],
            num_layers=len(dino_layer_features),
            out_channels=fusion_config.dino_fused_channels,
        ).to(device).eval()
        dino_fused = dino_fuser(dino_layer_features)

        pyramid_fuser = FPNDinoPyramidFusion(
            fpn_channels=fusion_config.fpn_channels,
            dino_channels=fusion_config.dino_fused_channels,
            out_channels=fusion_config.fused_channels,
        ).to(device).eval()
        fused_features = pyramid_fuser(fpn_features, dino_fused)

    output_dir = Path(output_root) / sample["sample_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    dino_layer_names = {
        f"DINO layer {layer_number}": feature.detach().cpu()
        for layer_number, feature in zip(fusion_config.dino_layer_numbers, dino_layer_features)
    }
    fpn_features_cpu = {level: feature.detach().cpu() for level, feature in fpn_features.items()}
    fused_features_cpu = {level: feature.detach().cpu() for level, feature in fused_features.items()}
    dino_fused_cpu = dino_fused.detach().cpu()
    image_cpu = image.detach().cpu()

    paths_out = {
        "reference": save_reference_image(image_cpu, output_dir / "reference_image.png"),
        "dinov3_layers_and_fused": save_feature_strip(
            "DINOv3 layer 3/7/11 before fusion and concat+1x1 result",
            {**dino_layer_names, "DINO fused": dino_fused_cpu},
            output_dir / "dinov3_layers_and_fused.png",
            cmap="magma",
        ),
        "fpn_pyramid": save_feature_strip(
            "FPN four-level pyramid P2-P5",
            fpn_features_cpu,
            output_dir / "fpn_pyramid.png",
            cmap="viridis",
        ),
        "fpn_dinov3_fused_pyramid": save_feature_strip(
            "FPN + resized DINO fused feature at each pyramid level",
            fused_features_cpu,
            output_dir / "fpn_dinov3_fused_pyramid.png",
            cmap="plasma",
        ),
        "overview": save_fusion_overview(
            image_cpu,
            dino_layer_names,
            dino_fused_cpu,
            fpn_features_cpu,
            fused_features_cpu,
            output_dir / "fusion_overview.png",
        ),
    }

    return {
        "sample_name": sample["sample_name"],
        "scan_name": sample["scan_name"],
        "ref_view": int(sample["ref_view"]),
        "device": str(device),
        "image_shape": tuple(image_cpu.shape),
        "dino_input_hw": dino_output["input_hw"],
        "dino_native_feature_hw": dino_output["native_feature_hw"],
        "dino_layer_numbers": fusion_config.dino_layer_numbers,
        "dino_layer_shapes": {name: tuple(feature.shape) for name, feature in dino_layer_names.items()},
        "dino_fused_shape": tuple(dino_fused_cpu.shape),
        "fpn_shapes": {level: tuple(feature.shape) for level, feature in fpn_features_cpu.items()},
        "fused_shapes": {level: tuple(feature.shape) for level, feature in fused_features_cpu.items()},
        "output_dir": output_dir,
        "paths": paths_out,
    }
