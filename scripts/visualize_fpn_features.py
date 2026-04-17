#!/usr/bin/env python
"""Visualize a Conv2d backbone plus lateral/smooth FPN on the first DTU sample."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dtu import build_dtu_dataset
from experiments.fpn import ConvFPNVisualizationNet
from upr_mvs.config import DTUConfig, ProjectPaths


CHANNEL_NAMES = [
    "blur-like",
    "sobel-x-like",
    "sobel-y-like",
    "laplacian-like",
    "diagonal-like",
    "identity-like",
    "blur-like repeat",
    "sobel-x-like repeat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fpn_conv_visualization"))
    parser.add_argument("--dtu-root", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--max-side", type=int, default=0, help="Optional resize before FPN. 0 keeps original resolution.")
    parser.add_argument("--channels", type=int, default=16, help="Channels for each C2-C5 backbone feature.")
    parser.add_argument("--out-channels", type=int, default=16, help="FPN output channels for P2-P5.")
    return parser.parse_args()


def normalize_for_display(array: np.ndarray, signed: bool = False) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    if signed:
        vmax = np.percentile(np.abs(array[finite]), 99)
        vmax = max(float(vmax), 1e-6)
        return np.clip(array / (2.0 * vmax) + 0.5, 0.0, 1.0)
    lo, hi = np.percentile(array[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def tensor_map(feature: torch.Tensor, channel: int) -> np.ndarray:
    return feature[0, channel].detach().cpu().numpy()


def feature_energy(feature: torch.Tensor) -> np.ndarray:
    return feature[0].detach().abs().mean(dim=0).cpu().numpy()


def save_reference_image(image: torch.Tensor, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0.0, 1.0)
    plt.imsave(output_path, image_np)
    return output_path


def save_overview(image: torch.Tensor, features: dict[str, torch.Tensor], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))

    image_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
    axes[0, 0].imshow(np.clip(image_np, 0.0, 1.0))
    axes[0, 0].set_title("reference image")
    axes[0, 0].axis("off")

    axes[1, 0].text(
        0.0,
        0.62,
        "p5 = lateral5(c5)\n"
        "p4 = lateral4(c4) + upsample(p5)\n"
        "p3 = lateral3(c3) + upsample(p4)\n"
        "p2 = lateral2(c2) + upsample(p3)\n\n"
        "then smooth2~5: 3x3 conv",
        fontsize=11,
        va="center",
    )
    axes[1, 0].axis("off")

    for col, level in enumerate(["C2", "C3", "C4", "C5"], start=1):
        energy = normalize_for_display(feature_energy(features[level]))
        axes[0, col].imshow(energy, cmap="magma")
        axes[0, col].set_title(f"{level} conv feature energy {tuple(features[level].shape[-2:])}")
        axes[0, col].axis("off")

    for col, level in enumerate(["P2", "P3", "P4", "P5"], start=1):
        energy = normalize_for_display(feature_energy(features[level]))
        axes[1, col].imshow(energy, cmap="magma")
        axes[1, col].set_title(f"{level} fused+smoothed energy {tuple(features[level].shape[-2:])}")
        axes[1, col].axis("off")

    fig.suptitle("Conv2d backbone + FPN: lateral 1x1, top-down nearest upsample, smooth 3x3", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_channel_grid(level: str, feature: torch.Tensor, output_path: Path, max_channels: int = 8) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    num_channels = min(max_channels, feature.shape[1])
    for channel in range(num_channels):
        name = CHANNEL_NAMES[channel] if channel < len(CHANNEL_NAMES) else f"channel {channel:02d}"
        image = normalize_for_display(tensor_map(feature, channel), signed=True)
        axes[channel].imshow(image, cmap="coolwarm")
        axes[channel].set_title(f"{level} ch{channel:02d} {name}")
        axes[channel].axis("off")

    energy = normalize_for_display(feature_energy(feature))
    axes[8].imshow(energy, cmap="viridis")
    axes[8].set_title(f"{level} mean |activation|")
    axes[8].axis("off")

    for ax in axes[num_channels:8]:
        ax.axis("off")

    fig.suptitle(f"{level} feature channels, shape={tuple(feature.shape)}", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def maybe_resize(image: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return image
    height, width = image.shape[-2:]
    scale = float(max_side) / float(max(height, width))
    if scale >= 1.0:
        return image
    target_hw = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    return F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    if args.dtu_root is not None:
        paths = replace(paths, dtu_train_root=args.dtu_root)
    if args.list_file is not None:
        paths = replace(paths, dtu_list_path=args.list_file)

    dataset = build_dtu_dataset(paths=paths, config=DTUConfig())
    sample = dataset[args.sample_index]
    image = sample["imgs"][0].unsqueeze(0).float() / 255.0
    image = maybe_resize(image, args.max_side)

    model = ConvFPNVisualizationNet(
        c2_channels=args.channels,
        c3_channels=args.channels,
        c4_channels=args.channels,
        c5_channels=args.channels,
        out_channels=args.out_channels,
    ).eval()
    with torch.no_grad():
        features = model(image)

    output_dir = args.output_root / sample["sample_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = save_reference_image(image, output_dir / "reference_image.png")
    overview_path = save_overview(image, features, output_dir / "fpn_overview.png")

    channel_paths = {}
    for level, feature in features.items():
        channel_paths[level] = save_channel_grid(level, feature, output_dir / f"{level.lower()}_channels.png")

    print("sample_name:", sample["sample_name"])
    print("scan_name:", sample["scan_name"])
    print("ref_view:", int(sample["ref_view"]))
    print("image shape:", tuple(image.shape))
    print("model parameter count:", sum(p.numel() for p in model.parameters()))
    for level, feature in features.items():
        print(f"{level} shape:", tuple(feature.shape))
    print("reference image:", ref_path)
    print("overview:", overview_path)
    print("channel grids:")
    for level, path in channel_paths.items():
        print(f" - {level}: {path}")


if __name__ == "__main__":
    main()
