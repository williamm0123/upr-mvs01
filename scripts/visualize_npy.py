#!/usr/bin/env python
"""Visualize one .npy file or a directory of .npy files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input .npy file or a directory containing .npy files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path for one file, or output directory for a directory input.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "raw", "inverse-depth", "rgb"),
        default="auto",
        help="Visualization mode. auto uses RGB for 3-channel arrays and inverse-depth for 2D arrays.",
    )
    parser.add_argument("--cmap", default="Spectral", help="Matplotlib colormap for scalar arrays.")
    parser.add_argument("--low", type=float, default=2.0, help="Low percentile for scalar normalization.")
    parser.add_argument("--high", type=float, default=98.0, help="High percentile for scalar normalization.")
    parser.add_argument("--recursive", action="store_true", help="Recursively find .npy files under directory input.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def normalize_scalar(array: np.ndarray, low: float, high: float) -> np.ndarray:
    valid = np.isfinite(array)
    if not valid.any():
        return np.zeros(array.shape, dtype=np.float32)

    values = array[valid]
    vmin = np.percentile(values, low)
    vmax = np.percentile(values, high)
    if np.isclose(vmin, vmax):
        vmin = float(values.min())
        vmax = float(values.max())
    if np.isclose(vmin, vmax):
        return np.zeros(array.shape, dtype=np.float32)

    normalized = (array - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~valid] = 0.0
    return normalized.astype(np.float32)


def to_rgb_array(array: np.ndarray, low: float, high: float) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"RGB mode expects HxWx3/4 or 3/4xHxW array, got shape {array.shape}")

    array = array[..., :3].astype(np.float32)
    if array.max(initial=0.0) > 1.5:
        array = normalize_scalar(array, low, high)
    else:
        array = np.clip(array, 0.0, 1.0)
    return array


def scalar_from_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0].astype(np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0].astype(np.float32)
    if array.ndim == 3 and array.shape[0] <= 4:
        return np.abs(array.astype(np.float32)).mean(axis=0)
    if array.ndim == 3 and array.shape[-1] <= 4:
        return np.abs(array.astype(np.float32)).mean(axis=-1)
    raise ValueError(f"Cannot convert array with shape {array.shape} to a scalar image.")


def visualize_array(array: np.ndarray, mode: str, cmap: str, low: float, high: float) -> tuple[np.ndarray, str]:
    array = np.asarray(array)
    if mode == "auto":
        if array.ndim == 3 and (array.shape[-1] in (3, 4) or array.shape[0] in (3, 4)):
            mode = "rgb"
        else:
            mode = "inverse-depth"

    if mode == "rgb":
        return to_rgb_array(array, low, high), "rgb"

    scalar = scalar_from_array(array)
    if mode == "inverse-depth":
        valid = np.isfinite(scalar) & (scalar > 0)
        scalar_for_vis = np.zeros_like(scalar, dtype=np.float32)
        scalar_for_vis[valid] = 1.0 / scalar[valid]
        title_mode = "inverse-depth"
    elif mode == "raw":
        scalar_for_vis = scalar
        title_mode = "raw"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    normalized = normalize_scalar(scalar_for_vis, low, high)
    rgb = matplotlib.colormaps[cmap](normalized)[..., :3]
    return rgb, title_mode


def default_output_path(input_path: Path, output: Path | None, root_input: Path | None = None) -> Path:
    if output is None:
        return input_path.with_name(f"{input_path.stem}_vis.png")
    if output.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return output
    if root_input is not None:
        relative = input_path.relative_to(root_input)
        return output / relative.with_name(f"{relative.stem}_vis.png")
    return output / f"{input_path.stem}_vis.png"


def save_npy_visualization(input_path: Path, output_path: Path, args: argparse.Namespace) -> Path:
    array = np.load(input_path)
    image, title_mode = visualize_array(array, args.mode, args.cmap, args.low, args.high)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(image)
    ax.set_title(f"{input_path.name} | shape={array.shape} | mode={title_mode}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    return output_path


def iter_input_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            raise ValueError(f"Input file must be .npy: {input_path}")
        return [input_path]
    if input_path.is_dir():
        pattern = "**/*.npy" if recursive else "*.npy"
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    args = parse_args()
    input_path = args.input
    files = iter_input_files(input_path, args.recursive)
    if not files:
        raise FileNotFoundError(f"No .npy files found under: {input_path}")

    for npy_path in files:
        root_input = input_path if input_path.is_dir() else None
        output_path = default_output_path(npy_path, args.output, root_input=root_input)
        saved = save_npy_visualization(npy_path, output_path, args)
        print(saved)


if __name__ == "__main__":
    main()
