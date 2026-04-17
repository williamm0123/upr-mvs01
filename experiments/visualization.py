"""Visualization helpers for cost-volume experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def image_tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image, 0, 255).astype(np.uint8)


def save_depth_result_image(feature_name: str, row: dict, maps: dict, depth_gt_target: torch.Tensor, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    soft_depth = maps["soft_depth"][0].numpy()
    argmin_depth = maps["argmin_depth"][0].numpy()
    confidence = maps["confidence"][0].numpy()
    abs_error = maps["abs_error"][0].numpy()
    valid_mask = maps["valid_mask"][0].numpy()
    gt_depth = depth_gt_target[0].detach().cpu().numpy()

    finite_error = abs_error[np.isfinite(abs_error)]
    error_vmax = float(np.nanpercentile(finite_error, 95)) if finite_error.size else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    im = axes[0].imshow(soft_depth, cmap="turbo")
    axes[0].set_title("soft depth")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(argmin_depth, cmap="turbo")
    axes[1].set_title("argmin depth")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    im = axes[2].imshow(gt_depth, cmap="turbo")
    axes[2].set_title("GT depth")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    im = axes[3].imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[3].set_title("confidence")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    im = axes[4].imshow(abs_error, cmap="inferno", vmin=0.0, vmax=error_vmax)
    axes[4].set_title("soft abs error")
    axes[4].axis("off")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    axes[5].imshow(valid_mask, cmap="gray")
    axes[5].set_title("valid eval mask")
    axes[5].axis("off")

    fig.suptitle(
        f"{feature_name} | soft med={row['soft_median']:.2f}mm, "
        f"soft mean={row['soft_mean']:.2f}mm, conf med={row['confidence_median']:.3f}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_metrics_summary_plots(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dino_df = metrics_df[metrics_df["feature"].str.startswith("DINOv3")].copy()
    rgb_row = metrics_df[metrics_df["feature"].str.startswith("RGB")].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(dino_df["layer"], dino_df["argmin_median"], marker="o", label="argmin median")
    axes[0].plot(dino_df["layer"], dino_df["soft_median"], marker="o", label="soft median")
    axes[0].axhline(float(rgb_row["soft_median"]), color="gray", linestyle="--", label="RGB soft median")
    axes[0].set_xlabel("DINOv3 layer")
    axes[0].set_ylabel("depth error median (mm)")
    axes[0].set_title("Median depth error")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(dino_df["layer"], dino_df["soft_mean"], marker="o", label="soft mean")
    axes[1].plot(dino_df["layer"], dino_df["soft_p90"], marker="o", label="soft p90")
    axes[1].axhline(float(rgb_row["soft_mean"]), color="gray", linestyle="--", label="RGB soft mean")
    axes[1].set_xlabel("DINOv3 layer")
    axes[1].set_ylabel("depth error (mm)")
    axes[1].set_title("Mean / P90 soft depth error")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(dino_df["layer"], dino_df["confidence_median"], marker="o", label="confidence median")
    axes[2].plot(dino_df["layer"], dino_df["valid_ratio"], marker="o", label="valid ratio")
    axes[2].set_xlabel("DINOv3 layer")
    axes[2].set_ylabel("ratio / probability")
    axes[2].set_title("Confidence and valid ratio")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_adapter_training_summary(history_df: pd.DataFrame, comparison_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history_df["step"], history_df["loss"], marker="o")
    axes[0].set_title("Adapter training loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(True)

    axes[1].plot(history_df["step"], history_df["median_abs_error"], marker="o", label="median")
    axes[1].plot(history_df["step"], history_df["mean_abs_error"], marker="o", label="mean")
    axes[1].set_title("Train-grid depth error")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("depth error (mm)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].bar(comparison_df["feature"], comparison_df["soft_median"])
    axes[2].set_title("Eval-grid soft median error")
    axes[2].set_ylabel("depth error (mm)")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_adapter_ablation_summary(history_df: pd.DataFrame, metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for feature_name, group in history_df.groupby("feature"):
        axes[0].plot(group["step"], group["loss"], marker="o", label=feature_name)
        axes[1].plot(group["step"], group["median_abs_error"], marker="o", label=feature_name)

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Train-grid median error")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("depth error (mm)")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    axes[2].bar(metrics_df["feature"], metrics_df["soft_median"])
    axes[2].set_title("Eval-grid soft median error")
    axes[2].set_ylabel("depth error (mm)")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
