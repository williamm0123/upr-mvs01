"""Compare DA3 mono/metric depth predictions with DTU ground-truth depth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from data.dtu import expected_depth_path, expected_image_path
from data.io import read_pfm
from experiments.da3_scan_pointcloud import collect_scan_images, parse_rectified_image_name
from experiments.depth_anything_v3 import DA3VisualizationConfig, load_da3_mono_model, predict_da3_depth, visualize_depth
from upr_mvs.config import DEFAULT_DA3_MONO_MODEL_DIR, ProjectPaths


@dataclass(frozen=True)
class DA3DepthGTCompareConfig:
    """Configuration for comparing one DTU view against DA3 depth outputs."""

    scan_name: str = "scan1"
    view_id: int = 0
    light_id: int = 0
    split: str = "train"
    image_dir: str = "Rectified_raw"
    depth_dir: str = "Depths_raw"
    mono_model_dir: Path = DEFAULT_DA3_MONO_MODEL_DIR
    metric_depth_root: Path = Path("outputs/da3metric_first_scan_depths")
    mono_output_root: Path = Path("outputs/da3mono_single_depths")
    comparison_output_root: Path = Path("outputs/da3_depth_gt_comparison")
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    force_mono: bool = False


def resize_depth_nearest(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize depth-like data without interpolating invalid/background values."""

    target_h, target_w = target_hw
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    return np.asarray(image.resize((target_w, target_h), resample=Image.Resampling.NEAREST), dtype=np.float32)


def minmax_normalize(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max normalize on valid pixels and keep invalid pixels as NaN."""

    valid_values = values[mask].astype(np.float32)
    vmin = float(np.min(valid_values))
    vmax = float(np.max(valid_values))
    denom = vmax - vmin
    normalized = np.full(values.shape, np.nan, dtype=np.float32)
    if denom <= 1e-12:
        normalized[mask] = 0.0
    else:
        normalized[mask] = np.clip((values[mask] - vmin) / denom, 0.0, 1.0)
    return normalized, vmin, vmax


def summarize_values(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_max": float(np.max(values)),
    }


def summarize_depth(prefix: str, depth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = depth[mask].astype(np.float32)
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def metric_depth_candidates(root: Path, scan_name: str, image_stem: str) -> list[Path]:
    filename = f"{image_stem}_depth.npy"
    return [
        root / "depths" / filename,
        root / scan_name / "depths" / filename,
        root / filename,
    ]


def find_metric_depth(root: str | Path, scan_name: str, image_stem: str) -> Path:
    root = Path(root)
    for candidate in metric_depth_candidates(root, scan_name, image_stem):
        if candidate.is_file():
            return candidate
    recursive = sorted(root.glob(f"**/{image_stem}_depth.npy"))
    if recursive:
        return recursive[0]
    tried = "\n".join(str(path) for path in metric_depth_candidates(root, scan_name, image_stem))
    raise FileNotFoundError(f"Cannot find metric depth npy. Tried:\n{tried}")


def generate_or_load_mono_depth(
    image_path: str | Path,
    config: DA3DepthGTCompareConfig,
    device: str | torch.device | None,
) -> tuple[np.ndarray, np.ndarray, Path, Path, dict]:
    image_path = Path(image_path)
    depth_dir = Path(config.mono_output_root) / config.scan_name / "depths"
    preview_dir = Path(config.mono_output_root) / config.scan_name / "depth_preview"
    rgb_dir = Path(config.mono_output_root) / config.scan_name / "processed_rgb"
    depth_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    depth_path = depth_dir / f"{image_path.stem}_depth.npy"
    preview_path = preview_dir / f"{image_path.stem}_depth.png"
    rgb_path = rgb_dir / f"{image_path.stem}_processed_rgb.png"

    if depth_path.is_file() and rgb_path.is_file() and not config.force_mono:
        depth = np.load(depth_path).astype(np.float32)
        rgb = imageio.imread(rgb_path)
        return rgb, depth, depth_path, preview_path, {"loaded_from_cache": True}

    model_config = DA3VisualizationConfig(
        model_dir=config.mono_model_dir,
        process_res=config.process_res,
        process_res_method=config.process_res_method,
        view_id=config.view_id,
        light_id=config.light_id,
        image_dir=config.image_dir,
        split=config.split,
    )
    model, load_info = load_da3_mono_model(model_config.model_dir, device=device)
    rgb, depth, _ = predict_da3_depth(
        model,
        image_path,
        process_res=model_config.process_res,
        process_res_method=model_config.process_res_method,
    )
    np.save(depth_path, depth.astype(np.float32))
    imageio.imwrite(preview_path, visualize_depth(depth, cmap="Spectral"))
    imageio.imwrite(rgb_path, rgb)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rgb, depth, depth_path, preview_path, load_info


def mono_depth_path_for_image(output_root: str | Path, scan_name: str, image_stem: str) -> Path:
    return Path(output_root) / scan_name / "depths" / f"{image_stem}_depth.npy"


def generate_scan_mono_depths(
    paths: ProjectPaths | None = None,
    config: DA3DepthGTCompareConfig | None = None,
    include_max_images: bool = True,
    light_id: int | None = None,
    device: str | torch.device | None = None,
) -> dict:
    """Generate DA3 mono depth npy files for a full DTU scan."""

    paths = paths or ProjectPaths()
    config = config or DA3DepthGTCompareConfig()
    image_paths = collect_scan_images(
        paths.dtu_train_root,
        config.scan_name,
        config.split,
        config.image_dir,
        include_max_images=include_max_images,
        light_id=light_id,
    )

    depth_dir = Path(config.mono_output_root) / config.scan_name / "depths"
    preview_dir = Path(config.mono_output_root) / config.scan_name / "depth_preview"
    depth_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    model, load_info = load_da3_mono_model(config.mono_model_dir, device=device)
    rows = []
    for index, image_path in enumerate(image_paths):
        depth_path = depth_dir / f"{image_path.stem}_depth.npy"
        preview_path = preview_dir / f"{image_path.stem}_depth.png"
        view_id, image_light = parse_rectified_image_name(image_path)
        if depth_path.is_file() and not config.force_mono:
            depth = np.load(depth_path).astype(np.float32)
            generated = False
        else:
            _, depth, _ = predict_da3_depth(
                model,
                image_path,
                process_res=config.process_res,
                process_res_method=config.process_res_method,
            )
            np.save(depth_path, depth.astype(np.float32))
            imageio.imwrite(preview_path, visualize_depth(depth, cmap="Spectral"))
            generated = True
        valid = np.isfinite(depth) & (depth > 0)
        values = depth[valid]
        rows.append(
            {
                "image_index": index,
                "scan_name": config.scan_name,
                "image_name": image_path.name,
                "view_id": view_id,
                "light": image_light,
                "generated": generated,
                "depth_shape": str(tuple(depth.shape)),
                "depth_path": str(depth_path),
                "depth_vis_path": str(preview_path),
                "depth_min": float(np.min(values)) if values.size else np.nan,
                "depth_median": float(np.median(values)) if values.size else np.nan,
                "depth_mean": float(np.mean(values)) if values.size else np.nan,
                "depth_max": float(np.max(values)) if values.size else np.nan,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(rows)
    summary_csv_path = Path(config.mono_output_root) / config.scan_name / f"{config.scan_name}_da3mono_depth_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    return {
        "summary_df": summary_df,
        "summary_csv_path": summary_csv_path,
        "output_root": Path(config.mono_output_root) / config.scan_name,
        "load_info": load_info,
    }


def compare_depth_pair(
    name: str,
    pred: np.ndarray,
    gt_resized: np.ndarray,
    compute_abs_error: bool,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(gt_resized) & (gt_resized > 0) & np.isfinite(pred) & (pred > 0)
    if valid_mask.sum() == 0:
        raise ValueError(f"No valid pixels for {name} comparison.")

    gt_norm, gt_min, gt_max = minmax_normalize(gt_resized, valid_mask)
    pred_norm, pred_min, pred_max = minmax_normalize(pred, valid_mask)
    norm_abs_diff = np.abs(pred_norm - gt_norm)
    norm_abs_diff[~valid_mask] = np.nan

    row = {
        "name": name,
        "valid_pixels": int(valid_mask.sum()),
        "valid_ratio": float(valid_mask.mean()),
        "gt_norm_min": gt_min,
        "gt_norm_max": gt_max,
        "pred_norm_min": pred_min,
        "pred_norm_max": pred_max,
        **summarize_depth("gt", gt_resized, valid_mask),
        **summarize_depth("pred", pred, valid_mask),
        **summarize_values("norm_abs_diff", norm_abs_diff[valid_mask]),
    }
    if compute_abs_error:
        abs_error = np.abs(pred - gt_resized).astype(np.float32)
        abs_error[~valid_mask] = np.nan
        row.update(summarize_values("abs_error", abs_error[valid_mask]))
    else:
        abs_error = np.full(pred.shape, np.nan, dtype=np.float32)
    return row, valid_mask, abs_error, norm_abs_diff


def metric_depth_files(root: str | Path, scan_name: str) -> list[Path]:
    root = Path(root)
    candidate_dirs = [root / "depths", root / scan_name / "depths", root]
    for candidate_dir in candidate_dirs:
        if candidate_dir.is_dir():
            files = sorted(candidate_dir.glob("rect_*_depth.npy"))
            if files:
                return files
    files = sorted(root.glob("**/rect_*_depth.npy"))
    if files:
        return files
    raise FileNotFoundError(f"No metric depth npy files found under: {root}")


def save_scan_metrics_overview(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_df = metrics_df[metrics_df["name"] == "metric"].copy()
    mono_df = metrics_df[metrics_df["name"] == "mono"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(metric_df["abs_error_mean"].dropna(), bins=40, color="#c43c39", alpha=0.85)
    axes[0, 0].set_title("metric vs GT: abs error mean")
    axes[0, 0].set_xlabel("depth units")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(metric_df["norm_abs_diff_mean"].dropna(), bins=40, alpha=0.75, label="metric")
    axes[0, 1].hist(mono_df["norm_abs_diff_mean"].dropna(), bins=40, alpha=0.75, label="mono")
    axes[0, 1].set_title("normalized abs diff mean")
    axes[0, 1].set_xlabel("[0, 1] depth diff")
    axes[0, 1].legend()

    axes[1, 0].scatter(metric_df["gt_median"], metric_df["pred_median"], s=10, alpha=0.55, label="metric")
    axes[1, 0].set_title("metric median depth vs GT median")
    axes[1, 0].set_xlabel("GT median")
    axes[1, 0].set_ylabel("metric median")
    axes[1, 0].grid(True, linewidth=0.3, alpha=0.35)

    axes[1, 1].scatter(mono_df["gt_median"], mono_df["pred_median"], s=10, alpha=0.55, label="mono", color="#3572b0")
    axes[1, 1].set_title("mono raw median depth vs GT median")
    axes[1, 1].set_xlabel("GT median")
    axes[1, 1].set_ylabel("mono raw median")
    axes[1, 1].grid(True, linewidth=0.3, alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def summarize_scan_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "valid_ratio",
        "gt_median",
        "pred_median",
        "norm_abs_diff_mean",
        "norm_abs_diff_median",
        "norm_abs_diff_rmse",
        "norm_abs_diff_p90",
        "abs_error_mean",
        "abs_error_median",
        "abs_error_rmse",
        "abs_error_p90",
    ]
    rows = []
    for name, group in metrics_df.groupby("name"):
        row = {"name": name, "num_images": int(len(group))}
        for column in metric_columns:
            if column in group.columns:
                values = group[column].dropna().to_numpy(dtype=np.float64)
                row[f"{column}_avg"] = float(np.mean(values)) if values.size else np.nan
                row[f"{column}_median"] = float(np.median(values)) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_da3_scan_depth_gt_comparison(
    paths: ProjectPaths | None = None,
    config: DA3DepthGTCompareConfig | None = None,
) -> dict:
    """Compare all matched metric/mono depth npy files in one scan against GT."""

    paths = paths or ProjectPaths()
    config = config or DA3DepthGTCompareConfig()
    metric_files = metric_depth_files(config.metric_depth_root, config.scan_name)
    output_dir = Path(config.comparison_output_root) / f"{config.scan_name}_all_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric_depth_path in metric_files:
        image_stem = metric_depth_path.name[: -len("_depth.npy")]
        view_id, image_light = parse_rectified_image_name(f"{image_stem}.png")
        mono_depth_path = mono_depth_path_for_image(config.mono_output_root, config.scan_name, image_stem)
        if not mono_depth_path.is_file():
            raise FileNotFoundError(f"Missing mono depth for {image_stem}: {mono_depth_path}")

        depth_path = expected_depth_path(
            paths.dtu_train_root,
            config.scan_name,
            config.split,
            config.depth_dir,
            view_id,
        )
        gt = read_pfm(str(depth_path)).astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]

        metric_depth = np.load(metric_depth_path).astype(np.float32)
        mono_depth = np.load(mono_depth_path).astype(np.float32)
        if metric_depth.shape != mono_depth.shape:
            raise ValueError(f"Shape mismatch for {image_stem}: metric={metric_depth.shape}, mono={mono_depth.shape}")
        gt_resized = resize_depth_nearest(gt, metric_depth.shape)

        metric_row, _, _, _ = compare_depth_pair("metric", metric_depth, gt_resized, compute_abs_error=True)
        mono_row, _, _, _ = compare_depth_pair("mono", mono_depth, gt_resized, compute_abs_error=False)
        common = {
            "scan_name": config.scan_name,
            "image_stem": image_stem,
            "view_id": view_id,
            "light": image_light,
            "gt_depth_path": str(depth_path),
            "metric_depth_path": str(metric_depth_path),
            "mono_depth_path": str(mono_depth_path),
            "depth_shape": str(tuple(metric_depth.shape)),
            "gt_original_shape": str(tuple(gt.shape)),
        }
        rows.append({**common, **metric_row})
        rows.append({**common, **mono_row})

    metrics_df = pd.DataFrame(rows)
    metrics_csv_path = output_dir / f"{config.scan_name}_metric_mono_gt_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    summary_df = summarize_scan_metrics(metrics_df)
    summary_csv_path = output_dir / f"{config.scan_name}_metric_mono_gt_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    overview_path = save_scan_metrics_overview(metrics_df, output_dir / f"{config.scan_name}_metric_mono_gt_overview.png")
    return {
        "output_dir": output_dir,
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "metrics_csv_path": metrics_csv_path,
        "summary_csv_path": summary_csv_path,
        "overview_path": overview_path,
    }


def save_comparison_figure(
    rgb: np.ndarray,
    gt_resized: np.ndarray,
    metric_depth: np.ndarray,
    mono_depth: np.ndarray,
    metric_abs_error: np.ndarray,
    metric_norm_diff: np.ndarray,
    mono_norm_diff: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("RGB", rgb, None),
        ("GT depth resized", gt_resized, "Spectral"),
        ("DA3 metric depth", metric_depth, "Spectral"),
        ("metric abs error", metric_abs_error, "magma"),
        ("DA3 mono depth", mono_depth, "Spectral"),
        ("metric norm abs diff", metric_norm_diff, "magma"),
        ("mono norm abs diff", mono_norm_diff, "magma"),
        ("GT valid mask", np.isfinite(gt_resized) & (gt_resized > 0), "gray"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5))
    for ax, (title, data, cmap) in zip(axes.ravel(), panels):
        if cmap is None:
            ax.imshow(data)
        else:
            im = ax.imshow(data, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_da3_depth_gt_comparison(
    paths: ProjectPaths | None = None,
    config: DA3DepthGTCompareConfig | None = None,
    device: str | torch.device | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    config = config or DA3DepthGTCompareConfig()

    image_path = expected_image_path(
        paths.dtu_train_root,
        config.scan_name,
        config.split,
        config.image_dir,
        config.light_id,
        config.view_id,
    )
    depth_path = expected_depth_path(
        paths.dtu_train_root,
        config.scan_name,
        config.split,
        config.depth_dir,
        config.view_id,
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Missing GT depth: {depth_path}")

    metric_depth_path = find_metric_depth(config.metric_depth_root, config.scan_name, image_path.stem)
    metric_depth = np.load(metric_depth_path).astype(np.float32)
    rgb, mono_depth, mono_depth_path, mono_preview_path, mono_info = generate_or_load_mono_depth(
        image_path,
        config,
        device=device,
    )
    if mono_depth.shape != metric_depth.shape:
        raise ValueError(f"Metric and mono depth shapes differ: metric={metric_depth.shape}, mono={mono_depth.shape}")

    gt = read_pfm(str(depth_path)).astype(np.float32)
    if gt.ndim == 3:
        gt = gt[..., 0]
    gt_resized = resize_depth_nearest(gt, metric_depth.shape)

    metric_row, metric_mask, metric_abs_error, metric_norm_diff = compare_depth_pair(
        "metric",
        metric_depth,
        gt_resized,
        compute_abs_error=True,
    )
    mono_row, mono_mask, _, mono_norm_diff = compare_depth_pair(
        "mono",
        mono_depth,
        gt_resized,
        compute_abs_error=False,
    )

    output_dir = Path(config.comparison_output_root) / f"{config.scan_name}_view{config.view_id:03d}_light{config.light_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gt_resized.npy", gt_resized.astype(np.float32))
    np.save(output_dir / "metric_abs_error.npy", metric_abs_error.astype(np.float32))
    np.save(output_dir / "metric_norm_abs_diff.npy", metric_norm_diff.astype(np.float32))
    np.save(output_dir / "mono_norm_abs_diff.npy", mono_norm_diff.astype(np.float32))

    rows = []
    for row in (metric_row, mono_row):
        rows.append(
            {
                "scan_name": config.scan_name,
                "view_id": config.view_id,
                "light_id": config.light_id,
                "image_path": str(image_path),
                "gt_depth_path": str(depth_path),
                "metric_depth_path": str(metric_depth_path),
                "mono_depth_path": str(mono_depth_path),
                "depth_shape": str(tuple(metric_depth.shape)),
                "gt_original_shape": str(tuple(gt.shape)),
                **row,
            }
        )
    metrics_df = pd.DataFrame(rows)
    metrics_csv_path = output_dir / "da3_metric_mono_gt_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    figure_path = save_comparison_figure(
        rgb,
        gt_resized,
        metric_depth,
        mono_depth,
        metric_abs_error,
        metric_norm_diff,
        mono_norm_diff,
        output_dir / "da3_metric_mono_gt_comparison.png",
    )
    imageio.imwrite(output_dir / "gt_resized_vis.png", visualize_depth(gt_resized, cmap="Spectral"))
    imageio.imwrite(output_dir / "metric_depth_vis.png", visualize_depth(metric_depth, cmap="Spectral"))
    imageio.imwrite(output_dir / "mono_depth_vis.png", visualize_depth(mono_depth, cmap="Spectral"))

    return {
        "output_dir": output_dir,
        "metrics_df": metrics_df,
        "metrics_csv_path": metrics_csv_path,
        "figure_path": figure_path,
        "mono_depth_path": mono_depth_path,
        "mono_preview_path": mono_preview_path,
        "metric_depth_path": metric_depth_path,
        "gt_depth_path": depth_path,
        "metric_valid_pixels": int(metric_mask.sum()),
        "mono_valid_pixels": int(mono_mask.sum()),
        "mono_info": mono_info,
    }
