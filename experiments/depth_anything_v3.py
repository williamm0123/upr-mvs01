"""Depth Anything 3 monocular depth visualization on DTU scans."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from data.dtu import expected_image_path
from upr_mvs.config import DEFAULT_DA3_MONO_MODEL_DIR, ProjectPaths


DA3_ROOT = Path(__file__).resolve().parents[1] / "models/Depth-Anything-3"
DA3_SRC = DA3_ROOT / "src"
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

from depth_anything_3.cfg import create_object  # noqa: E402
from depth_anything_3.utils.io.input_processor import InputProcessor  # noqa: E402
from depth_anything_3.utils.io.output_processor import OutputProcessor  # noqa: E402
from depth_anything_3.utils.visualize import visualize_depth  # noqa: E402


@dataclass(frozen=True)
class DA3VisualizationConfig:
    """Configuration for the local DA3MONO-LARGE DTU visualization test."""

    model_dir: Path = DEFAULT_DA3_MONO_MODEL_DIR
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    view_id: int = 0
    light_id: int = 3
    image_dir: str = "Rectified_raw"
    split: str = "train"


def read_scan_list(list_path: str | Path) -> list[str]:
    return [line.strip() for line in Path(list_path).read_text().splitlines() if line.strip()]


def load_da3_mono_model(
    model_dir: str | Path,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load DA3 from a local Hugging Face-style directory without importing export modules."""

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing DA3 config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing DA3 weights: {weights_path}")

    with config_path.open("r") as file:
        config_json = json.load(file)
    model_config = OmegaConf.create(config_json["config"])
    model = create_object(model_config)

    state_dict = load_file(str(weights_path), device="cpu")
    if state_dict and all(key.startswith("model.") for key in state_dict.keys()):
        state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}
    load_result = model.load_state_dict(state_dict, strict=False)

    device_t = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device_t).eval()
    load_info = {
        "model_name": config_json.get("model_name", "unknown"),
        "config": str(config_path),
        "weights": str(weights_path),
        "device": str(device_t),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    return model, load_info


def denormalize_processed_images(images: torch.Tensor) -> np.ndarray:
    """Convert DA3-normalized NCHW tensor to NHWC uint8."""

    images_np = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    images_np = images_np * std + mean
    images_np = np.clip(images_np, 0.0, 1.0)
    return (images_np * 255.0).round().astype(np.uint8)


def predict_da3_depth(
    model: torch.nn.Module,
    image_path: str | Path,
    process_res: int,
    process_res_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run DA3 mono depth for one image and return processed RGB, depth, confidence."""

    input_processor = InputProcessor()
    output_processor = OutputProcessor()
    device = next(model.parameters()).device

    images_cpu, _, _ = input_processor(
        [str(image_path)],
        process_res=process_res,
        process_res_method=process_res_method,
        num_workers=1,
        sequential=True,
    )
    images = images_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            raw_output = model(
                images,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy="first",
            )
    prediction = output_processor(raw_output)
    processed = denormalize_processed_images(images_cpu)[0]
    depth = prediction.depth[0].astype(np.float32)
    conf = None if prediction.conf is None else prediction.conf[0].astype(np.float32)
    return processed, depth, conf


def depth_stats(depth: np.ndarray) -> dict:
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        return {"depth_min": np.nan, "depth_median": np.nan, "depth_max": np.nan}
    values = depth[valid]
    return {
        "depth_min": float(np.min(values)),
        "depth_median": float(np.median(values)),
        "depth_max": float(np.max(values)),
    }


def save_scan_visualization(
    rgb: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray | None,
    scan_name: str,
    image_path: str | Path,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_vis = visualize_depth(depth, cmap="Spectral")

    ncols = 3 if conf is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 5.4))
    if ncols == 2:
        axes = np.asarray(axes)
    axes[0].imshow(rgb)
    axes[0].set_title(f"{scan_name} RGB\n{Path(image_path).name}")
    axes[0].axis("off")
    axes[1].imshow(depth_vis)
    axes[1].set_title("DA3MONO-LARGE depth")
    axes[1].axis("off")
    if conf is not None:
        axes[2].imshow(conf, cmap="magma")
        axes[2].set_title("confidence")
        axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_overview(rows: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No DA3 rows to visualize.")

    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, ...]
    for row_index, row in enumerate(rows):
        rgb = imageio.imread(row["processed_rgb_path"])
        depth_vis = imageio.imread(row["depth_vis_path"])
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].set_title(f"{row['scan_name']} RGB")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].imshow(depth_vis)
        axes[row_index, 1].set_title(
            "DA3 depth "
            f"median={row['depth_median']:.3f}, "
            f"range=[{row['depth_min']:.3f}, {row['depth_max']:.3f}]"
        )
        axes[row_index, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_da3_test_scan_visualization(
    paths: ProjectPaths | None = None,
    config: DA3VisualizationConfig | None = None,
    output_root: str | Path = "outputs/depth_anything_v3_test_scans",
    device: str | torch.device | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    config = config or DA3VisualizationConfig()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scans = read_scan_list(paths.dtu_list_path)
    model, load_info = load_da3_mono_model(config.model_dir, device=device)

    rows = []
    for scan_name in scans:
        image_path = expected_image_path(
            paths.dtu_train_root,
            scan_name,
            config.split,
            config.image_dir,
            config.light_id,
            config.view_id,
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing DTU scan image for DA3 test: {image_path}")

        rgb, depth, conf = predict_da3_depth(
            model,
            image_path,
            process_res=config.process_res,
            process_res_method=config.process_res_method,
        )
        scan_dir = output_root / scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = scan_dir / f"{scan_name}_processed_rgb.png"
        depth_path = scan_dir / f"{scan_name}_depth.npy"
        depth_vis_path = scan_dir / f"{scan_name}_depth_vis.png"
        conf_path = scan_dir / f"{scan_name}_confidence.npy"
        combined_path = scan_dir / f"{scan_name}_da3mono_depth.png"

        imageio.imwrite(rgb_path, rgb)
        np.save(depth_path, depth)
        imageio.imwrite(depth_vis_path, visualize_depth(depth, cmap="Spectral"))
        if conf is not None:
            np.save(conf_path, conf)
        save_scan_visualization(rgb, depth, conf, scan_name, image_path, combined_path)

        stats = depth_stats(depth)
        if conf is not None:
            conf_valid = np.isfinite(conf)
            conf_mean = float(np.mean(conf[conf_valid])) if conf_valid.any() else np.nan
        else:
            conf_mean = np.nan
        rows.append(
            {
                "scan_name": scan_name,
                "image_path": str(image_path),
                "processed_shape": str(tuple(rgb.shape)),
                "depth_shape": str(tuple(depth.shape)),
                "confidence_available": conf is not None,
                "confidence_mean": conf_mean,
                "processed_rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "depth_vis_path": str(depth_vis_path),
                "confidence_path": str(conf_path) if conf is not None else "",
                "combined_path": str(combined_path),
                **stats,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(rows)
    summary_csv_path = output_root / "da3mono_test_scans_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    overview_path = save_overview(rows, output_root / "da3mono_test_scans_overview.png")

    return {
        "load_info": load_info,
        "summary_df": summary_df,
        "summary_csv_path": summary_csv_path,
        "overview_path": overview_path,
        "output_root": output_root,
    }
