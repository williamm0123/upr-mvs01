"""Reusable experiment entry points extracted from the notebook."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .adapter import (
    Conv1x1Adapter,
    Conv1x1Conv3x3Adapter,
    GeometryAdapter,
    adapter_features_for_grid,
    make_adapter_native_input,
    make_depth_training_targets,
    raw_selected_dino_features_for_grid,
    train_geometry_adapter,
)
from .cost_volume import evaluate_feature_cost_volume_highres, prepare_rgb_feature_grid_baseline
from .geometry import make_projection_for_feature_grid, resize_gt_to_feature_grid
from .pointcloud import (
    ref_image_colors_for_grid,
    save_depth_point_cloud,
    save_pixel_depth_point_cloud,
    scaled_ref_camera_for_grid,
)
from .visualization import (
    save_adapter_ablation_summary,
    save_adapter_training_summary,
    save_depth_result_image,
    save_metrics_summary_plots,
)
from data.dtu import build_dtu_dataset
from models.dinov3.extractor import (
    dino_normalization_tensors,
    extract_dinov3_native_features,
    load_dinov3_vit_base,
    project_and_resize_dino_layer,
)
from upr_mvs.config import AdapterConfig, CostVolumeConfig, DINOConfig, DTUConfig, ProjectPaths
from upr_mvs.external import sample_depth_planes


METRIC_COLUMNS = [
    "feature",
    "layer",
    "channels",
    "height",
    "width",
    "valid_ratio",
    "argmin_mean",
    "argmin_median",
    "argmin_p90",
    "soft_mean",
    "soft_median",
    "soft_p90",
    "soft_<10mm",
    "soft_<25mm",
    "soft_<50mm",
    "confidence_mean",
    "confidence_median",
    "num_eval_pixels",
    "visualization",
]


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sample(
    sample_index: int = 0,
    paths: ProjectPaths | None = None,
    dtu_config: DTUConfig | None = None,
) -> tuple[object, dict]:
    dataset = build_dtu_dataset(paths=paths, config=dtu_config)
    sample = dataset[sample_index]
    return dataset, sample


def feature_hw_from_scale(sample: dict, scale: float) -> tuple[int, int]:
    image_h, image_w = sample["imgs"].shape[-2:]
    return max(1, int(round(image_h * scale))), max(1, int(round(image_w * scale)))


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slugify_feature_name(name: str) -> str:
    return (
        name.lower()
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace("__", "_")
    )


def _export_depth_maps_as_point_clouds(
    sample: dict,
    feature_hw: tuple[int, int],
    maps: dict,
    output_dir: Path,
    output_stem: str,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    colors,
) -> dict:
    pointcloud_dir = output_dir / "pointclouds"
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    soft_path, soft_count = save_depth_point_cloud(
        depth=maps["soft_depth"][0],
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        valid_mask=maps["valid_mask"][0],
        colors=colors,
        output_path=pointcloud_dir / f"{output_stem}_soft_depth.ply",
    )
    saved["soft_pointcloud"] = str(soft_path)
    saved["soft_point_count"] = soft_count

    argmin_path, argmin_count = save_depth_point_cloud(
        depth=maps["argmin_depth"][0],
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        valid_mask=maps["valid_mask"][0],
        colors=colors,
        output_path=pointcloud_dir / f"{output_stem}_argmin_depth.ply",
    )
    saved["argmin_pointcloud"] = str(argmin_path)
    saved["argmin_point_count"] = argmin_count

    pixel_soft_path, pixel_soft_count = save_pixel_depth_point_cloud(
        depth=maps["soft_depth"][0],
        valid_mask=maps["valid_mask"][0],
        colors=colors,
        output_path=pointcloud_dir / f"{output_stem}_pixel_soft_depth.ply",
    )
    saved["pixel_soft_pointcloud"] = str(pixel_soft_path)
    saved["pixel_soft_point_count"] = pixel_soft_count

    pixel_argmin_path, pixel_argmin_count = save_pixel_depth_point_cloud(
        depth=maps["argmin_depth"][0],
        valid_mask=maps["valid_mask"][0],
        colors=colors,
        output_path=pointcloud_dir / f"{output_stem}_pixel_argmin_depth.ply",
    )
    saved["pixel_argmin_pointcloud"] = str(pixel_argmin_path)
    saved["pixel_argmin_point_count"] = pixel_argmin_count
    return saved


def _export_gt_point_cloud(
    sample: dict,
    feature_hw: tuple[int, int],
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    output_dir: Path,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    colors,
) -> dict:
    pointcloud_dir = output_dir / "pointclouds"
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    gt_path, gt_count = save_depth_point_cloud(
        depth=depth_gt[0],
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        valid_mask=mask[0],
        colors=colors,
        output_path=pointcloud_dir / "gt_depth_ref_view.ply",
    )
    pixel_gt_path, pixel_gt_count = save_pixel_depth_point_cloud(
        depth=depth_gt[0],
        valid_mask=mask[0],
        colors=colors,
        output_path=pointcloud_dir / "gt_pixel_depth_ref_view.ply",
    )
    return {
        "gt_pointcloud": str(gt_path),
        "gt_point_count": gt_count,
        "gt_pixel_pointcloud": str(pixel_gt_path),
        "gt_pixel_point_count": pixel_gt_count,
    }


def run_dinov3_cost_volume_comparison(
    sample_index: int = 0,
    paths: ProjectPaths | None = None,
    dtu_config: DTUConfig | None = None,
    cost_config: CostVolumeConfig | None = None,
    dino_config: DINOConfig | None = None,
    output_root: str | Path = "outputs/dinov3_cost_volume",
    device: str | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    dtu_config = dtu_config or DTUConfig()
    cost_config = cost_config or CostVolumeConfig()
    dino_config = dino_config or DINOConfig()
    device_t = resolve_device(device)

    _, sample = load_sample(sample_index=sample_index, paths=paths, dtu_config=dtu_config)
    target_feature_hw = feature_hw_from_scale(sample, cost_config.scale)
    output_dir = Path(output_root) / sample["sample_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    mean, std = dino_normalization_tensors(device_t, dino_config)
    model = load_dinov3_vit_base(
        device=device_t,
        weights_file=paths.dinov3_weights_file,
        patch_size=dino_config.patch_size,
    )
    _empty_cuda_cache()

    dinov3_native = extract_dinov3_native_features(
        sample=sample,
        model=model,
        device=device_t,
        max_side=dino_config.input_max_side,
        patch_size=dino_config.patch_size,
        layers=dino_config.layers,
        mean=mean,
        std=std,
    )
    dino_projection_matrices = make_projection_for_feature_grid(sample, target_feature_hw, device_t)
    dino_depth_range = sample["depth_range"].view(1, 2).to(device=device_t, dtype=torch.float32)
    dino_depth_values = sample_depth_planes(dino_depth_range, cost_config.num_depths)
    dino_depth_gt_target, dino_mask_target = resize_gt_to_feature_grid(sample, target_feature_hw, device_t)

    rows = []
    visualization_paths = []

    rgb_features, rgb_projection, rgb_depth_values, rgb_depth_gt, rgb_mask = prepare_rgb_feature_grid_baseline(
        sample=sample,
        feature_hw=target_feature_hw,
        num_depths=cost_config.num_depths,
        device=device_t,
    )
    rgb_row, rgb_maps = evaluate_feature_cost_volume_highres(
        feature_name="RGB image feature",
        features_for_cost=rgb_features,
        projection_matrices=rgb_projection,
        depth_values=rgb_depth_values,
        depth_gt_target=rgb_depth_gt,
        mask_target=rgb_mask,
        temperature=cost_config.temperature,
        channel_chunk=cost_config.channel_chunk,
    )
    rgb_row["layer"] = np.nan
    rgb_row["visualization"] = str(
        save_depth_result_image("RGB image feature", rgb_row, rgb_maps, rgb_depth_gt, output_dir / "rgb_feature_result.png")
    )
    rows.append(rgb_row)
    visualization_paths.append(rgb_row["visualization"])
    del rgb_features, rgb_maps
    _empty_cuda_cache()

    for layer_order, native_layer_features in enumerate(dinov3_native["layer_features"]):
        layer_id = dino_config.layers[layer_order] + 1
        layer_features = project_and_resize_dino_layer(
            native_layer_features,
            target_feature_hw=target_feature_hw,
            out_channels=dino_config.project_channels,
            device=device_t,
            seed=dino_config.random_projection_seed,
        )
        row, maps = evaluate_feature_cost_volume_highres(
            feature_name=f"DINOv3 layer {layer_id:02d}",
            features_for_cost=layer_features,
            projection_matrices=dino_projection_matrices,
            depth_values=dino_depth_values,
            depth_gt_target=dino_depth_gt_target,
            mask_target=dino_mask_target,
            temperature=cost_config.temperature,
            channel_chunk=cost_config.channel_chunk,
        )
        row["layer"] = layer_id
        row["visualization"] = str(
            save_depth_result_image(
                f"DINOv3 layer {layer_id:02d}",
                row,
                maps,
                dino_depth_gt_target,
                output_dir / f"dinov3_layer_{layer_id:02d}_result.png",
            )
        )
        rows.append(row)
        visualization_paths.append(row["visualization"])
        print(
            f"Layer {layer_id:02d}: soft median={row['soft_median']:.3f}, "
            f"soft mean={row['soft_mean']:.3f}, argmin median={row['argmin_median']:.3f}, "
            f"conf median={row['confidence_median']:.3f}"
        )
        del layer_features, maps
        _empty_cuda_cache()

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[[col for col in METRIC_COLUMNS if col in metrics_df.columns]]
    metrics_csv_path = output_dir / "dinov3_cost_volume_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    summary_plot_path = save_metrics_summary_plots(metrics_df, output_dir / "dinov3_cost_volume_metric_summary.png")
    return {
        "sample": sample,
        "target_feature_hw": target_feature_hw,
        "metrics_df": metrics_df,
        "metrics_csv_path": metrics_csv_path,
        "summary_plot_path": summary_plot_path,
        "visualization_paths": visualization_paths,
    }


def run_geometry_adapter_test(
    sample_index: int = 0,
    paths: ProjectPaths | None = None,
    dtu_config: DTUConfig | None = None,
    cost_config: CostVolumeConfig | None = None,
    dino_config: DINOConfig | None = None,
    adapter_config: AdapterConfig | None = None,
    output_root: str | Path = "outputs/geometry_adapter",
    device: str | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    dtu_config = dtu_config or DTUConfig()
    cost_config = cost_config or CostVolumeConfig()
    dino_config = dino_config or DINOConfig()
    adapter_config = adapter_config or AdapterConfig()
    device_t = resolve_device(device)

    _, sample = load_sample(sample_index=sample_index, paths=paths, dtu_config=dtu_config)
    eval_feature_hw = feature_hw_from_scale(sample, cost_config.scale)
    train_feature_hw = (
        max(1, eval_feature_hw[0] // adapter_config.train_grid_divisor),
        max(1, eval_feature_hw[1] // adapter_config.train_grid_divisor),
    )
    output_dir = Path(output_root) / sample["sample_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    mean, std = dino_normalization_tensors(device_t, dino_config)
    model = load_dinov3_vit_base(
        device=device_t,
        weights_file=paths.dinov3_weights_file,
        patch_size=dino_config.patch_size,
    )
    _empty_cuda_cache()

    dinov3_native = extract_dinov3_native_features(
        sample=sample,
        model=model,
        device=device_t,
        max_side=dino_config.input_max_side,
        patch_size=dino_config.patch_size,
        layers=dino_config.layers,
        mean=mean,
        std=std,
    )
    native_input = make_adapter_native_input(dinov3_native, adapter_config.layer_ids).to(device_t)
    adapter = GeometryAdapter(
        in_ch=int(native_input.shape[1]),
        hidden_ch=adapter_config.hidden_ch,
        out_ch=adapter_config.out_ch,
    ).to(device_t)

    print("adapter native input shape:", tuple(native_input.shape))
    print("adapter parameter count   :", sum(p.numel() for p in adapter.parameters()))
    print("adapter train feature_hw  :", train_feature_hw)
    print("adapter eval feature_hw   :", eval_feature_hw)

    history_df = train_geometry_adapter(
        adapter=adapter,
        native_input=native_input,
        sample=sample,
        train_feature_hw=train_feature_hw,
        num_depths=cost_config.num_depths,
        temperature=cost_config.temperature,
        config=adapter_config,
        device=device_t,
        channel_chunk=cost_config.channel_chunk,
    )
    history_csv_path = output_dir / "geometry_adapter_training_history.csv"
    history_df.to_csv(history_csv_path, index=False)

    projection_eval, depth_values_eval, depth_gt_eval, mask_eval, all_views_valid_eval = make_depth_training_targets(
        sample=sample,
        feature_hw=eval_feature_hw,
        device=device_t,
        num_depths=cost_config.num_depths,
    )

    rows = []
    with torch.no_grad():
        raw_features = raw_selected_dino_features_for_grid(
            native_input,
            target_feature_hw=eval_feature_hw,
            out_channels=adapter_config.out_ch,
            device=device_t,
            seed=dino_config.random_projection_seed,
        )
    raw_row, raw_maps = evaluate_feature_cost_volume_highres(
        feature_name=f"Raw DINO layers {adapter_config.layer_ids}",
        features_for_cost=raw_features,
        projection_matrices=projection_eval,
        depth_values=depth_values_eval,
        depth_gt_target=depth_gt_eval,
        mask_target=mask_eval,
        temperature=cost_config.temperature,
        channel_chunk=cost_config.channel_chunk,
        all_views_valid=all_views_valid_eval,
    )
    raw_row["visualization"] = str(
        save_depth_result_image(
            raw_row["feature"],
            raw_row,
            raw_maps,
            depth_gt_eval,
            output_dir / "raw_selected_dino_result.png",
        )
    )
    rows.append(raw_row)
    del raw_features, raw_maps
    _empty_cuda_cache()

    adapter.eval()
    with torch.no_grad():
        adapted_features = adapter_features_for_grid(adapter, native_input, eval_feature_hw)
    adapted_row, adapted_maps = evaluate_feature_cost_volume_highres(
        feature_name=f"GeometryAdapter layers {adapter_config.layer_ids}",
        features_for_cost=adapted_features,
        projection_matrices=projection_eval,
        depth_values=depth_values_eval,
        depth_gt_target=depth_gt_eval,
        mask_target=mask_eval,
        temperature=cost_config.temperature,
        channel_chunk=cost_config.channel_chunk,
        all_views_valid=all_views_valid_eval,
    )
    adapted_row["visualization"] = str(
        save_depth_result_image(
            adapted_row["feature"],
            adapted_row,
            adapted_maps,
            depth_gt_eval,
            output_dir / "geometry_adapter_result.png",
        )
    )
    rows.append(adapted_row)
    del adapted_features, adapted_maps
    _empty_cuda_cache()

    metrics_df = pd.DataFrame(rows)
    metrics_csv_path = output_dir / "geometry_adapter_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    summary_plot_path = save_adapter_training_summary(history_df, metrics_df, output_dir / "geometry_adapter_summary.png")
    return {
        "sample": sample,
        "train_feature_hw": train_feature_hw,
        "eval_feature_hw": eval_feature_hw,
        "history_df": history_df,
        "metrics_df": metrics_df,
        "history_csv_path": history_csv_path,
        "metrics_csv_path": metrics_csv_path,
        "summary_plot_path": summary_plot_path,
        "output_dir": output_dir,
    }


def run_adapter_ablation_test(
    sample_index: int = 0,
    paths: ProjectPaths | None = None,
    dtu_config: DTUConfig | None = None,
    cost_config: CostVolumeConfig | None = None,
    dino_config: DINOConfig | None = None,
    adapter_config: AdapterConfig | None = None,
    output_root: str | Path = "outputs/adapter_ablation",
    device: str | None = None,
    export_pointclouds: bool = True,
) -> dict:
    paths = paths or ProjectPaths()
    dtu_config = dtu_config or DTUConfig()
    cost_config = cost_config or CostVolumeConfig()
    dino_config = dino_config or DINOConfig()
    adapter_config = adapter_config or AdapterConfig()
    device_t = resolve_device(device)

    _, sample = load_sample(sample_index=sample_index, paths=paths, dtu_config=dtu_config)
    eval_feature_hw = feature_hw_from_scale(sample, cost_config.scale)
    train_feature_hw = (
        max(1, eval_feature_hw[0] // adapter_config.train_grid_divisor),
        max(1, eval_feature_hw[1] // adapter_config.train_grid_divisor),
    )
    output_dir = Path(output_root) / sample["sample_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    mean, std = dino_normalization_tensors(device_t, dino_config)
    model = load_dinov3_vit_base(
        device=device_t,
        weights_file=paths.dinov3_weights_file,
        patch_size=dino_config.patch_size,
    )
    _empty_cuda_cache()

    dinov3_native = extract_dinov3_native_features(
        sample=sample,
        model=model,
        device=device_t,
        max_side=dino_config.input_max_side,
        patch_size=dino_config.patch_size,
        layers=dino_config.layers,
        mean=mean,
        std=std,
    )
    single_input = make_adapter_native_input(dinov3_native, (adapter_config.single_layer_id,)).to(device_t)
    fusion_input = make_adapter_native_input(dinov3_native, adapter_config.layer_ids).to(device_t)

    projection_eval, depth_values_eval, depth_gt_eval, mask_eval, all_views_valid_eval = make_depth_training_targets(
        sample=sample,
        feature_hw=eval_feature_hw,
        device=device_t,
        num_depths=cost_config.num_depths,
    )
    pointcloud_info = {}
    if export_pointclouds:
        ref_colors = ref_image_colors_for_grid(sample, eval_feature_hw)
        ref_intrinsics, ref_extrinsics = scaled_ref_camera_for_grid(sample, eval_feature_hw, device="cpu")
        pointcloud_info = _export_gt_point_cloud(
            sample=sample,
            feature_hw=eval_feature_hw,
            depth_gt=depth_gt_eval.detach().cpu(),
            mask=mask_eval.detach().cpu(),
            output_dir=output_dir,
            intrinsics=ref_intrinsics,
            extrinsics=ref_extrinsics,
            colors=ref_colors,
        )
    else:
        ref_colors = None
        ref_intrinsics = None
        ref_extrinsics = None

    rows = []
    history_frames = []

    with torch.no_grad():
        raw_features = raw_selected_dino_features_for_grid(
            single_input,
            target_feature_hw=eval_feature_hw,
            out_channels=adapter_config.out_ch,
            device=device_t,
            seed=dino_config.random_projection_seed,
        )
    raw_name = f"raw DINO layer {adapter_config.single_layer_id:02d}"
    raw_row, raw_maps = evaluate_feature_cost_volume_highres(
        feature_name=raw_name,
        features_for_cost=raw_features,
        projection_matrices=projection_eval,
        depth_values=depth_values_eval,
        depth_gt_target=depth_gt_eval,
        mask_target=mask_eval,
        temperature=cost_config.temperature,
        channel_chunk=cost_config.channel_chunk,
        all_views_valid=all_views_valid_eval,
    )
    raw_row["variant"] = "raw_dino"
    raw_row["trainable_params"] = 0
    raw_row["dino_layers"] = str((adapter_config.single_layer_id,))
    if export_pointclouds:
        raw_row.update(pointcloud_info)
        raw_row.update(
            _export_depth_maps_as_point_clouds(
                sample=sample,
                feature_hw=eval_feature_hw,
                maps=raw_maps,
                output_dir=output_dir,
                output_stem="raw_dino",
                intrinsics=ref_intrinsics,
                extrinsics=ref_extrinsics,
                colors=ref_colors,
            )
        )
    raw_row["visualization"] = str(
        save_depth_result_image(
            raw_name,
            raw_row,
            raw_maps,
            depth_gt_eval,
            output_dir / "raw_dino_result.png",
        )
    )
    rows.append(raw_row)
    del raw_features, raw_maps
    _empty_cuda_cache()

    variants = [
        (
            "DINO + 1x1 conv",
            "dino_1x1_conv",
            Conv1x1Adapter(
                in_ch=int(single_input.shape[1]),
                out_ch=adapter_config.out_ch,
            ).to(device_t),
            single_input,
            (adapter_config.single_layer_id,),
        ),
        (
            "DINO + 1x1 + 3x3 conv",
            "dino_1x1_3x3_conv",
            Conv1x1Conv3x3Adapter(
                in_ch=int(single_input.shape[1]),
                hidden_ch=adapter_config.hidden_ch,
                out_ch=adapter_config.out_ch,
            ).to(device_t),
            single_input,
            (adapter_config.single_layer_id,),
        ),
        (
            "DINO + residual adapter",
            "dino_residual_adapter",
            GeometryAdapter(
                in_ch=int(single_input.shape[1]),
                hidden_ch=adapter_config.hidden_ch,
                out_ch=adapter_config.out_ch,
            ).to(device_t),
            single_input,
            (adapter_config.single_layer_id,),
        ),
        (
            "DINO multi-layer fusion + adapter",
            "dino_multilayer_fusion_adapter",
            GeometryAdapter(
                in_ch=int(fusion_input.shape[1]),
                hidden_ch=adapter_config.hidden_ch,
                out_ch=adapter_config.out_ch,
            ).to(device_t),
            fusion_input,
            adapter_config.layer_ids,
        ),
    ]

    for feature_name, variant_slug, adapter, native_input, dino_layers in variants:
        print("=" * 80)
        print(feature_name)
        print("dino layers       :", dino_layers)
        print("trainable params  :", sum(p.numel() for p in adapter.parameters()))
        history_df = train_geometry_adapter(
            adapter=adapter,
            native_input=native_input,
            sample=sample,
            train_feature_hw=train_feature_hw,
            num_depths=cost_config.num_depths,
            temperature=cost_config.temperature,
            config=adapter_config,
            device=device_t,
            channel_chunk=cost_config.channel_chunk,
        )
        history_df["feature"] = feature_name
        history_df["variant"] = variant_slug
        history_df["dino_layers"] = str(dino_layers)
        history_frames.append(history_df)
        history_df.to_csv(output_dir / f"{variant_slug}_training_history.csv", index=False)

        adapter.eval()
        with torch.no_grad():
            adapted_features = adapter_features_for_grid(adapter, native_input, eval_feature_hw)
        row, maps = evaluate_feature_cost_volume_highres(
            feature_name=feature_name,
            features_for_cost=adapted_features,
            projection_matrices=projection_eval,
            depth_values=depth_values_eval,
            depth_gt_target=depth_gt_eval,
            mask_target=mask_eval,
            temperature=cost_config.temperature,
            channel_chunk=cost_config.channel_chunk,
            all_views_valid=all_views_valid_eval,
        )
        row["variant"] = variant_slug
        row["trainable_params"] = sum(p.numel() for p in adapter.parameters())
        row["dino_layers"] = str(dino_layers)
        if export_pointclouds:
            row.update(pointcloud_info)
            row.update(
                _export_depth_maps_as_point_clouds(
                    sample=sample,
                    feature_hw=eval_feature_hw,
                    maps=maps,
                    output_dir=output_dir,
                    output_stem=variant_slug,
                    intrinsics=ref_intrinsics,
                    extrinsics=ref_extrinsics,
                    colors=ref_colors,
                )
            )
        row["visualization"] = str(
            save_depth_result_image(
                feature_name,
                row,
                maps,
                depth_gt_eval,
                output_dir / f"{variant_slug}_result.png",
            )
        )
        rows.append(row)
        print(
            f"{feature_name}: soft median={row['soft_median']:.3f}, "
            f"soft mean={row['soft_mean']:.3f}, argmin median={row['argmin_median']:.3f}, "
            f"conf median={row['confidence_median']:.4f}"
        )
        del adapted_features, maps, adapter
        _empty_cuda_cache()

    metrics_df = pd.DataFrame(rows)
    metrics_csv_path = output_dir / "adapter_ablation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    if history_frames:
        history_all_df = pd.concat(history_frames, ignore_index=True)
    else:
        history_all_df = pd.DataFrame()
    history_csv_path = output_dir / "adapter_ablation_training_history.csv"
    history_all_df.to_csv(history_csv_path, index=False)
    summary_plot_path = save_adapter_ablation_summary(
        history_all_df,
        metrics_df,
        output_dir / "adapter_ablation_summary.png",
    )

    return {
        "sample": sample,
        "train_feature_hw": train_feature_hw,
        "eval_feature_hw": eval_feature_hw,
        "metrics_df": metrics_df,
        "history_df": history_all_df,
        "metrics_csv_path": metrics_csv_path,
        "history_csv_path": history_csv_path,
        "summary_plot_path": summary_plot_path,
        "output_dir": output_dir,
    }
