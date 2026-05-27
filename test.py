from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parent
DA3_SRC = REPO_ROOT / "models" / "Depth-Anything-3" / "src"
VGGT_ROOT = REPO_ROOT / "models" / "vggt"
for path in (REPO_ROOT, DA3_SRC, VGGT_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from base.config import ProjectPaths
from experiments.vggt_project_denoised_points_to_depth import (
    load_ref_camera,
    npz_stem_from_denoised_ply,
    project_points_to_depth,
)
import models.depth_fill as depth_fill
import models.general as G
import models.visual_tools as V
from models.depth_fill import read_binary_rgb_ply
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(
        description="Test DA3-normal-constrained depth completion from denoised VGGT PLY point clouds."
    )
    parser.add_argument(
        "--pointcloud-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "vggt_5view_reconstruction" / "pointclouds_denoised" / "knn5",
    )
    parser.add_argument(
        "--pose-npz-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "vggt_dtu_test",
    )
    parser.add_argument(
        "--vggt-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "vggt_5view_reconstruction",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "vggt_5view_reconstruction" / "depth_fill_normal_constraint_test_png",
    )
    parser.add_argument(
        "--da3-cache-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "vggt_5view_reconstruction"
        / "guided_fill_denoised"
        / "da3_resized_npy",
    )
    parser.add_argument("--da3-model-path", type=Path, default=paths.da3_weights_file)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="scan*_ref*_light*_denoised.ply")
    parser.add_argument("--image-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--ref-index", type=int, default=0)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=0.0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_image_path_map(vggt_root: Path) -> dict[str, str]:
    summary = read_json(vggt_root / "summary.json")
    image_paths: dict[str, str] = {}
    for sample in summary.get("samples", []):
        scan = sample["scan"]
        ref_view = int(sample["ref_view"])
        light_idx = int(sample["light_idx"])
        output_name = f"{scan}_ref{ref_view:03d}_light{light_idx}"
        image_paths[output_name] = sample["image_paths"][0]
    return image_paths


def load_cached_or_generate_da3(
    output_name: str,
    image_path: str,
    target_shape: tuple[int, int],
    args: argparse.Namespace,
    da3_state: dict,
) -> np.ndarray:
    cache_path = args.da3_cache_root / f"{output_name}_da3_resized.npy"
    if cache_path.is_file():
        return np.load(cache_path).astype(np.float32)

    if da3_state.get("model") is None:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        da3_state["model"], da3_state["device"] = depth_fill.load_da3_model(args.da3_model_path, device=device)
        print(f"Loaded DA3 on {da3_state['device']}")

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    da3_depth = depth_fill.generate_da3_depth_maps(image, da3_model=da3_state["model"])
    return depth_fill.resize_dense_depth(da3_depth, target_shape)


def preprocess_ref_color(image_path: str, image_mode: str, expected_shape: tuple[int, int]) -> np.ndarray:
    image = load_and_preprocess_images([image_path], mode=image_mode)[0]
    color = image.permute(1, 2, 0).cpu().numpy()
    color = np.clip(color * 255.0, 0, 255).astype(np.uint8)
    if color.shape[:2] != expected_shape:
        raise ValueError(f"Preprocessed color shape {color.shape[:2]} != depth shape {expected_shape}")
    return color


def save_filled_depth_pointcloud(
    filled_depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_path: str,
    image_mode: str,
    output_path: Path,
) -> int:
    valid = np.isfinite(filled_depth) & (filled_depth > 0)
    points = G.backproject_depth_to_points(filled_depth, intrinsic, extrinsic=extrinsic)
    colors = preprocess_ref_color(image_path, image_mode, filled_depth.shape)[valid]
    G.save_binary_ply(points, colors.astype(np.uint8), output_path)
    return int(points.shape[0])


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pointcloud_root = args.output_root / "pointclouds"
    pointcloud_root.mkdir(parents=True, exist_ok=True)

    image_path_map = build_image_path_map(args.vggt_root)
    ply_paths = sorted(args.pointcloud_root.glob(args.pattern))
    if args.max_files > 0:
        ply_paths = ply_paths[: args.max_files]
    if not ply_paths:
        raise FileNotFoundError(f"No PLY files matched {args.pointcloud_root / args.pattern}")

    da3_state: dict = {"model": None, "device": None}
    fill_config = depth_fill.NormalConstraintDepthFillConfig()

    for idx, ply_path in enumerate(ply_paths, start=1):
        output_name = npz_stem_from_denoised_ply(ply_path)
        npz_path = args.pose_npz_root / f"{output_name}.npz"
        image_path = image_path_map[output_name]
        print(f"[{idx}/{len(ply_paths)}] Normal-constrained fill: {output_name}")

        points, _ = read_binary_rgb_ply(ply_path)
        extrinsic, intrinsic, image_size = load_ref_camera(npz_path, args.ref_index)
        sparse_depth, project_info = project_points_to_depth(
            points=points,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            image_size=image_size,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            splat_radius=args.splat_radius,
        )
        da3_depth = load_cached_or_generate_da3(
            output_name=output_name,
            image_path=image_path,
            target_shape=sparse_depth.shape,
            args=args,
            da3_state=da3_state,
        )
        filled_depth, fill_info = depth_fill.fill_vggt_depth_by_da3_normals(
            sparse_depth=sparse_depth,
            da3_depth=da3_depth,
            intrinsic=intrinsic,
            config=fill_config,
        )

        png_path = args.output_root / f"{output_name}_normal_constraint_filled_depth.png"
        ply_path = pointcloud_root / f"{output_name}_normal_constraint_filled_depth.ply"
        V.save_depth_png(filled_depth, png_path, cmap="turbo", percentile=(1.0, 99.0))
        point_count = save_filled_depth_pointcloud(
            filled_depth=filled_depth,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            image_path=image_path,
            image_mode=args.image_mode,
            output_path=ply_path,
        )
        print(
            f"  sparse valid {project_info['valid_pixels']} / {sparse_depth.size}, "
            f"filled valid {fill_info['filled_valid_pixels']} / {sparse_depth.size}, "
            f"solver {fill_info['normal_fill']['solver']}({fill_info['normal_fill']['cg_info']})"
        )
        print(f"  saved {png_path}")
        print(f"  saved {ply_path} ({point_count} points)")


if __name__ == "__main__":
    main()
