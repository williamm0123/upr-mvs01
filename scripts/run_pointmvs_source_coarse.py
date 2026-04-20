#!/usr/bin/env python
"""Run the original PointMVSNet source coarse depth visualization test."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pointmvs_source_coarse import (  # noqa: E402
    PointMVSSourceCoarseConfig,
    run_pointmvs_source_coarse_depth_test,
)
from upr_mvs.config import DTUConfig, ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/pointmvs_source_coarse"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtu-root", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--img-base-channels", type=int, default=8)
    parser.add_argument("--num-depths", type=int, default=48)
    parser.add_argument("--volume-base-channels", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--point-chunk-size", type=int, default=200000)
    parser.add_argument(
        "--pointmvs-checkpoint",
        type=Path,
        default=Path("models/PointMVSNet/outputs/dtu_wde3/model_pretrained.pth"),
        help="PointMVSNet checkpoint used to initialize coarse_img_conv and coarse_vol_conv.",
    )
    parser.add_argument("--no-load-weights", action="store_true")
    parser.add_argument("--no-rgb-to-bgr", action="store_true")
    parser.add_argument("--no-normalize-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    if args.dtu_root is not None:
        paths = replace(paths, dtu_train_root=args.dtu_root)
    if args.list_file is not None:
        paths = replace(paths, dtu_list_path=args.list_file)

    config = PointMVSSourceCoarseConfig(
        max_side=args.max_side,
        img_base_channels=args.img_base_channels,
        volume_base_channels=args.volume_base_channels,
        num_depths=args.num_depths,
        temperature=args.temperature,
        point_chunk_size=args.point_chunk_size,
        rgb_to_bgr=not args.no_rgb_to_bgr,
        normalize_images=not args.no_normalize_images,
        load_weights=not args.no_load_weights,
        pointmvs_checkpoint=args.pointmvs_checkpoint,
    )

    result = run_pointmvs_source_coarse_depth_test(
        sample_index=args.sample_index,
        paths=paths,
        dtu_config=DTUConfig(),
        config=config,
        output_root=args.output_root,
        device=args.device,
    )

    print("image shape:", result["image_shape"])
    print("feature shape:", result["feature_shape"])
    print("variance volume shape:", result["variance_volume_shape"])
    print("filtered cost shape:", result["filtered_cost_shape"])
    print("world points shape:", result["world_points_shape"])
    print("depth values shape:", result["depth_values_shape"])
    print("source weight load:", result["load_info"])
    print(result["metrics_df"].to_string(index=False))
    print("metrics csv:", result["metrics_csv_path"])
    print("ImageConv feature overview:", result["feature_visualization_path"])
    print("coarse depth result:", result["depth_visualization_path"])
    print("cost volume overview:", result["cost_visualization_path"])


if __name__ == "__main__":
    main()
