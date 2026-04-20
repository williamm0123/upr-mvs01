#!/usr/bin/env python
"""Run the PointMVSNet-style coarse depth test with FPN features."""

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

from experiments.pointmvs_coarse import PointMVSCoarseConfig, run_pointmvs_coarse_depth_test
from upr_mvs.config import DTUConfig, ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/pointmvs_coarse"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtu-root", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--level", type=int, default=4, help="FPN level used as the coarse feature. P4 is 1/8 scale.")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--fpn-channels", type=int, default=64)
    parser.add_argument("--num-depths", type=int, default=48)
    parser.add_argument("--volume-base-channels", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--point-chunk-size", type=int, default=200000)
    parser.add_argument(
        "--pointmvs-checkpoint",
        type=Path,
        default=Path("models/PointMVSNet/outputs/dtu_wde3/model_pretrained.pth"),
        help="Optional PointMVSNet checkpoint used to initialize only coarse_vol_conv.",
    )
    parser.add_argument("--no-load-volume-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    if args.dtu_root is not None:
        paths = replace(paths, dtu_train_root=args.dtu_root)
    if args.list_file is not None:
        paths = replace(paths, dtu_list_path=args.list_file)

    config = PointMVSCoarseConfig(
        pyramid_level=args.level,
        max_side=args.max_side,
        fpn_channels=args.fpn_channels,
        num_depths=args.num_depths,
        volume_base_channels=args.volume_base_channels,
        temperature=args.temperature,
        point_chunk_size=args.point_chunk_size,
        load_volume_weights=not args.no_load_volume_weights,
        pointmvs_checkpoint=args.pointmvs_checkpoint,
    )

    result = run_pointmvs_coarse_depth_test(
        sample_index=args.sample_index,
        paths=paths,
        dtu_config=DTUConfig(),
        config=config,
        output_root=args.output_root,
        device=args.device,
    )

    print("level:", result["level"])
    print("image shape:", result["image_shape"])
    print("feature shape:", result["feature_shape"])
    print("variance volume shape:", result["variance_volume_shape"])
    print("filtered cost shape:", result["filtered_cost_shape"])
    print("world points shape:", result["world_points_shape"])
    print("depth values shape:", result["depth_values_shape"])
    print("volume weight load:", result["load_info"])
    print(result["metrics_df"].to_string(index=False))
    print("metrics csv:", result["metrics_csv_path"])
    print("FPN feature overview:", result["fpn_feature_path"])
    print("coarse depth result:", result["depth_visualization_path"])
    print("cost volume overview:", result["cost_visualization_path"])


if __name__ == "__main__":
    main()
