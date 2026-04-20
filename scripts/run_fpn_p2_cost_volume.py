#!/usr/bin/env python
"""Build a P2 cost volume from FPN-only matching features."""

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

from experiments.fpn_cost_volume import FPNCostVolumeConfig, run_fpn_p2_cost_volume_test
from upr_mvs.config import DTUConfig, ProjectPaths


def parse_kernel(value: str) -> tuple[int, int, int]:
    kernel = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(kernel) != 3:
        raise argparse.ArgumentTypeError("Kernel must have three integers, for example 3,3,3")
    if any(item <= 0 or item % 2 == 0 for item in kernel):
        raise argparse.ArgumentTypeError("Kernel values must be positive odd integers")
    return kernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fpn_p2_cost_volume"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtu-root", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--level", type=int, default=2, help="FPN pyramid level l. This test defaults to l=2.")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--fpn-channels", type=int, default=16)
    parser.add_argument("--matching-channels", type=int, default=16)
    parser.add_argument("--num-depths", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--channel-chunk", type=int, default=4)
    parser.add_argument("--regularization", choices=("avg3d", "none"), default="avg3d")
    parser.add_argument("--regularization-blend", type=float, default=0.5)
    parser.add_argument("--regularization-kernel", type=parse_kernel, default=(3, 3, 3))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    if args.dtu_root is not None:
        paths = replace(paths, dtu_train_root=args.dtu_root)
    if args.list_file is not None:
        paths = replace(paths, dtu_list_path=args.list_file)

    config = FPNCostVolumeConfig(
        pyramid_level=args.level,
        max_side=args.max_side,
        fpn_channels=args.fpn_channels,
        matching_channels=args.matching_channels,
        num_depths=args.num_depths,
        temperature=args.temperature,
        channel_chunk=args.channel_chunk,
        regularization=args.regularization,
        regularization_blend=args.regularization_blend,
        regularization_kernel=args.regularization_kernel,
    )

    result = run_fpn_p2_cost_volume_test(
        sample_index=args.sample_index,
        paths=paths,
        dtu_config=DTUConfig(),
        config=config,
        output_root=args.output_root,
        device=args.device,
    )

    print("level:", result["level"])
    print("image shape:", result["image_shape"])
    print("feature shape:", result["features_shape"])
    print("projection shape:", result["projection_shape"])
    print("depth values shape:", result["depth_values_shape"])
    print("cost volume shape:", result["cost_volume_shape"])
    print("probability volume shape:", result["probability_volume_shape"])
    print(result["metrics_df"].to_string(index=False))
    print("metrics csv:", result["metrics_csv_path"])
    print("feature overview:", result["feature_overview_path"])
    print("raw depth result:", result["raw_depth_visualization_path"])
    print("raw cost diagnostics:", result["raw_cost_volume_diagnostics_path"])
    print("regularized depth result:", result["regularized_depth_visualization_path"])
    print("regularized cost diagnostics:", result["regularized_cost_volume_diagnostics_path"])


if __name__ == "__main__":
    main()
