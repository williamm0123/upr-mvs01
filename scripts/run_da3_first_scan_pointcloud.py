#!/usr/bin/env python
"""Run DA3 on all images of the first test scan and export a raw-depth point cloud."""

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

from experiments.da3_scan_pointcloud import DA3ScanPointCloudConfig, run_da3_first_scan_pointcloud  # noqa: E402
from experiments.depth_anything_v3 import DA3VisualizationConfig  # noqa: E402
from upr_mvs.config import DEFAULT_DA3_MONO_MODEL_DIR, ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/da3_first_scan_pointcloud"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtu-root", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DA3_MONO_MODEL_DIR)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument(
        "--depth-scale-mode",
        choices=("constant", "metric_focal_300", "auto"),
        default="constant",
        help="Use metric_focal_300 for DA3METRIC-LARGE, optionally with --depth-scale 1000 for meters-to-mm.",
    )
    parser.add_argument("--light-id", type=int, default=None, help="Optionally keep only one DTU light id.")
    parser.add_argument("--no-max-images", action="store_true", help="Skip rect_*_max.png images.")
    parser.add_argument("--save-depths", action="store_true", help="Save every DA3 depth .npy and depth preview image.")
    parser.add_argument("--preview-max-points", type=int, default=180000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    if args.dtu_root is not None:
        paths = replace(paths, dtu_train_root=args.dtu_root)
    if args.list_file is not None:
        paths = replace(paths, dtu_list_path=args.list_file)

    da3_config = DA3VisualizationConfig(
        model_dir=args.model_dir,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
    )
    config = DA3ScanPointCloudConfig(
        da3=da3_config,
        point_stride=args.point_stride,
        depth_scale=args.depth_scale,
        depth_scale_mode=args.depth_scale_mode,
        include_max_images=not args.no_max_images,
        light_id=args.light_id,
        save_depths=args.save_depths,
        preview_max_points=args.preview_max_points,
    )
    result = run_da3_first_scan_pointcloud(
        paths=paths,
        config=config,
        output_root=args.output_root,
        device=args.device,
    )

    print("DA3 load info:", result["load_info"])
    print("scan:", result["scan_name"])
    print("num images:", result["num_images"])
    print("num points:", result["num_points"])
    print("pointcloud:", result["pointcloud_path"])
    print("preview:", result["preview_path"])
    print("summary csv:", result["summary_csv_path"])
    print(result["summary_df"].head(12).to_string(index=False))
    print("output root:", result["output_root"])


if __name__ == "__main__":
    main()
