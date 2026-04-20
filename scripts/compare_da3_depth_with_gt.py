"""CLI for comparing DA3 mono/metric depth against DTU GT depth."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.da3_depth_gt_compare import (  # noqa: E402
    DA3DepthGTCompareConfig,
    generate_scan_mono_depths,
    run_da3_depth_gt_comparison,
    run_da3_scan_depth_gt_comparison,
)
from upr_mvs.config import DEFAULT_DA3_MONO_MODEL_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-name", default="scan1")
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--light-id", type=int, default=0)
    parser.add_argument("--depth-dir", default="Depths_raw")
    parser.add_argument("--metric-depth-root", type=Path, default=Path("outputs/da3metric_first_scan_depths"))
    parser.add_argument("--mono-model-dir", type=Path, default=DEFAULT_DA3_MONO_MODEL_DIR)
    parser.add_argument("--mono-output-root", type=Path, default=Path("outputs/da3mono_single_depths"))
    parser.add_argument("--comparison-output-root", type=Path, default=Path("outputs/da3_depth_gt_comparison"))
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-mono", action="store_true")
    parser.add_argument(
        "--all-scan-images",
        action="store_true",
        help="Generate/cache mono depth for the whole scan and compare all matched metric npy files.",
    )
    parser.add_argument("--no-max-images", action="store_true", help="When generating all scan images, skip rect_*_max.png.")
    parser.add_argument("--only-light-id", type=int, default=None, help="When generating all scan images, keep only one light id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DA3DepthGTCompareConfig(
        scan_name=args.scan_name,
        view_id=args.view_id,
        light_id=args.light_id,
        depth_dir=args.depth_dir,
        mono_model_dir=args.mono_model_dir,
        metric_depth_root=args.metric_depth_root,
        mono_output_root=args.mono_output_root,
        comparison_output_root=args.comparison_output_root,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        force_mono=args.force_mono,
    )
    if args.all_scan_images:
        mono_result = generate_scan_mono_depths(
            config=config,
            include_max_images=not args.no_max_images,
            light_id=args.only_light_id,
            device=args.device,
        )
        result = run_da3_scan_depth_gt_comparison(config=config)
        df = result["summary_df"]

        with pd.option_context("display.max_columns", None, "display.width", 240):
            print("mono depth summary:")
            print(mono_result["summary_df"].tail(8).to_string(index=False))
            print("\ncomparison summary:")
            print(df.to_string(index=False))
        print(f"\nmono summary csv: {mono_result['summary_csv_path']}")
        print(f"metrics csv: {result['metrics_csv_path']}")
        print(f"summary csv: {result['summary_csv_path']}")
        print(f"overview figure: {result['overview_path']}")
    else:
        result = run_da3_depth_gt_comparison(config=config, device=args.device)
        df = result["metrics_df"]

        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))
        print(f"\nmono depth: {result['mono_depth_path']}")
        print(f"metrics csv: {result['metrics_csv_path']}")
        print(f"comparison figure: {result['figure_path']}")


if __name__ == "__main__":
    main()
