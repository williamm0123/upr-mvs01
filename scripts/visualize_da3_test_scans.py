#!/usr/bin/env python
"""Visualize one DA3MONO-LARGE depth prediction for each scan in test.txt."""

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

from experiments.depth_anything_v3 import (  # noqa: E402
    DA3VisualizationConfig,
    run_da3_test_scan_visualization,
)
from upr_mvs.config import DEFAULT_DA3_MONO_MODEL_DIR, ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/depth_anything_v3_test_scans"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtu-root", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DA3_MONO_MODEL_DIR)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--view-id", type=int, default=0)
    parser.add_argument("--light-id", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    if args.dtu_root is not None:
        paths = replace(paths, dtu_train_root=args.dtu_root)
    if args.list_file is not None:
        paths = replace(paths, dtu_list_path=args.list_file)

    config = DA3VisualizationConfig(
        model_dir=args.model_dir,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        view_id=args.view_id,
        light_id=args.light_id,
    )
    result = run_da3_test_scan_visualization(
        paths=paths,
        config=config,
        output_root=args.output_root,
        device=args.device,
    )

    print("DA3 load info:", result["load_info"])
    print(result["summary_df"].to_string(index=False))
    print("summary csv:", result["summary_csv_path"])
    print("overview:", result["overview_path"])
    print("output root:", result["output_root"])


if __name__ == "__main__":
    main()
