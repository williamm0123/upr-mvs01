#!/usr/bin/env python
"""Print the notebook-style sample/input alignment checks."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dtu import build_dtu_dataset
from experiments.validation import build_sample_alignment_report, print_sample_alignment_report
from upr_mvs.config import DTUConfig, ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--dtu-root", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--light-id", type=int, default=3)
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-dir", default="Rectified_raw")
    parser.add_argument("--depth-dir", default="Depths_raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    if args.dtu_root is not None:
        paths = replace(paths, dtu_train_root=args.dtu_root)
    if args.list_file is not None:
        paths = replace(paths, dtu_list_path=args.list_file)
    dtu_config = DTUConfig(
        n_views=args.n_views,
        light_id=args.light_id,
        split=args.split,
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
    )
    dataset = build_dtu_dataset(paths=paths, config=dtu_config)
    sample = dataset[args.sample_index]
    report = build_sample_alignment_report(dataset, sample, args.sample_index)
    print_sample_alignment_report(report)


if __name__ == "__main__":
    main()
