#!/usr/bin/env python
"""Visualize DINOv3 layer fusion and four-level FPN+DINO fusion."""

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

from experiments.dino_fpn_fusion import DinoFPNFusionConfig, run_dino_fpn_fusion_visualization
from upr_mvs.config import DTUConfig, ProjectPaths


def parse_layer_numbers(value: str) -> tuple[int, ...]:
    layer_numbers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(layer_numbers) != 3:
        raise argparse.ArgumentTypeError("Please provide exactly three DINO layer numbers, for example 3,7,11")
    return layer_numbers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dino_fpn_fusion"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtu-root", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--dinov3-weights", type=Path, default=None)
    parser.add_argument(
        "--max-side",
        type=int,
        default=768,
        help="Resize the reference image before both FPN and DINO. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--dino-input-max-side",
        type=int,
        default=0,
        help="Optional DINO-only input max side after image resize. 0 reuses the current image size.",
    )
    parser.add_argument("--dino-layers", type=parse_layer_numbers, default=(3, 7, 11))
    parser.add_argument("--fpn-channels", type=int, default=16)
    parser.add_argument("--dino-fused-channels", type=int, default=16)
    parser.add_argument("--fused-channels", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    if args.dtu_root is not None:
        paths = replace(paths, dtu_train_root=args.dtu_root)
    if args.list_file is not None:
        paths = replace(paths, dtu_list_path=args.list_file)
    if args.dinov3_weights is not None:
        paths = replace(paths, dinov3_weights_file=args.dinov3_weights)

    fusion_config = DinoFPNFusionConfig(
        dino_layer_numbers=args.dino_layers,
        max_side=args.max_side,
        dino_input_max_side=args.dino_input_max_side,
        fpn_channels=args.fpn_channels,
        dino_fused_channels=args.dino_fused_channels,
        fused_channels=args.fused_channels,
        patch_size=args.patch_size,
    )

    result = run_dino_fpn_fusion_visualization(
        sample_index=args.sample_index,
        paths=paths,
        dtu_config=DTUConfig(),
        fusion_config=fusion_config,
        output_root=args.output_root,
        device=args.device,
    )

    print("sample_name:", result["sample_name"])
    print("scan_name:", result["scan_name"])
    print("ref_view:", result["ref_view"])
    print("device:", result["device"])
    print("image shape:", result["image_shape"])
    print("DINO layers:", result["dino_layer_numbers"])
    print("DINO input hw:", result["dino_input_hw"])
    print("DINO native feature hw:", result["dino_native_feature_hw"])
    print("DINO layer shapes:")
    for name, shape in result["dino_layer_shapes"].items():
        print(f" - {name}: {shape}")
    print("DINO fused shape:", result["dino_fused_shape"])
    print("FPN shapes:")
    for level, shape in result["fpn_shapes"].items():
        print(f" - {level}: {shape}")
    print("FPN + DINO fused shapes:")
    for level, shape in result["fused_shapes"].items():
        print(f" - {level}: {shape}")
    print("outputs:")
    for name, path in result["paths"].items():
        print(f" - {name}: {path}")


if __name__ == "__main__":
    main()
