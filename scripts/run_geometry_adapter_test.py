#!/usr/bin/env python
"""Train and evaluate the lightweight DINOv3 GeometryAdapter test."""

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

from experiments.runners import run_geometry_adapter_test
from upr_mvs.config import AdapterConfig, CostVolumeConfig, DINOConfig, DTUConfig, ProjectPaths


def parse_layer_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/geometry_adapter"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--num-depths", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--channel-chunk", type=int, default=4)
    parser.add_argument("--dino-input-max-side", type=int, default=384)
    parser.add_argument("--dino-project-channels", type=int, default=32)
    parser.add_argument("--dino-layers", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--adapter-layers", default="1,6,12")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--hidden-ch", type=int, default=128)
    parser.add_argument("--out-ch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--l1-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-grid-divisor", type=int, default=2)
    parser.add_argument("--dtu-root", type=Path, default=None)
    parser.add_argument("--list-file", type=Path, default=None)
    parser.add_argument("--upr-mvs-root", type=Path, default=None)
    parser.add_argument("--dinov3-weights", type=Path, default=None)
    return parser.parse_args()


def parse_dino_layers(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) - 1 for item in value.split(",") if item.strip())


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    if args.dtu_root is not None:
        paths = replace(paths, dtu_train_root=args.dtu_root)
    if args.list_file is not None:
        paths = replace(paths, dtu_list_path=args.list_file)
    if args.upr_mvs_root is not None:
        paths = replace(paths, upr_mvs_root=args.upr_mvs_root)
    if args.dinov3_weights is not None:
        paths = replace(paths, dinov3_weights_file=args.dinov3_weights)

    cost_config = CostVolumeConfig(
        scale=args.scale,
        num_depths=args.num_depths,
        temperature=args.temperature,
        channel_chunk=args.channel_chunk,
    )
    dino_config = DINOConfig(
        input_max_side=args.dino_input_max_side,
        layers=parse_dino_layers(args.dino_layers),
        project_channels=args.dino_project_channels,
    )
    adapter_config = AdapterConfig(
        layer_ids=parse_layer_ids(args.adapter_layers),
        hidden_ch=args.hidden_ch,
        out_ch=args.out_ch,
        train_steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ce_weight=args.ce_weight,
        l1_weight=args.l1_weight,
        grad_clip=args.grad_clip,
        train_grid_divisor=args.train_grid_divisor,
    )

    result = run_geometry_adapter_test(
        sample_index=args.sample_index,
        paths=paths,
        dtu_config=DTUConfig(),
        cost_config=cost_config,
        dino_config=dino_config,
        adapter_config=adapter_config,
        output_root=args.output_root,
        device=args.device,
    )
    print(result["metrics_df"].to_string(index=False))
    print("history csv:", result["history_csv_path"])
    print("metrics csv:", result["metrics_csv_path"])
    print("summary plot:", result["summary_plot_path"])


if __name__ == "__main__":
    main()
