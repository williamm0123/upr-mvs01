from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from base.config import ProjectPaths, build_mvs_config
from data.dtu import DTUMVSDataset
import data.norm_fill as norm_fill
import data.camera_utils as C
from models.depth_range import initial_range_from_prior
from models.sfm import SfMConfig, generate_sparse_depth_from_sample
from data.norm_fill import _tensor_to_uint8_hwc

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def select_first_ref_per_scan_metas(metas):
    selected = []
    seen_scans = set()

    for scan, light_idx, ref_view, src_views in metas:
        if scan in seen_scans:
            continue

        selected.append((scan, light_idx, ref_view, src_views))
        seen_scans.add(scan)

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["local", "umhpc"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-views", type=int, default=None)
    parser.add_argument("--max-scans", type=int, default=0)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--image-mode", choices=["resize", "crop", "pad"], default="resize")
    parser.add_argument("--conf-percentile", type=float, default=10.0)
    parser.add_argument("--target-w", type=int, default=518)
    parser.add_argument("--target-h", type=int, default=420)
    parser.add_argument("--sfm-only", action="store_true")
    parser.add_argument("--sfm-max-features", type=int, default=8000)
    parser.add_argument("--sfm-ratio", type=float, default=0.75)
    parser.add_argument("--sfm-max-reproj-error", type=float, default=2.0)
    return parser.parse_args()




def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = ProjectPaths()
    dataset = DTUMVSDataset(
        datapath=paths.dtu_train_root,
        listfile=paths.dtu_list_path,
        nviews=args.num_views or 5,
        ndepths=192,
        mode="test",
        resize_scale=0.5,
    )
    dataset.metas = select_first_ref_per_scan_metas(dataset.metas)

    for i, sample in enumerate(dataset):
        if i>0:
            break

        scan, _light_idx, _ref_view, _src_views = dataset.metas[i]
        print("-" * 60)
        print("scan number:", scan)

        out = norm_fill.generate_priors_from_sample(
            sample,
            device,
            args.image_mode,
            conf_percentile=args.conf_percentile,
            image_target_wh=(args.target_w, args.target_h),
        )
        depth_filled = out["depth_filled"]
        depth_gt = sample["depth_gt"]
        intrinsics = sample["intrinsics"][0]
        extrinsics = sample["extrinsics"][0]
        depth_filled = C.backproject_depth_to_world_points(depth_filled, intrinsics, extrinsics)
        depth_gt = C.backproject_depth_to_world_points(depth_gt, intrinsics, extrinsics)

        colors_filled = np.tile(np.array([153, 204, 255], dtype=np.uint8), (depth_filled.shape[0], 1))
        colors_gt = np.tile(np.array([255, 255, 255], dtype=np.uint8), (depth_gt.shape[0], 1))
        merged_points = np.concatenate([depth_filled, depth_gt], axis=0)
        merged_colors = np.concatenate([colors_filled, colors_gt], axis=0)
        C.save_pointcloud_ply(merged_points, f"outputs/depth_hypo/{scan}_merged_pointcloud.ply", colors=merged_colors)
if __name__ == "__main__":
    main()
