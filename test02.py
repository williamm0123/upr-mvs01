from __future__ import annotations

import argparse
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np


from data.dtu import DTUDataset
from base.config import ProjectPaths
import models.sfm as sfm
import models.depth_fill2 as depth_fill
import models.general as G
import models.visual_tools as V

from PIL import Image
from models.sam3.sam3.model_builder import build_sam3_image_model
from models.sam3.sam3.model.sam3_image_processor import Sam3Processor


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

    parser.add_argument("--sfm-output-root", type=Path, default=Path("outputs/depth_fill2_test02"))
    parser.add_argument("--max-image-size", type=int, default=1200)
    parser.add_argument("--max-num-features", type=int, default=8192)
    parser.add_argument("--max-ratio", type=float, default=0.8)
    parser.add_argument("--max-view-gap", type=int, default=8)
    parser.add_argument("--min-pair-matches", type=int, default=30)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=2000.0)
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument("--min-tri-angle", type=float, default=1.0)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--gpu-sfm", action="store_true")
    parser.add_argument("--clean-sfm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--depthfill3-output-root", type=Path, default=Path("outputs/depth fill3 test01"))
    parser.add_argument("--depthfill3-w-data", type=float, default=1.0)
    parser.add_argument("--depthfill3-w-grad", type=float, default=10.0)
    parser.add_argument("--depthfill3-w-smooth", type=float, default=0.1)
    parser.add_argument("--depthfill3-edge-alpha", type=float, default=10.0)
    parser.add_argument("--depthfill3-max-solver-pixels", type=int, default=350000)
    parser.add_argument("--depthfill3-lsqr-iter-lim", type=int, default=600)
    parser.add_argument("--depthfill3-save-pointcloud", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()

def main():
    args = parse_args()
    paths = ProjectPaths()

    dataset = DTUDataset(
        datapath=paths.dtu_train_root,
        listfile=paths.dtu_list_path,
        nviews=5,
        ndepths=192,
        mode="test",
        resize_scale=1.0,
    )
    dataset.metas = select_first_ref_per_scan_metas(dataset.metas)
    print("selected first-ref metas:", len(dataset.metas))

    # sample = dataset[0]

    sfm_config = sfm.SFMConfig(
        output_root=paths.project_path / args.sfm_output_root,
        max_image_size=args.max_image_size,
        max_num_features=args.max_num_features,
        max_ratio=args.max_ratio,
        max_view_gap=args.max_view_gap,
        min_pair_matches=args.min_pair_matches,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_reproj_error=args.max_reproj_error,
        min_tri_angle=args.min_tri_angle,
        voxel_size=args.voxel_size,
        gpu=args.gpu_sfm,
        clean=args.clean_sfm,
    )
    print("-"*60)
    output_path = paths.project_path / args.depthfill3_output_root

    model_da3, device = depth_fill.load_da3_model(paths.da3_weights_file)
    for i,sample in enumerate(dataset):
        if args.max_samples > 0 and i >= args.max_samples:
            break
        

        # Load the model
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        # Load an image
        image = Image.open("<YOUR_IMAGE_PATH.jpg>")
        inference_state = processor.set_image(image)
        # Prompt the model with text
        output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

        # Get the masks, bounding boxes, and scores
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

                
        scan, light_idx, ref_view, _ = dataset.metas[i]
        output_name = f"{scan}_ref{ref_view:03d}_light{light_idx}"
      
        depth_da3 = depth_fill.generate_da3_depth_maps(sample["images"][0],
                                                    da3_model=model_da3)

        print("-"*60)
        
        
if __name__ == "__main__":
    main()
