# UPR-MVS Jupyter Experiments

This folder now contains project code extracted from `upr-mvs.ipynb`.

## Layout

- `data/`: non-test data loading and file I/O code.
  - `data/dtu.py`: DTU sample loading and path helpers.
  - `data/io.py`: PFM/camera/pair-file readers.
- `models/dinov3/`: the trimmed DINOv3 code used by this project.
  - `models/dinov3/vision_transformer.py`: minimal ViT-B/16 implementation for 12-layer feature extraction.
  - `models/dinov3/extractor.py`: DINOv3 weight loading, feature extraction, projection, resizing.
- `experiments/`: step-by-step test modules extracted from the notebook.
  - `experiments/cost_volume.py`: homography warping based variance cost volume, soft argmin, metrics.
  - `experiments/adapter.py`: `GeometryAdapter` and its single-sample training loss.
  - `experiments/fpn.py`: Conv2d C2-C5 FPN feature visualization.
  - `experiments/fpn_cost_volume.py`: l=2 FPN-only matching head cost-volume and depth regression test.
  - `experiments/pointmvs_coarse.py`: PointMVSNet-style coarse depth prediction using FPN features.
  - `experiments/pointmvs_source_coarse.py`: original PointMVSNet `ImageConv` + `VolumeConv` coarse depth visualization test.
  - `experiments/dino_fpn_fusion.py`: DINOv3 layer 3/7/11 concat compression plus four-level FPN fusion visualization.
  - `experiments/dino_fpn_cost_volume.py`: l=2 fused FPN+DINO feature cost-volume and depth regression test.
  - `experiments/depth_anything_v3.py`: local DA3MONO-LARGE loading and one-image-per-test-scan depth visualization.
  - `experiments/da3_scan_pointcloud.py`: raw DA3 depth projection over a full DTU scan.
  - `experiments/runners.py`: reusable experiment entry points.
  - `experiments/validation.py`: lightweight sample/view/depth alignment checks.
- `upr_mvs/`: shared project configuration and bridges to the original UPR-MVS utilities.
  - `upr_mvs/config.py`: default paths and experiment config dataclasses.
  - `upr_mvs/external.py`: explicit loader for upstream UPR-MVS transformer utilities.
- `scripts/check_sample_geometry.py`: print the geometry/input sanity report.
- `scripts/run_dinov3_cost_volume_comparison.py`: run RGB baseline plus DINOv3 layer comparison.
- `scripts/run_geometry_adapter_test.py`: train/evaluate the GeometryAdapter test.
- `scripts/run_adapter_ablation_test.py`: fixed-module ablation for raw DINO and adapter variants, including PLY point-cloud export.
- `scripts/visualize_fpn_features.py`: visualizes a Conv2d C2-C5 backbone plus lateral/smooth FPN on the first DTU sample.
- `scripts/run_fpn_p2_cost_volume.py`: builds a cost volume from FPN-only l=2/P2 matching features.
- `scripts/run_pointmvs_coarse.py`: runs the PointMVSNet-style coarse depth module and visualizes FPN/depth outputs.
- `scripts/run_pointmvs_source_coarse.py`: runs the original PointMVSNet source coarse module for comparison.
- `scripts/visualize_dino_fpn_fusion.py`: visualizes DINOv3 internal fusion and FPN+DINO fusion at P2-P5.
- `scripts/run_dino_fpn_p2_cost_volume.py`: builds a cost volume from the fused l=2/P2 feature and regresses depth.
- `scripts/visualize_da3_test_scans.py`: runs Depth Anything 3 on one DTU image for every scan in `lists/dtu/test.txt`.
- `scripts/run_da3_first_scan_pointcloud.py`: runs DA3 on every image in the first test scan and exports an unoptimized PLY point cloud.
- `scripts/visualize_npy.py`: visualizes scalar/RGB `.npy` files or directories.

## Example Commands

```bash
python scripts/check_sample_geometry.py --sample-index 0
python scripts/run_dinov3_cost_volume_comparison.py --sample-index 0
python scripts/run_geometry_adapter_test.py --sample-index 0 --steps 30
python scripts/run_adapter_ablation_test.py --sample-index 0 --steps 30
python scripts/visualize_fpn_features.py --sample-index 0
python scripts/run_fpn_p2_cost_volume.py --sample-index 0
python scripts/run_pointmvs_coarse.py --sample-index 0
python scripts/run_pointmvs_source_coarse.py --sample-index 0
python scripts/visualize_dino_fpn_fusion.py --sample-index 0
python scripts/run_dino_fpn_p2_cost_volume.py --sample-index 0
python scripts/visualize_da3_test_scans.py
python scripts/run_da3_first_scan_pointcloud.py --point-stride 8
python scripts/visualize_npy.py outputs/depth_anything_v3_test_scans/scan1/scan1_depth.npy
```

The adapter ablation script writes prediction and GT point clouds under
`outputs/adapter_ablation/<sample_name>/pointclouds/` by default.

The default paths match the notebook. Override them with CLI flags or environment variables:

- `UPR_MVS_PROJECT_ROOT`
- `DTU_TRAIN_ROOT`
- `DTU_TEST_ROOT`
- `DINOV3_WEIGHTS_FILE`
