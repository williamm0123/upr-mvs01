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

## Example Commands

```bash
python scripts/check_sample_geometry.py --sample-index 0
python scripts/run_dinov3_cost_volume_comparison.py --sample-index 0
python scripts/run_geometry_adapter_test.py --sample-index 0 --steps 30
python scripts/run_adapter_ablation_test.py --sample-index 0 --steps 30
python scripts/visualize_fpn_features.py --sample-index 0
```

The adapter ablation script writes prediction and GT point clouds under
`outputs/adapter_ablation/<sample_name>/pointclouds/` by default.

The default paths match the notebook. Override them with CLI flags or environment variables:

- `UPR_MVS_PROJECT_ROOT`
- `DTU_TRAIN_ROOT`
- `DTU_TEST_ROOT`
- `DINOV3_WEIGHTS_FILE`
