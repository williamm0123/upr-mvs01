"""Fine-tune a DTU-trained MoAMVSNet (moa1) on BlendedMVS (balanced with DTU by default).

    python train_blended.py --name MOA1_BLD_30K --init-from log/experiments/MOA_E15/model/latest.pth \
        --blended-root /scr/user/qinglong/dataset/BlendedMVS_low --max-steps 30000 --lr 1e-4

Same training loop as train_moa.py (train_moa.run); only the config source and
the dataset change:

* The architecture comes from the ``--init-from`` checkpoint's snapshot, not from
  today's defaults — main has since moved to warp 128x4 / RANSAC / moa_gain /
  edge_snap. The three behaviour-only fields that moa1's snapshot predates are
  pinned to moa1's semantics (``--moa1-semantics on``, the default):
      global_solver = huber x3   (moa1 had only the Huber IRLS solver)
      moa_gain      = 1, 1, 1    (no per-stage cap on the mono experts)
      edge_snap     = off        (no hard surface pick at DA3 edges)
  The init is strict: every trainable weight must be in the checkpoint.
* Weights only: a fresh optimizer and a fresh warmup + cosine schedule over
  ``--max-steps``. Requeues resume from this run's own latest.pth as usual.
* Data: data/blended_moa.py, multi-scale crops up to the native 576x768, DA3 from
  a separate cache (scripts/build_da3_cache_mvs.py --dataset blended). Default
  lists are the official BlendedMVS split MVSFormer++ uses (lists/blended/, 106/7
  scenes of the original low-res release, not BlendedMVS+).
* ``--mix-dtu on`` (default) = MVSFormer++'s ``--balanced_training``: each epoch
  takes min(len) samples from DTU-train and from Blended-train and shuffles them
  together (data/blended_moa.BalancedMixDataset). DTU keeps its own 1600 DA3
  cache; both use BLENDED_SCALES so a batch can mix them. Validation stays on
  Blended-val only. ``--mix-dtu off`` = Blended only.
* Metrics: errors are multiplied by each sample's ``metric_scale`` (DTU's depth
  range / the scene's), so ``abs_err`` / ``acc_2mm`` read like DTU's.
  The per-stage ``err_s*_mm`` / ``moa*_err_*_mm`` diagnostics stay in raw scene units.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

import train_moa
from base.config_moa import apply_arch_snapshot
from data.augment import PhotometricAug
from data.blended_moa import BLENDED_SCALES, BalancedMixDataset, MoABlendedDataset
from data.dtu_moa import MoADTUDataset

REPO = Path(__file__).resolve().parent
MOA1_SEMANTICS = {"global_solver": ("huber",) * 3, "moa_gain": (1.0,) * 3, "edge_snap": (False,) * 3}


def parse_args(argv=None):
    p = train_moa.build_parser()
    p.description = __doc__
    p.add_argument("--blended-root", default="/scr/user/qinglong/dataset/BlendedMVS_low")
    p.add_argument("--mix-dtu", choices=["on", "off"], default="on",
                   help="on: balanced DTU + Blended training (MVSFormer++ --balanced_training)")
    p.add_argument("--dtu-da3-root", default=None, help="DTU DA3 cache (default cfg.paths.da3_cache_path)")
    p.add_argument("--dtu-train-list", default=None, help="default cfg.paths.train_list_file")
    p.add_argument("--arch-from", default=None,
                   help="checkpoint whose architecture snapshot builds the net (default: --init-from)")
    p.add_argument("--moa1-semantics", choices=["on", "off"], default="on",
                   help="on: global_solver=huber, moa_gain=1,1,1, edge_snap=off (moa1)")
    p.set_defaults(train_list=str(REPO / "lists/blended/training_list.txt"),
                   val_list=str(REPO / "lists/blended/validation_list.txt"),
                   da3_root=str(REPO / "log/da3_cache_blended"))
    args = p.parse_args(argv)
    if args.warp_channels or args.moa_dim:
        p.error("--warp-channels / --moa-dim come from the checkpoint's architecture here")
    if not (args.arch_from or args.init_from):
        p.error("need --init-from (or --arch-from): the architecture is taken from that checkpoint")
    return args


def build_config(args):
    cfg = train_moa.build_config(args)
    src = args.arch_from or args.init_from
    arch = torch.load(src, map_location="cpu", weights_only=False)["arch"]
    cfg = apply_arch_snapshot(cfg, arch)
    if args.moa1_semantics == "on":
        cfg = dataclasses.replace(cfg, moa=dataclasses.replace(cfg.moa, **MOA1_SEMANTICS))
    cfg = dataclasses.replace(cfg, moa=dataclasses.replace(cfg.moa, enabled=args.moa == "on"),
                              augment=dataclasses.replace(cfg.augment, scales=BLENDED_SCALES))
    m = cfg.moa
    print(f"[blended] mix_dtu={args.mix_dtu}  scales={BLENDED_SCALES}")
    print(f"[blended] arch from {src}: warp={cfg.cascade.warp_channels} depths={cfg.cascade.num_depths} "
          f"moa={'on' if m.enabled else 'off'} global_solver={m.global_solver} "
          f"moa_gain={m.moa_gain} edge_snap={m.edge_snap}")
    return cfg


def build_datasets(cfg, args):
    t, aug = cfg.train, cfg.augment
    common = dict(nviews=t.num_views, seed=t.seed, da3_root=Path(args.da3_root),
                  load_mono=cfg.moa.enabled, da3_missing=t.da3_missing)
    train_kw = dict(
        mode="train",
        aug=PhotometricAug(brightness=aug.brightness, contrast=aug.contrast, saturation=aug.saturation,
                           hue=aug.hue, min_gamma=aug.min_gamma, max_gamma=aug.max_gamma)
        if aug.photometric else None,
        scales=aug.scales if t.multi_scale else (), resize_range=aug.resize_range,
        height=t.height, width=t.width)
    train_ds = MoABlendedDataset(args.blended_root, args.train_list, **train_kw, **common)
    if args.mix_dtu == "on":
        dtu = MoADTUDataset(
            cfg.paths.dtu_train_root, args.dtu_train_list or str(cfg.paths.train_list_file), **train_kw,
            **{**common, "da3_root": Path(args.dtu_da3_root or cfg.paths.da3_cache_path)})
        train_ds = BalancedMixDataset({"dtu": dtu, "blended": train_ds}, seed=t.seed)
    val_ds = MoABlendedDataset(args.blended_root, args.val_list, mode="val", **common)
    if args.max_val_samples and len(val_ds.metas) > args.max_val_samples:
        idx = np.linspace(0, len(val_ds.metas) - 1, args.max_val_samples).round().astype(int)
        val_ds.metas = [val_ds.metas[i] for i in idx]
    return train_ds, val_ds


def main(argv=None) -> None:
    args = parse_args(argv)
    args.init_strict = True
    train_moa.run(args, config_fn=build_config, datasets_fn=build_datasets)


if __name__ == "__main__":
    main()
