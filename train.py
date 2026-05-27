from __future__ import annotations

import argparse
import os
from dataclasses import replace

from base.config import build_mvs_config
from engine.trainer import Trainer


def _resolve_distributed(flag: str, profile_default: bool) -> bool:
    if flag == "on":
        return True
    if flag == "off":
        return False
    if "LOCAL_RANK" in os.environ:
        return True
    return profile_default


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UprMVSNet on DTU")
    parser.add_argument("--profile", choices=["local", "umhpc"], default=None,
                        help="hyperparameter preset (see base/config.py)")
    parser.add_argument("--ddp", choices=["on", "off", "auto"], default="auto",
                        help="DDP switch. auto: follow profile / detect torchrun env")
    parser.add_argument("--name", default="uprmvs")
    args = parser.parse_args()

    cfg = build_mvs_config(profile=args.profile)
    use_ddp = _resolve_distributed(args.ddp, cfg.train.distributed)
    if use_ddp != cfg.train.distributed:
        cfg = type(cfg)(
            paths=cfg.paths,
            data=cfg.data,
            dino=cfg.dino,
            fpn=cfg.fpn,
            vggt_prior=cfg.vggt_prior,
            geo_fusion=cfg.geo_fusion,
            depth_range=cfg.depth_range,
            cost_volume=cfg.cost_volume,
            anchor_pe=cfg.anchor_pe,
            points_alignment=cfg.points_alignment,
            decoder=cfg.decoder,
            loss=cfg.loss,
            stage_weights=cfg.stage_weights,
            train=replace(cfg.train, distributed=use_ddp),
        )

    trainer = Trainer(cfg, run_name=args.name)
    trainer.train()


if __name__ == "__main__":
    main()
