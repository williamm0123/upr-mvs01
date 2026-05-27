from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from base.config import build_mvs_config
from data.dtu import DTUMVSDataset
from engine.evaluator import evaluate
from engine.trainer import collate_batch
from models.mvsnet import UprMVSNet
from utils.logging_utils import dump_metrics, get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--profile", choices=["local", "umhpc"], default=None)
    parser.add_argument("--listfile", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = build_mvs_config(profile=args.profile)
    paths = cfg.paths
    logger = get_logger("eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UprMVSNet(cfg, device=device).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["model"], strict=False)
    logger.info(f"loaded checkpoint from {args.ckpt}")

    list_path = Path(args.listfile) if args.listfile else paths.test_list_file
    prior_root = paths.offline_prior_root if cfg.vggt_prior.prior_source in ("offline", "auto") else None
    dataset = DTUMVSDataset(
        datapath=paths.dtu_test_root,
        listfile=list_path,
        nviews=cfg.train.num_views,
        target_h=cfg.data.target_h,
        target_w=cfg.data.target_w,
        feature_strides=cfg.data.feature_strides,
        mode="test",
        use_pair_filter=cfg.data.use_pair_filter,
        prior_root=prior_root,
        prior_confidence=cfg.vggt_prior.offline_confidence,
        require_prior=cfg.vggt_prior.offline_prior_required,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_batch)
    metrics = evaluate(model, loader, device, max_batches=None)
    logger.info("metrics: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    out_path = Path(args.out) if args.out else Path(args.ckpt).with_suffix(".metrics.json")
    dump_metrics(out_path, metrics)
    logger.info(f"saved metrics to {out_path}")


if __name__ == "__main__":
    main()
