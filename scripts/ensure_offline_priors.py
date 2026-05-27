from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base.config import build_mvs_config
from data.prior_precompute import ensure_offline_priors


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate missing offline VGGT+DA3 prior depth files.")
    parser.add_argument("--profile", choices=["local", "umhpc"], default=None)
    parser.add_argument("--device", default=None, help="torch device, for example cuda:0 or cpu")
    parser.add_argument("--dry-run", action="store_true", help="only report missing files")
    parser.add_argument("--max-groups", type=int, default=None, help="debug limit for generated ref-view groups")
    args = parser.parse_args()

    cfg = build_mvs_config(profile=args.profile)
    if args.max_groups is not None:
        cfg = replace(
            cfg,
            vggt_prior=replace(cfg.vggt_prior, offline_generation_max_groups=args.max_groups),
        )

    device = torch.device(args.device) if args.device is not None else None
    ensure_offline_priors(cfg, device=device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
