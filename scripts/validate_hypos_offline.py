"""Offline stage-1 hypothesis coverage validation — the gate before retraining.

Runs the NEW dual-branch hypothesis construction (models/depth_range.py) on
cached priors + GT, no network, no GPU training, and reports:

  * global-branch coverage  (target: >= 99% of valid pixels)
  * full-axis coverage and GT-to-nearest-bin distance (quantization error)
  * local-branch hit rate, split by prior confidence quartile and edge band
  * the same numbers with prior corruption applied (guard robustness)

Usage:
    python scripts/validate_hypos_offline.py --split val --num-samples 50
    python scripts/validate_hypos_offline.py --split train --corrupt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from base.config import build_mvs_config
from data.dtu import DTUMVSDataset
from data.prior_corruption import corrupt_prior
from models.depth_range import build_stage1_hypotheses


def _stage_res(x: torch.Tensor, hw: tuple[int, int], nearest: bool) -> torch.Tensor:
    mode = "nearest" if nearest else "bilinear"
    kwargs = {} if nearest else {"align_corners": False}
    return F.interpolate(x[None, None].float(), size=hw, mode=mode, **kwargs)[0, 0]


class Accum:
    FIELDS = ("n", "global_inr", "full_inr", "local_hit", "q_err_sum", "prior_err_sum")

    def __init__(self) -> None:
        self.d = {f: 0.0 for f in self.FIELDS}

    def add(self, valid, g_inr, f_inr, l_hit, q_err, p_err) -> None:
        n = float(valid.sum())
        if n == 0:
            return
        self.d["n"] += n
        self.d["global_inr"] += float(g_inr[valid].sum())
        self.d["full_inr"] += float(f_inr[valid].sum())
        self.d["local_hit"] += float(l_hit[valid].sum())
        self.d["q_err_sum"] += float(q_err[valid].sum())
        self.d["prior_err_sum"] += float(p_err[valid].sum())

    def row(self, name: str) -> str:
        n = max(self.d["n"], 1.0)
        return (f"{name:<22} n={int(self.d['n']):>10} "
                f"global_inr={self.d['global_inr'] / n:7.4f} "
                f"full_inr={self.d['full_inr'] / n:7.4f} "
                f"local_hit={self.d['local_hit'] / n:7.4f} "
                f"nearest_bin_mm={self.d['q_err_sum'] / n:7.3f} "
                f"prior_err_mm={self.d['prior_err_sum'] / n:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--corrupt", action="store_true", help="also report with prior corruption applied")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = build_mvs_config()
    listfile = cfg.paths.val_list_file if args.split == "val" else cfg.paths.train_list_file
    ds = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=listfile,
        nviews=cfg.train.num_views,
        mode=args.split,
    )
    if len(ds) == 0:
        raise SystemExit(f"empty dataset for split={args.split} ({listfile})")

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)

    variants = [("clean", False)] + ([("corrupted", True)] if args.corrupt else [])
    accums: dict[str, dict[str, Accum]] = {
        v: {"all": Accum(), "edge": Accum(), "smooth": Accum(),
            "conf_q1": Accum(), "conf_q4": Accum()} for v, _ in variants
    }
    n_out_of_scene = 0
    n_all_valid = 0

    for count, idx in enumerate(idxs):
        sample = ds[int(idx)]
        gt_full = torch.from_numpy(np.ascontiguousarray(sample["depth_gt"])).float()
        mask_full = torch.from_numpy(np.ascontiguousarray(sample["mask"])).float()
        dv = torch.from_numpy(np.ascontiguousarray(sample["depth_values"])).float()
        d_min, d_max = dv.min()[None], dv.max()[None]

        h1, w1 = gt_full.shape[0] // 4, gt_full.shape[1] // 4
        gt = _stage_res(gt_full, (h1, w1), nearest=True)
        valid = _stage_res(mask_full, (h1, w1), nearest=True).bool() & (gt > 0)
        # GT beyond the scene's physical range is unreachable by any in-range
        # hypothesis (DTU backgrounds past depth_values max); loss/metrics
        # exclude it, so coverage is measured over the same denominator.
        in_scene = (gt >= d_min.item()) & (gt <= d_max.item())
        n_out_of_scene += int((valid & ~in_scene).sum())
        n_all_valid += int(valid.sum())
        valid &= in_scene

        for vname, do_corrupt in variants:
            dp = np.asarray(sample["depth_prior"], dtype=np.float32)
            cp = np.asarray(sample["conf_prior"], dtype=np.float32)
            if do_corrupt:
                dp, cp, _ = corrupt_prior(dp, cp, sample_prob=1.0, rng=np.random.default_rng(args.seed + int(idx)))

            s1 = build_stage1_hypotheses(
                torch.from_numpy(dp)[None],
                torch.from_numpy(cp)[None],
                d_min, d_max, cfg.depth_range, target_hw=(h1, w1),
            )
            hy = s1.hypos[0]
            g_inr = (gt >= s1.global_lo.item()) & (gt <= s1.global_hi.item())
            f_inr = (gt >= hy[0]) & (gt <= hy[-1])
            l_hit = (gt >= s1.local_lo[0]) & (gt <= s1.local_hi[0])
            q_err = (hy - gt[None]).abs().amin(dim=0)
            p_err = (s1.prior[0] - gt).abs()

            a = accums[vname]
            a["all"].add(valid, g_inr, f_inr, l_hit, q_err, p_err)
            edge = s1.edge[0] > 0.5
            a["edge"].add(valid & edge, g_inr, f_inr, l_hit, q_err, p_err)
            a["smooth"].add(valid & ~edge, g_inr, f_inr, l_hit, q_err, p_err)
            conf = s1.conf[0]
            a["conf_q1"].add(valid & (conf < 0.25), g_inr, f_inr, l_hit, q_err, p_err)
            a["conf_q4"].add(valid & (conf >= 0.75), g_inr, f_inr, l_hit, q_err, p_err)

        if (count + 1) % 10 == 0:
            print(f"  processed {count + 1}/{len(idxs)} samples")

    for vname, _ in variants:
        print(f"\n===== {args.split} / {vname} =====")
        for bucket in ("all", "smooth", "edge", "conf_q1", "conf_q4"):
            print(accums[vname][bucket].row(bucket))
    print(f"\nGT outside physical scene range (excluded, unreachable): "
          f"{n_out_of_scene}/{n_all_valid} = {n_out_of_scene / max(n_all_valid, 1):.4f}")
    print("targets: global_inr >= 0.99 in EVERY bucket (incl. corrupted); "
          "nearest_bin_mm well below the stage-2 capture half-range")


if __name__ == "__main__":
    main()
