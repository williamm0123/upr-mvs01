"""Batch-0 measurement driver: MVSFormer++-comparable metrics + mode_window ablation.

Answers the three questions the training logs cannot:

1. **What is our body precision?**  ``body_n`` = mean |err| over pixels already
   within n mm — MVSFormer++'s ``abs_depth_thres0-{n}mm_error``.  Their DTU-test
   number is 0.4920 mm (<2mm) / 0.6119 mm (<4mm).  We had never logged this, so
   every "refinement gap" claim so far rested on an assumption.

2. **What do we score under *their* protocol?**  ``e_n`` = fraction of masked
   pixels with |err| > n mm — exactly ``1 - acc_nmm``, and exactly their
   ``thres{n}mm_error``.  Their eval is DTU *test* at 1152x1536 with **no
   in-scene mask filter**; ``--no-in-scene`` removes ours so the rulers match.

3. **Is ``mode_window`` costing us sub-bin precision?**  ``--mode-window 1,2,3``
   re-runs the stage-3 mode-centred regression from the *same* forward pass, so
   scanning several values costs one inference.  (Stage 1/2 windows change the
   hypothesis axes downstream, so a strict full-cascade sweep needs
   ``--mw-full-cascade``, which re-runs inference per value.)

Model/data plumbing is imported from ``test.py`` — this script only changes what
is measured, never how the network runs.

Examples
--------
# protocol-aligned baseline (the number that actually compares to MVSFormer++)
python eval_ablation.py --split test --full-image --resize-scale 0.96 --no-in-scene

# zero-cost sub-bin ablation on the val split
python eval_ablation.py --split val --mode-window 1,2,3

# strict (expensive) full-cascade version of the same ablation
python eval_ablation.py --split val --mode-window 1,2,3 --mw-full-cascade
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base.config import build_mvs_config
from models.depth_range import mode_centered_regression
from test import _collate, build_dataset, ensure_priors, load_model

# MVSFormer++ local reproduction, DTU test @1152x1536, epoch 13 (best) — the bar.
# Source: reference/mvsformer++forCD traininglog, config/mvsformer++.json.
MVSFORMERPP_REF = {
    "e2": 14.477, "e4": 9.854, "e8": 7.396, "e14": 6.129,
    "body2": 0.4920, "body4": 0.6119, "body8": 0.7492, "body14": 0.8867,
}
MVSFORMERPP_PAPER = {"e2": 12.41, "e4": 7.90, "e8": 5.69}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("UprMVSNet batch-0 measurement / ablation")
    # --- passthrough to test.py's dataset+model builders (same semantics) ---
    p.add_argument("--profile", choices=["local", "umhpc"], default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt", default=None, help="explicit checkpoint file; overrides --ckpt-dir")
    p.add_argument("--ckpt-dir", default="log/model_eval")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--list", default=None)
    p.add_argument("--num-views", type=int, default=None)
    p.add_argument("--resize-scale", type=float, default=0.5,
                   help="0.96 + --full-image gives 1152x1536, MVSFormer++'s eval resolution")
    p.add_argument("--full-image", action="store_true")
    p.add_argument("--prior-target-w", type=int, default=None)
    p.add_argument("--prior-target-h", type=int, default=None)
    p.add_argument("--max-scans", type=int, default=0)
    p.add_argument("--max-refs", type=int, default=0, help="limit ref views per scan (0 = all 49)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--build-priors", choices=["auto", "skip", "force"], default="auto")
    p.add_argument("--spre", choices=["auto", "on", "off"], default="auto",
                   help="SPRE DINOv3 prior-reliability head: 'auto' (default) mirrors the checkpoint")
    # --- what this script adds ---
    p.add_argument("--no-in-scene", action="store_true",
                   help="drop the in-scene GT filter from the metric mask. MVSFormer++ has no such "
                        "filter, so this is REQUIRED for a like-for-like e_n comparison. The fraction "
                        "it removes is reported either way.")
    p.add_argument("--mode-window", default=None,
                   help="comma list of stage-3 mode_window values to score, e.g. '1,2,3'. "
                        "Default: the configured value only.")
    p.add_argument("--mw-full-cascade", action="store_true",
                   help="apply each mode_window to ALL stages and re-run inference (strict but "
                        "N x slower). Without it only stage 3 is rescanned, reusing one forward pass.")
    p.add_argument("--out", default=None, help="output json (default outputs/ablation_<split>/metrics.json)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class ErrorMeter:
    """Pixel-weighted MVSFormer++-comparable depth metrics, per scan and overall.

    ``e_n``    = 100 * fraction(|err| >  n mm)   == their ``thres{n}mm_error``
                                                 == 100 * (1 - our acc_nmm)
    ``body_n`` = mean |err| over pixels with |err| <= n mm
                                                 == their ``abs_depth_thres0-{n}mm_error``

    ``body_n`` is the number the training logs never had: it isolates precision
    on the pixels that already match, independent of the outlier tail that
    dominates the plain mean.
    """

    THRESH = (1.0, 2.0, 4.0, 8.0, 14.0)
    # slots: [err_sum, n] + per threshold [over_count, body_sum, body_count]
    N_SLOTS = 2 + 3 * len(THRESH)

    def __init__(self) -> None:
        self.sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.N_SLOTS, dtype=np.float64))
        self.pool: dict[str, list[np.ndarray]] = defaultdict(list)

    def update(self, scan: str, err: torch.Tensor) -> None:
        e = err.detach().float()
        if e.numel() == 0:
            return
        s = self.sums[scan]
        s[0] += e.sum().item()
        s[1] += e.numel()
        for i, t in enumerate(self.THRESH):
            keep = e <= t
            s[2 + 3 * i] += (~keep).sum().item()
            s[3 + 3 * i] += e[keep].sum().item()
            s[4 + 3 * i] += keep.sum().item()
        # subsample for quantiles (~4k values per batch keeps memory flat)
        self.pool[scan].append(e[:: max(e.numel() // 4096, 1)].cpu().numpy())

    @classmethod
    def _finish(cls, s: np.ndarray, pool: np.ndarray) -> dict[str, float]:
        n = max(s[1], 1.0)
        out: dict[str, float] = {
            "pixels": int(s[1]),
            "abs_err": s[0] / n,
            "median": float(np.median(pool)),
            "p90": float(np.percentile(pool, 90)),
        }
        for i, t in enumerate(cls.THRESH):
            tag = f"{t:g}"
            over, bsum, bn = s[2 + 3 * i], s[3 + 3 * i], s[4 + 3 * i]
            out[f"e{tag}"] = 100.0 * over / n           # % of pixels worse than t mm
            out[f"acc_{tag}mm"] = bn / n                # our historical convention
            out[f"body{tag}"] = bsum / max(bn, 1.0)     # mean err among pixels within t mm
        # the >8mm tail, reported directly instead of being back-solved
        body8_sum, body8_n = s[3 + 3 * 3], s[4 + 3 * 3]
        tail_n = max(s[1] - body8_n, 0.0)
        out["tail8_frac"] = 100.0 * tail_n / n
        out["tail8_mean"] = (s[0] - body8_sum) / max(tail_n, 1.0)
        out["tail8_share"] = 100.0 * (s[0] - body8_sum) / max(s[0], 1e-9)
        return out

    def scan_metrics(self, scan: str) -> dict[str, float]:
        pool = np.concatenate(self.pool[scan]) if self.pool[scan] else np.zeros(1)
        return self._finish(self.sums[scan], pool)

    def overall(self) -> dict[str, float]:
        if not self.sums:
            return {}
        tot = np.sum(list(self.sums.values()), axis=0)
        pool = np.concatenate([v for vs in self.pool.values() for v in vs]) if self.pool else np.zeros(1)
        return self._finish(tot, pool)


def build_metric_mask(batch: dict, use_in_scene: bool) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Return (gt, mask, kept_px, dropped_px).

    ``dropped_px`` counts pixels the in-scene filter removes — GT outside
    [depth_min, depth_max], unreachable by any hypothesis. We normally exclude
    them; MVSFormer++ does not, so the count quantifies that protocol gap even
    when the filter stays on.
    """
    gt = batch["depth_gt"].float()
    base = batch["mask"].bool() & (gt > 0)
    dv = batch["depth_values"].float()
    in_scene = (gt >= dv.amin(dim=1).view(-1, 1, 1)) & (gt <= dv.amax(dim=1).view(-1, 1, 1))
    dropped = float((base & ~in_scene).sum().item())
    mask = (base & in_scene) if use_in_scene else base
    return gt, mask, float(base.sum().item()), dropped


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, ds, cfg, args, device, windows: list[int]) -> tuple[dict[int, ErrorMeter], dict]:
    """One pass over the split, scoring every window in ``windows``.

    The extra windows are free: ``mode_centered_regression`` only needs stage 3's
    posterior and hypothesis axis, both already in ``outputs``. Stage 1/2 keep
    the model's configured window (see ``--mw-full-cascade`` for the strict
    variant).
    """
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=_collate, pin_memory=True)
    use_amp = cfg.train.amp and device.type == "cuda"
    meters = {w: ErrorMeter() for w in windows}
    native_w = model.decoders[-1].mode_window
    stats = {"masked_px": 0.0, "oos_px": 0.0, "samples": 0}

    for i, batch in enumerate(loader):
        scan = ds.metas[i][0]
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)

        gt, m, base_px, oos_px = build_metric_mask(batch, use_in_scene=not args.no_in_scene)
        stats["masked_px"] += base_px
        stats["oos_px"] += oos_px
        stats["samples"] += 1
        if not m.any():
            continue

        full_hw = batch["images"].shape[-2:]
        s3 = outputs["stage3"]
        for w in windows:
            if w == native_w:
                pred = outputs["depth_full"].float()
            else:
                d3, _, _ = mode_centered_regression(s3["prob"].float(), s3["depth_hypos"].float(), w)
                pred = F.interpolate(d3.unsqueeze(1), size=full_hw,
                                     mode="bilinear", align_corners=False).squeeze(1)
            meters[w].update(scan, (pred[m] - gt[m]).abs())

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(ds)}] {scan}", flush=True)

    return meters, stats


def print_report(results: dict[int, dict], per_scan: dict[str, dict], stats: dict,
                 args, eval_hw: tuple[int, int], native_w: int) -> None:
    oos_pct = 100.0 * stats["oos_px"] / max(stats["masked_px"], 1.0)
    print("\n" + "=" * 100)
    print(f"split={args.split}  eval_res={eval_hw[0]}x{eval_hw[1]}  "
          f"in_scene_filter={'OFF' if args.no_in_scene else 'ON'}  "
          f"cascade={'full' if args.mw_full_cascade else 'stage3-only'}")
    print(f"GT-out-of-scene pixels: {oos_pct:.2f}% of the DTU mask "
          f"({'counted as error — matches MVSFormer++' if args.no_in_scene else 'EXCLUDED — MVSFormer++ does not exclude them'})")
    print("=" * 100)

    hdr = (f"{'mw':>4} | {'e2':>6} {'e4':>6} {'e8':>6} {'e14':>6} | "
           f"{'body2':>6} {'body4':>6} {'body8':>6} | "
           f"{'mean':>7} {'med':>6} {'p90':>7} | {'tail>8':>7} {'tailmm':>7} {'share':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for w in sorted(results):
        r = results[w]
        star = " *" if w == native_w else "  "
        print(f"{w:>2}{star} | {r['e2']:6.2f} {r['e4']:6.2f} {r['e8']:6.2f} {r['e14']:6.2f} | "
              f"{r['body2']:6.4f} {r['body4']:6.4f} {r['body8']:6.4f} | "
              f"{r['abs_err']:7.3f} {r['median']:6.3f} {r['p90']:7.3f} | "
              f"{r['tail8_frac']:6.2f}% {r['tail8_mean']:7.2f} {r['tail8_share']:5.1f}%")
    print("  (* = configured mode_window; body_n = mean |err| among pixels already within n mm)")

    ref = MVSFORMERPP_REF
    print(f"\n{'ref':>4} | {ref['e2']:6.2f} {ref['e4']:6.2f} {ref['e8']:6.2f} {ref['e14']:6.2f} | "
          f"{ref['body2']:6.4f} {ref['body4']:6.4f} {ref['body8']:6.4f} |"
          "   MVSFormer++ repro, DTU test @1152x1536, ep13")
    p = MVSFORMERPP_PAPER
    print(f"{'pap':>4} | {p['e2']:6.2f} {p['e4']:6.2f} {p['e8']:6.2f} {'':>6} | "
          f"{'':>6} {'':>6} {'':>6} |   MVSFormer++ paper Table 4, best ablation row")
    if not (args.no_in_scene and args.split == "test" and args.full_image):
        print("\n  NOTE: the reference rows are NOT comparable to the rows above unless you ran\n"
              "        --split test --full-image --resize-scale 0.96 --no-in-scene")

    if per_scan:
        print(f"\nper-scan @ mw={native_w}:")
        for scan in sorted(per_scan, key=lambda s: int(s.replace("scan", "") or 0)):
            s = per_scan[scan]
            print(f"  {scan:>8s}: e2={s['e2']:6.2f} e8={s['e8']:6.2f} body2={s['body2']:.4f} "
                  f"mean={s['abs_err']:7.3f} tail>8={s['tail8_frac']:5.2f}%")


def main() -> None:
    args = parse_args()
    cfg = build_mvs_config(profile=args.profile)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds = build_dataset(cfg, args)
    scans = sorted({m[0] for m in ds.metas}, key=lambda s: int(s.replace("scan", "") or 0))
    prior_wh = (args.prior_target_w or cfg.prior.target_w, args.prior_target_h or cfg.prior.target_h)
    print(f"[ablation] split={args.split} scans={len(scans)} samples={len(ds)} "
          f"resize={args.resize_scale} full_image={args.full_image} prior_target_wh={prior_wh}")

    ensure_priors(ds, device, args.build_priors, prior_wh)
    model, ckpt_path = load_model(cfg, args, device)

    native_w = cfg.depth_range.mode_window
    windows = [int(x) for x in args.mode_window.split(",")] if args.mode_window else [native_w]

    if args.mw_full_cascade and len(windows) > 1:
        # strict: every stage uses w, so the hypothesis axes change too
        results, per_scan, stats = {}, {}, None
        for w in windows:
            for dec in model.decoders:
                dec.mode_window = w
            print(f"\n[ablation] full-cascade pass, mode_window={w}")
            meters, stats = evaluate(model, ds, cfg, args, device, [w])
            results[w] = meters[w].overall()
            if w == native_w:
                per_scan = {s: meters[w].scan_metrics(s) for s in scans if s in meters[w].sums}
        for dec in model.decoders:
            dec.mode_window = native_w
    else:
        meters, stats = evaluate(model, ds, cfg, args, device, windows)
        results = {w: meters[w].overall() for w in windows}
        ref_w = native_w if native_w in meters else windows[0]
        per_scan = {s: meters[ref_w].scan_metrics(s) for s in scans if s in meters[ref_w].sums}

    print_report(results, per_scan, stats, args, (ds.height, ds.width), native_w)

    out = Path(args.out) if args.out else Path("outputs") / f"ablation_{args.split}" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": str(ckpt_path),
        "split": args.split,
        "eval_hw": [ds.height, ds.width],
        "resize_scale": args.resize_scale,
        "full_image": args.full_image,
        "in_scene_filter": not args.no_in_scene,
        "full_cascade": args.mw_full_cascade,
        "oos_frac_pct": 100.0 * stats["oos_px"] / max(stats["masked_px"], 1.0),
        "native_mode_window": native_w,
        "by_mode_window": {str(w): results[w] for w in results},
        "per_scan": per_scan,
        "reference_mvsformerpp": MVSFORMERPP_REF,
    }, indent=2), encoding="utf-8")
    print(f"\n[ablation] wrote {out}")


if __name__ == "__main__":
    main()
