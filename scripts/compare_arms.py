#!/usr/bin/env python
"""并排比较 log/experiments/ 下各个消融 arm。

判读规则（噪声地板实测自 0816/0817 那对同配置复现实验）:
  * 单个 val 点的复现噪声约 0.109mm —— 所以不要比 best 单点
  * 8k-12k 共 9 个 val 点的窗口均值把它压到约 0.048mm
  * 改善 <0.05mm 不解释; >0.15mm 才算确认; 中间需要再跑一个 seed

用法:
    python scripts/compare_arms.py                    # 全部 arm, 默认 8000-12000 窗口
    python scripts/compare_arms.py --lo 4000 --hi 6000
    python scripts/compare_arms.py --ref D0           # 以某个 arm 为基准算差值
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

# 8.16 / 7.29 两次历史实验在同一窗口的成绩，作为固定参照。
HIST = {"7.29 (b69ee46)": 2.7688, "8.16 (c8057bc)": 3.0111}

DIAGS = [
    ("train/diag_stage1_bp_raw_abs_err", "bp_raw"),
    ("train/diag_stage1_bp_post_abs_err", "bp_post"),
    ("train/diag_stage1_bp_flip_help", "flip_help"),
    ("train/diag_stage2_floor_binding", "s2_floor"),
    ("train/diag_stage3_floor_binding", "s3_floor"),
    ("train/diag_stage4_in_range", "s4_in_rng"),
    ("train/diag_stage1_guard_win_rate", "guard_win"),
]


def load(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """一个 arm 可能有多份事件文件（被 slurm 重排队过）——全部读进来按 step 合并。"""
    out: dict[str, dict[int, float]] = {}
    for d in sorted(glob.glob(str(run_dir / "tensorboard" / "*"))):
        acc = ea.EventAccumulator(d, size_guidance={ea.SCALARS: 0, ea.IMAGES: 1,
                                                    ea.HISTOGRAMS: 1, ea.TENSORS: 1})
        try:
            acc.Reload()
        except Exception:
            continue
        for t in acc.Tags()["scalars"]:
            out.setdefault(t, {}).update({e.step: e.value for e in acc.Scalars(t)})
    return {t: (np.array(sorted(v)), np.array([v[s] for s in sorted(v)])) for t, v in out.items()}


def window_mean(series, lo: int, hi: int) -> tuple[float, int]:
    if series is None:
        return float("nan"), 0
    s, v = series
    m = (s >= lo) & (s <= hi)
    return (float(v[m].mean()), int(m.sum())) if m.any() else (float("nan"), 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="log/experiments")
    ap.add_argument("--lo", type=int, default=8000)
    ap.add_argument("--hi", type=int, default=12000)
    ap.add_argument("--ref", default="D0", help="基准 arm 名")
    args = ap.parse_args()

    runs = sorted(p for p in Path(args.root).iterdir() if p.is_dir()) \
        if Path(args.root).is_dir() else []
    if not runs:
        raise SystemExit(f"{args.root} 下没有 arm —— 训练还没产出事件文件？")

    data = {}
    for r in runs:
        d = load(r)
        if "val/abs_err" in d:
            data[r.name] = d

    print(f"\n窗口 [{args.lo}, {args.hi}] 的 val 均值      "
          f"(噪声地板: 单点 ~0.109mm, 9 点窗口 ~0.048mm)\n")
    hdr = f"{'arm':<16}{'abs_err':>9}{'Δ vs ref':>10}{'acc_2mm':>9}{'tail_frac':>10}{'#pts':>6}{'last':>8}"
    print(hdr); print("-" * len(hdr))

    ref_v = window_mean(data.get(args.ref, {}).get("val/abs_err"), args.lo, args.hi)[0] \
        if args.ref in data else float("nan")
    rows = []
    for name, d in data.items():
        v, n = window_mean(d.get("val/abs_err"), args.lo, args.hi)
        a, _ = window_mean(d.get("val/acc_2mm"), args.lo, args.hi)
        t, _ = window_mean(d.get("val/tail_frac_8mm"), args.lo, args.hi)
        last = int(d["val/abs_err"][0].max())
        rows.append((v, name, v, a, t, n, last))
    for _, name, v, a, t, n, last in sorted(rows):
        dv = v - ref_v
        mark = "" if not np.isfinite(dv) else ("  ✓确认" if dv <= -0.15 else
                                               ("  ~待定" if dv <= -0.05 else ""))
        print(f"{name:<16}{v:>9.4f}{dv:>+10.4f}{a:>9.4f}{t:>10.4f}{n:>6d}{last:>8d}{mark}")
    print("-" * len(hdr))
    for k, v in HIST.items():
        print(f"{k:<16}{v:>9.4f}{v - ref_v:>+10.4f}    (历史, 同窗口)")

    print(f"\n关键诊断 (同窗口均值)\n")
    names = list(data)
    print(f"{'':<16}" + "".join(f"{n[:11]:>12}" for n in names))
    for tag, label in DIAGS:
        vals = [window_mean(data[n].get(tag), args.lo, args.hi)[0] for n in names]
        if all(not np.isfinite(x) for x in vals):
            continue
        print(f"{label:<16}" + "".join(
            (f"{x:>12.4f}" if np.isfinite(x) else f"{'-':>12}") for x in vals))
    print("\n  bp_post < bp_raw 才说明分支先验有正贡献; 反之应保持 branch_prior=off")
    print("  s2/s3_floor 高 = 窗宽由 range_min_gi*global_interval 决定 (改 num_global 会整体缩放)")

    print(f"\n深度分桶 val abs_err (q0 近 -> q3 远)\n")
    print(f"{'':<16}" + "".join(f"{n[:11]:>12}" for n in names))
    for b in range(4):
        vals = [window_mean(data[n].get(f"val/q{b}_abs_err"), args.lo, args.hi)[0] for n in names]
        if all(not np.isfinite(x) for x in vals):
            continue
        print(f"{'q'+str(b)+'_abs_err':<16}" + "".join(
            (f"{x:>12.4f}" if np.isfinite(x) else f"{'-':>12}") for x in vals))
    for b in range(4):
        vals = [window_mean(data[n].get(f"val/q{b}_s4_in_range"), args.lo, args.hi)[0] for n in names]
        if all(not np.isfinite(x) for x in vals):
            continue
        print(f"{'q'+str(b)+'_s4_in_rng':<16}" + "".join(
            (f"{x:>12.4f}" if np.isfinite(x) else f"{'-':>12}") for x in vals))
    print("\n  远端桶 (q3) 明显更差 = 值得做深度自适应窗宽; 四桶接近 = 瓶颈不在这里\n")


if __name__ == "__main__":
    main()
