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

# 历史 30k run 的窗口均值曾经放在这里作参照, 已删除: 它们的余弦退火 horizon 是
# 30000, 而 12k 筛选 arm 的 horizon 是 12000 —— 同一步数窗口里 lr 差一个数量级
# (step 10000: 2.34e-4 vs 1.59e-5), 12k arm 是已退火的模型。跨 horizon 横比会
# 系统性偏袒短 horizon 的 arm。只有 lr_schedule_steps 完全一致才可比。

DIAGS = [
    ("train/diag_grad_norm_unclipped", "grad_norm"),
    ("train/diag_grad_amp_scale", "amp_scale"),
    ("train/diag_grad_nonfinite_frac", "nonfin_frac"),
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
    for _, name, v, a, t, n, last in sorted(rows, key=lambda r: (np.isnan(r[0]), r[0])):
        dv = v - ref_v
        mark = "" if not np.isfinite(dv) else ("  ✓确认" if dv <= -0.15 else
                                               ("  ~待定" if dv <= -0.05 else ""))
        print(f"{name:<16}{v:>9.4f}{dv:>+10.4f}{a:>9.4f}{t:>10.4f}{n:>6d}{last:>8d}{mark}")
    print("-" * len(hdr))

    print(f"\n关键诊断 (同窗口均值)\n")
    names = list(data)
    print(f"{'':<16}" + "".join(f"{n[:11]:>12}" for n in names))
    for tag, label in DIAGS:
        vals = [window_mean(data[n].get(tag), args.lo, args.hi)[0] for n in names]
        if all(not np.isfinite(x) for x in vals):
            continue
        print(f"{label:<16}" + "".join(
            (f"{x:>12.4f}" if np.isfinite(x) else f"{'-':>12}") for x in vals))
    print("\n  abs_err 是 nan / #pts 远小于 9 = 这个 arm 中途发散或被杀, 先看它的 slurm 日志")
    print("  bp_post < bp_raw 才说明分支先验有正贡献; 反之应保持 branch_prior=off")
    print("  s2/s3_floor 高 = 窗宽由 range_min_gi*global_interval 决定 (改 num_global 会整体缩放)")
    print("  amp_scale 一路下滑 / nonfin_frac 非 0 = 溢出在变频繁, 是发散的提前量")
    print("  need_half90 > have_half = 窗口不够宽; cover_x2 = 窗宽翻倍后的覆盖率")
    print("  gt_rank<=2 高 = 正确候选常在 top-2, 双模态才值得做")

    print(f"\n深度分桶: abs_err / stage4 覆盖率 / 覆盖条件下的误差 (q0 近 -> q3 远)\n")
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
    # 覆盖率和精度必须分开看: 抬宽窗口一定同时改变两者, 只看总 abs_err 分不出
    # 是覆盖变好还是精度变差。
    for suf, lbl in (("_abs_err_s4_in", "_err_in"), ("_abs_err_s4_out", "_err_out")):
        for b in range(4):
            vals = [window_mean(data[n].get(f"val/q{b}{suf}"), args.lo, args.hi)[0] for n in names]
            if all(not np.isfinite(x) for x in vals):
                continue
            print(f"{'q' + str(b) + lbl:<16}" + "".join(
                (f"{x:>12.4f}" if np.isfinite(x) else f"{'-':>12}") for x in vals))
    print("\n  远端桶 (q3) 明显更差 = 值得做深度自适应窗宽; 四桶接近 = 瓶颈不在这里")
    print("  err_in 基本不变而 in_rng 上升 = 抬宽窗口是净赚; err_in 同时变差 = 在拿精度换覆盖\n")


if __name__ == "__main__":
    main()
