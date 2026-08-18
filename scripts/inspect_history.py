#!/usr/bin/env python
"""把 tensorboard 里已经存在、但没人读过的那几个数捞出来。

回答两个具体问题:

1. **那四个目标 (2.66 / 0.900 / 0.039 / 0.80) 取自 7.29 的哪个 step?**
   这决定它们是第 2 步 (12k arm) 的验收线还是第 3 步 (30k) 的。若 7.29 是在
   ~30k 才收敛到 2.66, 那么拿 12k arm 的 8k-12k 窗口均值去对它, 是拿 12k 的
   成绩比 30k 的终点 —— 判"没达标"就判错了对象。
   本脚本对每个 run 报告: 各目标第一次被达到的 step、run 的总长度、以及同一个
   run 在 8k-12k 窗口的值, 三者一对照口径就清楚了。

2. **stage4 的 in_range 现在是多少?**
   四个目标里唯一不在 slurm 日志里的一个。窗口几何 (32/16 vs 40/8) 的影响就
   落在这个标签上 —— 实测 32/16 是 0.809, 40/8 掉到 0.690。

用法:
    python scripts/inspect_history.py                      # 扫所有能找到的 run
    python scripts/inspect_history.py --runs R D0          # 只看这几个
    python scripts/inspect_history.py --lo 25000 --hi 30000   # 换判读窗口
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

# Codex 给的四个目标。"更好"的方向不同, 所以每个都带上比较方向。
TARGETS = [
    ("val/abs_err", 2.66, "lower"),
    ("val/acc_2mm", 0.900, "higher"),
    ("val/tail_frac_8mm", 0.039, "lower"),
    ("train/diag_stage4_in_range", 0.80, "higher"),
]

# stage4 in_range 在不同版本里可能挂在这几个 tag 之一。
S4_TAGS = ["train/diag_stage4_in_range", "val/q0_s4_in_range", "val/q1_s4_in_range",
           "val/q2_s4_in_range", "val/q3_s4_in_range"]


def find_runs(project: Path, only: list[str] | None) -> dict[str, list[str]]:
    """run 名 -> 事件目录列表。同时覆盖新旧两种布局。"""
    runs: dict[str, list[str]] = {}
    # 新布局: log/experiments/<run>/tensorboard/<timestamp>/
    for d in sorted(glob.glob(str(project / "log/experiments/*/tensorboard/*"))):
        runs.setdefault(Path(d).parents[1].name, []).append(d)
    # 旧布局 (7.29 / 8.16 那两次很可能在这里): log/tensorboard/<something>/
    for d in sorted(glob.glob(str(project / "log/tensorboard/*"))):
        runs.setdefault(f"legacy:{Path(d).name}", []).append(d)
    if only:
        runs = {k: v for k, v in runs.items() if any(o in k for o in only)}
    return runs


def load(dirs: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    merged: dict[str, dict[int, float]] = {}
    for d in dirs:
        acc = ea.EventAccumulator(d, size_guidance={ea.SCALARS: 0, ea.IMAGES: 1,
                                                    ea.HISTOGRAMS: 1, ea.TENSORS: 1})
        try:
            acc.Reload()
        except Exception:
            continue
        for t in acc.Tags()["scalars"]:
            merged.setdefault(t, {}).update({e.step: e.value for e in acc.Scalars(t)})
    return {t: (np.array(sorted(v)), np.array([v[s] for s in sorted(v)]))
            for t, v in merged.items()}


def first_reached(series, target: float, direction: str):
    """第一次达到目标的 (step, 值); 没达到过返回 (None, 最好值)。"""
    if series is None:
        return None, float("nan")
    s, v = series
    if v.size == 0:
        return None, float("nan")
    hit = (v <= target) if direction == "lower" else (v >= target)
    best = float(v.min() if direction == "lower" else v.max())
    if not hit.any():
        return None, best
    i = int(np.argmax(hit))
    return int(s[i]), float(v[i])


def window(series, lo: int, hi: int):
    if series is None:
        return float("nan"), 0
    s, v = series
    m = (s >= lo) & (s <= hi)
    return (float(v[m].mean()), int(m.sum())) if m.any() else (float("nan"), 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--runs", nargs="*", default=None, help="只看名字里含这些子串的 run")
    ap.add_argument("--lo", type=int, default=8000)
    ap.add_argument("--hi", type=int, default=12000)
    args = ap.parse_args()

    runs = find_runs(Path(args.project), args.runs)
    if not runs:
        print("没找到任何 tensorboard 事件目录。看过这两处:")
        print(f"    {args.project}/log/experiments/*/tensorboard/*")
        print(f"    {args.project}/log/tensorboard/*")
        return

    print(f"找到 {len(runs)} 个 run。判读窗口 = [{args.lo}, {args.hi}]\n")
    for name, dirs in runs.items():
        data = load(dirs)
        if not data:
            continue
        val = data.get("val/abs_err")
        last_step = int(val[0].max()) if val is not None and val[0].size else -1
        print("=" * 74)
        print(f"run: {name}    事件目录 {len(dirs)} 份    最后一个 val step = {last_step}")
        print("=" * 74)

        print("  四个目标 —— 第一次达到是在哪一步:")
        for tag, tgt, direction in TARGETS:
            series = data.get(tag)
            if series is None and tag == "train/diag_stage4_in_range":
                for alt in S4_TAGS:                       # in_range 换过 tag 名
                    if alt in data:
                        series, tag = data[alt], alt
                        break
            if series is None:
                print(f"    {tag:<34s} 目标 {tgt:<7g} —— 这个 run 里没有这个标签")
                continue
            step, v = first_reached(series, tgt, direction)
            w, n = window(series, args.lo, args.hi)
            if step is None:
                print(f"    {tag:<34s} 目标 {tgt:<7g} 从未达到 (最好 {v:.4f})"
                      f"   窗口均值 {w:.4f} (n={n})")
            else:
                print(f"    {tag:<34s} 目标 {tgt:<7g} 首次达到 @ step {step:<7d} ({v:.4f})"
                      f"   窗口均值 {w:.4f} (n={n})")

        print("  stage4 in_range (窗口均值):")
        any_s4 = False
        for tag in S4_TAGS:
            if tag in data:
                w, n = window(data[tag], args.lo, args.hi)
                print(f"    {tag:<34s} {w:.4f} (n={n})")
                any_s4 = True
        if not any_s4:
            print("    这个 run 没有记录 in_range —— 诊断标签是 25afa8f 之后才加的")
        print()

    print("=" * 74)
    print("怎么读:")
    print("  * 若 7.29 的 2.66 是在 ~30k 才首次达到, 那四个目标就是 30k 的验收线,")
    print("    不该拿来判 12k 的 arm。同口径的对照是 7.29 自己的 8k-12k 窗口均值。")
    print("  * in_range 若 32/16 ≈ 0.809 而 40/8 ≈ 0.690, 就坐实了窗口几何这一路,")
    print("    与 gate / branch_prior / visibility 无关。")
    print("=" * 74)


if __name__ == "__main__":
    main()
