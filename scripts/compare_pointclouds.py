#!/usr/bin/env python3
"""两组点云的逐场景配对比较 —— 工单 v5.3 §8.1 的终审判据。

**必须在看到结果之前写好。** 先跑出数再决定怎么比, 等于让结论挑自己的检验方法;
这个脚本把判据钉死在代码里, 跑的时候只输入两个目录。

    python scripts/compare_pointclouds.py \
        --base log/pred_points_W0_final \
        --cand log/pred_points_UPRMVS_vNext_final

判据 (v5.3 §8.1, 全部逐场景配对):
    delta_acc      <= -0.02 mm      主判据
    delta_overall  <  0
    comp           不显著退化        <- 配对 bootstrap 的 95% CI, 不是肉眼
    scan48/77      报均值与单场景, **不设"一场不改善就否决"的硬门**
    scan29/33/49   只做分解, 不作判据

为什么是**配对** bootstrap: 22 个 scan 的难度差着好几倍 (scan77 的 acc 0.66,
scan114 只有 0.22)。两组各自算均值再相减, 方差被场景难度支配, 22 个样本根本
分不出 0.02mm。配对之后每个 scan 自己做对照, 剩下的才是方法的差异。

重采样的是 **scan**, 不是像素 —— 同一个 scan 内的点不独立, 按点重采样会把
置信区间压得虚窄。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# 论文 (MonoMVSNet Fig.4) 直接点名的两个反光/深度不连续场景
PAPER_SCANS = (48, 77)
# UPRMVS 自身的困难集 —— 只做分解, 不作判据
OWN_HARD = (29, 33, 49)

_ROW = re.compile(r"scan(\d+)\s+acc:\s*([\d.]+)\s+comp:\s*([\d.]+)\s+overall:\s*([\d.]+)")


def read_result(d: Path) -> tuple[dict[int, tuple[float, float, float]], dict | None]:
    """读 Fast-DTU-Evaluation 的 result.txt, 外加同目录的 run_manifest.json。"""
    f = d / "result.txt"
    if not f.exists():
        raise SystemExit(f"{f} 不存在 —— 先用 Fast-DTU-Evaluation 给这个目录打分")
    rows = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if m:
            rows[int(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
    if not rows:
        raise SystemExit(f"{f} 里没有解析出任何 scan 行")
    man = d / "run_manifest.json"
    return rows, (json.loads(man.read_text(encoding="utf-8")) if man.exists() else None)


def paired_bootstrap(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float, float]:
    """返回 (mean, lo95, hi95, P(diff_mean < 0))。重采样单位是 scan。"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return (float(diff.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), float((means < 0).mean()))


def _manifest_line(tag: str, man: dict | None) -> str:
    if man is None:
        return (f"  {tag}: **没有 run_manifest.json** —— 谱系不明, "
                f"按 v5.3 §6.1 不得作为基线")
    ck = man.get("checkpoint", {}) or {}
    fu = man.get("fusion", {}) or {}
    inf = man.get("inference", {}) or {}
    return (f"  {tag}: {ck.get('path')}\n"
            f"        step={ck.get('step')}  git={(man.get('code_git') or {}).get('sha', '?')[:8]}"
            f"  dirty={(man.get('code_git') or {}).get('dirty')}\n"
            f"        views={inf.get('num_views')} resize={inf.get('resize_scale')}"
            f"  conf={fu.get('conf_source')} keep_ratio={fu.get('photo_keep_ratio')}"
            f"  geo_views={fu.get('geo_views')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基线 ply 目录 (W0)")
    ap.add_argument("--cand", required=True, help="候选 ply 目录 (vNext)")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260526)
    ap.add_argument("--acc-target", type=float, default=-0.02,
                    help="v5.3 §8.1 的主判据: delta_acc 必须 <= 这个值")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    base, man_b = read_result(Path(a.base))
    cand, man_c = read_result(Path(a.cand))

    print("[谱系]")
    print(_manifest_line("base", man_b))
    print(_manifest_line("cand", man_c))
    if man_b and man_c:
        fb, fc = man_b.get("fusion", {}), man_c.get("fusion", {})
        for k in ("photo_keep_ratio", "geo_views", "geo_pix", "geo_rel", "backend"):
            if fb.get(k) != fc.get(k):
                print(f"  !! 融合参数 {k} 两边不同 ({fb.get(k)} vs {fc.get(k)}) —— "
                      f"这不是同一套协议, 比较无效")
        ib, ic = man_b.get("inference", {}), man_c.get("inference", {})
        for k in ("num_views", "resize_scale", "full_image", "split"):
            if ib.get(k) != ic.get(k):
                print(f"  !! 推理参数 {k} 两边不同 ({ib.get(k)} vs {ic.get(k)})")

    shared = sorted(set(base) & set(cand))
    only_b, only_c = sorted(set(base) - set(cand)), sorted(set(cand) - set(base))
    if only_b or only_c:
        print(f"\n  !! 只在一边出现的 scan: base {only_b}  cand {only_c} —— "
              f"配对比较只用两边都有的 {len(shared)} 个")
    if len(shared) < 2:
        raise SystemExit("配对样本不足")

    names = ("acc", "comp", "overall")
    diffs = {n: np.array([cand[s][i] - base[s][i] for s in shared]) for i, n in enumerate(names)}

    print(f"\n[逐场景] {len(shared)} 个配对场景 (Δ = cand − base, 负 = 变好)")
    print(f"  {'scan':>7}{'acc_b':>9}{'acc_c':>9}{'Δacc':>9}"
          f"{'Δcomp':>9}{'Δover':>9}")
    for s in shared:
        mark = " *" if s in PAPER_SCANS else ("  ~" if s in OWN_HARD else "")
        print(f"  {('scan' + str(s)):>7}{base[s][0]:>9.4f}{cand[s][0]:>9.4f}"
              f"{cand[s][0] - base[s][0]:>+9.4f}{cand[s][1] - base[s][1]:>+9.4f}"
              f"{cand[s][2] - base[s][2]:>+9.4f}{mark}")
    print("  (* = 论文点名的 scan48/77   ~ = UPRMVS 自身困难集, 只做分解)")

    print(f"\n[配对 bootstrap] 重采样单位 = scan, n={a.n_boot}")
    print(f"  {'指标':<10}{'base':>9}{'cand':>9}{'Δ均值':>10}{'95% CI':>22}{'P(Δ<0)':>9}")
    res = {}
    for i, n in enumerate(names):
        mb = float(np.mean([base[s][i] for s in shared]))
        mc = float(np.mean([cand[s][i] for s in shared]))
        m, lo, hi, p = paired_bootstrap(diffs[n], a.n_boot, a.seed + i)
        res[n] = dict(base=mb, cand=mc, delta=m, lo=lo, hi=hi, p_better=p)
        print(f"  {n:<10}{mb:>9.4f}{mc:>9.4f}{m:>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:>9.3f}")

    print("\n[判据 · v5.3 §8.1]")
    verdict = {}
    d_acc, d_over, d_comp = res["acc"], res["overall"], res["comp"]

    verdict["acc"] = d_acc["delta"] <= a.acc_target
    print(f"  {'PASS' if verdict['acc'] else 'FAIL'}  Δacc = {d_acc['delta']:+.4f} "
          f"(要求 ≤ {a.acc_target:+.4f})")

    verdict["overall"] = d_over["delta"] < 0
    print(f"  {'PASS' if verdict['overall'] else 'FAIL'}  Δoverall = {d_over['delta']:+.4f} "
          f"(要求 < 0)")

    # comp 只要求"不显著退化": 允许变差, 但 95% CI 的下界不能整体落在 0 以上。
    verdict["comp"] = d_comp["lo"] <= 0.0
    print(f"  {'PASS' if verdict['comp'] else 'FAIL'}  comp 未显著退化: "
          f"Δcomp = {d_comp['delta']:+.4f}, CI 下界 {d_comp['lo']:+.4f} "
          f"(要求 ≤ 0; 显著变差才算 FAIL)")

    have = [s for s in PAPER_SCANS if s in shared]
    if have:
        dm = float(np.mean([cand[s][2] - base[s][2] for s in have]))
        detail = "  ".join(f"scan{s} {cand[s][2] - base[s][2]:+.4f}" for s in have)
        verdict["paper_scans"] = dm < 0
        print(f"  {'PASS' if verdict['paper_scans'] else 'FAIL'}  scan48/77 overall 均值 "
              f"{dm:+.4f} (要求 < 0; 单场景不设硬门) —— {detail}")
    own = [s for s in OWN_HARD if s in shared]
    if own:
        print("  [分解, 不作判据] " + "  ".join(
            f"scan{s} {cand[s][2] - base[s][2]:+.4f}" for s in own))

    allpass = all(verdict.values())
    print(f"\n[结论] {'通过' if allpass else '未通过'} "
          f"({sum(verdict.values())}/{len(verdict)} 条判据)")
    if not allpass:
        print("  按 v5.3 §9: 先看护栏曲线定位是工程故障还是模型问题, "
              "**不要**立刻拆成三个 arm 逐项归因。")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(
            {"base": str(a.base), "cand": str(a.cand), "n_scans": len(shared),
             "scans": shared, "metrics": res, "verdict": verdict, "pass": allpass,
             "per_scan": {str(s): {"base": base[s], "cand": cand[s]} for s in shared}},
            indent=2), encoding="utf-8")
        print(f"[写入] {a.json_out}")


if __name__ == "__main__":
    main()
