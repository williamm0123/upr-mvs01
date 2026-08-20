#!/usr/bin/env python
"""验证「逆深度域 scale+shift」是否真的优于现在的单一乘性 scale。

背景: ``models/sfm.metric_scale_from_sparse`` 取 ``sparse/depth`` 比值的中位数,
只有乘性 scale。实测先验误差对 GT 深度呈 U 形 (最低点 ~620mm, 往两边发散),
而用 GT 做的 oracle 逆深度仿射把 log-log 斜率从 1.99 压到 0.70、最远档误差
从 9.62mm 降到 4.55mm —— 那正是"只拟合了 scale、漏掉 shift"留下的残差形状。

但 oracle 用了 GT, 是上界。这个脚本回答的是**能不能只用 SfM 稀疏点复现出来**::

    # HPC 上 (log/sfm_depth 有缓存时): 用真实 SfM 点拟合, held-out 上比较
    python scripts/validate_prior_affine.py --source sfm --n 60

    # 本地无 SfM 缓存时: 用空间抽稀的 GT 冒充稀疏点, 只验证代码路径与量级
    python scripts/validate_prior_affine.py --source gt --n 20

两种模式都按**空间块**划分 fit / held-out —— SfM 点在空间上高度聚集, 随机划
分会让同一 track 的邻域同时进两边, held-out 就失去意义。

判据: 只有 held-out 残差稳定优于 scale-only, 才值得重建缓存进主线。
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.sfm import apply_affine_inverse, metric_affine_from_sparse  # noqa: E402

GT_DIR = "/home/william/project/dataset/DTU/dtu_training/Depths_raw"
DMIN, DMAX = 425.0, 931.15


def read_pfm(fn):
    with open(fn, "rb") as f:
        f.readline()
        w, h = map(int, re.match(r"^(\d+)\s(\d+)\s*$", f.readline().decode()).groups())
        sc = float(f.readline().decode())
        return np.flipud(np.reshape(np.fromfile(f, ("<" if sc < 0 else ">") + "f"), (h, w)))


def load_gt(scan, view, shape):
    g = f"{GT_DIR}/{scan}/depth_map_{view:04d}.pfm"
    m = f"{GT_DIR}/{scan}/depth_visual_{view:04d}.png"
    if not os.path.exists(g):
        return None, None
    from PIL import Image
    gt = read_pfm(g).astype(np.float32)
    mk = np.array(Image.open(m), dtype=np.float32)
    f_ = gt.shape[0] // shape[0]
    if f_ > 1:
        gt, mk = gt[::f_, ::f_], mk[::f_, ::f_]
    if gt.shape != shape:
        return None, None
    return gt, (mk > 10)


def sparse_from_gt(gt, valid, stride: int, rng):
    """空间抽稀的 GT 当作稀疏点。抖动起点, 避免总是取同一批像素。"""
    out = np.zeros_like(gt)
    oy, ox = rng.integers(0, stride, size=2)
    sel = np.zeros(gt.shape, bool)
    sel[oy::stride, ox::stride] = True
    sel &= valid
    out[sel] = gt[sel]
    return out


def bucket_report(prior, target, valid, label):
    e = np.abs(prior[valid] - target[valid])
    d = target[valid]
    qs = np.quantile(d, [0, .25, .5, .75, 1.0])
    row = [f"{label:<10}{np.median(e):8.3f}{e.mean():8.3f}{np.quantile(e, .95):9.3f}  |"]
    for i in range(4):
        m = (d >= qs[i]) & (d < qs[i + 1] if i < 3 else d <= qs[i + 1])
        row.append(f"{np.median(e[m]):8.3f}" if m.any() else f"{'-':>8}")
    return "".join(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sfm", "gt"], default="gt")
    ap.add_argument("--n", type=int, default=20, help="抽多少个 (scan, view)")
    ap.add_argument("--stride", type=int, default=16, help="--source gt 时的抽稀步长")
    ap.add_argument("--cache", default=str(_ROOT / "log" / "prior_cache"))
    ap.add_argument("--sfm-cache", default=str(_ROOT / "log" / "sfm_depth"))
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.cache}/scan*/prior_*_3.npz"))
    if not files:
        raise SystemExit(f"{args.cache} 下没有先验缓存")
    rng = np.random.default_rng(0)
    random.seed(0)
    sel = random.sample(files, min(args.n, len(files)))

    ok_n = fail = 0
    reasons: dict[str, int] = {}
    hold_a, hold_s = [], []
    all_prior, all_aff, all_gt, all_val = [], [], [], []

    for fn in sel:
        scan = os.path.basename(os.path.dirname(fn))
        view = int(re.search(r"prior_(\d+)_", fn).group(1))
        z = np.load(fn)
        if float(z["sfm_valid"]) < 0.5:
            continue
        prior = z["depth_prior"].astype(np.float32)
        gt, valid = load_gt(scan, view, prior.shape)
        if gt is None:
            continue
        valid = valid & np.isfinite(gt) & (gt > 1) & np.isfinite(prior) & (prior > 0)
        if valid.sum() < 5000:
            continue

        if args.source == "gt":
            sparse = sparse_from_gt(gt, valid, args.stride, rng)
        else:
            sp = f"{args.sfm_cache}/{scan}/sfm_{view:04d}_3.npy"
            if not os.path.exists(sp):
                continue
            sparse = np.load(sp).astype(np.float32)
            if sparse.shape != prior.shape:
                continue

        good, (a, b), info = metric_affine_from_sparse(
            prior, sparse, sparse > 0, depth_min=DMIN, depth_max=DMAX)
        if good:
            ok_n += 1
            hold_a.append(info["hold_affine_median"])
            hold_s.append(info["hold_scale_median"])
            all_aff.append(apply_affine_inverse(prior, a, b))
        else:
            fail += 1
            key = info["reason"].split("(")[0].strip() or "未知"
            reasons[key] = reasons.get(key, 0) + 1
            all_aff.append(prior)
        all_prior.append(prior)
        all_gt.append(gt)
        all_val.append(valid)

    n = ok_n + fail
    if not n:
        raise SystemExit("没有可用样本 —— 检查 --cache / --sfm-cache 路径与 GT 是否存在")

    print(f"\n来源={args.source}  样本 {n} 个   仿射通过 {ok_n} ({100 * ok_n / n:.0f}%)   "
          f"退回 scale-only {fail}")
    if reasons:
        print("  退回原因:")
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {v:3d}x  {k}")
    if hold_a:
        print(f"\n  held-out |z - sparse| 中位数:  affine {np.median(hold_a):.3f}  "
              f"vs  scale-only {np.median(hold_s):.3f}   "
              f"({100 * (np.median(hold_a) / np.median(hold_s) - 1):+.1f}%)")

    P = np.concatenate([p[v] for p, v in zip(all_prior, all_val)])
    A = np.concatenate([p[v] for p, v in zip(all_aff, all_val)])
    G = np.concatenate([g[v] for g, v in zip(all_gt, all_val)])
    full = np.ones_like(G, bool)
    print(f"\n  对 GT 的误差 (全部像素, 这是最终判据)\n")
    print(f"{'':<10}{'median':>8}{'mean':>8}{'p95':>9}  |{'q0':>8}{'q1':>8}{'q2':>8}{'q3':>8}  (按 GT 深度四分位)")
    print("  " + "-" * 76)
    print("  " + bucket_report(P, G, full, "scale-only"))
    print("  " + bucket_report(A, G, full, "affine"))

    def slope(pred):
        e = np.abs(pred - G)
        qs = np.quantile(G, np.linspace(0, 1, 9))
        c, y = [], []
        for i in range(8):
            m = (G >= qs[i]) & (G < qs[i + 1])
            if m.sum() > 100:
                c.append(np.median(G[m]))
                y.append(max(np.median(e[m]), 1e-3))
        return np.polyfit(np.log(c), np.log(y), 1)[0]

    print(f"\n  误差随深度的 log-log 斜率:  scale-only {slope(P):+.2f}   affine {slope(A):+.2f}")
    print("  (0 = 与深度无关, 2 = 视差域恒定。越接近 0 说明系统偏差修得越干净)\n")


if __name__ == "__main__":
    main()
