#!/usr/bin/env python
"""验证「逆深度域 scale+shift」是否真的优于现在的单一乘性 scale。

背景: ``models/sfm.metric_scale_from_sparse`` 取 ``sparse/depth`` 比值的中位数,
只有乘性 scale, 依据是 "VGGT depth is metric-consistent up to a single global
scale"。实测这条假设不完全成立 —— 先验误差对 GT 深度呈 U 形 (最低点 ~620mm,
往两边发散), 而逆深度仿射能把 log-log 斜率从 ~2 压到 ~0.4、最远档误差降到四分
之一。那正是"输出本质是仿射-in-逆深度、却只拟合了乘性 scale"留下的残差形状。

两种稀疏点来源::

    # 真实检验: 按 pre_prior 的同一配方现算 SfM 稀疏深度 (纯 OpenCV, 不用 GPU)
    python scripts/validate_prior_affine.py --source sfm --n 40

    # 上界参考: 用空间抽稀的 GT 冒充稀疏点
    python scripts/validate_prior_affine.py --source gt --n 40

注意 ``log/sfm_depth`` **没有缓存** —— data/dtu.py 里那段 load_or_compute 是注释
掉的, 稀疏深度是建先验时现算的。所以 ``--source sfm`` 走的也是现算, 每个样本
要跑一次特征匹配 + 三角化, 比 ``--source gt`` 慢。

判据: 只有 held-out 残差稳定优于 scale-only, 才值得重建缓存进主线。fit/held-out
按**空间块**划分 —— SfM 点在空间上高度聚集, 随机划分会让同一 track 的邻域同时
进两边, held-out 就失去意义。
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "models", _ROOT / "models" / "Depth-Anything-3" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from base.config import build_mvs_config, resolve_split  # noqa: E402
from models.sfm import apply_affine_inverse, metric_affine_from_sparse  # noqa: E402


def read_pfm(fn):
    with open(fn, "rb") as f:
        f.readline()
        w, h = map(int, re.match(r"^(\d+)\s(\d+)\s*$", f.readline().decode()).groups())
        sc = float(f.readline().decode())
        return np.flipud(np.reshape(np.fromfile(f, ("<" if sc < 0 else ">") + "f"), (h, w)))


def sparse_from_gt(gt, valid, stride, rng):
    """空间抽稀的 GT 当稀疏点。抖动起点, 免得每次都取同一批像素。"""
    out = np.zeros_like(gt)
    oy, ox = rng.integers(0, stride, size=2)
    sel = np.zeros(gt.shape, bool)
    sel[oy::stride, ox::stride] = True
    sel &= valid
    out[sel] = gt[sel]
    return out


def bucket_row(err, depth, label):
    qs = np.quantile(depth, [0, .25, .5, .75, 1.0])
    row = f"{label:<12}{np.median(err):8.3f}{err.mean():8.3f}{np.quantile(err, .95):9.3f}  |"
    for i in range(4):
        m = (depth >= qs[i]) & ((depth < qs[i + 1]) if i < 3 else (depth <= qs[i + 1]))
        row += f"{np.median(err[m]):8.3f}" if m.any() else f"{'-':>8}"
    return row


def slope(err, depth):
    qs = np.quantile(depth, np.linspace(0, 1, 9))
    c, y = [], []
    for i in range(8):
        m = (depth >= qs[i]) & (depth < qs[i + 1])
        if m.sum() > 100:
            c.append(np.median(depth[m]))
            y.append(max(np.median(err[m]), 1e-3))
    return np.polyfit(np.log(c), np.log(y), 1)[0] if len(c) >= 4 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sfm", "gt"], default="sfm")
    ap.add_argument("--n", type=int, default=40, help="抽多少个 (scan, view)")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--stride", type=int, default=16, help="--source gt 时的抽稀步长")
    ap.add_argument("--no-clean-lists", action="store_true", default=True)
    args = ap.parse_args()

    from data.dtu import DTUMVSDataset
    cfg = build_mvs_config("umhpc")
    root = Path(cfg.paths.dtu_train_root)
    lf, ef = resolve_split(getattr(cfg.paths, f"{args.split}_list_file"),
                           args.split, not args.no_clean_lists)
    ds = DTUMVSDataset(datapath=root, listfile=lf, exclude_file=ef,
                       nviews=cfg.train.num_views, mode="val",
                       use_src_weights=False, seed=cfg.train.seed)
    print(f"[data] {root}   {args.split} metas={len(ds)}   "
          f"prior_cache={cfg.paths.prior_cache_path}")

    random.seed(0)
    idxs = random.sample(range(len(ds)), min(args.n, len(ds)))
    rng = np.random.default_rng(0)

    skip = Counter()
    ok_n = 0
    reasons = Counter()
    hold_a, hold_s, stats = [], [], []
    P, A, G = [], [], []

    for idx in idxs:
        scan, light, view, _ = ds.metas[idx]
        cpath = ds.prior_cache_path_for(idx)
        if not os.path.exists(cpath):
            skip["先验缓存缺失"] += 1
            continue
        z = np.load(cpath)
        if float(z.get("sfm_valid", 0.0)) < 0.5:
            skip["sfm_valid=0"] += 1
            continue
        prior_raw = z["depth_prior"].astype(np.float32)

        # GT: 与 dtu.py 同一份路径与命名
        gfn = root / "Depths_raw" / scan / f"depth_map_{view:04d}.pfm"
        mfn = root / "Depths_raw" / scan / f"depth_visual_{view:04d}.png"
        if not gfn.exists() or not mfn.exists():
            skip["GT 缺失"] += 1
            continue
        gt_full = read_pfm(str(gfn)).astype(np.float32)
        mk_full = ds.read_mask(str(mfn))

        if args.source == "sfm":
            # 按 pre_prior.PriorPrecomputer.compute 的同一配方现算
            import models.sfm as S
            pc = ds.precrop_inputs(idx)
            try:
                sfm_out = S.generate_sparse_depth_from_sample(pc, ref_idx=0)
            except Exception as exc:
                skip[f"SfM 失败: {type(exc).__name__}"] += 1
                continue
            sparse = np.asarray(sfm_out["sparse_depth"], np.float32)
            if not (sparse > 0).any():
                skip["SfM 没产出稀疏点"] += 1
                continue
            hw = sparse.shape
        else:
            hw = prior_raw.shape

        prior = ds._match_hw(prior_raw, hw, is_depth=True)
        gt = ds._match_hw(gt_full, hw, is_depth=True)
        mk = ds._match_hw(mk_full.astype(np.float32), hw, is_depth=False) > 0.5
        valid = mk & np.isfinite(gt) & (gt > 1) & np.isfinite(prior) & (prior > 0)
        if valid.sum() < 5000:
            skip["有效像素不足"] += 1
            continue
        if args.source == "gt":
            sparse = sparse_from_gt(gt, valid, args.stride, rng)

        dv = np.asarray(ds.precrop_inputs(idx)["depth_values"], np.float32) \
            if args.source == "gt" else np.asarray(pc["depth_values"], np.float32)
        good, (a, b), info = metric_affine_from_sparse(
            prior, sparse, sparse > 0,
            depth_min=float(dv[0]), depth_max=float(dv[-1]))
        stats.append((info.get("num_pairs", 0), info.get("inv_span_rel", float("nan"))))
        if good:
            ok_n += 1
            hold_a.append(info["hold_affine_median"])
            hold_s.append(info["hold_scale_median"])
            A.append(apply_affine_inverse(prior, a, b)[valid])
        else:
            reasons[info["reason"].split("(")[0].strip() or "未知"] += 1
            A.append(prior[valid])
        P.append(prior[valid])
        G.append(gt[valid])

    n = len(P)
    if not n:
        print("\n没有可用样本。逐项跳过原因:")
        for k, v in skip.most_common():
            print(f"  {v:4d}x  {k}")
        print(f"\n检查: 先验缓存 {cfg.paths.prior_cache_path} 是否存在; "
              f"GT 是否在 {root}/Depths_raw")
        raise SystemExit(1)

    print(f"\n来源={args.source}   可用样本 {n}/{len(idxs)}   "
          f"仿射通过 {ok_n} ({100 * ok_n / n:.0f}%)   退回 scale-only {n - ok_n}")
    if skip:
        print("  跳过:  " + "   ".join(f"{k} x{v}" for k, v in skip.most_common()))
    if reasons:
        print("  退回原因:")
        for k, v in reasons.most_common():
            print(f"    {v:3d}x  {k}")
    if stats:
        npair = np.array([x[0] for x in stats], float)
        spans = np.array([x[1] for x in stats], float)
        print(f"  稀疏点: 中位 {np.median(npair):.0f} 个 (最少 {npair.min():.0f}); "
              f"逆深度相对跨度中位 {np.nanmedian(spans):.3f}")
    if hold_a:
        print(f"\n  held-out |z - sparse| 中位数:  affine {np.median(hold_a):.3f}  vs  "
              f"scale-only {np.median(hold_s):.3f}   "
              f"({100 * (np.median(hold_a) / np.median(hold_s) - 1):+.1f}%)")

    P, A, G = np.concatenate(P), np.concatenate(A), np.concatenate(G)
    ep, ea = np.abs(P - G), np.abs(A - G)
    print("\n  对 GT 的误差 (全部像素, 这是最终判据)\n")
    print(f"  {'':<12}{'median':>8}{'mean':>8}{'p95':>9}  |{'q0':>8}{'q1':>8}{'q2':>8}{'q3':>8}"
          "   (按 GT 深度四分位)")
    print("  " + "-" * 78)
    print("  " + bucket_row(ep, G, "scale-only"))
    print("  " + bucket_row(ea, G, "affine"))
    print(f"\n  误差随深度的 log-log 斜率:  scale-only {slope(ep, G):+.2f}   "
          f"affine {slope(ea, G):+.2f}")
    if hold_a and np.median(ea) >= np.median(ep):
        print("\n  !! held-out 改善但对 GT 没改善 —— 拟合过拟到 SfM 点的分布上了。")
        print("     held-out 是在**稀疏点**上量的, 而稀疏点集中在有纹理的区域、"
              "空间分布有偏;")
        print("     它变好不代表整幅深度图变好。建缓存时拿不到 GT, 所以这道守卫"
              "天然测不到这件事。")
    print("  (0 = 与深度无关, 2 = 视差域恒定。越接近 0 说明系统偏差修得越干净)\n")


if __name__ == "__main__":
    main()
