#!/usr/bin/env python3
"""用离线 SfM 稀疏点 (log/sfm_cache) 把 DA3 单目相对深度 (log/da3_cache) 标定成 metric 深度。

每个 (scan, view) 的 SfM 点只在 light 3 上建过一次 (``scripts/build_sfm_cache_all.py``),
点是世界系里的 3D 几何, 与光照无关, 所以同一个视角的 **7 种光照的 DA3 都用这一份点标定**,
各自拟合各自的参数 (不同光照下 DA3 的输出量级/形状并不相同, 不能共用 a, b)。

## 标定模型: 深度域 a*d+b, 迭代截尾最小二乘

DA3 输出的是相对深度, 缺的不只是尺度还有 shift。前面的对比实验 (都是 DA3 + SfM 稀疏点):

  * 纯乘子 scale (``models/sfm.metric_scale_from_sparse``): 中位 158.7mm —— 错的模型
  * 逆深度域 scale+shift: 16.4mm, 偶发灾难
  * **深度域 a*d+b: 11.3mm** (同批 GT 上界 5.9mm, 差距来自当时 SfM 点太少)

见 [[sfm-scale-is-the-bottleneck]] / [[test16-prior-cache-scale-verdict]]。所以这里用深度域
仿射, 拟合算法与 ``models.norm_fill.robust_affine_align_depth`` 相同 (3 轮, 每轮按残差
中位数 ± trim_mad * 1.4826 * MAD 截尾), 只是作用在 SfM 点上的一维样本, 不依赖 torch。

DA3 在 SfM 点的亚像素位置 (uv) 上双线性取值, 四个邻点任一无效就丢掉这个点。

## 守卫与回退

  1. 参与拟合的点 >= ``--min-points`` 且 DA3 取值的相对跨度 (p90-p10)/中位 >= ``--min-rel-span``
     (点都挤在同一个深度上时 b 不可辨识) -> 仿射; 要求 a > 0。
  2. 否则点数 >= ``--min-scale-points`` -> 纯乘子 (中位比值), ``mode="scale"``。已知对 DA3
     偏差很大, 只作兜底, 下游可以按 mode 过滤。
  3. 再不行就不写文件, 记进失败表。

## 输出

``<out>/<scan>/da3_{view:04d}_{light}.npz`` —— 文件名、目录结构、``depth`` 这个 key 都与
da3_cache 完全一样, 下游读 DA3 的代码换个根目录就能拿到 metric 版本:

    depth         float16/32 [1200,1600]  metric 深度 (mm), a*d+b <= 0 的像素置 0 (无效)
    a, b          float64  标定参数 (depth = a * da3 + b), 想要无损的 float32 可以从 da3_cache 现算
    mode          "affine" / "scale"
    n_points      参与拟合的 SfM 点数;  n_inliers  截尾后剩下的点数
    fit_res_med   截尾内点上 |a*d+b - z_sfm| 的中位数 (mm), 不碰 GT 的自检指标
    support_depth float32 [2]  截尾内点 DA3 取值 p1/p99 对应的 metric 深度 (mm): SfM 点真正
                  支撑的深度区间。区间外的像素是全局仿射外推出来的, 可信度低
    extrap_frac   有效像素里落在支撑区间外的比例 —— 白桌面这类整片没有 SfM 点的视角会很高
    sfm_light / scan / view / light

GT (``Depths_raw``) **只用于汇总表里的误差统计**, 不参与拟合。

## 用法

    python scripts/calibrate_da3_with_sfm.py                   # 全量 (da3_cache 里有的都做), 断点续跑
    python scripts/calibrate_da3_with_sfm.py --dry-run
    python scripts/calibrate_da3_with_sfm.py --scans 1 13 --lights 3 --workers 1
    python scripts/calibrate_da3_with_sfm.py --gt-oracle       # 汇总表里再加一列 GT 仿射上界 (慢一点)

按 (scan, view) 分组并行 (``--workers`` 个 CPU 进程, 不需要 GPU)。汇总写
``<out>/_calib_summary_<时间戳>.csv``, 失败写 ``_calib_failures_<时间戳>.csv``。
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base.config import build_mvs_config  # noqa: E402

NATIVE_H, NATIVE_W = 1200, 1600
ALL_LIGHTS = tuple(range(7))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scans", type=int, nargs="*", default=None,
                   help="只做这些 scan (默认: da3_cache 下的全部)")
    p.add_argument("--views", type=int, nargs="*", default=None, help="0 起算的视角号 (默认全部)")
    p.add_argument("--lights", type=int, nargs="*", default=list(ALL_LIGHTS),
                   help="标定哪些光照的 DA3 (默认 0~6, da3_cache 里没有的自动跳过)")
    p.add_argument("--sfm-light", type=int, default=3, help="SfM 点云是在哪个光照上建的")
    p.add_argument("--min-obs", type=int, default=1,
                   help="只用 n_obs >= 该值的 SfM 点拟合 (建点时已经过滤过, 默认全用)")
    p.add_argument("--min-points", type=int, default=100, help="仿射拟合需要的最少点数")
    p.add_argument("--min-scale-points", type=int, default=20, help="纯乘子兜底需要的最少点数")
    p.add_argument("--min-rel-span", type=float, default=0.02,
                   help="DA3 取值 (p90-p10)/中位 低于它时 b 不可辨识, 退回纯乘子")
    p.add_argument("--trim-mad", type=float, default=3.5, help="截尾阈值 (MAD 倍数)")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                   help="depth 存盘精度。float16 在 700mm 处量化步长 0.5mm, 与 DA3 源缓存 (float16) "
                        "同量级; 要无损就用 a,b 从 da3_cache 现算")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-gt-eval", action="store_true", help="汇总表里不算与 GT 的误差")
    p.add_argument("--gt-oracle", action="store_true",
                   help="汇总表里再算一列 DA3 直接对 GT 仿射的误差 (上界, 只做对照)")
    p.add_argument("--force", action="store_true", help="已存在的文件也重算")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sfm-root", default=None, help="覆盖 cfg.paths.sfm_sparse_cache_path")
    p.add_argument("--da3-root", default=None, help="覆盖 cfg.paths.da3_cache_path")
    p.add_argument("--out", default=None, help="覆盖 cfg.paths.da3_sfm_cache_path")
    p.add_argument("--dtu-root", default=None, help="覆盖 cfg.paths.dtu_train_root (只用于 GT 统计)")
    return p.parse_args()


def da3_path(root: Path, scan: str, view: int, light: int) -> Path:
    return root / scan / f"da3_{view:04d}_{light}.npz"


def sfm_path(root: Path, scan: str, view: int, light: int) -> Path:
    return root / scan / f"sfm_{view:04d}_{light}.npz"


# --------------------------------------------------------------------------- #
# 拟合
# --------------------------------------------------------------------------- #
def robust_affine(x: np.ndarray, y: np.ndarray, trim_mad: float, min_points: int):
    """y ≈ a*x + b, 与 models.norm_fill.robust_affine_align_depth 同一算法。返回 (a, b, keep)。"""
    keep = np.ones_like(x, dtype=bool)
    a, b = 1.0, 0.0
    for _ in range(3):
        A = np.stack([x[keep], np.ones(int(keep.sum()))], axis=1)
        a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = y - (a * x + b)
        med = float(np.median(r[keep]))
        sigma = max(1.4826 * float(np.median(np.abs(r[keep] - med))), 1e-8)
        nk = np.abs(r - med) <= trim_mad * sigma
        if int(nk.sum()) < min_points or int(nk.sum()) == int(keep.sum()):
            break
        keep = nk
    return float(a), float(b), keep


def bilinear(img: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """在亚像素 uv (整数=像素中心) 上双线性取值; 越界或四邻点任一 <=0/非有限 的记无效。"""
    h, w = img.shape
    x, y = uv[:, 0].astype(np.float64), uv[:, 1].astype(np.float64)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    q = [img[y0c + dy, x0c + dx].astype(np.float64) for dy in (0, 1) for dx in (0, 1)]
    for v in q:
        ok &= np.isfinite(v) & (v > 0)
    fx, fy = x - x0, y - y0
    val = (q[0] * (1 - fx) * (1 - fy) + q[1] * fx * (1 - fy)
           + q[2] * (1 - fx) * fy + q[3] * fx * fy)
    return val, ok


def fit_view(da3: np.ndarray, uv: np.ndarray, z: np.ndarray, args) -> dict:
    x, ok = bilinear(da3, uv)
    x, y = x[ok], z[ok].astype(np.float64)
    n = len(x)
    info = {"n_points": n, "n_inliers": 0, "mode": "none", "a": np.nan, "b": np.nan,
            "rel_span": np.nan, "reason": ""}
    if n >= 2:
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        info["rel_span"] = float((p90 - p10) / max(p50, 1e-12))
    keep = np.zeros(n, bool)
    if n >= args.min_points and info["rel_span"] >= args.min_rel_span:
        a, b, k_aff = robust_affine(x, y, args.trim_mad, args.min_points)
        if a > 0:
            keep = k_aff
            info.update(mode="affine", a=a, b=b, n_inliers=int(keep.sum()))
        else:
            info["reason"] = f"仿射 a={a:.4g} <= 0"
    elif n >= args.min_points:
        info["reason"] = f"DA3 相对跨度 {info['rel_span']:.4f} < {args.min_rel_span}"
    else:
        info["reason"] = f"点数 {n} < {args.min_points}"
    if info["mode"] == "none" and n >= args.min_scale_points:
        ratio = y / x
        s = float(np.median(ratio))
        keep = np.abs(ratio - s) <= args.trim_mad * 1.4826 * np.median(np.abs(ratio - s)) + 1e-12
        info.update(mode="scale", a=s, b=0.0, n_inliers=int(keep.sum()))
    if info["mode"] != "none" and keep.any():
        res = np.abs(info["a"] * x[keep] + info["b"] - y[keep])
        info["fit_res_med"] = float(np.median(res))
        info["fit_res_p90"] = float(np.percentile(res, 90))
        info["support_da3"] = tuple(float(t) for t in np.percentile(x[keep], [1, 99]))
    return info


def err_stats(pred: np.ndarray, gt: np.ndarray, prefix: str) -> dict:
    m = (gt > 0) & np.isfinite(gt) & (pred > 0) & np.isfinite(pred)
    n_gt = int(((gt > 0) & np.isfinite(gt)).sum())
    if not m.any():
        return {f"{prefix}cover": 0.0}
    e = np.abs(pred[m].astype(np.float64) - gt[m])
    return {f"{prefix}med": float(np.median(e)), f"{prefix}mean": float(e.mean()),
            f"{prefix}lt2": float((e < 2).mean()), f"{prefix}lt5": float((e < 5).mean()),
            f"{prefix}cover": float(m.sum() / max(n_gt, 1))}


def atomic_savez(path: Path, **arrays) -> None:
    """临时名带 pid+随机串再 os.replace (与 build_da3_cache_all.save_depth 同一个理由)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 一个 (scan, view): 读一次 SfM/GT, 标定它的各个光照
# --------------------------------------------------------------------------- #
def process_view(job) -> list[dict]:
    scan, view, lights, a = job
    rows = []
    base = {"scan": scan, "view": view}
    try:
        with np.load(sfm_path(Path(a.sfm_root), scan, view, a.sfm_light)) as z:
            uv, depth, n_obs = z["uv"], z["depth"], z["n_obs"]
        sel = n_obs >= a.min_obs
        uv, depth = uv[sel], depth[sel]
    except Exception as exc:  # noqa: BLE001
        return [{**base, "light": l, "status": "fail",
                 "error": f"读 SfM 失败 {type(exc).__name__}: {exc}"} for l in lights]
    gt = None
    if a.gt_eval:
        f = Path(a.dtu_root) / "Depths_raw" / scan / f"depth_map_{view:04d}.pfm"
        if f.is_file():
            from data.io import read_pfm
            gt = read_pfm(str(f)).astype(np.float64)
    for light in lights:
        row = {**base, "light": light, "status": "ok"}
        t0 = time.time()
        try:
            with np.load(da3_path(Path(a.da3_root), scan, view, light)) as z:
                da3 = z["depth"].astype(np.float32)
            if da3.shape != (NATIVE_H, NATIVE_W):
                raise ValueError(f"DA3 形状 {da3.shape} != {(NATIVE_H, NATIVE_W)}")
            info = fit_view(da3, uv, depth, a)
            row.update({k: v for k, v in info.items() if k != "support_da3"})
            if info["mode"] == "none":
                row.update(status="fail", error=f"无法标定: {info['reason']}")
                rows.append(row)
                continue
            metric = info["a"] * da3.astype(np.float64) + info["b"]
            metric = np.where(np.isfinite(metric) & (metric > 0) & (da3 > 0), metric, 0.0)
            lo, hi = info.get("support_da3", (np.nan, np.nan))
            valid = metric > 0
            info["extrap_frac"] = float(((da3 < lo) | (da3 > hi))[valid].mean()) if valid.any() else 1.0
            support_depth = np.float32([info["a"] * lo + info["b"], info["a"] * hi + info["b"]])
            row["extrap_frac"] = info["extrap_frac"]
            atomic_savez(
                da3_path(Path(a.out), scan, view, light), depth=metric.astype(a.dtype),
                a=np.float64(info["a"]), b=np.float64(info["b"]), mode=np.asarray(info["mode"]),
                n_points=np.int32(info["n_points"]), n_inliers=np.int32(info["n_inliers"]),
                fit_res_med=np.float32(info.get("fit_res_med", np.nan)),
                support_depth=support_depth, extrap_frac=np.float32(info["extrap_frac"]),
                sfm_light=np.int32(a.sfm_light), scan=np.asarray(scan),
                view=np.int32(view), light=np.int32(light))
            if gt is not None:
                row.update(err_stats(metric, gt, "gt_"))
                if a.gt_oracle:
                    m = (gt > 0) & (da3 > 0)
                    if m.sum() >= 100:
                        oa, ob, _ = robust_affine(da3[m].astype(np.float64), gt[m], a.trim_mad, 100)
                        row.update(err_stats(oa * da3.astype(np.float64) + ob, gt, "oracle_"))
        except Exception as exc:  # noqa: BLE001 - 单个样本失败不能带崩整轮
            row.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        row["time_s"] = round(time.time() - t0, 3)
        rows.append(row)
    return rows


SUMMARY_COLS = ["scan", "view", "light", "status", "mode", "a", "b", "n_points", "n_inliers",
                "rel_span", "fit_res_med", "fit_res_p90", "extrap_frac", "gt_med", "gt_mean", "gt_lt2", "gt_lt5",
                "gt_cover", "oracle_med", "oracle_mean", "oracle_lt2", "time_s", "reason", "error"]


def _fmt(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main() -> None:
    args = parse_args()
    cfg = build_mvs_config()
    args.dtu_root = str(Path(args.dtu_root) if args.dtu_root else cfg.paths.dtu_train_root)
    args.sfm_root = str(Path(args.sfm_root) if args.sfm_root else cfg.paths.sfm_sparse_cache_path)
    args.da3_root = str(Path(args.da3_root) if args.da3_root else cfg.paths.da3_cache_path)
    args.out = str(Path(args.out) if args.out else cfg.paths.da3_sfm_cache_path)
    args.gt_eval = not args.no_gt_eval
    da3_root, sfm_root, out_root = Path(args.da3_root), Path(args.sfm_root), Path(args.out)

    if not da3_root.is_dir():
        raise SystemExit(f"找不到 da3_cache: {da3_root}")
    scans = ([f"scan{s}" for s in args.scans] if args.scans else
             sorted((d.name for d in da3_root.iterdir() if d.is_dir()), key=lambda s: (len(s), s)))
    jobs, n_done, n_todo, n_nosfm = [], 0, 0, 0
    for scan in scans:
        d = da3_root / scan
        if not d.is_dir():
            continue
        by_view: dict[int, list[int]] = {}
        for f in d.glob("da3_*_*.npz"):
            v, l = (int(t) for t in f.stem.split("_")[1:3])
            if (args.views is not None and v not in args.views) or l not in args.lights:
                continue
            if da3_path(out_root, scan, v, l).is_file() and not args.force:
                n_done += 1
                continue
            by_view.setdefault(v, []).append(l)
        for v in sorted(by_view):
            if not sfm_path(sfm_root, scan, v, args.sfm_light).is_file():
                n_nosfm += len(by_view[v])
            jobs.append((scan, v, sorted(by_view[v]), args))
            n_todo += len(by_view[v])
    print(f"[da3-sfm] da3_root={da3_root}\n[da3-sfm] sfm_root={sfm_root}\n[da3-sfm] out={out_root}")
    print(f"[da3-sfm] scans={len(scans)}  已完成 {n_done}  待做 {n_todo} ({len(jobs)} 个视角)  "
          f"其中缺 SfM 点云的 {n_nosfm} (会记为失败)")
    if args.dry_run or not jobs:
        print("[da3-sfm] dry-run 或无待办, 退出。")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    summary_csv = out_root / f"_calib_summary_{stamp}.csv"
    fail_csv = out_root / f"_calib_failures_{stamp}.csv"
    workers = max(1, min(args.workers, len(jobs)))
    t0 = time.time()
    n_ok = n_fail = done = 0
    modes: dict[str, int] = {}
    gt_meds = []
    with summary_csv.open("w", newline="") as fs, fail_csv.open("w", newline="") as ff:
        ws = csv.DictWriter(fs, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        wf = csv.writer(ff)
        ws.writeheader()
        wf.writerow(["scan", "view", "light", "error"])
        pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
        it = pool.imap_unordered(process_view, jobs, chunksize=2) if pool else map(process_view, jobs)
        try:
            for k, rows in enumerate(it, 1):
                for row in rows:
                    ws.writerow(row)
                    if row["status"] == "ok":
                        n_ok += 1
                        modes[row["mode"]] = modes.get(row["mode"], 0) + 1
                        if "gt_med" in row:
                            gt_meds.append(row["gt_med"])
                    else:
                        n_fail += 1
                        wf.writerow([row["scan"], row["view"], row["light"], row.get("error", "")])
                done += len(rows)
                if k % 50 == 0 or k == len(jobs):
                    fs.flush()
                    ff.flush()
                    el = time.time() - t0
                    print(f"[da3-sfm] {k}/{len(jobs)} 视角  {done}/{n_todo} 文件  ok={n_ok} fail={n_fail}  "
                          f"模式 {modes}  GT误差中位 {np.median(gt_meds) if gt_meds else float('nan'):.2f}mm  "
                          f"用时 {_fmt(el)}  预计剩余 {_fmt(el / done * (n_todo - done))}", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    print(f"[da3-sfm] 完成: ok={n_ok} fail={n_fail}  模式 {modes}  总用时 {_fmt(time.time() - t0)}")
    if gt_meds:
        g = np.asarray(gt_meds)
        print(f"[da3-sfm] 逐文件 GT 误差中位数: 中位 {np.median(g):.2f}mm  p90 {np.percentile(g, 90):.2f}mm")
    print(f"[da3-sfm] 汇总表: {summary_csv}")
    if n_fail:
        print(f"[da3-sfm] 失败清单: {fail_csv}")
    else:
        fail_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
