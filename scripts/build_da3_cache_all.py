#!/usr/bin/env python3
"""对全数据集每个 (scan, view, light) 跑一次 DA3 单目深度, 存到 cfg.paths.da3_cache_path。

和 log/prior_cache 的区别: 这里**没有 VGGT, 没有融合, 没有标尺**, 就是 DA3 的原始相对深度,
在原生 1200x1600 (H x W) 分辨率上存下来, 给以后任何需要 DA3 深度的实验直接读, 不用每次现跑。
命名/目录结构照抄 prior_cache 的约定 (每 scan 一个子目录, 缺什么建什么, 可断点续跑)。

## 枚举范围

scan 列表来自 ``<dtu_train_root>/Rectified_raw/`` 目录的实际内容 (不是某个 split 的
listfile) —— dtu_training 下 79 个 train + 18 个 val + 22 个 test + 5 个不在任何列表里的
scan, 一共 124 个 (2026-09-18 audit), 这样才是真正的"所有 scan"。view 数取
``Cameras/pair.txt`` 第一行 (全数据集共用一份标定, 目前是 49)。light 固定尝试 0~6 这 7 种,
每一个 (scan, view, light) 组合先看 ``Rectified_raw/<scan>/rect_{view+1:03d}_{light}_r5000.png``
在不在盘上, 不在就跳过不算失败 —— 不假设每个 scan 都有全部 7 个光照, 让磁盘说了算。

## 分辨率: 原生跑, 不是先跑 518x420 再放大

``log/prior_cache`` 里 VGGT/DA3 固定跑在 ``cfg.prior.target_wh`` (518x420, 见
prior-cache-subsystem 那条记忆), 因为它要跟 VGGT 的点云对齐、且融合环节本来就要重采样。
这里没有这层限制, 实测原生分辨率 (process_res=1600, DA3 内部按 patch14 取整到约
1596x1204) 单张只要约 0.6s、显存约 4.2GiB (A100 80GB 绰绰有余), 比 518 档只慢 5 倍左右,
不是想象中的量级, 所以默认按 ``--process-res`` (默认 1600, 即长边) 原生跑, 出来的
~1596x1204 再 cv2.resize (线性, 只差几个像素) 到严格的 1200x1600 存盘。想省时间/空间
换回旧档可以 ``--process-res 518``。

## 存储: float16 压缩

DA3 输出是无量纲的相对深度 (量级 O(1)), float16 的相对精度 (~1e-3) 比这套数据后续任何
标定/仿射步骤引入的误差 (百分之几) 小得多, 换来单文件约 670KB (float32 是 6.3MB, 10 倍)。
42532 个组合全部建满约 28GB, float32 要 268GB —— 磁盘紧张就别用 ``--dtype float32``。

## 用法

    python scripts/build_da3_cache_all.py                    # 全量, 断点续跑
    python scripts/build_da3_cache_all.py --dry-run           # 只统计, 不加载模型/不跑
    python scripts/build_da3_cache_all.py --scans 1 4 9        # 只建这几个 scan
    python scripts/build_da3_cache_all.py --limit 50           # 调试: 只跑前 50 个组合
    python scripts/build_da3_cache_all.py --force               # 忽略已存在的文件, 全部重建

失败 (读图/推理异常) 记到 ``<cache_root>/_failures_<时间戳>.csv``, 不会带崩整轮; 结尾打印
完成率与失败清单路径。Ctrl-C 或作业超时后重新跑同一条命令会跳过已完成的文件继续。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base.config import ProjectPaths, build_mvs_config  # noqa: E402

NATIVE_H, NATIVE_W = 1200, 1600
ALL_LIGHTS = tuple(range(7))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scans", type=int, nargs="*", default=None,
                   help="只建这些 scan 编号 (默认: Rectified_raw/ 下的全部)")
    p.add_argument("--lights", type=int, nargs="*", default=list(ALL_LIGHTS),
                   help="尝试的光照条件 (默认 0~6, 磁盘上没有的组合自动跳过)")
    p.add_argument("--views", type=int, nargs="*", default=None,
                   help="0 起算的视角号 (默认: 按 Cameras/pair.txt 第一行取 0..N-1)")
    p.add_argument("--process-res", type=int, default=1600,
                   help="DA3 处理分辨率 (长边像素, upper_bound_resize, 内部再取整到 14 的倍数); "
                        "默认原生 1600, 想省时间用旧档传 518")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--force", action="store_true", help="已存在的文件也重新计算覆盖")
    p.add_argument("--dry-run", action="store_true", help="只统计目标组合数, 不装模型不跑")
    p.add_argument("--limit", type=int, default=0, help="调试用: 只处理前 N 个待办组合 (0=不限)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default=None, help="覆盖 cfg.paths.da3_cache_path")
    p.add_argument("--dtu-root", default=None, help="覆盖 cfg.paths.dtu_train_root")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def num_views(dtu_root: Path) -> int:
    return int((dtu_root / "Cameras" / "pair.txt").read_text().split()[0])


def scan_list(dtu_root: Path) -> list[str]:
    rect = dtu_root / "Rectified_raw"
    return sorted((d.name for d in rect.iterdir() if d.is_dir()),
                 key=lambda s: (len(s), s))     # scan2 < scan10 (数值序, 不是字典序)


def image_path(dtu_root: Path, scan: str, view: int, light: int) -> Path:
    return dtu_root / "Rectified_raw" / scan / f"rect_{view + 1:03d}_{light}_r5000.png"


def cache_path(cache_root: Path, scan: str, view: int, light: int) -> Path:
    return cache_root / scan / f"da3_{view:04d}_{light}.npz"


def build_todo(dtu_root: Path, cache_root: Path, scans: list[str], views: list[int],
               lights: list[int], force: bool) -> tuple[list[tuple[str, int, int]], dict]:
    """target 全集对磁盘现状取差集 (build_prior_cache_all.py 同款思路): 图不存在的组合不算
    "缺", 直接从 todo 里剔除, 不进失败统计。"""
    todo: list[tuple[str, int, int]] = []
    n_no_image = n_done = 0
    for scan in scans:
        for view in views:
            for light in lights:
                if not image_path(dtu_root, scan, view, light).is_file():
                    n_no_image += 1
                    continue
                dst = cache_path(cache_root, scan, view, light)
                if dst.is_file() and not force:
                    n_done += 1
                    continue
                todo.append((scan, view, light))
    return todo, {"n_no_image": n_no_image, "n_already_done": n_done}


def save_depth(path: Path, depth: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:          # savez_compressed 会给不以 .npz 结尾的名字再加 .npz,
        np.savez_compressed(fh, depth=depth, **meta)   # 必须传文件对象而不是路径字符串
    os.replace(tmp, path)


def _fmt(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main() -> None:
    args = parse_args()
    cfg = build_mvs_config()
    dtu_root = Path(args.dtu_root) if args.dtu_root else cfg.paths.dtu_train_root
    cache_root = Path(args.out) if args.out else cfg.paths.da3_cache_path

    scans = [f"scan{s}" for s in args.scans] if args.scans else scan_list(dtu_root)
    missing_scans = [s for s in scans if not (dtu_root / "Rectified_raw" / s).is_dir()]
    if missing_scans:
        raise SystemExit(f"--scans 里这些在 Rectified_raw/ 下找不到: {missing_scans}")
    views = args.views if args.views is not None else list(range(num_views(dtu_root)))
    lights = args.lights

    todo, stats = build_todo(dtu_root, cache_root, scans, views, lights, args.force)
    total_slots = len(todo) + stats["n_already_done"]
    print(f"[da3-cache] dtu_root={dtu_root}  cache_root={cache_root}")
    print(f"[da3-cache] scans={len(scans)}  views/scan={len(views)}  lights={lights}  "
         f"process_res={args.process_res}  dtype={args.dtype}")
    print(f"[da3-cache] 目标组合 {total_slots} (磁盘上有图的)  "
         f"已完成 {stats['n_already_done']}  待建 {len(todo)}  "
         f"(图不存在, 不计入待建/失败: {stats['n_no_image']})")
    if args.limit:
        todo = todo[: args.limit]
        print(f"[da3-cache] --limit {args.limit}: 只跑前 {len(todo)} 个")
    if args.dry_run or not todo:
        print("[da3-cache] dry-run 或无待办, 不装模型退出。")
        return

    import torch
    import models.norm_fill as nf

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("需要 CUDA (A100 作业请确认 --gres=gpu:1 生效)")
    print("[da3-cache] 装 DA3 ...", flush=True)
    da3_model = nf.load_da3_model(ProjectPaths().da3_weights_file, device)

    cache_root.mkdir(parents=True, exist_ok=True)
    fail_csv = cache_root / f"_failures_{datetime.now():%Y%m%d_%H%M%S}.csv"
    n_ok = n_fail = 0
    t_start = time.time()

    with fail_csv.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["scan", "view", "light", "error"])
        for i, (scan, view, light) in enumerate(todo, 1):
            t0 = time.time()
            try:
                img_path = image_path(dtu_root, scan, view, light)
                image = np.asarray(Image.open(img_path).convert("RGB"))
                depth = nf._get_depth_da3(image, da3_model, args.process_res)
                if depth.shape != (NATIVE_H, NATIVE_W):
                    depth = cv2.resize(depth, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_LINEAR)
                depth = depth.astype(args.dtype)
                meta = {"scan": np.asarray(scan), "view": np.asarray(view, np.int32),
                       "light": np.asarray(light, np.int32),
                       "process_res": np.asarray(args.process_res, np.int32),
                       "raw_prediction_hw": np.asarray(depth.shape, np.int32)}
                save_depth(cache_path(cache_root, scan, view, light), depth, meta)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 - 单样本失败不能带崩整轮 (跑几小时的作业)
                n_fail += 1
                wr.writerow([scan, view, light, f"{type(exc).__name__}: {exc}"])
                fh.flush()
                print(f"    !! {scan} v{view} l{light}: {type(exc).__name__}: {exc}", flush=True)
            if i % args.log_every == 0 or i == len(todo):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1e-6)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"[da3-cache] {i}/{len(todo)}  ok={n_ok} fail={n_fail}  "
                     f"{rate:.2f}/s  用时 {_fmt(elapsed)}  预计剩余 {_fmt(eta)}  "
                     f"最新 {scan} v{view} l{light} {time.time()-t0:.2f}s", flush=True)

    print(f"[da3-cache] 完成: ok={n_ok} fail={n_fail} / {len(todo)}  总用时 {_fmt(time.time()-t_start)}")
    if n_fail:
        print(f"[da3-cache] 失败清单: {fail_csv}")
    else:
        fail_csv.unlink(missing_ok=True)   # 全部成功就不留一个空的失败表


if __name__ == "__main__":
    main()
