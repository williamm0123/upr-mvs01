"""补全 log/prior_cache —— 按 (scan, ref_view, light) 枚举目标文件, 缺什么建什么。

## 为什么要它 (2026-08-15 全量元数据扫描, 27979 个文件)

| 组 | 现状 |
|---|---|
| 79 个 train scan | 各 343 个 (49 视角 × 7 光照) —— 齐 |
| 18 个 val scan | 各 49 个 (仅 light 3) —— 齐 |
| **22 个 test scan** | **一个都没有** —— 整个 split 缺失 |
| 5 个不在任何列表里的 scan (25/26/27/54/73) | 一个都没有 |
| 2336 个 (8.35%) | `sfm_valid=0` (标尺失败, 网络会退回 global-only), 集中在 scan85-92 |

全部是 `pipeline_version=3` / `target_wh=518×420` / `num_views=3`, 没有损坏文件。

## 和 scripts/rebuild_priors.py 的区别

那个脚本按 `DTUMVSDataset.metas` 枚举, 于是"该建哪些"由 listfile + mode 决定:
mode='train' 才有 7 个光照, 否则只有 light 3。想补 test 就得换 list 和 mode 跑
好几遍, 而且漏了谁不会有人告诉你。这个脚本反过来 —— **先算出目标文件名的全集**
(scan × pair.txt 的 49 个 ref_view × 光照策略), 再和磁盘对差集, 所以"完整"是
可验证的, `--dry-run` 直接把缺口按 split 列出来。

另外两个实际差别:
  * **每个样本都有 try/except**: 旧脚本一个样本抛异常就带走整轮 (跑了 3 小时,
    第 900 个炸掉, 前面白跑)。这里坏样本记进 CSV 继续跑, 结尾汇总。
  * **没有 --slim**: 旧脚本的 slim 把 conf 存成 uint8 (0..255), 但 `save_prior`
    紧接着 `astype(np.float32)` —— 存进去的置信度是 0..255 而不是 0..1, 网络读
    到的是放大 255 倍的 conf。所以这里不提供 slim。

## 尺度回退链 (2026-08-15 加)

未标尺的先验绝不能进训练: 它停在 VGGT 的归一化尺度上 (中位数 ~1 而不是 ~650mm),
网络会拿它当 stage1 的候选中心。所以标尺失败时按顺序回退:

  1. **换光照重跑 SfM** —— 光照 0..6 依次试, 同一个参考视角。几何完全没变, 变的
     只是 SIFT 能不能找到足够的匹配点。**VGGT/DA3 不重算**: 它们对光照不敏感,
     而且标尺失败时存下来的 depth_prior 就是未标尺的稠密深度 (scale=1.0), 直接
     乘上新尺度即可。这条路求出来的尺度不是"借"的 —— 它是拿本样本自己的稠密
     深度和那批稀疏点求的比值中位数。2026-08-15 拿 GT 深度验过 scan48 v42/44/45:
     GT/prior 中位比 0.987~0.992 (偏 0.8~1.3%), 绝对误差中位 5.2~8.3mm, 和同
     scan 自有尺度的样本 (0.4~1.6%, 3.9~10.6mm) 没有区别。
  2. **借相邻视角的尺度** —— 七个光照全失败才走这条。同 scan 内按"同光照优先 →
     视角距离近优先 → 同距离时优先前一个视角"挑, 也就是先向前追溯, 前面没有再
     向后找 (view 0 失败就等 view 1 建完借它的)。这是近似: 实测相邻视角的尺度
     差约 5% (scan48 v43=680.1 vs v44 的真值 648.4)。借来的尺度只从"自有/换光照"
     得到的视角里挑, 不会借二手的, 免得误差累积。
  3. 整个 scan 都没有有效尺度才认输, 留 sfm_valid=0 (网络退回 global-only)。

**尺度合理性闸门** (2026-08-16 加): ``num_pairs >= 20`` 只说明"有支撑", 不说明
"标对了"。实测 32 个样本的尺度是本 scan 中位的 1.6~2.4 倍, **全部落在 v37/v38/v39**
—— pair.txt 里 v38 的 source 评分是 39(2189) 37(1834) 40(825) 36(772), 后两个只有
别处的 1/3, 三角化条件最差, SIFT 匹配错一批就整体偏 2 倍。这种样本 sfm_valid=1,
``dtu.py`` 的 in_range 门 [128, 2802]mm 也拦不住, 会带着 2.3 倍错误的公制先验进训练。
所以每个候选尺度都要过一遍 ``cam.txt`` 的深度假设范围: 乘完之后 depth 中位数必须落在
[dmin*0.9, dmax*1.05] (默认, 见 --scale-band) 内 —— 全库 28844 个有尺度样本的 99.9%
分位是 933.9, 而 dmax=934, 所以正常样本贴着这条线以下, 离群的从 987 起跳。不过闸门
就走回退链: 换光照 -> 借邻视角。scan71 v38 的 l0/l1/l2/l5/l6 都是 1418, 但 l3/l4 是
623, 换光照正好能救。

来源记在 npz 的 ``scale_source`` (0 自有 / 1 换光照 / 2 借邻视角) +
``scale_light`` + ``scale_ref_view`` 里, 也逐样本写进 CSV。

## 用法

    python scripts/build_prior_cache_all.py --dry-run          # 只报缺口和预估
    python scripts/build_prior_cache_all.py                    # 建, 已有的跳过
    python scripts/build_prior_cache_all.py --scans test       # 只补 test split
    python scripts/build_prior_cache_all.py --redo-unscaled    # 顺带重算 sfm_valid=0 的
    python scripts/build_prior_cache_all.py --lights all       # 每个 scan 都建 7 个光照
    python scripts/build_prior_cache_all.py --shard 0/2        # 两块卡分片并行

中断后重跑同一条命令即可续跑 (`save_prior` 是 tmp+os.replace 原子写, 不会留半个
文件)。长跑建议:

    nohup python -u scripts/build_prior_cache_all.py > log/rebuild/all.log 2>&1 &
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models.sfm as S
from base.config import ProjectPaths, build_mvs_config
from data.dtu import DTUMVSDataset
from models.pre_prior import (PIPELINE_VERSION, PriorPrecomputer,
                              cache_signature_from_file, load_prior, save_prior)

# 每样本耗时/体积的经验值 (RTX 5060 Ti, nviews=5, 2026-08-15 实测), 只用来给
# --dry-run 报预估; 真实 ETA 跑起来后按实际速率算。体积几乎和 resize 无关 ——
# 先验是从 518x420 升采样上来的平滑图, 存 1200x1600 也就比 600x800 大一点点。
SEC_PER_SAMPLE = {0.5: 3.3, 1.0: 4.2}
MB_PER_SAMPLE = {0.5: 3.7, 1.0: 3.8}
ALL_LIGHTS = tuple(range(7))
# dtu.py 的 build_list: mode='train' 才展开 7 个光照, 其余模式钉死 light 3。
# 光照策略必须跟它一致, 否则要么建了没人读, 要么训练时 FileNotFoundError。
VAL_TEST_LIGHTS = (3,)


def _fmt(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _scan_key(name: str) -> tuple[int, str]:
    digits = "".join(c for c in name if c.isdigit())
    return (int(digits) if digits else 1 << 30, name)


def read_pairs(root: Path) -> list[tuple[int, list[int]]]:
    """Cameras/pair.txt —— 全数据集共用一份 (不是每个 scan 一份), 49 个 ref_view。"""
    with open(root / "Cameras" / "pair.txt") as f:
        n = int(f.readline())
        out = []
        for _ in range(n):
            ref = int(f.readline().rstrip())
            src = [int(x) for x in f.readline().rstrip().split()[1::2]]
            out.append((ref, src))
    return out


def read_list(path: Path) -> list[str]:
    if not Path(path).exists():
        return []
    return [ln.strip() for ln in open(path) if ln.strip()]


def probe(path: Path) -> dict | None:
    """元数据 + sfm_valid。

    版本/分辨率那几个字段直接用 pre_prior 的共享实现 (训练启动时的过期判定走的
    是同一条路), 这里只多读一个 sfm_valid —— 未标尺的样本要单独统计。
    """
    sig = cache_signature_from_file(path)
    if sig is None:
        return None
    try:
        with np.load(path) as z:
            sig["sfm_valid"] = float(z["sfm_valid"]) if "sfm_valid" in z.files else 0.0
            sig["sfm_scale"] = float(z["sfm_scale"]) if "sfm_scale" in z.files else 0.0
    except Exception:
        return None
    return sig


def view_depth_ranges(root: Path, ndepths: int = 192) -> dict[int, tuple[float, float]]:
    """每个参考视角的深度假设范围 —— 和 data/dtu.py 的 depth_values 同一套算法。"""
    out: dict[int, tuple[float, float]] = {}
    for f in sorted((root / "Cameras").glob("*_cam.txt")):
        try:
            v = int(f.stem.split("_")[0])
            L = f.read_text().splitlines()
            dmin = float(L[11].split()[0])
            dint = float(L[11].split()[1]) * 1.06
        except Exception:
            continue
        out[v] = (dmin, dmin + dint * ndepths)
    return out


def scan_median_scale(cache: Path, scan: str, ranges: dict, band: tuple[float, float]) -> float:
    """该 scan 所有"绝对闸门内"的尺度的中位数 —— 相对闸门的基准。

    同一个 scan 的 49 个视角看的是同一个物体、同一个转台半径, 尺度天然接近
    (实测四分位 628~719, 同 scan 内更紧)。所以"比本 scan 中位大 40%"基本就是
    三角化错了。用中位数而不是均值: 一个 scan 里混几个离群值也不影响基准。
    返回 0 表示没有可用基准 (新 scan / 全是坏的), 调用方跳过相对判据。
    """
    vals = [v for v in scan_scale_book(cache, scan, ranges, band).values()]
    return float(np.median(vals)) if len(vals) >= 8 else 0.0


def scale_plausible_rel(scale: float, scan_med: float, rel: float) -> bool:
    """尺度相对本 scan 中位的偏离是否可接受。scan_med<=0 或 rel<=0 时不判。"""
    if scan_med <= 0 or rel <= 0 or not np.isfinite(scale) or scale <= 0:
        return True
    return (1.0 / (1.0 + rel)) <= (scale / scan_med) <= (1.0 + rel)


def scale_plausible(median_depth: float, view: int, ranges: dict, band: tuple[float, float]) -> bool:
    """乘完尺度的 depth 中位数是否落在该视角的深度假设范围内 (带余量)。"""
    if view not in ranges or not np.isfinite(median_depth) or median_depth <= 0:
        return True                       # 没有 cam.txt 就不判, 免得误杀
    dmin, dmax = ranges[view]
    return band[0] * dmin <= median_depth <= band[1] * dmax


def scale_from_other_light(ds, idx: int, meta: tuple, light: int, resize: float,
                          dense_depth: np.ndarray, min_pairs: int):
    """在另一个光照下重跑 SfM 三角化, 用它的稀疏点给 ``dense_depth`` 定标。

    只跑 ``generate_sparse_depth_from_sample`` (SIFT + 匹配 + 三角化), 不碰
    VGGT/DA3 —— 换光照不改变几何, 稠密深度照用。约 1.4 s/次 @resize 1.0。
    返回 ``(scale, info, src_weights)``, ``info["valid"]`` 说明成不成。
    """
    scan, _l, view, src = meta
    ds.metas[idx] = (scan, light, view, src)      # 临时换光照, precrop_inputs 读它
    try:
        pc = ds.precrop_inputs(idx, resize_scale=resize)
    finally:
        ds.metas[idx] = meta
    out = S.generate_sparse_depth_from_sample(pc, ref_idx=0)
    scale, info = S.metric_scale_from_sparse(
        dense_depth, out["sparse_depth"], out["valid_mask"], min_pairs=min_pairs)
    return scale, info, out["source_weights"]


def scan_scale_book(cache: Path, scan: str, ranges: dict, band: tuple[float, float]
                    ) -> dict[tuple[int, int], float]:
    """该 scan 缓存里所有**自有**尺度 (scale_source<2), 作为邻域回退的候选池。

    只收自有的: 借来的尺度再借一次会让误差累积。元数据读取 0.18 ms/个, 一个
    scan 最多 343 个文件, 60 ms。
    """
    book: dict[tuple[int, int], float] = {}
    d = cache / scan
    if not d.is_dir():
        return book
    for f in d.glob("prior_*.npz"):
        parts = f.stem.split("_")
        if len(parts) != 3:
            continue
        try:
            with np.load(f) as z:
                if "sfm_valid" not in z.files or float(z["sfm_valid"]) <= 0.5:
                    continue
                src = float(z["scale_source"]) if "scale_source" in z.files else 0.0
                if src >= 1.5:                    # 借来的, 不做二手来源
                    continue
                sc = float(z["sfm_scale"])
                v_ = int(parts[1])
                if not scale_plausible(sc, v_, ranges, band):
                    continue                      # 离群的尺度不进候选池
                book[(v_, int(parts[2]))] = sc
        except Exception:
            continue
    return book


def neighbor_candidates(book: dict[tuple[int, int], float], view: int, light: int):
    """按"同光照优先 -> 视角距离近优先 -> 同距离时优先前一个视角"排好的候选尺度。

    返回列表而不是单个: 借来的尺度也要过合理性闸门, 第一个不行就试下一个。
    """
    return sorted(book.items(),
                  key=lambda kv: (0 if kv[0][1] == light else 1,
                                  abs(kv[0][0] - view),
                                  0 if kv[0][0] < view else 1))


def resolve_scan_set(spec: str, splits: dict[str, list[str]], disk: list[str]) -> list[str]:
    sel: list[str] = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok == "all":
            sel += disk
        elif tok == "lists":
            sel += splits["train"] + splits["val"] + splits["test"]
        elif tok in splits:
            sel += splits[tok]
        elif tok == "other":
            listed = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
            sel += [s for s in disk if s not in listed]
        else:
            sel.append(tok)
    seen, out = set(), []
    for s in sorted(dict.fromkeys(sel), key=_scan_key):
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="补全 prior 缓存: 遍历 scan × ref_view × light, 缺什么建什么",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", default="all",
                    help="all(磁盘上全部) | lists(train+val+test 列表) | train | val | "
                         "test | other(不在任何列表里的) | 显式 scan 名, 逗号分隔")
    ap.add_argument("--lights", default="auto",
                    help="auto = train split 建 7 个光照, val/test/其他只建 light 3 "
                         "(和 data/dtu.py 的 build_list 一致); all = 一律 7 个; "
                         "也可以给 '0,3,6' 这样的显式清单")
    ap.add_argument("--resize", default="train:0.5,val:0.5,test:1.0,other:0.5",
                    help="每个 split 的建缓存分辨率倍率 (1.0 = 原生 1200x1600)。"
                         "ViT 恒在 target_wh 上跑, 这个只决定存图大小和 SfM 标尺"
                         "用的图有多清楚 —— test 用 1.0 是因为推理跑 0.8, 降采样"
                         "无损而升采样造不出细节")
    ap.add_argument("--num-views", type=int, default=None,
                    help="1 ref + (N-1) src, 默认取 cfg.train.num_views。只影响先验"
                         "*质量* (src 越多 SfM 三角化点越多), 不是兼容性判据")
    ap.add_argument("--target-w", type=int, default=518)
    ap.add_argument("--target-h", type=int, default=420)
    ap.add_argument("--redo-unscaled", action="store_true",
                    help="把已存在但 sfm_valid=0 的样本也重建 (当前 2336 个)")
    ap.add_argument("--force", action="store_true", help="忽略已有文件, 全部重算")
    ap.add_argument("--check", choices=["meta", "exists"], default="meta",
                    help="meta(默认): 读元数据, 版本/分辨率不符或文件损坏都重建; "
                         "exists: 只看文件在不在 (快, 但放过旧版缓存)")
    ap.add_argument("--scale-fallback", choices=["full", "light", "off"], default="full",
                    help="标尺失败时的回退: full(默认)=换光照 + 借邻视角; "
                         "light=只换光照; off=不回退, 直接写 sfm_valid=0")
    ap.add_argument("--scale-band", default="0.9,1.05",
                    help="尺度合理性闸门: 乘完的 depth 中位数必须落在 "
                         "[dmin*lo, dmax*hi] 内 (dmin/dmax 来自 cam.txt 的深度假设范围)。"
                         "设 '0,99' 等于关掉")
    ap.add_argument("--scale-rel", type=float, default=0.30,
                    help="相对闸门: 尺度偏离本 scan 中位超过这个比例就否决 (默认 0.30)。"
                         "绝对闸门在 900~1000mm 这段分不开真假 —— 实测 scan48 v39 的 "
                         "1020 (本 scan 中位 732) 和 scan8 v37 的 986 (中位 620) 都能过"
                         "绝对闸门, 但对 GT 分别远了 43% 和 61%。设 0 关掉")
    ap.add_argument("--redo-bad-scale", action="store_true",
                    help="把已存在但尺度落在闸门外的文件也重建 (当前 25 个, 全在 v37/38/39)")
    ap.add_argument("--min-pairs", type=int, default=20,
                    help="metric_scale_from_sparse 的支撑像素门槛 (默认 20)。"
                         "调低会让更多样本'通过', 但十几个像素的中位数比值噪声很大, "
                         "标错尺比不标尺更糟")
    ap.add_argument("--quarantine", action="store_true",
                    help="标尺失败时额外往 log/prior_cache_quarantine/ 存一份副本 "
                         "(默认关: CSV 报告 + npz 里的 sfm_valid 已经记了同样的事, "
                         "副本每个 3.6MB)")
    ap.add_argument("--shard", default="0/1", help="i/N, 只做第 i 片")
    ap.add_argument("--limit", type=int, default=0, help="只做前 N 个 (调试)")
    ap.add_argument("--dry-run", action="store_true", help="只报缺口和预估, 不建")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--profile", choices=["local", "umhpc"], default=None)
    ap.add_argument("--report", default="",
                    help="逐样本 CSV, 默认 log/rebuild/build_all_<时间戳>.csv")
    args = ap.parse_args()

    cfg = build_mvs_config(profile=args.profile)
    paths = ProjectPaths()
    root = Path(cfg.paths.dtu_train_root)
    cache = Path(paths.prior_cache_path)
    nviews = args.num_views or cfg.train.num_views
    twh = (args.target_w, args.target_h)
    si, sn = (int(x) for x in args.shard.split("/"))

    # 用*原始*列表, 不是 audit 产出的 *_clean.txt —— 重建的目的就是把坏 scan 修好
    splits = {"train": read_list(cfg.paths.train_list_file),
              "val": read_list(cfg.paths.val_list_file),
              "test": read_list(cfg.paths.test_list_file)}
    split_of = {s: sp for sp, names in splits.items() for s in names}
    disk = sorted((d.name for d in (root / "Rectified_raw").iterdir() if d.is_dir()),
                  key=_scan_key)

    resize = {"train": 0.5, "val": 0.5, "test": 1.0, "other": 0.5}
    for kv in filter(None, (x.strip() for x in args.resize.split(","))):
        k, v = kv.split(":")
        resize[k.strip()] = float(v)

    if args.lights == "auto":
        lights_for = lambda sp: ALL_LIGHTS if sp == "train" else VAL_TEST_LIGHTS
    elif args.lights == "all":
        lights_for = lambda sp: ALL_LIGHTS
    else:
        explicit = tuple(int(x) for x in args.lights.split(","))
        lights_for = lambda sp: explicit

    band = tuple(float(x) for x in args.scale_band.split(","))
    ranges = view_depth_ranges(root)
    _dmin = min(a for a, _ in ranges.values()); _dmax = max(b for _, b in ranges.values())
    print(f"[build] 尺度闸门: depth 中位数须落在 [{band[0]*_dmin:.0f}, {band[1]*_dmax:.0f}] mm "
          f"(cam.txt 深度假设范围 {_dmin:.0f}~{_dmax:.0f} × {band})")

    scans = resolve_scan_set(args.scans, splits, disk)
    missing_on_disk = [s for s in scans if not (root / "Rectified_raw" / s).is_dir()]
    if missing_on_disk:
        print(f"[build] 数据集里没有这些 scan, 跳过: {missing_on_disk}")
        scans = [s for s in scans if s not in missing_on_disk]
    pairs = read_pairs(root)

    print(f"[build] === prior cache 补全 ===")
    print(f"[build] 缓存目录 {cache}")
    print(f"[build] scans={len(scans)}  ref_views={len(pairs)}  num_views={nviews} "
          f"(1 ref + {nviews-1} src)  target_wh={twh}  pipeline_version={PIPELINE_VERSION}")
    print(f"[build] 光照策略={args.lights}  分辨率倍率={resize}  检查方式={args.check}"
          f"{'  [--force]' if args.force else ''}"
          f"{'  [--redo-unscaled]' if args.redo_unscaled else ''}")

    # ---------------- 1) 枚举目标全集, 对差集 ----------------
    t0 = time.time()
    todo: list[tuple[str, str, int, int, list[int], float]] = []  # scan,split,light,view,src,resize
    stat = {}   # split -> dict(want/have/stale/unscaled/todo)
    scan_med_cache: dict[str, float] = {}
    for scan in scans:
        sp = split_of.get(scan, "other")
        rs = resize.get(sp, resize["other"])
        scan_med_cache[scan] = scan_median_scale(cache, scan, ranges, band)
        st = stat.setdefault(sp, {"scans": 0, "want": 0, "have": 0, "stale": 0,
                                  "unscaled": 0, "badscale": 0, "todo": 0})
        st["scans"] += 1
        for light in lights_for(sp):
            for view, src in pairs:
                st["want"] += 1
                f = cache / scan / f"prior_{view:0>4}_{light}.npz"
                need = True
                if not args.force and f.exists():
                    st["have"] += 1
                    if args.check == "exists":
                        need = False
                    else:
                        meta = probe(f)
                        if meta is None:
                            st["stale"] += 1          # 损坏/截断
                        elif not (meta["pipeline_version"] >= PIPELINE_VERSION
                                  and meta["target_w"] == twh[0]
                                  and meta["target_h"] == twh[1]):
                            st["stale"] += 1          # 旧 pipeline 或别的 target_wh
                        elif meta["sfm_valid"] < 0.5:
                            st["unscaled"] += 1
                            need = bool(args.redo_unscaled)
                        elif not (scale_plausible(meta["sfm_scale"], view, ranges, band)
                                  and scale_plausible_rel(meta["sfm_scale"],
                                                          scan_med_cache[scan], args.scale_rel)):
                            # 用 sfm_scale 当 depth 中位数的代理: 稠密图归一化后中位
                            # 数约 1.0 (实测 0.93~1.0), 判离群足够了
                            st["badscale"] += 1
                            need = bool(args.redo_bad_scale)
                        else:
                            need = False
                if need:
                    st["todo"] += 1
                    todo.append((scan, sp, light, view, src, rs))
    print(f"[build] 差集扫描用时 {time.time()-t0:.1f}s")

    print(f"\n{'split':<6} {'scans':>5} {'目标':>7} {'已有':>7} {'过期':>6} "
          f"{'未标尺':>7} {'尺度离群':>8} {'待建':>7} {'倍率':>5} {'预估':>10} {'磁盘':>8}")
    tot_sec = tot_mb = 0.0
    for sp in ("train", "val", "test", "other"):
        if sp not in stat:
            continue
        st = stat[sp]
        rs = resize.get(sp, resize["other"])
        sec = st["todo"] * SEC_PER_SAMPLE.get(rs, 7.5)
        mb = st["todo"] * MB_PER_SAMPLE.get(rs, 4.0)
        tot_sec += sec
        tot_mb += mb
        print(f"{sp:<6} {st['scans']:>5} {st['want']:>7} {st['have']:>7} {st['stale']:>6} "
              f"{st['unscaled']:>7} {st['badscale']:>8} {st['todo']:>7} {rs:>5} "
              f"{_fmt(sec):>10} {mb/1024:>7.1f}G")
    print(f"{'合计':<6} {len(scans):>5} {sum(s['want'] for s in stat.values()):>7} "
          f"{sum(s['have'] for s in stat.values()):>7} "
          f"{sum(s['stale'] for s in stat.values()):>6} "
          f"{sum(s['unscaled'] for s in stat.values()):>7} "
          f"{sum(s['badscale'] for s in stat.values()):>8} "
          f"{len(todo):>7} {'':>5} {_fmt(tot_sec):>10} {tot_mb/1024:>7.1f}G")

    if sn > 1:
        todo = [j for k, j in enumerate(todo) if k % sn == si]
        print(f"[build] 分片 {si}/{sn}: 本片 {len(todo)} 个")
    if args.limit:
        todo = todo[:args.limit]
        print(f"[build] --limit: 只做前 {len(todo)} 个")

    free_gb = os.statvfs(cache.parent if cache.exists() else Path("."))
    free_gb = free_gb.f_bavail * free_gb.f_frsize / 2**30
    print(f"[build] 磁盘剩余 {free_gb:.0f}G")
    if tot_mb / 1024 > free_gb * 0.9:
        print(f"[build] !! 预估写入 {tot_mb/1024:.0f}G 接近可用空间, 先腾地方")

    if not todo:
        print("[build] 没有要建的, 缓存已完整")
        return
    if args.dry_run:
        print("[build] --dry-run: 到此为止")
        return

    # VGGT+DA3 常驻约 5.9 GiB, 峰值 7.3 GiB —— 卡上有别的进程就会 OOM
    try:
        busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
        if busy:
            print(f"[build] !! GPU 上已有进程 (峰值需 7.3 GiB, 可能 OOM):\n     "
                  + "\n     ".join(busy.splitlines()))
    except Exception:
        pass

    # ---------------- 2) 建 ----------------
    # 借 DTUMVSDataset 的 precrop_inputs 读图/相机 (和训练同一条路径, 保证一致),
    # 但 metas 换成我们自己枚举的全集 —— dataset 的 build_list 只会按 listfile +
    # mode 给出一部分。
    ds = DTUMVSDataset(datapath=root, listfile=str(cfg.paths.test_list_file),
                       nviews=nviews, mode="val", random_crop=False)
    ds.metas = [(scan, light, view, src) for scan, _sp, light, view, src, _rs in todo]

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    rpt_path = Path(args.report) if args.report else Path("log/rebuild") / f"build_all_{stamp}.csv"
    rpt_path.parent.mkdir(parents=True, exist_ok=True)
    rpt = open(rpt_path, "a", newline="")
    wr = csv.writer(rpt)
    wr.writerow(["ts", "scan", "split", "view", "light", "resize", "status",
                 "sfm_valid", "sfm_scale", "num_pairs", "depth_median", "h", "w",
                 "scale_source", "scale_light", "scale_ref_view", "light_tries", "sec"])
    # 本轮写过的文件清单 —— 直接喂 rsync --files-from 做增量上传
    list_path = rpt_path.with_suffix(".filelist")
    flist = open(list_path, "a")
    written: set[str] = set()

    def record_written(path: Path) -> None:
        try:                      # 相对仓库根 —— rsync --files-from 要的就是相对路径
            rel = str(Path(path).resolve().relative_to(Path(paths.project_path).resolve()))
        except ValueError:
            rel = str(path)
        if rel not in written:
            written.add(rel)
            flist.write(rel + "\n")
            flist.flush()

    print(f"[build] 逐样本报告 -> {rpt_path}")
    print(f"[build] 改动文件清单 -> {list_path}  (rsync --files-from 可直接用)")
    print(f"[build] 尺度回退: {args.scale_fallback}  min_pairs={args.min_pairs}")
    print(f"[build] 装 VGGT + DA3 ...", flush=True)

    pre = PriorPrecomputer(torch.device(args.device), image_target_wh=twh)
    n_own = n_light = n_neighbor = n_unresolved = n_rejected = failed = 0
    errors: list[str] = []
    # 本 scan 里换光照也救不回来的, 攒到 scan 结束再借邻视角 (那时前后视角都建好了)
    deferred: list[dict] = []
    cur_scan: str | None = None
    t0 = time.time()

    def resolve_deferred(scan: str) -> None:
        """给本 scan 攒下的未标尺样本借一个邻视角的尺度 (借来的也要过闸门)。"""
        nonlocal n_neighbor, n_unresolved
        if not deferred:
            return
        book = scan_scale_book(cache, scan, ranges, band)
        print(f"[build] {scan}: {len(deferred)} 个换光照仍失败, 邻域候选 {len(book)} 个",
              flush=True)
        for job in deferred:
            got = None
            for (ref_v, ref_l), cand in neighbor_candidates(book, job["view"], job["light"]):
                # 借来的尺度同样要过闸门 —— 邻视角自己也可能是 v37/38/39 那种离群
                if (scale_plausible(job["med_dense"] * cand, job["view"], ranges, band)
                        and scale_plausible_rel(cand, scan_med_cache.get(scan, 0.0), args.scale_rel)):
                    got = (cand, ref_v, ref_l)
                    break
            if got is None:
                n_unresolved += 1
                print(f"[build] !! {scan} v{job['view']} l{job['light']}: 整个 scan 都没有"
                      f"能过闸门的尺度 ({len(book)} 个候选), 留 sfm_valid=0", flush=True)
                wr.writerow([f"{datetime.now():%F %T}", scan, job["split"], job["view"],
                             job["light"], job["resize"], "unresolved", 0, 1, job["num_pairs"],
                             f"{job['median']:.3f}", job["h"], job["w"], "", "", "", job["tries"], ""])
                continue
            scale, ref_v, ref_l = got
            prior = load_prior(job["path"])          # 存的是未标尺的稠密深度 (scale=1.0)
            prior["depth_prior"] = np.asarray(prior["depth_prior"], np.float32) * scale
            prior["sfm_valid"] = 1.0
            prior["sfm_scale"] = float(scale)
            prior["scale_source"] = 2.0
            prior["scale_light"] = float(ref_l)
            prior["scale_ref_view"] = float(ref_v)
            save_prior(job["path"], prior)
            record_written(job["path"])
            n_neighbor += 1
            d = prior["depth_prior"]
            v = d[np.isfinite(d) & (d > 0)]
            print(f"[build] {scan} v{job['view']} l{job['light']}: 借 v{ref_v} l{ref_l} 的"
                  f" scale={scale:.1f} -> 中位数 {float(np.median(v)):.1f}mm", flush=True)
            wr.writerow([f"{datetime.now():%F %T}", scan, job["split"], job["view"],
                         job["light"], job["resize"], "ok_neighbor", 1, f"{scale:.6g}",
                         job["num_pairs"], f"{float(np.median(v)):.2f}", job["h"], job["w"],
                         2, ref_l, ref_v, job["tries"], ""])
        rpt.flush()
        deferred.clear()

    try:
        for n, (scan, sp, light, view, _src, rs) in enumerate(todo, 1):
            if cur_scan is not None and scan != cur_scan:
                resolve_deferred(cur_scan)           # 上一个 scan 收尾
            cur_scan = scan
            dst = cache / scan / f"prior_{view:0>4}_{light}.npz"
            ts = time.time()
            try:
                prior = pre.compute(ds.precrop_inputs(n - 1, resize_scale=rs))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                try:                              # OOM 多半是碎片, 清一次再来一遍
                    prior = pre.compute(ds.precrop_inputs(n - 1, resize_scale=rs))
                except Exception as e:
                    prior, err = None, f"OOM->{type(e).__name__}: {e}"
            except Exception as e:
                prior, err = None, f"{type(e).__name__}: {e}"
            if prior is None:
                # 单个样本失败不该带走整轮: 记下来继续, 结尾汇总, 重跑时会再试
                failed += 1
                errors.append(f"{scan}/prior_{view:0>4}_{light}: {err}")
                wr.writerow([f"{datetime.now():%F %T}", scan, sp, view, light, rs,
                             "failed", "", "", "", "", "", "", "", "", "", 0,
                             f"{time.time()-ts:.1f}"])
                rpt.flush()
                print(f"[build] !! 失败 {errors[-1]}", flush=True)
                continue

            # --- 闸门: num_pairs>=20 只说明有支撑, 不说明标对了 ---
            applied = float(prior["sfm_scale"]) if prior["sfm_valid"] > 0.5 else 1.0
            applied = applied if applied > 0 else 1.0
            dense = np.asarray(prior["depth_prior"], np.float32) / applied
            _dv = dense[np.isfinite(dense) & (dense > 0)]
            med_dense = float(np.median(_dv)) if _dv.size else 0.0
            if scan not in scan_med_cache:
                scan_med_cache[scan] = scan_median_scale(cache, scan, ranges, band)
            scan_med = scan_med_cache[scan]
            own_ok = (prior["sfm_valid"] > 0.5
                      and scale_plausible(med_dense * applied, view, ranges, band)
                      and scale_plausible_rel(applied, scan_med, args.scale_rel))
            if prior["sfm_valid"] > 0.5 and not own_ok:
                n_rejected += 1
                print(f"[build] {scan} v{view} l{light}: 自有尺度 {applied:.1f} 过不了闸门 "
                      f"(depth 中位 {med_dense*applied:.0f}mm, 本 scan 中位尺度 {scan_med:.0f}), "
                      f"走回退链", flush=True)
            if not own_ok:                      # 退回未标尺状态, 让回退链在干净的稠密图上试
                prior["depth_prior"] = dense
                prior["sfm_valid"] = 0.0
                prior["sfm_scale"] = 1.0

            # --- 回退链 1: 换光照重跑 SfM (不重算 VGGT/DA3) ---
            tries = 0
            if not own_ok and args.scale_fallback != "off":
                for l2 in [x for x in ALL_LIGHTS if x != light]:
                    tries += 1
                    try:
                        scale, info, sw = scale_from_other_light(
                            ds, n - 1, (scan, light, view, _src), l2, rs, dense, args.min_pairs)
                    except Exception as e:
                        print(f"[build] !! {scan} v{view} 换光照 l{l2} 出错: "
                              f"{type(e).__name__}: {e}", flush=True)
                        continue
                    if info["valid"] and not (scale_plausible(med_dense * scale, view, ranges, band)
                                              and scale_plausible_rel(scale, scan_med, args.scale_rel)):
                        print(f"[build]   l{l2} 的尺度 {scale:.1f} 过不了闸门 "
                              f"(depth 中位 {med_dense*scale:.0f}mm), 继续换", flush=True)
                        continue
                    if info["valid"]:
                        prior["depth_prior"] = dense * scale
                        prior["sfm_valid"] = 1.0
                        prior["sfm_scale"] = float(scale)
                        prior["sfm_num_pairs"] = float(info["num_pairs"])
                        prior["src_weights"] = np.asarray(sw, np.float32)
                        prior["scale_source"] = 1.0
                        prior["scale_light"] = float(l2)
                        prior["scale_ref_view"] = float(view)
                        print(f"[build] {scan} v{view} l{light}: 本光照 SfM 不够, 换 l{l2} "
                              f"成功 (num_pairs={info['num_pairs']}, scale={scale:.1f})",
                              flush=True)
                        break

            d = np.asarray(prior["depth_prior"], np.float32)
            v = d[np.isfinite(d) & (d > 0)]
            med = float(np.median(v)) if v.size else -1.0
            ok = prior["sfm_valid"] > 0.5
            src_tag = int(float(prior.get("scale_source", 0.0)))
            if ok:
                if src_tag == 1:
                    n_light += 1
                else:
                    n_own += 1
            elif args.quarantine:
                # sfm_valid=0 照样写主缓存: 删了 dataloader 会 FileNotFoundError,
                # 而 dtu.py 读到 sfm_valid=0 会自己把 prior_valid 置 0 退回 global-only
                save_prior(cache.with_name(cache.name + "_quarantine") / scan / dst.name, prior)
            save_prior(dst, prior)
            record_written(dst)

            # --- 回退链 2: 攒到 scan 结束再借邻视角 ---
            if not ok and args.scale_fallback == "full":
                deferred.append({"scan": scan, "split": sp, "view": view, "light": light,
                                 "resize": rs, "path": dst, "median": med,
                                 "med_dense": med_dense,
                                 "num_pairs": int(prior["sfm_num_pairs"]),
                                 "h": int(prior["prior_h"]), "w": int(prior["prior_w"]),
                                 "tries": tries})
            elif not ok:
                n_unresolved += 1

            wr.writerow([f"{datetime.now():%F %T}", scan, sp, view, light, rs,
                         "ok" if ok else "deferred",
                         int(ok), f"{prior['sfm_scale']:.6g}", int(prior["sfm_num_pairs"]),
                         f"{med:.2f}", int(prior["prior_h"]), int(prior["prior_w"]),
                         src_tag, int(float(prior.get("scale_light", 0.0))),
                         int(float(prior.get("scale_ref_view", view))), tries,
                         f"{time.time()-ts:.1f}"])
            if n % args.log_every == 0 or n == len(todo):
                rpt.flush()
                el = time.time() - t0
                eta = el / n * (len(todo) - n)
                print(f"[build] {n}/{len(todo)}  {scan} v{view} l{light}  "
                      f"{el/n:.2f}s/个  已用 {_fmt(el)}  剩余 {_fmt(eta)}  "
                      f"换光照救回 {n_light}  借邻视角 {n_neighbor}  "
                      f"待借 {len(deferred)}  无解 {n_unresolved}  异常 {failed}", flush=True)
        resolve_deferred(cur_scan or "")
    except KeyboardInterrupt:
        print("\n[build] 收到 Ctrl-C —— 先把攒下的未标尺样本借完尺度再退出 ...")
        try:
            resolve_deferred(cur_scan or "")
        except Exception as e:
            print(f"[build] !! 收尾时出错: {type(e).__name__}: {e}")
        print("[build] 已写完的文件都是完整的 (原子写), 重跑同一条命令续跑")
    finally:
        rpt.close()
        flist.close()

    el = time.time() - t0
    done = n_own + n_light + n_neighbor + n_unresolved + failed
    print(f"\n[build] 完成 {done}/{len(todo)}, 用时 {_fmt(el)}")
    print(f"[build]   自有尺度      {n_own}")
    print(f"[build]   换光照救回    {n_light}")
    print(f"[build]   借邻视角      {n_neighbor}  (近似, npz 里 scale_source=2)")
    print(f"[build]   自有尺度被闸门否决 {n_rejected}  (随后走了回退链)")
    print(f"[build]   仍未标尺      {n_unresolved}  (sfm_valid=0, 网络退回 global-only)")
    print(f"[build]   异常          {failed}")
    if errors:
        print(f"[build] 异常样本 (前 10 个, 全量见 {rpt_path}):")
        for e in errors[:10]:
            print(f"[build]   {e}")
    print(f"[build] 改动文件 {len(written)} 个 -> {list_path}")
    print(f"[build] 增量上传 umhpc:")
    print(f"[build]   rsync -avP --files-from={list_path} ./ "
          f"<user>@<umhpc>:/scr/user/qinglong/projects/upr-mvs01/")
    print(f"[build] 报告: {rpt_path}")
    print(f"[build] 复查完整性: python scripts/build_prior_cache_all.py "
          f"--scans {args.scans} --lights {args.lights} --dry-run")


if __name__ == "__main__":
    main()
