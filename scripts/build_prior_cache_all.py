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

from base.config import ProjectPaths, build_mvs_config
from data.dtu import DTUMVSDataset
from models.pre_prior import (PIPELINE_VERSION, PriorPrecomputer,
                              cache_signature_from_file, save_prior)

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
    except Exception:
        return None
    return sig


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
    for scan in scans:
        sp = split_of.get(scan, "other")
        rs = resize.get(sp, resize["other"])
        st = stat.setdefault(sp, {"scans": 0, "want": 0, "have": 0, "stale": 0,
                                  "unscaled": 0, "todo": 0})
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
                        else:
                            need = False
                if need:
                    st["todo"] += 1
                    todo.append((scan, sp, light, view, src, rs))
    print(f"[build] 差集扫描用时 {time.time()-t0:.1f}s")

    print(f"\n{'split':<6} {'scans':>5} {'目标':>7} {'已有':>7} {'过期':>6} "
          f"{'未标尺':>7} {'待建':>7} {'倍率':>5} {'预估':>10} {'磁盘':>8}")
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
              f"{st['unscaled']:>7} {st['todo']:>7} {rs:>5} {_fmt(sec):>10} {mb/1024:>7.1f}G")
    print(f"{'合计':<6} {len(scans):>5} {sum(s['want'] for s in stat.values()):>7} "
          f"{sum(s['have'] for s in stat.values()):>7} "
          f"{sum(s['stale'] for s in stat.values()):>6} "
          f"{sum(s['unscaled'] for s in stat.values()):>7} "
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

    rpt_path = Path(args.report) if args.report else \
        Path("log/rebuild") / f"build_all_{datetime.now():%Y%m%d_%H%M%S}.csv"
    rpt_path.parent.mkdir(parents=True, exist_ok=True)
    rpt = open(rpt_path, "a", newline="")
    wr = csv.writer(rpt)
    wr.writerow(["ts", "scan", "split", "view", "light", "resize", "status",
                 "sfm_valid", "sfm_scale", "num_pairs", "depth_median", "h", "w", "sec"])
    print(f"[build] 逐样本报告 -> {rpt_path}")
    print(f"[build] 装 VGGT + DA3 ...", flush=True)

    pre = PriorPrecomputer(torch.device(args.device), image_target_wh=twh)
    built = invalid = failed = 0
    errors: list[str] = []
    t0 = time.time()
    try:
        for n, (scan, sp, light, view, _src, rs) in enumerate(todo, 1):
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
                             "failed", "", "", "", "", "", "", f"{time.time()-ts:.1f}"])
                rpt.flush()
                print(f"[build] !! 失败 {errors[-1]}", flush=True)
                continue

            d = prior["depth_prior"]
            v = d[np.isfinite(d) & (d > 0)]
            ok = prior["sfm_valid"] > 0.5
            if not ok:
                invalid += 1
                if args.quarantine:
                    # sfm_valid=0 照样写主缓存: 删了 dataloader 会 FileNotFoundError,
                    # 而 dtu.py 读到 sfm_valid=0 会自己把 prior_valid 置 0 退回 global-only
                    save_prior(cache.with_name(cache.name + "_quarantine") / scan / dst.name,
                               prior)
            else:
                built += 1
            save_prior(dst, prior)
            wr.writerow([f"{datetime.now():%F %T}", scan, sp, view, light, rs,
                         "ok" if ok else "scale_invalid",
                         int(ok), f"{prior['sfm_scale']:.6g}", int(prior["sfm_num_pairs"]),
                         f"{float(np.median(v)) if v.size else -1:.2f}",
                         int(prior["prior_h"]), int(prior["prior_w"]),
                         f"{time.time()-ts:.1f}"])
            if n % args.log_every == 0 or n == len(todo):
                rpt.flush()
                el = time.time() - t0
                eta = el / n * (len(todo) - n)
                print(f"[build] {n}/{len(todo)}  {scan} v{view} l{light}  "
                      f"{el/n:.2f}s/个  已用 {_fmt(el)}  剩余 {_fmt(eta)}  "
                      f"标尺失败 {invalid}  异常 {failed}", flush=True)
    except KeyboardInterrupt:
        print("\n[build] 收到 Ctrl-C —— 已写完的文件都是完整的 (原子写), "
              "重跑同一条命令续跑")
    finally:
        rpt.close()

    el = time.time() - t0
    done = built + invalid + failed
    print(f"\n[build] 完成 {done}/{len(todo)}: {built} 成功, {invalid} 标尺失败 "
          f"(已写主缓存但 sfm_valid=0, 网络会退回 global-only), {failed} 异常, "
          f"用时 {_fmt(el)}")
    if errors:
        print(f"[build] 异常样本 (前 10 个, 全量见 {rpt_path}):")
        for e in errors[:10]:
            print(f"[build]   {e}")
    print(f"[build] 报告: {rpt_path}")
    print(f"[build] 复查完整性: python scripts/build_prior_cache_all.py "
          f"--scans {args.scans} --lights {args.lights} --dry-run")


if __name__ == "__main__":
    main()
