"""prior cache 审计 / 统一 / 修复。

已知两个问题:
  A. 两代分辨率共存 —— 22 个 test scan 是 1200x1600 (旧 build), train/val 全是
     600x800。dtu.py 的 _match_hw 会 resize 掉, 所以不报错, 但 train 和 test
     的 prior 来自不同 pipeline。
  B. scan85-92 大面积未标尺 —— sfm.metric_scale_from_sparse 在重叠点 <20 时静默
     返回 scale=1.0 (models/sfm.py:273), 于是 depth_prior 留在 VGGT 的任意尺度,
     中位数 ~1.0 而非 ~650mm。norm_fill 没有检查 scale_info["valid"]。

用法:
  audit                 全量扫描, 写 manifest.csv, 打印分组统计
  audit --clean-lists   额外产出剔除坏 scan 的 lists/dtu/*_clean.txt
  audit --quarantine    把坏文件移到 log/prior_cache_quarantine/ (可 --undo 撤回)
  audit --rebuild SCANS 用当前 pipeline 重建这些 scan (需要 GPU + VGGT/DA3)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base.config import ProjectPaths

# 当前 pipeline 的目标分辨率 = pre_prior.build_prior_cache 的 image_target_wh
# 经 inverse_transform_map 还原到 precrop_inputs 的分辨率 (resize_scale=0.5 -> 600x800)
CURRENT_GEN_SHAPE = (600, 800)
UNSCALED_MEDIAN_MAX = 10.0     # 正常 DTU 公制深度 ~400-900mm; <10 必是未标尺


def scan_cache(root: Path) -> list[dict]:
    rows = []
    files = sorted(root.rglob("prior_*.npz"))
    print(f"[audit] 扫描 {len(files)} 个文件 ...")
    for i, f in enumerate(files):
        rec = {"path": str(f.relative_to(root)), "scan": f.parent.name,
               "view": -1, "light": -1, "h": 0, "w": 0,
               "depth_median": float("nan"), "valid_frac": 0.0,
               "flag_unscaled": 0, "flag_oldgen": 0, "flag_unreadable": 0}
        parts = f.stem.split("_")
        if len(parts) == 3:
            rec["view"], rec["light"] = int(parts[1]), int(parts[2])
        try:
            with np.load(f) as z:
                d = z["depth_prior"]
            v = d[np.isfinite(d) & (d > 0)]
            rec["h"], rec["w"] = int(d.shape[0]), int(d.shape[1])
            rec["valid_frac"] = float(v.size / d.size)
            if v.size:
                rec["depth_median"] = float(np.median(v))
                rec["flag_unscaled"] = int(rec["depth_median"] < UNSCALED_MEDIAN_MAX)
            rec["flag_oldgen"] = int((rec["h"], rec["w"]) != CURRENT_GEN_SHAPE)
        except Exception as e:
            rec["flag_unreadable"] = 1
            print(f"[audit]   读不了 {f}: {e}")
        rows.append(rec)
        if (i + 1) % 2000 == 0:
            print(f"[audit]   {i+1}/{len(files)}")
    return rows


def summarize(rows: list[dict], lists_dir: Path) -> tuple[set[str], set[str]]:
    from collections import Counter, defaultdict
    per_scan = defaultdict(lambda: {"n": 0, "unscaled": 0, "oldgen": 0, "bad": 0})
    shapes = Counter()
    for r in rows:
        s = per_scan[r["scan"]]
        s["n"] += 1
        s["unscaled"] += r["flag_unscaled"]
        s["oldgen"] += r["flag_oldgen"]
        s["bad"] += int(r["flag_unreadable"])
        shapes[(r["h"], r["w"])] += 1

    print(f"\n=== 分辨率分布 (当前代 = {CURRENT_GEN_SHAPE}) ===")
    for k, v in shapes.most_common():
        tag = "  <- 当前代" if k == CURRENT_GEN_SHAPE else "  <- 旧代, 需重建"
        print(f"  {k}: {v} files{tag}")

    unscaled_scans = {k for k, v in per_scan.items() if v["unscaled"] > 0}
    oldgen_scans = {k for k, v in per_scan.items() if v["oldgen"] > 0}

    print(f"\n=== 未标尺 (depth median < {UNSCALED_MEDIAN_MAX}mm) ===")
    print(f"  受影响 scan: {len(unscaled_scans)}")
    for k in sorted(unscaled_scans, key=lambda x: -per_scan[x]["unscaled"] / max(per_scan[x]["n"], 1)):
        v = per_scan[k]
        print(f"    {k:10s} {v['unscaled']:4d}/{v['n']:4d} = {100*v['unscaled']/max(v['n'],1):5.1f}%")

    splits = {}
    for name in ("train", "val", "test"):
        p = lists_dir / f"{name}.txt"
        if p.exists():
            splits[name] = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    print(f"\n=== 坏 scan 的 split 归属 ===")
    for name, lst in splits.items():
        u = [s for s in lst if s in unscaled_scans]
        o = [s for s in lst if s in oldgen_scans]
        print(f"  {name:6s} (n={len(lst):3d})  未标尺 {len(u):3d} {sorted(u)[:9]}")
        print(f"  {'':6s}              旧分辨率 {len(o):3d} {sorted(o)[:9]}")
    return unscaled_scans, oldgen_scans


def write_clean_lists(lists_dir: Path, rows: list[dict], max_bad_frac: float) -> None:
    """按坏文件*比例*剔除 scan, 不是"有一个就剔除"。

    全量审计: 51/119 个 scan 至少有一个未标尺文件, 但绝大多数只有 1-3%。
    按"有就剔"会砍掉 79 个训练 scan 里的 49 个。按比例阈值 10% 只剔 10 个,
    残余文件级污染 1.2%。剩下的零散坏文件由 exclude_*.csv 列出, 需要在
    dtu.py 的 meta 构建里过滤才能真正生效 (尚未接线)。
    """
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        p = per[r["scan"]]
        p[0] += 1
        p[1] += int(r["flag_unscaled"]) or int(r["flag_unreadable"])
    drop = {k for k, v in per.items() if v[0] and v[1] / v[0] > max_bad_frac}
    print(f"\n[audit] clean lists: 阈值 {max_bad_frac:.0%} -> 剔除 {len(drop)} 个 scan "
          f"{sorted(drop)}")
    for name in ("train", "val", "trainval"):
        p = lists_dir / f"{name}.txt"
        if not p.exists():
            continue
        orig = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        keep = [s for s in orig if s not in drop]
        out = lists_dir / f"{name}_clean.txt"
        out.write_text("\n".join(keep) + "\n")
        rem_n = sum(per[s][0] for s in keep if s in per)
        rem_b = sum(per[s][1] for s in keep if s in per)
        print(f"[audit]   {out.name}  {len(orig)} -> {len(keep)} scans "
              f"(剔除 {len(orig)-len(keep)}), 残余污染 {rem_b}/{rem_n} = "
              f"{100*rem_b/max(rem_n,1):.2f}%")
        excl = [r for r in rows if r["scan"] in keep
                and (int(r["flag_unscaled"]) or int(r["flag_unreadable"]))]
        ep = lists_dir / f"exclude_{name}.csv"
        with ep.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["scan", "view", "light", "depth_median"])
            w.writeheader()
            w.writerows({k: r[k] for k in ("scan", "view", "light", "depth_median")}
                        for r in excl)
        print(f"[audit]   {ep.name}  {len(excl)} 个零散坏样本 (需在 dtu.py 里过滤)")


def quarantine(root: Path, rows: list[dict], undo: bool) -> None:
    qroot = root.parent / "prior_cache_quarantine"
    if undo:
        moved = list(qroot.rglob("prior_*.npz")) if qroot.exists() else []
        for f in moved:
            dst = root / f.relative_to(qroot)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
        print(f"[audit] 撤回 {len(moved)} 个文件到 {root}")
        return
    bad = [r for r in rows if r["flag_unscaled"] or r["flag_unreadable"]]
    for r in bad:
        src = root / r["path"]
        dst = qroot / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))
    print(f"[audit] 隔离 {len(bad)} 个文件 -> {qroot}  (--undo 可撤回)")
    print("[audit] 注意: 隔离后这些 (scan,view) 会让 dataloader 抛 FileNotFoundError,"
          " 必须配合 *_clean.txt 或先重建。")


def rebuild(scans: list[str], lists_dir: Path) -> None:
    import torch
    from base.config import build_mvs_config
    from data.dtu import DTUMVSDataset
    from models.pre_prior import build_prior_cache

    cfg = build_mvs_config()
    tmp = lists_dir / "_rebuild_tmp.txt"
    tmp.write_text("\n".join(scans) + "\n")
    print(f"[audit] 重建 {len(scans)} 个 scan: {scans}")
    ds = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root, listfile=str(tmp),
        nviews=cfg.train.num_views, mode="train",
        use_src_weights=cfg.cost_volume.use_src_weights,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = build_prior_cache(ds, device, overwrite=True, verbose=True)
    tmp.unlink(missing_ok=True)
    print(f"[audit] 重建完成 {n} 个 prior。请重跑 audit 确认 flag_unscaled 归零。")
    print("[audit] 若 scan85-92 重建后仍未标尺, 说明是 sfm.metric_scale_from_sparse")
    print("        的 <20 重叠点 fallback 又触发了 —— 那是代码问题, 不是缓存问题。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="log/prior_cache_manifest.csv")
    ap.add_argument("--clean-lists", action="store_true")
    ap.add_argument("--max-bad-frac", type=float, default=0.10,
                    help="坏文件比例超过它才剔除整个 scan (默认 0.10)")
    ap.add_argument("--quarantine", action="store_true")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--rebuild", default="", help="逗号分隔 scan 名, 或 'unscaled'/'oldgen'/'all-bad'")
    args = ap.parse_args()

    paths = ProjectPaths()
    root = Path(paths.prior_cache_path)
    lists_dir = Path(paths.project_path) / "lists/dtu"

    if args.undo:
        quarantine(root, [], undo=True)
        return

    rows = scan_cache(root)
    mp = Path(args.manifest)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with mp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[audit] manifest -> {mp}")

    unscaled_scans, oldgen_scans = summarize(rows, lists_dir)

    if args.clean_lists:
        write_clean_lists(lists_dir, rows, args.max_bad_frac)
    if args.quarantine:
        quarantine(root, rows, undo=False)
    if args.rebuild:
        sel = {"unscaled": sorted(unscaled_scans), "oldgen": sorted(oldgen_scans),
               "all-bad": sorted(unscaled_scans | oldgen_scans)}.get(
                   args.rebuild, [s for s in args.rebuild.split(",") if s])
        if sel:
            rebuild(sel, lists_dir)


if __name__ == "__main__":
    main()
