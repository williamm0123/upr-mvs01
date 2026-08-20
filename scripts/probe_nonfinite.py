#!/usr/bin/env python
"""定位 stage4 前向非有限值的第一处源头, 并对比 fp16 / bf16 / fp32。

背景: 2026-08-20 那一轮八个 arm 里有五个死于同一签名 —— 只有 stage4 的 loss
分量非有限、输入全部有限、``depth_full`` 的 finite_frac 恰好 0.5。梯度范数、
AMP scale、nonfinite_frac 都没有分离度 (跑满的 D0 grad_norm 11.96 反而高于
崩掉的 R 的 6.44), 所以只能靠内部激活探针。

用法::

    # 扫验证集, 找出第一个能复现非有限值的样本, 打印整条链
    python scripts/probe_nonfinite.py --ckpt log/experiments/R/model/latest.pth

    # 拿到坏样本后, 同一批数据依次用三种精度跑, 比较
    python scripts/probe_nonfinite.py --ckpt ... --compare --sample-index 40213

结论怎么读: ``first_bad`` 给出的就是链上第一个出现非有限元素的张量。若它是
``logits_raw``, 源头在 stage4 的 3D UNet (相关体已经是 fp32, 但 UNet 在
autocast 下跑 fp16); 若是 ``warped`` 或 ``cv_s``, 源头在投影/采样那一段。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "models", _ROOT / "models" / "Depth-Anything-3" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

from base.config import build_mvs_config, resolve_split
from data.dtu import DTUMVSDataset
from models.network import UprMVSNet
from models.probe import Probe
import train as T

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}


def build(args):
    cfg = build_mvs_config(profile=args.profile)
    ck = torch.load(args.ckpt, map_location="cpu") if args.ckpt else None
    if ck is not None and ck.get("config"):
        fp = ck.get("fingerprint") or {}
        print(f"[ckpt] step={ck.get('step','?')} fingerprint 摘要: "
              f"{fp.get('num_global')}/{fp.get('num_local')} "
              f"bp={fp.get('branch_prior')} vis={fp.get('visibility_weighting')} "
              f"lr={fp.get('lr')} seed={fp.get('seed')} amp={fp.get('amp_dtype','fp16')}")
        # 用 checkpoint 里记录的开关重建配置, 否则拿当前默认值加载会静默改语义
        from dataclasses import replace
        c = ck["config"]
        cfg = replace(
            cfg,
            depth_range=replace(cfg.depth_range, **{k: c["depth_range"][k] for k in
                                                    ("num_global", "num_local", "gate_local_branch",
                                                     "branch_prior") if k in c.get("depth_range", {})}),
            cost_volume=replace(cfg.cost_volume, **{k: c["cost_volume"][k] for k in
                                                    ("num_depths_stage1", "visibility_weighting")
                                                    if k in c.get("cost_volume", {})}),
        )
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UprMVSNet(cfg).to(dev).eval()
    if ck is not None:
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"[ckpt] 非严格加载: missing={len(missing)} unexpected={len(unexpected)}")
    lf, ef = resolve_split(cfg.paths.val_list_file, "val", not args.no_clean_lists)
    ds = DTUMVSDataset(datapath=cfg.paths.dtu_train_root, listfile=lf, exclude_file=ef,
                       nviews=cfg.train.num_views, mode="val",
                       use_src_weights=cfg.cost_volume.use_src_weights,
                       seed=cfg.train.seed)
    return cfg, model, ds, dev


def run_once(model, batch, dev, dtype):
    """在指定精度下跑一次前向, 返回 (rows, first_bad)。fp32 = 关掉 autocast。"""
    with Probe.session():
        with torch.no_grad(), torch.autocast(device_type=dev.type,
                                             dtype=dtype or torch.float16,
                                             enabled=(dtype is not None and dev.type == "cuda")):
            model(batch)
        return Probe.rows(), Probe.first_bad(), Probe.format(only_bad=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="要复现的 checkpoint (留空 = 随机初始化)")
    ap.add_argument("--profile", default="umhpc")
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    ap.add_argument("--max-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--sample-index", type=int, default=-1, help="只跑这一个样本")
    ap.add_argument("--compare", action="store_true", help="对同一批依次跑 fp16/bf16/fp32")
    ap.add_argument("--no-clean-lists", action="store_true", default=True)
    args = ap.parse_args()

    cfg, model, ds, dev = build(args)
    if args.sample_index >= 0:
        ds = torch.utils.data.Subset(ds, [args.sample_index])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         collate_fn=T._collate, num_workers=2)
    print(f"[probe] 样本 {len(ds)} 个, batch={args.batch_size}, dtype={args.dtype}, device={dev}")

    hits = 0
    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        rows, bad, text = run_once(model, batch, dev, DTYPES[args.dtype])
        if bad is None:
            continue
        hits += 1
        print("\n" + "=" * 78)
        print(f"batch {bi} 复现非有限值 —— 第一处: {bad['stage']}.{bad['name']}"
              + (f"[s{bad['src']}]" if bad["src"] is not None else ""))
        print("=" * 78)
        print("  样本身份:")
        print("    " + T._sample_ids(batch).replace("\n                ", "\n    "))
        print(f"\n  完整链 ({args.dtype}):")
        print(text)

        if args.compare:
            for name, dt in DTYPES.items():
                if name == args.dtype:
                    continue
                _, b2, t2 = run_once(model, batch, dev, dt)
                verdict = "仍然非有限" if b2 is not None else "全部有限"
                print(f"\n  —— 同一批改用 {name}: {verdict} ——")
                if b2 is not None:
                    print(f"     第一处: {b2['stage']}.{b2['name']}")
                else:
                    tail = [l for l in t2.split("\n") if "stage4." in l or "out." in l]
                    print("\n".join(tail))
        if hits >= 3:
            break

    if not hits:
        print(f"\n[probe] 扫了 {min(bi + 1, args.max_batches)} 个 batch, {args.dtype} 下没有复现非有限值。")
        print("        换 --ckpt (崩溃那一步的 latest.pth) 或放宽 --max-batches。")


if __name__ == "__main__":
    main()
