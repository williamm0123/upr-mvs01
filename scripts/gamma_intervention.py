#!/usr/bin/env python3
"""SPRE 门的推理干预 —— gamma_k <- 0, 零训练成本。

为什么需要这个: ``gamma_k`` 的**数值本身不是消融读数**。门和下游卷积之间存在
尺度不可辨识性

    gamma * r * W  ==  (c * gamma) * r * (W / c)

所以 gamma 很小不一定代表 SPRE 没信息, 不为零也不代表模型真的依赖它 —— 下游
卷积完全可以补偿门的大小。要判断依赖度, 只能在**同一个 checkpoint** 上把某一级
的 gamma 强制置 0, 重跑一遍验证集, 看指标怎么变。

    python scripts/gamma_intervention.py --ckpt log/experiments/W1_vnext/model/best.pth

不需要重新训练, 也不是新的小实验 —— 它是读 gate 数值的正确替代品。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import test as testmod
from base.config import build_mvs_config


@torch.no_grad()
def evaluate(model, ds, cfg, args, device, kill: int | None):
    """kill=None 表示不干预; kill=k 表示把 gamma_k (1-based) 之后置 0。"""
    orig = model.spre_gates.forward

    if kill is not None:
        def patched():
            g = orig().clone()
            g[kill - 1] = 0.0
            return g
        model.spre_gates.forward = patched

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=testmod._collate, pin_memory=True)
    err_sum = n = acc2 = 0.0
    cov4 = cov4_n = 0.0
    for i, batch in enumerate(loader):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type,
                            enabled=(cfg.train.amp and device.type == "cuda")):
            out = model(batch)
        gt = batch["depth_gt"].float()
        m = batch["mask"].bool() & (gt > 0)
        dv = batch["depth_values"].float()
        m &= (gt >= dv.amin(1).view(-1, 1, 1)) & (gt <= dv.amax(1).view(-1, 1, 1))
        if m.any():
            e = (out["depth_full"].float() - gt).abs()[m]
            err_sum += float(e.sum()); n += float(e.numel())
            acc2 += float((e < 2.0).sum())
        h4 = out["stage4"]["depth_hypos"].float()
        hw = h4.shape[-2:]
        g4 = F.interpolate(gt.unsqueeze(1), size=hw, mode="nearest").squeeze(1)
        v4 = F.interpolate(m.float().unsqueeze(1), size=hw, mode="nearest").squeeze(1).bool()
        if v4.any():
            inr = (g4 >= h4[:, 0]) & (g4 <= h4[:, -1])
            cov4 += float(inr[v4].sum()); cov4_n += float(v4.sum())
        if args.limit and i + 1 >= args.limit:
            break

    model.spre_gates.forward = orig
    return {"abs_err": err_sum / max(n, 1), "acc_2mm": acc2 / max(n, 1),
            "s4_in_range": cov4 / max(cov4_n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--profile", default="umhpc")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="experiments/out/gamma_intervention.json")
    ap.add_argument("--cfg-override", default="auto")
    ap.add_argument("--resize-scale", type=float, default=0.8,
                    help="相对 DTU 原始 1200x1600 的缩放。0.8 整幅是 A100 的部署口径; "
                         "16GB 卡放不下, 用 0.5 (600x800) 或 0.6 (720x960)。"
                         "只用能被 8 整除的值 —— FPN 要下采样三次")
    ap.add_argument("--num-views", type=int, default=None,
                    help="视角数, 默认跟 cfg (5)。显存不够时**先降分辨率再降视角** —— "
                         "视角数对匹配质量的影响比分辨率大")
    ap.add_argument("--full-image", choices=["on", "off"], default="on",
                    help="on=整幅 (只缩放不裁剪); off=按 cfg 的裁剪窗口")
    ap.add_argument("--max-refs", type=int, default=0,
                    help="每个 scan 最多取几个参考视图, 0=全部 (子采样用)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_mvs_config(profile=a.profile)
    ns = testmod.eval_namespace(ckpt=a.ckpt, num_workers=a.num_workers, limit=a.limit,
                                     resize_scale=a.resize_scale,
                                     num_views=a.num_views,
                                     full_image=(a.full_image == "on"),
                                     max_refs=a.max_refs,
                                cfg_override=a.cfg_override)
    ds = testmod.build_dataset(cfg, ns)
    model, ckpt = testmod.load_model(cfg, ns, device)
    model.eval()
    g = model.spre_gates().detach()
    print(f"[gamma] ckpt={ckpt}")
    print(f"[gamma] 学到的门: {[round(float(x), 4) for x in g]}")
    print("[gamma] 提醒: 上面这几个数**不能**当消融读数 —— 下游卷积可以补偿门的大小。\n")

    rows = {"baseline": evaluate(model, ds, cfg, ns, device, None)}
    for k in (2, 3, 4):
        rows[f"gamma{k}=0"] = evaluate(model, ds, cfg, ns, device, k)

    b = rows["baseline"]
    print(f"{'干预':<14}{'abs_err':>10}{'Δ':>9}{'acc@2mm':>10}{'Δ':>9}{'s4_cov':>9}{'Δ':>9}")
    for name, r in rows.items():
        print(f"{name:<14}{r['abs_err']:>10.4f}{r['abs_err'] - b['abs_err']:>+9.4f}"
              f"{r['acc_2mm']:>10.4f}{r['acc_2mm'] - b['acc_2mm']:>+9.4f}"
              f"{r['s4_in_range']:>9.4f}{r['s4_in_range'] - b['s4_in_range']:>+9.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"gamma": [float(x) for x in g], "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n写入 {a.out}")
    print("读法: 把某一级的门置 0 后 abs_err 明显变差 = 那一级真的在用 SPRE;")
    print("      几乎不动 = 那一级的 SPRE 输入是冗余的, 无论门的数值是多少。")


if __name__ == "__main__":
    main()
