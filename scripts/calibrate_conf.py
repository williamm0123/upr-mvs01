#!/usr/bin/env python3
"""W3-C 的温度标定 —— 训练之后, 在**独立验证集**上跑一次, 零训练成本。

为什么不能和 BCE 一起训: 一起训的话温度会把训练集的过拟合程度一并吸收掉,
校准的意义正好没了。温度缩放的前提就是 "在一批模型没见过的数据上, 单参数地
把 logit 拉回正确的尺度"。

    python scripts/calibrate_conf.py --ckpt log/experiments/W3_vnext/model/best.pth --write

``--write`` 把拟合出来的 T 写回 checkpoint 的 ``conf_head.log_T`` buffer。写完
之后 ``test.py --conf-source learned`` 输出的就是标定过的概率, ``--photo-thresh
0.5`` 第一次真的表示 "五成把握"。

报四个数, 它们回答的是**两件不同的事**:
  * ECE / Brier  —— 置信度准不准 (校准)
  * AURC / R(0.6) —— 当过滤器好不好用 (排序)
两者可以一好一坏。AURC 好而 ECE 差 = 只需要重新校准, 不必重训。
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
from torch.utils.data import DataLoader

import test as testmod
from base.config import build_mvs_config
from models.fusion_conf import (
    brier_score, expected_calibration_error, fit_temperature,
    risk_at_coverage, risk_coverage,
)


@torch.no_grad()
def collect(model, ds, cfg, args, device, tau_mm: float, stride: int):
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=testmod._collate, pin_memory=True)
    zs, ys, es = [], [], []
    for i, batch in enumerate(loader):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type,
                            enabled=(cfg.train.amp and device.type == "cuda")):
            out = model(batch)
        fc = out.get("fusion_conf")
        if fc is None:
            raise SystemExit(
                "这个 checkpoint 没有融合置信度头 —— 训练时要加 --conf-head on。")
        gt = batch["depth_gt"].float()
        m = batch["mask"].bool() & (gt > 0)
        dv = batch["depth_values"].float()
        m &= (gt >= dv.amin(1).view(-1, 1, 1)) & (gt <= dv.amax(1).view(-1, 1, 1))
        if not m.any():
            continue
        err = (out["depth_full"].float() - gt).abs()[m]
        # 抽样是为了让整个验证集的 logit 能装进内存; 逐样本同一个 stride,
        # 不做任何按误差的筛选 —— 那会把标定推向乐观。
        zs.append(fc["logit"].float()[m].flatten()[::stride].cpu())
        es.append(err.flatten()[::stride].cpu())
        ys.append((err < tau_mm).float().flatten()[::stride].cpu())
        if (i + 1) % 50 == 0 or i + 1 == len(ds):
            print(f"[calib] {i + 1}/{len(ds)}", flush=True)
        if args.limit and i + 1 >= args.limit:
            break
    return torch.cat(zs), torch.cat(ys), torch.cat(es)


def _report(tag, prob, y, err):
    cov, rk, aurc = risk_coverage(prob, err)
    return {
        "tag": tag,
        "ece": expected_calibration_error(prob, y),
        "brier": brier_score(prob, y),
        "aurc_mm": aurc,
        "risk_at_0.6_mm": risk_at_coverage(prob, err, 0.6),
        "mean_conf": float(prob.mean()),
        "pos_frac": float(y.mean()),
        "keep_at_0.5": float((prob > 0.5).float().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--profile", default="umhpc")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=13,
                    help="逐样本的像素抽样步长, 只为了装得进内存")
    ap.add_argument("--tau-mm", type=float, default=2.0,
                    help="标签阈值, 必须与训练时的 conf_tau_mm 一致")
    ap.add_argument("--write", action="store_true",
                    help="把拟合出的 T 写回 checkpoint 的 conf_head.log_T")
    ap.add_argument("--out", default="experiments/out/calibrate_conf.json")
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
    model, ckpt_path = testmod.load_model(cfg, ns, device)
    model.eval()
    print(f"[calib] ckpt={ckpt_path}  样本数={len(ds)}  tau={a.tau_mm}mm")

    z, y, err = collect(model, ds, cfg, ns, device, a.tau_mm, a.stride)
    print(f"[calib] 收集到 {z.numel()} 个像素, 正例比例 {float(y.mean()):.4f}")

    T = fit_temperature(z, y)
    before = _report("before(T=1)", torch.sigmoid(z), y, err)
    after = _report(f"after(T={T:.4f})", torch.sigmoid(z / T), y, err)

    print(f"\n{'':<18}{'ECE':>9}{'Brier':>9}{'AURC(mm)':>11}{'R(0.6)mm':>11}"
          f"{'mean':>8}{'keep@.5':>9}")
    for r in (before, after):
        print(f"{r['tag']:<18}{r['ece']:>9.4f}{r['brier']:>9.4f}{r['aurc_mm']:>11.4f}"
              f"{r['risk_at_0.6_mm']:>11.4f}{r['mean_conf']:>8.3f}{r['keep_at_0.5']:>9.3f}")
    # 温度缩放是**单调**变换, 所以它按定义不改变排序 —— AURC / R(0.6) 两行必须
    # 完全相同。不同就说明哪里错了 (最常见的是把 T 也拿去训了)。
    if abs(before["aurc_mm"] - after["aurc_mm"]) > 1e-6:
        print("[calib] WARNING: 温度缩放是单调的, AURC 不该变 —— 检查实现")
    print(f"\n[calib] T = {T:.4f}  ({'过自信, 需要压平' if T > 1 else '欠自信, 需要锐化'})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"ckpt": str(ckpt_path), "T": T, "tau_mm": a.tau_mm,
         "n_pixels": int(z.numel()), "before": before, "after": after},
        indent=2), encoding="utf-8")
    print(f"[calib] 写入 {a.out}")

    if a.write:
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(ckpt_path, map_location="cpu")
        key = next((k for k in ck["model"] if k.endswith("conf_head.log_T")), None)
        if key is None:
            raise SystemExit("checkpoint 的 state_dict 里没有 conf_head.log_T")
        ck["model"][key] = torch.tensor(float(torch.log(torch.tensor(T))))
        torch.save(ck, ckpt_path)
        print(f"[calib] 已写回 {ckpt_path} 的 {key} (T={T:.4f})")
        print("[calib] 现在 test.py --conf-source learned 输出的是标定过的概率, "
              "--photo-thresh 0.5 才真的表示 '五成把握'。")
    else:
        print("[calib] 只报告不写入。加 --write 才会改 checkpoint。")


if __name__ == "__main__":
    main()
