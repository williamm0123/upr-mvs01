#!/usr/bin/env python3
"""W3-C 的温度标定 —— 训练之后, 在**独立验证集**上跑一次, 零训练成本。

为什么不能和 BCE 一起训: 一起训的话温度会把训练集的过拟合程度一并吸收掉,
校准的意义正好没了。温度缩放的前提就是 "在一批模型没见过的数据上, 单参数地
把 logit 拉回正确的尺度"。

    python scripts/calibrate_conf.py \
        --ckpt log/experiments/UPRMVS_vNext/model/latest.pth \
        --require-step 30000 \
        --write-out log/experiments/UPRMVS_vNext/model/latest_calibrated.pth

拟合的是**二参数** Platt ``p = sigma(z/T + b)``, 不是单参数温度。原因: 置信度
损失用的是类别平衡 BCE (losses/composite.py), 加权会平移最优 logit 的截距,
只拟合 T 时 ``prob=0.5`` 并不表示 "五成把握"。

``--write-out`` 产出**一份新的** checkpoint, 不覆盖原文件 —— 标定是可以推倒
重来的后处理, 覆盖之后就再也拿不到未标定的 logit。

报四个数, 它们回答的是**两件不同的事**:
  * ECE / Brier  —— 置信度准不准 (校准)
  * AURC / R(0.6) —— 当过滤器好不好用 (排序)
两者可以一好一坏。AURC 好而 ECE 差 = 只需要重新校准, 不必重训。
"""
from __future__ import annotations

import argparse
import json
import math
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
    brier_score, expected_calibration_error, fit_platt,
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
        # dtype 必须跟 checkpoint 走 (cfg 是 load_model 对齐过的那份, 见 main)。
        with torch.autocast(device_type=device.type, dtype=testmod.amp_dtype_of(cfg),
                            enabled=(cfg.train.amp and device.type == "cuda")):
            out = model(batch)
        fc = out.get("fusion_conf")
        if fc is None:
            raise SystemExit(
                "这个 checkpoint 没有融合置信度头 —— 训练时要加 --conf-head on。")
        if i == 0:
            # **第一个样本就查**, 不要跑完 833 个再报一屏 nan (job 415228 就是
            # 跑满 40 分钟才在最后一步失败)。顺带把链上第一处非有限的位置指出来。
            bad = [k for k, t in (("depth_full", out["depth_full"]),
                                  ("fusion_conf.logit", fc["logit"]))
                   if not torch.isfinite(t).all()]
            if bad:
                dt = testmod.amp_dtype_of(cfg)
                raise SystemExit(
                    f"第一个样本的 {', '.join(bad)} 含非有限值 (autocast dtype={dt})。\n"
                    f"不要继续标定 —— 先定位:\n"
                    f"  python scripts/probe_nonfinite.py --ckpt {args.ckpt} --compare\n"
                    f"它会打印链上第一处非有限的张量, 并对同一批依次跑 fp16/bf16/fp32。")
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
    ap.add_argument("--write-out", default=None,
                    help="把拟合出的 (T, bias) 写进**一份新的** checkpoint。"
                         "不允许覆盖原 ckpt —— 标定是可以推倒重来的后处理, "
                         "覆盖掉之后就再也拿不到未标定的 logit 了。")
    ap.add_argument("--require-step", type=int, default=None,
                    help="断言 checkpoint 的 step 等于该值 (正式协议用 30000)。"
                         "防止拿一个中途的 ckpt 去标定然后当终审模型。")
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
    # **必须换成对齐后的 cfg**: load_model 内部按 fingerprint 重建了 cfg (amp_dtype
    # 等), 但它只改了自己的局部名字。不换的话下面 collect() 里的 autocast dtype
    # 用的还是 profile 默认的 fp16 —— job 415228 的直接死因。
    cfg = getattr(testmod.load_model, "last_cfg", cfg)
    model.eval()
    meta = getattr(testmod.load_model, "last_meta", None) or {}
    if a.require_step is not None and int(meta.get("step", -1)) != int(a.require_step):
        raise SystemExit(
            f"checkpoint 的 step={meta.get('step')}, 但 --require-step {a.require_step}。"
            f"正式协议只标定跑满的最终模型 —— 拿中途的 ckpt 标定再当终审, 等于用"
            f"另一个模型的置信度去解释这个模型。")
    print(f"[calib] ckpt={ckpt_path} (step {meta.get('step')})  样本数={len(ds)}  tau={a.tau_mm}mm")

    z, y, err = collect(model, ds, cfg, ns, device, a.tau_mm, a.stride)
    print(f"[calib] 收集到 {z.numel()} 个像素, 正例比例 {float(y.mean()):.4f}")

    log_T, bias = fit_platt(z, y)
    T = float(torch.tensor(log_T).exp())
    before = _report("before(T=1,b=0)", torch.sigmoid(z), y, err)
    after = _report(f"after(T={T:.4f},b={bias:+.4f})", torch.sigmoid(z / T + bias), y, err)

    print(f"\n{'':<18}{'ECE':>9}{'Brier':>9}{'AURC(mm)':>11}{'R(0.6)mm':>11}"
          f"{'mean':>8}{'keep@.5':>9}")
    for r in (before, after):
        print(f"{r['tag']:<18}{r['ece']:>9.4f}{r['brier']:>9.4f}{r['aurc_mm']:>11.4f}"
              f"{r['risk_at_0.6_mm']:>11.4f}{r['mean_conf']:>8.3f}{r['keep_at_0.5']:>9.3f}")
    # z -> z/T + b 在 T > 0 时是**严格单调**的仿射变换, 所以它按定义不改变排序
    # —— AURC / R(0.6) 必须逐位相同。不同就说明实现错了 (最常见的是把 T 也拿去
    # 训了, 或者拟合时 T 跑成了负数)。这里 raise 而不是 warn: 标定完之后紧跟着
    # 就是终审点云, 一个悄悄改了排序的 "标定" 会直接污染最终结论。
    # 容差不能取 0 或 1e-6: sigmoid 在 fp32 下会把接近的 logit 映射成**完全相等**
    # 的概率 (|z| 大的时候尤其多), 而排序对并列不作保证, 于是并列元素的先后可能
    # 变, AURC 随之抖动一点点。这不是错误。真正的实现错误 (T 训进了模型、T 拟合
    # 成负数、误用了另一套 logit) 会让这两个数差好几个百分点, 相对 1e-3 足够抓。
    for k in ("aurc_mm", "risk_at_0.6_mm", "ece", "brier"):
        # nan 的比较恒为 False, 所以下面那条相对误差判据对全 nan 是**空的** ——
        # 必须先单独查有限性, 否则一份全 nan 的报告会一路通过。
        for tag, r in (("before", before), ("after", after)):
            if not math.isfinite(r[k]):
                raise SystemExit(f"[calib] {tag}[{k}] = {r[k]} —— 模型输出非有限, 先修那个。")
    for k in ("aurc_mm", "risk_at_0.6_mm"):
        ref = max(abs(before[k]), 1e-9)
        if abs(before[k] - after[k]) / ref > 1e-3:
            raise SystemExit(
                f"[calib] {k} 在标定前后不一致 ({before[k]:.6f} -> {after[k]:.6f})。"
                f"正斜率仿射不可能改变排序 —— 实现有问题, 不要继续。")
    print(f"\n[calib] T = {T:.4f}  bias = {bias:+.4f}  "
          f"({'过自信, 需要压平' if T > 1 else '欠自信, 需要锐化'})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"ckpt": str(ckpt_path), "T": T, "log_T": log_T, "bias": bias, "tau_mm": a.tau_mm,
         "n_pixels": int(z.numel()), "before": before, "after": after},
        indent=2), encoding="utf-8")
    print(f"[calib] 写入 {a.out}")

    if a.write_out:
        dst = Path(a.write_out)
        if dst.resolve() == Path(ckpt_path).resolve():
            raise SystemExit("--write-out 不能等于 --ckpt: 覆盖原 checkpoint 之后就"
                             "再也拿不到未标定的 logit, 标定也就无法重做。")
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(ckpt_path, map_location="cpu")
        kt = next((k for k in ck["model"] if k.endswith("conf_head.log_T")), None)
        if kt is None:
            raise SystemExit("checkpoint 的 state_dict 里没有 conf_head.log_T")
        ck["model"][kt] = torch.tensor(float(log_T))
        kb = kt[: -len("log_T")] + "calib_bias"
        ck["model"][kb] = torch.tensor(float(bias))
        ck["calibration"] = {"log_T": log_T, "bias": bias, "tau_mm": a.tau_mm,
                             "n_pixels": int(z.numel()), "source_ckpt": str(ckpt_path)}
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ck, dst)
        print(f"[calib] 写入新 checkpoint {dst} ({kt}={log_T:.4f}, {kb}={bias:+.4f})")
        print("[calib] 现在 test.py --conf-source learned 输出的是标定过的概率, "
              "--photo-thresh 0.5 才真的表示 '五成把握'。")
    else:
        print("[calib] 只报告不写入。加 --write-out <路径> 才会产出标定过的 checkpoint。")


if __name__ == "__main__":
    main()
