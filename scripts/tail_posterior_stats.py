#!/usr/bin/env python3
"""W0-B: 尾部条件的 stage1 后验统计 —— **不训练**, 只读一个 checkpoint。

这是决定 W2 (双峰双分支) 做不做的唯一依据。现有的两个数都不够:

  * ``stage1/gt_rank_le2 = 0.77`` 是在**全部**有效像素上统计的
    (losses/composite.py 里的诊断), 不是在尾部像素上;
  * ``stage4/oor_recoverable = 0.876`` 只说 "GT 最近 bin 的概率不低于峰值的
    10%", 它可能是同一个宽峰的邻近 bin、平坦后验里的普通质量, 也可能是真正的
    第二表面 —— 这三种情况需要的对策完全不同。

本脚本在 A = {stage4 GT 出界} 与 B = {最终 |d-gt| > 8mm} 的并集上统计, 而且:

  * 峰的**分离性和邻域质量全部用物理逆深度**, 不用 bin 索引 ——
    stage1 是 global+local 混合轴, 相邻 local bin 可能只差零点几毫米;
  * 同时报 **oracle** 与 **deployable** 两套口径。oracle 用了 GT (离 GT 最近的
    分离峰), 只给收益上界; deployable 是 winner 之外质量最大的分离峰, 那才是
    部署时模型拿得到的。**判据只看 deployable。**

预注册判据 (动手前就写死, 别看完数再定):
    P(deployable 峰存在 且 M2 >= 0.10 且 GT 落在它邻域 | A∪B)
        >= 0.35  -> 做 W2
        <  0.20  -> 放弃, 中心偏移就够
        之间     -> 只留日志, 不扩模型

用法 (与 test.py 共用参数):
    python scripts/tail_posterior_stats.py --ckpt log/experiments/R32_f10_30k/model/best.pth
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

import test as testmod                      # 复用它的 dataset / model / collate
from base.config import build_mvs_config


def _peaks(prob: torch.Tensor) -> torch.Tensor:
    p_prev = torch.cat([prob[:, :1], prob[:, :-1]], dim=1)
    p_next = torch.cat([prob[:, 1:], prob[:, -1:]], dim=1)
    return (prob >= p_prev) & (prob >= p_next)


@torch.no_grad()
def collect(model, ds, cfg, args, device):
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=testmod._collate, pin_memory=True)
    acc = {k: 0.0 for k in (
        "n_tail", "n_valid", "dep_found", "dep_mass_ok", "dep_gt_in", "orc_found",
        "orc_gt_in", "dep_top2", "orc_top2", "win_cov", "rank1", "rank2", "rank3", "rank5",
        "rank1_all", "rank2_all", "rank3_all", "n_all")}
    gaps_dep, gaps_orc, masses = [], [], []

    for i, batch in enumerate(loader):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=(cfg.train.amp and device.type == "cuda")):
            out = model(batch)

        gt_full = batch["depth_gt"].float()
        m_full = batch["mask"].bool() & (gt_full > 0)
        dv = batch["depth_values"].float()
        m_full &= (gt_full >= dv.amin(1).view(-1, 1, 1)) & (gt_full <= dv.amax(1).view(-1, 1, 1))
        err_full = (out["depth_full"].float() - gt_full).abs()

        s1, s4 = out["stage1"], out["stage4"]
        hw = tuple(s1["depth_hypos"].shape[-2:])
        rs = lambda x: F.interpolate(x.unsqueeze(1).float(), size=hw, mode="nearest").squeeze(1)
        gt = rs(gt_full)
        valid = rs(m_full.float()).bool() & (gt > 0)
        err = rs(err_full)

        h4 = s4["depth_hypos"].float()
        oor4 = rs((gt_full < F.interpolate(h4[:, :1], size=gt_full.shape[-2:],
                                           mode="nearest").squeeze(1)).float()).bool() \
            | rs((gt_full > F.interpolate(h4[:, -1:], size=gt_full.shape[-2:],
                                          mode="nearest").squeeze(1)).float()).bool()
        tail = valid & (oor4 | (err > 8.0))

        prob = s1["prob"].float()
        hyp = s1["depth_hypos"].float()
        v = 1.0 / hyp.clamp_min(1e-6)
        v_gt = 1.0 / gt.clamp_min(1e-6)
        m_idx = prob.argmax(1, keepdim=True)
        v_m = v.gather(1, m_idx)

        # rank of the GT-nearest bin
        g_idx = (v - v_gt.unsqueeze(1)).abs().argmin(1, keepdim=True)
        p_at = prob.gather(1, g_idx)
        rank = (prob > p_at).sum(1) + 1

        # 物理分离尺度: stage2 的逆深度半宽
        h2 = out["stage2"]["depth_hypos"].float()
        v2ax = 1.0 / h2.clamp_min(1e-6)
        h_sep = ((v2ax.amax(1) - v2ax.amin(1)) * 0.5).clamp_min(1e-12)
        h_sep = F.interpolate(h_sep.unsqueeze(1), size=hw, mode="nearest").squeeze(1)

        pk = _peaks(prob)
        sep = (v - v_m).abs() > 2.0 * h_sep.unsqueeze(1)
        cand = pk & sep

        # deployable: 分离峰里概率最高的
        sc = torch.where(cand, prob, torch.full_like(prob, -1.0))
        j_d = sc.argmax(1, keepdim=True)
        found_d = (sc.gather(1, j_d) >= 0).squeeze(1)
        # oracle: 离 GT 最近的分离峰 (用了 GT, 只给上界)
        dist = torch.where(cand, (v - v_gt.unsqueeze(1)).abs(),
                           torch.full_like(prob, float("inf")))
        j_o = dist.argmin(1, keepdim=True)
        found_o = torch.isfinite(dist.gather(1, j_o)).squeeze(1)

        win_cov = (v_gt - v_m.squeeze(1)).abs() <= h_sep
        acc["win_cov"] += float((tail & win_cov).sum())
        for nm, j, found in (("dep", j_d, found_d), ("orc", j_o, found_o)):
            v_j = v.gather(1, j)
            mass = torch.where((v - v_j).abs() <= h_sep.unsqueeze(1),
                               prob, torch.zeros_like(prob)).sum(1)
            gt_in = ((v_gt - v_j.squeeze(1)).abs() <= h_sep) & found
            sel = tail & found
            acc[f"{nm}_found"] += float(sel.sum())
            acc[f"{nm}_gt_in"] += float((tail & gt_in).sum())
            # top-2 覆盖: 以 v_m 与 v_j 为中心各开一个当前半宽, 合起来盖住多少。
            # 这是"双模态相对单峰能多拿到什么"的直接读数 —— 单看 gt_in 会低估,
            # 因为 winner 窗口本来就盖住了一部分尾部像素。
            acc[f"{nm}_top2"] += float((tail & (gt_in | win_cov)).sum())
            if nm == "dep":
                acc["dep_mass_ok"] += float((tail & found & (mass >= 0.10) & gt_in).sum())
                if sel.any():
                    masses.append(mass[sel].flatten().cpu())
            g = ((v_j.squeeze(1) - v_m.squeeze(1)).abs() / h_sep.clamp_min(1e-12))
            (gaps_dep if nm == "dep" else gaps_orc).append(g[sel].flatten().cpu())

        acc["n_tail"] += float(tail.sum())
        acc["n_valid"] += float(valid.sum())
        acc["n_all"] += float(valid.sum())
        for r in (1, 2, 3, 5):
            acc[f"rank{r}"] += float((tail & (rank <= r)).sum())
        for r in (1, 2, 3):
            acc[f"rank{r}_all"] += float((valid & (rank <= r)).sum())
        if (i + 1) % 50 == 0 or i + 1 == len(ds):
            print(f"[tail] {i + 1}/{len(ds)}  tail_frac="
                  f"{acc['n_tail'] / max(acc['n_valid'], 1):.4f}", flush=True)
        if args.limit and i + 1 >= args.limit:
            break

    def q(lst, p):
        if not lst:
            return float("nan")
        t = torch.cat(lst)
        return float(t.quantile(p)) if t.numel() else float("nan")

    n = max(acc["n_tail"], 1.0)
    res = {
        "tail_frac": acc["n_tail"] / max(acc["n_valid"], 1.0),
        "P_dep_found": acc["dep_found"] / n,
        "P_dep_mass_ok_and_gt_in": acc["dep_mass_ok"] / n,       # <-- 判据用这个
        "P_dep_gt_in": acc["dep_gt_in"] / n,
        "P_orc_found": acc["orc_found"] / n,
        "P_orc_gt_in": acc["orc_gt_in"] / n,                     # oracle 上界
        # 单峰窗口本身在尾部盖住多少 —— top2 要跟它比, 不是跟 0 比
        "P_winner_only_cov": acc["win_cov"] / n,
        "P_top2_cov_dep": acc["dep_top2"] / n,
        "P_top2_cov_orc": acc["orc_top2"] / n,
        "gap_dep_p50_in_s2_half": q(gaps_dep, 0.5),
        "gap_orc_p50_in_s2_half": q(gaps_orc, 0.5),
        "mass_dep_p50": q(masses, 0.5),
        "rank_le1_tail": acc["rank1"] / n, "rank_le2_tail": acc["rank2"] / n,
        "rank_le3_tail": acc["rank3"] / n, "rank_le5_tail": acc["rank5"] / n,
        "rank_le1_all": acc["rank1_all"] / max(acc["n_all"], 1.0),
        "rank_le2_all": acc["rank2_all"] / max(acc["n_all"], 1.0),
        "rank_le3_all": acc["rank3_all"] / max(acc["n_all"], 1.0),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(parents=[], add_help=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--profile", default="local")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个样本 (调试用)")
    ap.add_argument("--out", default="experiments/out/tail_posterior_stats.json")
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
    args_ns = testmod.eval_namespace(ckpt=a.ckpt, num_workers=a.num_workers,
                                     resize_scale=a.resize_scale,
                                     num_views=a.num_views,
                                     full_image=(a.full_image == "on"),
                                     max_refs=a.max_refs,
                                     limit=a.limit, cfg_override=a.cfg_override)
    ds = testmod.build_dataset(cfg, args_ns)
    model, ckpt = testmod.load_model(cfg, args_ns, device)
    model.eval()
    print(f"[tail] ckpt={ckpt}  样本数={len(ds)}  device={device}")

    res = collect(model, ds, cfg, args_ns, device)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ 尾部条件后验统计 (A∪B) ================")
    for k, v in res.items():
        print(f"  {k:32s} {v:.4f}")
    p = res["P_dep_mass_ok_and_gt_in"]
    print("\n---- 预注册判据 (只看 deployable) ----")
    print(f"  P(dep 峰存在 ∧ M2>=0.10 ∧ GT 在其邻域 | A∪B) = {p:.4f}")
    gap = res["gap_dep_p50_in_s2_half"]
    gain_pp = 100.0 * (res["P_top2_cov_dep"] - res["P_winner_only_cov"])
    if p >= 0.35:
        verdict = "做 W2 —— 尾部确实主要是可分离、可检出的第二表面"
    elif p < 0.20:
        # 放弃的**理由**要由 gap 决定, 不能硬编码。gap<1 才是"中心偏移就够";
        # gap>2 是"只有模态切换够得着", 那时放弃的理由是没质量/不值得。
        if gap < 1.0:
            why = (f"第二峰离 winner 只有 {gap:.2f}x stage2 半宽 —— "
                   f"范围控制器的中心偏移就够得着, 不需要模态切换")
        elif gap > 2.0:
            why = (f"第二峰远在 {gap:.2f}x stage2 半宽处 (只有模态切换够得着), "
                   f"但它没有质量 (M2 中位数 {res['mass_dep_p50']:.4f}) 且只多盖 "
                   f"{gain_pp:+.1f}pp —— 够得着也不值得")
        else:
            why = f"第二峰在 {gap:.2f}x stage2 半宽处, 质量与收益都不足"
        verdict = f"放弃 W2 —— {why}"
    else:
        verdict = "灰区: 只保留日志与诊断, 不扩模型"
    print(f"  -> {verdict}")
    print(f"  top-2 覆盖收益: winner-only {res['P_winner_only_cov']:.4f} -> "
          f"top2 {res['P_top2_cov_dep']:.4f} ({gain_pp:+.1f}pp); "
          f"oracle 上界 {res['P_top2_cov_orc']:.4f}")
    print(f"  oracle 上界 P_orc_gt_in = {res['P_orc_gt_in']:.4f}; 与 deployable 的差距"
          f" = {res['P_orc_gt_in'] - res['P_dep_gt_in']:+.4f}"
          f"  (差距大 = 峰存在但选不出来, 那是 ModeHead 的难度, 不是收益)")
    if res["P_dep_found"] > 0.99:
        print("  注: P_*_found ~ 1.0 不代表'总有第二个表面' —— 48 bin 的轴上, "
              "任何离 winner 超过 2x h_sep 的局部极大都算'找到'。有没有东西看 M2, "
              "不看 found。")
    print(f"  rank(g): 尾部 top1 {res['rank_le1_tail']:.3f} / top3 {res['rank_le3_tail']:.3f}"
          f"  vs 全体 top1 {res['rank_le1_all']:.3f} / top3 {res['rank_le3_all']:.3f}")
    if res["rank_le3_tail"] > 0.6 and res["rank_le1_tail"] < 0.5:
        print("  ^ 读法: 尾部里 GT bin 多半仍在 stage1 的 top-3, 只是不在 top-1。"
              "信息在 stage1 就有, 丢在'只把 argmax 传给下一级'这个交接上 —— "
              "这是范围中心/交接的问题, 不是再加一个模态能解决的。")
    print(f"\n  写入 {a.out}")


if __name__ == "__main__":
    main()
