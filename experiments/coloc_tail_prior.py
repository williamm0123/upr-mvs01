"""共位测试: 模型的重误差像素上, prior 是不是也错的?

纯推理, 不训练。回答一个问题: 模型 |err|>20mm 的那批像素, prior 在那里的
误差分布是什么样。

  重合   (prior 在那里也是 100mm 量级) -> prior 提供不了独立信息
  不重合 (prior 在那里反而准)          -> prior 是独立信号, 架构没用上

顺带记录 stage1 的分支归因, 把"没用上"拆成两种可能:
  covered_by_local  真值落在 local(先验)分支的覆盖范围内吗
  winner_is_local   stage1 的 argmax 落在 local 分支上吗
两者结合区分「先验根本没覆盖」和「覆盖了但没选中」。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base.config import build_mvs_config, resolve_split
from data.dtu import DTUMVSDataset
from models.network import UprMVSNet
from train import _collate


def _pct(x: np.ndarray, q):
    return np.percentile(x, q) if x.size else np.full(len(q) if hasattr(q, "__len__") else 1, np.nan)


def _describe(name: str, e: np.ndarray, total: int) -> str:
    if e.size == 0:
        return f"  {name:<22s}  (空)"
    p = _pct(e, [50, 90, 95])
    return (f"  {name:<22s} n={e.size/1e6:6.2f}M ({100*e.size/total:5.2f}%)  "
            f"median {p[0]:7.2f}  mean {e.mean():8.2f}  p90 {p[1]:8.2f}  "
            f">20mm {100*(e>20).mean():5.1f}%  >50mm {100*(e>50).mean():5.1f}%")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="log/model/best.pth")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--listfile", default="", help="覆盖 split 的列表文件 (例: lists/dtu/val_clean.txt)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--spre", default="on", choices=["on", "off"])
    ap.add_argument("--legacy-arch", action="store_true",
                    help="按 2026-08-13 之前的架构建模型 (num_global/local=32/16, 无可见性头, "
                         "无分支门控) —— log/model/best.pth 是那个架构训的")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = 全部")
    ap.add_argument("--tail-mm", type=float, default=20.0)
    ap.add_argument("--out", default="experiments/out/coloc_tail_prior.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_mvs_config(profile=args.profile)
    cfg = replace(cfg, spre=replace(cfg.spre, enabled=(args.spre == "on")))
    # checkpoint 里有指纹就照它建模型 —— 忘记传 --legacy-arch 会用随机初始化的
    # VisibilityHead + 40/8 + 双模态做 strict=False 推理, 结果无意义。
    _peek = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    _fp = _peek.get("fingerprint")
    del _peek
    if _fp and not args.legacy_arch:
        print(f"[coloc] 按 checkpoint 指纹建模型: {_fp}")
        cfg = replace(cfg,
                      depth_range=replace(cfg.depth_range,
                                          num_global=_fp["num_global"], num_local=_fp["num_local"],
                                          gate_local_branch=_fp["gate_local_branch"],
                                          branch_prior=_fp["branch_prior"],
                                          dual_mode_stage2=_fp["dual_mode_stage2"]),
                      cost_volume=replace(cfg.cost_volume,
                                          visibility_weighting=_fp["visibility_weighting"]))
    elif not _fp and not args.legacy_arch:
        print("[coloc] WARNING: checkpoint 没有 fingerprint (2026-08-14 之前存的)。"
              "log/model/best.pth 是 32/16 架构 —— 请加 --legacy-arch, 否则结果无效。")
    if args.legacy_arch:
        cfg = replace(cfg,
                      depth_range=replace(cfg.depth_range, num_global=32, num_local=16,
                                          gate_local_branch=False, branch_prior=False,
                                          dual_mode_stage2=False),
                      cost_volume=replace(cfg.cost_volume, visibility_weighting=False))
        print("[coloc] legacy-arch: num_global/local=32/16, 门控/双模态/可见性头全关")

    if args.listfile:
        listfile, exclude_file = args.listfile, None
    else:
        _base = {"val": cfg.paths.val_list_file,
                 "train": cfg.paths.train_list_file,
                 "test": cfg.paths.test_list_file}[args.split]
        listfile, exclude_file = resolve_split(_base, args.split)

    ds = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=str(listfile),
        exclude_file=exclude_file,
        nviews=cfg.train.num_views,
        mode="val",                       # 确定性 center crop, 无 prior 腐蚀
        use_src_weights=cfg.cost_volume.use_src_weights,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=_collate, pin_memory=True, drop_last=False)
    print(f"[coloc] list={listfile} samples={len(ds)} views={cfg.train.num_views} "
          f"spre={cfg.spre.enabled} device={device}")

    model = UprMVSNet(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    _crit = [k for k in list(missing) + list(unexpected) if "vis_head" not in k]
    print(f"[coloc] ckpt={args.ckpt} step={ckpt.get('step')} best_metric={ckpt.get('best_metric')}")
    if missing or unexpected:
        print(f"[coloc]   WARNING missing={len(missing)} unexpected={len(unexpected)}"
              f" (非 vis_head 的: {len(_crit)})")
        for k in list(missing)[:5]:
            print("            missing:", k)
        for k in list(unexpected)[:5]:
            print("            unexpected:", k)
    model.eval()

    m_err_all, p_err_all = [], []
    # stage1 分支归因, 只在重误差像素上收集
    s1 = {"cover_local": [], "cover_global": [], "cover_axis": [], "win_local": [],
          "gt_in_local_span": [], "winner_err": [], "reg_err": [], "reg_shift": [],
          "win_cross_branch": []}

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(batch)

        pred = out["depth_full"].float()
        gt = batch["depth_gt"].float()
        prior = batch["depth_prior"].float()
        m = batch["mask"].bool() & (gt > 0) & torch.isfinite(prior) & (prior > 0)
        if "depth_values" in batch:
            dv = batch["depth_values"].float()
            m &= (gt >= dv.amin(dim=1).view(-1, 1, 1)) & (gt <= dv.amax(dim=1).view(-1, 1, 1))
        if not m.any():
            continue

        m_err = (pred - gt).abs()
        p_err = (prior - gt).abs()
        m_err_all.append(m_err[m].cpu().numpy())
        p_err_all.append(p_err[m].cpu().numpy())

        # ---- stage1 归因 (在 stage1 分辨率上, 只看模型重误差像素) ----
        st1 = out["stage1"]
        hyp = st1["depth_hypos"].float()               # [B,48,h,w]
        isl = st1["is_local"].float()                  # [B,48,h,w]
        mode_idx = st1["mode_idx"].long()              # [B,1,h,w] (见 network.py:267)
        h, w = hyp.shape[-2:]
        # 尾巴在 stage1 自己的分辨率上重新定义: 用 stage1 的预测而不是把全分辨率
        # 的掩码最近邻降采样 (1/8 降采样只抽 64 个像素里的 1 个, 不自洽)。
        gt_s = F.interpolate(gt.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)
        vm_s = F.interpolate(m.float().unsqueeze(1), size=(h, w), mode="nearest").squeeze(1) > 0.5
        d1_err = (st1["depth"].float() - gt_s).abs()
        m_s = vm_s & (gt_s > 0) & (d1_err > args.tail_mm)
        if m_s.any():
            d = (hyp - gt_s.unsqueeze(1)).abs()                       # [B,48,h,w]
            big = torch.full_like(d, 1e9)
            loc_err = torch.where(isl > 0.5, d, big).amin(dim=1)      # local 分支最优
            glb_err = torch.where(isl <= 0.5, d, big).amin(dim=1)     # global 分支最优
            axis_err = d.amin(dim=1)
            win_local = isl.gather(1, mode_idx).squeeze(1) > 0.5
            in_span = (gt_s >= st1["local_lo"].float()) & (gt_s <= st1["local_hi"].float())
            s1["cover_local"].append(loc_err[m_s].cpu().numpy())
            s1["cover_global"].append(glb_err[m_s].cpu().numpy())
            s1["cover_axis"].append(axis_err[m_s].cpu().numpy())
            s1["win_local"].append(win_local[m_s].cpu().numpy())
            s1["gt_in_local_span"].append(in_span[m_s].cpu().numpy())
            # "100% 选择失败" 里还混着两种情况: argmax 真选错, 和 argmax 对了
            # 但 mode_centered_regression 把它拉偏。分开它们决定该改聚合还是
            # 改回归。
            win_hyp = hyp.gather(1, mode_idx).squeeze(1)
            reg_d = st1["depth"].float()
            mw = int(getattr(model.range_cfg, "mode_window", 2))
            lo_i = (mode_idx - mw).clamp_min(0)
            hi_i = (mode_idx + mw).clamp_max(hyp.shape[1] - 1)
            cross = (isl.gather(1, lo_i) != isl.gather(1, hi_i)).squeeze(1)
            s1["winner_err"].append((win_hyp - gt_s).abs()[m_s].cpu().numpy())
            s1["reg_err"].append((reg_d - gt_s).abs()[m_s].cpu().numpy())
            s1["reg_shift"].append((reg_d - win_hyp).abs()[m_s].cpu().numpy())
            s1["win_cross_branch"].append(cross[m_s].cpu().numpy())

        if (bi + 1) % 20 == 0:
            print(f"[coloc]   {bi+1} batches")

    me = np.concatenate(m_err_all)
    pe = np.concatenate(p_err_all)
    n = me.size
    T = args.tail_mm

    print(f"\n=== 全体有效像素 n={n/1e6:.2f}M ===")
    print(_describe("model |err|", me, n))
    print(_describe("prior |err|", pe, n))

    print(f"\n=== 按 model |err| 分桶, 看该桶内 prior |err| ===")
    buckets = [("model err <2mm", me < 2),
               ("model err 2-8mm", (me >= 2) & (me < 8)),
               ("model err 8-20mm", (me >= 8) & (me < 20)),
               (f"model err >{T:.0f}mm  <<TAIL", me >= T),
               ("model err >50mm", me >= 50)]
    for name, sel in buckets:
        print(_describe(name, pe[sel], n))

    tail = me >= T
    print(f"\n=== 判决 (尾巴 = model |err| >= {T:.0f}mm, 占 {100*tail.mean():.2f}%) ===")
    if tail.any():
        pt = pe[tail]
        frac_prior_good = float((pt < 8).mean())
        print(f"  尾巴像素上 prior median = {np.median(pt):.2f} mm   mean = {pt.mean():.2f} mm")
        print(f"  尾巴像素上 prior <8mm 的比例 = {100*frac_prior_good:.2f}%   "
              f"(全体基线 = {100*float((pe<8).mean()):.2f}%)")
        lift = frac_prior_good / max(float((pe < 8).mean()), 1e-9)
        print(f"  提升比 (尾巴上 prior 好的比例 / 全体) = {lift:.2f}x")
        if frac_prior_good < 0.15:
            print("  >>> 重合: prior 在模型崩的地方也崩。DTU 线到此为止, 转 T&T。")
        elif lift > 1.0:
            print("  >>> 不重合: prior 在模型崩的地方反而更好。是独立信号, 架构没用上。")
        else:
            print("  >>> 部分重合: 看下面 stage1 归因决定是覆盖问题还是选择问题。")

    if s1["cover_local"]:
        cl = np.concatenate(s1["cover_local"]); cg = np.concatenate(s1["cover_global"])
        ca = np.concatenate(s1["cover_axis"]); wl = np.concatenate(s1["win_local"])
        sp = np.concatenate(s1["gt_in_local_span"])
        print(f"\n=== stage1 归因 (stage1 自身 |err|>{T:.0f}mm 的像素, n={cl.size}) ===")
        print(f"  真值落在 local 分支跨度内       : {100*sp.mean():5.2f}%")
        print(f"  stage1 winner 来自 local 分支   : {100*wl.mean():5.2f}%")
        print(f"  local  分支最优 hypo 误差 median: {np.median(cl):8.2f} mm")
        print(f"  global 分支最优 hypo 误差 median: {np.median(cg):8.2f} mm")
        print(f"  整条轴最优 hypo 误差     median : {np.median(ca):8.2f} mm")
        print(f"  轴上根本没有 <20mm 的候选       : {100*(ca>20).mean():5.2f}%  <- 覆盖失败")
        if s1["winner_err"]:
            we = np.concatenate(s1["winner_err"]); re_ = np.concatenate(s1["reg_err"])
            rs = np.concatenate(s1["reg_shift"]); xb = np.concatenate(s1["win_cross_branch"])
            print(f"\n  --- argmax 错 vs 回归拉偏 ---")
            print(f"  winner hypothesis 误差 median : {np.median(we):8.2f} mm")
            print(f"  回归后深度       误差 median : {np.median(re_):8.2f} mm")
            print(f"  |回归 - winner|       median : {np.median(rs):8.2f} mm")
            print(f"  winner 已 >20mm (argmax 选错) : {100*(we>20).mean():5.2f}%  <- 改聚合/匹配")
            print(f"  winner <20mm 但回归后 >20mm  : {100*((we<=20)&(re_>20)).mean():5.2f}%  <- 改回归")
            print(f"  回归窗口跨了 local/global 边界: {100*xb.mean():5.2f}%")
        covered = ca <= 20
        if covered.any():
            print(f"  覆盖了却没选中 (轴内有<20mm 但最终 err>{T:.0f}mm): {100*covered.mean():5.2f}% "
                  f"of tail  <- 选择失败")

    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_p, model_err=me.astype(np.float32), prior_err=pe.astype(np.float32),
                        **{f"s1_{k}": np.concatenate(v) for k, v in s1.items() if v})
    print(f"\n[coloc] 原始误差已存到 {out_p}")


if __name__ == "__main__":
    main()
