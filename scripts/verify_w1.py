#!/usr/bin/env python3
"""W1 的实现校验 —— 不是性能实验。

三个子命令, 都不碰数据集:

  --units        新模块的单元检查 (逆 softplus 往返、SPRE 门单调、p-范数不下溢、
                 rho 缩放后轴仍升序、MAP 头恒落在候选上、残差头零初始化 ...)
  --equivalence  **最重要的一条**: 用同一个 seed 分别构造 legacy 与 inverse 两个
                 网络, 确认 (a) 共享参数逐元素相同 (RNG 流没被打乱),
                 (b) step 0 (blend=0) 的三级候选轴与 depth_full 逐元素一致。
                 不要求逐比特 —— GPU 的卷积/插值/归约未必逐比特确定, 用容差。
  --mem          合成全分辨率数据的前向 + 反向, 报 torch.cuda.max_memory_allocated。
                 nvidia-smi 的采样在短跑里经常采不到峰值, 这个才是确切值。

    python scripts/verify_w1.py --units
    python scripts/verify_w1.py --equivalence --tol 1e-5
    python scripts/verify_w1.py --mem --batch 1 --views 5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from base.config import MVSConfig
from models.depth_range import (
    blend_axes, build_axis_inverse, local_intervals,
    robust_local_scale, second_mode_physical,
)
from models.decoder import DepthDecoder
from models.range_controller import (
    RangeController, SpreGates, Stage4ResidualHead, inverse_softplus,
)

D_MIN, D_MAX = 425.0, 935.0


def _cfg(axis="legacy_depth", stage4="expect", spre_cascade=False,
         geo_valid=False, vis_mode="softmax", conf_head=False,
         vis_supervise=False, visibility=None, weights=None,
         h=None, w=None, views=None):
    c = MVSConfig()
    c = replace(c, dino=replace(c.dino, mode="off", feed_fpn=False))
    c = replace(c, spre=replace(c.spre, enabled=False, reliability_source="cached"))
    c = replace(c, depth_range=replace(
        c.depth_range, num_global=32, num_local=16, range_min_gi=(0.66, 0.20, 0.10),
        axis_space=axis, stage4_head=stage4, spre_cascade=spre_cascade))
    c = replace(c, cost_volume=replace(
        c.cost_volume, num_depths_stage1=48,
        geo_valid_aggregation=geo_valid, vis_mode=vis_mode,
        vis_supervise=vis_supervise,
        visibility_weighting=(visibility if visibility is not None
                              else vis_mode == "sigmoid")))
    c = replace(c, decoder=replace(c.decoder, fusion_conf=conf_head))
    lw = dict(w_conf=1.0 if conf_head else 0.0,
              w_vis=1.0 if vis_supervise else 0.0)
    lw.update(weights or {})
    c = replace(c, loss=replace(c.loss, **lw))
    if h is not None:
        c = replace(c, data=replace(c.data, target_h=h, target_w=w, nviews=views))
    return c


def _batch(B, V, H, W, device="cpu"):
    K = torch.eye(3).view(1, 1, 3, 3).repeat(B, V, 1, 1)
    K[:, :, 0, 0] = K[:, :, 1, 1] = 2.8 * W
    K[:, :, 0, 2] = W / 2
    K[:, :, 1, 2] = H / 2
    E = torch.eye(4).view(1, 1, 4, 4).repeat(B, V, 1, 1)
    for v in range(1, V):
        E[:, v, 0, 3] = 40.0 * v
    gt = torch.rand(B, H, W) * (D_MAX - D_MIN) * 0.6 + D_MIN + 50
    b = {
        "images": torch.rand(B, V, 3, H, W) * 255,
        "intrinsics": K, "extrinsics": E,
        "depth_prior": gt + torch.randn(B, H, W) * 8,
        "conf_prior": torch.rand(B, H, W),
        "prior_valid": torch.ones(B),
        "depth_values": torch.linspace(D_MIN, D_MAX, 192).view(1, -1).repeat(B, 1),
        "depth_gt": gt, "mask": torch.ones(B, H, W),
    }
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


# ----------------------------------------------------------------- units
def run_units() -> None:
    torch.manual_seed(0)
    B, H, W = 2, 8, 10
    ok = lambda m: print(f"  [ok] {m}")

    for v in (1.5, 0.9, 0.6, 0.66, 0.20, 0.10, 0.5, 0.05):
        got = float(F.softplus(torch.tensor(inverse_softplus(v))))
        assert abs(got - v) < 1e-5, (v, got)
    ok("inverse_softplus 往返 —— softplus(1.5) != 1.5, 初始化必须取逆")

    g = SpreGates()().detach()
    assert abs(float(g[0]) - 1.0) < 1e-6 and all(
        float(g[i]) >= float(g[i + 1]) > 0 for i in range(3)), g
    ok(f"SpreGates 单调且有限 (logit(1.0)=inf 已避开): {[round(float(x), 4) for x in g]}")

    Dn = 16
    hyp = torch.linspace(500, 600, Dn).view(1, Dn, 1, 1).expand(B, Dn, H, W).contiguous()
    iv = local_intervals(hyp)
    assert torch.allclose(iv[:, 1:-1], torch.full_like(iv[:, 1:-1], 100 / (Dn - 1)), atol=1e-4)
    mode = torch.randint(0, Dn, (B, 1, H, W))
    g_v = torch.full((B, 1, 1), 1e-5)
    assert torch.allclose(robust_local_scale(torch.full_like(iv, 1e-12), mode, g_v),
                          g_v.expand(B, H, W))
    ok("local_intervals 逐 bin 正确; robust_local_scale 在近重复候选上回退 g_v")

    # 回归守卫: local_intervals 的 min_interval 是**有量纲**的。默认 1e-4 只对
    # 毫米深度轴成立; 逆深度间距 ~1e-5, 用默认值会把 w_bar 整张图钉成同一个常数
    # (窗口宽 3-20 倍), 而且完全静默 —— 这个 bug 只能靠 "p50 == p90" 这类
    # 退化检查抓出来, 不会报错。
    d_ax = torch.linspace(D_MAX, D_MIN, 48).view(1, 48, 1, 1).expand(B, 48, H, W)
    v_ax = 1.0 / d_ax                                       # 逆深度升序, 间距 ~1e-5
    iv_bad = local_intervals(v_ax)                          # 默认 1e-4
    iv_good = local_intervals(v_ax, min_interval=0.0)
    assert abs(float(iv_bad.min()) - 1e-4) < 1e-9 and abs(float(iv_bad.max()) - 1e-4) < 1e-9, \
        "构造前提: 这条轴的真实间距全部小于 1e-4"
    assert float(iv_good.max()) / float(iv_good.min()) > 2.0, \
        "min_interval=0 之后逆深度间距必须重新有 d^2 的动态范围"
    wb = robust_local_scale(iv_good, torch.full((B, 1, H, W), 24), g_v)
    assert float(wb.max()) / float(wb.min()) >= 1.0 and torch.isfinite(wb).all()
    ok(f"local_intervals 的下限有量纲: 逆深度轴必须传 min_interval=0 "
       f"(默认 1e-4 会把 {float(iv_bad.mean()):.1e} 钉死, 真实值 {float(iv_good.mean()):.2e})")

    v_min = torch.full((B, 1, 1), 1.0 / D_MAX)
    v_max = torch.full((B, 1, 1), 1.0 / D_MIN)
    v_c = ((v_min + v_max) / 2).expand(B, H, W).contiguous()
    ax = build_axis_inverse(v_c, torch.full((B, H, W), 2e-6),
                           torch.full((B, H, W), 5e-6), 4, v_min, v_max)
    assert (ax[:, 1:] > ax[:, :-1]).all()
    for lam in (0.0, 0.3, 1.0):
        bl = blend_axes(torch.linspace(500, 520, 4).view(1, 4, 1, 1).expand(B, 4, H, W), ax, lam)
        assert (bl[:, 1:] >= bl[:, :-1]).all()
    ok("build_axis_inverse 深度升序且在物理范围内; blend_axes 凸组合保序")

    D1 = 48
    hyp1 = torch.linspace(D_MIN, D_MAX, D1).view(1, D1, 1, 1).expand(B, D1, H, W).contiguous()
    p = torch.zeros(B, D1, H, W); p[:, 10] = 0.6; p[:, 40] = 0.3; p += 1e-3
    p = p / p.sum(1, keepdim=True)
    hsep = torch.full((B, H, W), 2e-6)
    _, mass, dv2, found, j2 = second_mode_physical(p, 1.0 / hyp1, p.argmax(1, keepdim=True), hsep, hsep)
    assert j2.shape == (B, 1, H, W) and (j2 == 40).all(), "winner 在 bin 10, 次峰应当是 bin 40"
    assert found.all() and (mass > 0.25).all()
    ok(f"second_mode_physical(物理逆深度邻域) mass={float(mass.mean()):.3f}")

    cfg = MVSConfig().depth_range
    ctrl = RangeController(range_k=cfg.range_k, range_min_gi=(0.66, 0.20, 0.10),
                           entropy_a=cfg.range_entropy_a, edge_b=cfg.range_edge_b)
    feats = torch.zeros(B, ctrl.in_ch, H, W)
    ent, edge, r = torch.rand(B, H, W), torch.rand(B, H, W), torch.rand(B, H, W)
    wbar = torch.full((B, H, W), 3e-6)
    for k in range(3):
        vc, hl, hh, _ = ctrl(k, feats, v_c, wbar, g_v, ent, edge, r,
                             torch.tensor(0.0), v_min, v_max, blend=0.0)
        assert torch.allclose(vc, v_c, atol=1e-9)
        a = (float(F.softplus(ctrl.kappa[k])) * wbar
             * (1 + float(F.softplus(ctrl.w_ent[k])) * ent
                + float(F.softplus(ctrl.w_edge[k])) * edge)).double()
        b = (float(F.softplus(ctrl.eta[k])) * g_v.expand_as(wbar)).double()
        # 参考值必须 float64 —— fp32 下 a**16 ~ 1e-80 直接下溢成 0。
        h0 = ((a ** 16 + b ** 16) ** (1 / 16)).float()
        hmax = torch.maximum(a, b).float()
        assert torch.allclose(hl, h0, rtol=2e-3) and torch.allclose(hl, hh)
        assert (hl / hmax <= 1.045).all() and (hl / hmax >= 0.999).all()
    ok("RangeController blend=0 复现 max(k*w*(1+aH+bE), f*g), 软最大高 <=4.4%")

    _, hl2, _, _ = ctrl(2, feats, v_c, torch.full((B, H, W), 1e-5),
                        torch.full((B, 1, 1), 1e-5), ent, edge, r,
                        torch.tensor(0.0), v_min, v_max, blend=0.0)
    assert (hl2 > 0).all() and torch.isfinite(hl2).all()
    ok(f"p-范数走 log 域 logsumexp, 逆深度量纲下不下溢 (h={float(hl2.mean()):.3e})")

    vc3, hl3, hh3, _ = ctrl(0, feats, v_c, torch.full((B, H, W), 1.0), g_v, ent, edge, r,
                            torch.tensor(0.0), v_min, v_max, blend=0.0)
    assert float((hl3 + hh3).max()) <= float((v_max - v_min)[0]) * 1.001
    assert (build_axis_inverse(vc3, hl3, hh3, 4, v_min, v_max).diff(dim=1) > 0).all()
    ok("区间宽于物理域时先按 rho 整体缩放再夹中心, 轴仍严格升序")

    dec = DepthDecoder(in_channels=8, base=8, depth=2, mode_window=2, head_mode="map")
    hyp4 = torch.linspace(500, 504, 4).view(1, 4, 1, 1).expand(B, 4, H, W).contiguous()
    d, sig, pr, _, mi, _ = dec(torch.randn(B, 8, 4, H, W), hyp4)
    assert torch.allclose(d, hyp4.gather(1, mi).squeeze(1))
    ok("DepthDecoder(map) 的输出恒等于某个真实候选, 不会落在两个表面之间")

    rh = Stage4ResidualHead(num_depths=4, feat_channels=16, hidden=16)
    res = rh(pr, hyp4, local_intervals(hyp4), mi, sig,
             torch.randn(B, 16, H, W), torch.rand(B, 2, H, W))
    assert res.shape == (B, 4, H, W) and float(res.abs().max()) == 0.0
    ok("Stage4ResidualHead 逐候选输出 D 个残差, 零初始化 -> step 0 残差恒为 0")

    from models.fusion_conf import (
        FusionConfidenceHead, brier_score, expected_calibration_error,
        fit_temperature, risk_at_coverage, risk_coverage,
    )
    ch = FusionConfidenceHead(feat_channels=16, hidden=16, prior_pos=0.9)
    lg = ch(torch.randn(B, ch.IN_CH, H, W), torch.randn(B, 16, H, W))
    assert lg.shape == (B, H, W)
    assert abs(float(torch.sigmoid(lg).mean()) - 0.9) < 1e-4, "零权重 -> 输出恒为先验"
    assert torch.allclose(ch.calibrated(lg), torch.sigmoid(lg))     # 未标定 T=1
    ok("FusionConfidenceHead 零权重时输出恒等于先验 0.9; 未标定时 T=1")

    # 温度标定: 真实 logit 是 u, 模型报的是 2u —— 恰好过自信 2 倍, 正确的 T = 2。
    # (不要用 "y 决定 z" 那种构造: 那样 z 几乎完美可分, 最优 T 反而 < 1 —— 那是
    #  欠自信, 不是过自信。)
    torch.manual_seed(3)
    u = torch.randn(200000)
    y = (torch.rand(200000) < torch.sigmoid(u)).float()
    z = 2.0 * u
    T = fit_temperature(z, y)
    assert abs(T - 2.0) < 0.1, f"过自信 2 倍时最优温度应当接近 2.0, 得到 {T}"
    e0 = expected_calibration_error(torch.sigmoid(z), y)
    e1 = expected_calibration_error(torch.sigmoid(z / T), y)
    assert T > 1.0 and e1 < e0, (T, e0, e1)
    ok(f"fit_temperature 修正过自信: T={T:.3f}, ECE {e0:.4f} -> {e1:.4f}")

    # 风险-覆盖: 置信度与误差完全反相关时 AURC 必须低于随机排序
    conf = torch.rand(5000)
    riskv = 1.0 - conf                       # 完美排序
    _, _, a_good = risk_coverage(conf, riskv)
    _, _, a_rand = risk_coverage(torch.rand(5000), riskv)
    assert a_good < a_rand, (a_good, a_rand)
    assert risk_at_coverage(conf, riskv, 0.6) < risk_at_coverage(conf, riskv, 1.0)
    assert brier_score(torch.full((100,), 0.5), torch.ones(100)) == 0.25
    ok(f"risk_coverage: 完美排序 AURC={a_good:.4f} < 随机 {a_rand:.4f}; Brier 口径正确")

    # ---- W3-B: uv_img 的坐标约定 ----
    # 这是整个 W3-B 里唯一会**静默**写错的地方。特征网格的归一化坐标直接拿去采
    # 全分辨率 GT 会差半个 stride, 而遮挡标签恰恰只在遮挡边界上有意义 —— 那正是
    # 半个 stride 会取到另一个表面的地方。
    # 构造: 零基线 (E_ref == E_src), 于是投影退化成参考视图自己的像素网格,
    # uv_img 必须精确对应 "参考特征像素 x stride"。用一张 val(i,j)=j 的斜坡图
    # 采样, 采回来的值就应当等于 j_ref * stride。
    from utils.geometry import homography_warp_features
    st, Hf, Wf, Dn2 = 8, 6, 7, 3
    Ki = torch.eye(3).unsqueeze(0)
    Ki[0, 0, 0] = Ki[0, 1, 1] = 700.0
    Ki[0, 0, 2], Ki[0, 1, 2] = Wf * st / 2, Hf * st / 2
    Ei = torch.eye(4).unsqueeze(0)
    dh = torch.full((1, Dn2, Hf, Wf), 600.0)
    _, _, uv_img, z_src = homography_warp_features(
        torch.zeros(1, 4, Hf, Wf), Ki, Ki, Ei, Ei, dh, st, return_geom=True)
    Hi, Wi = Hf * st, Wf * st
    ramp = torch.arange(Wi, dtype=torch.float32).view(1, 1, 1, Wi).expand(1, 1, Hi, Wi)
    got = F.grid_sample(ramp, uv_img.reshape(1, Dn2 * Hf, Wf, 2),
                        mode="bilinear", align_corners=True).view(1, Dn2, Hf, Wf)
    want = (torch.arange(Wf, dtype=torch.float32) * st).view(1, 1, 1, Wf).expand(1, Dn2, Hf, Wf)
    assert torch.allclose(got, want, atol=1e-3), (got[0, 0, 0], want[0, 0, 0])
    # 零基线下 z_src 必须恒等于候选深度本身
    assert torch.allclose(z_src, dh, atol=1e-3)
    ok("uv_img 归一化到整幅图像且无半 stride 偏移 (零基线下精确落在 j*stride)")
    print("  单元检查全部通过")


# --------------------------------------------------------- equivalence
def run_equivalence(tol: float, device: str) -> None:
    from models.network import UprMVSNet
    B, V, H, W = 1, 3, 64, 80
    torch.manual_seed(1234); nl = UprMVSNet(_cfg("legacy_depth", "expect")).to(device).eval()
    torch.manual_seed(1234); ni = UprMVSNet(_cfg("inverse", "expect")).to(device).eval()

    sd_l, sd_i = nl.state_dict(), ni.state_dict()
    shared = [k for k in sd_l if k in sd_i and sd_l[k].shape == sd_i[k].shape]
    diff = [k for k in shared if not torch.equal(sd_l[k], sd_i[k])]
    print(f"  共享参数 {len(shared)}/{len(sd_l)}, 初值不同的 {len(diff)}")
    assert not diff, f"RNG 流被打乱了: {diff[:5]}"
    print("  [ok] 两种配置的初始权重完全相同 (新模块无条件构造, 不改 RNG 流)")

    torch.manual_seed(7)
    b = _batch(B, V, H, W, device)
    with torch.no_grad():
        ol, oi = nl(b, step=0), ni(b, step=0)          # blend = 0
    worst = 0.0
    for k in ("stage2", "stage3", "stage4"):
        m = float((ol[k]["depth_hypos"] - oi[k]["depth_hypos"]).abs().max())
        worst = max(worst, m)
        print(f"    {k:8s} max|d_new - d_old| = {m:.3e}")
    md = float((ol["depth_full"] - oi["depth_full"]).abs().max())
    print(f"    depth_full max|Δ|      = {md:.3e}")
    assert worst <= tol and md <= tol, f"step 0 必须与 legacy 一致 (容差 {tol})"
    print(f"  [ok] axis_space=inverse 在 step 0 与 legacy 一致 (容差 {tol})")

    # 工单第 3 项的验收 (2): depth / probability / 各项 loss 都要在容差内相同。
    # 只比深度是不够的 —— 损失里换掉了 Huber 的 scale (span/(D-1) -> 局部间距),
    # legacy 路径下两者恒等, 不等就说明 interval 记账写错了。
    from losses.composite import MVSLoss
    zero_range = {"w_range": 0.0, "w_center": 0.0}
    lf_l = MVSLoss(_cfg("legacy_depth", "expect", weights=zero_range).loss, _cfg().stage_weights)
    lf_i = MVSLoss(_cfg("inverse", "expect", weights=zero_range).loss, _cfg().stage_weights)
    tl, logs_l = lf_l(ol, b, step=0)
    ti, logs_i = lf_i(oi, b, step=0)
    dl = abs(float(tl) - float(ti))
    print(f"    total loss (w_range=0)  legacy={float(tl):.6f}  inverse={float(ti):.6f}"
          f"  |Δ|={dl:.3e}")
    # 把新增的监督关掉之后, 两条路的**共享**计算必须逐项一致。这一条专门盯
    # Huber 的 scale: 它从 span/(D-1) 换成了逐像素的 winner 局部间距, legacy
    # 轴上两者恒等 —— 不等就说明 interval 记账写错了。
    worst_k, worst_v = None, 0.0
    for k in (k for k in logs_l if k in logs_i):
        d = abs(float(logs_l[k]) - float(logs_i[k]))
        if d > worst_v:
            worst_k, worst_v = k, d
    print(f"    共享 loss 项最大差异: {worst_k}={worst_v:.3e}")
    assert dl <= max(tol, 1e-4) and worst_v <= max(tol, 1e-4), \
        f"关掉新监督之后 legacy 路径的损失必须一致 ({worst_k}={worst_v})"
    print("  [ok] 关掉 range 监督后各项 loss 逐项一致 (Huber scale 换算无误)")

    # 打开 range 监督之后, 两者的差**必须恰好等于**新增的那一项。差得更多就说明
    # 新代码顺手改了共享路径上的东西。
    lf_r = MVSLoss(_cfg("inverse", "expect").loss, _cfg().stage_weights)
    tr, logs_r = lf_r(oi, b, step=0)
    added = float(logs_r.get("range/total", 0.0)) * float(_cfg().loss.w_range)
    resid = abs((float(tr) - float(ti)) - added)
    print(f"    w_range=1 时增量={float(tr) - float(ti):.6f}, range/total={added:.6f}, "
          f"残差={resid:.3e}")
    assert resid <= max(tol, 1e-3), "总损失的增量必须恰好是 range 项"
    print("  [ok] 新增损失只来自 pinball+center, 没有动共享路径")

    # 验收 (3): checkpoint 参数加载无缺失、无意外重映射。
    #
    # 注意**哪些开关会改参数形状**: spre_cascade 给 stage2-4 的正则器 +2 通道,
    # geo_valid +1 通道。所以 state_dict 只在**同一组开关**下可互换 —— 这正是
    # test.py 必须从 fingerprint 恢复这几个开关的原因, 也是这里要验的东西。
    for tag, kw in (("W1", dict(axis="inverse", stage4="map", spre_cascade=True)),
                    ("W3AC", dict(axis="inverse", stage4="map", spre_cascade=True,
                                  geo_valid=True, conf_head=True)),
                    ("W3ABC", dict(axis="inverse", stage4="map", spre_cascade=True,
                                   geo_valid=True, conf_head=True,
                                   vis_mode="sigmoid", vis_supervise=True))):
        torch.manual_seed(11)
        src = UprMVSNet(_cfg(**kw)).to(device)
        sd = src.state_dict()
        torch.manual_seed(999)                       # 故意换种子: 必须靠加载对齐
        dst = UprMVSNet(_cfg(**kw)).to(device)
        miss, unexp = dst.load_state_dict(sd, strict=False)
        assert not miss and not unexp, (tag, list(miss)[:5], list(unexp)[:5])
        dst.load_state_dict(sd)                      # strict=True 也必须过
        bad = [k for k in sd if not torch.equal(sd[k], dst.state_dict()[k])]
        assert not bad, (tag, bad[:5])
        print(f"    {tag:6s} {len(sd):3d} 键: 无缺失 / 无意外 / 逐元素一致")
        del src, dst

    # 辅助头一律**无条件构造** (只用开关决定参不参与前向), 所以 legacy 与
    # inverse 的键集必须相同 —— 否则换个 arm 连键名都对不上。
    kl, ki = set(nl.state_dict()), set(ni.state_dict())
    assert kl == ki, (sorted(kl - ki)[:5], sorted(ki - kl)[:5])
    for pre in ("range_ctrl.", "conf_head.", "res_head.", "spre_gates."):
        assert any(k.startswith(pre) for k in kl), f"{pre} 不在 state_dict 里"
    print("  [ok] legacy / inverse 键集相同, 四个辅助头无条件出现在 checkpoint 里")

    with torch.no_grad():
        oi2 = ni(b, step=10 ** 6)                       # blend = 1
    print("  step 迁移完成后与 legacy 的轴差 —— 这是坐标变化本身, 是要记录的量:")
    for k in ("stage2", "stage3", "stage4"):
        print(f"    {k:8s} mean|Δ| = "
              f"{float((ol[k]['depth_hypos'] - oi2[k]['depth_hypos']).abs().mean()):.3f} mm")
        assert (oi2[k]["depth_hypos"].diff(dim=1) >= 0).all(), f"{k} 轴非升序"
    print("  [ok] 迁移后三级候选轴仍严格升序")


# ---------------------------------------------------------------- memory
def run_mem(batch: int, views: int, device: str) -> None:
    from losses.composite import MVSLoss
    from models.network import UprMVSNet
    if device == "cpu":
        print("  跳过: 没有 CUDA, 显存峰值只能在 GPU 上量")
        return
    H, W = MVSConfig().data.target_h, MVSConfig().data.target_w
    print(f"  分辨率 {H}x{W}, per-GPU batch={batch}, views={views}")
    rows = []
    for tag, kw in (("legacy", dict(axis="legacy_depth", stage4="expect")),
                    ("W1", dict(axis="inverse", stage4="map", spre_cascade=True)),
                    ("W1+W3A", dict(axis="inverse", stage4="map", spre_cascade=True,
                                    geo_valid=True, vis_mode="sigmoid")),
                    ("W1+W3AC", dict(axis="inverse", stage4="map", spre_cascade=True,
                                     geo_valid=True, vis_mode="sigmoid", conf_head=True)),
                    ("W1+W3ABC", dict(axis="inverse", stage4="map", spre_cascade=True,
                                      geo_valid=True, vis_mode="sigmoid", conf_head=True,
                                      vis_supervise=True))):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        cfg = _cfg(**kw)
        net = UprMVSNet(cfg).to(device).train()
        lf = MVSLoss(cfg.loss, cfg.stage_weights)
        b = _batch(batch, views, H, W, device)
        if cfg.cost_volume.vis_supervise:
            b["src_depth_gt"] = torch.rand(batch, views - 1, H, W, device=device) * 300 + 500
            b["src_mask_gt"] = torch.ones(batch, views - 1, H, W, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(b, step=10 ** 6)
            tot, _ = lf(out, b, step=10 ** 6)
        tot.float().backward()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        rows.append((tag, peak, float(tot)))
        print(f"    {tag:8s} peak={peak:6.2f} GiB   loss={float(tot):.3f}")
        del net, lf, out, tot, b
        torch.cuda.empty_cache()
    base = rows[0][1]
    for tag, peak, _ in rows[1:]:
        print(f"  {tag} 相对 legacy: {peak - base:+.2f} GiB ({100 * (peak / base - 1):+.1f}%)")
    print(f"  A100-80GB 余量: {80 - max(r[1] for r in rows):.1f} GiB")


# ------------------------------------------------------------------ ddp
def run_ddp_precondition(device: str) -> None:
    """DDP(find_unused_parameters=False) 的**充要前提**: 每一个 requires_grad
    的参数都必须在每一步收到梯度。

    单卡上违反它什么都不会发生, 多卡上第二个 iteration 直接抛
    "Expected to have finished reduction in the prior iteration"。所以这条要在
    提交双卡任务**之前**在单卡上查掉 —— 排队几个小时再撞上它太贵了。
    """
    from losses.composite import MVSLoss
    from models.network import UprMVSNet
    B, V, H, W = 1, 3, 64, 80
    cfgs = {
        "legacy": dict(axis="legacy_depth", stage4="expect"),
        "W1": dict(axis="inverse", stage4="map", spre_cascade=True),
        "W1+W3AC": dict(axis="inverse", stage4="map", spre_cascade=True,
                        geo_valid=True, conf_head=True),
        "W1+W3ABC": dict(axis="inverse", stage4="map", spre_cascade=True,
                         geo_valid=True, conf_head=True,
                         vis_mode="sigmoid", vis_supervise=True),
        "vis_softmax": dict(axis="inverse", stage4="map", vis_mode="softmax",
                            visibility=True),
        # 下面四个是"开关打开但对应的损失权重是 0"。这类组合单卡完全正常,
        # 双卡在第二个 iteration 才炸 —— 必须由冻结策略兜住, 所以要查。
        "w_range=0": dict(axis="inverse", stage4="map", weights={"w_range": 0.0}),
        "w_residual=0": dict(axis="inverse", stage4="map", weights={"w_residual": 0.0}),
        "w_conf=0": dict(axis="inverse", stage4="map", geo_valid=True,
                         conf_head=True, weights={"w_conf": 0.0}),
        "all_w=0": dict(axis="inverse", stage4="map", geo_valid=True, conf_head=True,
                        weights={"w_range": 0.0, "w_residual": 0.0, "w_conf": 0.0}),
    }
    bad_total = 0
    for tag, kw in cfgs.items():
        torch.manual_seed(0)
        cfg = _cfg(**kw)
        net = UprMVSNet(cfg).to(device).train()
        lf = MVSLoss(cfg.loss, cfg.stage_weights)
        b = _batch(B, V, H, W, device)
        if cfg.cost_volume.vis_supervise:
            b["src_depth_gt"] = torch.rand(B, V - 1, H, W, device=device) * 300 + 500
            b["src_mask_gt"] = torch.ones(B, V - 1, H, W, device=device)
        out = net(b, step=10 ** 6)
        tot, _ = lf(out, b, step=10 ** 6)
        tot.float().backward()
        missing = [n for n, p in net.named_parameters()
                   if p.requires_grad and p.grad is None]
        n_train = sum(1 for p in net.parameters() if p.requires_grad)
        n_froz = sum(1 for p in net.parameters() if not p.requires_grad)
        flag = "ok " if not missing else "FAIL"
        print(f"  [{flag}] {tag:10s} 可训练 {n_train:3d} / 冻结 {n_froz:3d}"
              f" / 无梯度 {len(missing)}")
        for n in missing[:6]:
            print(f"          缺梯度: {n}")
        bad_total += len(missing)
        del net, lf, out, tot, b
    if bad_total:
        raise SystemExit("有参数要梯度却没收到 —— 双卡 DDP 会在第二步报错")
    print("  所有配置都满足 DDP find_unused_parameters=False 的前提")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--ddp-check", action="store_true")
    ap.add_argument("--equivalence", action="store_true")
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--views", type=int, default=5)
    a = ap.parse_args()
    if not (a.units or a.equivalence or a.mem or a.ddp_check):
        a.units = a.equivalence = a.ddp_check = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.units:
        print("== 单元检查 =="); run_units()
    if a.equivalence:
        print("== legacy 等价性 =="); run_equivalence(a.tol, dev)
    if a.ddp_check:
        print("== DDP 前提 (每个可训练参数都要收到梯度) =="); run_ddp_precondition(dev)
    if a.mem:
        print("== 显存峰值 =="); run_mem(a.batch, a.views, dev)


if __name__ == "__main__":
    main()
