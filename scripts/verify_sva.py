#!/usr/bin/env python3
"""ARM=sva (2026-09-18) 的实现校验 —— **不是性能实验**。

只回答 "代码对不对", 任何一条挂掉都不要通过调参绕过去:

   1  NormalizedPE2D 与 MVSFormer++ PositionEncodingSineNorm 逐元素一致
   2  线性注意力与 MVSFormer++ CrossLinearAttention 一致
   3  HighResSVA (FMT) 把 V-1 个 source 拼在 token 维上查 reference, 与逐视角循环一致
   4  SVAPathway 与 FMT_with_pathway 的逐视角参考实现 (同一组权重) 一致
   5  FPN stage1_head=out0: 1/8 输出 = out0(p8), 普通 top-down 从 out0 之前的 p8 出发
   6  sva.full=off 时是旧结构 (to_fpn 1x1 / smooth_p8 / 没有 deconv、out0、sva_pathway)
   7  stage1 = 44 global + 4 local, 候选轴升序
   8  全网前向 + 反向: 每个可训练参数都有梯度, depth 有限
   9  fingerprint 往返: test._align_cfg_to_ckpt 恢复出 sva.full / sva_layout / 44/4 /
      range_min_gi, 按它重建的模型能 strict 加载; 缺 sva_layout 的 (cdaa5d5) 被拒绝
  10  CVPE 已卸载: cvpe.enabled=True 构造网络直接报错; 带 cvpe 的旧 fingerprint 被 test 拒绝

需要 DINOv3 权重 (cfg.paths.dinov3_weights_file)。有 GPU 就在 GPU 上跑。

    python scripts/verify_sva.py
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import replace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import build_mvs_config
from models.sva import HighResSVA, NormalizedPE2D, SVAPathway, _linear_attn

_OK: list[str] = []


def ok(msg: str) -> None:
    _OK.append(msg)
    print(f"  [ok] {msg}")


# ---------------------------------------------------------------------------
# MVSFormer++ 的参考实现, 原样抄录 (reference/MVSFormerPlusPlus/models/
# position_encoding.py 与 dino/layers/attention.py)。集群上没有 reference 目录,
# 所以不 import。
# ---------------------------------------------------------------------------
def _ref_pe(d_model: int, H: int, W: int, max_shape=(128, 128)) -> torch.Tensor:
    pe = torch.zeros((d_model, H, W))
    y_position = torch.ones((H, W)).cumsum(0).float().unsqueeze(0) * max_shape[0] / H
    x_position = torch.ones((H, W)).cumsum(1).float().unsqueeze(0) * max_shape[1] / W
    div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
    div_term = div_term[:, None, None]
    pe[0::4, :, :] = torch.sin(x_position * div_term)
    pe[1::4, :, :] = torch.cos(x_position * div_term)
    pe[2::4, :, :] = torch.sin(y_position * div_term)
    pe[3::4, :, :] = torch.cos(y_position * div_term)
    return pe.unsqueeze(0)


def _ref_linear_attn(q_proj, k_proj, v_proj, x, key, value, num_heads):
    eps = 1e-6
    B, N, C = x.shape
    q = q_proj(x).reshape(B, N, num_heads, C // num_heads).to(dtype=torch.float32)
    k = k_proj(key).reshape(B, N, num_heads, C // num_heads).to(dtype=torch.float32)
    v = v_proj(value).reshape(B, N, num_heads, C // num_heads).to(dtype=torch.float32)
    q = torch.nn.functional.elu(q) + 1
    k = torch.nn.functional.elu(k) + 1
    KV = torch.einsum("nshd,nshm->nhmd", k, v)
    Z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + eps)
    V = torch.einsum("nlhd,nhmd,nlh->nlhm", q, KV, Z)
    return V.reshape(B, N, C).contiguous()


@torch.no_grad()
def check_modules() -> None:
    torch.manual_seed(0)
    # 1
    for (h, w) in ((64, 80), (120, 160), (37, 51)):
        d = float((_ref_pe(128, h, w) - NormalizedPE2D(128).encoding(h, w, "cpu", torch.float32)).abs().max())
        assert d < 1e-4, f"PE 与 MVSFormer++ 不一致 ({h}x{w}): max|diff|={d}"
    ok("NormalizedPE2D == PositionEncodingSineNorm (64x80 / 120x160 / 37x51, fp32 舍入内)")

    # 2
    C, heads = 64, 4
    lin = [nn.Linear(C, C, bias=False) for _ in range(3)]
    x, kv = torch.randn(2, 50, C), torch.randn(2, 50, C)
    ref = _ref_linear_attn(*lin, x, kv, kv, heads)
    mine = _linear_attn(lin[0](x), lin[1](kv), lin[2](kv), heads)
    d = float((ref - mine).abs().max())
    assert d < 1e-5, f"线性注意力与 CrossLinearAttention 不一致: {d}"
    ok(f"线性注意力 == CrossLinearAttention (max|diff| {d:.1e})")

    # 3
    hr = HighResSVA(128).double()
    feat = torch.randn(2, 5, 128, 12, 16, dtype=torch.float64)
    y = hr(feat)
    B, V, Cc, h, w = feat.shape
    tok = hr.pe(feat.reshape(B * V, Cc, h, w)).view(B, V, Cc, h, w).flatten(3).transpose(2, 3)
    r = tok[:, 0]
    refs = []
    for L, n in zip(hr.layers, hr.layer_names):
        if n == "self":
            r = L(r)
            refs.append(r)
    outs = [r]
    for v in range(1, V):
        s = tok[:, v]
        for i, (L, n) in enumerate(zip(hr.layers, hr.layer_names)):
            s = L(s) if n == "self" else L(s, kv=refs[i // 2])
        outs.append(s)
    y_loop = torch.stack(outs, 1).transpose(2, 3).reshape(B, V, Cc, h, w)
    d = float((y - y_loop).abs().max())
    assert d < 1e-5, f"批量 cross 与逐视角循环不一致: {d}"
    ok(f"HighResSVA 批量 cross == 逐视角循环 (max|diff| {d:.1e})")

    # 4  FMT_with_pathway 的逐视角写法, 原样照抄其 forward 的数据流, 共用同一组权重
    pw = SVAPathway(32, heads=4).double().eval()
    fs = {8: torch.randn(2, 3, 32, 6, 8, dtype=torch.float64),
          4: torch.randn(2, 3, 32, 12, 16, dtype=torch.float64),
          2: torch.randn(2, 3, 32, 24, 32, dtype=torch.float64),
          1: torch.randn(2, 3, 32, 48, 64, dtype=torch.float64)}
    got = pw(fs)
    s1 = pw.fmt(fs[8])                      # FMT 已由 3 单独核对, 这里核对 pathway

    def up_add(x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="bilinear") + y
    for v in range(3):
        r2 = pw.smooth[0](up_add(pw.reduce[0](s1[:, v]), fs[4][:, v]))
        r3 = pw.smooth[1](up_add(pw.reduce[1](r2), fs[2][:, v]))
        r4 = pw.smooth[2](up_add(pw.reduce[2](r3), fs[1][:, v]))
        for st, ref in ((8, s1[:, v]), (4, r2), (2, r3), (1, r4)):
            d = float((got[st][:, v] - ref).abs().max())
            assert d < 1e-9, f"SVAPathway stride {st} view {v} 与 FMT_with_pathway 不一致: {d}"
    ok("SVAPathway == FMT_with_pathway 的逐视角数据流 (四级, 同一组权重)")

    # 5  普通 FPN: out0 头 + 从 out0 之前出发的 top-down
    from models.fpn import FPNFeatureExtractor
    f = FPNFeatureExtractor(out_channels=32, base_channel=8, stage1_head="out0").eval()
    x = torch.rand(2, 3, 64, 80)
    dn = torch.randn(2, 32, 8, 10)
    o = f(x, dino=dn)
    fh = f.tower_half(x); fq = f.tower_quarter(fh); fe = f.tower_eighth(fq)
    p8 = f.lateral_eighth(fe) + dn
    p4 = f.lateral_quarter(fq) + F.interpolate(p8, size=fq.shape[-2:], mode="bilinear", align_corners=False)
    assert torch.equal(o[8], f.out0(p8)), "1/8 输出不是 out0(p8)"
    assert torch.equal(o[4], f.smooth_p4(p4)), "1/4 的普通 top-down 没有从 out0 之前的 p8 出发"
    assert not hasattr(f, "smooth_p8"), "out0 模式下不该再有 smooth_p8 (不前向的参数)"
    ok("FPN stage1_head=out0: 1/8 = out0(p8), 普通 top-down 从 out0 之前的 p8 出发")


def _sva_cfg(full: bool = True):
    cfg = build_mvs_config()
    return replace(
        cfg,
        spre=replace(cfg.spre, enabled=True, reliability_source="spre"),
        dino=replace(cfg.dino, mode="all_view", feed_fpn=True),
        sva=replace(cfg.sva, full=full),
        cvpe=replace(cfg.cvpe, enabled=False),
        depth_range=replace(cfg.depth_range, num_global=44, num_local=4,
                            range_min_gi=(0.9155, 0.2774, 0.1387), axis_space="legacy_depth",
                            stage4_head="expect", spre_cascade=False,
                            gate_local_branch=False, branch_prior=False),
        cost_volume=replace(cfg.cost_volume, num_depths_stage1=48, geo_valid_aggregation=True,
                            visibility_weighting=False),
        decoder=replace(cfg.decoder, fusion_conf=True, fusion_conf_detach=True),
        loss=replace(cfg.loss, w_conf=1.0, w_branch=0.0),
        train=replace(cfg.train, batch_size=2, num_views=5, amp_dtype="bf16"),
    )


def check_network(device: torch.device) -> None:
    from losses.composite import MVSLoss
    from models.network import UprMVSNet
    import train as trainmod
    import test as testmod

    # 6
    net_off = UprMVSNet(_sva_cfg(full=False))
    assert net_off.dino_sva.to_fpn is not None and net_off.dino_sva.deconv is None \
        and net_off.sva_pathway is None and hasattr(net_off.fpn.fpn, "smooth_p8") \
        and not hasattr(net_off.fpn.fpn, "out0"), "sva.full=off 不是旧结构"
    ok("sva.full=off: 1x1 to_fpn + smooth_p8, 没有 deconv / out0 / sva_pathway")
    del net_off

    # 7, 8
    cfg = _sva_cfg(full=True)
    torch.manual_seed(0)
    model = UprMVSNet(cfg).to(device).train()
    loss_fn = MVSLoss(cfg.loss, cfg.stage_weights)
    batch = trainmod._synthetic_batch(cfg, device, 2, hw=(256, 320))
    use_amp = device.type == "cuda"
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        out = model(batch, step=0)
        loss, _ = loss_fn(out, batch, step=0)
    s1 = out["stage1"]
    D = s1["depth_hypos"].shape[1]
    n_loc = s1["is_local"].float().sum(1)
    assert D == 48 and bool((n_loc == 4).all()), f"stage1 候选 D={D}, local={n_loc.unique().tolist()}"
    assert bool((s1["depth_hypos"][:, 1:] >= s1["depth_hypos"][:, :-1]).all()), "stage1 候选轴不是升序"
    ok("stage1 = 44 global + 4 local, 候选轴升序")
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"这些可训练参数没收到梯度: {missing[:8]} (共 {len(missing)})"
    _sva_prefix = ("sva_pathway.", "dino_sva.deconv.", "fpn.fpn.out0.")
    hr_grad = [n for n, p in model.named_parameters()
               if n.startswith(_sva_prefix) and float(p.grad.abs().max()) == 0]
    assert not hr_grad, f"完整 SVA 的参数梯度恒为 0: {hr_grad[:8]}"
    assert torch.isfinite(out["depth_full"]).all() and torch.isfinite(loss), "前向出现非有限值"
    n_new = sum(p.numel() for n, p in model.named_parameters() if n.startswith(_sva_prefix)) / 1e6
    ok(f"前向 + 反向: 每个可训练参数都有梯度, 完整 SVA 新增 {n_new:.2f}M 参数, "
       f"sva/delta_rel={out['range_diag']['sva']['delta_rel']:.3f}")

    # 9
    fp = trainmod._arch_fingerprint(cfg)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    restored, _ = testmod._align_cfg_to_ckpt(build_mvs_config(), state, "auto", fingerprint=fp)
    assert restored.sva.full and restored.depth_range.num_global == 44 \
        and restored.depth_range.num_local == 4, "fingerprint 没恢复出 sva.full / 44/4"
    assert tuple(restored.depth_range.range_min_gi) == (0.9155, 0.2774, 0.1387), \
        f"range_min_gi 没恢复: {restored.depth_range.range_min_gi}"
    assert restored.decoder.fusion_conf and restored.cost_volume.geo_valid_aggregation
    UprMVSNet(restored).load_state_dict(state, strict=True)
    try:
        testmod._align_cfg_to_ckpt(build_mvs_config(), state, "auto",
                                   fingerprint={k: v for k, v in fp.items() if k != "sva_layout"})
    except SystemExit:
        pass
    else:
        raise AssertionError("缺 sva_layout 的 (cdaa5d5 单路径) fingerprint 没有被 test 拒绝")
    ok("fingerprint 往返: sva.full / sva_layout / 44/4 / range_min_gi / conf_head / geo_valid "
       "恢复, strict 加载通过; cdaa5d5 的单路径 checkpoint 被拒绝")

    # 10
    try:
        UprMVSNet(replace(cfg, cvpe=replace(cfg.cvpe, enabled=True)))
    except ValueError as e:
        assert "CVPE" in str(e)
    else:
        raise AssertionError("cvpe.enabled=True 居然构造成功了 —— CVPE 没卸干净")
    try:
        testmod._align_cfg_to_ckpt(build_mvs_config(), state, "auto",
                                   fingerprint=dict(fp, cvpe_enabled=True))
    except SystemExit:
        pass
    else:
        raise AssertionError("带 CVPE 的旧 fingerprint 没有被 test 拒绝")
    ok("CVPE 已卸载: 网络拒绝 cvpe.enabled=True, test 拒绝带 CVPE 的 checkpoint")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify_sva] device={device}")
    print("[modules]")
    check_modules()
    print("[network]")
    check_network(device)
    print(f"\n[verify_sva] 全部通过 ({len(_OK)} 条检查)")


if __name__ == "__main__":
    main()
