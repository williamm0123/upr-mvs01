#!/usr/bin/env python3
"""UPRMVS vNext 的实现校验 —— **不是性能实验**。

工单 v5.3 只跑一次 30k 训练, 所以代码必须在提交作业之前就是对的。这里的每一条
都只回答 "代码跑不跑得对", 不回答 "模型好不好"; 任何一条挂掉都不允许通过调参
或改模型来"绕过"。

    python scripts/verify_vnext.py --all
    python scripts/verify_vnext.py --cvpe --equivalence     # 只跑其中两组

覆盖 (与工单 v5.3 §11 一一对应):
   1  cvpe=off 时 FPN 重构前后四级输出逐位一致
   2  K_ref==K_src, E_ref==E_src 时 warp 与输入特征对齐 (同一个投影函数)
   3  CVPE 输出形状 [B, V-1, C_in, h8, w8]
   4  reference 前向输出未被 CVPE 覆盖
   5  source 只做一次残差相加; delta=0 时输出严格等于输入
   6  CVPE 输出全部有限
   7  两步梯度: 零初始化时 out_proj 有梯度; 更新一次后 in_proj/cam/attn 也有
   8  旧 W0 checkpoint 在新增 calib_bias 与 CVPE 配置之后仍能 strict 加载
   9  fingerprint 保存/恢复之后 CVPE 配置完全相同
  10  photo_keep_ratio=0.6 精确保留 ceil(0.6 N)
  11  Platt 标定保持排序, AURC 与 risk@0.6 不变
  12  CPU 小张量前向; --cuda 时再加一次 BF16 CUDA smoke
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
from models.cvpe import CrossViewPE, img2world_params, inverse_depth_planes
from models.fpn import MultiViewFPN
from models.fusion_conf import FusionConfidenceHead, fit_platt, risk_at_coverage, risk_coverage
from models.network import UprMVSNet
from utils.geometry import homography_warp_features

D_MIN, D_MAX = 425.0, 935.0
_OK = []


def ok(msg: str) -> None:
    _OK.append(msg)
    print(f"  [ok] {msg}")


def _cfg(cvpe: bool = False, conf_head: bool = False, geo_valid: bool = False) -> MVSConfig:
    """vNext 的配置, 但把 DINO/SPRE 关掉 —— 它们要下载 ViT 权重, 而这里校验的
    是 CVPE / 融合 / checkpoint 兼容性, 与 DINO 无关。"""
    c = MVSConfig()
    c = replace(c, dino=replace(c.dino, mode="off", feed_fpn=False))
    c = replace(c, spre=replace(c.spre, enabled=False, reliability_source="cached"))
    c = replace(c, depth_range=replace(
        c.depth_range, num_global=32, num_local=16, range_min_gi=(0.66, 0.20, 0.10),
        axis_space="legacy_depth", stage4_head="expect", spre_cascade=False,
        gate_local_branch=False, branch_prior=False))
    c = replace(c, cost_volume=replace(
        c.cost_volume, num_depths_stage1=48, geo_valid_aggregation=geo_valid,
        visibility_weighting=False, vis_supervise=False))
    c = replace(c, decoder=replace(c.decoder, fusion_conf=conf_head, fusion_conf_detach=True))
    c = replace(c, loss=replace(c.loss, w_conf=1.0 if conf_head else 0.0, w_branch=0.0))
    c = replace(c, cvpe=replace(c.cvpe, enabled=cvpe))
    return c


def _arch_fp_min() -> dict:
    """_align_cfg_to_ckpt 必读的那几个字段 (其余都有 .get 默认值)。"""
    import train as trainmod
    return trainmod._arch_fingerprint(_cfg())


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


# ===================================================================== 1, 4, 5
def run_equivalence(tol: float = 0.0) -> None:
    """cvpe=off 的 FPN 与重构前逐位一致; delta=0 时 CVPE 也逐位一致。

    容差默认 **0** —— mid_hook=None 是纯粹的控制流分支, 一条算子都没多; 而
    ``x + 0`` 在 IEEE-754 下精确等于 x。这两条如果不是逐位相同, 说明重构引入了
    别的东西 (最常见的是 contiguous / view 顺序变了导致归约次序变化)。
    """
    torch.manual_seed(0)
    B, V, C, H, W = 2, 3, 128, 32, 40
    fpn = MultiViewFPN(out_channels=C, base_channel=8).eval()
    imgs = torch.rand(B, V, 3, H, W)

    with torch.no_grad():
        base = fpn(imgs)                                   # 走 mid_hook=None
        hooked = fpn(imgs, cvpe_fn=lambda p8: p8)          # 走 mid_hook, 恒等
    for s in (8, 4, 2, 1):
        assert torch.equal(base[s], hooked[s]), f"stride {s} 不是逐位一致"
    ok("cvpe=off 与恒等 mid_hook 的四级 FPN 输出**逐位**一致")

    # delta=0: out_proj 零初始化, 所以刚构造出来的 CVPE 必须是恒等。
    cv = CrossViewPE(in_channels=C).eval()
    p8 = torch.randn(B, V, C, H // 8, W // 8)
    K = torch.eye(3).view(1, 1, 3, 3).repeat(B, V, 1, 1) * 100.0
    K[:, :, 2, 2] = 1.0
    E = torch.eye(4).view(1, 1, 4, 4).repeat(B, V, 1, 1)
    E[:, 1:, 0, 3] = 40.0
    dmin = torch.full((B,), D_MIN)
    dmax = torch.full((B,), D_MAX)
    with torch.no_grad():
        delta = cv(p8=p8, K=K, E=E, depth_min=dmin, depth_max=dmax, feature_stride=8)
    assert torch.count_nonzero(delta) == 0, "out_proj 零初始化 -> delta 必须恒为 0"
    ok("out_proj 零初始化 ⇒ step 0 的 delta 恒为 0")

    merged = torch.cat([p8[:, :1], p8[:, 1:] + delta], dim=1)
    assert torch.equal(merged, p8), "delta=0 时残差相加必须逐位还原输入"
    ok("delta=0 时 src+delta **逐位**等于 src (残差只加了一次)")

    # 只加一次: 人为把 delta 设成已知常数, 输出必须正好差这个常数, 不是两倍。
    d1 = torch.full_like(delta, 0.25)
    m1 = torch.cat([p8[:, :1], p8[:, 1:] + d1], dim=1)
    assert torch.allclose(m1[:, 1:] - p8[:, 1:], d1), "source 被加了不止一次"
    assert torch.equal(m1[:, :1], p8[:, :1]), "reference 被 CVPE 改动了"
    ok("source 恰好加一次 delta; reference 逐位不变 (#4, #5)")

    # 端到端: 整个网络在 cvpe=on / off 下, 除了多一个零输出的模块以外应当一致。
    torch.manual_seed(7)
    net_off = UprMVSNet(_cfg(cvpe=False)).eval()
    torch.manual_seed(7)
    net_on = UprMVSNet(_cfg(cvpe=True)).eval()
    shared_off = {k: v for k, v in net_off.state_dict().items()}
    for k, v in net_on.state_dict().items():
        if k.startswith("cvpe."):
            continue
        assert k in shared_off and torch.equal(v, shared_off[k]), \
            f"参数 {k} 在 cvpe on/off 之间不同 —— CVPE 打乱了 RNG 流, 两者不再是配对实验"
    ok("cvpe=on 与 off 的**全部共享参数逐位相同** (CVPE 构造在最后, 没动 RNG 流)")

    batch = _batch(1, 3, 64, 80)
    with torch.no_grad():
        o_off = net_off(batch, step=0)
        o_on = net_on(batch, step=0)
    d = (o_off["depth_full"] - o_on["depth_full"]).abs().max()
    assert float(d) <= tol, f"cvpe=on 在 step 0 的 depth_full 与 off 差 {float(d):.3e} > {tol}"
    ok(f"端到端 step 0: cvpe=on 的 depth_full 与 off 一致 (max |Δ| = {float(d):.3e})")


# ======================================================================== 2, 3, 6
def run_cvpe() -> None:
    torch.manual_seed(0)
    B, V, C, h, w = 2, 4, 128, 12, 16

    # --- #2 同一个相机 => warp 必须是恒等 --------------------------------------
    # 这是投影方向、K 缩放约定、align_corners 三者一致性的**唯一**免争议判据:
    # 相机相同的时候, 无论深度平面取什么值, 采样点都必须落回原像素。
    feat = torch.randn(B, 8, h, w)
    K = torch.eye(3).view(1, 3, 3).repeat(B, 1, 1)
    K[:, 0, 0] = K[:, 1, 1] = 120.0
    K[:, 0, 2], K[:, 1, 2] = w * 8 / 2, h * 8 / 2
    E = torch.eye(4).view(1, 4, 4).repeat(B, 1, 1)
    planes = inverse_depth_planes(torch.full((B,), D_MIN), torch.full((B,), D_MAX), 8, (h, w))
    warped = homography_warp_features(feat, K, K, E, E, planes, feature_stride=8)
    for j in range(planes.shape[1]):
        # 边界一圈交给 grid_sample 的 padding, 只查内部。
        a = warped[:, :, j, 1:-1, 1:-1]
        b = feat[:, :, 1:-1, 1:-1]
        assert torch.allclose(a, b, atol=1e-4), \
            f"平面 {j}: 相同相机下 warp 不是恒等 (max {float((a - b).abs().max()):.3e})"
    ok("K_ref==K_src 且 E_ref==E_src 时, warp 逐平面恒等 (投影约定与 cost volume 同源)")

    # --- #3 / #6 形状与有限性 --------------------------------------------------
    cv = CrossViewPE(in_channels=C)
    # 非零 delta 才验证得了有限性, 所以给 out_proj 一个真实的权重。
    torch.nn.init.xavier_uniform_(cv.out_proj.weight)
    torch.nn.init.normal_(cv.out_proj.bias, std=0.1)
    cv.eval()
    p8 = torch.randn(B, V, C, h, w)
    Kb = K.view(B, 1, 3, 3).repeat(1, V, 1, 1).clone()
    Eb = E.view(B, 1, 4, 4).repeat(1, V, 1, 1).clone()
    Eb[:, 1:, 0, 3] = torch.tensor([40.0, -55.0, 90.0])[: V - 1]
    with torch.no_grad():
        delta = cv(p8=p8, K=Kb, E=Eb, depth_min=torch.full((B,), D_MIN),
                   depth_max=torch.full((B,), D_MAX), feature_stride=8)
    assert tuple(delta.shape) == (B, V - 1, C, h, w), delta.shape
    ok(f"CVPE 输出形状 {tuple(delta.shape)} == [B, V-1, C_in, h8, w8]")
    assert torch.isfinite(delta).all(), "CVPE 输出含 nan/inf"
    assert float(delta.abs().max()) > 0, "构造前提: out_proj 已随机化, delta 不应恒为 0"
    ok(f"CVPE 输出全部有限 (|delta| max {float(delta.abs().max()):.3e})")
    assert cv.last_stats is not None and math.isfinite(cv.last_stats["delta_rel"])
    ok(f"诊断可用: delta_rel={cv.last_stats['delta_rel']:.4f}")

    # V=1 (没有源视图) 不能崩
    with torch.no_grad():
        z = cv(p8=p8[:, :1], K=Kb[:, :1], E=Eb[:, :1],
               depth_min=torch.full((B,), D_MIN), depth_max=torch.full((B,), D_MAX),
               feature_stride=8)
    assert tuple(z.shape) == (B, 0, C, h, w)
    ok("V=1 时返回空 delta, 不抛异常")

    # 逆深度平面: 升序覆盖 [d_min, d_max], 且在逆深度上等距
    pl = inverse_depth_planes(torch.full((1,), D_MIN), torch.full((1,), D_MAX), 8, (2, 2))
    d = pl[0, :, 0, 0]
    assert abs(float(d[0]) - D_MAX) < 1e-3 and abs(float(d[-1]) - D_MIN) < 1e-3, d
    dv = torch.diff(1.0 / d)
    assert float(dv.max() - dv.min()) < 1e-9, "平面必须在逆深度上等距"
    ok(f"逆深度平面: {float(d[0]):.1f} -> {float(d[-1]):.1f} mm, 逆深度等距")

    # 相机向量: K 必须按 feature_stride 缩过, 否则描述的是另一个分辨率的相机
    c8 = img2world_params(K, E, 8)
    c1 = img2world_params(K, E, 1)
    assert c8.shape == (B, 16) and not torch.allclose(c8, c1)
    ok("img2world_params 按 feature_stride 缩放 K (与 warp 同一约定)")


# ============================================================================ 7
def run_grad() -> None:
    """两步梯度。

    为什么必须分两步: out_proj 零初始化时 ``delta = W2 @ y`` 对 y 的雅可比是
    ``W2 = 0``, 所以**第一步** in_proj / cam_encoder / 注意力层全部收不到梯度 ——
    这是零初始化的正常表现, 不是 bug。但如果 out_proj 自己也没梯度, 那就是接线
    断了。第二步 (out_proj 已经非零之后) 才轮到上游必须有梯度。

    只测 CVPE 模块本身: 端到端两步反向要跑整个网络, 而这里要定位的是 CVPE 的
    接线, 混进级联只会让失败原因变模糊。
    """
    torch.manual_seed(0)
    B, V, C, h, w = 1, 3, 128, 10, 12
    cv = CrossViewPE(in_channels=C)
    p8 = torch.randn(B, V, C, h, w)
    K = torch.eye(3).view(1, 1, 3, 3).repeat(B, V, 1, 1)
    K[:, :, 0, 0] = K[:, :, 1, 1] = 120.0
    K[:, :, 0, 2], K[:, :, 1, 2] = w * 8 / 2, h * 8 / 2
    E = torch.eye(4).view(1, 1, 4, 4).repeat(B, V, 1, 1)
    E[:, 1:, 0, 3] = torch.tensor([40.0, -55.0])[: V - 1]
    kw = dict(K=K, E=E, depth_min=torch.full((B,), D_MIN),
              depth_max=torch.full((B,), D_MAX), feature_stride=8)

    groups = {
        "out_proj": [cv.out_proj.weight, cv.out_proj.bias],
        "in_proj": list(cv.in_proj.parameters()),
        "cam_encoder": list(cv.cam_encode.parameters()),
        "attention": list(cv.layers.parameters()),
    }

    # 损失必须在 delta=0 处有非零梯度。``mean(delta^2)`` 不行 ——
    # d/ddelta = 2*delta/N, 在零初始化那一步恒为 0, 于是 out_proj 也拿不到梯度,
    # 测出来的是损失的性质而不是接线的性质。用一个非零靶: 在 delta=0 处
    # d/ddelta = -2*target/N != 0。
    target = torch.randn(B, V - 1, C, h, w)

    def step() -> dict[str, float]:
        cv.zero_grad(set_to_none=True)
        (cv(p8=p8, **kw) - target).square().mean().backward()
        out = {}
        for name, ps in groups.items():
            g = [p.grad for p in ps if p.grad is not None]
            n = float(torch.stack([x.norm() for x in g]).norm()) if g else 0.0
            assert math.isfinite(n), f"{name} 的梯度非有限"
            out[name] = n
        return out

    g1 = step()
    assert g1["out_proj"] > 0, "第一步 out_proj 就没梯度 —— CVPE 根本没接进计算图"
    ok(f"第一步: out_proj 梯度 {g1['out_proj']:.3e} > 0 (上游因零初始化为 0, 正常)")

    with torch.no_grad():                       # 手动跨出零初始化
        cv.out_proj.weight.normal_(std=0.05)
        cv.out_proj.bias.normal_(std=0.05)
    g2 = step()
    for name in ("in_proj", "cam_encoder", "attention", "out_proj"):
        assert g2[name] > 0, f"第二步 {name} 仍然没有梯度"
    ok("第二步: in_proj / cam_encoder / attention / out_proj 梯度均非零且有限 "
       + " ".join(f"{k}={v:.2e}" for k, v in g2.items()))


# ========================================================================= 8, 9
def run_compat() -> None:
    # --- #8 旧 checkpoint (无 calib_bias, 无 CVPE) 严格加载 -------------------
    torch.manual_seed(3)
    old = UprMVSNet(_cfg(cvpe=False, conf_head=True))
    sd = {k: v.clone() for k, v in old.state_dict().items()}
    dropped = [k for k in list(sd) if k.endswith("conf_head.calib_bias")]
    for k in dropped:
        sd.pop(k)
    assert dropped, "构造前提: conf_head 里应当有 calib_bias"
    torch.manual_seed(3)
    new = UprMVSNet(_cfg(cvpe=False, conf_head=True))
    new.load_state_dict(sd, strict=True)          # strict=True 必须仍然成功
    assert float(new.conf_head.calib_bias) == 0.0
    ok(f"缺 {dropped[0]} 的旧 state_dict 仍能 strict=True 加载, 自动补 0")

    # 旧 checkpoint 里没有任何 cvpe.* key; cvpe=off 时模型也不该有
    assert not any(k.startswith("cvpe.") for k in new.state_dict()), \
        "cvpe=off 时不应构造 CVPE 模块 (否则旧 ckpt 会缺 key)"
    ok("cvpe=off 时模型不含 cvpe.* 参数 —— 旧 checkpoint 不会缺 key")

    # --- #9 fingerprint 往返 --------------------------------------------------
    import train as trainmod
    import test as testmod
    from pathlib import Path
    cfg_on = _cfg(cvpe=True, conf_head=True)
    fp = trainmod._arch_fingerprint(cfg_on)
    for k in ("cvpe_enabled", "cvpe_d_model", "cvpe_num_planes", "cvpe_n_heads",
              "cvpe_cam_mid_channels", "cvpe_layer_pattern", "cvpe_feature_stride",
              "cvpe_plane_space", "cvpe_align_corners"):
        assert k in fp, f"fingerprint 缺 {k}"
    assert fp["cvpe_enabled"] is True
    # 从一个 **CVPE 关闭** 的 cfg 出发恢复, 才真的在测 "从 fingerprint 恢复"
    restored, _ = testmod._align_cfg_to_ckpt(_cfg(cvpe=False), old.state_dict(),
                                             "auto", fingerprint=fp)
    for field in ("enabled", "d_model", "num_planes", "n_heads",
                  "cam_mid_channels", "layer_pattern"):
        a, b = getattr(restored.cvpe, field), getattr(cfg_on.cvpe, field)
        assert a == b, f"cvpe.{field} 恢复后不同: {a!r} != {b!r}"
    ok("fingerprint 保存/恢复之后 CVPE 配置完全相同")

    fp_off = trainmod._arch_fingerprint(_cfg(cvpe=False))
    assert fp_off["cvpe_enabled"] is False
    legacy = {k: v for k, v in fp_off.items() if not k.startswith("cvpe_")}
    r2, _ = testmod._align_cfg_to_ckpt(_cfg(cvpe=True), old.state_dict(),
                                       "auto", fingerprint=legacy)
    assert r2.cvpe.enabled is False, "没有 cvpe_* 字段的旧 fingerprint 必须恢复成 off"
    ok("旧 fingerprint (无 cvpe_* 字段) 恢复成 cvpe=off")

    # --- load_model 必须把**对齐后**的 cfg 交出来 -----------------------------
    # job 415228: load_model 内部 `cfg, has_spre = _align_cfg_to_ckpt(...)` 只重新
    # 绑定了自己的局部名字, 调用方 (calibrate_conf) 手里的 cfg 仍是 profile 默认的
    # amp_dtype='fp16'。于是"推理跟着 checkpoint 的 dtype 走"这条修复对它完全没
    # 生效, 又白跑了一次 833 样本的全 nan。
    import tempfile
    with tempfile.TemporaryDirectory() as _d:
        _cf = _cfg(cvpe=False, conf_head=True)
        _net = UprMVSNet(_cf)
        _fp = dict(trainmod._arch_fingerprint(_cf))
        _fp["amp_dtype"] = "bf16"
        _ck = {"model": _net.state_dict(), "step": 30000, "fingerprint": _fp}
        _p = Path(_d) / "latest.pth"
        torch.save(_ck, _p)
        _ns = testmod.eval_namespace(ckpt=str(_p))
        _caller_cfg = replace(_cf, train=replace(_cf.train, amp_dtype="fp16"))
        _model, _ = testmod.load_model(_caller_cfg, _ns, torch.device("cpu"))
        _aligned = getattr(testmod.load_model, "last_cfg", None)
        assert _aligned is not None, "load_model 没有交出对齐后的 cfg"
        assert testmod.amp_dtype_of(_caller_cfg) is torch.float16
        assert testmod.amp_dtype_of(_aligned) is torch.bfloat16, \
            "load_model.last_cfg 的 amp_dtype 没跟着 checkpoint —— 这是 job 415228 的死因"
        # **这条才是真正管用的**: dtype 挂在 model 上, 调用方不可能忘了换。
        # 415273 就是因为 test.py 自己的 main() 仍把没对齐的 cfg 传给
        # run_inference —— 只断言 last_cfg 是抓不到那次的。
        assert getattr(_model, "inference_amp_dtype", None) is torch.bfloat16, \
            "load_model 没把 inference_amp_dtype 挂到 model 上 —— 这是 job 415273 的死因"
    ok("load_model: last_cfg 与 model.inference_amp_dtype 都跟着 checkpoint 解析成 bf16")


# ======================================================================== 10, 11
def run_fusion() -> None:
    # --- #10 固定保留率 --------------------------------------------------------
    # 复刻 test.fuse_scan 里的门, 逐字对齐 (那里在 GPU 张量上跑, 这里在 CPU)。
    torch.manual_seed(0)
    for n_valid, ratio in ((1000, 0.6), (7, 0.6), (1, 0.6), (999, 0.6), (100, 1.0)):
        N = 1500
        conf = torch.rand(N)
        valid = torch.zeros(N, dtype=torch.bool)
        valid[torch.randperm(N)[:n_valid]] = True
        photo = torch.zeros_like(valid)
        vidx = valid.nonzero(as_tuple=False).squeeze(1)
        k = int(math.ceil(ratio * float(vidx.numel())))
        k = max(1, min(k, int(vidx.numel())))
        order = torch.argsort(conf[vidx].float(), descending=True, stable=True)
        photo[vidx[order[:k]]] = True
        assert int(photo.sum()) == math.ceil(ratio * n_valid), \
            f"n_valid={n_valid}: 保留 {int(photo.sum())}, 应为 {math.ceil(ratio * n_valid)}"
        assert not bool((photo & ~valid).any()), "无效深度像素被保留了"
        # 保留的必须是置信度最高的那一批
        assert float(conf[photo].min()) >= float(conf[valid & ~photo].max()) \
            if bool((valid & ~photo).any()) else True
    ok("photo_keep_ratio: 每个视角精确保留 ceil(r x N_valid), 且只在有效深度里选")

    # 打平时也必须可复现 (stable 排序)
    conf = torch.zeros(64)
    valid = torch.ones(64, dtype=torch.bool)
    picks = []
    for _ in range(3):
        order = torch.argsort(conf.float(), descending=True, stable=True)
        picks.append(order[:38].tolist())
    assert picks[0] == picks[1] == picks[2], "置信度打平时排序不可复现"
    ok("置信度打平时 stable 排序给出可复现的选择")

    # --- #11 Platt 保持排序 -----------------------------------------------------
    torch.manual_seed(1)
    n = 20000
    z = torch.randn(n) * 2.0
    err = torch.rand(n) * 6.0 - 1.5 * z          # 让置信度与误差真的相关
    y = (err < 2.0).float()
    log_T, bias = fit_platt(z, y)
    T = math.exp(log_T)
    assert T > 0 and math.isfinite(bias)
    p0, p1 = torch.sigmoid(z), torch.sigmoid(z / T + bias)
    # 不能用 ``argsort(p0) == argsort(p1)``: sigmoid 在 fp32 下会把接近的 logit
    # 映射成**完全相等**的概率 (这里 20000 个里有十几个), 而 argsort 对并列的
    # 顺序不作保证 —— 那样连恒等映射都过不了这一条。正确的判据是成对单调:
    # 只要 p0 真的分得开, p1 的大小关系就必须一致。
    i = torch.randperm(n)[:4000]
    j = torch.randperm(n)[:4000]
    d0, d1 = p0[i] - p0[j], p1[i] - p1[j]
    sep = d0.abs() > 1e-7
    assert bool(sep.any())
    assert bool((torch.sign(d0[sep]) == torch.sign(d1[sep])).all()), "正斜率仿射改变了排序"
    ok(f"Platt (T={T:.4f}, b={bias:+.4f}) 成对单调: {int(sep.sum())} 对可分样本全部同向")
    _, _, a0 = risk_coverage(p0, err)
    _, _, a1 = risk_coverage(p1, err)
    r0 = risk_at_coverage(p0, err, 0.6)
    r1 = risk_at_coverage(p1, err, 0.6)
    assert abs(a0 - a1) < 1e-6, f"AURC 变了: {a0:.6f} -> {a1:.6f}"
    assert abs(r0 - r1) < 1e-6, f"risk@0.6 变了: {r0:.6f} -> {r1:.6f}"
    ok(f"AURC {a0:.4f} 与 risk@0.6 {r0:.4f} 在标定前后逐位不变")

    # 头的 calibrated() 与手算一致
    head = FusionConfidenceHead(feat_channels=128)
    with torch.no_grad():
        head.log_T.fill_(log_T)
        head.calib_bias.fill_(bias)
        got = head.calibrated(z)
    assert torch.allclose(got, p1, atol=1e-6)
    ok("FusionConfidenceHead.calibrated() == sigmoid(z/T + b)")


# ============================================================================ 12
def run_smoke(cuda: bool = False) -> None:
    torch.manual_seed(11)
    net = UprMVSNet(_cfg(cvpe=True, conf_head=True, geo_valid=True)).eval()
    batch = _batch(1, 3, 64, 80)
    with torch.no_grad():
        out = net(batch, step=0)
    assert torch.isfinite(out["depth_full"]).all()
    assert "fusion_conf" in out, "conf_head=on 时前向必须产出 fusion_conf"
    ok(f"CPU 前向 (cvpe+geo_valid+conf_head): depth_full {tuple(out['depth_full'].shape)}, 全部有限")

    # risk@0.6 的排序信号在训练与推理两侧是同一份实现
    from models.fusion_conf import cascade_confidence as cc_model
    from test import cascade_confidence as cc_test
    from train import cascade_confidence as cc_train
    assert cc_model is cc_test is cc_train, "cascade_confidence 有多份实现"
    conf = cc_model(out, window=1, mode="product")
    assert conf.shape == out["depth_full"].shape and torch.isfinite(conf).all()
    ok("cascade_confidence 在 train / test / models 三处是同一个对象")

    if not cuda:
        print("  [--] 跳过 CUDA smoke (加 --cuda 开启)")
        return
    if not torch.cuda.is_available():
        raise SystemExit("--cuda 但没有可用的 GPU")

    # ---- 回归守卫: **部署 token 数**下的三种 dtype ---------------------------
    # job 415038 (2026-08-26): 22 个 scan 全 nan、点云 0 点。根因是 test.py 的
    # autocast 没传 dtype (CUDA 上默认 fp16) 而模型是 bf16 训的, CVPE 的线性
    # 注意力对 S 个 token 求和, 0.8 整幅下 S=19200, einsum(q, k.sum) 约 1.5e5
    # **溢出 fp16** -> inf -> nan。
    # 之所以没被之前的 smoke 抓到: 那里是 64x80 -> p8 = 8x10 = 80 个 token,
    # 比部署少 240 倍。**这条必须用真实的部署尺寸**, 小张量在这里是测不出来的。
    _h, _w = (1200 * 8 // 10) // 8, (1600 * 8 // 10) // 8      # 0.8 整幅的 p8
    _cv = CrossViewPE(in_channels=128).to("cuda").eval()
    torch.nn.init.xavier_uniform_(_cv.out_proj.weight)          # 跨出零初始化
    _p8 = torch.randn(1, 3, 128, _h, _w, device="cuda")
    _K = torch.eye(3, device="cuda").view(1, 1, 3, 3).repeat(1, 3, 1, 1)
    _K[:, :, 0, 0] = _K[:, :, 1, 1] = 1200.0
    _K[:, :, 0, 2], _K[:, :, 1, 2] = _w * 8 / 2, _h * 8 / 2
    _E = torch.eye(4, device="cuda").view(1, 1, 4, 4).repeat(1, 3, 1, 1)
    _E[:, 1:, 0, 3] = torch.tensor([40.0, -55.0], device="cuda")
    _kw = dict(K=_K, E=_E, depth_min=torch.full((1,), D_MIN, device="cuda"),
               depth_max=torch.full((1,), D_MAX, device="cuda"), feature_stride=8)
    _ref = None
    for _nm, _dt in (("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        with torch.no_grad():
            if _dt is None:
                _d = _cv(p8=_p8, **_kw)
            else:
                with torch.autocast("cuda", dtype=_dt):
                    _d = _cv(p8=_p8, **_kw)
        _d = _d.float()
        assert torch.isfinite(_d).all(), \
            f"CVPE 在部署 token 数 ({_h}x{_w}={_h * _w}) 下 {_nm} 出现非有限值"
        if _ref is None:
            _ref = _d
        else:
            assert float((_d - _ref).abs().max() / _ref.abs().max()) < 0.05, f"{_nm} 与 fp32 偏差过大"
    ok(f"CVPE 在部署 token 数 {_h}x{_w}={_h * _w} 下 fp32/bf16/fp16 全部有限且互相一致")
    del _cv, _p8, _d, _ref
    torch.cuda.empty_cache()

    # ---- 推理侧的 autocast dtype 必须跟 checkpoint 走 -----------------------
    import test as _tm
    from dataclasses import replace as _rep
    for _tag, _want in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        _c = _rep(_cfg(), train=_rep(_cfg().train, amp_dtype=_tag))
        assert _tm.amp_dtype_of(_c) is _want, f"amp_dtype_of({_tag}) != {_want}"
    _fp = {"amp_dtype": "bf16"}
    _restored, _ = _tm._align_cfg_to_ckpt(_rep(_cfg(), train=_rep(_cfg().train, amp_dtype="fp16")),
                                          UprMVSNet(_cfg()).state_dict(), "auto",
                                          fingerprint={**_arch_fp_min(), **_fp})
    assert _tm.amp_dtype_of(_restored) is torch.bfloat16, \
        "fingerprint 里的 amp_dtype=bf16 没有恢复到推理侧 —— 这正是 job 415038 的根因"
    ok("推理的 autocast dtype 从 checkpoint fingerprint 恢复 (bf16), 不再用 CUDA 默认的 fp16")
    dev = torch.device("cuda")
    net = net.to(dev).train()
    batch = _batch(1, 3, 64, 80, device=dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = net(batch, step=0)
        loss = out["depth_full"].float().square().mean()
    loss.backward()
    gs = [p.grad for p in net.cvpe.parameters() if p.grad is not None]
    assert gs and all(torch.isfinite(g).all() for g in gs)
    ok(f"CUDA BF16 前向+反向通过; CVPE 收到 {len(gs)} 个有限梯度; "
       f"峰值显存 {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--equivalence", action="store_true")
    ap.add_argument("--cvpe", action="store_true")
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--compat", action="store_true")
    ap.add_argument("--fusion", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cuda", action="store_true", help="额外跑一次 BF16 CUDA smoke")
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()
    sel = dict(equivalence=a.equivalence, cvpe=a.cvpe, grad=a.grad,
               compat=a.compat, fusion=a.fusion, smoke=a.smoke)
    if a.all or not any(sel.values()):
        sel = {k: True for k in sel}

    torch.use_deterministic_algorithms(False)
    if sel["cvpe"]:
        print("[cvpe] 模块级: 投影约定 / 形状 / 有限性")
        run_cvpe()
    if sel["equivalence"]:
        print("[equivalence] cvpe=off 逐位一致 + 残差只加一次")
        run_equivalence(a.tol)
    if sel["grad"]:
        print("[grad] 两步梯度")
        run_grad()
    if sel["compat"]:
        print("[compat] checkpoint 与 fingerprint 兼容性")
        run_compat()
    if sel["fusion"]:
        print("[fusion] 固定保留率 + Platt 标定")
        run_fusion()
    if sel["smoke"]:
        print("[smoke] 端到端前向")
        run_smoke(a.cuda)
    print(f"\n[verify_vnext] 全部通过 ({len(_OK)} 条检查)")


if __name__ == "__main__":
    main()
