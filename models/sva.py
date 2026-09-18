"""MVSFormer++ 完整 SVA 的后半段: deconv 上采样 + 1/8 尺度的 normalized 2D-PE
与 self/cross 注意力。前半段 (DINOv3 1/16 token 上的 SVA) 在 ``models/spre.py``
的 ``SVAFusion``。

对应 MVSFormer++ 的 Fig. 2(a):

    DINO 多层 token --SVA(self: ref / cross: src)--> proj + 2x deconv (x4 上采样)
        --> 与 FPN 的 1/8 特征相加 --> + normalized 2D-PE
        --> SVA(self, cross, self, cross) --> 沿 FPN top-down 传到 1/4、1/2、1/1

代码对应关系 (reference/MVSFormerPlusPlus):
  * ``SVADeconv``      <- ``CrossVITDecoder.proj / upsampler0 / upsampler1``
  * ``NormalizedPE2D`` <- ``position_encoding.PositionEncodingSineNorm``
  * ``SVABlock``       <- ``dino/layers/block.CrossBlock`` (pre-norm, LayerScale,
                          ``pre_norm_query=False``: key/value 与 query 共用 norm1)
  * ``_linear_attn``   <- ``dino/layers/attention.CrossLinearAttention``
  * ``HighResSVA``     <- ``FMT.FMT`` (``FMT_config.layer_names = self,cross,self,cross``)

论文的原话是 "after the upsampling to the 1/8 scale, we further incorporate two
additional SVA blocks to high-resolution features with normalized 2D-PE", 以及
"SVA performs cross-view learning for both DINOv2 (1/32) and coarse MVS (1/8)
features" —— 第二段注意力作用在**融合了 DINO 的 FPN 1/8 特征**上, 不是只作用
在 DINO 分支上。代码 (``DINOv2MVSNet.forward``) 也是先 ``conv31 + vit_feat``、
过 FPN decoder, 再进 ``FMT_with_pathway``。这里的 FPN 把 DINO 注入在 p8 上、
top-down 之前, 所以 ``HighResSVA`` 挂在同一个注入点上, 它的输出由现有的
lateral/smooth 传遍四级 —— 与 ``FMT_with_pathway`` 的 upsample-add-smooth
是同一件事, 不需要再建一条平行链。

与 MVSFormer++ 的差别 (都是适配, 不是改设计):
  * 宽度: 它们的 1/8 特征是 64 通道 (``feat_chs[3]``), 这里 FPN 是 128 通道, 所以
    d_model=128; 头数沿用它们的 nhead=4。
  * 线性注意力的 key/value 长度可以与 query 不同 (它们的实现里 k 按 query 的 N
    reshape, 只在等长时成立)。于是 cross 把 V-1 个 source 视角**拼在 token 维上**
    一次查 reference —— 注意力对每个 query 独立, 这与逐视角循环逐位同义。
  * 注意力的数值部分固定 fp32 并关掉 autocast。线性注意力要对全部 key 求和
    (0.8 整幅推理时 p8 有 120x160 = 19200 个 token), fp16 会溢出 (job 415038
    就是 CVPE 的同一个问题: 22 个 scan 全 nan)。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SVADeconv(nn.Module):
    """SVA 输出 (patch 网格) -> 1/8 尺度的 FPN 宽度特征。

    ``CrossVITDecoder`` 的 proj(3x3) + 两层 ConvTranspose2d(4, s2, p1), 每层
    BN + SiLU。x4 上采样: 它们的 1/32 token 网格 -> 1/8; 这里 DINOv3 patch16 +
    max_side 512 下 token 网格约为图像的 1/28 (训练最大尺度 640x896), x4 后约 1/7,
    最后由调用方双线性对齐到 FPN 的 p8 尺寸 —— 与它们 ``F.interpolate(vit_feat,
    size=conv31)`` 同一处理。
    """

    def __init__(self, in_dim: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_dim, out_ch * 4, 3, padding=1),
                                  nn.BatchNorm2d(out_ch * 4), nn.SiLU())
        self.up0 = nn.Sequential(nn.ConvTranspose2d(out_ch * 4, out_ch * 2, 4, stride=2, padding=1),
                                 nn.BatchNorm2d(out_ch * 2), nn.SiLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(out_ch * 2, out_ch, 4, stride=2, padding=1),
                                 nn.BatchNorm2d(out_ch), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up1(self.up0(self.proj(x)))


class NormalizedPE2D(nn.Module):
    """``PositionEncodingSineNorm``: 正弦 2D 位置编码, 坐标先线性归一化到
    ``max_shape`` —— 无论特征图多大, 行/列坐标的最大值都是 128。

    这是它们高分辨率泛化的关键 (Tab. 7/11): 训练在 1/8 的 64x80 上, 推理在
    120x160 上, 不归一化的话推理时一半的位置都是训练没见过的频率相位。

    编码是确定性的, 不是参数, 按 (h, w, device, dtype) 缓存, 不进 state_dict。
    """

    def __init__(self, d_model: int, max_shape: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"NormalizedPE2D 需要 d_model 能被 4 整除, 收到 {d_model}")
        self.d_model = int(d_model)
        self.max_shape = (float(max_shape[0]), float(max_shape[1]))
        self._cache: dict = {}

    def encoding(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w, str(device), dtype)
        pe = self._cache.get(key)
        if pe is None:
            c = self.d_model
            # ones.cumsum(0) = 1..H —— 与原实现一样从 1 开始
            y = torch.arange(1, h + 1, dtype=torch.float32).view(h, 1).expand(h, w) * (self.max_shape[0] / h)
            x = torch.arange(1, w + 1, dtype=torch.float32).view(1, w).expand(h, w) * (self.max_shape[1] / w)
            div = torch.exp(torch.arange(0, c // 2, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / (c // 2))).view(-1, 1, 1)
            pe = torch.zeros(c, h, w, dtype=torch.float32)
            pe[0::4] = torch.sin(x.unsqueeze(0) * div)
            pe[1::4] = torch.cos(x.unsqueeze(0) * div)
            pe[2::4] = torch.sin(y.unsqueeze(0) * div)
            pe[3::4] = torch.cos(y.unsqueeze(0) * div)
            pe = pe.unsqueeze(0).to(device=device, dtype=dtype)
            self._cache[key] = pe
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, C, H, W]"""
        return x + self.encoding(x.shape[-2], x.shape[-1], x.device, x.dtype)


def _linear_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int,
                 eps: float = 1e-6) -> torch.Tensor:
    """``CrossLinearAttention`` 的数值部分: phi(x) = elu(x) + 1。

    q: [B, N, C], k/v: [B, M, C] (M 可以 != N)。fp32 且关掉 autocast。
    """
    B, N, C = q.shape
    M = k.shape[1]
    d = C // heads
    with torch.autocast(device_type=q.device.type, enabled=False):
        q = F.elu(q.float().view(B, N, heads, d)) + 1.0
        k = F.elu(k.float().view(B, M, heads, d)) + 1.0
        v = v.float().view(B, M, heads, d)
        kv = torch.einsum("bmhd,bmhe->bhde", k, v)                      # [B,h,d,d]
        z = 1.0 / (torch.einsum("bnhd,bhd->bnh", q, k.sum(dim=1)) + eps)
        out = torch.einsum("bnhd,bhde,bnh->bnhe", q, kv, z)
    return out.reshape(B, N, C)


class SVABlock(nn.Module):
    """``CrossBlock`` (pre-norm + LayerScale init 1.0, qkv 无 bias)。

    self-attention 时 ``kv=None``; cross 时 key/value 也过 ``norm1`` ——
    FMT 配置 ``pre_norm_query=False`` 的行为。
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 init_values: float = 1.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} 不能被 heads={heads} 整除")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        xn = self.norm1(x)
        kvn = xn if kv is None else self.norm1(kv)
        a = _linear_attn(self.q_proj(xn), self.k_proj(kvn), self.v_proj(kvn), self.heads)
        x = x + self.ls1 * self.proj(a.to(xn.dtype))
        return x + self.ls2 * self.mlp(self.norm2(x))


class HighResSVA(nn.Module):
    """1/8 尺度上的第二段 SVA —— ``FMT`` 的移植。

    layer_names = (self, cross, self, cross), 各层权重在 reference 与 source 之间
    **共享** (FMT 里只有一个 ``self.layers``):

        reference:  PE -> self_0 -> r1 -> self_2 -> r2           (只走 self 层)
        source:     PE -> self_0 -> cross_1(ref=r1) -> self_2 -> cross_3(ref=r2)

    输出 view 0 = r2 (最后一个 reference 状态), 其余 = 各 source 的最终状态。
    与原实现一样, PE 进入残差流, 所以输出特征里含有 PE —— 这是它们的设计。
    """

    def __init__(self, dim: int, heads: int = 4, layer_names: tuple[str, ...] = ("self", "cross", "self", "cross"),
                 mlp_ratio: float = 4.0, pe_max_shape: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        names = tuple(str(n).strip() for n in layer_names)
        # FMT 的 ref_idx = i // 2 只对 (self, cross) 交替、以 self 开头的排布成立。
        if not names or len(names) % 2 or names != ("self", "cross") * (len(names) // 2):
            raise ValueError(f"layer_names 必须是 self,cross 交替 (如 self,cross,self,cross), 收到 {names}")
        self.layer_names = names
        self.layers = nn.ModuleList(SVABlock(dim, heads, mlp_ratio) for _ in names)
        self.pe = NormalizedPE2D(dim, pe_max_shape)
        self._reset_parameters()
        self.last_stats: dict | None = None

    def _reset_parameters(self) -> None:
        # FMT._reset_parameters: 所有 >1 维的参数 xavier_uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, V, C, h, w] (view 0 = reference) -> 同形状。"""
        B, V, C, h, w = feat.shape
        x = self.pe(feat.reshape(B * V, C, h, w)).view(B, V, C, h, w)
        tok = x.flatten(3).transpose(2, 3).contiguous()       # [B, V, N, C]
        N = h * w

        ref = tok[:, 0]
        ref_states = []
        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                ref = layer(ref)
                ref_states.append(ref)

        out = [ref]
        if V > 1:
            src = tok[:, 1:].reshape(B * (V - 1), N, C)
            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == "self":
                    src = layer(src)                          # 每个 source 视角内部
                else:
                    r = ref_states[i // 2]
                    # 注意力对每个 query 独立: V-1 个视角拼在 token 维上查同一个 ref
                    # 与逐视角循环逐位同义, 但 ref 的 k/v 只投影一次。
                    q = src.reshape(B, (V - 1) * N, C)
                    src = layer(q, kv=r).reshape(B * (V - 1), N, C)
            out.append(src.reshape(B, V - 1, N, C))
            y = torch.cat([out[0].unsqueeze(1), out[1]], dim=1)
        else:
            y = out[0].unsqueeze(1)

        with torch.no_grad():
            # 纯诊断: 注意力块相对 (特征 + PE) 的改动量。恒 ~0 = 什么都没学到;
            # >>1 = 在覆盖 FPN 特征而不是修正它。
            d = y.float() - tok.float()
            self.last_stats = {
                "delta_rel": float(d.norm() / tok.float().norm().clamp_min(1e-6)),
                "ls1_mean": float(torch.stack([l.ls1.abs().mean() for l in self.layers]).mean()),
            }
        return y.transpose(2, 3).reshape(B, V, C, h, w)
