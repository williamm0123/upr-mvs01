from __future__ import annotations

"""Cross-View Position Encoding (CVPE) —— MonoMVSNet (ICCV 2025) 的跨视位置编码,
移植到 UPRMVS 的 FPN p8 注入点。

**为什么放在 p8, 而不是 cost volume 之前**
    ``models/fpn.py`` 的 top-down 是 p8 -> p4 -> p2 -> p1, DINO 已经注入在 p8
    (top-down **之前**, 见那里的 docstring), 所以改 p8 会沿现有的 lateral /
    interpolate / smooth 通路自动传遍四级。MonoMVSNet 之所以要另写一条
    dim_reduction + upsample 的平行链 (``FMT_with_pathway``), 是因为它的 FPN4
    是外部模块、拿不到 top-down 的中间态; 这里没有这个约束, 也就不该多一条
    会随训练漂移的平行链。

**为什么只改 source, ref 不动**
    cost volume 是 ref 与 warp 后 src 的相关。只增强一侧, "增强" 才表现为相关
    峰的锐化; 两侧同时改会把增益部分抵消。而且 ref 是 SPRE / prior 的宿主,
    改它会串到别的模块。MonoMVSNet 同样只写回 src (见其 ``FMT_with_pathway``
    里的 ``# attention only for src_features``)。

**为什么是线性注意力而不是 softmax**
    p8 在 640x896 下是 80x112 = 8960 个 token。普通注意力的 N^2 矩阵是
    8960^2 x heads x B, 显存不可接受。ELU(x)+1 的线性注意力把复杂度降到
    O(N d^2), 与 MonoMVSNet 一致 (它们的 ``LinearAttention``)。

**梯度与数值边界**
    * 逆深度平面与 warp 网格全程 fp32 且 no_grad —— 直接复用
      ``utils.geometry.homography_warp_features``, **不另写一份投影**。两处各写
      一份迟早分叉, 而且分叉时两边都不报错。
    * 特征采样与注意力走训练 dtype (bf16); LayerNorm / 归约由 autocast 处理。
    * ``out_proj`` 零初始化 ⇒ 训练第 0 步 delta 恒为 0, 与 CVPE 关闭**逐位**
      一致 (x + 0 在 IEEE 下精确等于 x)。这条性质比 golden test 更强, 也让
      "开着但还没学到东西" 与 "关着" 在数值上无法区分。
    * 本模块**只返回 delta**, 由调用方做且只做一次残差相加。模块内部不再自加
      一次 —— 那会让 source 被加两遍。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry import homography_warp_features


# --------------------------------------------------------------------------- #
# 小组件 (与 MonoMVSNet 的 Mlp / SELayer / LinearAttention / EncoderLayer 对齐)
# --------------------------------------------------------------------------- #
class _Mlp(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _SELayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1)
        self.act = nn.ReLU()
        self.conv_expand = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, x_se: torch.Tensor) -> torch.Tensor:
        x_se = self.conv_expand(self.act(self.conv_reduce(x_se)))
        return x * torch.sigmoid(x_se)


class CamParamEncoder(nn.Module):
    """把 [B, d*D, H, W] 的 warp 特征压成 [B, d, H, W] 的跨视位置编码。

    相机参数 (img2world 的 16 个数) 经 MLP 变成通道级的 SE 门 —— 这就是
    MonoMVSNet 说的 "把 camera-parameter-related spatial information 注入投影
    特征, 并沿深度维压缩"。深度维是靠 ``reduce_conv`` 的 in_channels = d*D
    一次性吃掉的, 不是 pooling。
    """

    CAM_LEN = 16

    def __init__(self, d_model: int, num_planes: int, mid_channels: int) -> None:
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(d_model * num_planes, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # **偏离 MonoMVSNet 的一处**: 它这里是 nn.BatchNorm1d(16)
        # (FMT_with_CVPE.py:143), 换成 LayerNorm。三个理由, 每条都是实打实的故障:
        #   1. BN1d 在 B=1 的训练前向下直接抛
        #      "Expected more than 1 value per channel" —— scripts/fit_batch.sh
        #      的显存扫描从 batch 1 起步, smoke 也是 batch 1, 一上来就崩。
        #   2. 它把同一个 batch 里不同样本的相机耦合在一起: 训练用 batch 统计、
        #      推理用 running 统计, 而推理恒为 B=1。同一对相机在训练和测试下拿到
        #      不同的编码, 且**不报错**。
        #   3. 每个 source 要调两次 (cam_src 与 cam_ref), 两种分布会被写进同一份
        #      running stats; DDP 下各 rank 的 running stats 也不同步。
        # LayerNorm 是逐样本的, 上面三条全不存在; 而 16 个分量之间的量级差异
        # (焦距 ~1e3 / 旋转 ~1 / 平移 ~1e2) 由紧随其后的 16->mid 全连接逐分量
        # 重新定标 —— 那一层本来就有逐输入分量的权重。
        self.bn = nn.LayerNorm(self.CAM_LEN)
        self.context_mlp = _Mlp(self.CAM_LEN, mid_channels, mid_channels)
        self.context_se = _SELayer(mid_channels)
        self.context_conv = nn.Conv2d(mid_channels, d_model, 1)

    def forward(self, feat: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
        # cam 的量级跨好几个数量级 (焦距 ~1e3, 旋转 ~1, 平移 ~1e2), BatchNorm1d
        # 是 MonoMVSNet 的做法, 保留 —— 但必须 fp32, bf16 下这个归一化会把
        # 平移项的有效位数吃掉。
        with torch.autocast(device_type=feat.device.type, enabled=False):
            se = self.context_mlp(self.bn(cam.float().view(cam.shape[0], -1)))
        se = se.to(feat.dtype)[..., None, None]
        x = self.reduce_conv(feat)
        return self.context_conv(self.context_se(x, se))


class _LinearAttention(nn.Module):
    """ELU(x)+1 的线性注意力。**不显式构造注意力矩阵**, 所以 8960 个 token 也放得下。"""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum("nshd,nshm->nhmd", k, v)
        z = 1.0 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps)
        return torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, z).contiguous()


class _AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        dk = d_model // n_heads
        self.q_proj = nn.Linear(d_model, dk * n_heads)
        self.k_proj = nn.Linear(d_model, dk * n_heads)
        self.v_proj = nn.Linear(d_model, dk * n_heads)
        self.out_proj = nn.Linear(dk * n_heads, d_model)
        self.n_heads = n_heads
        self.inner = _LinearAttention()

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        N, L, _ = q.shape
        S = kv.shape[1]
        H = self.n_heads
        qq = self.q_proj(q).view(N, L, H, -1)
        kk = self.k_proj(kv).view(N, S, H, -1)
        vv = self.v_proj(kv).view(N, S, H, -1)
        return self.out_proj(self.inner(qq, kk, vv).view(N, L, -1))


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attention = _AttentionLayer(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, 2 * d_model)
        self.linear2 = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x, source)
        y = x = self.norm1(x)
        y = self.linear2(F.relu(self.linear1(y)))
        return self.norm2(x + y)


class _PositionEncodingSine(nn.Module):
    """标准 2D 正弦位置编码。它只提供**图像内**的位置; 跨视图的几何由 CVPE 提供。"""

    def __init__(self, d_model: int, max_shape: tuple[int, int] = (320, 320)) -> None:
        super().__init__()
        pe = torch.zeros((d_model, *max_shape), dtype=torch.float32)
        y_pos = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_pos = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div = torch.exp(torch.arange(0, d_model // 2, 2).float()
                        * (-math.log(10000.0) / (d_model // 2)))
        div = div[:, None, None]
        pe[0::4] = torch.sin(x_pos * div)
        pe[1::4] = torch.cos(x_pos * div)
        pe[2::4] = torch.sin(y_pos * div)
        pe[3::4] = torch.cos(y_pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, : x.shape[-2], : x.shape[-1]].to(x.dtype)


# --------------------------------------------------------------------------- #
def inverse_depth_planes(depth_min: torch.Tensor, depth_max: torch.Tensor,
                         num_planes: int, hw: tuple[int, int]) -> torch.Tensor:
    """逆深度均匀的 ``num_planes`` 个全局平面, 广播成 [B, D, H, W]。

        v_j = 1/d_max + j/(D-1) * (1/d_min - 1/d_max),   d_j = 1/v_j

    这些平面**只服务于位置编码**, 与主 cost volume 的 48/16/8/4 无关, 也不随
    级联变化 —— CVPE 要的是 "两个视图之间的几何关系", 不是深度假设。
    """
    B = depth_min.shape[0]
    v_min = 1.0 / depth_max.float().clamp_min(1e-6)          # 远端 -> 小 v
    v_max = 1.0 / depth_min.float().clamp_min(1e-6)
    t = torch.arange(num_planes, device=depth_min.device, dtype=torch.float32)
    t = t / max(num_planes - 1, 1)
    v = v_min.view(B, 1) + t.view(1, -1) * (v_max - v_min).view(B, 1)
    d = (1.0 / v.clamp_min(1e-12))                            # [B, D]
    return d.view(B, num_planes, 1, 1).expand(B, num_planes, hw[0], hw[1]).contiguous()


def img2world_params(K: torch.Tensor, E: torch.Tensor, feature_stride: int) -> torch.Tensor:
    """``inv(E) @ inv(K_4x4)`` 展平成 16 维, 作为 CamParamEncoder 的条件输入。

    K 先按 ``feature_stride`` 缩到特征分辨率 —— 与 ``homography_warp_features``
    里的缩放**同一个约定**, 否则相机嵌入描述的是另一个分辨率的相机。
    """
    B = K.shape[0]
    Ks = K.float().clone()
    Ks[:, 0, :] = Ks[:, 0, :] / feature_stride
    Ks[:, 1, :] = Ks[:, 1, :] / feature_stride
    K4 = torch.eye(4, device=K.device, dtype=torch.float32).view(1, 4, 4).repeat(B, 1, 1)
    K4[:, :3, :3] = Ks
    return (torch.inverse(E.float()) @ torch.inverse(K4)).reshape(B, 16)


class CrossViewPE(nn.Module):
    """CVPE: 用相机与固定逆深度平面把对侧特征 warp 过来做位置编码, 再跨视注意力。

    ``forward`` 返回**只给 source 的 delta**, 形状 [B, V-1, C_in, h, w]。
    调用方负责且只负责一次 ``src = src + delta``。
    """

    def __init__(
        self,
        in_channels: int = 128,
        d_model: int = 64,
        num_planes: int = 8,
        n_heads: int = 8,
        layer_pattern: tuple[str, ...] = ("self", "cross") * 4,
        cam_mid_channels: int = 64,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} 必须被 n_heads {n_heads} 整除")
        for nm in layer_pattern:
            if nm not in ("self", "cross"):
                raise ValueError(f"layer_pattern 只能含 'self'/'cross', 收到 {nm!r}")
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.num_planes = int(num_planes)
        self.n_heads = int(n_heads)
        self.layer_pattern = tuple(layer_pattern)

        self.in_proj = nn.Conv2d(in_channels, d_model, 1, bias=False)
        self.pos_encoding = _PositionEncodingSine(d_model)
        self.cam_encode = CamParamEncoder(d_model, num_planes, cam_mid_channels)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads) for _ in self.layer_pattern]
        )
        self.out_proj = nn.Conv2d(d_model, in_channels, 1)
        # 零初始化: step 0 的 delta 恒为 0 ⇒ 与 CVPE 关闭逐位一致。
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        for p in self.in_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.layers:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        self.last_stats: dict[str, float] | None = None

    def forward(
        self,
        p8: torch.Tensor,                  # [B, V, C_in, h, w]
        K: torch.Tensor,                   # [B, V, 3, 3]  (图像分辨率的内参)
        E: torch.Tensor,                   # [B, V, 4, 4]
        depth_min: torch.Tensor,           # [B]
        depth_max: torch.Tensor,           # [B]
        feature_stride: int,
    ) -> torch.Tensor:
        B, V, C, h, w = p8.shape
        if V < 2:
            return p8.new_zeros(B, 0, C, h, w)

        x = self.in_proj(p8.reshape(B * V, C, h, w)).view(B, V, self.d_model, h, w)
        ref = x[:, 0]                                            # [B, d, h, w]
        planes = inverse_depth_planes(depth_min, depth_max, self.num_planes, (h, w))
        with torch.no_grad():
            cam_ref = img2world_params(K[:, 0], E[:, 0], feature_stride)

        pe_ref_in = self.pos_encoding(ref)
        deltas = []
        for s in range(1, V):
            src = x[:, s]
            with torch.no_grad():
                cam_src = img2world_params(K[:, s], E[:, s], feature_stride)

            # ref -> src 帧: 在 source 的像素网格上采 reference 特征。
            # 复用同一个 homography_warp_features, 只是把 ref/src 的角色对调。
            w_ref = homography_warp_features(
                ref, K[:, s], K[:, 0], E[:, s], E[:, 0], planes, feature_stride)
            # src -> ref 帧: 与 cost volume 走的是同一个方向、同一个函数。
            w_src = homography_warp_features(
                src, K[:, 0], K[:, s], E[:, 0], E[:, s], planes, feature_stride)

            pe_src = self.cam_encode(w_ref.reshape(B, self.d_model * self.num_planes, h, w), cam_src)
            pe_r = self.cam_encode(w_src.reshape(B, self.d_model * self.num_planes, h, w), cam_ref)
            del w_ref, w_src

            q = (self.pos_encoding(src) + pe_src).flatten(2).transpose(1, 2).contiguous()
            kv = (pe_ref_in + pe_r).flatten(2).transpose(1, 2).contiguous()
            for layer, name in zip(self.layers, self.layer_pattern):
                q = layer(q, q if name == "self" else kv)
            y = q.transpose(1, 2).reshape(B, self.d_model, h, w)
            deltas.append(self.out_proj(y))

        delta = torch.stack(deltas, dim=1)                       # [B, V-1, C_in, h, w]
        with torch.no_grad():
            # 相对幅度 —— 判断 CVPE 到底改了多少。恒为 0 说明它没学到东西,
            # 远大于 1 说明它在覆盖而不是修正 p8。
            den = p8[:, 1:].detach().float().abs().mean().clamp_min(1e-8)
            self.last_stats = {
                "delta_rel": float(delta.detach().float().abs().mean() / den),
                "delta_abs": float(delta.detach().float().abs().mean()),
            }
        return delta
