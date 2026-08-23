from __future__ import annotations

"""可学习的级联范围控制器 (W1-B / W1-C)。

替代 ``depth_range.refine_range_from_posterior`` 里那套
``half = max(range_k * winner_interval * (1 + a*H + b*E), range_min_gi * gi)``。

三点设计约束, 每一条都对应一个实测出来的问题:

1. ``max`` 是不可微的悬崖, 实测把 78–91% 的像素钉在常数上, 自适应项整段失效。
   这里换成 log 域的 p-范数软最大 (p=16): A=B 时只比严格 max 高 4.4%,
   两项相差一个量级时几乎等于 max, 而两个尺度系数都可学。

2. 中心偏移必须卡在 ±1 个上一级 bin (``tanh`` 的界)。它只负责修 "量化偏差";
   跨表面是离散模态选择的职责 (stage4 MAP / stage2 双模态), 两件事不能混进
   同一个连续量, 否则就退化成"回归到两个表面之间"。

3. 初始化必须能复现旧行为, 所以:
     * ``softplus`` 的参数取逆 softplus 初始化 (softplus(1.5) != 1.5);
     * ``(1 + a*H + b*E)`` 这一项必须放回 A_k —— 零初始化控制器补不回它;
     * 深度均匀轴与逆深度均匀轴的中间候选本来就不同, 靠 network.py 里的
       legacy<->inverse 凸组合迁移解决, 不在这里解决。

梯度边界: 本模块的输入全部 detach, 输出的候选轴对 matching 路径也 detach,
唯一的梯度来自 losses/composite.py 里的 pinball + center 损失。原因见那里的
注释 —— 各级 CE 的掩码是 ``valid & in_range``, 若窗口位置对 CE 可导, "缩窗把
难像素排除出监督" 就能直接降 loss, 这是一个退化激励。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(x: float) -> float:
    """softplus 的逆: 给定目标值返回原始参数。

    稳定写法 ``x + log1p(-exp(-x))``; 直接写 ``log(exp(x) - 1)`` 在 x 大时溢出。
    """
    x = float(x)
    if x <= 0:
        raise ValueError(f"softplus 的输出恒为正, 目标值 {x} 不可达")
    return x + math.log1p(-math.exp(-x))


def _pnorm_softmax(log_a: torch.Tensor, log_b: torch.Tensor, p: float) -> torch.Tensor:
    """log 域的 p-范数软最大: log[(A^p + B^p)^(1/p)]。

    **必须在 log 域算。** 逆深度量纲下 A ~ 1e-5, 直接算 A**16 ~ 1e-80 在 fp32
    里下溢成 0, 软最大就退化成 max(0, B^16)^(1/16) = B —— 而且是静默的。
    """
    stacked = torch.stack([p * log_a, p * log_b], dim=0)
    return torch.logsumexp(stacked, dim=0) / p


class SpreGates(nn.Module):
    """SPRE 可靠度在四级之间的单调衰减门 (W1-F)。

    ``gamma_1 = 1`` 固定, ``gamma_k = gamma_{k-1} * sigmoid(a_k)`` ——
    累乘形式同时做到两件事:
      * 避开 ``logit(1.0) = inf``;
      * **结构性**保证 1 = g1 >= g2 >= g3 >= g4 > 0, 独立 sigmoid 做不到。

    注意 gamma 本身 **不 detach**, 它要从 range loss 和匹配 loss 收梯度;
    detach 的是 r 和 e。另外 gamma 的数值 **不能**当成 "SPRE 还有没有用" 的
    消融读数 —— gamma*r*W = (c*gamma)*r*(W/c), 下游卷积可以补偿门的大小。
    要判断依赖度用 scripts/gamma_intervention.py 做 gamma<-0 的推理干预。
    """

    def __init__(self, init=(1.0, 0.60, 0.35, 0.20)) -> None:
        super().__init__()
        if len(init) != 4 or abs(init[0] - 1.0) > 1e-6:
            raise ValueError("spre_gate_init 必须是 4 个值且第一个为 1.0")
        ratios = []
        for k in range(1, 4):
            r = init[k] / max(init[k - 1], 1e-8)
            if not (0.0 < r < 1.0):
                raise ValueError(f"spre_gate_init 必须严格递减且为正: {init}")
            ratios.append(math.log(r / (1.0 - r)))          # logit
        self.a = nn.Parameter(torch.tensor(ratios, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """返回 [4] 的 gamma, 保持 1 = g1 >= g2 >= g3 >= g4 > 0。"""
        s = torch.sigmoid(self.a)
        g = torch.cumprod(s, dim=0)
        return torch.cat([s.new_ones(1), g], dim=0)


class RangeController(nn.Module):
    """三个 transition 共享的范围控制器 (stage k -> k+1, k = 1,2,3)。

    共享主干 + per-stage FiLM: 三个独立控制器在这个数据量下容易过拟合, 而三段
    转移的物理机制是同一个。

    forward 返回 ``(v_c, h_lo, h_hi, diag)``, 全部在上一级分辨率上、逆深度单位。
    """

    IN_CH = 11

    def __init__(
        self,
        range_k=(1.5, 0.9, 0.6),
        range_min_gi=(0.66, 0.20, 0.10),
        entropy_a: float = 0.5,
        edge_b: float = 0.5,
        hidden: int = 32,
        in_ch: int = 11,
        pnorm_p: float = 16.0,
        rho_max: float = 8.0,
        beta_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.pnorm_p = float(pnorm_p)
        self.log_rho = math.log(float(rho_max))

        # --- 尺度系数: 逆 softplus 初始化, 让 step 0 复现旧公式的两项 ---
        self.kappa = nn.Parameter(torch.tensor([inverse_softplus(v) for v in range_k]))
        self.eta = nn.Parameter(torch.tensor([inverse_softplus(v) for v in range_min_gi]))
        self.w_ent = nn.Parameter(torch.tensor([inverse_softplus(entropy_a)] * 3))
        self.w_edge = nn.Parameter(torch.tensor([inverse_softplus(edge_b)] * 3))
        # 低可靠度 -> 更宽。init 取一个小正数, 不用 softplus(0)=0.693 (那在
        # r=0 时就等于 exp(ln8 * tanh(0.69)) = 3.4x, 一上来就把窗口撑开)。
        self.beta = nn.Parameter(torch.tensor([inverse_softplus(beta_init)] * 3))

        # --- 共享主干, 最后一层零初始化 ---
        self.film = nn.Embedding(3, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        self.in_ch = int(in_ch)
        self.c1 = nn.Conv2d(self.in_ch, hidden, 1)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.head = nn.Conv2d(hidden, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        stage_idx: int,
        feats: torch.Tensor,
        v_m: torch.Tensor,
        w_bar: torch.Tensor,
        g_v: torch.Tensor,
        entropy: torch.Tensor,
        edge: torch.Tensor,
        reliability: torch.Tensor,
        gamma: torch.Tensor,
        v_min: torch.Tensor,
        v_max: torch.Tensor,
        blend: float = 1.0,
    ):
        """
        stage_idx    0/1/2  对应 s1->s2, s2->s3, s3->s4
        feats        [B, IN_CH, H, W]  控制器输入 (全部 detach)
        v_m          [B, H, W]         上一级 winner 的逆深度
        w_bar        [B, H, W]         稳健局部逆深度间隔 (见 depth_range.robust_local_scale)
        g_v          [B, 1, 1]         全局参考逆深度间隔
        entropy/edge [B, H, W]         归一化熵与边缘图
        reliability  [B, H, W]         SPRE 的 r
        gamma        标量张量           该级的 SPRE 门 gamma_{k+1}
        v_min/v_max  [B, 1, 1]         物理范围 (v_min = 1/d_max)
        blend        float             lambda(t) = min(1, step/warm), 同时乘中心与对数尺度
        """
        k = stage_idx
        B, _, H, W = feats.shape

        h = F.gelu(self.c1(feats))
        gs, bs = self.film(torch.full((1,), k, device=feats.device, dtype=torch.long)).chunk(2, dim=-1)
        h = h * (1.0 + gs.view(1, -1, 1, 1)) + bs.view(1, -1, 1, 1)
        h = F.gelu(self.c2(h))
        delta, s_lo, s_hi = self.head(h).float().unbind(dim=1)      # 各 [B,H,W]

        eps = 1e-12
        # --- 基准尺度: 把熵/边缘项放回来, 否则 step 0 对不上旧实现 ---
        ent_term = 1.0 + F.softplus(self.w_ent[k]) * entropy + F.softplus(self.w_edge[k]) * edge
        a_scale = F.softplus(self.kappa[k]) * w_bar.clamp_min(eps) * ent_term.clamp_min(eps)
        b_scale = F.softplus(self.eta[k]) * g_v.clamp_min(eps).expand_as(a_scale)
        log_h0 = _pnorm_softmax(a_scale.clamp_min(eps).log(),
                                b_scale.clamp_min(eps).log(), self.pnorm_p)

        # --- 中心: 最多移动一个上一级 bin ---
        v_c = v_m + blend * w_bar * torch.tanh(delta)

        # --- 半宽: 低可靠度更宽, 但作用随级衰减。
        #     写成 beta*gamma*(1-r) 而不是 beta*(1-gamma*r) —— 后者在 gamma 小时
        #     会把 r=1 (完全可靠) 也判成不可靠。
        unrel = F.softplus(self.beta[k]) * gamma * (1.0 - reliability.clamp(0.0, 1.0))
        h_lo = torch.exp(log_h0 + blend * self.log_rho * torch.tanh(s_lo + unrel))
        h_hi = torch.exp(log_h0 + blend * self.log_rho * torch.tanh(s_hi + unrel))

        # --- 物理范围: 先整体缩放再夹中心。只平移在 "区间比物理域还宽" 时无解。
        span = (v_max - v_min).clamp_min(eps).expand_as(h_lo)
        rho = torch.clamp(span / (h_lo + h_hi + eps), max=1.0)
        h_lo = h_lo * rho
        h_hi = h_hi * rho
        v_c = torch.max(torch.min(v_c, v_max.expand_as(v_c) - h_hi), v_min.expand_as(v_c) + h_lo)

        sat = ((s_lo + unrel).abs() > 2.6465).float().mean() * 0.5 \
            + ((s_hi + unrel).abs() > 2.6465).float().mean() * 0.5    # atanh(0.99)
        diag = {
            "half_lo_mean": h_lo.detach(),
            "half_hi_mean": h_hi.detach(),
            "sat_frac": sat.detach(),
            "rho_bind_frac": (rho < 0.999).float().mean().detach(),
            "delta_abs_mean": delta.detach().abs().mean(),
        }
        return v_c, h_lo, h_hi, diag

    @torch.no_grad()
    def scalar_log(self) -> dict:
        """把五组尺度系数打出来 —— eta -> 0 就说明全局尺度项其实没用。

        键名不带 ``ctrl/`` 前缀: 调用方把它整个塞进 ``range_diag["ctrl"]``,
        损失侧会拼成 ``ctrl/<key>``, 自带前缀就会变成 ``ctrl/ctrl/...``。
        """
        out = {}
        for k in range(3):
            out[f"kappa_s{k + 2}"] = float(F.softplus(self.kappa[k]))
            out[f"eta_s{k + 2}"] = float(F.softplus(self.eta[k]))
            out[f"ent_a_s{k + 2}"] = float(F.softplus(self.w_ent[k]))
            out[f"edge_b_s{k + 2}"] = float(F.softplus(self.w_edge[k]))
            out[f"beta_s{k + 2}"] = float(F.softplus(self.beta[k]))
        return out


class Stage4ResidualHead(nn.Module):
    """stage4 的 **逐候选** 残差头 (W1-D)。

    为什么是逐候选而不是只给 MAP 一个:
    ``m = argmax p`` 不可导, 候选轴又是 detach 的。如果只输出一个残差并在所有
    有效像素上用 SmoothL1 训练, 那么 MAP 选错表面或 GT 出界时, 梯度既改不了
    MAP、也改不了范围、更移不动概率 —— 它只能把残差推向饱和, 让残差头在错误
    模式上长期学习 "尽量顶到边界"。

    改成每个候选一个残差之后, 第 j 个残差只负责自己那个 Voronoi 单元内的修正,
    与它是否赢得 argmax 无关; 训练时只监督离 GT 最近的那个 bin, 推理时才取
    MAP 对应的那个。
    """

    def __init__(self, num_depths: int, feat_channels: int, hidden: int = 32,
                 feat_proj: int = 8, extra_channels: int = 2) -> None:
        super().__init__()
        self.num_depths = num_depths
        self.proj = nn.Conv2d(feat_channels, feat_proj, 1)
        in_ch = num_depths * 2 + 3 + extra_channels + feat_proj
        self.c1 = nn.Conv2d(in_ch, hidden, 1)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.head = nn.Conv2d(hidden, num_depths, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, prob, hypos, interval, mode_idx, sigma, ref_feat, extra):
        """返回未过 tanh 的 [B, D, H, W] 残差 logits。"""
        d_map = hypos.gather(1, mode_idx)                                  # [B,1,H,W]
        dd_map = interval.gather(1, mode_idx).clamp_min(1e-6)
        rel = ((hypos - d_map) / dd_map).clamp(-8.0, 8.0)
        p = prob.float()
        ent = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(1, keepdim=True) \
            / max(math.log(float(prob.shape[1])), 1e-6)
        pmax = p.amax(dim=1, keepdim=True)
        sig = (sigma.unsqueeze(1) / dd_map).clamp(0.0, 8.0)
        f = self.proj(ref_feat.float())
        x = torch.cat([p, rel.float(), ent, pmax, sig, extra.float(), f], dim=1)
        h = F.gelu(self.c1(x))
        h = F.gelu(self.c2(h))
        return self.head(h)
