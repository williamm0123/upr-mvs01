from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.probe import Probe
from utils.geometry import homography_warp_features


def group_wise_correlation(ref: torch.Tensor, warped: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Correlate ref against each depth-warped source, group-wise.

    ref    : [B, C, H, W]       (already projected to warp_channels)
    warped : [B, C, D, H, W]
    return : [B, num_groups, D, H, W]
    """
    B, C, D, H, W = warped.shape
    assert C % num_groups == 0, f"channels {C} not divisible by groups {num_groups}"
    g = C // num_groups
    ref_g = ref.view(B, num_groups, g, 1, H, W)
    warp_g = warped.view(B, num_groups, g, D, H, W)
    # Multiply in fp32: features can be sampled/stored in fp16 (use_half), but the
    # correlation product grows ~O(C * feat^2) and overflows fp16's ~65504 range on
    # the full-res stage, producing inf -> NaN downstream. Casting after .mean() is
    # too late; the overflow happens in the elementwise product.
    return (ref_g.float() * warp_g.float()).mean(dim=2)


class VisibilityHead(nn.Module):
    """Per-(source, pixel) 可见性权重, 由该 source 自己的相关体统计量预测。

    动机 (experiments/out/coloc_val_*.log): stage1 尾巴里 58.9% 是 global 分支
    赢了却选错平面 —— 而聚合原本是所有 source 的等权平均, 一个被遮挡的 source
    和一个完全可见的 source 权重相同。遮挡是逐像素的, per-view 标量表达不了。

    输入三个与 D 无关的统计量, 所以参数量和深度假设数无关:
      peak  相关性在深度维上的最大值   (匹配到没有)
      ent   深度维 softmax 的归一化熵  (峰是否尖锐)
      mean  深度维均值                 (整体相关强度基线)
    """

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, cv_s: torch.Tensor) -> torch.Tensor:
        """``cv_s`` [B,G,D,H,W] -> 未归一化的 logits [B,1,H,W]。

        调用方在收齐所有 source 之后在 source 维做 softmax —— 每个 source 独立
        sigmoid 不构成竞争, 无法表达"这块区域该信谁"。
        """
        c = cv_s.mean(dim=1)                                   # [B,D,H,W]
        peak = c.amax(dim=1, keepdim=True)
        mean = c.mean(dim=1, keepdim=True)
        p = F.softmax(c.float(), dim=1).clamp_min(1e-8)
        D = c.shape[1]
        ent = (-(p * p.log()).sum(dim=1, keepdim=True) / max(torch.log(torch.tensor(float(D))).item(), 1e-6))
        x = torch.cat([peak, mean, ent.to(peak.dtype)], dim=1)
        return self.net(x)                                      # [B,1,H,W] logits


class CostVolumeBuilder(nn.Module):
    """One cascade stage: plane-sweep matching -> group-correlation cost volume.

    Pipeline per call:
        1. project ref + every source FPN feature (in_channels) down to a small
           ``warp_channels`` width via a shared 1x1 conv  (memory control);
        2. homography-warp each source into the ref frame at every depth
           hypothesis  -> [B, warp_channels, D, H, W]  (the memory hot spot);
        3. group-wise correlate with the ref feature and average over sources
           -> cost volume [B, num_groups, D, H, W].

    ``warp_channels`` shrinks across stages (64/32/16) so the full-resolution
    stage stays off the OOM cliff; ``use_half`` samples/correlates in fp16 on
    CUDA (geometry stays fp32 inside the warp) for a further ~2x cut.
    """

    def __init__(
        self,
        in_channels: int,
        warp_channels: int,
        num_groups: int = 8,
        use_half: bool = True,
        visibility_weighting: bool = False,
    ) -> None:
        super().__init__()
        if warp_channels % num_groups != 0:
            raise ValueError(
                f"warp_channels {warp_channels} must be divisible by num_groups {num_groups}"
            )
        self.proj = nn.Conv2d(in_channels, warp_channels, kernel_size=1, bias=False)
        self.warp_channels = warp_channels
        self.num_groups = num_groups
        # 无条件构造, 只用 use_vis 开关它是否参与前向。
        # 条件构造 (``VisibilityHead() if visibility_weighting else None``) 会改变
        # 之后每个模块从全局 RNG 取到的随机数, 于是 "只关掉可见性头" 的消融实际上
        # 连 decoder 的初始权重都换了一套, 不是配对实验。这里始终建、始终消耗同样
        # 的 RNG, 关掉时它只是不参与前向也不产生梯度。
        self.vis_head = VisibilityHead()
        self.use_vis = bool(visibility_weighting)
        self.tag = "stage?"          # network.py 构造后写入, 只给探针用
        if not self.use_vis:
            # 关掉时冻结: 它拿不到梯度, 不冻结的话 DDP (find_unused_parameters=False)
            # 会因为"有梯度需求却没收到梯度"报错, 优化器也会白白为它建 state。
            # 权重仍在 state_dict 里, 所以两种配置的 checkpoint 互相可加载。
            self.vis_head.requires_grad_(False)
        self.use_half = use_half

    def _sample_dtype(self, ref: torch.Tensor) -> torch.dtype:
        """warp / 相关体采样用的精度。

        以前这里硬返回 torch.float16, 于是即使外层 autocast 换成 bf16, 投影特征
        仍会被转回 fp16 —— stage4 全分辨率那条溢出路径原封不动。现在跟随当前
        autocast 的 dtype: bf16 下就是 bf16 (指数范围同 fp32), 关掉 AMP 时退回
        输入精度。CUDA 的 grid_sample 对 bf16 有原生支持 (已实测)。
        """
        if not (self.use_half and ref.is_cuda):
            return ref.dtype
        try:
            if torch.is_autocast_enabled():
                return torch.get_autocast_dtype("cuda")
        except Exception:
            pass
        return torch.float16

    def forward(
        self,
        ref_feat: torch.Tensor,
        src_feats: torch.Tensor,
        K_ref: torch.Tensor,
        K_src: torch.Tensor,
        E_ref: torch.Tensor,
        E_src: torch.Tensor,
        depth_hypos: torch.Tensor,
        feature_stride: int,
        src_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        ref_feat    : [B, C, H, W]
        src_feats   : [B, S, C, H, W]
        K_ref/E_ref : [B, 3, 3] / [B, 4, 4]
        K_src/E_src : [B, S, 3, 3] / [B, S, 4, 4]
        depth_hypos : [B, D, H, W]
        return      : cost volume [B, num_groups, D, H, W]
        """
        B, S, C, H, W = src_feats.shape
        D = depth_hypos.shape[1]
        sample_dtype = self._sample_dtype(ref_feat)
        
        ref_p = self.proj(ref_feat).to(sample_dtype)
        Probe.log(self.tag, "ref_p", ref_p)

        # 缓存的 src_weights 可能比当前 source 数短 (见 config 注释) —— 与其
        # 让它在中途 IndexError, 不如显式忽略。
        if src_weights is not None and src_weights.shape[1] != S:
            # 必须严格等长: 更长也不能用, 那是按另一组视角算出来的权重。
            src_weights = None
        agg = ref_feat.new_zeros(B, self.num_groups, D, H, W, dtype=torch.float32)
        weight_sum = ref_feat.new_zeros(B, 1, 1, 1, 1, dtype=torch.float32)
        # 两遍: 先收齐每个 source 的相关体和可见性 logits, 再在 source 维竞争
        # 归一化。一遍式的 per-source sigmoid 不构成竞争。
        cvs, vlogits = [], []
        for s in range(S):
            src_p = self.proj(src_feats[:, s]).to(sample_dtype)
            Probe.log(self.tag, "src_p", src_p, src=s)
            warped = homography_warp_features(
                src_p,
                K_ref,
                K_src[:, s],
                E_ref,
                E_src[:, s],
                depth_hypos,
                feature_stride,
            )
            Probe.log(self.tag, "warped", warped, src=s)
            cv_s = group_wise_correlation(ref_p, warped, self.num_groups).float()
            Probe.log(self.tag, "cv_s", cv_s, src=s)
            cvs.append(cv_s)
            if self.use_vis:
                vlogits.append(self.vis_head(cv_s))
        self.last_vis_stats = None
        if self.use_vis:
            pi = torch.softmax(torch.stack(vlogits, dim=0).float(), dim=0)   # [S,B,1,H,W]
            vis = pi * S                                                     # 均值恒为 1
            with torch.no_grad():
                # 记 mean 没有信息 (softmax*S 的 source 维均值按构造就是 1)。
                # 有信息的是这个分布有多集中:
                p_ = pi.squeeze(2).clamp_min(1e-8)                           # [S,B,H,W]
                ent = (-(p_ * p_.log()).sum(0)) / max(float(np.log(S)), 1e-6)  # 归一化熵
                mx = p_.amax(0)
                self.last_vis_stats = {
                    "ent": float(ent.mean()),          # 1 = 完全均匀, 0 = 只信一个源视
                    "max_w": float(mx.mean()),
                    "eff_src": float(torch.exp(-(p_ * p_.log()).sum(0)).mean()),
                    "concentrated": float((mx > 0.7).float().mean()),
                    "num_src": float(S),
                }
        for s, cv_s in enumerate(cvs):
            w = (
                src_weights[:, s].view(B, 1, 1, 1, 1).float()
                if src_weights is not None
                else ref_feat.new_ones(B, 1, 1, 1, 1, dtype=torch.float32)
            )
            if self.use_vis:
                w = w * vis[s].unsqueeze(2)          # [B,1,1,H,W]
            agg = agg + cv_s * w
            weight_sum = weight_sum + w
        cost = agg / weight_sum.clamp(min=1e-6)
        Probe.log(self.tag, "cost", cost)
        # 相关体的幅度 —— 2026-08-21 那次事故的唯一直接指标。
        # stage4 的 cost 量级是 stage1 的约 200 倍 (特征幅度 ~13 -> ~500, 而组内
        # 平均的通道数 16 -> 2), 实测最大值经常越过 fp16 的 65504: 越过的那个样本
        # 在 autocast 转 fp16 的瞬间就变 inf, 于是 logits - logits.amax() = nan。
        # bf16 下这条线不再是威胁, 但幅度本身仍是训练是否健康的信号。
        with torch.no_grad():
            self.last_cost_max = float(cost.detach().abs().amax())
        return cost
