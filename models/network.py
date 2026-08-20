from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import MVSConfig
from models.cost_volume import CostVolumeBuilder
from models.probe import Probe
from models.decoder import DepthDecoder
from models.depth_range import (
    Stage1Hypotheses,
    build_stage1_hypotheses,
    apply_branch_prior,
    merge_dual_mode,
    refine_range_from_posterior,
    second_mode,
)
from models.fpn import MultiViewFPN


class UprMVSNet(nn.Module):
    """End-to-end cascade MVS network: FPN -> 4-stage cost volume + 3D-UNet.

    Stage-1 hypothesis axis is dual-branch (see models/depth_range.py):

      * global branch — prior-INDEPENDENT guard bins over the robust scene
        range; always unique, never shrunk by prior confidence. Its job is
        coverage and rescue when the prior is wrong.
      * local branch  — dense bins around a spike-robust prior center; its job
        is sub-interval precision and a small stage-2 range when it wins.

    The 3D regularizer receives hypothesis metadata channels (normalized depth
    / spacing / branch id / distance-to-prior / confidence / edge) so it can
    tell the two populations apart instead of learning a sampling-density bias.
    Matching features themselves stay prior-free: correlation evidence must be
    independent of the thing it arbitrates.

    Depth per stage comes from mode-centered regression (not a global
    soft-argmin), and the next stage's range is sized by the *winning bin's*
    sampling interval, widened by posterior entropy and the edge map. All
    hypothesis geometry is detached — each stage is trained by its own loss.

    Expected ``batch`` keys
    -----------------------
        images      [B, V, 3, H, W]   view 0 is the reference
        intrinsics  [B, V, 3, 3]      DTU metric cameras at image resolution
        extrinsics  [B, V, 4, 4]      (same metric frame as depth_prior)
        depth_prior [B, H, W]         ref metric depth   (norm_fill["depth_filled"])
        conf_prior  [B, H, W]         ref confidence      (norm_fill["conf_map"])
        depth_values [B, D]           ref metric depth range; min/max derived
                                      from it (or explicit depth_min/depth_max)
        src_weights [B, V-1]          optional per-source cost-volume weights
    """

    # FPN feature strides for the four cascade stages (coarse -> fine).
    fpn_stage_strides: tuple[int, int, int, int] = (8, 4, 2, 1)

    def __init__(self, cfg: MVSConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MVSConfig()
        cv_cfg = self.cfg.cost_volume
        fpn_cfg = self.cfg.fpn
        dec_cfg = self.cfg.decoder
        self.range_cfg = self.cfg.depth_range

        expected_d1 = self.range_cfg.num_global + self.range_cfg.num_local
        if cv_cfg.num_depths_stage1 != expected_d1:
            raise ValueError(
                f"num_depths_stage1={cv_cfg.num_depths_stage1} must equal "
                f"depth_range.num_global+num_local={expected_d1}"
            )

        self.fpn = MultiViewFPN(
            out_channels=fpn_cfg.out_channels,
            base_channel=fpn_cfg.base_channel,
        )
        fpn_c = fpn_cfg.out_channels

        # Per-stage cost-volume builders: warp-channel width shrinks as
        # resolution grows so the full-res stage does not OOM.
        warp_chs = (cv_cfg.warp_channels_stage1, cv_cfg.warp_channels_stage2,
                    cv_cfg.warp_channels_stage3, cv_cfg.warp_channels_stage4)
        self.cost_builders = nn.ModuleList([
            # 只在 stage1 启用: 全分辨率 stage4 只有 4 个候选, 深度维统计量
            # (peak/entropy) 在那里几乎没有信息量。
            CostVolumeBuilder(fpn_c, wc, cv_cfg.num_groups, cv_cfg.warp_use_half,
                              visibility_weighting=(cv_cfg.visibility_weighting and i == 0))
            for i, wc in enumerate(warp_chs)
        ])
        # Per-stage 3D-UNet decoders. Stage 1 additionally sees the hypothesis
        # metadata channels (its axis is irregular); stages 2-4 use uniform
        # axes and need none.
        mw = self.range_cfg.mode_window
        self.decoders = nn.ModuleList(
            [DepthDecoder(
                in_channels=cv_cfg.num_groups + cv_cfg.stage1_meta_channels,
                base=dec_cfg.unet_base_channels, depth=dec_cfg.unet_depth, mode_window=mw,
            )]
            + [DepthDecoder(in_channels=cv_cfg.num_groups, base=dec_cfg.unet_base_channels,
                            depth=dec_cfg.unet_depth, mode_window=mw) for _ in range(3)]
        )

        # 给探针标记归属, 否则记录里分不清是哪一级
        for i, m in enumerate(self.cost_builders):
            m.tag = f"stage{i + 1}"
        for i, m in enumerate(self.decoders):
            m.tag = f"stage{i + 1}"

        self.num_depths = (cv_cfg.num_depths_stage1, cv_cfg.num_depths_stage2,
                           cv_cfg.num_depths_stage3, cv_cfg.num_depths_stage4)

        # Optional DINOv3 path. One frozen backbone + SVA fusion feeds two
        # consumers: the FPN bottleneck (per-view matching features) and the
        # SPRE reliability head (reference view). Off by default; when off the
        # network is byte-for-byte the previous model and DINOv3 never loads.
        # 三个正交开关 (从前全被 spre.enabled 一个开关捆住, 收益无法归因):
        #   dino.mode              off / all_view / ref_only —— 跑不跑 DINO, 跑几个视角
        #   dino.feed_fpn          DINO 特征喂不喂 FPN (matching 路径)
        #   spre.reliability_source cached / edge / spre —— 可靠度从哪来
        self.prior_mode = str(getattr(self.cfg.prior, "mode", "on")).lower()
        self.dino_mode = str(getattr(self.cfg.dino, "mode", "all_view"))
        self.reliability_source = str(getattr(self.cfg.spre, "reliability_source", "spre"))
        if self.dino_mode == "ref_only":
            raise NotImplementedError(
                "dino_mode='ref_only' 需要 MonoMVSNet 式异构融合 (source 用 CNN/FPN "
                "做 query 去查 reference DINO), 而当前 SVA 与 prior_view_consistency "
                "都依赖 source DINO token。见 models/spre.py:106 与 :246。"
            )
        # 三个开关真正正交, 非法组合直接报错而不是静默降级。
        # spre.enabled 是总闸: 它为 False 时无论 reliability_source 写什么都不会
        # 有 SPRE —— 但这个降级会打印出来, 不是静默的。从前这里完全没读
        # cfg.spre.enabled, 于是 `--spre off` 关不掉 SPRE, 而默认配置
        # (enabled=False + reliability_source="spre") 反而会加载 DINO+SPRE。
        if not self.cfg.spre.enabled and self.reliability_source == "spre":
            print("[net] spre.enabled=False -> reliability_source 由 'spre' 降级为 'cached'")
            self.reliability_source = "cached"
        need_spre = self.reliability_source == "spre"
        self.feed_fpn = self.dino_mode != "off" and bool(self.cfg.dino.feed_fpn)
        self.dino_enabled = self.dino_mode != "off" and (self.feed_fpn or need_spre)
        self.spre_enabled = need_spre
        if need_spre and self.dino_mode == "off":
            raise ValueError(
                "reliability_source='spre' 需要 DINO (SPRE 建在 DinoSVA 的 reference "
                "stream 上), 但 dino_mode='off'。要纯 FPN 消融请同时设 "
                "--reliability cached。"
            )
        if self.dino_mode != "off" and not self.feed_fpn and not need_spre:
            raise ValueError(
                "dino_mode != 'off' 但 feed_fpn=False 且 reliability_source != 'spre' "
                "—— DINO 会被加载却无人使用。请设 dino_mode='off'。"
            )
        if self.dino_enabled:
            from models.spre import SPRE, DinoSVA
            self.dino_sva = DinoSVA(
                self.cfg.spre, self.cfg.dino, self.cfg.paths.dinov3_weights_file,
                fpn_channels=fpn_c if self.feed_fpn else None,
            )
            # 只在真的要用时实例化: 多卡 DDP 用 find_unused_parameters=False,
            # 建了却不 forward 的参数会在 reduction 时报错。
            self.spre = SPRE(self.cfg.spre, self.dino_sva.out_dim) if self.spre_enabled else None

    def _resolve_depth_bounds(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if "depth_min" in batch and "depth_max" in batch:
            return batch["depth_min"].float(), batch["depth_max"].float()
        depth_values = batch["depth_values"].float()
        return depth_values.amin(dim=1), depth_values.amax(dim=1)

    def _resolve_src_weights(self, batch: dict) -> torch.Tensor | None:
        if not self.cfg.cost_volume.use_src_weights:
            return None
        src_weights = batch.get("src_weights")
        if src_weights is None:
            return None
        # floor so a source that sfm failed to match isn't dropped entirely
        return src_weights.float().clamp(min=0.1)

    @staticmethod
    def _stage1_meta(s1: Stage1Hypotheses, global_interval: torch.Tensor,
                     prior_off: bool = False) -> torch.Tensor:
        """[B, 6, D, H, W] hypothesis descriptors for the stage-1 regularizer.

        All detached by construction (the whole bundle is built under no_grad).
        """
        B, D, H, W = s1.hypos.shape
        span = (s1.global_hi - s1.global_lo).view(B, 1, 1, 1).clamp_min(1e-4)
        lo = s1.global_lo.view(B, 1, 1, 1)
        gi = global_interval.view(B, 1, 1, 1).clamp_min(1e-4)

        norm_depth = ((s1.hypos - lo) / span).clamp(0.0, 1.0)
        norm_interval = (s1.interval / gi).clamp(0.0, 4.0)
        dist_prior = ((s1.hypos - s1.prior.unsqueeze(1)) / gi).clamp(-8.0, 8.0) / 8.0
        conf = s1.conf.unsqueeze(1).expand(B, D, H, W)
        edge = s1.edge.unsqueeze(1).expand(B, D, H, W)
        if prior_off:
            # 光把 prior 置零不够: dist_prior = (hypos - 0)/gi 就变成了归一化深度的
            # 另一种写法, 仍是一路有信息的输入。这三通道必须显式中性化, 否则
            # "无先验"对照里还残留着先验位置带来的结构。
            dist_prior = torch.zeros_like(dist_prior)
            conf = torch.zeros_like(conf)
            edge = torch.zeros_like(edge)
        return torch.stack(
            [norm_depth, norm_interval, s1.is_local, dist_prior, conf, edge], dim=1
        ).float()

    def _run_stage(
        self,
        stage_idx: int,
        feats_stage: torch.Tensor,
        K: torch.Tensor,
        E: torch.Tensor,
        depth_hypos: torch.Tensor,
        feature_stride: int,
        src_weights: torch.Tensor | None,
        meta: torch.Tensor | None = None,
        branch_prior=None,
    ) -> tuple[torch.Tensor, ...]:
        cost = self.cost_builders[stage_idx](
            feats_stage[:, 0],
            feats_stage[:, 1:],
            K[:, 0],
            K[:, 1:],
            E[:, 0],
            E[:, 1:],
            depth_hypos,
            feature_stride=feature_stride,
            src_weights=src_weights,
        )
        if meta is not None:
            cost = torch.cat([cost, meta.to(cost.dtype)], dim=1)
        return self.decoders[stage_idx](cost, depth_hypos, branch_prior=branch_prior)

    @staticmethod
    def _upsample_hypos(hypos: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if tuple(hypos.shape[-2:]) == tuple(target_hw):
            return hypos
        return F.interpolate(hypos, size=target_hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _resize_map(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        if tuple(x.shape[-2:]) == tuple(hw):
            return x
        return F.interpolate(x.unsqueeze(1), size=hw, mode="bilinear", align_corners=False).squeeze(1)

    def forward(self, batch: dict, step: int | None = None) -> dict:
        # Normalize to [0, 1] before the FPN: raw 0-255 pixels flow through the
        # full-res input_proj/smooth_p1 branch (whose final conv + smooth carry no
        # norm), so their magnitude propagates into the cost-volume correlation and
        # blows up fp16.
        images = batch["images"].float() / 255.0
        K = batch["intrinsics"].float()
        E = batch["extrinsics"].float()
        depth_prior = batch["depth_prior"].float()
        conf_prior = batch["conf_prior"].float()
        prior_valid = batch.get("prior_valid")
        # prior_mode="off": 严格的无先验对照。置零之后 build_stage1_hypotheses 里
        # `valid = isfinite & >0` 全 False, local 分支退化成 guard 范围内的半偏移
        # 均匀网格 —— 也就是一条不含先验的 48 候选轴。模块**照常构造**, 所以
        # RNG 流与 prior_mode="on" 完全一致, 两者是配对实验。
        prior_off = self.prior_mode == "off"
        if prior_off:
            depth_prior = torch.zeros_like(depth_prior)
            conf_prior = torch.zeros_like(conf_prior)
            prior_valid = torch.zeros(depth_prior.shape[0], device=depth_prior.device)
        depth_min, depth_max = self._resolve_depth_bounds(batch)
        src_weights = self._resolve_src_weights(batch)

        strides = self.fpn_stage_strides
        s1_stride = strides[0]
        coarse_hw = (images.shape[-2] // s1_stride, images.shape[-1] // s1_stride)

        # DINOv3 + SVA runs before the FPN so its per-view features can be
        # injected at the 1/8 bottleneck and propagate down the top-down path.
        fused = raw_tokens = grid = None
        if self.dino_enabled:
            fused, raw_tokens, grid = self.dino_sva(images)   # [B, V, N, dim], [B, V, N, 768]

        dino_fpn = (self.dino_sva.fpn_feature(fused, grid, coarse_hw)
                    if self.feed_fpn else None)
        feats = self.fpn(images, dino=dino_fpn)  # {8: [B,V,C,h,w], 4: ..., 2: ..., 1: ...}

        # ---------- Stage 1 (coarsest, 1/8): dual-branch hypotheses ----------
        feat1 = feats[s1_stride]
        # SPRE: learned reliability from the fused reference tokens + online
        # prior stats replaces the cached conf (which is degenerate). Computed at
        # the stage-1 resolution so the hypothesis builder's resize is a no-op.
        if self.spre_enabled and not prior_off:
            # Cross-view evidence: reproject the prior into every source view and
            # measure agreement. SPRE runs before the cost volume, so this is its
            # only multi-view signal.
            from models.spre import prior_view_consistency
            consistency = prior_view_consistency(
                raw_tokens, depth_prior, K, E, images.shape[-2:], grid
            )
            spre_logits = self.spre(fused[:, 0], grid, depth_prior,
                                    target_hw=feat1.shape[-2:], consistency=consistency)
            conf_used = torch.sigmoid(spre_logits)
        elif self.reliability_source == "edge":
            # 手工门控对照组 (MonoMVSNet 的 edge gate): 可靠度 = 1 - edge
            spre_logits = None
            from models.depth_range import edge_map_from_prior
            _v = torch.isfinite(depth_prior) & (depth_prior > 0)
            conf_used = (1.0 - edge_map_from_prior(
                depth_prior, _v, self.range_cfg.edge_grad_rel)).clamp(0.0, 1.0)
        else:
            spre_logits = None
            conf_used = conf_prior
        s1 = build_stage1_hypotheses(
            depth_prior,
            conf_used,
            depth_min,
            depth_max,
            self.range_cfg,
            target_hw=feat1.shape[-2:],
            prior_valid=prior_valid,
        )
        global_interval = (s1.global_hi - s1.global_lo) / max(self.range_cfg.num_global - 1, 1)  # [B]
        meta1 = self._stage1_meta(s1, global_interval, prior_off=prior_off)
        # 分支先验用 *可微的* conf_used 构造 —— hypothesis 几何仍在 no_grad 里,
        # 但 q 必须带梯度, 否则 branch loss 只训练 decoder 抵消常数, 传不回 SPRE。
        if self.range_cfg.branch_prior and not prior_off:
            q1 = self._resize_map(conf_used, feat1.shape[-2:])
            # beta 退火: 实测 hard prior 只在 0-2k 有益 (post-raw -3.20mm),
            # 2k 之后一路有害 (+0.42mm)。退火让它做完引导就把决定权交还给
            # matching evidence, 而不是全程硬压。
            _n = int(getattr(self.range_cfg, "branch_prior_anneal_steps", 0) or 0)
            _b = float(getattr(self.range_cfg, "branch_prior_beta", 1.0))
            if _n > 0 and step is not None:
                _b = _b * max(0.0, 1.0 - float(step) / float(_n))
            _mode = str(getattr(self.range_cfg, "branch_prior_mode", "hard"))
            bp = lambda lg: apply_branch_prior(
                lg, s1.global_idx, s1.local_idx, q1, s1.branch_active,
                self.range_cfg.branch_q_min, mode=_mode, beta=_b)
            self.last_bp_beta = _b
        else:
            q1, bp = None, None
            self.last_bp_beta = 0.0
        depth1, sigma1, prob1, logits1, mode_idx1, logits1_raw = self._run_stage(
            0, feat1, K, E, s1.hypos, s1_stride, src_weights, meta=meta1,
            branch_prior=bp,
        )

        stage_out = {
            "stage1": {
                "depth": depth1, "sigma": sigma1, "prob": prob1,
                "logits": logits1, "logits_raw": logits1_raw, "depth_hypos": s1.hypos,
                "mode_idx": mode_idx1,
                # branch bookkeeping for the loss / diagnostics
                "is_local": s1.is_local,
                "interval": s1.interval,
                "global_idx": s1.global_idx,
                "local_idx": s1.local_idx,
                "global_lo": s1.global_lo, "global_hi": s1.global_hi,
                "local_lo": s1.local_lo, "local_hi": s1.local_hi,
                "prior": s1.prior, "conf": s1.conf, "edge": s1.edge,
                "global_interval": global_interval,
                "branch_active": s1.branch_active,
                "branch_q": q1,
            },
        }

        # ---------- Stages 2-4: range from the winning candidate ----------
        # winner's sampling interval decides how much correction room the next
        # stage keeps: local win -> narrow, global win -> wide.
        winner_interval = s1.interval.gather(1, mode_idx1).squeeze(1)
        # 第二模态: 把 winner 邻域屏蔽后的次峰。stage1 一旦选错, 正确的候选
        # 即使存在也会被 winner-centered 窗口永久丢弃 (实测 100% 的尾巴都是
        # 选择失败, 轴上一直有 <20mm 的候选)。
        second_depth, second_mass = second_mode(
            prob1, s1.hypos, mode_idx1, self.range_cfg.second_mode_guard)
        range_diag: dict[str, dict] = {}
        prev = {
            "depth": depth1, "prob": prob1, "winner_interval": winner_interval,
            "edge": s1.edge, "hw": feat1.shape[-2:],
            "second_depth": second_depth, "second_mass": second_mass,
        }
        for k in (1, 2, 3):
            feat_k = feats[strides[k]]
            hypos_k, range_stats = refine_range_from_posterior(
                center=prev["depth"],
                winner_interval=prev["winner_interval"],
                prob=prev["prob"],
                edge=prev["edge"],
                config=self.range_cfg,
                num_depths=self.num_depths[k],
                global_interval=global_interval,
                depth_min=depth_min,
                depth_max=depth_max,
                stage_idx=k - 1,
            )
            # stage2 双模态: 一半候选留给次峰, 仅在分支分歧/高熵像素上启用
            if k == 1 and self.range_cfg.dual_mode_stage2 and prev.get("second_depth") is not None:
                hypos_k = merge_dual_mode(
                    hypos_k, prev["second_depth"], prev["second_mass"],
                    prev["winner_interval"], self.range_cfg)
            hypos_k = self._upsample_hypos(hypos_k, feat_k.shape[-2:])
            depth_k, sigma_k, prob_k, logits_k, mode_idx_k, _ = self._run_stage(
                k, feat_k, K, E, hypos_k, strides[k], src_weights
            )
            range_diag[f"stage{k + 1}"] = range_stats
            edge_k = self._resize_map(s1.edge, feat_k.shape[-2:])
            stage_out[f"stage{k + 1}"] = {
                "depth": depth_k, "sigma": sigma_k, "prob": prob_k,
                "logits": logits_k, "depth_hypos": hypos_k,
                "mode_idx": mode_idx_k, "edge": edge_k,
            }
            # uniform axis: per-pixel interval = span / (D - 1)
            interval_k = (hypos_k[:, -1] - hypos_k[:, 0]) / max(self.num_depths[k] - 1, 1)
            prev = {
                "depth": depth_k, "prob": prob_k, "winner_interval": interval_k,
                "edge": edge_k, "hw": feat_k.shape[-2:], "second_depth": None,
            }

        depth_last = stage_out[f"stage{len(strides)}"]["depth"]
        depth_full = F.interpolate(
            depth_last.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        # spre_logits 为 None 有两种情况: reliability_source != "spre", 或者
        # prior_mode="off" 时 SPRE 头根本没参与前向。两种都不该产出 SPRE 监督 ——
        # 否则 loss 侧会拿 None 去取 shape, 而且无先验对照里也不该有先验可靠度损失。
        if self.spre_enabled and spre_logits is not None:
            stage_out["spre"] = {
                "logits": spre_logits, "r": conf_used, "hw": tuple(feat1.shape[-2:]),
            }

        vs = getattr(self.cost_builders[0], "last_vis_stats", None)
        if vs is not None:
            stage_out["vis"] = vs
        stage_out["range_diag"] = range_diag
        stage_out["range_diag"]["branch"] = {"bp_beta": float(getattr(self, "last_bp_beta", 0.0))}
        stage_out["range_diag"]["cost"] = {
            f"max_s{i + 1}": float(getattr(m, "last_cost_max", 0.0))
            for i, m in enumerate(self.cost_builders)
        }
        Probe.log("out", "depth_full", depth_full)
        return {"depth_full": depth_full, **stage_out}
