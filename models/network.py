from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import MVSConfig
from models.cost_volume import CostVolumeBuilder
from models.probe import Probe
from models.decoder import DepthDecoder
from models.fusion_conf import FusionConfidenceHead
from models.range_controller import (
    RangeController,
    SpreGates,
    Stage4ResidualHead,
)
from models.depth_range import (
    Stage1Hypotheses,
    blend_axes,
    build_axis_inverse,
    local_intervals,
    robust_local_scale,
    second_mode_physical,
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
        geo_valid = bool(getattr(cv_cfg, "geo_valid_aggregation", False))
        self.geo_valid = geo_valid
        self.vis_supervise = bool(getattr(cv_cfg, "vis_supervise", False))
        self.cost_builders = nn.ModuleList([
            # 只在 stage1 启用: 全分辨率 stage4 只有 4 个候选, 深度维统计量
            # (peak/entropy) 在那里几乎没有信息量。
            CostVolumeBuilder(fpn_c, wc, cv_cfg.num_groups, cv_cfg.warp_use_half,
                              visibility_weighting=(cv_cfg.visibility_weighting and i == 0),
                              vis_mode=str(getattr(cv_cfg, "vis_mode", "softmax")),
                              geo_valid=geo_valid)
            for i, wc in enumerate(warp_chs)
        ])
        # Per-stage 3D-UNet decoders. Stage 1 additionally sees the hypothesis
        # metadata channels (its axis is irregular); stages 2-4 use uniform
        # axes and need none.
        mw = self.range_cfg.mode_window
        mws = getattr(self.range_cfg, "mode_window_stages", None) or (mw, mw, mw, mw)
        self.mode_windows = tuple(mws)
        # --- 逐级通道数记账 ---
        #   n_geo : W3-A 的 n_valid/S 通道 (每级 +1)
        #   n_spre: W1-F 的 gamma_k*r / gamma_k*edge (stage2-4 各 +2)
        n_geo = 1 if geo_valid else 0
        self.spre_cascade = bool(getattr(self.range_cfg, "spre_cascade", False))
        n_spre = 2 if self.spre_cascade else 0
        self.learn_range = str(getattr(self.range_cfg, "axis_space", "legacy_depth")).lower() == "inverse"
        self.stage4_map = str(getattr(self.range_cfg, "stage4_head", "expect")).lower() == "map"
        head_modes = ("expect", "expect", "expect", "map" if self.stage4_map else "expect")
        self.decoders = nn.ModuleList(
            [DepthDecoder(
                in_channels=cv_cfg.num_groups + n_geo + cv_cfg.stage1_meta_channels,
                base=dec_cfg.unet_base_channels, depth=dec_cfg.unet_depth,
                mode_window=mws[0], head_mode=head_modes[0],
            )]
            + [DepthDecoder(in_channels=cv_cfg.num_groups + n_geo + n_spre,
                            base=dec_cfg.unet_base_channels, depth=dec_cfg.unet_depth,
                            mode_window=mws[i + 1], head_mode=head_modes[i + 1])
               for i in range(3)]
        )

        # --- W1: 可学习范围控制器 / SPRE 门 / stage4 逐候选残差头 ---
        # 无条件构造, 只用开关决定是否参与前向 —— 与 VisibilityHead 同一个理由:
        # 条件构造会改变之后每个模块从全局 RNG 取到的随机数, 于是 "只换一个开关"
        # 的消融连初始权重都换了一套, 不再是配对实验。关掉时冻结, 避免 DDP
        # (find_unused_parameters=False) 因 "有梯度需求却没收到梯度" 报错。
        self.spre_gates = SpreGates(tuple(getattr(self.range_cfg, "spre_gate_init",
                                                  (1.0, 0.60, 0.35, 0.20))))
        self.range_ctrl = RangeController(
            range_k=tuple(self.range_cfg.range_k),
            range_min_gi=tuple(self.range_cfg.range_min_gi),
            entropy_a=float(self.range_cfg.range_entropy_a),
            edge_b=float(self.range_cfg.range_edge_b),
            hidden=int(getattr(self.range_cfg, "ctrl_hidden", 32)),
            pnorm_p=float(getattr(self.range_cfg, "pnorm_p", 16.0)),
            rho_max=(tuple(self.range_cfg.rho_stages)
                     if getattr(self.range_cfg, "rho_stages", None)
                     else float(getattr(self.range_cfg, "rho_max", 8.0))),
            beta_init=float(getattr(self.range_cfg, "ctrl_beta_init", 0.05)),
            refine_ratio_init=tuple(getattr(self.range_cfg, "refine_ratio_init",
                                            (0.4, 0.5142857142857142, 0.8))),
            refine_cap_p=float(getattr(self.range_cfg, "refine_cap_p", 16.0)),
        )
        self.res_head = Stage4ResidualHead(
            num_depths=cv_cfg.num_depths_stage4, feat_channels=fpn_c,
            hidden=int(getattr(self.range_cfg, "ctrl_hidden", 32)),
        )
        # ---- 冻结策略: 开关**和**损失权重一起决定 ----
        # 这几个头的梯度只有一个来源, 权重置 0 就等于"要梯度却收不到梯度"。
        # DDP 是 find_unused_parameters=False, 那会在**第二个 iteration** 抛
        # "Expected to have finished reduction in the prior iteration" ——
        # 单卡跑得好好的, 排队几小时之后才炸。所以这里按实际能否收到梯度来冻结。
        lcfg = getattr(cfg, "loss", None)
        w_range = float(getattr(lcfg, "w_range", 0.0)) if lcfg else 0.0
        w_residual = float(getattr(lcfg, "w_residual", 0.0)) if lcfg else 0.0
        w_conf = float(getattr(lcfg, "w_conf", 0.0)) if lcfg else 0.0
        ctrl_trains = self.learn_range and w_range > 0
        res_trains = self.stage4_map and w_residual > 0
        # gamma 的梯度有三条来源: meta 通道 (spre_cascade) -> 各级 CE;
        # 半宽里的 beta*gamma*(1-r) -> range loss; stage4 残差头的 extra -> l_res。
        if not (self.spre_cascade or ctrl_trains or res_trains):
            self.spre_gates.requires_grad_(False)
        if not ctrl_trains:
            self.range_ctrl.requires_grad_(False)
        self.conf_head = FusionConfidenceHead(
            feat_channels=fpn_c,
            hidden=int(getattr(dec_cfg, "fusion_conf_hidden", 32)),
            prior_pos=float(getattr(dec_cfg, "fusion_conf_prior", 0.9)),
        )
        self.use_conf_head = bool(getattr(dec_cfg, "fusion_conf", False))
        self.conf_detach = bool(getattr(dec_cfg, "fusion_conf_detach", True))
        if not res_trains:
            self.res_head.requires_grad_(False)
        if not (self.use_conf_head and w_conf > 0):
            # 置信度头的 logit **只**流向 L_conf (输出里的 prob 是 detach 的),
            # w_conf=0 时它一条梯度都收不到。
            self.conf_head.requires_grad_(False)
        if self.use_conf_head:
            # source 间相关性统计只有 stage4 用得上, 只在那一级收集。
            self.cost_builders[3].collect_src_stats = True
        self.child_interval_cap = bool(getattr(self.range_cfg, "child_interval_cap", False))
        if not (self.child_interval_cap and ctrl_trains):
            # xi_k 只在 cap 打开时进前向 (关掉时 q 恒为 1, 与它无关)。
            # 不冻结的话它就是一个"要梯度却永远收不到"的参数 —— 白占优化器
            # 状态, 而且 DDP(find_unused_parameters=False) 会在第二步直接报错。
            self.range_ctrl.refine_ratio_raw.requires_grad_(False)
        self.residual_scale = float(getattr(self.range_cfg, "residual_scale", 0.5))
        self.axis_blend_steps = int(getattr(self.range_cfg, "axis_blend_steps", 2000))

        if self.vis_supervise:
            # 可见性头只在 stage1 参与前向, 所以投影几何也只在那一级留。
            self.cost_builders[0].collect_vis_geom = True

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

        # ---- CVPE (工单 v5.3 的主线模块) --------------------------------------
        # **必须构造在最后**: 关闭时不构造, 开启时也只在所有既有模块之后从全局
        # RNG 取数, 于是 cvpe=off 的 checkpoint 与改动前逐位一致, cvpe=on 也不会
        # 让前面任何模块拿到不同的初始权重。两者是严格配对的实验。
        cvpe_cfg = getattr(self.cfg, "cvpe", None)
        self.cvpe = None
        if cvpe_cfg is not None and bool(cvpe_cfg.enabled):
            from models.cvpe import CrossViewPE
            self.cvpe = CrossViewPE(
                in_channels=fpn_c,
                d_model=int(cvpe_cfg.d_model),
                num_planes=int(cvpe_cfg.num_planes),
                n_heads=int(cvpe_cfg.n_heads),
                layer_pattern=tuple(x.strip() for x in cvpe_cfg.layer_pattern.split(",")),
                cam_mid_channels=int(cvpe_cfg.cam_mid_channels),
            )

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


    def _range_ctrl_feats(self, prev: dict, g_v, r_map, gamma, n_valid):
        """组装范围控制器的 11 通道输入。**全部 detach** —— 控制器是一条自带监督
        的旁路, 它的梯度只来自 pinball/center 损失; 而候选轴对 matching 路径也
        detach。原因见 losses/composite 里的注释: 各级 CE 的掩码是
        ``valid & in_range``, 若窗口位置对 CE 可导, 缩窗排除难像素就能直接降 loss。
        """
        with torch.no_grad():
            hw = tuple(prev["hw"])
            prob = prev["prob"].float()
            D = prob.shape[1]
            ent = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(1) \
                / max(math.log(float(D)), 1e-6)
            pmax = prob.amax(dim=1)
            v_m = 1.0 / prev["depth"].float().clamp_min(1e-6)
            wbar_v = prev["w_bar_v"]
            sig_rel = (prev["sigma"].float() / prev["winner_interval"].clamp_min(1e-6)).clamp(0, 8)
            log_scale = (wbar_v / g_v.expand_as(wbar_v)).clamp_min(1e-8).log().clamp(-8, 8) / 8.0
            e_map = self._resize_map(prev["edge"].float(), hw)
            r_k = self._resize_map(r_map.float(), hw)
            nd = ((prev["depth"].float() - prev["d_min"]) / prev["d_span"]).clamp(0, 1)
            nv = n_valid if n_valid is not None else torch.ones_like(ent)
            feats = torch.stack([
                ent, pmax, sig_rel, log_scale,
                prev["mass2"], (prev["dv2"] / wbar_v.clamp_min(1e-12)).clamp(0, 8) / 8.0,
                gamma * r_k, gamma * e_map,
                nd, nv.clamp(0, 1), prev["found2"].float(),
            ], dim=1)
            return feats.float(), e_map, r_k, v_m

    @staticmethod
    def _mode_mass(prob: torch.Tensor, mode_idx: torch.Tensor, window: int = 1) -> torch.Tensor:
        """mode 邻域 +-window 个 bin 的后验质量 —— 现有 cascade_confidence 的原料。"""
        D = prob.shape[1]
        offs = torch.arange(-window, window + 1, device=prob.device).view(1, -1, 1, 1)
        idx = (mode_idx + offs).clamp(0, D - 1)
        return prob.float().gather(1, idx).sum(dim=1)

    def _fusion_conf_feats(self, stage_out, hw, gamma4, r_full, g_d, n_valid, src_stats,
                           prob4, hypos4, interval4, mode_idx4, sigma4, depth4, depth3,
                           residual):
        """组装融合置信度头的 12 通道输入。

        全部是**前向阶段拿得到**的量。n_geo 和真正的跨视重投影一致性要等所有
        参考视图的深度图存完、在融合阶段才有 —— 它们只能进标签, 不能进这里。
        """
        eps = 1e-6
        dd = interval4.gather(1, mode_idx4).squeeze(1).clamp_min(eps)
        p = prob4.float()
        ent = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(1) \
            / max(math.log(float(p.shape[1])), 1e-6)
        top2 = p.topk(min(2, p.shape[1]), dim=1).values
        pmax = top2[:, 0]
        margin = pmax - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(pmax))
        res_mag = torch.zeros_like(pmax) if residual is None else \
            torch.tanh(residual.gather(1, mode_idx4).squeeze(1).float()).abs()
        d3 = self._resize_map(depth3.float(), hw)
        move = ((depth4.float() - d3) / dd).clamp(-4.0, 4.0) / 4.0
        width = (dd / g_d.clamp_min(eps).expand_as(dd)).clamp(0.0, 8.0) / 8.0
        rk = self._resize_map(r_full, hw)
        nv = torch.ones_like(pmax) if n_valid is None else n_valid.clamp(0.0, 1.0)
        if src_stats is None:
            cmean = torch.zeros_like(pmax)
            cstd = torch.zeros_like(pmax)
        else:
            idx = mode_idx4.unsqueeze(1).expand(-1, 2, -1, -1, -1)
            st = src_stats.gather(2, idx).squeeze(2).float()          # [B,2,H,W]
            m_, s_ = st[:, 0], st[:, 1]
            # 尺度无关的一致性: stage4 的 cost 量级是 stage1 的约 200 倍, 原始值
            # 直接喂进去会被幅度而不是一致性主导。
            den = (m_.abs() + s_ + eps)
            cmean, cstd = m_ / den, s_ / den
        casc = torch.ones_like(pmax)
        for name in ("stage1", "stage2", "stage3", "stage4"):
            st_k = stage_out.get(name)
            if st_k is None or "mode_idx" not in st_k:
                continue
            casc = casc * self._resize_map(
                self._mode_mass(st_k["prob"], st_k["mode_idx"], 1), hw)
        sig = (sigma4.float() / dd).clamp(0.0, 8.0) / 8.0
        return torch.stack([ent, pmax, margin, sig, res_mag, move, width,
                            gamma4 * rk, nv, cmean, cstd, casc], dim=1)

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

        # CVPE 挂在 FPN 的 1/8 注入点上 (DINO 之后、top-down 之前), 所以它的修改
        # 沿 p8->p4->p2->p1 传遍四级 —— 不需要另建一条平行的降维/上采样链。
        # 模块只返回 delta, **残差相加只在这里做一次**; 用 cat 重建张量而不是
        # 对 p8[:, 1:] 原地赋值 (原地写 autograd 叶子会静默地丢梯度)。
        cvpe_fn = None
        if self.cvpe is not None:
            def cvpe_fn(p8: torch.Tensor) -> torch.Tensor:
                delta = self.cvpe(
                    p8=p8, K=K, E=E,
                    depth_min=depth_min.reshape(-1), depth_max=depth_max.reshape(-1),
                    feature_stride=s1_stride,
                )
                return torch.cat([p8[:, :1], p8[:, 1:] + delta], dim=1)

        feats = self.fpn(images, dino=dino_fpn, cvpe_fn=cvpe_fn)  # {8: [B,V,C,h,w], ...}

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
        # 第二模态: 把 winner 邻域屏蔽后的次峰 (旧口径, 只喂给 legacy 双模态)。
        second_depth, second_mass = second_mode(
            prob1, s1.hypos, mode_idx1, self.range_cfg.second_mode_guard)

        # --- 逆深度空间的公共量 ---
        v_min = (1.0 / depth_max.clamp_min(1e-6)).view(-1, 1, 1)
        v_max = (1.0 / depth_min.clamp_min(1e-6)).view(-1, 1, 1)
        g_v = ((v_max - v_min) / max(self.range_cfg.num_global - 1, 1)).clamp_min(1e-12)
        d_min_b = depth_min.view(-1, 1, 1)
        d_span_b = (depth_max - depth_min).view(-1, 1, 1).clamp_min(1e-3)
        gammas = self.spre_gates()                       # [4], gamma 不 detach
        r_full = conf_used if (self.spre_enabled and not prior_off) else torch.zeros_like(s1.edge)
        r_full = r_full.detach()                         # stop-grad 只作用在 r/e 上
        # lambda(t): 同时乘在中心偏移、对数尺度和 legacy<->inverse 的轴凸组合上。
        # 只给损失做 warm-up 挡不住一个未训练的控制器在第一步就改变深度轴。
        blend = 1.0 if step is None else min(1.0, float(step) / max(self.axis_blend_steps, 1))

        def _n_valid_at(builder_idx, mode_idx):
            if not self.geo_valid:
                return None
            nv = getattr(self.cost_builders[builder_idx], "last_n_valid", None)
            if nv is None:
                return None
            S = max(images.shape[1] - 1, 1)
            return (nv.gather(1, mode_idx).squeeze(1) / S).detach()

        # stage1 的第二峰 —— 物理逆深度口径 (deployable), 只当控制器特征
        # v_axis 按深度升序排列 -> 在逆深度上是降序; local_intervals 要求单调,
        # 所以 flip 过去算完再 flip 回来, 索引与 hypos 保持一致。
        v_axis1 = 1.0 / s1.hypos.float().clamp_min(1e-6)
        # min_interval=0: local_intervals 的默认下限 1e-4 是**毫米**量纲的,
        # 逆深度间距 ~1e-5, 用默认值会把 w_bar 整张图钉在 1e-4 上 (窗口宽 3-20 倍,
        # 而且不报错)。逆深度轴的退化交给 robust_local_scale 的 g_v 回退。
        wbar_v1, wbar_fb1 = robust_local_scale(
            local_intervals(v_axis1.flip(dims=[1]), min_interval=0.0).flip(dims=[1]),
            mode_idx1, g_v, return_fallback=True)
        # h_sep: 分离判据的物理尺度, 用下一级 (stage2) 的基准半宽, 逐像素
        h_sep1 = (float(self.range_cfg.range_k[0]) * wbar_v1).clamp_min(1e-12)
        v2_1, mass2_1, dv2_1, found2_1, j2_1 = second_mode_physical(
            prob1.float(), v_axis1, mode_idx1, h_sep1, h_sep1)
        # 期望窗口跨两个表面的比例: 窗口内出现一个**物理分离**的次峰, 那次期望
        # 就落在两个真实表面之间。stage3 偏高 = stage3 也该收窗或改 MAP。
        bimodal_diag = {"stage1": {"regress_window_bimodal_frac": float(
            (found2_1 & ((j2_1.squeeze(1) - mode_idx1.squeeze(1)).abs()
                         <= self.mode_windows[0])).float().mean())}}

        range_diag: dict[str, dict] = {}
        range_ctrl_out: dict[str, dict] = {}
        conf_ctx: dict | None = None
        prev = {
            "depth": depth1, "prob": prob1, "sigma": sigma1,
            "winner_interval": winner_interval, "mode_idx": mode_idx1,
            "edge": s1.edge, "hw": feat1.shape[-2:],
            "second_depth": second_depth, "second_mass": second_mass,
            "w_bar_v": wbar_v1, "mass2": mass2_1, "dv2": dv2_1, "found2": found2_1,
            "d_min": d_min_b, "d_span": d_span_b, "wbar_fb": wbar_fb1,
            "n_valid": _n_valid_at(0, mode_idx1),
        }
        for k in (1, 2, 3):
            feat_k = feats[strides[k]]
            hypos_leg, range_stats = refine_range_from_posterior(
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
            if self.learn_range:
                ctrl_feats, e_map, r_k, v_m = self._range_ctrl_feats(
                    prev, g_v, r_full, gammas[k].detach(), prev["n_valid"])
                v_c, h_lo, h_hi, ctrl_diag = self.range_ctrl(
                    k - 1, ctrl_feats, v_m, prev["w_bar_v"], g_v,
                    entropy=ctrl_feats[:, 0], edge=e_map, reliability=r_k,
                    gamma=gammas[k], v_min=v_min, v_max=v_max,
                    num_depths=self.num_depths[k],
                    apply_interval_cap=self.child_interval_cap, blend=blend,
                )
                hypos_inv = build_axis_inverse(v_c, h_lo, h_hi, self.num_depths[k], v_min, v_max)
                # legacy -> inverse 的凸组合: 两条轴都按深度升序, 组合仍然升序。
                hypos_k = blend_axes(hypos_leg, hypos_inv, blend)
                range_ctrl_out[f"stage{k + 1}"] = {
                    "v_lo": v_c - h_lo, "v_hi": v_c + h_hi, "v_c": v_c,
                    "v_m": v_m, "w_bar": prev["w_bar_v"], "hw": tuple(prev["hw"]),
                    # 与当前窗口**无关**的固定尺度, 供 pinball_scale=global 用。
                    # 切断 "轴变宽 -> 分母变大 -> 宽度惩罚变小" 那条反馈。
                    "w_fixed": g_v.expand_as(prev["w_bar_v"]),
                    "tau": float(self.range_cfg.tau_stages[k - 1]),
                }
                with torch.no_grad():
                    # **deprecated**: 一阶换算 h_d ~ h_v/v_c^2, 大窗口下误差明显。
                    # 保留只为历史曲线可比, 判读用 ctrl_exact_half_mm_*。
                    hm = (0.5 * (h_lo + h_hi) / v_c.clamp_min(1e-12) ** 2).flatten(1)[:, ::97].float()
                    range_stats["ctrl_half_p50"] = float(hm.quantile(0.5)) if hm.numel() else 0.0
                    range_stats["ctrl_half_p90"] = float(hm.quantile(0.9)) if hm.numel() else 0.0
                    range_stats["ctrl_sat_frac"] = float(ctrl_diag["sat_frac"])
                    range_stats["ctrl_rho_bind"] = float(ctrl_diag["rho_bind_frac"])
                    range_stats["ctrl_delta_abs"] = float(ctrl_diag["delta_abs_mean"])
                    # 范围分解 + 级联分辨率。只看 h 分不出 "上一级把基准撑大了"
                    # (A/h0 大) 和 "控制器自己要更宽" (mult 大); 也分不出 cap 拦了
                    # 多少 (gap_cap_bind) 与物理域拦了多少 (physical_bind)。
                    for _k in ("A_over_B_p50", "A_over_B_p90",
                               "h0_over_gv_p50", "h0_over_gv_p90",
                               "mult_raw_p50", "mult_raw_p90", "refine_ratio",
                               "gap_ratio_raw_p50", "gap_ratio_raw_p90",
                               "gap_ratio_final_p50", "gap_ratio_final_p90",
                               "gap_ratio_final_p99",
                               "gap_cap_bind_frac", "gap_cap_q_p50", "gap_cap_q_p10",
                               "physical_bind_frac",
                               "exact_half_mm_p50", "exact_half_mm_p90"):
                        if _k in ctrl_diag:
                            range_stats[f"ctrl_{_k}"] = float(ctrl_diag[_k])
                    range_stats["ctrl_blend"] = float(blend)
                    # legacy 轴与 inverse 轴的逐元素差 —— 迁移完成后它仍然很大,
                    # 说明 "深度均匀 -> 逆深度均匀" 这个坐标变化本身影响不小,
                    # 是要记录的量, 不是故障。
                    range_stats["axis_l1_gap"] = float(
                        (hypos_inv - hypos_leg).abs().mean())
            else:
                hypos_k = hypos_leg
            # stage2 双模态 (legacy 路径; W2 的正式实现见工单, 需要 W0-B 判据通过)
            if k == 1 and self.range_cfg.dual_mode_stage2 and prev.get("second_depth") is not None:
                hypos_k = merge_dual_mode(
                    hypos_k, prev["second_depth"], prev["second_mass"],
                    prev["winner_interval"], self.range_cfg)
            hypos_k = self._upsample_hypos(hypos_k, feat_k.shape[-2:]).detach()

            meta_k = None
            if self.spre_cascade:
                # gamma_k * sg(r) / gamma_k * sg(e), 沿 D 广播 —— SPRE 贯穿四级
                hw_k = feat_k.shape[-2:]
                rk = self._resize_map(r_full, hw_k)
                ek = self._resize_map(s1.edge.detach(), hw_k)
                mk = torch.stack([gammas[k] * rk, gammas[k] * ek], dim=1)
                meta_k = mk.unsqueeze(2).expand(-1, -1, hypos_k.shape[1], -1, -1)

            depth_k, sigma_k, prob_k, logits_k, mode_idx_k, _ = self._run_stage(
                k, feat_k, K, E, hypos_k, strides[k], src_weights, meta=meta_k,
            )
            # 逐 bin 局部间距 —— 逆深度轴在深度空间非均匀, span/(D-1) 不再成立
            interval_k = local_intervals(hypos_k)
            wi_k = interval_k.gather(1, mode_idx_k).squeeze(1)
            with torch.no_grad():
                # 级联分辨率的直接读数。均值会被少数极宽像素拉走, 所以报分位数:
                # "stage4 是不是比 stage3 粗" 要看 p50/p90 而不是 mean。
                _iv = wi_k.flatten()[::97].float()
                if _iv.numel():
                    range_stats["interval_mm_p50"] = float(_iv.quantile(0.5))
                    range_stats["interval_mm_p90"] = float(_iv.quantile(0.9))
            range_stats["wbar_fallback_frac"] = float(prev.get("wbar_fb", 0.0))
            range_diag[f"stage{k + 1}"] = range_stats
            edge_k = self._resize_map(s1.edge, feat_k.shape[-2:])
            out_k = {
                "depth": depth_k, "sigma": sigma_k, "prob": prob_k,
                "logits": logits_k, "depth_hypos": hypos_k,
                "mode_idx": mode_idx_k, "edge": edge_k,
                "interval_local": interval_k,
            }
            if k == 3 and self.stage4_map:
                # 逐候选残差: 训练只监督离 GT 最近的那个 bin (见 losses/composite),
                # 推理才取 MAP 对应的那个。MAP 选错时不会污染残差监督。
                hw4 = feat_k.shape[-2:]
                nv4 = _n_valid_at(3, mode_idx_k)
                nv4 = torch.ones_like(depth_k) if nv4 is None else nv4
                extra = torch.stack([gammas[3] * self._resize_map(r_full, hw4), nv4], dim=1)
                res = self.res_head(prob_k, hypos_k, interval_k, mode_idx_k,
                                    sigma_k, feat_k[:, 0], extra)
                dd = interval_k.gather(1, mode_idx_k).squeeze(1)
                depth_k = depth_k + self.residual_scale * torch.tanh(
                    res.gather(1, mode_idx_k).squeeze(1)) * dd
                out_k["residual"] = res
                out_k["depth_map_only"] = out_k["depth"]
                out_k["depth"] = depth_k
            stage_out[f"stage{k + 1}"] = out_k
            if k == 3 and self.use_conf_head:
                # prev 还没被重新赋值, prev["depth"] 仍然是 stage3 的深度。
                conf_ctx = {
                    "hw": feat_k.shape[-2:], "ref_feat": feat_k[:, 0],
                    "depth3": prev["depth"], "depth4": out_k["depth"],
                    "prob": prob_k, "hypos": hypos_k, "interval": interval_k,
                    "mode_idx": mode_idx_k, "sigma": sigma_k,
                    "residual": out_k.get("residual"),
                    "n_valid": _n_valid_at(3, mode_idx_k),
                    "src_stats": getattr(self.cost_builders[3], "last_src_stats", None),
                }

            if k < 3:
                v_axis_k = 1.0 / hypos_k.float().clamp_min(1e-6)
                wbar_v_k, wbar_fb_k = robust_local_scale(
                    local_intervals(v_axis_k.flip(dims=[1]), min_interval=0.0).flip(dims=[1]),
                    mode_idx_k, g_v, return_fallback=True)
                h_sep_k = (float(self.range_cfg.range_k[k]) * wbar_v_k).clamp_min(1e-12)
                v2_k, mass2_k, dv2_k, found2_k, j2_k = second_mode_physical(
                    prob_k.float(), v_axis_k, mode_idx_k, h_sep_k, h_sep_k)
                bimodal_diag[f"stage{k + 1}"] = {"regress_window_bimodal_frac": float(
                    (found2_k & ((j2_k.squeeze(1) - mode_idx_k.squeeze(1)).abs()
                                 <= self.mode_windows[k])).float().mean())}
            else:
                wbar_v_k = torch.zeros_like(depth_k)
                mass2_k = dv2_k = torch.zeros_like(depth_k)
                found2_k = torch.zeros_like(depth_k, dtype=torch.bool)
                wbar_fb_k = 0.0
            prev = {
                "depth": depth_k, "prob": prob_k, "sigma": sigma_k,
                "winner_interval": wi_k, "mode_idx": mode_idx_k,
                "edge": edge_k, "hw": feat_k.shape[-2:], "second_depth": None,
                "w_bar_v": wbar_v_k, "mass2": mass2_k, "dv2": dv2_k, "found2": found2_k,
                "d_min": d_min_b, "d_span": d_span_b, "wbar_fb": wbar_fb_k,
                "n_valid": _n_valid_at(k, mode_idx_k),
            }

        depth_last = stage_out[f"stage{len(strides)}"]["depth"]
        depth_full = F.interpolate(
            depth_last.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        # ---------------- W3-C: 学出来的最终重建置信度 ----------------
        if self.use_conf_head and conf_ctx is not None:
            g_d = d_span_b / max(self.range_cfg.num_global - 1, 1)
            cf = self._fusion_conf_feats(
                stage_out, tuple(conf_ctx["hw"]), gammas[3].detach(), r_full, g_d,
                conf_ctx["n_valid"], conf_ctx["src_stats"], conf_ctx["prob"],
                conf_ctx["hypos"], conf_ctx["interval"], conf_ctx["mode_idx"],
                conf_ctx["sigma"], conf_ctx["depth4"], conf_ctx["depth3"],
                conf_ctx["residual"])
            rf = conf_ctx["ref_feat"]
            if self.conf_detach:
                # 自带监督的旁路: 学 "怎么读后验", 不许用 BCE 反过来重塑深度特征。
                cf, rf = cf.detach(), rf.detach()
            logit = self.conf_head(cf, rf)
            if logit.shape[-2:] != depth_full.shape[-2:]:
                logit = F.interpolate(logit.unsqueeze(1), size=depth_full.shape[-2:],
                                      mode="bilinear", align_corners=False).squeeze(1)
            stage_out["fusion_conf"] = {
                "logit": logit,
                "prob": self.conf_head.calibrated(logit.detach()),
                "T": float(self.conf_head.log_T.exp()),
            }

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
        vsup = getattr(self.cost_builders[0], "last_vis_sup", None)
        if vsup is not None:
            # 只交出 "我在哪、投到了哪儿"; 遮挡标签怎么造是损失侧的事 ——
            # 把源视图 GT 深度传进模型是一条没必要的耦合。
            vsup = dict(vsup)
            vsup["stride"] = int(strides[0])
            stage_out["vis_sup"] = vsup
        stage_out["range_diag"] = range_diag
        if range_ctrl_out:
            stage_out["range_ctrl"] = range_ctrl_out
        stage_out["spre_gamma"] = gammas
        if self.learn_range:
            # 五个尺度系数学到哪去了。eta -> 0 就说明全局尺度项其实没用。
            stage_out["range_diag"]["ctrl"] = self.range_ctrl.scalar_log()
        if self.geo_valid:
            agg = {}
            for i, m in enumerate(self.cost_builders):
                st = getattr(m, "last_n_valid_stats", None)
                if st:
                    agg.update({f"n_valid_{k}_s{i + 1}": v for k, v in st.items()})
            if agg:
                stage_out["range_diag"]["agg"] = agg
        # 逐 stage **合并**, 不是 update —— dict.update 会整个替换掉同名的
        # sub-dict, 把 refine_range_from_posterior 写进去的 range_stats 全冲掉。
        for _sname, _d in bimodal_diag.items():
            stage_out["range_diag"].setdefault(_sname, {}).update(_d)
        if self.cvpe is not None and self.cvpe.last_stats is not None:
            # 纯诊断, 不进损失。delta_rel 恒为 0 = CVPE 什么也没学到;
            # 远大于 1 = 它在覆盖 p8 而不是修正 p8, 两头都是故障信号。
            stage_out["range_diag"]["cvpe"] = dict(self.cvpe.last_stats)
        stage_out["range_diag"]["branch"] = {"bp_beta": float(getattr(self, "last_bp_beta", 0.0))}
        stage_out["range_diag"]["cost"] = {
            f"max_s{i + 1}": float(getattr(m, "last_cost_max", 0.0))
            for i, m in enumerate(self.cost_builders)
        }
        Probe.log("out", "depth_full", depth_full)
        return {"depth_full": depth_full, **stage_out}
