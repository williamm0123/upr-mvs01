from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import MVSConfig
from models.cost_volume import CostVolumeBuilder
from models.decoder import DepthDecoder
from models.depth_range import initial_range_from_prior, refine_range_from_prob
from models.fpn import MultiViewFPN


class UprMVSNet(nn.Module):
    """End-to-end cascade MVS network: FPN -> 3-stage cost volume + 3D-UNet.

    Data flow
    ---------
        images ──► FPN ──► multi-scale features {1/4, 1/2, 1}
        depth_prior + conf_prior (norm_fill) ──► depth_range ──► stage-1 hypotheses
        for each stage k:
            ref/src features + cameras + hypotheses ──► CostVolumeBuilder ──► cost volume
            cost volume ──► DepthDecoder (3D UNet) ──► prob volume ──► soft-argmin ──► depth
            depth + prob ──► refine_range_from_prob ──► next-stage hypotheses (upsampled)

    Stage resolutions follow ``fpn_stage_strides`` = (4, 2, 1): 1/4 -> 1/2 -> full.

    Expected ``batch`` keys
    -----------------------
        images      [B, V, 3, H, W]   view 0 is the reference
        intrinsics  [B, V, 3, 3]      DTU metric cameras at image resolution
        extrinsics  [B, V, 4, 4]      (same metric frame as depth_prior)
        depth_prior [B, H, W]         ref metric depth   (norm_fill["depth_filled"])
        conf_prior  [B, H, W]         ref confidence      (norm_fill["conf_map"])
        depth_values [B, D]           ref metric depth range (dtu["depth_values"]);
                                      min/max derived from it. Optional explicit
                                      depth_min/depth_max [B] override it.
        src_weights [B, V-1]          optional per-source cost-volume weights
                                      (used only when cfg.cost_volume.use_src_weights)
    """

    # FPN feature strides for the three cascade stages (coarse -> fine).
    fpn_stage_strides: tuple[int, int, int] = (4, 2, 1)

    def __init__(self, cfg: MVSConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MVSConfig()
        cv_cfg = self.cfg.cost_volume
        fpn_cfg = self.cfg.fpn
        dec_cfg = self.cfg.decoder
        self.range_cfg = self.cfg.depth_range

        self.fpn = MultiViewFPN(
            out_channels=fpn_cfg.out_channels,
            base_channel=fpn_cfg.base_channel,
        )
        fpn_c = fpn_cfg.out_channels

        # Per-stage cost-volume builders: warp-channel width shrinks as
        # resolution grows so the full-res stage does not OOM.
        self.cost_builders = nn.ModuleList([
            CostVolumeBuilder(fpn_c, cv_cfg.warp_channels_stage1, cv_cfg.num_groups, cv_cfg.warp_use_half),
            CostVolumeBuilder(fpn_c, cv_cfg.warp_channels_stage2, cv_cfg.num_groups, cv_cfg.warp_use_half),
            CostVolumeBuilder(fpn_c, cv_cfg.warp_channels_stage3, cv_cfg.num_groups, cv_cfg.warp_use_half),
        ])
        # Per-stage 3D-UNet decoders: cost volume [B, G, D, H, W] -> prob volume.
        self.decoders = nn.ModuleList([
            DepthDecoder(in_channels=cv_cfg.num_groups, base=dec_cfg.unet_base_channels, depth=dec_cfg.unet_depth)
            for _ in range(3)
        ])

        self.num_depths = (cv_cfg.num_depths_stage1, cv_cfg.num_depths_stage2, cv_cfg.num_depths_stage3)
        self.interval_ratios = (cv_cfg.interval_ratio_stage2, cv_cfg.interval_ratio_stage3)

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

    def _run_stage(
        self,
        stage_idx: int,
        feats_stage: torch.Tensor,
        K: torch.Tensor,
        E: torch.Tensor,
        depth_hypos: torch.Tensor,
        feature_stride: int,
        src_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        depth, sigma, prob = self.decoders[stage_idx](cost, depth_hypos)
        return depth, sigma, prob

    @staticmethod
    def _upsample_hypos(hypos: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if tuple(hypos.shape[-2:]) == tuple(target_hw):
            return hypos
        return F.interpolate(hypos, size=target_hw, mode="bilinear", align_corners=False)

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
        depth_min, depth_max = self._resolve_depth_bounds(batch)
        src_weights = self._resolve_src_weights(batch)

        feats = self.fpn(images)  # {4: [B,V,C,h,w], 2: ..., 1: ...}
        s1, s2, s3 = self.fpn_stage_strides
        strides = (s1, s2, s3)

        # ---------- Stage 1 (coarsest, 1/4): hypotheses from the prior ----------
        feat1 = feats[s1]
        depth_hypos1, _ = initial_range_from_prior(
            depth_prior,
            conf_prior,
            depth_min,
            depth_max,
            self.range_cfg,
            num_depths=self.num_depths[0],
            target_hw=feat1.shape[-2:],
        )
        depth1, sigma1, prob1 = self._run_stage(0, feat1, K, E, depth_hypos1, s1, src_weights)

        # ---------- Stages 2 & 3: refine range from the previous prob volume ----------
        stage_out = {
            "stage1": {"depth": depth1, "sigma": sigma1, "prob": prob1, "depth_hypos": depth_hypos1},
        }
        prev_prob, prev_hypos, prev_depth = prob1, depth_hypos1, depth1
        for k in (1, 2):
            feat_k = feats[strides[k]]
            hypos_k = refine_range_from_prob(
                prev_prob,
                prev_hypos,
                prev_depth,
                self.range_cfg,
                num_depths=self.num_depths[k],
                interval_ratio=self.interval_ratios[k - 1],
            )
            hypos_k = self._upsample_hypos(hypos_k, feat_k.shape[-2:])
            # refine_range_from_prob recenters on the previous prediction and, when
            # uncertain, re-expands the half-range; without a clamp the hypotheses can
            # drift below depth_min (even negative), which makes the warp project to
            # degenerate pixels. Keep them inside the scene's valid depth range.
            hypos_k = hypos_k.clamp(
                min=depth_min.view(-1, 1, 1, 1),
                max=depth_max.view(-1, 1, 1, 1),
            )
            depth_k, sigma_k, prob_k = self._run_stage(k, feat_k, K, E, hypos_k, strides[k], src_weights)
            stage_out[f"stage{k + 1}"] = {
                "depth": depth_k, "sigma": sigma_k, "prob": prob_k, "depth_hypos": hypos_k,
            }
            prev_prob, prev_hypos, prev_depth = prob_k, hypos_k, depth_k

        depth3 = stage_out["stage3"]["depth"]
        depth_full = F.interpolate(
            depth3.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        return {"depth_full": depth_full, **stage_out}
