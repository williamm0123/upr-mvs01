from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import MVSConfig
from models.cost_volume import CostVolumeBuilder
from models.decoder import DepthDecoder
from models.depth_range import (
    Stage1Hypotheses,
    build_stage1_hypotheses,
    refine_range_from_posterior,
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
            CostVolumeBuilder(fpn_c, wc, cv_cfg.num_groups, cv_cfg.warp_use_half)
            for wc in warp_chs
        ])
        # Per-stage 3D-UNet decoders. Stage 1 additionally sees the hypothesis
        # metadata channels (its axis is irregular); stages 2-4 use uniform
        # axes and need none.
        mw = self.range_cfg.mode_window
        mr = self.range_cfg.mode_radius_mm
        self.decoders = nn.ModuleList(
            [DepthDecoder(
                in_channels=cv_cfg.num_groups + cv_cfg.stage1_meta_channels,
                base=dec_cfg.unet_base_channels, depth=dec_cfg.unet_depth,
                mode_window=mw, mode_radius_mm=mr,
            )]
            + [DepthDecoder(in_channels=cv_cfg.num_groups, base=dec_cfg.unet_base_channels,
                            depth=dec_cfg.unet_depth, mode_window=mw, mode_radius_mm=mr)
               for _ in range(3)]
        )

        self.num_depths = (cv_cfg.num_depths_stage1, cv_cfg.num_depths_stage2,
                           cv_cfg.num_depths_stage3, cv_cfg.num_depths_stage4)

        # Optional DINOv3 path. One frozen backbone + SVA fusion feeds two
        # consumers: the FPN bottleneck (per-view matching features) and the
        # SPRE reliability head (reference view). Off by default; when off the
        # network is byte-for-byte the previous model and DINOv3 never loads.
        self.spre_enabled = bool(self.cfg.spre.enabled)
        self.feed_fpn = self.spre_enabled and bool(self.cfg.dino.feed_fpn)
        if self.spre_enabled:
            from models.spre import SPRE, DinoSVA
            self.dino_sva = DinoSVA(
                self.cfg.spre, self.cfg.dino, self.cfg.paths.dinov3_weights_file,
                fpn_channels=fpn_c if self.feed_fpn else None,
            )
            self.spre = SPRE(self.cfg.spre, self.dino_sva.out_dim)

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
    def _stage1_meta(s1: Stage1Hypotheses, global_interval: torch.Tensor) -> torch.Tensor:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return self.decoders[stage_idx](cost, depth_hypos)

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
        depth_min, depth_max = self._resolve_depth_bounds(batch)
        src_weights = self._resolve_src_weights(batch)

        strides = self.fpn_stage_strides
        s1_stride = strides[0]
        coarse_hw = (images.shape[-2] // s1_stride, images.shape[-1] // s1_stride)

        # DINOv3 + SVA runs before the FPN so its per-view features can be
        # injected at the 1/8 bottleneck and propagate down the top-down path.
        fused = raw_tokens = grid = None
        if self.spre_enabled:
            fused, raw_tokens, grid = self.dino_sva(images)   # [B, V, N, dim], [B, V, N, 768]

        dino_fpn = (self.dino_sva.fpn_feature(fused, grid, coarse_hw)
                    if self.feed_fpn else None)
        feats = self.fpn(images, dino=dino_fpn)  # {8: [B,V,C,h,w], 4: ..., 2: ..., 1: ...}

        # ---------- Stage 1 (coarsest, 1/8): dual-branch hypotheses ----------
        feat1 = feats[s1_stride]
        # SPRE: learned reliability from the fused reference tokens + online
        # prior stats replaces the cached conf (which is degenerate). Computed at
        # the stage-1 resolution so the hypothesis builder's resize is a no-op.
        if self.spre_enabled:
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
        )
        global_interval = (s1.global_hi - s1.global_lo) / max(self.range_cfg.num_global - 1, 1)  # [B]
        meta1 = self._stage1_meta(s1, global_interval)
        depth1, sigma1, prob1, logits1, mode_idx1 = self._run_stage(
            0, feat1, K, E, s1.hypos, s1_stride, src_weights, meta=meta1
        )

        stage_out = {
            "stage1": {
                "depth": depth1, "sigma": sigma1, "prob": prob1,
                "logits": logits1, "depth_hypos": s1.hypos,
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
            },
        }

        # ---------- Stages 2-4: range from the winning candidate ----------
        # winner's sampling interval decides how much correction room the next
        # stage keeps: local win -> narrow, global win -> wide.
        winner_interval = s1.interval.gather(1, mode_idx1).squeeze(1)
        prev = {
            "depth": depth1, "prob": prob1, "winner_interval": winner_interval,
            "edge": s1.edge, "hw": feat1.shape[-2:],
        }
        for k in (1, 2, 3):
            feat_k = feats[strides[k]]
            hypos_k = refine_range_from_posterior(
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
            hypos_k = self._upsample_hypos(hypos_k, feat_k.shape[-2:])
            depth_k, sigma_k, prob_k, logits_k, mode_idx_k = self._run_stage(
                k, feat_k, K, E, hypos_k, strides[k], src_weights
            )
            edge_k = self._resize_map(s1.edge, feat_k.shape[-2:])
            # The interval that PLACED this stage's window, resampled to this
            # stage's resolution. The loss normalises the regression by it
            # instead of by this stage's own bin, so that changing num_depths or
            # range_k cannot silently reweight the stage (losses/composite.py).
            parent_interval = self._resize_map(prev["winner_interval"], feat_k.shape[-2:])
            stage_out[f"stage{k + 1}"] = {
                "depth": depth_k, "sigma": sigma_k, "prob": prob_k,
                "logits": logits_k, "depth_hypos": hypos_k,
                "mode_idx": mode_idx_k, "edge": edge_k,
                "parent_interval": parent_interval,
            }
            # uniform axis: per-pixel interval = span / (D - 1)
            interval_k = (hypos_k[:, -1] - hypos_k[:, 0]) / max(self.num_depths[k] - 1, 1)
            prev = {
                "depth": depth_k, "prob": prob_k, "winner_interval": interval_k,
                "edge": edge_k, "hw": feat_k.shape[-2:],
            }

        depth_last = stage_out[f"stage{len(strides)}"]["depth"]
        depth_full = F.interpolate(
            depth_last.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        if self.spre_enabled:
            stage_out["spre"] = {
                "logits": spre_logits, "r": conf_used, "hw": tuple(feat1.shape[-2:]),
            }

        return {"depth_full": depth_full, **stage_out}
