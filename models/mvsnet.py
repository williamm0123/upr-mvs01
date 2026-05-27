from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import (
    AnchorPEConfig,
    CostVolumeConfig,
    DecoderConfig,
    DepthRangeConfig,
    DINOConfig,
    FPNConfig,
    GeoFusionConfig,
    MVSConfig,
    VGGTPriorConfig,
)
from models.anchor_pe import (
    AnchorPositionalEncoder,
    compute_anchor_visibility,
    lambda_schedule,
    select_global_anchors,
)
from models.cost_volume import CostVolumeBuilder
from models.decoder import DepthDecoder
from models.depth_range import (
    fallback_global_range,
    initial_range_from_prior,
    refine_range_from_prob,
)
from models.dino_adapter import DINOv3Adapter
from models.fpn import MultiViewFPN
from models.geo_fusion import GatedGeoFusion
from models.vggt_prior import VGGTPrior
from utils.geometry import depth_to_normal, unproject_depth


class UprMVSNet(nn.Module):
    """End-to-end MVS network combining FPN + DINOv3 + VGGT prior + cascade cost volume."""

    def __init__(self, cfg: MVSConfig | None = None, device: torch.device | str = "cuda") -> None:
        super().__init__()
        self.cfg = cfg or MVSConfig()
        fpn_cfg: FPNConfig = self.cfg.fpn
        dino_cfg: DINOConfig = self.cfg.dino
        vggt_cfg: VGGTPriorConfig = self.cfg.vggt_prior
        geo_cfg: GeoFusionConfig = self.cfg.geo_fusion
        cv_cfg: CostVolumeConfig = self.cfg.cost_volume
        anchor_cfg: AnchorPEConfig = self.cfg.anchor_pe
        dec_cfg: DecoderConfig = self.cfg.decoder
        self.range_cfg: DepthRangeConfig = self.cfg.depth_range

        self.fpn = MultiViewFPN(fpn_cfg.backbone, fpn_cfg.out_channels, fpn_cfg.pretrained)
        self.dino = DINOv3Adapter(
            out_channels=dino_cfg.project_channels,
            max_side=dino_cfg.input_max_side,
            patch_size=dino_cfg.patch_size,
            layer_index=dino_cfg.layers[-1],
        )
        if self.cfg.train.use_vggt_prior and vggt_cfg.prior_source in ("online", "auto"):
            self.vggt_prior: VGGTPrior | None = VGGTPrior(vggt_cfg, device=device)
        else:
            self.vggt_prior = None

        self.fusion = GatedGeoFusion(
            rgb_channels=fpn_cfg.out_channels,
            geo_channels=geo_cfg.geo_channels,
            config=geo_cfg,
        ) if self.cfg.train.use_geo_fusion else None

        rgb_dim = fpn_cfg.out_channels + dino_cfg.project_channels
        self.merge_rgb = nn.Conv2d(rgb_dim, fpn_cfg.out_channels, 1)

        self.anchor_pe = AnchorPositionalEncoder(
            num_anchors=anchor_cfg.num_anchors,
            hidden=anchor_cfg.pe_hidden,
            out_channels=anchor_cfg.pe_out_channels,
        ) if self.cfg.train.use_anchor_pe else None
        if self.anchor_pe is not None:
            self.pe_fuse = nn.Conv2d(fpn_cfg.out_channels + anchor_cfg.pe_out_channels, fpn_cfg.out_channels, 1)

        self.cost_builder = CostVolumeBuilder(num_groups=cv_cfg.num_groups)
        self.decoder_stage1 = DepthDecoder(in_channels=cv_cfg.num_groups, base=dec_cfg.unet_base_channels, depth=dec_cfg.unet_depth)
        self.decoder_stage2 = DepthDecoder(in_channels=cv_cfg.num_groups, base=dec_cfg.unet_base_channels, depth=dec_cfg.unet_depth)
        self.decoder_stage3 = DepthDecoder(in_channels=cv_cfg.num_groups, base=dec_cfg.unet_base_channels, depth=dec_cfg.unet_depth)

        self.num_depths = (
            self.cfg.train.num_depths_stage1,
            self.cfg.train.num_depths_stage2,
            self.cfg.train.num_depths_stage3,
        )

    def _compute_view_features(
        self,
        imgs_norm: torch.Tensor,
        prior: dict[str, torch.Tensor] | None,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        step: int | None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        fpn_feats = self.fpn(imgs_norm)
        target_hw_p3 = fpn_feats[8].shape[-2:]
        dino_feat = self.dino(imgs_norm, target_hw=target_hw_p3)

        B, V, C_rgb, H, W = fpn_feats[8].shape
        merged_p3 = torch.cat([fpn_feats[8], dino_feat], dim=2)
        merged_p3 = self.merge_rgb(merged_p3.view(B * V, -1, H, W)).view(B, V, C_rgb, H, W)

        if self.fusion is not None and prior is not None:
            normals = self._compute_normals(prior["depth_sparse"], intrinsics)
            geo_feat = self.fusion.encode_geo(prior["depth_sparse"], normals, target_hw=(H, W))
            merged_p3 = self.fusion.fuse(merged_p3, geo_feat, prior["confidence"], step=step)

        return {4: fpn_feats[4], 8: merged_p3, 16: fpn_feats[16]}, dino_feat

    def _scale_intrinsics_to_hw(self, K: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = target_hw
        out = K.clone()
        out[..., 0, :] = out[..., 0, :] * (float(target_w) / float(self.cfg.data.target_w))
        out[..., 1, :] = out[..., 1, :] * (float(target_h) / float(self.cfg.data.target_h))
        return out

    def _compute_normals(self, depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        B, V, H, W = depth.shape
        d = depth.view(B * V, H, W)
        K = self._scale_intrinsics_to_hw(intrinsics, (H, W)).view(B * V, 3, 3)
        n = depth_to_normal(d, K).view(B, V, 3, H, W)
        return n

    def _world_points_from_depth(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        B, V, H, W = depth.shape
        K = self._scale_intrinsics_to_hw(intrinsics, (H, W)).view(B * V, 3, 3)
        E_inv = torch.inverse(extrinsics.view(B * V, 4, 4))
        world = unproject_depth(
            depth.view(B * V, H, W),
            torch.inverse(K),
            E_inv,
        )
        return world.view(B, V, 3, H, W).permute(0, 1, 3, 4, 2).contiguous()

    def _prepare_prior(
        self,
        prior: dict[str, torch.Tensor] | None,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if prior is None:
            return None
        depth = prior["depth_sparse"].float()
        confidence = prior.get("confidence", torch.ones_like(depth)).float().clamp(0.0, 1.0)
        valid = prior.get("valid_mask", confidence > 0).bool()
        valid = valid & torch.isfinite(depth) & (depth > 0) & torch.isfinite(confidence) & (confidence > 0)
        if not valid.any():
            return None
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
        world_points = self._world_points_from_depth(depth, intrinsics, extrinsics)
        return {
            "depth_sparse": depth,
            "confidence": confidence,
            "valid_mask": valid,
            "world_points": world_points,
        }

    def _maybe_apply_anchor_pe(
        self,
        ref_feat: torch.Tensor,
        prior: dict[str, torch.Tensor] | None,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        step: int | None,
    ) -> torch.Tensor:
        if self.anchor_pe is None or prior is None:
            return ref_feat
        B, C, H, W = ref_feat.shape
        anchors = select_global_anchors(
            prior["world_points"],
            prior["confidence"],
            prior["valid_mask"],
            num_anchors=self.anchor_pe.num_anchors,
            min_confidence=self.cfg.anchor_pe.min_confidence,
        )
        prior_h, prior_w = prior["depth_sparse"].shape[-2:]
        visibility = compute_anchor_visibility(anchors, intrinsics, extrinsics, image_hw=(prior_h, prior_w))
        prior_depth_ref = prior["depth_sparse"][:, 0]
        prior_depth_ref = F.interpolate(prior_depth_ref.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        K_ref = self._scale_intrinsics_to_hw(intrinsics[:, 0], (H, W))
        E_ref_inv = torch.inverse(extrinsics[:, 0])
        pe = self.anchor_pe(prior_depth_ref, K_ref, E_ref_inv, anchors, visibility[:, 0])
        lam = lambda_schedule(step or 0, self.cfg.anchor_pe)
        return self.pe_fuse(torch.cat([ref_feat, lam * pe], dim=1))

    def _scale_intrinsics(self, K: torch.Tensor, stride: int) -> torch.Tensor:
        out = K.clone()
        out[..., 0, :] = out[..., 0, :] / stride
        out[..., 1, :] = out[..., 1, :] / stride
        return out

    def _run_stage(
        self,
        feats: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        depth_hypos: torch.Tensor,
        feature_stride: int,
        decoder: DepthDecoder,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device_type = "cuda" if feats.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            feats_f = feats.float()
            K = intrinsics.float()
            E = extrinsics.float()
            depth_hypos_f = depth_hypos.float()
            ref_feat = feats_f[:, 0]
            src_feats = feats_f[:, 1:]
            cv = self.cost_builder(
                ref_feat,
                src_feats,
                K[:, 0],
                K[:, 1:],
                E[:, 0],
                E[:, 1:],
                depth_hypos_f,
                feature_stride=feature_stride,
            )
            depth, sigma, prob = decoder(cv, depth_hypos_f)
        return depth, sigma, prob

    def forward(
        self,
        batch: dict,
        step: int | None = None,
    ) -> dict[str, torch.Tensor]:
        imgs_norm = batch["imgs"]
        imgs_raw = batch["imgs_raw"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        depth_min = batch["depth_min"]
        depth_max = batch["depth_max"]
        prior = self._prepare_prior(batch.get("prior"), intrinsics, extrinsics)

        if prior is None and self.vggt_prior is not None:
            prior = self.vggt_prior(imgs_raw, intrinsics, extrinsics, depth_min, depth_max)
            prior = self._prepare_prior(prior, intrinsics, extrinsics)

        feats, dino_features = self._compute_view_features(imgs_norm, prior, intrinsics, extrinsics, step)

        target_hw = feats[8].shape[-2:]
        if prior is not None:
            depth_hypos1, half_range = initial_range_from_prior(
                prior["depth_sparse"][:, 0],
                prior["confidence"][:, 0],
                depth_min,
                depth_max,
                self.range_cfg,
                num_depths=self.num_depths[0],
                target_hw=target_hw,
            )
        else:
            depth_hypos1 = fallback_global_range(depth_min, depth_max, self.num_depths[0], target_hw)

        feat_p3 = feats[8]
        if self.anchor_pe is not None and prior is not None:
            B, V, C, H, W = feat_p3.shape
            ref_with_pe = self._maybe_apply_anchor_pe(feat_p3[:, 0], prior, intrinsics, extrinsics, step)
            feat_p3 = torch.cat([ref_with_pe.unsqueeze(1), feat_p3[:, 1:]], dim=1)

        K_s8 = self._scale_intrinsics(intrinsics, 1)
        depth1, sigma1, prob1 = self._run_stage(
            feat_p3,
            K_s8,
            extrinsics,
            depth_hypos1,
            feature_stride=8,
            decoder=self.decoder_stage1,
        )

        depth_hypos2 = refine_range_from_prob(
            prob1,
            depth_hypos1,
            depth1,
            self.range_cfg,
            num_depths=self.num_depths[1],
            interval_ratio=self.cfg.cost_volume.interval_ratio_stage2,
        )
        feat_p2 = feats[4]
        H2, W2 = feat_p2.shape[-2:]
        depth_hypos2_up = F.interpolate(depth_hypos2, size=(H2, W2), mode="bilinear", align_corners=False)
        depth2, sigma2, prob2 = self._run_stage(
            feat_p2,
            K_s8,
            extrinsics,
            depth_hypos2_up,
            feature_stride=4,
            decoder=self.decoder_stage2,
        )

        depth_hypos3 = refine_range_from_prob(
            prob2,
            depth_hypos2_up,
            depth2,
            self.range_cfg,
            num_depths=self.num_depths[2],
            interval_ratio=self.cfg.cost_volume.interval_ratio_stage3,
        )
        depth3, sigma3, prob3 = self._run_stage(
            feat_p2,
            K_s8,
            extrinsics,
            depth_hypos3,
            feature_stride=4,
            decoder=self.decoder_stage3,
        )

        depth_full = F.interpolate(
            depth3.unsqueeze(1), size=imgs_norm.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        return {
            "depth_full": depth_full,
            "stage1": {"depth": depth1, "sigma": sigma1, "prob": prob1, "hypos": depth_hypos1},
            "stage2": {"depth": depth2, "sigma": sigma2, "prob": prob2, "hypos": depth_hypos2_up},
            "stage3": {"depth": depth3, "sigma": sigma3, "prob": prob3, "hypos": depth_hypos3},
            "prior": prior,
            "dino_features": dino_features,
        }
