"""MoAMVSNet: DINO+FPN+SVA, pure-MVS stage 1, MoA-centred stages 2-4.

    images -> DinoSVA -> MultiViewFPN(out0) -> SVAPathway -> feats {8,4,2,1}
    stage 1: full-range uniform inverse-depth axis (no DA3), cost volume + 3D UNet
    for s in 2..4:
        centre_s = MoA_s(stage s-1 MVS evidence, DA3 depth/edge)   (or MVS centre if MoA off)
        window  = uniform in u around centre_s, wide enough to keep the MVS centre
        stage s: cost volume + 3D UNet on that window

MoA sets only where each stage searches; every stage's depth is still its own
MVS estimate. None of the old prior / SPRE / depth-range machinery is imported.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from base.config_moa import MoAMVSConfig
from models.decoder import DepthDecoder
from models.fpn import MultiViewFPN
from models.moa.cost_volume import MoACostVolume
from models.moa.edge import log_depth_edge
from models.moa.geometry import (
    axis_spacing, depth_to_u, inverse_bounds, stage1_u_hypotheses, u_to_depth,
    upsample_nearest, window_u_hypotheses,
)
from models.moa.moa import MoACascade
from models.spre import DinoSVA
from models.sva import SVAPathway


class MoAMVSNet(nn.Module):
    strides: tuple[int, int, int, int] = (8, 4, 2, 1)

    def __init__(self, cfg: MoAMVSConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MoAMVSConfig()
        cc = self.cfg.cascade
        if len(cc.num_depths) != 4 or len(cc.warp_channels) != 4:
            raise ValueError("cascade.num_depths / warp_channels need 4 entries")
        fpn_c = self.cfg.fpn.out_channels
        self.full_sva = bool(self.cfg.sva.full)

        self.dino_sva = DinoSVA(self.cfg.dino_fusion, self.cfg.dino, self.cfg.paths.dinov3_weights_file,
                                fpn_channels=fpn_c, sva_cfg=self.cfg.sva)
        self.fpn = MultiViewFPN(out_channels=fpn_c, base_channel=self.cfg.fpn.base_channel,
                                stage1_head="out0" if self.full_sva else "smooth")
        self.sva_pathway = None
        if self.full_sva:
            sva = self.cfg.sva
            self.sva_pathway = SVAPathway(
                fpn_c, heads=int(sva.hr_heads),
                layer_names=tuple(x.strip() for x in str(sva.hr_layers).split(",")),
                mlp_ratio=float(sva.hr_mlp_ratio), pe_max_shape=tuple(sva.pe_max_shape),
                strides=self.strides)

        self.cost_volumes = nn.ModuleList(
            MoACostVolume(fpn_c, wc, cc.num_groups, cc.warp_use_half) for wc in cc.warp_channels)
        self.decoders = nn.ModuleList(
            DepthDecoder(in_channels=cc.num_groups + 1, base=cc.unet_base_channels,
                         depth=cc.unet_depth, mode_window=mw, head_mode="expect")
            for mw in cc.mode_windows)
        for i, d in enumerate(self.decoders):
            d.tag = f"stage{i + 1}"
        self.moa = (MoACascade(self.cfg.moa, fpn_c, cc.num_groups, cc.num_depths[0])
                    if self.cfg.moa.enabled else None)

    @property
    def uses_mono(self) -> bool:
        return self.moa is not None

    def _features(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        coarse_hw = (images.shape[-2] // self.strides[0], images.shape[-1] // self.strides[0])
        fused, _, grid = self.dino_sva(images)
        dino_fpn = self.dino_sva.fpn_feature(fused, grid, coarse_hw)
        feats = self.fpn(images, dino=dino_fpn)
        if self.sva_pathway is not None:
            feats = self.sva_pathway(feats)
        return feats

    def _run_stage(self, k: int, feat: torch.Tensor, K: torch.Tensor, E: torch.Tensor,
                   u_hyp: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> dict:
        hypos = u_to_depth(u_hyp, vmin, vmax)
        cvo = self.cost_volumes[k](feat[:, 0], feat[:, 1:], K[:, 0], K[:, 1:], E[:, 0], E[:, 1:],
                                   hypos, feature_stride=self.strides[k])
        depth, sigma, prob, logits, mode_idx, _ = self.decoders[k](cvo.decoder_input(), hypos)
        return {"depth": depth.unsqueeze(1), "sigma": sigma.unsqueeze(1), "prob": prob,
                "logits": logits, "mode_idx": mode_idx, "depth_hypos": hypos, "u_hypos": u_hyp,
                "_cv": cvo}

    def forward(self, batch: dict) -> dict:
        images = batch["images"].float() / 255.0
        K = batch["intrinsics"].float()
        E = batch["extrinsics"].float()
        B = images.shape[0]
        vmin, vmax = inverse_bounds(batch["depth_values"])
        feats = self._features(images)

        mono = None
        if self.moa is not None:
            if "mono_depth" not in batch:
                raise KeyError("MoA is enabled but the batch has no 'mono_depth' (DA3 cache)")
            md = batch["mono_depth"].float().unsqueeze(1)
            mv = torch.isfinite(md) & (md > 0)
            md = torch.where(mv, md, torch.ones_like(md))
            mono = {"depth": md, "valid": mv.float(), "edge": log_depth_edge(md, mv)}

        cc = self.cfg.cascade
        f1 = feats[self.strides[0]]
        u1 = stage1_u_hypotheses(B, cc.num_depths[0], tuple(f1.shape[-2:]), f1.device)
        prev = self._run_stage(0, f1, K, E, u1, vmin, vmax)
        out = {"stage1": prev, "vmin": vmin, "vmax": vmax}

        for k in (1, 2, 3):
            feat = feats[self.strides[k]]
            du = axis_spacing(prev["u_hypos"])
            y = depth_to_u(prev["depth"], vmin, vmax).detach()
            cvo = prev.pop("_cv")
            if self.moa is not None:
                mo = self.moa(
                    k - 1, mono_depth=mono["depth"], mono_valid=mono["valid"], mono_edge=mono["edge"],
                    z_prev=prev["depth"].detach(), prob=prev["prob"].detach(),
                    u_hyp=prev["u_hypos"].detach(), cv=cvo.cv.detach(), n_valid=cvo.n_valid.detach(),
                    src_std=cvo.src_std.detach(), num_src=cvo.num_src,
                    ref_feat=feats[self.strides[k - 1]][:, 0].detach(), vmin=vmin, vmax=vmax)
                out[f"moa{k + 1}"] = mo
                center = mo.center_u.detach()
            else:
                center = y
            h_base = float(cc.window_halfwidth_bins[k - 1]) * du
            h_keep = float(cc.keep_margin_bins[k - 1]) * du
            half = torch.maximum(h_base, (center - y).abs() + h_keep)
            hw = tuple(feat.shape[-2:])
            u_k = window_u_hypotheses(upsample_nearest(center, hw), upsample_nearest(half, hw),
                                      cc.num_depths[k]).detach()
            del cvo
            prev = self._run_stage(k, feat, K, E, u_k, vmin, vmax)
            out[f"stage{k + 1}"] = prev
        prev.pop("_cv")
        out["depth_full"] = out["stage4"]["depth"][:, 0]
        return out
