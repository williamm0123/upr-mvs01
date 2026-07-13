from __future__ import annotations

import torch
import torch.nn as nn

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
    ) -> None:
        super().__init__()
        if warp_channels % num_groups != 0:
            raise ValueError(
                f"warp_channels {warp_channels} must be divisible by num_groups {num_groups}"
            )
        self.proj = nn.Conv2d(in_channels, warp_channels, kernel_size=1, bias=False)
        self.warp_channels = warp_channels
        self.num_groups = num_groups
        self.use_half = use_half

    def _sample_dtype(self, ref: torch.Tensor) -> torch.dtype:
        return torch.float16 if (self.use_half and ref.is_cuda) else ref.dtype

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

        agg = ref_feat.new_zeros(B, self.num_groups, D, H, W, dtype=torch.float32)
        weight_sum = ref_feat.new_zeros(B, 1, 1, 1, 1, dtype=torch.float32)
        for s in range(S):
            src_p = self.proj(src_feats[:, s]).to(sample_dtype)
            warped = homography_warp_features(
                src_p,
                K_ref,
                K_src[:, s],
                E_ref,
                E_src[:, s],
                depth_hypos,
                feature_stride,
            )
            cv_s = group_wise_correlation(ref_p, warped, self.num_groups).float()
            w = (
                src_weights[:, s].view(B, 1, 1, 1, 1).float()
                if src_weights is not None
                else ref_feat.new_ones(B, 1, 1, 1, 1, dtype=torch.float32)
            )
            agg = agg + cv_s * w
            weight_sum = weight_sum + w
        return agg / weight_sum.clamp(min=1e-6)
