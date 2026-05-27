from __future__ import annotations

import torch
import torch.nn as nn

from utils.geometry import homography_warp_features


def group_wise_correlation(ref: torch.Tensor, warped: torch.Tensor, num_groups: int) -> torch.Tensor:
    B, C, D, H, W = warped.shape
    assert C % num_groups == 0, f"channels {C} not divisible by groups {num_groups}"
    g = C // num_groups
    ref_d = ref.unsqueeze(2)
    ref_g = ref_d.view(B, num_groups, g, 1, H, W)
    warp_g = warped.view(B, num_groups, g, D, H, W)
    return (ref_g * warp_g).mean(dim=2)


class CostVolumeBuilder(nn.Module):
    def __init__(self, num_groups: int = 8) -> None:
        super().__init__()
        self.num_groups = num_groups

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
        B, V_minus_1, C, H, W = src_feats.shape
        D = depth_hypos.shape[1]
        agg = ref_feat.new_zeros(B, self.num_groups, D, H, W)
        weight_sum = ref_feat.new_zeros(B, 1, 1, 1, 1)
        for s in range(V_minus_1):
            warped = homography_warp_features(
                src_feats[:, s],
                K_ref,
                K_src[:, s],
                E_ref,
                E_src[:, s],
                depth_hypos,
                feature_stride,
            )
            cv_s = group_wise_correlation(ref_feat, warped, self.num_groups)
            if src_weights is None:
                w = ref_feat.new_ones(B, 1, 1, 1, 1)
            else:
                w = src_weights[:, s].view(B, 1, 1, 1, 1)
            agg = agg + cv_s * w
            weight_sum = weight_sum + w
        return agg / weight_sum.clamp(min=1e-6)
