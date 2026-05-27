from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.geometry import reproject_with_depth


def feature_cosine_loss(
    depth_pred: torch.Tensor,
    dino_features: torch.Tensor,
    K: torch.Tensor,
    E: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    B, V, C, H, W = dino_features.shape
    full_h, full_w = mask.shape[-2:]
    if depth_pred.shape[-2:] != (H, W):
        depth_pred = F.interpolate(
            depth_pred.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)
    if mask.shape[-2:] != (H, W):
        mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1)
    K_s = K.clone()
    K_s[..., 0, :] = K_s[..., 0, :] * (float(W) / float(full_w))
    K_s[..., 1, :] = K_s[..., 1, :] * (float(H) / float(full_h))
    ref = dino_features[:, 0]
    total = depth_pred.new_zeros(())
    cnt = 0
    for s in range(1, V):
        uv = reproject_with_depth(depth_pred, K_s[:, 0], E[:, 0], K_s[:, s], E[:, s])
        uv_x = uv[:, 0] / (W - 1) * 2.0 - 1.0
        uv_y = uv[:, 1] / (H - 1) * 2.0 - 1.0
        valid_src = (
            torch.isfinite(uv_x)
            & torch.isfinite(uv_y)
            & (uv_x >= -1.0)
            & (uv_x <= 1.0)
            & (uv_y >= -1.0)
            & (uv_y <= 1.0)
        )
        grid = torch.stack([uv_x, uv_y], dim=-1)
        warped = F.grid_sample(dino_features[:, s], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        cos = (F.normalize(ref, dim=1) * F.normalize(warped, dim=1)).sum(dim=1)
        m = mask.bool() & (depth_pred > 0) & valid_src
        if m.any():
            total = total + (1.0 - cos)[m].mean()
            cnt += 1
    if cnt == 0:
        return total
    return total / cnt
