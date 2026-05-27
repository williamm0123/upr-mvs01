from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.geometry import reproject_with_depth


def _ssim(x: torch.Tensor, y: torch.Tensor, window: int = 7) -> torch.Tensor:
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, 1, pad)
    mu_y = F.avg_pool2d(y, window, 1, pad)
    sigma_x = (F.avg_pool2d(x * x, window, 1, pad) - mu_x * mu_x).clamp(min=0.0)
    sigma_y = (F.avg_pool2d(y * y, window, 1, pad) - mu_y * mu_y).clamp(min=0.0)
    sigma_xy = F.avg_pool2d(x * y, window, 1, pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return torch.clamp((1 - n / d.clamp(min=1e-8)) * 0.5, 0, 1)


def ssim_reprojection_loss(
    depth_pred: torch.Tensor,
    imgs: torch.Tensor,
    K: torch.Tensor,
    E: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    B, V, C, H, W = imgs.shape
    if depth_pred.shape[-2:] != (H, W):
        depth_pred = F.interpolate(
            depth_pred.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)
    img_ref = imgs[:, 0]
    total = depth_pred.new_zeros(())
    cnt = 0
    for s in range(1, V):
        uv = reproject_with_depth(depth_pred, K[:, 0], E[:, 0], K[:, s], E[:, s])
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
        warped = F.grid_sample(
            imgs[:, s],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        s_val = _ssim(img_ref, warped)
        m = mask.bool().unsqueeze(1) & (depth_pred.unsqueeze(1) > 0) & valid_src.unsqueeze(1)
        if m.any():
            total = total + s_val[m.expand_as(s_val)].mean()
            cnt += 1
    if cnt == 0:
        return total
    return total / cnt
