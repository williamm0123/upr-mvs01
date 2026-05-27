from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import GeoFusionConfig


class GeometryEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, hidden: int = 64, out_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )

    def forward(self, sparse_input: torch.Tensor) -> torch.Tensor:
        return self.net(sparse_input)


class GatedGeoFusion(nn.Module):
    """F_fused = F_rgb + alpha * C * F_geo (per-pixel)."""

    def __init__(self, rgb_channels: int = 128, geo_channels: int = 128, config: GeoFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeoFusionConfig()
        self.geo_encoder = GeometryEncoder(in_channels=4, hidden=self.config.encoder_hidden, out_channels=geo_channels)
        self.alpha = nn.Parameter(torch.tensor(self.config.init_alpha, dtype=torch.float32))
        if rgb_channels != geo_channels:
            self.proj = nn.Conv2d(geo_channels, rgb_channels, 1)
        else:
            self.proj = nn.Identity()

    def _alpha_value(self, step: int | None) -> torch.Tensor:
        if step is None:
            return self.alpha
        cfg = self.config
        if step < cfg.alpha_warmup_steps:
            return torch.clamp(self.alpha, -cfg.alpha_max_during_warmup, cfg.alpha_max_during_warmup)
        if step < cfg.alpha_warmup_steps + cfg.alpha_release_steps:
            frac = (step - cfg.alpha_warmup_steps) / max(cfg.alpha_release_steps, 1)
            bound = cfg.alpha_max_during_warmup + frac * (1.0 - cfg.alpha_max_during_warmup)
            return torch.clamp(self.alpha, -bound, bound)
        return self.alpha

    def encode_geo(
        self,
        sparse_depth: torch.Tensor,
        normals: torch.Tensor | None,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        B, V, H, W = sparse_depth.shape
        if normals is None:
            normals = torch.zeros(B, V, 3, H, W, device=sparse_depth.device, dtype=sparse_depth.dtype)
        sd = sparse_depth.unsqueeze(2)
        sd_filled = torch.where(torch.isfinite(sd) & (sd > 0), sd, torch.zeros_like(sd))
        stacked = torch.cat([sd_filled, normals], dim=2)
        flat = stacked.view(B * V, 4, H, W)
        flat = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
        feat = self.geo_encoder(flat)
        return feat.view(B, V, -1, target_hw[0], target_hw[1])

    def fuse(
        self,
        rgb_features: torch.Tensor,
        geo_features: torch.Tensor,
        confidence: torch.Tensor,
        step: int | None = None,
    ) -> torch.Tensor:
        B, V, C, H, W = rgb_features.shape
        flat_geo = geo_features.view(B * V, geo_features.shape[2], H, W)
        flat_geo = self.proj(flat_geo).view(B, V, C, H, W)
        conf_resized = F.interpolate(confidence.view(B * V, 1, *confidence.shape[-2:]), size=(H, W), mode="bilinear", align_corners=False)
        conf_resized = conf_resized.view(B, V, 1, H, W)
        a = self._alpha_value(step)
        return rgb_features + a * conf_resized * flat_geo
