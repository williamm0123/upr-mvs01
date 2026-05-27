from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import DINOConfig, ProjectPaths
from models.dinov3.extractor import (
    compute_patch_aligned_size,
    dino_normalization_tensors,
    load_dinov3_vit_base,
)


class DINOv3Adapter(nn.Module):
    """Frozen DINOv3 backbone + trainable MLP adapter to FPN feature dim."""

    def __init__(
        self,
        out_channels: int = 128,
        max_side: int = 512,
        patch_size: int = 16,
        layer_index: int = -1,
        weights_file: str | Path | None = None,
        in_channels: int = 768,
    ) -> None:
        super().__init__()
        paths = ProjectPaths()
        weights_file = Path(weights_file) if weights_file is not None else paths.dinov3_weights_file
        self.backbone = load_dinov3_vit_base(device="cpu", weights_file=weights_file, patch_size=patch_size)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.max_side = max_side
        self.patch_size = patch_size
        self.layer_index = layer_index
        self.adapter = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_channels),
        )
        self.out_channels = out_channels
        self.config = DINOConfig()

    def _prepare_input(self, imgs_norm: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        BV, _, H, W = imgs_norm.shape
        in_h, in_w = compute_patch_aligned_size(H, W, max_side=self.max_side, patch_size=self.patch_size)
        device = imgs_norm.device
        mean = torch.tensor(self.config.mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.config.std, device=device).view(1, 3, 1, 1)
        denorm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        denorm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        imgs_raw = imgs_norm * denorm_std + denorm_mean
        resized = F.interpolate(imgs_raw, size=(in_h, in_w), mode="bilinear", align_corners=False)
        normed = (resized - mean) / std
        return normed, (in_h, in_w)

    def forward(self, imgs_norm: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        B, V, C, H, W = imgs_norm.shape
        flat = imgs_norm.view(B * V, C, H, W)
        dino_input, _ = self._prepare_input(flat)
        with torch.no_grad():
            feats = self.backbone.get_intermediate_layers(
                dino_input,
                n=[self.layer_index] if self.layer_index >= 0 else 1,
                reshape=True,
                norm=True,
            )
        feat = feats[-1].float()
        BV2, C2, Hf, Wf = feat.shape
        flat_tok = feat.permute(0, 2, 3, 1).reshape(BV2 * Hf * Wf, C2)
        projected = self.adapter(flat_tok).view(BV2, Hf, Wf, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(projected, size=target_hw, mode="bilinear", align_corners=False)
        out = F.normalize(out, p=2, dim=1)
        return out.view(B, V, -1, target_hw[0], target_hw[1])
