from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.depth_range import mode_centered_regression


class ConvGN3d(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UpConv3d(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, bias=False)
        self.norm = nn.GroupNorm(min(8, out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.up(x)))


class CostVolumeUNet(nn.Module):
    def __init__(self, in_channels: int = 8, base: int = 16, depth: int = 3) -> None:
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.input = ConvGN3d(in_channels, chs[0])
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(nn.Sequential(
                ConvGN3d(chs[i], chs[i + 1], stride=2),
                ConvGN3d(chs[i + 1], chs[i + 1]),
            ))
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(nn.ModuleList([
                UpConv3d(chs[i], chs[i - 1]),
                ConvGN3d(chs[i - 1] * 2, chs[i - 1]),
            ]))
        self.head = nn.Conv3d(chs[0], 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.input(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        feat = skips[-1]
        for i, (up, smooth) in enumerate(self.ups):
            feat = up(feat)
            skip = skips[-2 - i]
            if feat.shape != skip.shape:
                feat = F.interpolate(feat, size=skip.shape[2:], mode="trilinear", align_corners=False)
            feat = smooth(torch.cat([feat, skip], dim=1))
        return self.head(feat).squeeze(1)


class DepthDecoder(nn.Module):
    def __init__(self, in_channels: int = 8, base: int = 16, depth: int = 3, mode_window: int = 2) -> None:
        super().__init__()
        self.unet = CostVolumeUNet(in_channels=in_channels, base=base, depth=depth)
        self.mode_window = mode_window

    def forward(
        self,
        cost_volume: torch.Tensor,
        depth_hypos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.unet(cost_volume)
        # max-shift keeps softmax finite under AMP; log_softmax downstream is
        # shift-invariant so the loss sees the same distribution.
        logits = logits - logits.amax(dim=1, keepdim=True).detach()
        prob = F.softmax(logits.float(), dim=1)
        # Mode-centered regression instead of a global soft-argmin: over a
        # bimodal posterior (wrong local peak + correct global peak) the global
        # expectation lands between the peaks, on no real surface.
        depth, sigma, mode_idx = mode_centered_regression(prob, depth_hypos.float(), self.mode_window)
        return depth, sigma, prob, logits, mode_idx
