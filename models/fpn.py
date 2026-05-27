from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class FPNFeatureExtractor(nn.Module):
    """ResNet-50 FPN that produces 1/4, 1/8, 1/16 feature maps."""

    def __init__(self, backbone: str = "resnet50", out_channels: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise NotImplementedError(f"backbone={backbone} not supported")
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = tv_models.resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        c2, c3, c4 = 256, 512, 1024
        self.lateral_c2 = nn.Conv2d(c2, out_channels, 1)
        self.lateral_c3 = nn.Conv2d(c3, out_channels, 1)
        self.lateral_c4 = nn.Conv2d(c4, out_channels, 1)
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        return {
            4: self.smooth_p2(p2),
            8: self.smooth_p3(p3),
            16: self.smooth_p4(p4),
        }


class MultiViewFPN(nn.Module):
    def __init__(self, backbone: str = "resnet50", out_channels: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        self.fpn = FPNFeatureExtractor(backbone, out_channels, pretrained)
        self.out_channels = out_channels

    def forward(self, imgs: torch.Tensor) -> dict[int, torch.Tensor]:
        B, V, C, H, W = imgs.shape
        feats = self.fpn(imgs.view(B * V, C, H, W))
        return {s: f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for s, f in feats.items()}
