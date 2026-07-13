from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) # 注意这里换成了 nn.ReLU 模块
        )
    def forward(self, x):
        return self.conv_bn_relu(x)


class ResidualBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convbn1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.convbn2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convbn1(x)
        out = self.convbn2(out)
        return F.relu(out + x, inplace=True)


class FPNFeatureExtractor(nn.Module):
    """从头训练的特征塔 + FPN，产出 1/4、1/2、全分辨率三级特征。

    自底向上：原图 -> 1/2 尺度特征塔 -> 1/4 尺度特征塔（每个尺度第 1 层 stride=2 降采样）。
    自顶向下：p4(1/4) -> 上采样加到 p2(1/2) -> 上采样加到 p1(全分辨率)。
    """

    def __init__(
        self,
        out_channels: int = 128,
        base_channel: int = 32,
    ) -> None:
        super().__init__()
        c_half = base_channel * 2      # 1/2 尺度内部通道
        c_quarter = base_channel * 4   # 1/4 尺度内部通道

        # ---- 自底向上：特征塔（每尺度第 1 层 stride=2 降采样并换通道，同尺度用残差块增强） ----
        self.half_1 = ConvBnReLU(3, c_half, stride=2)        # 原图 -> 1/2
        self.half_res = ResidualBlock(c_half)                # 1/2 同尺度残差增强
        self.quarter_1 = ConvBnReLU(c_half, c_quarter, stride=2)  # 1/2 -> 1/4
        self.quarter_res = ResidualBlock(c_quarter)          # 1/4 同尺度残差增强

        # ---- FPN 横向连接：统一到 out_channels 才能逐元素相加 ----
        self.input_proj = nn.Conv2d(3, out_channels, 3, padding=1)   # 全分辨率分支(原图直接投影)
        self.lateral_half = nn.Conv2d(c_half, out_channels, 1)
        self.lateral_quarter = nn.Conv2d(c_quarter, out_channels, 1)

        # ---- 输出平滑：消除上采样混叠 ----
        self.smooth_p1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        # 自底向上：1/2 尺度
        c2 = self.half_1(x)              # [B, c_half, H/2, W/2] (stride=2 降采样)
        f_half = self.half_res(c2)       # [B, c_half, H/2, W/2] (残差增强)

        # 自底向上：1/4 尺度
        c4 = self.quarter_1(f_half)      # [B, c_quarter, H/4, W/4] (stride=2 降采样)
        f_quarter = self.quarter_res(c4) # [B, c_quarter, H/4, W/4] (残差增强)

        p4 = self.lateral_quarter(f_quarter)    # [B, out_channels, H/4, W/4]
        p2 = self.lateral_half(f_half) + F.interpolate(
            p4, size=f_half.shape[-2:], mode="bilinear", align_corners=False
        )                                        # [B, out_channels, H/2, W/2]
        p1 = self.input_proj(x) + F.interpolate(
            p2, size=x.shape[-2:], mode="bilinear", align_corners=False
        )                                        # [B, out_channels, H, W]

        return {
            4: self.smooth_p4(p4),
            2: self.smooth_p2(p2),
            1: self.smooth_p1(p1),
        }


class MultiViewFPN(nn.Module):
    def __init__(
        self,
        out_channels: int = 128,
        base_channel: int = 32,
    ) -> None:
        super().__init__()
        self.fpn = FPNFeatureExtractor(
            out_channels=out_channels,
            base_channel=base_channel,
        )
        self.out_channels = out_channels

    def forward(self, imgs: torch.Tensor) -> dict[int, torch.Tensor]:
        B, V, C, H, W = imgs.shape
        feats = self.fpn(imgs.view(B * V, C, H, W))
        return {s: f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for s, f in feats.items()}
