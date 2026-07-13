from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm 组数：取 <=max_groups 且能整除通道数的最大值。

    GN 的统计量按通道分组、每张图独立计算，与 batch size 无关，避免了
    B*V=3 张图时 BN 统计量高方差、train/eval 行为不一致的问题
    （3D 解码器 decoder.py 的 ConvGN3d 同样风格）。
    """
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvGnReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv_gn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_gn_relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convgn1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _group_norm(channels),
            nn.ReLU(inplace=True),
        )
        self.convgn2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _group_norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convgn1(x)
        out = self.convgn2(out)
        return F.relu(out + x, inplace=True)


class ScaleTower(nn.Module):
    """自底向上单个尺度塔：stride-2 降采样 + 一层同尺度 conv + 残差块。

    每尺度共 4 个 3x3 卷积层（2 个 ConvGnReLU + 1 个残差块），比原来的
    (stride conv + 残差) 更深，扩大感受野；下探一级（多一次 stride-2）
    对感受野的贡献是翻倍级的。
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = ConvGnReLU(in_ch, out_ch, stride=2)   # 降采样并换通道
        self.conv = ConvGnReLU(out_ch, out_ch)            # 同尺度加深
        self.res = ResidualBlock(out_ch)                  # 同尺度残差增强

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.conv(self.down(x)))


class FPNFeatureExtractor(nn.Module):
    """从头训练的特征塔 + FPN，产出 1/4、1/2、全分辨率三级特征。

    自底向上：原图 -> 1/2 塔 -> 1/4 塔 -> 1/8 塔（每塔第 1 层 stride=2 降采样，
    同尺度再叠加 conv + 残差块加深）。下探到 1/8 主要是为了扩感受野——1/8
    分辨率的特征图很便宜（显存/参数增量都很小），它不直接参与代价体，
    价值是让 1/4 特征经过更深的 top-down 路径、带上更全局的语义。

    自顶向下：p8(1/8) -> 上采样加到 p4(1/4) -> 加到 p2(1/2) -> 加到 p1(全分辨率)。
    全分辨率分支用 2 层 Conv-GN-ReLU-Conv 而不是单层线性投影，至少经过一次
    非线性组合来补高频细节（精细 stage 的判别力主力仍来自 top-down 语义）。
    """

    def __init__(
        self,
        out_channels: int = 128,
        base_channel: int = 32,
    ) -> None:
        super().__init__()
        c_half = base_channel * 2      # 1/2 尺度内部通道
        c_quarter = base_channel * 4   # 1/4 尺度内部通道
        c_eighth = base_channel * 8    # 1/8 尺度内部通道

        # ---- 自底向上：三级特征塔 ----
        self.tower_half = ScaleTower(3, c_half)             # 原图 -> 1/2
        self.tower_quarter = ScaleTower(c_half, c_quarter)  # 1/2 -> 1/4
        self.tower_eighth = ScaleTower(c_quarter, c_eighth)  # 1/4 -> 1/8

        # ---- FPN 横向连接：统一到 out_channels 才能逐元素相加 ----
        self.lateral_half = nn.Conv2d(c_half, out_channels, 1)
        self.lateral_quarter = nn.Conv2d(c_quarter, out_channels, 1)
        self.lateral_eighth = nn.Conv2d(c_eighth, out_channels, 1)

        # ---- 全分辨率分支：2 层 conv（Conv-GN-ReLU-Conv），非单层线性投影 ----
        self.input_proj = nn.Sequential(
            ConvGnReLU(3, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # ---- 输出平滑：消除上采样混叠 ----
        self.smooth_p1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        # 自底向上
        f_half = self.tower_half(x)          # [B, c_half, H/2, W/2]
        f_quarter = self.tower_quarter(f_half)  # [B, c_quarter, H/4, W/4]
        f_eighth = self.tower_eighth(f_quarter)  # [B, c_eighth, H/8, W/8]

        # 自顶向下：从 1/8 开始逐级上采样相加
        p8 = self.lateral_eighth(f_eighth)      # [B, out_channels, H/8, W/8]
        p4 = self.lateral_quarter(f_quarter) + F.interpolate(
            p8, size=f_quarter.shape[-2:], mode="bilinear", align_corners=False
        )                                        # [B, out_channels, H/4, W/4]
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
