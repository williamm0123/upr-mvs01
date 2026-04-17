"""Conv2d backbone plus standard top-down FPN for visualization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kernel_bank(dtype: torch.dtype = torch.float32) -> list[torch.Tensor]:
    blur = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
    ) / 16.0
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
    ) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
    ) / 8.0
    laplacian = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
    ) / 4.0
    diagonal = torch.tensor(
        [[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]],
        dtype=dtype,
    ) / 4.0
    identity = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=dtype,
    )
    return [blur, sobel_x, sobel_y, laplacian, diagonal, identity]


def _init_rgb_stride_conv(conv: nn.Conv2d) -> None:
    kernels = _kernel_bank(conv.weight.dtype)
    rgb_weights = torch.tensor([0.299, 0.587, 0.114], dtype=conv.weight.dtype)
    with torch.no_grad():
        conv.weight.zero_()
        for out_channel in range(conv.out_channels):
            kernel = kernels[out_channel % len(kernels)]
            for in_channel in range(min(3, conv.in_channels)):
                conv.weight[out_channel, in_channel] = kernel * rgb_weights[in_channel]
        if conv.bias is not None:
            conv.bias.zero_()


def _init_channelwise_stride_conv(conv: nn.Conv2d) -> None:
    kernels = _kernel_bank(conv.weight.dtype)
    with torch.no_grad():
        conv.weight.zero_()
        for out_channel in range(conv.out_channels):
            in_channel = out_channel % conv.in_channels
            conv.weight[out_channel, in_channel] = kernels[out_channel % len(kernels)]
        if conv.bias is not None:
            conv.bias.zero_()


def _init_lateral_conv(conv: nn.Conv2d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        num_identity = min(conv.in_channels, conv.out_channels)
        for channel in range(num_identity):
            conv.weight[channel, channel, 0, 0] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def _init_smooth_conv(conv: nn.Conv2d) -> None:
    blur = _kernel_bank(conv.weight.dtype)[0]
    with torch.no_grad():
        conv.weight.zero_()
        num_identity = min(conv.in_channels, conv.out_channels)
        for channel in range(num_identity):
            conv.weight[channel, channel] = blur
        if conv.bias is not None:
            conv.bias.zero_()


class ConvFPNVisualizationNet(nn.Module):
    """Four-level Conv2d FPN matching the requested lateral/smooth formula.

    The model is intentionally deterministic for visualization: all convolution
    layers are `nn.Conv2d`, but weights are initialized to simple blur/edge
    filters so untrained features are readable.
    """

    def __init__(
        self,
        c2_channels: int = 16,
        c3_channels: int = 16,
        c4_channels: int = 16,
        c5_channels: int = 16,
        out_channels: int = 16,
    ):
        super().__init__()
        self.stage2 = nn.Conv2d(3, c2_channels, kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Conv2d(c2_channels, c3_channels, kernel_size=3, stride=2, padding=1)
        self.stage4 = nn.Conv2d(c3_channels, c4_channels, kernel_size=3, stride=2, padding=1)
        self.stage5 = nn.Conv2d(c4_channels, c5_channels, kernel_size=3, stride=2, padding=1)

        self.lateral2 = nn.Conv2d(c2_channels, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3_channels, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4_channels, out_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(c5_channels, out_channels, kernel_size=1)

        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.reset_visualization_weights()

    def reset_visualization_weights(self) -> None:
        _init_rgb_stride_conv(self.stage2)
        _init_channelwise_stride_conv(self.stage3)
        _init_channelwise_stride_conv(self.stage4)
        _init_channelwise_stride_conv(self.stage5)
        for lateral in [self.lateral2, self.lateral3, self.lateral4, self.lateral5]:
            _init_lateral_conv(lateral)
        for smooth in [self.smooth2, self.smooth3, self.smooth4, self.smooth5]:
            _init_smooth_conv(smooth)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")

        c2 = self.stage2(image)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        return {
            "C2": c2,
            "C3": c3,
            "C4": c4,
            "C5": c5,
            "P2": p2,
            "P3": p3,
            "P4": p4,
            "P5": p5,
        }
