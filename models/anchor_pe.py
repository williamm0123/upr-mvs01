from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import AnchorPEConfig
from utils.geometry import unproject_depth


def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    B, N, _ = points.shape
    device = points.device
    if N <= num_samples:
        if N == 0:
            return torch.zeros(B, num_samples, dtype=torch.long, device=device)
        base = torch.arange(N, device=device, dtype=torch.long)
        pad = base.new_full((num_samples - N,), N - 1)
        idx = torch.cat([base, pad], dim=0).view(1, num_samples).expand(B, num_samples)
        return idx
    idx = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(num_samples):
        idx[:, i] = farthest
        centroid = points[torch.arange(B, device=device), farthest].view(B, 1, 3)
        d = torch.norm(points - centroid, dim=-1)
        dist = torch.minimum(dist, d)
        farthest = torch.argmax(dist, dim=-1)
    return idx


def select_global_anchors(
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    valid_mask: torch.Tensor,
    num_anchors: int,
    min_confidence: float,
) -> torch.Tensor:
    B, V, H, W, _ = world_points.shape
    out: list[torch.Tensor] = []
    for b in range(B):
        mask = (confidence[b] > min_confidence) & valid_mask[b]
        pts = world_points[b][mask]
        if pts.numel() == 0:
            pts = world_points[b].view(-1, 3)
        if pts.shape[0] > 4 * num_anchors:
            stride = pts.shape[0] // (4 * num_anchors)
            pts = pts[::stride][: 4 * num_anchors]
        idx = farthest_point_sampling(pts.unsqueeze(0), num_anchors)[0]
        out.append(pts[idx])
    return torch.stack(out, dim=0)


def compute_anchor_visibility(
    anchors_world: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    B, K, _ = anchors_world.shape
    V = intrinsics.shape[1]
    H, W = image_hw
    pts_h = torch.cat([anchors_world, torch.ones_like(anchors_world[..., :1])], dim=-1)
    pts_h = pts_h.unsqueeze(1).expand(B, V, K, 4)
    cam = (pts_h @ extrinsics.transpose(-1, -2))[..., :3]
    pix_h = cam @ intrinsics.transpose(-1, -2)
    z = pix_h[..., 2]
    u = pix_h[..., 0] / z.clamp(min=1e-6)
    v = pix_h[..., 1] / z.clamp(min=1e-6)
    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    return in_frame.float()


class AnchorPositionalEncoder(nn.Module):
    def __init__(self, num_anchors: int = 24, hidden: int = 64, out_channels: int = 64) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.mlp = nn.Sequential(
            nn.Linear(3 * num_anchors, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels),
        )
        self.out_channels = out_channels

    def forward(
        self,
        depth_ref: torch.Tensor,
        K_ref: torch.Tensor,
        E_ref_inv: torch.Tensor,
        anchors_world: torch.Tensor,
        visibility_ref: torch.Tensor,
    ) -> torch.Tensor:
        B, H, W = depth_ref.shape
        K_inv = torch.inverse(K_ref)
        world = unproject_depth(depth_ref, K_inv, E_ref_inv)
        K = self.num_anchors
        anchors = anchors_world.view(B, K, 3, 1, 1)
        rel = anchors - world.unsqueeze(1)
        vis = visibility_ref.view(B, K, 1, 1, 1)
        rel = rel * vis
        flat = rel.permute(0, 3, 4, 1, 2).reshape(B, H, W, K * 3)
        pe = self.mlp(flat).permute(0, 3, 1, 2).contiguous()
        return pe


def lambda_schedule(step: int, config: AnchorPEConfig) -> float:
    if step < config.lambda_warmup_steps:
        return 0.0
    if step < config.lambda_warmup_steps + config.lambda_release_steps:
        return (step - config.lambda_warmup_steps) / max(config.lambda_release_steps, 1)
    return 1.0
