from __future__ import annotations

import torch
import torch.nn.functional as F


def make_pixel_grid(h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    ones = torch.ones_like(xx)
    return torch.stack([xx, yy, ones], dim=0)


def unproject_depth(
    depth: torch.Tensor,
    K_inv: torch.Tensor,
    extrinsic_inv: torch.Tensor,
) -> torch.Tensor:
    B, H, W = depth.shape
    grid = make_pixel_grid(H, W, depth.device, depth.dtype)
    grid = grid.view(3, -1).unsqueeze(0).expand(B, -1, -1)
    cam_ray = torch.bmm(K_inv, grid)
    cam_pts = cam_ray * depth.view(B, 1, -1)
    cam_pts_h = torch.cat([cam_pts, torch.ones_like(cam_pts[:, :1])], dim=1)
    world = torch.bmm(extrinsic_inv, cam_pts_h)[:, :3]
    return world.view(B, 3, H, W)


def project_world_to_pixel(
    points_world: torch.Tensor,
    K: torch.Tensor,
    extrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = points_world.shape[0]
    N = points_world.shape[-1]
    pts_h = torch.cat([points_world, torch.ones_like(points_world[:, :1])], dim=1)
    cam = torch.bmm(extrinsic, pts_h)[:, :3]
    z = cam[:, 2:3]
    uv_h = torch.bmm(K, cam)
    uv = uv_h[:, :2] / torch.clamp(uv_h[:, 2:3], min=1e-6)
    return uv, z.view(B, N)


def homography_warp_features(
    src_features: torch.Tensor,
    K_ref: torch.Tensor,
    K_src: torch.Tensor,
    E_ref: torch.Tensor,
    E_src: torch.Tensor,
    depth_hypos: torch.Tensor,
    feature_stride: int,
) -> torch.Tensor:
    B, C, H, W = src_features.shape
    D = depth_hypos.shape[1]
    device = src_features.device
    dtype = src_features.dtype

    K_ref_s = K_ref.clone()
    K_src_s = K_src.clone()
    K_ref_s[:, 0, :] = K_ref_s[:, 0, :] / feature_stride
    K_ref_s[:, 1, :] = K_ref_s[:, 1, :] / feature_stride
    K_src_s[:, 0, :] = K_src_s[:, 0, :] / feature_stride
    K_src_s[:, 1, :] = K_src_s[:, 1, :] / feature_stride

    R_ref = E_ref[:, :3, :3]
    t_ref = E_ref[:, :3, 3:4]
    R_src = E_src[:, :3, :3]
    t_src = E_src[:, :3, 3:4]

    R_src_inv = R_src.transpose(1, 2)
    R_rel = R_ref @ R_src_inv
    t_rel = t_ref - R_rel @ t_src

    grid = make_pixel_grid(H, W, device, dtype).view(3, -1).unsqueeze(0).expand(B, -1, -1)
    K_ref_inv = torch.inverse(K_ref_s)
    rays = torch.bmm(K_ref_inv, grid)

    rays_d = rays.unsqueeze(1) * depth_hypos.view(B, D, 1, H * W)
    rays_d = rays_d.reshape(B * D, 3, H * W)

    R_rel_d = R_rel.unsqueeze(1).expand(B, D, 3, 3).reshape(B * D, 3, 3)
    t_rel_d = t_rel.unsqueeze(1).expand(B, D, 3, 1).reshape(B * D, 3, 1)
    cam_src = torch.bmm(R_rel_d.transpose(1, 2), rays_d - t_rel_d)
    K_src_d = K_src_s.unsqueeze(1).expand(B, D, 3, 3).reshape(B * D, 3, 3)
    pix_src = torch.bmm(K_src_d, cam_src)
    z_src = pix_src[:, 2:3].clamp(min=1e-6)
    uv = pix_src[:, :2] / z_src

    uv_x = uv[:, 0] / (W - 1) * 2.0 - 1.0
    uv_y = uv[:, 1] / (H - 1) * 2.0 - 1.0
    grid_sample = torch.stack([uv_x, uv_y], dim=-1).view(B * D, H, W, 2)

    src_features_d = src_features.unsqueeze(1).expand(B, D, C, H, W).reshape(B * D, C, H, W)
    warped = F.grid_sample(
        src_features_d,
        grid_sample,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped.view(B, D, C, H, W).permute(0, 2, 1, 3, 4).contiguous()


def soft_argmin(prob_volume: torch.Tensor, depth_hypos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if depth_hypos.dim() == 2:
        B, D = depth_hypos.shape
        depth_hypos = depth_hypos.view(B, D, 1, 1)
    depth = (prob_volume * depth_hypos).sum(dim=1)
    var = (prob_volume * (depth_hypos - depth.unsqueeze(1)) ** 2).sum(dim=1)
    sigma = torch.sqrt(var.clamp(min=1e-12))
    return depth, sigma


def depth_to_normal(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    B, H, W = depth.shape
    K_inv = torch.inverse(K)
    pad = F.pad(depth.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    dx = (pad[:, 1:-1, 2:] - pad[:, 1:-1, :-2]) * 0.5
    dy = (pad[:, 2:, 1:-1] - pad[:, :-2, 1:-1]) * 0.5
    grid = make_pixel_grid(H, W, depth.device, depth.dtype).view(3, -1).unsqueeze(0).expand(B, -1, -1)
    rays = torch.bmm(K_inv, grid).view(B, 3, H, W)
    pts = rays * depth.unsqueeze(1)
    px = torch.zeros_like(pts)
    py = torch.zeros_like(pts)
    px[:, :, :, 1:-1] = (pts[:, :, :, 2:] - pts[:, :, :, :-2]) * 0.5
    py[:, :, 1:-1] = (pts[:, :, 2:] - pts[:, :, :-2]) * 0.5
    n = torch.cross(px, py, dim=1)
    n = F.normalize(n, dim=1, eps=1e-6)
    _ = dx + dy
    return n


def reproject_with_depth(
    depth_ref: torch.Tensor,
    K_ref: torch.Tensor,
    E_ref: torch.Tensor,
    K_src: torch.Tensor,
    E_src: torch.Tensor,
) -> torch.Tensor:
    B, H, W = depth_ref.shape
    E_ref_inv = torch.inverse(E_ref)
    world = unproject_depth(depth_ref, torch.inverse(K_ref), E_ref_inv)
    world_flat = world.view(B, 3, -1)
    uv, _ = project_world_to_pixel(world_flat, K_src, E_src)
    return uv.view(B, 2, H, W)


def warp_image_by_depth(
    src_image: torch.Tensor,
    depth_ref: torch.Tensor,
    K_ref: torch.Tensor,
    E_ref: torch.Tensor,
    K_src: torch.Tensor,
    E_src: torch.Tensor,
) -> torch.Tensor:
    B, _, H, W = src_image.shape
    uv = reproject_with_depth(depth_ref, K_ref, E_ref, K_src, E_src)
    uv_x = uv[:, 0] / (W - 1) * 2.0 - 1.0
    uv_y = uv[:, 1] / (H - 1) * 2.0 - 1.0
    grid = torch.stack([uv_x, uv_y], dim=-1)
    return F.grid_sample(
        src_image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def make_depth_hypotheses(
    depth_center: torch.Tensor,
    half_range: torch.Tensor,
    num_depths: int,
) -> torch.Tensor:
    B, H, W = depth_center.shape
    device = depth_center.device
    dtype = depth_center.dtype
    steps = torch.linspace(-1.0, 1.0, num_depths, device=device, dtype=dtype)
    return depth_center.unsqueeze(1) + half_range.unsqueeze(1) * steps.view(1, num_depths, 1, 1)


def make_depth_hypotheses_global(
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    num_depths: int,
    h: int,
    w: int,
) -> torch.Tensor:
    B = depth_min.shape[0]
    device = depth_min.device
    dtype = depth_min.dtype
    steps = torch.linspace(0.0, 1.0, num_depths, device=device, dtype=dtype)
    span = (depth_max - depth_min).view(B, 1, 1, 1)
    base = depth_min.view(B, 1, 1, 1)
    return (base + span * steps.view(1, num_depths, 1, 1)).expand(B, num_depths, h, w).contiguous()
