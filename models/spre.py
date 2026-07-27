"""Semantic Prior Reliability Estimator (SPRE).

A learned, supervised, DINOv3-witnessed successor to the hand-engineered
offline ``conf`` (``models/conf.py``). SPRE predicts a per-pixel reliability
``r in [0, 1]`` for the (possibly corrupted) depth prior from:

  * frozen DINOv3 semantic features of the reference image — an *independent*
    witness that does not depend on the prior's own depth values, and
  * cheap online statistics of the depth prior (local MAD, gradient, validity)
    computed AFTER prior-corruption, so injected failures are visible.

``r`` replaces the cached ``conf_prior`` fed to ``build_stage1_hypotheses``:
low ``r`` widens the local search (leaning on the prior-independent guard),
high ``r`` keeps it tight. The head is supervised by the corruption mask and
the prior's true error (see ``losses/composite.py``). DINOv3 is frozen; only
the projection + head (+ optional attention) train.

The module emits raw LOGITS (not the sigmoid) so the loss can use the
autocast-safe ``binary_cross_entropy_with_logits``; the network applies the
sigmoid to obtain ``r`` for the hypothesis builder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import DINOConfig, SPREConfig
from models.dinov3.extractor import compute_patch_aligned_size, load_dinov3_vit_base


def _masked_standardize(d: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-sample zero-mean/unit-std over valid pixels; invalid pixels -> 0."""
    v = valid.float()
    n = v.flatten(1).sum(1).clamp_min(1.0)
    m = (d * v).flatten(1).sum(1) / n
    var = (((d - m[:, None, None]) ** 2) * v).flatten(1).sum(1) / n
    s = var.sqrt().clamp_min(1e-3)
    return ((d - m[:, None, None]) / s[:, None, None]) * v


def _local_mad_rel(d: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """3x3 median-absolute-deviation, made scale-free by dividing by depth."""
    B, H, W = d.shape
    nan = float("nan")
    dd = torch.where(valid, d, torch.full_like(d, nan))
    patches = F.unfold(dd.unsqueeze(1), kernel_size=3, padding=1).view(B, 9, H, W)
    med = patches.nanmedian(dim=1).values
    mad = (patches - med.unsqueeze(1)).abs().nanmedian(dim=1).values
    mad = torch.nan_to_num(mad, nan=0.0)
    return mad / (d.abs() + 1.0)


def _grad_mag_rel(d: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Scale-free forward-difference gradient magnitude."""
    dd = torch.nan_to_num(torch.where(valid, d, torch.zeros_like(d)), nan=0.0)
    gx = F.pad(dd[:, :, 1:] - dd[:, :, :-1], (0, 1, 0, 0))
    gy = F.pad(dd[:, 1:, :] - dd[:, :-1, :], (0, 0, 0, 1))
    return (gx.abs() + gy.abs()) / (d.abs() + 1.0)


class SPRE(nn.Module):
    N_STAT = 4  # depth_norm, mad_rel, grad_rel, valid

    def __init__(self, cfg: SPREConfig, dino_cfg: DINOConfig, weights_file) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = tuple(dino_cfg.layers)
        self.patch_size = int(dino_cfg.patch_size)
        self.max_side = int(dino_cfg.max_side)

        # Frozen DINOv3 witness (loaded on CPU; the outer .to(device) moves it).
        self.dino = load_dinov3_vit_base("cpu", weights_file, patch_size=self.patch_size)
        for p in self.dino.parameters():
            p.requires_grad_(False)

        self.register_buffer("dino_mean", torch.tensor(dino_cfg.mean).view(1, 3, 1, 1))
        self.register_buffer("dino_std", torch.tensor(dino_cfg.std).view(1, 3, 1, 1))

        dino_dim = 768 * len(self.layers)
        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_dim, cfg.proj_dim, kernel_size=1),
            nn.GroupNorm(min(8, cfg.proj_dim), cfg.proj_dim),
            nn.GELU(),
        )

        self.use_attention = bool(cfg.use_attention)
        if self.use_attention:
            self.attn = nn.MultiheadAttention(cfg.proj_dim, cfg.num_heads, batch_first=True)
            self.attn_norm = nn.LayerNorm(cfg.proj_dim)

        in_ch = cfg.proj_dim + self.N_STAT
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, cfg.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, cfg.hidden), cfg.hidden),
            nn.GELU(),
            nn.Conv2d(cfg.hidden, cfg.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, cfg.hidden), cfg.hidden),
            nn.GELU(),
            nn.Conv2d(cfg.hidden, 1, kernel_size=1),
        )

    def train(self, mode: bool = True):
        # Keep the frozen backbone in eval regardless of the parent's mode.
        super().train(mode)
        self.dino.eval()
        return self

    def _dino_features(self, ref_img: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        B, _, H, W = ref_img.shape
        dh, dw = compute_patch_aligned_size(H, W, self.max_side, self.patch_size)
        self.dino.eval()
        with torch.no_grad():
            x = F.interpolate(ref_img, size=(dh, dw), mode="bilinear", align_corners=False)
            x = (x - self.dino_mean) / self.dino_std
            feats = self.dino.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True)
            feats = [F.interpolate(f.float(), size=target_hw, mode="bilinear", align_corners=False)
                     for f in feats]
            f_sem = torch.cat(feats, dim=1)  # [B, 768*L, h, w], no grad (frozen witness)
        return f_sem

    def forward(
        self,
        ref_img: torch.Tensor,
        depth_prior: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Return reliability LOGITS ``[B, h, w]`` at ``target_hw``.

        ref_img     : [B, 3, H, W] in [0, 1] (reference view only)
        depth_prior : [B, H, W]    the possibly-corrupted metric prior
        """
        f_sem = self._dino_features(ref_img, target_hw)
        f_sem = self.dino_proj(f_sem)  # trainable; grad flows here, not into DINOv3

        if self.use_attention:
            B, C, h, w = f_sem.shape
            t = f_sem.flatten(2).transpose(1, 2)   # [B, h*w, C]
            a, _ = self.attn(t, t, t)              # self-attention (region consistency)
            t = self.attn_norm(t + a)
            f_sem = t.transpose(1, 2).reshape(B, C, h, w)

        # Online depth statistics at target_hw (fp32, corruption-consistent).
        d = F.interpolate(depth_prior.unsqueeze(1).float(), size=target_hw, mode="nearest").squeeze(1)
        valid = torch.isfinite(d) & (d > 0)
        d = torch.nan_to_num(d, nan=0.0)
        stats = torch.stack(
            [_masked_standardize(d, valid), _local_mad_rel(d, valid),
             _grad_mag_rel(d, valid), valid.float()],
            dim=1,
        )  # [B, 4, h, w]

        x = torch.cat([f_sem, stats.to(f_sem.dtype)], dim=1)
        return self.head(x).squeeze(1)  # logits [B, h, w]
