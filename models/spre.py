"""Semantic Prior Reliability Estimator (SPRE).

A learned, supervised, DINOv3-witnessed successor to the hand-engineered
offline ``conf`` (``models/conf.py``). SPRE predicts a per-pixel reliability
``r in [0, 1]`` for the (possibly corrupted) depth prior from:

  * frozen DINOv3 semantic features of every view, fused by ``CrossViTFusion``
    (ported from MVSFormer++) — an *independent* witness that does not depend
    on the prior's own depth values, and
  * cheap online statistics of the depth prior (local MAD, gradient, validity)
    computed AFTER prior-corruption, so injected failures are visible.

``r`` replaces the cached ``conf_prior`` fed to ``build_stage1_hypotheses``:
low ``r`` widens the local search (leaning on the prior-independent guard),
high ``r`` keeps it tight. The head is supervised by the corruption mask and
the prior's true error (see ``losses/composite.py``). DINOv3 is frozen; only
the fusion + projection + head train.

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


class _AttnBlock(nn.Module):
    """Pre-norm transformer block; self-attention when ``kv`` is None, else cross.

    Uses scaled_dot_product_attention so the [B, heads, N_q, N_kv] matrix is
    never materialised — with all source views as keys N_kv is (V-1)*N, which
    would otherwise cost hundreds of MB per block just to hold for backward.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, cross: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        # self-attention reuses norm_q for the keys; allocating an unused
        # norm_kv would leave parameters without gradients, which DDP rejects.
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6) if cross else None
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        B, N, C = t.shape
        return t.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        q_in = self.norm_q(x)
        kv_in = q_in if kv is None or self.norm_kv is None else self.norm_kv(kv)
        q = self._heads(self.to_q(q_in))
        k, v = self.to_kv(kv_in).chunk(2, dim=-1)
        a = F.scaled_dot_product_attention(q, self._heads(k), self._heads(v))
        B, _, N, _ = a.shape
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, -1))
        return x + self.mlp(self.norm2(x))


class CrossViTFusion(nn.Module):
    """MVSFormer++'s CrossVITDecoder, adapted to SPRE's single-output-map job.

    Reference branch (identical to theirs): the shallowest DINOv3 layer seeds the
    stream, then each further layer is folded in as
    ``norm(aas * self_attn(prev) + layer_i)`` — the AAS residual that carries
    their ablation gain, not a plain concatenation of the three layers.

    Cross branch (**direction reversed**): they use src as query and ref as
    key/value because every view needs its own feature for the plane sweep. SPRE
    needs one map in the *reference* frame, so here the ref tokens query the
    pooled source tokens. A ref token that finds no support across the sources is
    occluded or monocular-guessed — exactly the "do not trust the prior here"
    signal, and unavailable from the cost volume since SPRE runs before it.
    """

    def __init__(self, in_dim: int, cfg: SPREConfig, n_layers: int) -> None:
        super().__init__()
        dim, heads = int(cfg.attn_dim), int(cfg.num_heads)
        self.cross_view = bool(cfg.cross_view)
        self.in_proj = nn.Linear(in_dim, dim)

        self.self_blocks = nn.ModuleList(_AttnBlock(dim, heads) for _ in range(n_layers - 1))
        self.self_norms = nn.ModuleList(nn.LayerNorm(dim, eps=1e-6) for _ in range(n_layers - 1))
        self.self_aas = nn.ParameterList(
            nn.Parameter(torch.tensor(float(cfg.aas_init))) for _ in range(n_layers - 1)
        )
        if self.cross_view:
            self.cross_blocks = nn.ModuleList(_AttnBlock(dim, heads, cross=True) for _ in range(n_layers))
            self.cross_norms = nn.ModuleList(nn.LayerNorm(dim, eps=1e-6) for _ in range(n_layers - 1))
            self.cross_aas = nn.ParameterList(
                nn.Parameter(torch.tensor(float(cfg.aas_init))) for _ in range(n_layers - 1)
            )
        self.out_dim = dim * 2 if self.cross_view else dim

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        """``layers``: per DINOv3 layer, [B, V, N, in_dim]. Returns [B, N, out_dim]."""
        x = [self.in_proj(t) for t in layers]

        ref = [x[0][:, 0]]
        for i, (blk, norm, aas) in enumerate(zip(self.self_blocks, self.self_norms, self.self_aas)):
            ref.append(norm(aas * blk(ref[-1]) + x[i + 1][:, 0]))
        if not self.cross_view:
            return ref[-1]

        V = x[0].shape[1]
        ctx = None
        for i, blk in enumerate(self.cross_blocks):
            # V == 1 has no sources: degenerate to self-attention rather than
            # feeding an empty key tensor (SDPA cannot normalise over 0 keys).
            src = x[i][:, 1:].flatten(1, 2) if V > 1 else ref[i]
            if i == 0:
                q = ref[i]
            else:
                q = self.cross_norms[i - 1](self.cross_aas[i - 1] * ctx + ref[i])
            ctx = blk(q, kv=src)
        return torch.cat([ref[-1], ctx], dim=-1)


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

        self.fusion = CrossViTFusion(768, cfg, n_layers=len(self.layers))
        # 1x1 projection lives at token resolution: a pointwise conv commutes
        # exactly with bilinear upsampling, so projecting the 26x32 grid instead
        # of the 128x160 one is arithmetically identical and ~18x cheaper in the
        # activations held for backward.
        self.dino_proj = nn.Conv2d(self.fusion.out_dim, cfg.proj_dim, kernel_size=1)
        self.proj_post = nn.Sequential(
            nn.GroupNorm(min(8, cfg.proj_dim), cfg.proj_dim),
            nn.GELU(),
        )

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

    def _dino_features(self, images: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, int]]:
        """Frozen witness over every view. Returns per-layer [B, V, N, 768] token
        sequences (still on the patch grid) plus that grid's (h, w)."""
        B, V, _, H, W = images.shape
        dh, dw = compute_patch_aligned_size(H, W, self.max_side, self.patch_size)
        self.dino.eval()
        with torch.no_grad():
            x = images.reshape(B * V, 3, H, W)
            x = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)
            x = (x - self.dino_mean) / self.dino_std
            feats = self.dino.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True)
        gh, gw = feats[0].shape[-2:]
        tokens = [f.float().flatten(2).transpose(1, 2).view(B, V, gh * gw, -1) for f in feats]
        return tokens, (gh, gw)

    def forward(
        self,
        images: torch.Tensor,
        depth_prior: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Return reliability LOGITS ``[B, h, w]`` at ``target_hw``.

        images      : [B, V, 3, H, W] in [0, 1]; view 0 is the reference, the
                      rest are the sources the cross-attention pools over
        depth_prior : [B, H, W]       the possibly-corrupted metric prior
        """
        tokens, (gh, gw) = self._dino_features(images)
        fused = self.fusion(tokens)                                   # [B, N, out_dim]
        B, _, C = fused.shape
        f_sem = fused.transpose(1, 2).reshape(B, C, gh, gw)
        f_sem = self.dino_proj(f_sem)                                 # project on the patch grid
        f_sem = F.interpolate(f_sem, size=target_hw, mode="bilinear", align_corners=False)
        f_sem = self.proj_post(f_sem)

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
