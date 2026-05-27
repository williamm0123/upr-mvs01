from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import ProjectPaths, VGGTPriorConfig

_VGGT_REPO = Path(__file__).resolve().parent / "vggt"
if str(_VGGT_REPO) not in sys.path:
    sys.path.insert(0, str(_VGGT_REPO))

from vggt.models.vggt import VGGT


def _load_state_from_safetensors(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def load_vggt(device: torch.device | str = "cuda", weights_path: str | Path | None = None) -> nn.Module:
    paths = ProjectPaths()
    weights_path = Path(weights_path) if weights_path is not None else paths.vggt_weights_path

    if weights_path.is_dir():
        if (weights_path / "config.json").is_file():
            model = VGGT.from_pretrained(str(weights_path))
            print(f"[VGGT] loaded HF format from {weights_path}")
        else:
            st_files = list(weights_path.glob("*.safetensors"))
            if not st_files:
                raise FileNotFoundError(f"[VGGT] {weights_path} has neither config.json nor *.safetensors")
            model = VGGT()
            state = _load_state_from_safetensors(st_files[0])
            miss = model.load_state_dict(state, strict=False)
            print(f"[VGGT] loaded safetensors {st_files[0]} missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
    elif weights_path.is_file():
        model = VGGT()
        if weights_path.suffix == ".safetensors":
            state = _load_state_from_safetensors(weights_path)
        else:
            ckpt = torch.load(str(weights_path), map_location="cpu")
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        miss = model.load_state_dict(state, strict=False)
        print(f"[VGGT] loaded {weights_path} missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
    else:
        raise FileNotFoundError(f"[VGGT] weights path not found: {weights_path}")

    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


class VGGTPrior(nn.Module):
    """Wrapper that runs VGGT once and returns per-view sparse depth + confidence.

    Output keys:
        depth_sparse:        [B, V, H, W]  per-view depth from VGGT depth head
        confidence:          [B, V, H, W]  blended confidence in [0, 1]
        valid_mask:          [B, V, H, W]  bool, after threshold filter
    """

    def __init__(self, config: VGGTPriorConfig | None = None, weights_path: str | Path | None = None, device: torch.device | str = "cuda") -> None:
        super().__init__()
        self.config = config or VGGTPriorConfig()
        self.model = load_vggt(device=device, weights_path=weights_path)
        self._target_size = 518

    @torch.no_grad()
    def forward(
        self,
        imgs_raw: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        depth_min: torch.Tensor | None = None,
        depth_max: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, V, C, H, W = imgs_raw.shape
        device = imgs_raw.device

        vggt_in = imgs_raw.view(B * V, C, H, W)
        scale = self._target_size / max(H, W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        new_h = (new_h // 14) * 14
        new_w = (new_w // 14) * 14
        vggt_in = F.interpolate(vggt_in, size=(new_h, new_w), mode="bilinear", align_corners=False)
        vggt_in = vggt_in.view(B, V, C, new_h, new_w)

        preds = self.model(vggt_in)
        depth_pred = preds["depth"][..., 0]
        depth_conf = preds["depth_conf"]

        depth_full = F.interpolate(
            depth_pred.view(B * V, 1, new_h, new_w), size=(H, W), mode="bilinear", align_corners=False
        ).view(B, V, H, W)
        conf_full = F.interpolate(
            depth_conf.view(B * V, 1, new_h, new_w), size=(H, W), mode="bilinear", align_corners=False
        ).view(B, V, H, W)

        conf_norm = self._normalize_confidence(conf_full)
        depth_aligned = self._align_depth_scale(depth_full, conf_norm, depth_min, depth_max)
        valid_mask = conf_norm > 0.2

        return {
            "depth_sparse": depth_aligned,
            "confidence": conf_norm,
            "valid_mask": valid_mask,
        }

    @staticmethod
    def _normalize_confidence(conf: torch.Tensor) -> torch.Tensor:
        flat = conf.view(conf.shape[0], -1)
        lo = flat.quantile(0.02, dim=1).view(-1, 1, 1, 1)
        hi = flat.quantile(0.98, dim=1).view(-1, 1, 1, 1)
        return ((conf - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)

    @staticmethod
    def _align_depth_scale(
        depth_pred: torch.Tensor,
        confidence: torch.Tensor,
        depth_min: torch.Tensor | None,
        depth_max: torch.Tensor | None,
    ) -> torch.Tensor:
        depth_pred = torch.where(
            torch.isfinite(depth_pred) & (depth_pred > 0),
            depth_pred,
            torch.zeros_like(depth_pred),
        )
        if depth_min is None or depth_max is None:
            return depth_pred

        B, V, H, W = depth_pred.shape
        valid = (depth_pred > 0) & torch.isfinite(depth_pred) & (confidence > 0.2)
        flat = depth_pred.view(B, -1)
        valid_flat = valid.view(B, -1)
        medians = []
        for i in range(B):
            vals = flat[i][valid_flat[i]]
            if vals.numel() == 0:
                medians.append(flat.new_tensor(1.0))
            else:
                medians.append(vals.median())
        src_median = torch.stack(medians).view(B, 1, 1, 1).clamp(min=1e-6)
        target_mid = (0.5 * (depth_min + depth_max)).view(B, 1, 1, 1)
        aligned = depth_pred * (target_mid / src_median)
        return aligned.clamp(min=depth_min.view(B, 1, 1, 1), max=depth_max.view(B, 1, 1, 1))
