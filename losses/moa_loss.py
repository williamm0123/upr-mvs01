"""Loss for MoAMVSNet (MoA.md §11).

    L = sum_s w_s CE_s  +  sum_{s=2..4} lam_s (w_c L_center + w_sh L_shape + w_r L_conf)

CE_s is the two-bin soft-label CE of each MVS stage, restricted to pixels whose
GT lies inside that stage's window (a GT outside the window has no correct bin;
getting it inside is the job of the centre, i.e. of L_center). GT is used only
here, never in the forward pass.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from base.config_moa import MoALossConfig
from losses.depth_loss import soft_label_cross_entropy
from models.moa.geometry import depth_to_u, sample_at_feature_pixels, u_to_depth


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    return (x * m).sum() / m.sum().clamp_min(1.0)


def _gt_at(gt: torch.Tensor, mask: torch.Tensor, hw) -> tuple[torch.Tensor, torch.Tensor]:
    g = sample_at_feature_pixels(gt.unsqueeze(1), hw)
    m = sample_at_feature_pixels(mask.float().unsqueeze(1), hw) > 0.5
    return g, m & (g > 0)


class MoALoss:
    def __init__(self, cfg: MoALossConfig, num_depths_stage1: int) -> None:
        self.cfg = cfg
        self.du_global = 1.0 / float(num_depths_stage1 - 1)

    def __call__(self, outputs: dict, batch: dict, diagnostics: bool = False
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (loss, logs). Log values are detached 0-d tensors (no host sync);
        ``diagnostics`` adds coverage / error / MoA statistics — meant for log steps."""
        cfg = self.cfg
        gt = batch["depth_gt"].float()
        dv = batch["depth_values"].float()
        mask = batch["mask"].float() > 0.5
        mask &= (gt >= dv.amin(dim=1).view(-1, 1, 1)) & (gt <= dv.amax(dim=1).view(-1, 1, 1))
        vmin, vmax = outputs["vmin"], outputs["vmax"]
        total = gt.new_zeros(())
        logs: dict[str, torch.Tensor] = {}

        for s in range(1, 5):
            st = outputs[f"stage{s}"]
            hyp = st["depth_hypos"]
            g, m = _gt_at(gt, mask, tuple(hyp.shape[-2:]))
            inside = (g >= hyp[:, :1]) & (g <= hyp[:, -1:])
            ce = soft_label_cross_entropy(st["logits"], hyp, g[:, 0], (m & inside)[:, 0])
            total = total + cfg.stage_weights[s - 1] * ce
            logs[f"ce_s{s}"] = ce.detach()
            if diagnostics:
                with torch.no_grad():
                    logs[f"cover_s{s}"] = _masked_mean(inside.float(), m)
                    logs[f"err_s{s}_mm"] = _masked_mean((st["depth"] - g).abs(), m)

        for s in range(2, 5):
            mo = outputs.get(f"moa{s}")
            if mo is None:
                continue
            hw = tuple(mo.center_u.shape[-2:])
            g, m = _gt_at(gt, mask, hw)
            u_gt = depth_to_u(torch.where(m, g, torch.ones_like(g)), vmin, vmax)
            dg = self.du_global

            l_center = _masked_mean(
                F.smooth_l1_loss((mo.center_u - u_gt) / dg, torch.zeros_like(u_gt), reduction="none"), m)

            gxc = mo.center_u[..., :, 1:] - mo.center_u[..., :, :-1]
            gyc = mo.center_u[..., 1:, :] - mo.center_u[..., :-1, :]
            gxg = u_gt[..., :, 1:] - u_gt[..., :, :-1]
            gyg = u_gt[..., 1:, :] - u_gt[..., :-1, :]
            lg = torch.where(m, g, torch.ones_like(g)).log()
            mx = m[..., :, 1:] & m[..., :, :-1] & ((lg[..., :, 1:] - lg[..., :, :-1]).abs() < cfg.gt_edge_tau)
            my = m[..., 1:, :] & m[..., :-1, :] & ((lg[..., 1:, :] - lg[..., :-1, :]).abs() < cfg.gt_edge_tau)
            hx = F.smooth_l1_loss((gxc - gxg) / dg, torch.zeros_like(gxg), reduction="none")
            hy = F.smooth_l1_loss((gyc - gyg) / dg, torch.zeros_like(gyg), reduction="none")
            l_shape = ((hx * mx).sum() + (hy * my).sum()) / (mx.sum() + my.sum()).clamp_min(1)

            target = torch.exp(-(mo.mvs_u - u_gt).abs() / (mo.du + 1e-8)).detach()
            l_conf = _masked_mean(F.binary_cross_entropy_with_logits(
                mo.mvs_conf_logit, target, reduction="none"), m)

            lam = cfg.moa_stage_weights[s - 2]
            total = total + lam * (cfg.w_center * l_center + cfg.w_shape * l_shape + cfg.w_conf * l_conf)
            logs[f"moa{s}_center"] = l_center.detach()
            logs[f"moa{s}_shape"] = l_shape.detach()
            logs[f"moa{s}_conf"] = l_conf.detach()
            if diagnostics:
                logs.update(moa_diagnostics(mo, g, m, vmin, vmax, s))

        logs["loss"] = total.detach()
        return total, logs


@torch.no_grad()
def moa_diagnostics(mo, g: torch.Tensor, m: torch.Tensor, vmin, vmax, s: int) -> dict[str, torch.Tensor]:
    """Is MoA moving the centre closer to GT than the MVS centre it started from?"""
    p = f"moa{s}_"
    z_mvs = u_to_depth(mo.mvs_u, vmin, vmax)
    e_c = (mo.center_depth - g).abs()
    e_m = (z_mvs - g).abs()
    out = {
        p + "err_center_mm": _masked_mean(e_c, m),
        p + "err_mvs_mm": _masked_mean(e_m, m),
        p + "better_frac": _masked_mean((e_c < e_m).float(), m),
        p + "override_rate": (mo.alpha > 0.5).float().mean(),
        p + "conflict": mo.conflict.mean(),
        p + "r_mvs": mo.mvs_confidence.mean(),
        p + "mono_valid": mo.mono_valid.mean(),
        p + "global_ok": mo.global_ok.float().mean(),
        p + "edge_frac": mo.edge.mean(),
    }
    for j, name in enumerate(("mvs", "glob", "l3", "l5", "l7")):
        out[p + f"pi_{name}"] = mo.mixture_weights[:, j].mean()
    for j, name in enumerate(("3", "5", "7")):
        out[p + f"aff_valid_{name}"] = mo.affine_valid[:, j].mean()
        out[p + f"offset_only_{name}"] = mo.affine_offset_only[:, j].mean()
        out[p + f"barrier_{name}"] = mo.edge_barrier_rate[:, j].mean()
    return out
