from __future__ import annotations

import torch
import torch.nn.functional as F

from base.config import LossConfig, StageWeights

from .depth_loss import depth_cross_entropy_loss, depth_smooth_l1_loss


class MVSLoss:
    """Cascade loss: per-stage cross-entropy + interval-normalized smooth-L1.

    Both terms are O(1) regardless of scene depth scale, stage bin width, or
    dataset units, so neither can drown the other (a raw-mm smooth-L1 next to
    an O(1) CE makes up >90% of the total and its batch-to-batch variance
    dominates the training curve):

      * **Cross-entropy (classification)** on the logits volume: the GT depth
        snaps to its nearest hypothesis bin, maximised via ``log_softmax``.
        Main coarse signal, as in MVSFormer++.
      * **Smooth-L1 (regression)** on the soft-argmin depth, with the error in
        units of the per-pixel hypothesis interval and clamped at
        ``reg_err_clamp`` bins: sharpens sub-bin accuracy; pixels far off get
        no reg gradient and are driven back by CE alone.

    Pixels whose GT lies outside a stage's hypothesis range are excluded from
    both terms — the range there is the *previous* stage's responsibility, and
    supervising an unreachable target only injects conflicting gradients.

    Stages are weighted by ``StageWeights`` (finer stages weigh more, matching
    MVSFormer++'s ``depth_loss_weights``).
    Total = Σ_stage w_stage · (w_ce·CE + w_reg·SmoothL1).

    Per-stage diagnostics in ``logs`` (all cheap, all worth watching):
      * ``in_range``    — fraction of valid pixels whose GT the hypothesis
                          range actually covers (should climb to >0.95);
      * ``p_max``       — mean max probability (prob-volume sharpness);
      * ``interval_mm`` — mean hypothesis bin width in metric units.

    Required ``batch`` keys: ``depth_gt`` [B, H, W] and optionally ``mask``
    (defaults to depth_gt > 0). Required stage outputs: ``logits``, ``prob``,
    ``depth``, ``depth_hypos``.
    """

    stage_names = ("stage1", "stage2", "stage3")

    def __init__(self, cfg: LossConfig, stage_weights: StageWeights) -> None:
        self.cfg = cfg
        self.stage_weights = stage_weights

    @staticmethod
    def _to_stage_res(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        if tuple(x.shape[-2:]) == hw:
            return x
        return F.interpolate(x.unsqueeze(1).float(), size=hw, mode="nearest").squeeze(1)

    def __call__(
        self,
        outputs: dict,
        batch: dict,
        step: int = 0,
        **_: object,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        sw = self.stage_weights
        weights = {"stage1": sw.stage1, "stage2": sw.stage2, "stage3": sw.stage3}

        device = outputs["stage1"]["depth"].device
        depth_gt_full = batch["depth_gt"].to(device).float()
        mask_full = batch.get("mask")
        mask_full = (depth_gt_full > 0).float() if mask_full is None else mask_full.to(device).float()

        total = outputs["stage1"]["depth"].new_zeros(())
        logs: dict[str, float] = {}

        for name in self.stage_names:
            stage = outputs[name]
            hypos = stage["depth_hypos"]  # [B, D, h, w]
            logits = stage["logits"]
            depth = stage["depth"]
            hw = tuple(hypos.shape[-2:])

            gt = self._to_stage_res(depth_gt_full, hw)
            valid = self._to_stage_res(mask_full, hw).bool() & (gt > 0)

            hypo_min = hypos.amin(dim=1)
            hypo_max = hypos.amax(dim=1)
            in_range = (gt >= hypo_min) & (gt <= hypo_max)
            sup = valid & in_range
            # Per-pixel mean bin width (adjacent diffs of sorted hypos telescope
            # to span/(D-1)). Detached so the normalizer never feeds gradients
            # back into the learned hypothesis range.
            interval = ((hypo_max - hypo_min) / max(hypos.shape[1] - 1, 1)).detach()

            l_ce = depth_cross_entropy_loss(logits, hypos, gt, sup) if cfg.use_cross_entropy \
                else depth.new_zeros(())
            l_reg = depth_smooth_l1_loss(depth, gt, sup, interval, err_clamp=cfg.reg_err_clamp)

            total = total + weights[name] * (cfg.w_ce * l_ce + cfg.w_reg * l_reg)

            logs[f"{name}/ce"] = float(l_ce.detach())
            logs[f"{name}/reg"] = float(l_reg.detach())
            logs[f"{name}/in_range"] = float(in_range[valid].float().mean()) if valid.any() else 1.0
            logs[f"{name}/p_max"] = float(stage["prob"].detach().amax(dim=1).mean())
            logs[f"{name}/interval_mm"] = float(interval[valid].mean()) if valid.any() else 0.0

        logs["loss"] = float(total.detach())
        return total, logs
