from __future__ import annotations

import torch

from base.config import LossConfig, StageWeights

from .depth_loss import depth_cross_entropy_loss, depth_smooth_l1_loss


class MVSLoss:
    """MVSFormer++-style cascade loss.

    For every cascade stage we combine two complementary terms on that stage's
    outputs (both masked, both resampled to the stage resolution):

      * **Cross-entropy (classification)** on the probability volume: the GT
        depth is snapped to its nearest depth hypothesis and we maximise the
        predicted probability of that bin. This is the main signal in
        MVSFormer++ and gives a well-shaped probability volume.
      * **Smooth-L1 (regression)** on the soft-argmin depth: sharpens the final
        sub-pixel depth estimate.

    Stages are weighted by ``StageWeights`` (finer stages weigh more, matching
    MVSFormer++'s ``depth_loss_weights``). Total = Σ_stage w_stage · (w_ce·CE + w_reg·L1).

    Call signature ``(outputs, batch, step) -> (total_loss, logs)`` is kept so the
    existing trainer works unchanged.

    Required ``batch`` keys: ``depth_gt`` [B, H, W] (reference GT depth) and
    optionally ``mask`` [B, H, W] (defaults to depth_gt > 0).
    """

    stage_names = ("stage1", "stage2", "stage3")

    def __init__(self, cfg: LossConfig, stage_weights: StageWeights) -> None:
        self.cfg = cfg
        self.stage_weights = stage_weights

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
        depth_gt = batch["depth_gt"].to(device).float()
        mask = batch.get("mask")
        mask = (depth_gt > 0).float() if mask is None else mask.to(device).float()

        total = outputs["stage1"]["depth"].new_zeros(())
        logs: dict[str, float] = {}

        for name in self.stage_names:
            stage = outputs[name]
            prob = stage["prob"]
            hypos = stage["depth_hypos"]
            depth = stage["depth"]

            l_ce = depth_cross_entropy_loss(prob, hypos, depth_gt, mask) if cfg.use_cross_entropy \
                else depth.new_zeros(())
            l_reg = depth_smooth_l1_loss(depth, depth_gt, mask)

            l_stage = cfg.w_depth * l_ce + cfg.w_reg * l_reg
            total = total + weights[name] * l_stage

            logs[f"{name}/ce"] = float(l_ce.detach())
            logs[f"{name}/reg"] = float(l_reg.detach())

        logs["loss"] = float(total.detach())
        return total, logs
