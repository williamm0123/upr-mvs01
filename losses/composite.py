from __future__ import annotations

import torch
import torch.nn.functional as F

from base.config import LossConfig, StageWeights

from .depth_loss import normalized_huber_loss, soft_label_cross_entropy


class MVSLoss:
    """Cascade loss co-designed with the dual-branch stage-1 hypothesis axis.

    Stage 1 (global 48 + local 16, merged sorted axis):

      * ``L64``  — soft-label CE over all 64 candidates. When the local branch
        is right its dense bins carry most of the label mass; when it is wrong
        its candidates automatically become negatives and the correct global
        bin gets the positive signal. Supervised on ``valid & global-in-range``
        (the guard covers ~all valid pixels by construction).
      * ``L48``  — auxiliary CE over the *global branch only* (softmax over its
        48 gathered logits). This trains the guard to localize GT on every
        pixel at every step, even while the local branch wins the 64-way
        softmax, so its rescue ability never atrophies.
      * ``L16``  — auxiliary CE over the local branch, only where GT actually
        falls inside the local window (a wrong prior must not force the local
        branch to hallucinate; L64 already presses its candidates down).
      * ``reg``  — interval-normalized Huber on the mode-centered depth over
        ALL valid pixels (no in-range gating, no hard clamp): out-of-range
        pixels keep a bounded pull toward GT instead of a blind spot.

    Stages 2/3: soft-label CE (in-range) + all-valid Huber whose normalizer is
    the stage's own bin interval; the regression keeps correcting the previous
    stage's center even when GT fell outside the current window.

    Edge-band pixels (rule-based E map from the prior) get ``edge_reg_boost``x
    regression weight: they are ~5-10% of pixels and drown at uniform weight.

    Diagnostics (per batch, window-aggregated by the training loop):
      * stage1/global_in_range — guard coverage, should sit at ~1.0;
      * stage1/local_hit       — GT inside the local window (prior quality);
      * stage1/guard_win_rate  — argmax fell on a global bin (rescue firing);
      * stage1/prior_abs_err   — the prior's own error, the baseline the
        network must beat.
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

    def _edge_weight(self, edge: torch.Tensor) -> torch.Tensor:
        return 1.0 + (self.cfg.edge_reg_boost - 1.0) * edge.clamp(0.0, 1.0)

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
        # GT beyond the scene's physical depth range (DTU backgrounds past
        # depth_values max, ~2.5% of pixels) is unreachable by ANY hypothesis
        # confined to [dmin, dmax]: exclude it from supervision and metrics
        # instead of letting it pollute them as a permanent error floor.
        if "depth_values" in batch:
            dv = batch["depth_values"].to(device).float()
            d_lo = dv.amin(dim=1).view(-1, 1, 1)
            d_hi = dv.amax(dim=1).view(-1, 1, 1)
            in_scene = (depth_gt_full >= d_lo) & (depth_gt_full <= d_hi)
            gt_out_frac = float((~in_scene & (mask_full > 0) & (depth_gt_full > 0)).float().mean())
            mask_full = mask_full * in_scene.float()
        else:
            gt_out_frac = 0.0

        total = outputs["stage1"]["depth"].new_zeros(())
        logs: dict[str, float] = {}

        # ------------------------------ stage 1 ------------------------------ #
        s1 = outputs["stage1"]
        hypos = s1["depth_hypos"]
        logits = s1["logits"]
        hw1 = tuple(hypos.shape[-2:])
        gt1 = self._to_stage_res(depth_gt_full, hw1)
        valid1 = self._to_stage_res(mask_full, hw1).bool() & (gt1 > 0)

        B = hypos.shape[0]
        g_lo = s1["global_lo"].view(B, 1, 1)
        g_hi = s1["global_hi"].view(B, 1, 1)
        g_in_range = (gt1 >= g_lo) & (gt1 <= g_hi)
        l_in_range = (gt1 >= s1["local_lo"]) & (gt1 <= s1["local_hi"])
        sup64 = valid1 & g_in_range
        sup16 = valid1 & l_in_range

        edge1 = s1["edge"]
        w_edge1 = self._edge_weight(edge1)

        l_ce64 = soft_label_cross_entropy(logits, hypos, gt1, sup64) if cfg.use_cross_entropy \
            else logits.new_zeros(())

        logits48 = logits.gather(1, s1["global_idx"])
        hypos48 = hypos.gather(1, s1["global_idx"])
        l_ce48 = soft_label_cross_entropy(logits48, hypos48, gt1, sup64) if cfg.use_cross_entropy \
            else logits.new_zeros(())

        logits16 = logits.gather(1, s1["local_idx"])
        hypos16 = hypos.gather(1, s1["local_idx"])
        l_ce16 = soft_label_cross_entropy(logits16, hypos16, gt1, sup16) if cfg.use_cross_entropy \
            else logits.new_zeros(())

        gi1 = s1["global_interval"].view(B, 1, 1).expand_as(gt1)
        l_reg1 = normalized_huber_loss(s1["depth"], gt1, valid1, gi1, weight=w_edge1)

        total = total + weights["stage1"] * (
            cfg.w_ce * l_ce64
            + cfg.w_global_aux * l_ce48
            + cfg.w_local_aux * l_ce16
            + cfg.w_reg * l_reg1
        )

        # stage-1 diagnostics
        with torch.no_grad():
            winner_local = s1["is_local"].gather(1, s1["mode_idx"]).squeeze(1)
            logs["stage1/ce"] = float(l_ce64.detach())
            logs["stage1/ce_global_aux"] = float(l_ce48.detach())
            logs["stage1/ce_local_aux"] = float(l_ce16.detach())
            logs["stage1/reg"] = float(l_reg1.detach())
            logs["stage1/in_range"] = float(g_in_range[valid1].float().mean()) if valid1.any() else 1.0
            logs["stage1/global_in_range"] = logs["stage1/in_range"]
            logs["stage1/local_hit"] = float(l_in_range[valid1].float().mean()) if valid1.any() else 1.0
            logs["stage1/guard_win_rate"] = float((1.0 - winner_local)[valid1].mean()) if valid1.any() else 0.0
            logs["stage1/p_max"] = float(s1["prob"].detach().amax(dim=1).mean())
            logs["stage1/interval_mm"] = float(s1["interval"][valid1.unsqueeze(1).expand_as(s1["interval"])].mean()) \
                if valid1.any() else 0.0
            logs["stage1/edge_frac"] = float(edge1.mean())
            prior_err = (s1["prior"] - gt1).abs()
            prior_valid = valid1 & (s1["prior"] > 0)
            logs["stage1/prior_abs_err"] = float(prior_err[prior_valid].mean()) if prior_valid.any() else 0.0
            logs["stage1/gt_out_of_scene"] = gt_out_frac
            # supervised-pixel err split by whether the prior was corrupted (the
            # rescue-rate signal); mask arrives at full res from the dataset.
            if "prior_corrupt_mask" in batch:
                cm = self._to_stage_res(batch["prior_corrupt_mask"].to(device).float(), hw1).bool()
                err1 = (s1["depth"].detach() - gt1).abs()
                vc = valid1 & cm
                vk = valid1 & ~cm
                if vc.any():
                    logs["stage1/err_corrupted"] = float(err1[vc].mean())
                if vk.any():
                    logs["stage1/err_clean"] = float(err1[vk].mean())

        # ---------------------------- stages 2 / 3 ---------------------------- #
        for name in ("stage2", "stage3"):
            stage = outputs[name]
            hypos_k = stage["depth_hypos"]
            logits_k = stage["logits"]
            depth_k = stage["depth"]
            hw = tuple(hypos_k.shape[-2:])

            gt = self._to_stage_res(depth_gt_full, hw)
            valid = self._to_stage_res(mask_full, hw).bool() & (gt > 0)

            hypo_min = hypos_k[:, 0]
            hypo_max = hypos_k[:, -1]
            in_range = (gt >= hypo_min) & (gt <= hypo_max)
            sup = valid & in_range
            interval = ((hypo_max - hypo_min) / max(hypos_k.shape[1] - 1, 1)).detach().clamp(min=1e-4)

            w_edge = self._edge_weight(stage["edge"]) if "edge" in stage else None

            l_ce = soft_label_cross_entropy(logits_k, hypos_k, gt, sup) if cfg.use_cross_entropy \
                else depth_k.new_zeros(())
            # ALL valid pixels: even when GT fell outside this stage's window the
            # regression keeps a bounded pull on the previous stage's center.
            l_reg = normalized_huber_loss(depth_k, gt, valid, interval, weight=w_edge)

            total = total + weights[name] * (cfg.w_ce * l_ce + cfg.w_reg * l_reg)

            logs[f"{name}/ce"] = float(l_ce.detach())
            logs[f"{name}/reg"] = float(l_reg.detach())
            logs[f"{name}/in_range"] = float(in_range[valid].float().mean()) if valid.any() else 1.0
            logs[f"{name}/p_max"] = float(stage["prob"].detach().amax(dim=1).mean())
            logs[f"{name}/interval_mm"] = float(interval[valid].mean()) if valid.any() else 0.0

        logs["loss"] = float(total.detach())
        return total, logs
