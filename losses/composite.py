from __future__ import annotations

import torch

from base.config import LossConfig, StageWeights

from .depth_loss import depth_cross_entropy_loss, depth_l1_loss
from .feat_loss import feature_cosine_loss
from .grad_normal import depth_gradient_loss, normal_consistency_loss
from .residual import residual_laplacian_loss
from .ssim import ssim_reprojection_loss


def _phase_weight(step: int, warmup: int) -> float:
    if step < warmup:
        return 0.0
    return 1.0


class MVSLoss:
    def __init__(self, cfg: LossConfig, stage_weights: StageWeights) -> None:
        self.cfg = cfg
        self.stage_weights = stage_weights

    def __call__(
        self,
        outputs: dict,
        batch: dict,
        step: int,
        dino_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        sw = self.stage_weights
        logs: dict[str, float] = {}

        total = outputs["depth_full"].new_zeros(())
        stage_keys = [("stage1", sw.stage1, 8), ("stage2", sw.stage2, 4), ("stage3", sw.stage3, 4)]

        for name, weight, stride in stage_keys:
            stage = outputs[name]
            gt = batch["depth_gt_multiscale"][stride].to(stage["depth"].device)
            mask = batch["mask_multiscale"][stride].to(stage["depth"].device)
            l_d = depth_l1_loss(stage["depth"], gt, mask)
            l_g = depth_gradient_loss(stage["depth"], gt, mask)
            l = cfg.w_depth * l_d + cfg.w_grad * l_g
            if cfg.use_cross_entropy:
                l = l + cfg.w_depth * depth_cross_entropy_loss(
                    stage["prob"], stage["hypos"], gt, mask
                )
            total = total + weight * l
            logs[f"{name}/l_depth"] = float(l_d.detach())
            logs[f"{name}/l_grad"] = float(l_g.detach())

        depth_full = outputs["depth_full"]
        gt_full = batch["depth_gt_full"].to(depth_full.device)
        mask_full = batch["mask_full"].to(depth_full.device)
        K_full = batch["intrinsics"].to(depth_full.device)
        E_full = batch["extrinsics"].to(depth_full.device)

        if _phase_weight(step, cfg.residual_warmup_steps) > 0 and outputs.get("prior") is not None:
            prior_depth = outputs["prior"]["depth_sparse"][:, 0]
            prior_conf = outputs["prior"].get("confidence", None)
            prior_conf_ref = prior_conf[:, 0] if prior_conf is not None else None
            l_res = residual_laplacian_loss(
                depth_full,
                prior_depth,
                mask_full,
                confidence=prior_conf_ref,
                b_scale=cfg.residual_b_scale,
                min_confidence=cfg.residual_min_confidence,
                relative=cfg.residual_relative,
            )
            total = total + cfg.w_residual * l_res
            logs["l_residual"] = float(l_res.detach())

        l_norm = normal_consistency_loss(depth_full, gt_full, K_full[:, 0], mask_full)
        total = total + cfg.w_normal * l_norm
        logs["l_normal"] = float(l_norm.detach())

        if _phase_weight(step, cfg.ssim_warmup_steps) > 0:
            imgs = batch["imgs_raw"].to(depth_full.device)
            l_ssim = ssim_reprojection_loss(depth_full, imgs, K_full, E_full, mask_full)
            total = total + cfg.w_ssim * l_ssim
            logs["l_ssim"] = float(l_ssim.detach())

        if dino_features is None:
            dino_features = outputs.get("dino_features")
        if _phase_weight(step, cfg.feat_warmup_steps) > 0 and dino_features is not None:
            l_feat = feature_cosine_loss(depth_full, dino_features, K_full, E_full, mask_full)
            total = total + cfg.w_feat * l_feat
            logs["l_feat"] = float(l_feat.detach())

        logs["loss"] = float(total.detach().cpu())
        return total, logs
