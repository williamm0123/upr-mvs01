from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from base.config import MVSConfig
from data.dtu import DTUMVSDataset
from data.prior_precompute import ensure_offline_priors
from losses import MVSLoss
from models.mvsnet import UprMVSNet
from utils.logging_utils import MetricMeter, StepTimer, TensorBoardLogger, dump_metrics, get_logger
from utils.path_utils import make_run_dir
from utils.vis import save_depth_vis

from .ddp_utils import (
    barrier,
    cleanup_distributed,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    reduce_scalar_mean,
)
from .evaluator import evaluate


def collate_batch(samples: list[dict]) -> dict:
    out: dict = {}
    for k in samples[0]:
        v = samples[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        elif isinstance(v, dict):
            out[k] = {kk: torch.stack([s[k][kk] for s in samples], dim=0) for kk in v}
        elif isinstance(v, (int, float)):
            out[k] = torch.tensor([s[k] for s in samples])
        else:
            out[k] = [s[k] for s in samples]
    return out


def build_dataloaders(
    cfg: MVSConfig,
    distributed: bool,
    world_size: int,
    rank: int,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    paths = cfg.paths
    use_offline_prior = cfg.vggt_prior.prior_source in ("offline", "auto")
    prior_root = paths.offline_prior_root if use_offline_prior else None
    train_set = DTUMVSDataset(
        datapath=paths.dtu_train_root,
        listfile=paths.train_list_file,
        nviews=cfg.train.num_views,
        target_h=cfg.data.target_h,
        target_w=cfg.data.target_w,
        feature_strides=cfg.data.feature_strides,
        mode="train",
        use_pair_filter=cfg.data.use_pair_filter,
        pair_min_baseline_deg=cfg.data.pair_min_baseline_deg,
        pair_max_baseline_deg=cfg.data.pair_max_baseline_deg,
        prior_root=prior_root,
        prior_confidence=cfg.vggt_prior.offline_confidence,
        require_prior=cfg.vggt_prior.offline_prior_required,
    )
    val_set = DTUMVSDataset(
        datapath=paths.dtu_train_root,
        listfile=paths.val_list_file,
        nviews=cfg.train.num_views,
        target_h=cfg.data.target_h,
        target_w=cfg.data.target_w,
        feature_strides=cfg.data.feature_strides,
        mode="val",
        use_pair_filter=cfg.data.use_pair_filter,
        pair_min_baseline_deg=cfg.data.pair_min_baseline_deg,
        pair_max_baseline_deg=cfg.data.pair_max_baseline_deg,
        prior_root=prior_root,
        prior_confidence=cfg.vggt_prior.offline_confidence,
        require_prior=cfg.vggt_prior.offline_prior_required,
    )

    train_sampler: DistributedSampler | None = None
    if distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.train.batch_size,
            sampler=train_sampler,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
            drop_last=True,
            persistent_workers=cfg.train.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
            drop_last=True,
            persistent_workers=cfg.train.num_workers > 0,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, cfg.train.num_workers // 2),
        collate_fn=collate_batch,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


class Trainer:
    def __init__(self, cfg: MVSConfig, run_name: str = "uprmvs") -> None:
        self.cfg = cfg
        self.distributed = bool(cfg.train.distributed)

        if self.distributed:
            self.rank, self.world_size, self.local_rank = init_distributed()
        else:
            self.rank, self.world_size, self.local_rank = 0, 1, 0

        if torch.cuda.is_available():
            dev_id = self.local_rank if self.distributed else cfg.train.devices[0]
            self.device = torch.device(f"cuda:{dev_id}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.is_main = is_main_process()

        if self.is_main:
            self.run_dir = make_run_dir(f"{run_name}_{cfg.train.profile}")
            self.logger = get_logger("trainer", self.run_dir / "log" / "train.log")
            self.tb: TensorBoardLogger | None = TensorBoardLogger(self.run_dir / "tb")
            self.logger.info(f"Profile: {cfg.train.profile} | device: {self.device}")
            self.logger.info(f"Distributed: {self.distributed} world_size={self.world_size} rank={self.rank}")
            self.logger.info(f"Run dir: {self.run_dir}")
            self.logger.info(f"TensorBoard: {self.tb.log_dir}")
        else:
            self.run_dir = None
            self.logger = get_logger(f"trainer-rank{self.rank}")
            self.tb = None

        torch.manual_seed(cfg.train.seed + self.rank)

        self._ensure_offline_prior_cache()

        self.model = UprMVSNet(cfg, device=self.device).to(self.device)
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

        self.loss_fn = MVSLoss(cfg.loss, cfg.stage_weights)
        self.train_loader, self.val_loader, self.train_sampler = build_dataloaders(
            cfg, self.distributed, self.world_size, self.rank
        )
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.scaler = GradScaler(enabled=cfg.train.amp and self.device.type == "cuda")

    def _ensure_offline_prior_cache(self) -> None:
        should_check = (
            self.cfg.train.use_vggt_prior
            and self.cfg.vggt_prior.prior_source in ("offline", "auto")
            and self.cfg.vggt_prior.generate_missing_offline
        )
        if not should_check:
            return
        if self.is_main:
            ensure_offline_priors(self.cfg, device=self.device, logger=self.logger)
        if self.distributed:
            barrier()

    def _unwrap(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _lr_at(self, step: int) -> float:
        warmup = self.cfg.train.warmup_steps
        if step < warmup:
            return self.cfg.train.lr * (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(self.cfg.train.max_steps - warmup, 1)
        return self.cfg.train.lr * max(0.05, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()))

    def _save_ckpt(self, step: int, tag: str = "step") -> Path | None:
        if not self.is_main:
            return None
        path = self.run_dir / "ckpt" / f"{tag}_{step:07d}.pt"
        torch.save(
            {
                "step": step,
                "model": self._unwrap().state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "cfg_profile": self.cfg.train.profile,
            },
            path,
        )
        return path

    def _vis_step(self, batch: dict, outputs: dict, step: int) -> None:
        if not self.is_main or self.tb is None:
            return
        with torch.no_grad():
            depth_pred = outputs["depth_full"][0]
            depth_gt = batch["depth_gt_full"][0]
            mask_gt = batch["mask_full"][0]
            target_hw = tuple(depth_gt.shape[-2:])
            save_depth_vis(depth_pred, self.run_dir / "vis" / f"step{step:07d}_pred.png")
            save_depth_vis(depth_gt, self.run_dir / "vis" / f"step{step:07d}_gt.png")

            valid = (mask_gt.bool() & (depth_gt > 0)).cpu().numpy()
            gt_np = depth_gt.detach().cpu().numpy()
            vmin = float(gt_np[valid].min()) if valid.any() else None
            vmax = float(gt_np[valid].max()) if valid.any() else None
            depth_span = max((vmax or 1.0) - (vmin or 0.0), 1.0)
            err_vmax = max(depth_span * 0.1, 1.0)
            sigma_vmax = max(depth_span * 0.05, 1.0)

            imgs_raw = batch.get("imgs_raw")
            imgs_norm = batch.get("imgs")
            if imgs_raw is not None:
                for view_idx in range(min(imgs_raw.shape[1], self.cfg.train.vis_max_views)):
                    self.tb.add_image_raw(f"vis/images/view{view_idx}", imgs_raw[0, view_idx], step)
            elif imgs_norm is not None:
                for view_idx in range(min(imgs_norm.shape[1], self.cfg.train.vis_max_views)):
                    self.tb.add_image_norm(f"vis/images/view{view_idx}", imgs_norm[0, view_idx], step)

            self.tb.add_depth("vis/final/depth_pred", depth_pred, step, vmin=vmin, vmax=vmax)
            self.tb.add_depth("vis/final/depth_gt", depth_gt, step, vmin=vmin, vmax=vmax)
            err = (depth_pred - depth_gt).abs() * mask_gt
            self.tb.add_depth("vis/final/error_abs", err, step, vmin=0.0, vmax=err_vmax)

            if batch.get("prior") is not None:
                offline_prior_d = batch["prior"]["depth_sparse"][0, 0]
                offline_prior_c = batch["prior"]["confidence"][0, 0]
                offline_prior_m = batch["prior"]["valid_mask"][0, 0]
                save_depth_vis(offline_prior_d, self.run_dir / "vis" / f"step{step:07d}_offline_prior.png")
                self.tb.add_depth("vis/prior/offline_filled_depth", offline_prior_d, step, vmin=vmin, vmax=vmax)
                self.tb.add_depth("vis/prior/offline_confidence", offline_prior_c, step, vmin=0.0, vmax=1.0)
                self.tb.add_depth("vis/prior/offline_valid_mask", offline_prior_m, step, vmin=0.0, vmax=1.0)
                offline_err = (offline_prior_d - depth_gt).abs() * mask_gt * offline_prior_m
                self.tb.add_depth("vis/prior/offline_error_abs", offline_err, step, vmin=0.0, vmax=err_vmax)

            used_prior = outputs.get("prior")
            if used_prior is not None:
                used_prior_d = used_prior["depth_sparse"][0, 0]
                used_prior_c = used_prior["confidence"][0, 0]
                save_depth_vis(used_prior_d, self.run_dir / "vis" / f"step{step:07d}_used_prior.png")
                self.tb.add_depth("vis/prior/used_depth", used_prior_d, step, vmin=vmin, vmax=vmax)
                self.tb.add_depth("vis/prior/used_confidence", used_prior_c, step, vmin=0.0, vmax=1.0)

            for stage_name in ("stage1", "stage2", "stage3"):
                stage = outputs[stage_name]
                stage_depth = self._resize_vis_map(stage["depth"][0], target_hw)
                stage_err = (stage_depth - depth_gt).abs() * mask_gt
                self.tb.add_depth(f"vis/{stage_name}/depth", stage_depth, step, vmin=vmin, vmax=vmax)
                self.tb.add_depth(f"vis/{stage_name}/error_abs", stage_err, step, vmin=0.0, vmax=err_vmax)
                if "sigma" in stage:
                    stage_sigma = self._resize_vis_map(stage["sigma"][0], target_hw)
                    self.tb.add_depth(f"vis/{stage_name}/sigma", stage_sigma, step, vmin=0.0, vmax=sigma_vmax)
                if "prob" in stage:
                    stage_conf = stage["prob"][0].amax(dim=0)
                    stage_conf = self._resize_vis_map(stage_conf, target_hw)
                    self.tb.add_depth(f"vis/{stage_name}/prob_max", stage_conf, step, vmin=0.0, vmax=1.0)

            self.tb.flush()

    @staticmethod
    def _resize_vis_map(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if tuple(x.shape[-2:]) == target_hw:
            return x
        return F.interpolate(
            x.float().unsqueeze(0).unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    def _log_tb_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        if self.is_main and self.tb is not None:
            self.tb.add_scalars(prefix, values, step)

    def train(self) -> None:
        cfg = self.cfg
        step = 0
        epoch = 0
        meter = MetricMeter()
        timer = StepTimer()
        self.model.train()

        while step < cfg.train.max_steps:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            for batch in self.train_loader:
                if step >= cfg.train.max_steps:
                    break
                batch = move_batch_to_device(batch, self.device)

                lr = self._lr_at(step)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=cfg.train.amp and self.device.type == "cuda"):
                    outputs = self.model(batch, step=step)
                    loss, logs = self.loss_fn(outputs, batch, step=step)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.train.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                meter.update(**logs, lr=lr)
                self._log_tb_scalars("train", logs, step + 1)
                if self.is_main and self.tb is not None:
                    self.tb.add_scalar("train/lr", lr, step + 1)

                if (step + 1) % cfg.train.log_interval == 0:
                    avg = meter.avg()
                    dt = timer.tick()
                    avg_synced = {k: reduce_scalar_mean(v, self.device) for k, v in avg.items()}
                    if self.is_main:
                        msg = " ".join(f"{k}={v:.4f}" for k, v in avg_synced.items())
                        self.logger.info(f"step={step+1} dt={dt:.2f}s {msg}")
                        if self.tb is not None:
                            self.tb.add_scalar("train/step_time_sec", dt / max(cfg.train.log_interval, 1), step + 1)
                    meter.reset()

                if cfg.train.vis_interval > 0 and (step + 1) % cfg.train.vis_interval == 0:
                    self._vis_step(batch, outputs, step + 1)

                if (step + 1) % cfg.train.val_interval == 0:
                    if cfg.train.vis_interval <= 0 or (step + 1) % cfg.train.vis_interval != 0:
                        self._vis_step(batch, outputs, step + 1)
                    if self.is_main:
                        val_metrics = evaluate(self._unwrap(), self.val_loader, self.device, max_batches=16)
                        self.logger.info(f"[val step={step+1}] " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                        if self.tb is not None:
                            self.tb.add_scalars("val", val_metrics, step + 1)
                            self.tb.flush()
                        dump_metrics(self.run_dir / "log" / f"val_{step+1:07d}.json", val_metrics)
                    barrier()
                    self.model.train()

                if (step + 1) % cfg.train.ckpt_interval == 0:
                    path = self._save_ckpt(step + 1)
                    if path is not None:
                        self.logger.info(f"checkpoint -> {path}")
                    barrier()

                step += 1
            epoch += 1

        self._save_ckpt(step, tag="final")
        if self.is_main and self.tb is not None:
            self.tb.flush()
            self.tb.close()
            self.logger.info("training complete")
        cleanup_distributed()
