"""Training entry point for UprMVSNet.

Single-card and multi-card DistributedDataParallel training.  Multi-GPU jobs
should normally be launched with ``torchrun``:

    # single GPU
    python train.py --profile umhpc --gpus 1

    # 4 GPU DDP on one node
    torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py

Direct ``python`` multi-GPU launching is retained for backwards compatibility:

    python train.py --profile umhpc --gpus 4 --ddp on

    # pick specific device ids
    python train.py --devices 0,2 --ddp on

    # DDP off regardless of gpu count
    python train.py --gpus 4 --ddp off

    # validate the full model + loss + DDP + logging path on synthetic data
    python train.py --gpus 2 --ddp on --smoke

When launched by ``torchrun``, rank, world size, and the CUDA device are read
from ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK``.  ``--ddp auto`` applies only
to direct ``python`` launches and turns DDP on iff the selected GPU count > 1.

Artifacts (under <project>/log/):
    log/prior_cache/   precomputed {depth_prior, conf_prior, norm_depth_fill, src_weights}
    log/tensorboard/   TensorBoard event files (loss / lr / depth metrics / images)
    log/model/         latest.pth (every ckpt/val interval) and best.pth (best val abs_err)

Resume: ``--resume auto`` (default) continues from log/model/latest.pth when it
exists (model + optimizer + step + best metric), so a walltime-killed job can
simply be resubmitted. Pass ``--resume off`` to start fresh.

Validation: every ``val_interval`` steps the val split (cfg.paths.val_list_file)
is evaluated on all ranks (metrics all-reduced); best.pth tracks the lowest
validation abs_err. Val-scan priors must exist in the cache -- build them once
with ``--build-priors only`` (single process) before the first DDP run.

NOTE: real (non-smoke) training expects each dataset sample to carry the prior
keys the network consumes -- ``depth_prior`` / ``conf_prior`` (from norm_fill),
``images`` / ``intrinsics`` / ``extrinsics`` / ``depth_values`` (from dtu),
``depth_gt`` / ``mask`` (for the loss), and optionally ``src_weights``. The prior
keys are produced automatically by the offline precompute (models/pre_prior.py).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from base.config import ProjectPaths, build_mvs_config
from losses import MVSLoss
from models.network import UprMVSNet

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard not installed
    SummaryWriter = None


# --------------------------------------------------------------------------- #
# Device / DDP resolution
# --------------------------------------------------------------------------- #
def _parse_devices(args) -> list[int]:
    if args.devices:
        ids = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    else:
        ids = list(range(max(1, args.gpus)))
    if torch.cuda.is_available():
        ids = [i for i in ids if i < torch.cuda.device_count()] or [0]
    else:
        ids = [0]  # CPU: a single logical device
    return ids


def _use_ddp(args, world_size: int) -> bool:
    if args.ddp == "on":
        return world_size > 1
    if args.ddp == "off":
        return False
    return world_size > 1  # auto


def _lr_at(cfg, step: int, max_steps: int) -> float:
    """Linear warmup then cosine decay to 5% of the base LR.

    ``max_steps`` is the *effective* horizon (--steps override included), so the
    schedule always anneals within the steps that will actually run.
    """
    warm = cfg.train.warmup_steps
    if step < warm:
        return cfg.train.lr * (step + 1) / max(warm, 1)
    prog = (step - warm) / max(max_steps - warm, 1)
    return cfg.train.lr * max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0))))


# --------------------------------------------------------------------------- #
# Depth metrics (DTU-style thresholds) + TensorBoard logging / checkpointing
# --------------------------------------------------------------------------- #
def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    m = mask.bool() & (gt > 0)
    if not m.any():
        return {}
    err = (pred[m] - gt[m]).abs()
    return {
        "abs_err": err.mean().item(),
        "acc_2mm": (err < 2).float().mean().item(),
        "acc_4mm": (err < 4).float().mean().item(),
        "acc_8mm": (err < 8).float().mean().item(),
    }


def _norm_map(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    x = (x.float() - vmin) / (vmax - vmin + 1e-8)
    return x.clamp(0, 1).unsqueeze(0)  # [1, H, W]


class TrainLogger:
    """TensorBoard scalars/images (MVSFormer++-style) + latest/best checkpoints.

    Every summary tag lives below ``train/``.  TensorBoard uses the text before
    the first slash as the card group, so this keeps all training charts and
    images expanded together in one ``train`` grid instead of creating one
    collapsible row per tag.
    """

    TAG_PREFIX = "train"

    @classmethod
    def _tag(cls, name: str) -> str:
        return f"{cls.TAG_PREFIX}/{name}"

    def __init__(self, run_name: str, enabled: bool) -> None:
        self.enabled = enabled
        self.best_metric = float("inf")
        if not enabled:
            return
        log_root = ProjectPaths().project_path / "log"
        self.model_dir = log_root / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Timestamp the run subdir so repeated runs with the same --name land in
        # distinct TensorBoard directories instead of piling onto one another.
        run_dir = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tb_dir = log_root / "tensorboard" / run_dir
        self.tb = SummaryWriter(str(tb_dir)) if SummaryWriter else None
        if self.tb is not None:
            print(f"[tensorboard] logging to {tb_dir}")

    def log_scalars(self, logs: dict, lr: float, metrics: dict, step: int) -> None:
        if not self.enabled or self.tb is None:
            return
        self.tb.add_scalar(self._tag("loss_total"), logs["loss"], step)
        for name in ("stage1", "stage2", "stage3"):
            self.tb.add_scalar(self._tag(f"loss_{name}_ce"), logs[f"{name}/ce"], step)
            self.tb.add_scalar(self._tag(f"loss_{name}_reg"), logs[f"{name}/reg"], step)
            # diag_*_in_range: fraction of valid pixels whose GT the stage's
            #   hypothesis range covers (should climb to >0.95 and stay there);
            # diag_*_p_max: prob-volume sharpness; diag_*_interval_mm: bin width.
            for diag in ("in_range", "p_max", "interval_mm"):
                key = f"{name}/{diag}"
                if key in logs:
                    self.tb.add_scalar(self._tag(f"diag_{name}_{diag}"), logs[key], step)
        self.tb.add_scalar(self._tag("learning_rate"), lr, step)
        # metric_abs_err_mm : single overall mean-|pred-gt| in mm (no per-scale
        #   variant -- one number over all valid pixels).
        # metric_acc_{2,4,8}mm : the *scale* metrics -- fraction of pixels whose
        #   error is below 2 / 4 / 8 mm. Logged as plain scalars (one chart each,
        #   in this run) so there is no ambiguity about which scale is which and
        #   no add_scalars sub-run folders.
        if "abs_err" in metrics:
            self.tb.add_scalar(self._tag("metric_abs_err"), metrics["abs_err"], step)
        for thr in ("2mm", "4mm", "8mm"):
            key = f"acc_{thr}"
            if key in metrics:
                self.tb.add_scalar(self._tag(f"metric_acc_{thr}"), metrics[key], step)

    def log_images(self, batch: dict, outputs: dict, step: int) -> None:
        if not self.enabled or self.tb is None:
            return
        depth_pred = outputs["depth_full"][0].detach()
        depth_gt = batch["depth_gt"][0].float()
        depth_prior = batch["depth_prior"][0].detach().float()
        mask = batch["mask"][0].bool() & (depth_gt > 0)
        if mask.any():
            vmin = float(depth_gt[mask].min())
            vmax = float(depth_gt[mask].max())
        else:
            vmin, vmax = 0.0, 1.0

        # TensorBoard sorts cards by tag, so numeric prefixes keep this visual
        # comparison in a stable left-to-right order.
        self.tb.add_image(
            self._tag("01_ref_image"),
            batch["images"][0, 0].detach().float() / 255.0,
            step,
        )
        self.tb.add_image(self._tag("02_depth_gt"), _norm_map(depth_gt, vmin, vmax), step)
        self.tb.add_image(
            self._tag("03_depth_prior"),
            _norm_map(depth_prior, vmin, vmax),
            step,
        )
        self.tb.add_image(self._tag("04_depth_pred"), _norm_map(depth_pred, vmin, vmax), step)
        prob = outputs["stage3"]["prob"][0].detach().amax(dim=0)  # confidence
        self.tb.add_image(self._tag("05_stage3_confidence"), _norm_map(prob, 0.0, 1.0), step)

    def log_val(self, metrics: dict[str, float], step: int) -> None:
        if not self.enabled or self.tb is None:
            return
        for key, value in metrics.items():
            self.tb.add_scalar(f"val/{key}", value, step)

    def save(self, model, optimizer, step: int, val_metric: float | None = None) -> None:
        """Write latest.pth; when a validation metric is supplied and improves
        on the best seen so far, also write best.pth. best.pth therefore tracks
        the *validation* abs_err, never the (noisy single-batch) train loss."""
        if not self.enabled:
            return
        is_best = val_metric is not None and val_metric < self.best_metric
        if is_best:
            self.best_metric = float(val_metric)
        state = (model.module if isinstance(model, DDP) else model).state_dict()
        ckpt = {
            "step": step,
            "model": state,
            "optimizer": optimizer.state_dict(),
            "best_metric": self.best_metric,
        }
        torch.save(ckpt, self.model_dir / "latest.pth")
        if is_best:
            torch.save(ckpt, self.model_dir / "best.pth")
            print(f"[ckpt] new best (val abs_err={val_metric:.4f}) -> {self.model_dir/'best.pth'}")

    def close(self) -> None:
        if self.enabled and self.tb is not None:
            self.tb.flush()
            self.tb.close()


# --------------------------------------------------------------------------- #
# Synthetic batch (for --smoke: exercises the whole path without a dataset)
# --------------------------------------------------------------------------- #
def _synthetic_batch(cfg, device: torch.device, batch_size: int) -> dict:
    B, V = batch_size, cfg.train.num_views
    H, W = 256, 320
    dmin, interval, nd = 425.0, 2.5, 192
    dv = torch.from_numpy(np.arange(dmin, dmin + interval * nd, interval, dtype=np.float32))
    batch = {
        "images": torch.rand(B, V, 3, H, W) * 255.0,
        "intrinsics": torch.tensor([[[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]]]).repeat(B, V, 1, 1),
        "extrinsics": torch.eye(4).repeat(B, V, 1, 1),
        "depth_prior": torch.rand(B, H, W) * 100 + dmin,
        "conf_prior": torch.rand(B, H, W),
        "depth_values": dv.unsqueeze(0).repeat(B, 1),
        "depth_gt": torch.rand(B, H, W) * 400 + dmin,
        "mask": (torch.rand(B, H, W) > 0.2).float(),
    }
    batch["extrinsics"][:, 1:, 0, 3] = 5.0
    return {k: v.to(device) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# Prior precompute
# --------------------------------------------------------------------------- #
def _ensure_priors(cfg, device, overwrite: bool = False) -> None:
    """Offline-precompute {depth_prior, conf_prior, norm_depth_fill, src_weights}
    for the train + val splits and cache to log/prior_cache (idempotent)."""
    from data.dtu import DTUMVSDataset
    from models.pre_prior import build_prior_cache

    for split, listfile, mode in [
        ("train", cfg.paths.train_list_file, "train"),
        ("val", cfg.paths.val_list_file, "val"),
    ]:
        ds = DTUMVSDataset(
            datapath=cfg.paths.dtu_train_root,
            listfile=listfile,
            nviews=cfg.train.num_views,
            mode=mode,
        )
        print(f"[pre_prior] ensuring priors for {split} split ({len(ds)} samples)")
        build_prior_cache(ds, device, overwrite=overwrite)


# --------------------------------------------------------------------------- #
# Per-process worker
# --------------------------------------------------------------------------- #
def main_worker(
    rank: int,
    world_size: int,
    device_ids: list[int],
    args,
    local_rank: int | None = None,
) -> None:
    cfg = build_mvs_config(profile=args.profile)
    train_overrides = {}
    if args.batch_size is not None:
        train_overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_overrides["num_workers"] = args.num_workers
    if args.num_views is not None:
        train_overrides["num_views"] = args.num_views
    if args.lr is not None:
        train_overrides["lr"] = args.lr
    if args.warmup_steps is not None:
        train_overrides["warmup_steps"] = args.warmup_steps
    if args.amp is not None:
        train_overrides["amp"] = args.amp == "on"
    if train_overrides:
        cfg = replace(cfg, train=replace(cfg.train, **train_overrides))

    is_ddp = world_size > 1
    is_main = rank == 0

    if is_ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if local_rank is not None:
            # torchrun supplies rendezvous information and ranks through the
            # environment; passing them again can conflict with elastic launch.
            dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        device_id = local_rank if local_rank is not None else device_ids[rank]
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    torch.manual_seed(cfg.train.seed + rank)

    # Build the prior cache once (rank 0), before the training model is on GPU so
    # VGGT/DA3 are freed first. Other ranks wait at the barrier.
    if not args.smoke and args.build_priors != "skip":
        if is_main:
            _ensure_priors(cfg, device, overwrite=(args.build_priors == "force"))
        if is_ddp:
            dist.barrier()
        if args.build_priors == "only":
            if is_main:
                print("[pre_prior] cache complete (--build-priors only); exiting before training")
            if is_ddp:
                dist.destroy_process_group()
            return

    model = UprMVSNet(cfg).to(device)
    if is_ddp:
        ddp_ids = [device.index] if device.type == "cuda" else None
        out_dev = device.index if device.type == "cuda" else None
        model = DDP(model, device_ids=ddp_ids, output_device=out_dev, find_unused_parameters=False)

    loss_fn = MVSLoss(cfg.loss, cfg.stage_weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_name = args.name + ("_smoke" if args.smoke else "")
    logger = TrainLogger(run_name, enabled=is_main)

    # Resume from latest.pth so a walltime-killed job continues where it left
    # off (model + optimizer + step + best val metric). Every rank loads the
    # same file; the LR is recomputed from the step counter, so nothing else
    # needs restoring.
    start_step = 0
    if not args.smoke and args.resume == "auto":
        ckpt_path = ProjectPaths().project_path / "log" / "model" / "latest.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt.get("step", -1)) + 1
            logger.best_metric = float(ckpt.get("best_metric", float("inf")))
            if is_main:
                print(f"[resume] loaded {ckpt_path} -> continuing from step {start_step} "
                      f"(best val abs_err so far: {logger.best_metric:.4f})")

    if is_main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[rank {rank}] device={device} ddp={is_ddp} world_size={world_size} "
              f"params={n_params:.1f}M profile={cfg.train.profile} "
              f"batch={cfg.train.batch_size} views={cfg.train.num_views} "
              f"workers={cfg.train.num_workers} lr={cfg.train.lr:g} "
              f"warmup={cfg.train.warmup_steps} amp={cfg.train.amp}")

    if args.smoke:
        _run_smoke(model, loss_fn, optimizer, scaler, cfg, device, args, logger, is_main)
    else:
        _run_training(model, loss_fn, optimizer, scaler, cfg, device, args, world_size, rank, is_ddp, logger, is_main, start_step)

    logger.close()
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


def _train_step(model, loss_fn, optimizer, scaler, batch, cfg, device, step, use_amp):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        outputs = model(batch, step=step)
        loss, logs = loss_fn(outputs, batch, step=step)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip
    )
    scaler.step(optimizer)
    scaler.update()
    return logs, outputs


def _run_smoke(model, loss_fn, optimizer, scaler, cfg, device, args, logger, is_main):
    model.train()
    use_amp = cfg.train.amp and device.type == "cuda"
    for step in range(args.smoke_steps):
        batch = _synthetic_batch(cfg, device, cfg.train.batch_size)
        logs, outputs = _train_step(model, loss_fn, optimizer, scaler, batch, cfg, device, step, use_amp)
        if is_main:
            metrics = depth_metrics(outputs["depth_full"], batch["depth_gt"], batch["mask"])
            logger.log_scalars(logs, cfg.train.lr, metrics, step)
            logger.log_images(batch, outputs, step)
            logger.save(model, optimizer, step, val_metric=logs["loss"])
            print(f"[smoke step {step}] loss={logs['loss']:.4f} abs_err={metrics.get('abs_err', float('nan')):.2f}")
    if is_main:
        print("[smoke] OK - model + loss + backward + tensorboard + ckpt path verified")


@torch.no_grad()
def _run_validation(model, loader, device, use_amp, is_ddp) -> dict[str, float]:
    """Masked depth metrics over the val split.

    All ranks run their DistributedSampler shard and the pixel-weighted sums are
    all-reduced, so the result is identical on every rank (and SyncBN-safe,
    should it ever be enabled)."""
    model.eval()
    # [abs_err_sum, pixel_count, hits<2mm, hits<4mm, hits<8mm]
    stats = torch.zeros(5, device=device, dtype=torch.float64)
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
        pred = outputs["depth_full"].float()
        gt = batch["depth_gt"].float()
        m = batch["mask"].bool() & (gt > 0)
        if m.any():
            err = (pred[m] - gt[m]).abs()
            stats[0] += err.sum()
            stats[1] += m.sum()
            stats[2] += (err < 2).sum()
            stats[3] += (err < 4).sum()
            stats[4] += (err < 8).sum()
    if is_ddp:
        dist.all_reduce(stats)
    model.train()
    n = stats[1].clamp(min=1)
    return {
        "abs_err": float(stats[0] / n),
        "acc_2mm": float(stats[2] / n),
        "acc_4mm": float(stats[3] / n),
        "acc_8mm": float(stats[4] / n),
    }


def _run_training(model, loss_fn, optimizer, scaler, cfg, device, args, world_size, rank, is_ddp, logger, is_main, start_step=0):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from data.dtu import DTUMVSDataset

    dataset = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=cfg.paths.train_list_file,
        nviews=cfg.train.num_views,
        mode="train",
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
    )

    # Validation: deterministic split (center crop, one light condition); used
    # to select best.pth. Never overlaps the training scans.
    val_dataset = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=cfg.paths.val_list_file,
        nviews=cfg.train.num_views,
        mode="val",
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=False,
    )

    max_steps = args.steps if args.steps else cfg.train.max_steps
    model.train()
    use_amp = cfg.train.amp and device.type == "cuda"
    step = start_step
    epoch = start_step // max(len(loader), 1)
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            lr = _lr_at(cfg, step, max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            logs, outputs = _train_step(model, loss_fn, optimizer, scaler, batch, cfg, device, step, use_amp)

            if is_main and step % cfg.train.log_interval == 0:
                metrics = depth_metrics(outputs["depth_full"], batch["depth_gt"], batch["mask"])
                logger.log_scalars(logs, lr, metrics, step)
                print(
                    f"[step {step}] loss={logs['loss']:.4f} "
                    f"abs_err={metrics.get('abs_err', float('nan')):.2f} "

                )
            if is_main and cfg.train.vis_interval > 0 and step % cfg.train.vis_interval == 0:
                logger.log_images(batch, outputs, step)
            # Validation runs on ALL ranks (metrics are all-reduced); the
            # elif keeps ckpt_interval multiples of val_interval from double-saving.
            if step > 0 and cfg.train.val_interval > 0 and step % cfg.train.val_interval == 0:
                val_metrics = _run_validation(model, val_loader, device, use_amp, is_ddp)
                if is_main:
                    logger.log_val(val_metrics, step)
                    logger.save(model, optimizer, step, val_metric=val_metrics["abs_err"])
                    print(f"[val step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
            elif is_main and step > 0 and step % cfg.train.ckpt_interval == 0:
                logger.save(model, optimizer, step)
            step += 1
        epoch += 1

    # Final validation so the last weights are also considered for best.pth.
    val_metrics = _run_validation(model, val_loader, device, use_amp, is_ddp)
    if is_main:
        logger.log_val(val_metrics, step)
        logger.save(model, optimizer, step, val_metric=val_metrics["abs_err"])
        print(f"[val final step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))


def _collate(samples: list[dict]) -> dict:
    out: dict = {}
    for k in samples[0]:
        v = samples[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        elif isinstance(v, np.ndarray):
            out[k] = torch.stack([torch.from_numpy(s[k]) for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train UprMVSNet (single / multi-GPU DDP)")
    parser.add_argument("--profile", choices=["local", "umhpc"], default=None)
    parser.add_argument("--name", default="uprmvs", help="run name (tensorboard subdir)")
    parser.add_argument("--ddp", choices=["auto", "on", "off"], default="auto",
                        help="auto: DDP iff >1 GPU; on: force DDP; off: never DDP")
    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs (ignored if --devices given)")
    parser.add_argument("--devices", type=str, default="", help="explicit CUDA ids, e.g. '0,1,2,3'")
    parser.add_argument("--steps", type=int, default=0, help="override max training steps (0 = config default)")
    parser.add_argument("--batch-size", type=int, default=None, help="per-GPU batch size override")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count override")
    parser.add_argument("--num-views", type=int, default=None, help="number of MVS input views override")
    parser.add_argument("--lr", type=float, default=None, help="learning-rate override")
    parser.add_argument("--warmup-steps", type=int, default=None, help="LR warmup steps override")
    parser.add_argument("--amp", choices=["on", "off"], default=None, help="AMP override")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--resume", choices=["auto", "off"], default="auto",
                        help="auto: continue from log/model/latest.pth if present; off: always start fresh")
    parser.add_argument("--build-priors", choices=["auto", "force", "skip", "only"], default="auto",
                        help="auto: precompute missing priors; force: recompute all; "
                             "skip: assume cached; only: build missing priors then exit")
    parser.add_argument("--smoke", action="store_true", help="run synthetic steps to validate the pipeline")
    parser.add_argument("--smoke-steps", type=int, default=3)
    args = parser.parse_args()

    # torchrun creates one process per GPU and assigns each process a local CUDA
    # rank.  Do not enter the legacy mp.spawn path in this mode.
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if rank == 0:
            print(
                f"launch: torchrun ddp={world_size > 1} "
                f"world_size={world_size} mode={'smoke' if args.smoke else 'train'}"
            )
        main_worker(rank, world_size, [], args, local_rank=local_rank)
        return

    device_ids = _parse_devices(args)
    world_size = len(device_ids)
    use_ddp = _use_ddp(args, world_size)
    if not use_ddp:
        device_ids = device_ids[:1]
        world_size = 1

    print(f"launch: ddp={use_ddp} world_size={world_size} devices={device_ids} "
          f"mode={'smoke' if args.smoke else 'train'}")

    if use_ddp and world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = args.master_port
        mp.spawn(main_worker, args=(world_size, device_ids, args), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, device_ids, args)


if __name__ == "__main__":
    main()
