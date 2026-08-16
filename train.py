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

from pathlib import Path
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

from base.config import resolve_split, ProjectPaths, build_mvs_config
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


class WindowedMeter:
    """Cross-rank, window-aggregated training metrics.

    A single batch of 3 samples on rank 0 is far too noisy to read training
    health from (the old curves' "oscillation" was mostly this). Every rank
    accumulates pixel-weighted sums each step; at flush they are all-reduced,
    so the logged point reflects the whole window across the whole world size.
    Also tracks the prior's own error (the baseline the network must beat) and
    the corrupted/clean split (the guard-rescue signal), plus the median/P90 of
    per-batch mean error, pooled over all ranks, for spike visibility.

    The loss's own scalars (loss, per-stage ce/reg, spre/*, in_range, ...) are
    merged across ranks too, each with its own count: the loss emits several
    diagnostics only when their mask is non-empty (``spre/r_corrupt``,
    ``stageN/oor_recoverable``, ...), so a rank that skipped a key must not be
    averaged as if it had reported zero. That also makes the key set
    rank-dependent, which is why the merge goes through ``all_gather_object``
    over the union instead of a fixed-layout ``all_reduce`` — a per-rank length
    mismatch in a tensor collective would hang the job rather than log a wrong
    number.
    """

    # sums layout: [err, n, hit2, hit4, hit8, prior_err, prior_n,
    #               corrupt_err, corrupt_n, clean_err, clean_n,
    #               body2_err, body4_err, body8_err]
    # body{n}_err sums the error of pixels already within n mm; divided by the
    # matching hit{n} count it gives the *precision on matched pixels*, which the
    # plain mean cannot show (a few percent of far-off pixels dominate it). This
    # is MVSFormer++'s abs_depth_thres0-{n}mm_error — their DTU-test value is
    # 0.4920 mm at 2 mm — and it is the only way to tell a refinement gain from
    # an outlier-count gain.
    N_SLOTS = 14

    def __init__(self, device: torch.device, is_ddp: bool) -> None:
        self.device = device
        self.is_ddp = is_ddp
        self.reset()

    def reset(self) -> None:
        self.sums = torch.zeros(self.N_SLOTS, dtype=torch.float64, device=self.device)
        self.log_sums: dict[str, float] = {}
        self.log_counts: dict[str, int] = {}
        self.batch_means: list[float] = []

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        prior: torch.Tensor | None = None,
        corrupt_mask: torch.Tensor | None = None,
        logs: dict | None = None,
    ) -> None:
        m = mask.bool() & (gt > 0)
        if m.any():
            err = (pred.float() - gt.float()).abs()
            e = err[m]
            self.sums[0] += e.sum()
            self.sums[1] += m.sum()
            for i, t in enumerate((2.0, 4.0, 8.0)):
                hit = e < t
                self.sums[2 + i] += hit.sum()
                self.sums[11 + i] += e[hit].sum()
            self.batch_means.append(float(e.mean()))
            if prior is not None:
                pm = m & (prior > 0)
                if pm.any():
                    self.sums[5] += (prior.float() - gt.float()).abs()[pm].sum()
                    self.sums[6] += pm.sum()
            if corrupt_mask is not None:
                cm = corrupt_mask.bool()
                mc, mk = m & cm, m & ~cm
                if mc.any():
                    self.sums[7] += err[mc].sum()
                    self.sums[8] += mc.sum()
                if mk.any():
                    self.sums[9] += err[mk].sum()
                    self.sums[10] += mk.sum()
        if logs:
            for k, v in logs.items():
                self.log_sums[k] = self.log_sums.get(k, 0.0) + float(v)
                self.log_counts[k] = self.log_counts.get(k, 0) + 1

    def flush(self) -> tuple[dict[str, float], dict[str, float]]:
        """All-reduce (collective — every rank must call this at the same step)
        and return (metrics, window-averaged loss logs)."""
        s = self.sums.clone()
        if self.is_ddp:
            dist.all_reduce(s)
        n = s[1].clamp(min=1)
        metrics = {
            "abs_err": float(s[0] / n),
            "acc_2mm": float(s[2] / n),
            "acc_4mm": float(s[3] / n),
            "acc_8mm": float(s[4] / n),
            # precision on already-matched pixels (vs MVSFormer++ 0.4920 at 2mm)
            "abs_err_body_2mm": float(s[11] / s[2].clamp(min=1)),
            "abs_err_body_4mm": float(s[12] / s[3].clamp(min=1)),
            # the >8mm tail, measured directly instead of back-solved from the mean
            "abs_err_tail_8mm": float((s[0] - s[13]) / (n - s[4]).clamp(min=1)),
            "tail_frac_8mm": float((n - s[4]) / n),
        }
        if s[6] > 0:
            metrics["prior_abs_err"] = float(s[5] / s[6])
        if s[8] > 0:
            metrics["abs_err_prior_corrupted"] = float(s[7] / s[8])
        if s[10] > 0:
            metrics["abs_err_prior_clean"] = float(s[9] / s[10])

        # Loss scalars and per-batch means travel as objects (see the class
        # docstring). Merging in rank order keeps the summation deterministic,
        # so every rank ends up with byte-identical numbers.
        if self.is_ddp:
            payload: list = [None] * dist.get_world_size()
            dist.all_gather_object(
                payload, (self.log_sums, self.log_counts, self.batch_means)
            )
        else:
            payload = [(self.log_sums, self.log_counts, self.batch_means)]
        log_sums: dict[str, float] = {}
        log_counts: dict[str, int] = {}
        batch_means: list[float] = []
        for rank_sums, rank_counts, rank_means in payload:
            for k, v in rank_sums.items():
                log_sums[k] = log_sums.get(k, 0.0) + v
                log_counts[k] = log_counts.get(k, 0) + rank_counts[k]
            batch_means.extend(rank_means)

        if batch_means:
            bm = np.asarray(batch_means)
            metrics["abs_err_batch_median"] = float(np.median(bm))
            metrics["abs_err_batch_p90"] = float(np.percentile(bm, 90))
        avg_logs = {k: log_sums[k] / max(log_counts[k], 1) for k in sorted(log_sums)}
        self.reset()
        return metrics, avg_logs


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

    def __init__(self, run_name: str, enabled: bool, cfg=None) -> None:
        self.enabled = enabled
        self.best_metric = float("inf")
        self.cfg = cfg          # 存进 checkpoint, 见 save()
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
        # Loss terms land under train/loss_*, everything else the loss reports
        # (in_range / p_max / guard_win_rate / prior_abs_err / ...) under
        # train/diag_*, and the window-aggregated pixel metrics under
        # train/metric_*. Generic so new diagnostics show up without edits here.
        self.tb.add_scalar(self._tag("loss_total"), logs.get("loss", 0.0), step)
        for key, value in logs.items():
            if key == "loss":
                continue
            stage, _, field = key.partition("/")
            prefix = "loss" if field.startswith(("ce", "reg")) else "diag"
            self.tb.add_scalar(self._tag(f"{prefix}_{stage}_{field}"), value, step)
        self.tb.add_scalar(self._tag("learning_rate"), lr, step)
        for key, value in metrics.items():
            self.tb.add_scalar(self._tag(f"metric_{key}"), value, step)

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
            # 完整配置 + 指纹。以下开关都无法从 state_dict 反推:
            # 40/8 还是 32/16、branch prior/hard gate、dual-mode、reliability
            # source、stage weights。少了它, 同一个 ckpt 用当前默认配置推理会
            # 静默改变语义 (log/model/best.pth 就是 32/16 训的)。
            "config": _config_snapshot(self.cfg) if getattr(self, "cfg", None) is not None else None,
            "fingerprint": _arch_fingerprint(self.cfg) if getattr(self, "cfg", None) is not None else None,
            "git": _git_state(),
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
        # exercise the SPRE supervision path under --smoke as well
        "prior_corrupt_mask": (torch.rand(B, H, W) > 0.7),
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
        build_prior_cache(ds, device, overwrite=overwrite, image_target_wh=cfg.prior.target_wh)


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
    if args.val_interval is not None:
        train_overrides["val_interval"] = args.val_interval
    if args.amp is not None:
        train_overrides["amp"] = args.amp == "on"
    if train_overrides:
        cfg = replace(cfg, train=replace(cfg.train, **train_overrides))

    prior_overrides = {}
    if args.prior_target_w is not None:
        prior_overrides["target_w"] = args.prior_target_w
    if args.prior_target_h is not None:
        prior_overrides["target_h"] = args.prior_target_h
    if prior_overrides:
        cfg = replace(cfg, prior=replace(cfg.prior, **prior_overrides))

    if args.spre is not None:
        on = args.spre == "on"
        cfg = replace(cfg, spre=replace(cfg.spre, enabled=on))
        # --spre off 就是"没有 SPRE": 在这里显式翻译成 cached, 免得后面
        # 靠模型内部降级。--reliability 若也给了, 下面会覆盖它。
        if not on:
            cfg = replace(cfg, spre=replace(cfg.spre, reliability_source="cached"))
    # DINO matching 与 SPRE reliability 解耦: 从前 spre.enabled 一个开关同时
    # 控制"是否加载 DINO / 是否跑 SVA / 是否喂 FPN / 用不用 SPRE 替换缓存
    # confidence", 任何收益都无法归因。
    if args.dino_mode is not None:
        cfg = replace(cfg, dino=replace(cfg.dino, mode=args.dino_mode))
    if args.feed_fpn is not None:
        cfg = replace(cfg, dino=replace(cfg.dino, feed_fpn=(args.feed_fpn == "on")))
    if args.reliability is not None:
        cfg = replace(cfg, spre=replace(cfg.spre, reliability_source=args.reliability))

    is_ddp = world_size > 1
    is_main = rank == 0

    # Prior generation loads the large VGGT and DA3 models and can take much
    # longer than a distributed collective's timeout.  Letting rank 0 build
    # while the other ranks wait also hides the useful rank-0 exception:
    # waiting ranks usually surface only a secondary "connection reset by
    # peer" from NCCL.  Precompute in a single-process invocation, then launch
    # DDP with --build-priors skip (the UMHPC script does this automatically).
    if is_ddp and not args.smoke and args.build_priors != "skip":
        raise RuntimeError(
            "prior precomputation is not supported inside a DDP launch; run "
            "`python train.py --profile <profile> --gpus 1 --ddp off "
            "--num-views <views> --build-priors only` first, then relaunch "
            "DDP with `--build-priors skip`"
        )

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
    # VGGT/DA3 are freed first. DDP jobs are required to use the prebuilt cache.
    if not args.smoke and args.build_priors != "skip":
        if is_main:
            _ensure_priors(cfg, device, overwrite=(args.build_priors == "force"))
        if args.build_priors == "only":
            if is_main:
                print("[pre_prior] cache complete (--build-priors only); exiting before training")
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
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = args.name + ("_smoke" if args.smoke else "")
    logger = TrainLogger(run_name, enabled=is_main, cfg=cfg)

    # Resume from latest.pth so a walltime-killed job continues where it left
    # off (model + optimizer + step + best val metric). Every rank loads the
    # same file; the LR is recomputed from the step counter, so nothing else
    # needs restoring.
    start_step = 0
    if not args.smoke and args.resume == "auto":
        ckpt_path = ProjectPaths().project_path / "log" / "model" / "latest.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            try:
                (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])
            except RuntimeError as exc:
                # An architecture change (SPRE's cross-ViT fusion, DINOv3's
                # LayerScale, ...) makes old checkpoints unloadable. Failing
                # here is correct — silently partial-loading would train a new
                # head on top of weights fitted to the old one — but say so.
                raise RuntimeError(
                    f"{ckpt_path} was written by a different architecture and cannot be resumed.\n"
                    "Start fresh with --resume off (and move the stale checkpoint aside), "
                    "or check out the commit the checkpoint came from.\n"
                    f"original error: {exc}"
                ) from exc
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
    # [abs_err_sum, pixel_count, hits<2mm, hits<4mm, hits<8mm,
    #  body2_err_sum, body4_err_sum, body8_err_sum]  (see WindowedMeter for why)
    stats = torch.zeros(8, device=device, dtype=torch.float64)
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
        pred = outputs["depth_full"].float()
        gt = batch["depth_gt"].float()
        m = batch["mask"].bool() & (gt > 0)
        if "depth_values" in batch:
            dv = batch["depth_values"].float()
            m &= (gt >= dv.amin(dim=1).view(-1, 1, 1)) & (gt <= dv.amax(dim=1).view(-1, 1, 1))
        if m.any():
            err = (pred[m] - gt[m]).abs()
            stats[0] += err.sum()
            stats[1] += m.sum()
            for i, t in enumerate((2.0, 4.0, 8.0)):
                hit = err < t
                stats[2 + i] += hit.sum()
                stats[5 + i] += err[hit].sum()
    if is_ddp:
        dist.all_reduce(stats)
    model.train()
    if stats[1].item() == 0:
        raise RuntimeError(
            "validation produced no valid depth pixels — the val split is "
            "misconfigured (empty list, missing GT, or all-zero masks)"
        )
    n = stats[1]
    return {
        "abs_err": float(stats[0] / n),
        "acc_2mm": float(stats[2] / n),
        "acc_4mm": float(stats[3] / n),
        "acc_8mm": float(stats[4] / n),
        "abs_err_body_2mm": float(stats[5] / stats[2].clamp(min=1)),
        "abs_err_body_4mm": float(stats[6] / stats[3].clamp(min=1)),
        "abs_err_tail_8mm": float((stats[0] - stats[7]) / (n - stats[4]).clamp(min=1)),
        "tail_frac_8mm": float((n - stats[4]) / n),
    }


class _EpochShuffleSampler(torch.utils.data.Sampler):
    """Non-DDP counterpart of DistributedSampler: a per-epoch permutation that is
    fixed once ``set_epoch`` is called, so ``list(sampler)`` is exactly what the
    loader will draw (``shuffle=True`` re-draws and cannot be inspected)."""

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.seed = int(n), int(seed)
        self.order = list(range(self.n))

    def set_epoch(self, epoch: int) -> None:
        self.order = np.random.default_rng(self.seed + int(epoch)).permutation(self.n).tolist()

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


def _config_snapshot(cfg) -> dict:
    """MVSConfig -> 纯 python 容器 (Path 转 str), 可无依赖反序列化。"""
    from dataclasses import asdict, is_dataclass

    def conv(o):
        if is_dataclass(o) and not isinstance(o, type):
            return {k: conv(v) for k, v in asdict(o).items()}
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(conv(v) for v in o)
        if isinstance(o, Path):
            return str(o)
        return o
    return conv(cfg)


def _arch_fingerprint(cfg) -> dict:
    """决定 state_dict 是否可加载 / 推理语义是否一致的那些开关。"""
    return {
        "num_global": cfg.depth_range.num_global,
        "num_local": cfg.depth_range.num_local,
        "num_depths": [cfg.cost_volume.num_depths_stage1, cfg.cost_volume.num_depths_stage2,
                       cfg.cost_volume.num_depths_stage3, cfg.cost_volume.num_depths_stage4],
        "gate_local_branch": cfg.depth_range.gate_local_branch,
        "branch_prior": cfg.depth_range.branch_prior,
        "dual_mode_stage2": cfg.depth_range.dual_mode_stage2,
        "visibility_weighting": cfg.cost_volume.visibility_weighting,
        "use_src_weights": cfg.cost_volume.use_src_weights,
        "dino_mode": getattr(cfg.dino, "mode", "all_view"),
        "feed_fpn": cfg.dino.feed_fpn,
        "reliability_source": getattr(cfg.spre, "reliability_source", "spre"),
        "spre_enabled": cfg.spre.enabled,
        "mode_window": cfg.depth_range.mode_window,
    }


def _git_state() -> dict:
    import subprocess
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}




def _run_training(model, loss_fn, optimizer, scaler, cfg, device, args, world_size, rank, is_ddp, logger, is_main, start_step=0):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from data.augment import PhotometricAug
    from data.dtu import DTUMVSDataset

    _use_clean = not getattr(args, "no_clean_lists", False)
    _tr_list, _tr_excl = resolve_split(cfg.paths.train_list_file, "train", _use_clean)
    dataset = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=_tr_list,
        exclude_file=_tr_excl,
        nviews=cfg.train.num_views,
        mode="train",
        prior_corruption_prob=cfg.train.prior_corruption_prob,
        use_src_weights=cfg.cost_volume.use_src_weights,
        # Augmentation is train-only: val must stay deterministic or best.pth
        # gets selected on a moving target.
        aug=PhotometricAug(
            brightness=cfg.augment.brightness, contrast=cfg.augment.contrast,
            saturation=cfg.augment.saturation, hue=cfg.augment.hue,
            min_gamma=cfg.augment.min_gamma, max_gamma=cfg.augment.max_gamma,
        ) if cfg.augment.photometric else None,
        scales=cfg.augment.scales if cfg.augment.multi_scale else (),
        resize_range=cfg.augment.resize_range,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"training dataset is empty — check {_tr_list}")
    # A sampler on both paths (not shuffle=True) so the epoch's draw order can be
    # read before the loader consumes it — multi-scale needs to bucket that exact
    # order into per-batch resolutions.
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if is_ddp else _EpochShuffleSampler(len(dataset), cfg.train.seed)
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=_collate,
        worker_init_fn=_worker_init,
        pin_memory=True,
        drop_last=True,
    )

    # Validation: deterministic split (center crop, one light condition); used
    # to select best.pth. Never overlaps the training scans.
    _va_list, _va_excl = resolve_split(cfg.paths.val_list_file, "val", _use_clean)
    val_dataset = DTUMVSDataset(
        datapath=cfg.paths.dtu_train_root,
        listfile=_va_list,
        exclude_file=_va_excl,
        nviews=cfg.train.num_views,
        mode="val",
        use_src_weights=cfg.cost_volume.use_src_weights,
    )
    # A silently-empty val split poisons best.pth (val abs_err computes to 0.0
    # at the first validation and can never be beaten). Fail loudly instead.
    if len(val_dataset) == 0:
        raise RuntimeError(f"validation dataset is empty — check {_va_list}")
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=_collate,
        worker_init_fn=_worker_init,
        pin_memory=True,
        drop_last=False,
    )

    max_steps = args.steps if args.steps else cfg.train.max_steps
    model.train()
    use_amp = cfg.train.amp and device.type == "cuda"
    meter = WindowedMeter(device, is_ddp)
    step = start_step
    epoch = start_step // max(len(loader), 1)
    while step < max_steps:
        sampler.set_epoch(epoch)
        # Multi-scale: bucket this epoch's sampling order so every batch shares
        # one resolution (collate cannot stack mixed H x W). Must run AFTER
        # set_epoch — the order it buckets is the order the loader will draw —
        # and before iterating, so forked workers inherit the plan.
        dataset.reset_scale_plan(list(sampler), cfg.train.batch_size)
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            lr = _lr_at(cfg, step, max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            logs, outputs = _train_step(model, loss_fn, optimizer, scaler, batch, cfg, device, step, use_amp)

            # every rank feeds the window; flush is a collective at log steps.
            # GT beyond the scene's physical depth range is unreachable by any
            # in-range hypothesis — excluded from metrics (loss excludes it too).
            gt_b = batch["depth_gt"].float()
            metric_mask = batch["mask"].float()
            if "depth_values" in batch:
                dv_b = batch["depth_values"].float()
                in_scene = (gt_b >= dv_b.amin(dim=1).view(-1, 1, 1)) & (gt_b <= dv_b.amax(dim=1).view(-1, 1, 1))
                metric_mask = metric_mask * in_scene.float()
            meter.update(
                outputs["depth_full"].detach(),
                gt_b,
                metric_mask,
                prior=batch.get("depth_prior"),
                corrupt_mask=batch.get("prior_corrupt_mask"),
                logs=logs,
            )
            if step % cfg.train.log_interval == 0:
                win_metrics, win_logs = meter.flush()
                if is_main:
                    logger.log_scalars(win_logs, lr, win_metrics, step)
                    print(
                        f"[step {step}] loss={win_logs.get('loss', float('nan')):.4f} "
                        f"abs_err={win_metrics.get('abs_err', float('nan')):.2f} "
                        f"prior_err={win_metrics.get('prior_abs_err', float('nan')):.2f} "
                        f"rescue_err={win_metrics.get('abs_err_prior_corrupted', float('nan')):.2f}"
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


def _worker_init(worker_id: int) -> None:
    """Keep每个 DataLoader worker 单线程。

    torch 在 worker 里本来就是单线程, 但 OpenCV 不是: 它的线程池默认取满核数。
    dtu.py 每个样本要调十几次 cv2.resize, 于是 N 个 worker 各自开 N_core 个线程:
    单卡 16 worker 在 64 核上已经超订, DDP 两个 rank 就是 32 个 worker, 直接翻倍。
    争用导致某个 rank 取一个 batch 要几分钟 —— 单卡时只是变慢, DDP 下另一个 rank
    会卡在梯度 all-reduce 上被 NCCL watchdog (默认 10 分钟) 判定超时并杀掉作业。
    """
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)


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
    parser.add_argument("--val-interval", type=int, default=None, help="steps between validation runs")
    parser.add_argument("--amp", choices=["on", "off"], default=None, help="AMP override")
    parser.add_argument("--prior-target-w", type=int, default=None,
                        help="VGGT/DA3 prior width override (default 518; must be a multiple of 14). "
                             "Raises true depth-prior resolution at VGGT compute/memory cost. "
                             "Changing it needs --build-priors force to rebuild the cache.")
    parser.add_argument("--prior-target-h", type=int, default=None,
                        help="VGGT/DA3 prior height override (default 420; must be a multiple of 14)")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--resume", choices=["auto", "off"], default="auto",
                        help="auto: continue from log/model/latest.pth if present; off: always start fresh")
    parser.add_argument("--build-priors", choices=["auto", "force", "skip", "only"], default="auto",
                        help="auto: precompute missing priors; force: recompute all; "
                             "skip: assume cached; only: build missing priors then exit; "
                             "DDP launches must use skip")
    parser.add_argument("--no-clean-lists", action="store_true",
                        help="不使用 audit 产出的 *_clean.txt / exclude_*.csv (默认自动使用)")
    parser.add_argument("--dino-mode", choices=["off", "all_view", "ref_only"], default=None,
                        help="DINO 路径: off / all_view (当前) / ref_only (MonoMVSNet 式)")
    parser.add_argument("--feed-fpn", choices=["on", "off"], default=None,
                        help="DINO 特征是否注入 FPN (与 SPRE 可靠度解耦)")
    parser.add_argument("--reliability", choices=["cached", "edge", "spre"], default=None,
                        help="prior 可靠度来源, 与 DINO matching 解耦")
    parser.add_argument("--spre", choices=["on", "off"], default=None,
                        help="enable/disable the SPRE DINOv3 prior-reliability head (default: config value)")
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
