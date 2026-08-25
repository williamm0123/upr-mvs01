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
    log/experiments/<run>/model/       latest.pth + best.pth (每个 arm 独立)

Resume: ``--resume auto`` (default) continues from log/experiments/<run>/model/latest.pth when it
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
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from base.config import resolve_split, ProjectPaths, build_mvs_config
from losses import MVSLoss
# risk@0.6 的排序信号必须与 test.py 融合门控用的是同一份实现, 所以从这里导入
# 而不是在 train.py 里再写一份 —— 见 models/fusion_conf.py 顶部的说明。
from models.fusion_conf import cascade_confidence
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

    退火 horizon 默认跟随实际训练步数, 但可以用 cfg.train.lr_schedule_steps
    (--lr-schedule-steps) 解耦。12k 的筛选 arm 必须跑在 30k 的 horizon 上,
    否则它是"退火到底的 12k 模型", 跟 30k run 在同一步数窗口的 lr 差一个数量级,
    既不能横比也不能无跳变续训。
    """
    warm = cfg.train.warmup_steps
    if step < warm:
        return cfg.train.lr * (step + 1) / max(warm, 1)
    horizon = getattr(cfg.train, "lr_schedule_steps", None) or max_steps
    prog = (step - warm) / max(horizon - warm, 1)
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
    #               body2_err, body4_err, body8_err, capped2_err]
    # body{n}_err sums the error of pixels already within n mm; divided by the
    # matching hit{n} count it gives the *precision on matched pixels*, which the
    # plain mean cannot show (a few percent of far-off pixels dominate it). This
    # is MVSFormer++'s abs_depth_thres0-{n}mm_error — their DTU-test value is
    # 0.4920 mm at 2 mm — and it is the only way to tell a refinement gain from
    # an outlier-count gain.
    # slot 14 = sum(min(e, 2mm)) —— 主指标 capped_2mm 的分子。**直接累加**,
    # 不用 body_2mm x acc_2mm + 2(1-acc_2mm) 反算: 那个恒等式只在同一批像素上
    # 成立, 拿两个已经各自平均过的日志标量去乘会引入窗口口径不一致的偏差。
    # 为什么用它当主指标: abs_err 有 71% 来自 3.7% 的像素 (融合本该丢掉的那些),
    # 而 body_2mm 可以被 "把 1.9mm 的像素推到 2.1mm" 改善。min(e,2) 对每个像素
    # 的误差单调不减, 且尾部贡献被截断在 2mm, 两种操纵都不成立。
    # 注意它是**单调不减**而不是"必然变好": 20mm -> 3mm 不计入, 所以
    # tail_frac_8mm 必须始终并列汇报。
    N_SLOTS = 15

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
            self.sums[14] += e.clamp(max=2.0).sum()
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
            "capped_2mm": float(s[14] / n),
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


def run_paths(run_name: str) -> dict:
    """``log/experiments/<run_name>/{model,tensorboard}`` —— 每个消融 arm 一套。"""
    root = ProjectPaths().project_path / "log" / "experiments" / run_name
    return {"root": root, "model": root / "model", "tensorboard": root / "tensorboard"}


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

    def __init__(self, run_name: str, enabled: bool, cfg=None, scaler=None) -> None:
        self.enabled = enabled
        self.best_metric = float("inf")
        self.cfg = cfg          # 存进 checkpoint, 见 save()
        self.scaler = scaler    # GradScaler 的 scale 也是训练状态, 必须一起存
        if not enabled:
            return
        # 每个 run 一套目录。以前所有实验共用 log/model/latest.pth, 并行提交
        # 多个 arm 会互相覆盖 checkpoint, --resume auto 还可能捡到别的 arm 的权重。
        self.model_dir = run_paths(run_name)["model"]
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # tensorboard 再按启动时间分子目录: 同一个 run 被 slurm 重排队后新开一份
        # 事件文件, 但 model 目录不变, 所以 resume 仍能接上。
        tb_dir = run_paths(run_name)["tensorboard"] / datetime.now().strftime('%Y%m%d_%H%M%S')
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
        state = (model.module if isinstance(model, DDP) else model).state_dict()
        # 非有限权重绝不落盘。latest.pth 是 resume 的唯一入口, 一旦被坏状态覆盖,
        # 这个 run 就再也回不到好点了 (arm L 的 latest 就是这么没的)。注意这里
        # 扫的是整个 state_dict, 包含 BN 的 running_mean/var —— 它们在 forward
        # 里无条件更新, 不受 GradScaler 跳步保护, 恰恰是最先被污染的那一批。
        bad = [k for k, v in state.items()
               if isinstance(v, torch.Tensor) and v.is_floating_point()
               and not bool(torch.isfinite(v).all())]
        if bad:
            print(f"[ckpt] 拒绝写入: state_dict 里有 {len(bad)} 个张量含非有限值 "
                  f"(例如 {bad[:3]})。latest.pth / best.pth 保持在上一个好状态。")
            return
        is_best = val_metric is not None and val_metric < self.best_metric
        if is_best:
            self.best_metric = float(val_metric)
        ckpt = {
            "step": step,
            "model": state,
            "optimizer": optimizer.state_dict(),
            "best_metric": self.best_metric,
            # 续训完整性: 少了这两样, "接着跑"其实是换了一条随机流从头凑 ——
            # scaler 的 scale 要重新爬回去 (前几十步全被跳过), RNG 归零则 dropout
            # / 初始化侧的序列跟不中断的那次对不上。样本位置不用存: 它是
            # step % len(loader) 的确定性函数, resume 时重新算 (见 _run_training)。
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "rng": _rng_state(),
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
def _synthetic_batch(cfg, device: torch.device, batch_size: int, hw=None) -> dict:
    B, V = batch_size, cfg.train.num_views
    H, W = hw if hw is not None else (256, 320)
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
    if cfg.cost_volume.vis_supervise:
        # W3-B 的遮挡监督也要走一遍 —— 否则 --smoke 覆盖不到 L_vis, 而那条路上
        # 有 grid_sample 的坐标约定, 恰恰是最容易静默写错的地方。
        batch["src_depth_gt"] = torch.rand(B, V - 1, H, W) * 400 + dmin
        batch["src_mask_gt"] = (torch.rand(B, V - 1, H, W) > 0.2).float()
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
    if args.log_interval is not None:
        train_overrides["log_interval"] = args.log_interval
    if args.amp is not None:
        train_overrides["amp"] = args.amp == "on"
    if args.amp_dtype is not None:
        train_overrides["amp_dtype"] = args.amp_dtype
    if args.lr_schedule_steps is not None:
        train_overrides["lr_schedule_steps"] = args.lr_schedule_steps
    if args.nan_watchdog is not None:
        train_overrides["nan_watchdog"] = args.nan_watchdog == "on"
    if train_overrides:
        cfg = replace(cfg, train=replace(cfg.train, **train_overrides))

    if args.seed is not None:
        train_overrides["seed"] = args.seed
        cfg = replace(cfg, train=replace(cfg.train, seed=args.seed))

    # --- 消融开关 -> 配置 ---
    if args.prior is not None:
        cfg = replace(cfg, prior=replace(cfg.prior, mode=args.prior))
    if args.prior_cache is not None:
        cfg = replace(cfg, paths=replace(cfg.paths, prior_cache_path=Path(args.prior_cache)))

    dr_over = {}
    if args.range_min_gi is not None:
        v = tuple(float(x) for x in args.range_min_gi.split(","))
        if len(v) != 3:
            raise SystemExit("--range-min-gi 需要三个值 (s1->s2, s2->s3, s3->s4)")
        dr_over["range_min_gi"] = v
    if args.depth_adaptive is not None:
        dr_over["depth_adaptive_range"] = args.depth_adaptive == "on"
    if args.depth_adaptive_max is not None:
        dr_over["depth_adaptive_max"] = args.depth_adaptive_max
    if args.depth_adaptive_apply is not None:
        dr_over["depth_adaptive_apply"] = args.depth_adaptive_apply
    if args.depth_adaptive_stages is not None:
        v = tuple(x.strip().lower() == "on" for x in args.depth_adaptive_stages.split(","))
        if len(v) != 3:
            raise SystemExit("--depth-adaptive-stages 需要三个 on/off")
        dr_over["depth_adaptive_stages"] = v
    if args.branch_prior_mode is not None:
        dr_over["branch_prior_mode"] = args.branch_prior_mode
    if args.branch_prior_anneal_steps is not None:
        dr_over["branch_prior_anneal_steps"] = args.branch_prior_anneal_steps
    if args.num_global is not None:
        dr_over["num_global"] = args.num_global
    if args.num_local is not None:
        dr_over["num_local"] = args.num_local
    if args.gate_local is not None:
        dr_over["gate_local_branch"] = args.gate_local == "on"
    if args.branch_prior is not None:
        dr_over["branch_prior"] = args.branch_prior == "on"
    if args.axis_space is not None:
        dr_over["axis_space"] = args.axis_space
    if args.axis_blend_steps is not None:
        dr_over["axis_blend_steps"] = args.axis_blend_steps
    if args.rho_max is not None:
        dr_over["rho_max"] = args.rho_max
    if args.rho_stages is not None:
        v = tuple(float(x) for x in args.rho_stages.split(","))
        if len(v) != 3 or not all(x > 1.0 for x in v):
            raise SystemExit("--rho-stages 需要三个 > 1.0 的值, 如 '8,4,2'")
        dr_over["rho_stages"] = v
    if args.child_interval_cap is not None:
        dr_over["child_interval_cap"] = args.child_interval_cap == "on"
    if args.refine_ratio_init is not None:
        v = tuple(float(x) for x in args.refine_ratio_init.split(","))
        if len(v) != 3 or not all(0.0 < x < 1.0 for x in v):
            raise SystemExit("--refine-ratio-init 需要三个 (0,1) 内的值")
        dr_over["refine_ratio_init"] = v
    if args.refine_cap_p is not None:
        if args.refine_cap_p < 2.0:
            raise SystemExit("--refine-cap-p 需要 >= 2.0")
        dr_over["refine_cap_p"] = args.refine_cap_p
    if args.tau_stages is not None:
        v = tuple(float(x) for x in args.tau_stages.split(","))
        if len(v) != 3 or not all(0.5 < x < 1.0 for x in v):
            raise SystemExit("--tau-stages 需要三个 (0.5, 1.0) 内的值")
        dr_over["tau_stages"] = v
    if args.spre_cascade is not None:
        dr_over["spre_cascade"] = args.spre_cascade == "on"
    if args.spre_gate_init is not None:
        v = tuple(float(x) for x in args.spre_gate_init.split(","))
        if len(v) != 4 or abs(v[0] - 1.0) > 1e-6:
            raise SystemExit("--spre-gate-init 需要四个值且第一个为 1.0")
        dr_over["spre_gate_init"] = v
    if args.stage4_head is not None:
        dr_over["stage4_head"] = args.stage4_head
    if args.mode_window_stages is not None:
        v = tuple(int(x) for x in args.mode_window_stages.split(","))
        if len(v) != 4:
            raise SystemExit("--mode-window-stages 需要四个整数")
        dr_over["mode_window_stages"] = v
    if dr_over:
        cfg = replace(cfg, depth_range=replace(cfg.depth_range, **dr_over))
    # num_depths_stage1 必须恒等于 num_global + num_local (network.py 会断言),
    # 所以它跟着分支预算走, 不单独暴露成开关。
    nd1 = cfg.depth_range.num_global + cfg.depth_range.num_local
    cv_over = {}
    if nd1 != cfg.cost_volume.num_depths_stage1:
        cv_over["num_depths_stage1"] = nd1
    if args.visibility is not None:
        cv_over["visibility_weighting"] = args.visibility == "on"
    if args.warp_channels_stage4 is not None:
        cv_over["warp_channels_stage4"] = args.warp_channels_stage4
    if args.num_groups is not None:
        cv_over["num_groups"] = args.num_groups
    if args.geo_valid is not None:
        cv_over["geo_valid_aggregation"] = args.geo_valid == "on"
    if args.vis_mode is not None:
        cv_over["vis_mode"] = args.vis_mode
    if args.vis_supervise is not None:
        cv_over["vis_supervise"] = args.vis_supervise == "on"
    if cv_over:
        cfg = replace(cfg, cost_volume=replace(cfg.cost_volume, **cv_over))
    dec_over = {}
    if args.conf_head is not None:
        dec_over["fusion_conf"] = args.conf_head == "on"
    if args.conf_detach is not None:
        dec_over["fusion_conf_detach"] = args.conf_detach == "on"
    if dec_over:
        cfg = replace(cfg, decoder=replace(cfg.decoder, **dec_over))
    if args.cvpe is not None:
        cfg = replace(cfg, cvpe=replace(cfg.cvpe, enabled=args.cvpe == "on"))
    if args.stage1_weight is not None:
        cfg = replace(cfg, stage_weights=replace(cfg.stage_weights, stage1=args.stage1_weight))
    loss_over = {}
    if args.w_branch is not None:
        loss_over["w_branch"] = args.w_branch
    if args.spre_balance_corrupt is not None:
        loss_over["spre_balance_corrupt"] = args.spre_balance_corrupt == "on"
    for _a, _k in (("w_range", "w_range"), ("w_center", "w_center"),
                   ("w_residual", "w_residual"), ("w_oor", "w_oor"),
                   ("w_conf", "w_conf"), ("conf_tau_mm", "conf_tau_mm"),
                   ("w_vis", "w_vis"), ("delta_occ_mm", "delta_occ_mm"),
                   ("pinball_scale", "pinball_scale")):
        _v = getattr(args, _a, None)
        if _v is not None:
            loss_over[_k] = _v
    if loss_over:
        cfg = replace(cfg, loss=replace(cfg.loss, **loss_over))

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

    _seed_everything(cfg.train.seed, rank, deterministic=args.deterministic)

    # 代码版本闸门: slurm 只保存 sbatch 脚本文本, 不保存 python 项目快照。排队
    # 期间主工作树被改过, 任务启动时读到的就是新代码 —— 提交时记下 SHA, 这里
    # 不符就退出, 免得跑出一份不知道对应哪版代码的结果。
    if args.expect_sha:
        _g = _git_state()
        cur = _g.get("commit", "")
        if not cur.startswith(args.expect_sha) and not args.expect_sha.startswith(cur[:len(args.expect_sha)]):
            raise SystemExit(
                f"git SHA mismatch: expected {args.expect_sha}, running {cur[:12]}. "
                f"用 git worktree 给实验单独开一份代码, 或重新提交任务。"
            )
        if _g.get("dirty"):
            raise SystemExit("工作树 dirty —— 实验必须跑在干净 commit 上")
        if is_main:
            print(f"[git] verified {cur[:12]} clean")

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
    # bf16 不需要 loss scaling: 它的指数范围跟 fp32 一样, 梯度不会下溢到 0,
    # 也就没有"放大 -> 检查 inf -> 跳过"这一整套。开着反而多一层跳步逻辑。
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and _amp_dtype(cfg) is torch.float16)

    if args.fit_batch:
        _fit_batch(model, loss_fn, optimizer, scaler, cfg, device, args, is_main)
        return

    run_name = args.name + ("_smoke" if args.smoke else "")
    logger = TrainLogger(run_name, enabled=is_main, cfg=cfg, scaler=scaler)

    # Resume from latest.pth so a walltime-killed job continues where it left
    # off (model + optimizer + step + best val metric). Every rank loads the
    # same file; the LR is recomputed from the step counter, so nothing else
    # needs restoring.
    start_step = 0
    if not args.smoke and args.resume == "auto":
        ckpt_path = run_paths(run_name)["model"] / "latest.pth"
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path, map_location=device)
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
            # 指纹闸门。存了却不校验等于没存: 8.16 那次回退里先验缓存换了一版,
            # 权重看起来完全可加载, 但跑的已经是另一个实验。lr / stage weights /
            # num_global / prior_cache_version 全在指纹里, 对不上就停。
            diff = _fingerprint_diff(ckpt.get("fingerprint"), _arch_fingerprint(cfg))
            if diff and not args.allow_fingerprint_mismatch:
                detail = "\n".join(f"    {k:<24s} checkpoint={s!r}  当前={c!r}"
                                   for k, (s, c) in diff.items())
                raise RuntimeError(
                    f"{ckpt_path} 的指纹与当前配置不符, 续训会静默变成另一个实验:\n"
                    f"{detail}\n"
                    "  想接着跑就先对齐配置; 确认无害 (例如只是重建过先验缓存) "
                    "再加 --allow-fingerprint-mismatch。"
                )
            if diff:
                print(f"[resume] 指纹有 {len(diff)} 处不符, 但 --allow-fingerprint-mismatch "
                      f"已放行: {sorted(diff)}")
            optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            rng_ok = _restore_rng(ckpt.get("rng"))
            start_step = int(ckpt.get("step", -1)) + 1
            logger.best_metric = float(ckpt.get("best_metric", float("inf")))
            if is_main:
                print(f"[resume] loaded {ckpt_path} -> continuing from step {start_step} "
                      f"(best val abs_err so far: {logger.best_metric:.4f}; "
                      f"scaler={'restored' if ckpt.get('scaler') is not None else 'reset'}, "
                      f"rng={'restored' if rng_ok else 'reset'})")

    if is_main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[rank {rank}] device={device} ddp={is_ddp} world_size={world_size} "
              f"params={n_params:.1f}M profile={cfg.train.profile} "
              f"batch={cfg.train.batch_size} views={cfg.train.num_views} "
              f"workers={cfg.train.num_workers} lr={cfg.train.lr:g} "
              f"warmup={cfg.train.warmup_steps} amp={cfg.train.amp}")

    # finally: 看门狗是靠抛异常终止的, 没有这层 tensorboard 的最后一批事件
    # (含事故前几步的 grad norm) 会留在缓冲区里丢掉 —— 正是最该看的那几个点。
    try:
        if args.smoke:
            _run_smoke(model, loss_fn, optimizer, scaler, cfg, device, args, logger, is_main)
        else:
            _run_training(model, loss_fn, optimizer, scaler, cfg, device, args, world_size, rank, is_ddp, logger, is_main, start_step)
    finally:
        logger.close()
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


class NonFiniteError(RuntimeError):
    """训练里出现第一个非有限张量 —— 立即终止, 不再往下跑。

    不终止的后果是实测过的 (arm L, 2026-08-18): GradScaler 发现 inf/nan 就把
    整步跳过, 权重原地不动, 下一步同样的前向再溢出、再跳过。日志上看是 loss 恒
    为 nan 而 step 照涨, 实际优化器一步没走 —— 12000 步里烧掉了 9050 步。同时
    BN 的 running stats 在 forward 里无条件更新, 不受跳步保护, 会被逐层污染,
    于是 val 从 acc_2mm=0.38 一路烂到 0。这也是为什么 save() 必须同时拒绝把
    这种状态写进 latest.pth: 否则连回退点都没有了。
    """


def _finite_stat(t: torch.Tensor) -> tuple[float, float]:
    """(有限值占比, 有限值中的 max|x|)。纯诊断路径, 不进计算图。"""
    x = t.detach().float()
    if x.numel() == 0:
        return 1.0, 0.0
    fin = torch.isfinite(x)
    n_fin = int(fin.sum())
    return n_fin / x.numel(), (float(x[fin].abs().max()) if n_fin else float("nan"))


def _sample_ids(batch: dict) -> str:
    """出事的是哪几个样本 —— 离线复现只能靠它定位。

    逐样本列出, 因为 stage4 的失效是 sample-conditioned 的: 同一个 batch 里
    往往只有一个样本坏掉 (finite_frac 恰好 0.5)。配上 data/dtu.py 的 _rng(idx),
    这几个字段足以离线重放出完全相同的那一个样本。
    """
    n = 0
    for k in ("sample_index", "scan"):
        v = batch.get(k)
        if v is not None:
            n = max(n, len(v) if isinstance(v, (list, tuple)) else int(v.shape[0]))
    if not n:
        return "<batch 内没有样本身份字段 —— data/dtu.py 是否是旧版?>"

    def get(k, i):
        v = batch.get(k)
        if v is None:
            return None
        try:
            x = v[i]
        except Exception:
            return None
        if hasattr(x, "tolist"):
            x = x.tolist()
        return x

    rows = []
    for i in range(n):
        parts = [f"[{i}]"]
        for k, fmt in (("scan", "{}"), ("ref_view", "view={}"), ("light_idx", "light={}"),
                       ("sample_index", "idx={}"), ("crop_hw", "crop={}"),
                       ("crop_xy", "at={}"), ("resize_scale", "scale={}")):
            x = get(k, i)
            if x is not None:
                parts.append(fmt.format(x))
        rows.append("  ".join(parts))
    return "\n                ".join(rows)


def _nonfinite_report(where, step, lr, scaler, batch, probes, extra_lines=()) -> str:
    try:
        scale = float(scaler.get_scale()) if scaler.is_enabled() else float("nan")
    except Exception:
        scale = float("nan")
    lines = [
        "", "=" * 78,
        f"[watchdog] 首个非有限值: {where}",
        "=" * 78,
        f"  step        = {step}",
        f"  lr          = {lr:.6e}",
        f"  amp scale   = {scale:g}",
        f"  batch       = {_sample_ids(batch)}",
        *extra_lines,
        "  张量                          finite_frac          max|x|",
    ]
    for name, t in probes.items():
        if isinstance(t, torch.Tensor):
            frac, mx = _finite_stat(t)
            lines.append(f"    {name:<28s} {frac:11.6f} {mx:15.6g}{'   <<<' if frac < 1.0 else ''}")
    lines += [
        "=" * 78,
        "  已终止。latest.pth 仍是最后一个全有限的状态, 可以直接 resume。",
        "  梯度范数的走势看 tensorboard 的 train/diag_grad_norm_unclipped。",
        "=" * 78, "",
    ]
    return "\n".join(lines)


def _train_step(model, loss_fn, optimizer, scaler, batch, cfg, device, step, use_amp,
                lr: float = float("nan")):
    watch = getattr(cfg.train, "nan_watchdog", True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=use_amp):
        outputs = model(batch, step=step)
        loss, logs = loss_fn(outputs, batch, step=step)

    # 关口 1 —— 前向。放在 backward 之前是有意的: 此刻还能看出是哪个 loss 分量
    # 先坏的; 一旦 backward 跑完, 梯度已经被污染成一片 nan, 源头就找不回来了。
    if watch:
        bad = sorted(k for k, v in logs.items()
                     if isinstance(v, (int, float)) and not math.isfinite(float(v)))
        if bad or not bool(torch.isfinite(loss)):
            probes = {"loss": loss}
            probes.update({f"out.{k}": v for k, v in outputs.items() if isinstance(v, torch.Tensor)})
            probes.update({f"in.{k}": v for k, v in batch.items()
                           if isinstance(v, torch.Tensor) and v.is_floating_point()})
            # 同一个 batch 重放一次带探针的前向, 把链上第一处坏张量一并写进报告。
            # 常关探针 + 出事重放, 而不是常开: 常开要在每步对二十个张量做逐样本
            # 归约, 全分辨率下不划算。模型只有 GroupNorm、没有 dropout, 加上
            # 逐样本播种, 重放是确定性的。
            chain = ["", "  前向链 (重放, 逐样本 finite_frac / max|x|):"]
            try:
                from models.probe import Probe
                with Probe.session():
                    with torch.no_grad(), torch.autocast(device_type=device.type,
                                                         dtype=_amp_dtype(cfg), enabled=use_amp):
                        model(batch, step=step)
                fb = Probe.first_bad()
                if fb is not None:
                    chain.append(f"  >> 第一处非有限: {fb['stage']}.{fb['name']}"
                                 + (f"[s{fb['src']}]" if fb["src"] is not None else "")
                                 + f"  dtype={fb['dtype']}")
                else:
                    chain.append("  >> 重放时全部有限 (非确定性来源? 检查 cudnn.benchmark)")
                chain.append(Probe.format())
            except Exception as exc:                       # 诊断失败不能盖住原始事故
                chain.append(f"  (探针重放失败: {exc})")
            raise NonFiniteError(_nonfinite_report(
                "forward — loss 或 loss 分量", step, lr, scaler, batch, probes,
                extra_lines=[f"  非有限的 loss 分量: {bad}" if bad
                             else "  loss 本身非有限 (各分量倒都是有限的)"] + chain,
            ))

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    named = [(n, p) for n, p in (model.module if isinstance(model, DDP) else model).named_parameters()
             if p.requires_grad and p.grad is not None]
    params = [p for _, p in named]
    grads = [p.grad for p in params]

    # 关口 2 —— 反向。逐参数 isfinite 是几百次 kernel launch, 每步都做太贵, 所以
    # 先用一次融合的 _foreach_norm 拿到 unclipped total norm; 只有它非有限时才
    # 回头逐参数定位。这一步必须在 clip_grad_norm_ 之前: total_norm 一旦是 nan,
    # clip 会把每个梯度都乘成 nan, 那时再扫就分不出谁是源头。
    if grads:
        try:
            per_norm = torch._foreach_norm(grads)
        except Exception:                      # 私有 API 的兜底
            per_norm = [g.detach().float().norm() for g in grads]
        total_norm = torch.linalg.vector_norm(torch.stack(per_norm))
    else:
        per_norm, total_norm = [], torch.zeros((), device=device)

    # 单步梯度非有限**不是**发散。AMP 的 GradScaler 正是靠"把 loss 放大到溢出 ->
    # 发现 inf -> 跳过这步并把 scale 减半"来自动找可用的缩放系数的, 而
    # scaler.unscale_() 只是除以 scale, inf/scale 仍然是 inf。在这里见到一次
    # inf 就终止, 等于把 AMP 的正常工作机制当成事故 —— L 那个 arm 恰恰是最容易
    # 触发溢出的, 会在第一次正常溢出时白死。
    # 真发散的特征是**连续**若干步都非有限 (scaler 已经把 scale 一路减半却仍然
    # 救不回来)。所以这里记连击数, 到 nan_grad_patience 才报。
    nonfinite_grad = bool(per_norm) and not bool(torch.isfinite(total_norm))
    streak = (getattr(_train_step, "_nf_streak", 0) + 1) if nonfinite_grad else 0
    _train_step._nf_streak = streak
    patience = max(int(getattr(cfg.train, "nan_grad_patience", 3)), 1)
    if watch and nonfinite_grad and streak >= patience:
        ok = torch.isfinite(torch.stack(per_norm)).tolist()
        bad_names = [named[i][0] for i, good in enumerate(ok) if not good]
        probes = {f"grad.{named[i][0]}": grads[i] for i, good in enumerate(ok) if not good}
        raise NonFiniteError(_nonfinite_report(
            "backward — 梯度", step, lr, scaler, batch, dict(list(probes.items())[:12]),
            extra_lines=[
                f"  连续 {streak} 步梯度非有限 (patience={patience}) —— 不是 AMP 的一次性溢出",
                f"  unclipped grad norm = {float(total_norm):g}",
                f"  非有限梯度的参数: {len(bad_names)}/{len(named)} 个",
                f"  最靠前的几个: {bad_names[:6]}",
            ],
        ))

    # error_if_nonfinite 必须是 False: 溢出那一步交给 scaler.step() 去跳过,
    # 这里抛异常就又把正常的 AMP 行为当成事故了。
    #
    # **分组裁剪**: conf_head 是一条自带监督的旁路, conf_detach=True 已经切断了它
    # 对深度特征的梯度 —— 但全模型一次 clip_grad_norm_ 会把它的范数算进总范数,
    # 于是 "旁路头梯度大" 就会等比例缩小主干的更新步长。那样 W3-C 就不再是
    # 一个可加可减的旁路, 而是一个悄悄改变主干学习率的东西。两组分别裁剪之后,
    # 主干的更新与 conf_head 开不开严格无关。
    # 不拆成两个 optimizer: 那会改变 AdamW 的 state 布局与 checkpoint 结构,
    # 而这里要解决的只是范数耦合。
    conf_params, base_params = [], []
    for _n, _p in named:
        (conf_params if _n.startswith("conf_head.") or ".conf_head." in _n
         else base_params).append(_p)
    _nb = torch.nn.utils.clip_grad_norm_(base_params, cfg.train.grad_clip,
                                         error_if_nonfinite=False) if base_params else None
    _nc = torch.nn.utils.clip_grad_norm_(conf_params, cfg.train.grad_clip,
                                         error_if_nonfinite=False) if conf_params else None
    if not nonfinite_grad:
        if _nb is not None and torch.isfinite(_nb):
            logs["grad/norm_base_unclipped"] = float(_nb)
        if _nc is not None and torch.isfinite(_nc):
            logs["grad/norm_conf_unclipped"] = float(_nc)
    # 这条曲线是提前量: L 那次事故之前梯度范数其实已经在涨, 只是没人记录。
    # 只在有限时记: WindowedMeter 按 key 各自累计均值 (log_sums/log_counts),
    # 缺席的步不进分母。写 nan 进去会把整个窗口的均值污染成 nan, 而这条曲线正是
    # 事故复盘要看的那条。溢出的信息由下面的 nonfinite_frac 承载。
    if not nonfinite_grad:
        logs["grad/norm_unclipped"] = float(total_norm)
    # 事故的提前量都在这两条曲线上:
    #   amp_scale 一路往下掉 = 溢出在变频繁 (每次溢出 scaler 把它减半)
    #   nonfinite_frac 是窗口内溢出步的占比, 0 -> 非 0 就是预警
    logs["grad/nonfinite_frac"] = 1.0 if nonfinite_grad else 0.0
    try:
        logs["grad/amp_scale"] = float(scaler.get_scale()) if scaler.is_enabled() else 0.0
    except Exception:
        pass
    scaler.step(optimizer)
    scaler.update()
    return logs, outputs


def _fit_batch(model, loss_fn, optimizer, scaler, cfg, device, args, is_main) -> None:
    """在**真实训练配置**下扫 per-GPU batch, 报每张卡的峰值显存与占用率。

    为什么长在 train.py 里而不是单独一个脚本: 它必须走完全同一条 cfg 覆盖路径
    (--axis-space / --geo-valid / --conf-head / dino / spre ...), 否则量出来的
    是另一个模型的显存。scripts/verify_w1.py --mem 那条路把 dino 和 spre 关掉了,
    数字系统性偏低, **不能**拿来定 batch。

    量的是**多尺度里最大的那个尺度**。训练是多尺度的, 显存在最小与最大尺度之间
    摆动 2.2 倍 —— 按平均尺度定 batch, 最大尺度那几步必 OOM; 所以只能按最大的定,
    代价是小尺度时卡是空着的。这不是可以调掉的, 是多尺度训练的固有性质。
    """
    if device.type != "cuda":
        print("[fit-batch] 需要 CUDA"); return
    scales = cfg.augment.scales if (cfg.augment.multi_scale and cfg.augment.scales) \
        else ((cfg.data.target_h, cfg.data.target_w),)
    hw = max(scales, key=lambda s: s[0] * s[1])
    if args.fit_hw:
        # 显式指定尺度: 在放不下最大尺度的小卡上, 用小尺度测出斜率再外推
        try:
            _h, _w = (int(v) for v in str(args.fit_hw).lower().split("x"))
        except ValueError:
            raise SystemExit(f"--fit-hw 需要 HxW, 收到 {args.fit_hw!r}")
        hw = (_h, _w)
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
    try:
        batches = [int(x) for x in str(args.fit_batch).split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--fit-batch 需要逗号分隔的整数, 收到 {args.fit_batch!r}")
    target = float(args.fit_target)

    print(f"[fit-batch] {torch.cuda.get_device_name(device)}  总显存 {total:.1f} GiB")
    print(f"[fit-batch] 最大尺度 {hw[0]}x{hw[1]} ({hw[0]*hw[1]/1e6:.4f} Mpx), views={cfg.train.num_views}")
    print(f"[fit-batch] 目标占用 {100*target:.0f}%  ({total*target:.1f} GiB)")
    print(f"[fit-batch] 注意: 报的是 reserved (nvidia-smi 看到的量), 另加 ~0.5GiB CUDA context\n")
    print(f"  {'batch':>5} {'allocated':>11} {'reserved':>10} {'占用':>8}  {'状态':<8}")

    rows = []
    for b in batches:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            # 跑两步: 第一步之后 AdamW 才会分配 exp_avg/exp_avg_sq (约 830MB),
            # 只跑一步会把优化器状态漏掉, 低估将近 1GiB。
            for _ in range(2):
                batch = _synthetic_batch(cfg, device, b, hw=hw)
                _train_step(model, loss_fn, optimizer, scaler, batch, cfg, device,
                            10 ** 6, cfg.train.amp)
                del batch
            alloc = torch.cuda.max_memory_allocated() / 2 ** 30
            resv = torch.cuda.max_memory_reserved() / 2 ** 30
            frac = (resv + 0.5) / total
            rows.append((b, alloc, resv, frac))
            flag = "OK" if frac <= target else "超目标"
            print(f"  {b:>5} {alloc:>10.2f}G {resv:>9.2f}G {100*frac:>7.1f}%  {flag:<8}")
        except torch.OutOfMemoryError:
            print(f"  {b:>5} {'-':>11} {'-':>10} {'-':>8}  OOM")
            torch.cuda.empty_cache()
            break
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    fit = [r for r in rows if r[3] <= target]
    print()
    if not fit:
        print("[fit-batch] 没有一个 batch 落在目标之下 —— 目标太低或尺度太大")
        return
    best = max(fit, key=lambda r: r[0])
    print(f"[fit-batch] 目标 {100*target:.0f}% 之下最大的 per-GPU batch = {best[0]} "
          f"(峰值 {best[2]:.1f} GiB, {100*best[3]:.1f}%)")
    # lr 的基准必须是**未缩放**的参考值。不能拿 cfg.train.lr —— 那个已经被
    # _arm_common.sh 按当前全局 batch 缩放过一次了, 再缩一次就是双重缩放。
    import math as _m
    ref_lr, ref_b = float(args.fit_lr_ref), float(args.fit_lr_ref_batch)
    print(f"    (lr 基准 {ref_lr:.3e} @ 全局 batch {ref_b:g}; 当前 cfg.train.lr="
          f"{cfg.train.lr:.3e} 已按当前设置缩放过, 不作基准)")
    for nproc in (1, 2):
        gb = best[0] * nproc
        print(f"    {nproc} 卡 -> 全局 batch {gb}; "
              f"sqrt 缩放 lr = {ref_lr * _m.sqrt(gb / ref_b):.3e}, "
              f"linear = {ref_lr * gb / ref_b:.3e}")
    print("[fit-batch] 提醒: 多尺度下这个占用只在**最大尺度**成立; 最小尺度 "
          f"({min(scales, key=lambda s: s[0]*s[1])}) 大约只有它的 "
          f"{100*min(s[0]*s[1] for s in scales)/max(s[0]*s[1] for s in scales):.0f}%。")


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
def _fmt_val(m: dict) -> str:
    """主指标一行, 深度分桶的 abs_err 再跟一行 —— 全部打出来是 28 个键, slurm
    日志会被淹掉, 而分桶数据在 tensorboard 里都有。"""
    main = " ".join(f"{k}={m[k]:.4f}" for k in
                    ("abs_err", "acc_2mm", "acc_8mm", "tail_frac_8mm") if k in m)
    q = " ".join(f"q{b}={m[f'q{b}_abs_err']:.3f}" for b in range(4) if f"q{b}_abs_err" in m)
    return main + (f" | depth-buckets {q}" if q else "")


def _run_validation(model, loader, device, use_amp, is_ddp, amp_dtype=None,
                    step: int | None = None, blend_steps: int = 2000) -> dict[str, float]:
    """Masked depth metrics over the val split.

    ``step`` **必须**传: 前向里的 ``blend = min(1, step/axis_blend_steps)`` 决定
    候选轴是 legacy 还是 inverse。不传的话验证恒在 blend=1 上跑, 而训练还在
    blend=0.3 —— 两条曲线量的是两个模型, 而且这个错误完全静默。
    独立的 test.py 不传 (step=None -> blend=1.0) 是对的: 部署用最终的完整逆深度轴。

    All ranks run their DistributedSampler shard and the pixel-weighted sums are
    all-reduced, so the result is identical on every rank (and SyncBN-safe,
    should it ever be enabled)."""
    model.eval()
    # [abs_err_sum, pixel_count, hits<2mm, hits<4mm, hits<8mm,
    #  body2_err_sum, body4_err_sum, body8_err_sum,
    #  capped2_err_sum, risk_err_sum, risk_count]  (see WindowedMeter for why)
    stats = torch.zeros(11, device=device, dtype=torch.float64)
    # 按归一化场景深度 (gt - dmin)/(dmax - dmin) 分 4 桶。
    # 逆深度采样、d^2 的三角测量分辨率、先验误差随深度的分布 —— 这些争论都只能
    # 靠"误差是否集中在远端"来定。整体 abs_err 把它平均掉了。
    #   bq: [err_sum, n, hit<2mm, err>=8mm]           每桶
    #   sq: [in_range_sum, n]                          每桶 x stage(2,3,4)
    bq = torch.zeros(4, 4, device=device, dtype=torch.float64)
    sq = torch.zeros(4, 3, 2, device=device, dtype=torch.float64)
    # 按桶 x {stage4 窗口内, 窗口外} 拆开的误差: [err_sum, n]。
    # 这是把"覆盖率"和"精度"分开的唯一办法 —— 抬宽窗口一定同时改变两者,
    # 只看总 abs_err 分不出是覆盖变好还是精度变差。
    cq = torch.zeros(4, 2, 2, device=device, dtype=torch.float64)
    # model.eval() 不关梯度: 没有 no_grad 的话每个 batch 的激活都被 stats 里的
    # 累加项通过计算图引用着, 整个 val 集 (882 样本) 的图会一直挂到循环结束。
    _blend_seen = None
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                             dtype=amp_dtype or torch.float16, enabled=use_amp):
            outputs = model(batch, step=step)
        if _blend_seen is None:
            rd = outputs.get("range_diag") or {}
            for _st in ("stage2", "stage3", "stage4"):
                if isinstance(rd.get(_st), dict) and "ctrl_blend" in rd[_st]:
                    _blend_seen = float(rd[_st]["ctrl_blend"]); break
            else:
                _blend_seen = -1.0        # legacy_depth: 没有 ctrl_blend, 跳过断言
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
            stats[8] += err.clamp(max=2.0).sum()
            # ---- risk@0.6: 固定保留率下的风险 -----------------------------
            # 点云融合本质上是一次**选择**, 所以无条件均值 (abs_err / capped_2mm)
            # 都不是它的代理; 保留率固定时的条件均值才是。排序信号用与推理端
            # 融合门控**完全同一份** cascade_confidence (models/fusion_conf.py),
            # 否则训练时量的排序和融合时用的排序是两个东西。
            # 逐图算 top-k 再按保留像素数加权 —— 各图 risk 直接平均会给小图过高
            # 权重, 而 DTU 每张图的有效像素数差别很大。
            try:
                conf = cascade_confidence(outputs, window=1, mode="product")
            except KeyError:
                conf = None
            if conf is not None:
                for b in range(pred.shape[0]):
                    mb = m[b]
                    nb = int(mb.sum())
                    if nb == 0:
                        continue
                    kb = int(math.ceil(0.6 * nb))
                    cb = conf[b][mb].float()
                    eb = (pred[b][mb] - gt[b][mb]).abs()
                    idx = cb.argsort(descending=True, stable=True)[:kb]
                    stats[9] += eb[idx].sum()
                    stats[10] += kb
        if m.any() and "depth_values" in batch:
            lo = dv.amin(dim=1).view(-1, 1, 1)
            hi = dv.amax(dim=1).view(-1, 1, 1)
            t01 = ((gt - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)
            bidx = (t01 * 4.0).long().clamp(0, 3)
            for b in range(4):
                mb = m & (bidx == b)
                if not mb.any():
                    continue
                eb = (pred[mb] - gt[mb]).abs()
                bq[b, 0] += eb.sum()
                bq[b, 1] += mb.sum()
                bq[b, 2] += (eb < 2.0).sum()
                bq[b, 3] += (eb >= 8.0).sum()
            for si, nm in enumerate(("stage2", "stage3", "stage4")):
                st = outputs.get(nm)
                if not isinstance(st, dict) or "depth_hypos" not in st:
                    continue
                hy = st["depth_hypos"].float()
                hw = tuple(hy.shape[-2:])
                g_ = F.interpolate(gt.unsqueeze(1), size=hw, mode="nearest").squeeze(1)
                m_ = F.interpolate(m.float().unsqueeze(1), size=hw, mode="nearest").squeeze(1) > 0.5
                b_ = F.interpolate(bidx.float().unsqueeze(1), size=hw, mode="nearest").squeeze(1).long()
                ir = (g_ >= hy[:, 0]) & (g_ <= hy[:, -1])
                for b in range(4):
                    mb = m_ & (b_ == b)
                    if mb.any():
                        sq[b, si, 0] += ir[mb].sum()
                        sq[b, si, 1] += mb.sum()
                # stage4 是 stride 1, 与 depth_full 同分辨率, 可以直接把误差按
                # 窗口内/外拆开
                if nm == "stage4" and tuple(hy.shape[-2:]) == tuple(pred.shape[-2:]):
                    e_ = (pred - gt).abs()
                    for b in range(4):
                        for j, sel in enumerate((ir, ~ir)):
                            mb = m & (bidx == b) & sel
                            if mb.any():
                                cq[b, j, 0] += e_[mb].sum()
                                cq[b, j, 1] += mb.sum()
    if is_ddp:
        dist.all_reduce(stats)
        dist.all_reduce(bq)
        dist.all_reduce(sq)
        dist.all_reduce(cq)
    model.train()
    if _blend_seen is not None and _blend_seen >= 0.0 and step is not None:
        want = min(1.0, float(step) / max(int(blend_steps), 1))
        if abs(_blend_seen - want) > 1e-6:
            raise RuntimeError(
                f"验证的 ctrl_blend={_blend_seen:.6f} 与 step={step} 期望的 {want:.6f} 不符 —— "
                f"说明 step 没传进前向, 验证跑的是另一条候选轴")
    if stats[1].item() == 0:
        raise RuntimeError(
            "validation produced no valid depth pixels — the val split is "
            "misconfigured (empty list, missing GT, or all-zero masks)"
        )
    n = stats[1]
    extra = {}
    if _blend_seen is not None and _blend_seen >= 0.0:
        # val/ctrl_blend: 一眼看出验证跑在哪条轴上。它必须与训练同步爬到 1.0。
        extra["ctrl_blend"] = float(_blend_seen)
    for b in range(4):
        nb = bq[b, 1]
        if nb <= 0:
            continue
        extra[f"q{b}_abs_err"] = float(bq[b, 0] / nb)
        extra[f"q{b}_acc_2mm"] = float(bq[b, 2] / nb)
        extra[f"q{b}_tail_frac_8mm"] = float(bq[b, 3] / nb)
        extra[f"q{b}_pixel_frac"] = float(nb / n)
        for si, nm in enumerate(("s2", "s3", "s4")):
            ns = sq[b, si, 1]
            if ns > 0:
                extra[f"q{b}_{nm}_in_range"] = float(sq[b, si, 0] / ns)
        for j, nm in enumerate(("in", "out")):
            nc = cq[b, j, 1]
            if nc > 0:
                extra[f"q{b}_abs_err_s4_{nm}"] = float(cq[b, j, 0] / nc)
    return {
        **extra,
        "abs_err": float(stats[0] / n),
        "acc_2mm": float(stats[2] / n),
        "acc_4mm": float(stats[3] / n),
        "acc_8mm": float(stats[4] / n),
        "abs_err_body_2mm": float(stats[5] / stats[2].clamp(min=1)),
        "abs_err_body_4mm": float(stats[6] / stats[3].clamp(min=1)),
        "abs_err_tail_8mm": float((stats[0] - stats[7]) / (n - stats[4]).clamp(min=1)),
        "tail_frac_8mm": float((n - stats[4]) / n),
        "capped_2mm": float(stats[8] / n),
        **({"risk_at_0.6": float(stats[9] / stats[10])} if stats[10] > 0 else {}),
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


def load_checkpoint(path, map_location=None):
    """读我们自己写的 checkpoint。

    torch 2.6 把 ``torch.load`` 的 ``weights_only`` 默认值改成了 True, 而
    TrainLogger.save 会把整份配置快照 (含 numpy 对象) 一起存进去 —— 于是
    ``--resume auto`` 和所有离线工具在新 torch 上都会 UnpicklingError。
    这些文件是本仓库自己产出的, 显式关掉即可; 老 torch 没有这个参数, 兜底重试。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _amp_dtype(cfg) -> torch.dtype:
    return torch.bfloat16 if str(getattr(cfg.train, "amp_dtype", "fp16")).lower() == "bf16" \
        else torch.float16


def _seed_everything(seed: int, rank: int, deterministic: bool = False) -> None:
    """固定 python / numpy / torch / cuda 的随机源。

    数据增广那一侧不靠全局状态: data/dtu.py 用 SeedSequence([seed, epoch, idx])
    给每个样本独立播种, 所以 worker 数变化不会改变增广序列。这里管的是模型
    初始化、dropout 之类的全局流。

    ``deterministic`` 只动 cudnn 的两个开关, 不调
    ``torch.use_deterministic_algorithms``: 后者会让 grid_sample 的反向直接抛
    异常 (没有确定性实现), 而单应 warp 全靠它。
    """
    import random as _random
    _random.seed(seed + rank)
    np.random.seed((seed + rank) % (2 ** 32))
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _rng_state() -> dict:
    """四条随机流的现场。数据增广不靠它们 (data/dtu.py 用 SeedSequence
    [seed, epoch, idx] 逐样本播种), 但模型侧的 dropout / 采样仍然走全局流。"""
    import random as _random
    return {
        "python": _random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict | None) -> bool:
    if not state:
        return False
    import random as _random
    try:
        _random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() if torch.is_tensor(s) else s for s in state["cuda"]])
        return True
    except Exception as exc:                   # 老 checkpoint 里没有 / 格式对不上
        print(f"[resume] RNG 状态无法恢复 ({exc}) —— 继续, 但这一次不是精确续训")
        return False


def _fingerprint_diff(saved: dict | None, current: dict) -> dict:
    """checkpoint 的指纹 vs 当前配置。返回 {key: (存的, 现在的)}。"""
    if not saved:
        return {}
    def norm(v):                               # torch.save 会把 tuple 原样带回来
        return list(v) if isinstance(v, (list, tuple)) else v
    return {k: (saved.get(k), current.get(k))
            for k in sorted(set(saved) | set(current))
            if norm(saved.get(k)) != norm(current.get(k))}


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
        # 以下都不改 state_dict 的形状, 但都会改变结果。少了它们, 两个
        # checkpoint 看起来"可互换"其实跑的是不同实验。
        "lr": cfg.train.lr,
        "warmup_steps": cfg.train.warmup_steps,
        "max_steps": cfg.train.max_steps,
        "batch_size": cfg.train.batch_size,
        "num_views": cfg.train.num_views,
        "seed": cfg.train.seed,
        "amp_dtype": getattr(cfg.train, "amp_dtype", "fp16"),
        "prior_mode": getattr(cfg.prior, "mode", "on"),
        "prior_cache_path": str(cfg.paths.prior_cache_path),
        "depth_adaptive": [cfg.depth_range.depth_adaptive_range,
                           cfg.depth_range.depth_adaptive_max,
                           cfg.depth_range.depth_adaptive_apply,
                           list(cfg.depth_range.depth_adaptive_stages)],
        "branch_prior_mode": cfg.depth_range.branch_prior_mode,
        "branch_prior_anneal_steps": cfg.depth_range.branch_prior_anneal_steps,
        "warp_channels": [cfg.cost_volume.warp_channels_stage1, cfg.cost_volume.warp_channels_stage2,
                          cfg.cost_volume.warp_channels_stage3, cfg.cost_volume.warp_channels_stage4],
        "num_groups": cfg.cost_volume.num_groups,
        "lr_schedule_steps": getattr(cfg.train, "lr_schedule_steps", None),
        "stage_weights": [cfg.stage_weights.stage1, cfg.stage_weights.stage2,
                          cfg.stage_weights.stage3, cfg.stage_weights.stage4],
        "w_branch": cfg.loss.w_branch,
        "w_spre": cfg.loss.w_spre,
        "spre_balance_corrupt": cfg.loss.spre_balance_corrupt,
        "range_k": list(cfg.depth_range.range_k),
        "range_min_gi": list(cfg.depth_range.range_min_gi),
        "axis_space": cfg.depth_range.axis_space,
        "axis_blend_steps": cfg.depth_range.axis_blend_steps,
        "rho_stages": (list(cfg.depth_range.rho_stages)
                       if cfg.depth_range.rho_stages else None),
        "child_interval_cap": cfg.depth_range.child_interval_cap,
        "refine_ratio_init": list(cfg.depth_range.refine_ratio_init),
        "refine_cap_p": cfg.depth_range.refine_cap_p,
        "pinball_scale": cfg.loss.pinball_scale,
        "stage4_head": cfg.depth_range.stage4_head,
        "mode_window_stages": (list(cfg.depth_range.mode_window_stages)
                               if cfg.depth_range.mode_window_stages else None),
        "spre_cascade": cfg.depth_range.spre_cascade,
        "tau_stages": list(cfg.depth_range.tau_stages),
        "geo_valid_aggregation": cfg.cost_volume.geo_valid_aggregation,
        "vis_mode": cfg.cost_volume.vis_mode,
        "vis_supervise": cfg.cost_volume.vis_supervise,
        "fusion_conf": cfg.decoder.fusion_conf,
        "fusion_conf_detach": cfg.decoder.fusion_conf_detach,
        # CVPE。plane_space / feature_stride / align_corners 是**写死的约定**而不是
        # 配置项, 但仍然进 fingerprint —— 将来若有人改了实现, 旧 checkpoint 必须
        # 立刻不匹配, 而不是安静地跑出另一套几何。
        "cvpe_enabled": cfg.cvpe.enabled,
        "cvpe_d_model": cfg.cvpe.d_model,
        "cvpe_num_planes": cfg.cvpe.num_planes,
        "cvpe_n_heads": cfg.cvpe.n_heads,
        "cvpe_cam_mid_channels": cfg.cvpe.cam_mid_channels,
        "cvpe_layer_pattern": cfg.cvpe.layer_pattern,
        "cvpe_feature_stride": 8,
        "cvpe_plane_space": "inverse_global",
        "cvpe_align_corners": True,
        "range_max_gi": cfg.depth_range.range_max_gi,
        "local_half_gi": [cfg.depth_range.local_half_min_gi, cfg.depth_range.local_half_max_gi],
        "gate_hard_conf": cfg.depth_range.gate_hard_conf,
        "prior_corruption_prob": cfg.train.prior_corruption_prob,
        "prior_cache_version": _prior_cache_version(),
    }


def _prior_cache_version() -> float:
    """缓存里 pipeline_version 的取值 (抽一个样本)。先验重建过就会变, 而它
    完全不体现在权重里 —— 8.16 那次回退里先验缓存正好也换了一版。"""
    try:
        import glob
        f = sorted(glob.glob(str(ProjectPaths().prior_cache_path / "*" / "*.npz")))
        if not f:
            return -1.0
        return float(np.asarray(np.load(f[0])["pipeline_version"]).reshape(-1)[0])
    except Exception:
        return -1.0


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
        seed=cfg.train.seed,
        use_src_weights=cfg.cost_volume.use_src_weights,
        load_src_depth=cfg.cost_volume.vis_supervise,
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
    _gen = torch.Generator()
    _gen.manual_seed(cfg.train.seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=_collate,
        worker_init_fn=_worker_init,
        generator=_gen,
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
        load_src_depth=cfg.cost_volume.vis_supervise,
    )
    # A silently-empty val split poisons best.pth (val abs_err computes to 0.0
    # at the first validation and can never be beaten). Fail loudly instead.
    if len(val_dataset) == 0:
        raise RuntimeError(f"validation dataset is empty — check {_va_list}")
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None
    val_bs = getattr(args, "val_batch_size", None) or cfg.train.batch_size
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=_collate,
        worker_init_fn=_worker_init,
        pin_memory=True,
        drop_last=False,
    )

    max_steps = args.steps if args.steps else cfg.train.max_steps
    # 已经跑完的 run 被原样重投: --resume auto 捡回 step=max_steps 的 latest.pth,
    # while 循环一次都不进, 然后照常走最后那次验证并打印 [val final step N+1] ——
    # 一行看起来完全正常的结果, 实际上零训练步, 白烧一个 GPU 名额 (2026-08-18,
    # D0/L/R 三个作业各中一次)。这里明确拦下来, 并说清楚三条出路。
    if start_step >= max_steps:
        if is_main:
            print(
                f"\n[已完成] {args.name}: latest.pth 已经在 step {start_step - 1}, "
                f"而 --steps={max_steps}。没有可训练的步数, 直接退出。\n"
                f"  想继续训练 -> 调大 --steps;\n"
                f"  想重跑一次 -> 换 --name (checkpoint 按 run 隔离, 不会打架), "
                f"或 --resume off 并先把旧的 latest.pth 挪走;\n"
                f"  只想重新验证 -> 用 scripts/eval_pointcloud.sh, 别走训练入口。\n"
            )
        return
    model.train()
    use_amp = cfg.train.amp and device.type == "cuda"
    meter = WindowedMeter(device, is_ddp)
    step = start_step
    epoch = start_step // max(len(loader), 1)
    # 续训位置。以前 resume 只恢复 epoch, 然后从 epoch 开头重新迭代 —— 本 epoch
    # 里已经训过的那些 batch 会被再训一遍, 样本顺序跟不中断的那次对不上。位置是
    # step % len(loader) 的确定性函数 (采样顺序由 sampler.set_epoch(epoch) 决定),
    # 所以不必存进 checkpoint, 这里重新算出来跳过即可。跳过仍要走一遍 DataLoader
    # (拿不到"直接 seek 到第 k 个 batch"的接口), 代价是几分钟 IO, 换的是顺序精确。
    skip_in_epoch = start_step % max(len(loader), 1)
    while step < max_steps:
        sampler.set_epoch(epoch)
        # 逐样本增广的种子含 epoch, 所以每个 epoch 是一组新的、但可复现的增广。
        dataset.set_epoch(epoch)
        # Multi-scale: bucket this epoch's sampling order so every batch shares
        # one resolution (collate cannot stack mixed H x W). Must run AFTER
        # set_epoch — the order it buckets is the order the loader will draw —
        # and before iterating, so forked workers inherit the plan.
        dataset.reset_scale_plan(list(sampler), cfg.train.batch_size)
        if skip_in_epoch and is_main:
            print(f"[resume] 跳过本 epoch 已消费的前 {skip_in_epoch} 个 batch, "
                  f"从 epoch {epoch} 的第 {skip_in_epoch} 个 batch 接上")
        for i, batch in enumerate(loader):
            if i < skip_in_epoch:
                continue
            if step >= max_steps:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            lr = _lr_at(cfg, step, max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            logs, outputs = _train_step(model, loss_fn, optimizer, scaler, batch, cfg,
                                        device, step, use_amp, lr=lr)

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
                val_metrics = _run_validation(model, val_loader, device, use_amp, is_ddp,
                                              _amp_dtype(cfg), step=step,
                                              blend_steps=cfg.depth_range.axis_blend_steps)
                if is_main:
                    logger.log_val(val_metrics, step)
                    logger.save(model, optimizer, step, val_metric=val_metrics["abs_err"])
                    print(f"[val step {step}] " + _fmt_val(val_metrics))
            elif is_main and step > 0 and step % cfg.train.ckpt_interval == 0:
                logger.save(model, optimizer, step)
            step += 1
        skip_in_epoch = 0            # 只对 resume 后的第一个 epoch 生效
        epoch += 1

    # Final validation so the last weights are also considered for best.pth.
    val_metrics = _run_validation(model, val_loader, device, use_amp, is_ddp,
                                  _amp_dtype(cfg), step=step,
                                  blend_steps=cfg.depth_range.axis_blend_steps)
    if is_main:
        logger.log_val(val_metrics, step)
        logger.save(model, optimizer, step, val_metric=val_metrics["abs_err"])
        print(f"[val final step {step}] " + _fmt_val(val_metrics))


def _worker_init(worker_id: int) -> None:
    """Keep每个 DataLoader worker 单线程。

    torch 在 worker 里本来就是单线程, 但 OpenCV 不是: 它的线程池默认取满核数。
    dtu.py 每个样本要调十几次 cv2.resize, 于是 N 个 worker 各自开 N_core 个线程:
    单卡 16 worker 在 64 核上已经超订, DDP 两个 rank 就是 32 个 worker, 直接翻倍。
    争用导致某个 rank 取一个 batch 要几分钟 —— 单卡时只是变慢, DDP 下另一个 rank
    会卡在梯度 all-reduce 上被 NCCL watchdog (默认 10 分钟) 判定超时并杀掉作业。
    """
    import cv2
    import random as _random

    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    # torch 已经给每个 worker 派了不同的 base seed; numpy / random 不会自动跟随。
    # 数据增广本身已改成逐样本 SeedSequence 播种, 不依赖这里 —— 这道只是兜底,
    # 保证以后有人在 dataset 里加了 np.random.* 也不会重新引入不可复现。
    _s = torch.initial_seed() % (2 ** 32)
    np.random.seed(_s)
    _random.seed(_s)


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
    parser.add_argument("--val-batch-size", type=int, default=None,
                        help="验证用的 batch (默认跟训练一致)。验证只做前向且在 no_grad 下, "
                             "显存远小于训练峰值, 而按像素加权的指标与 batch 无关 —— "
                             "调大它纯粹是省时间, 不会改变任何数字。")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count override")
    parser.add_argument("--num-views", type=int, default=None, help="number of MVS input views override")
    parser.add_argument("--lr", type=float, default=None, help="learning-rate override")
    parser.add_argument("--warmup-steps", type=int, default=None, help="LR warmup steps override")
    parser.add_argument("--val-interval", type=int, default=None, help="steps between validation runs")
    parser.add_argument("--log-interval", type=int, default=None,
                        help="每多少步往 tensorboard 刷一次窗口均值 (默认 50)。查发散时"
                             "调小到 10 —— 梯度范数的尖峰会被 50 步的均值抹平。")
    parser.add_argument("--amp", choices=["on", "off"], default=None, help="AMP override")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default=None,
                        help="AMP 的计算精度。bf16 指数范围与 fp32 相同, 用来消除 stage4 "
                             "全分辨率相关体的 inf 溢出; bf16 下自动关闭 GradScaler。")
    parser.add_argument("--lr-schedule-steps", type=int, default=None,
                        help="余弦退火 horizon (默认=实际训练步数)。12k 筛选应设 30000, "
                             "这样选中的 checkpoint 能无 lr 跳变续训到 30k。")
    parser.add_argument("--prior-target-w", type=int, default=None,
                        help="VGGT/DA3 prior width override (default 518; must be a multiple of 14). "
                             "Raises true depth-prior resolution at VGGT compute/memory cost. "
                             "Changing it needs --build-priors force to rebuild the cache.")
    parser.add_argument("--prior-target-h", type=int, default=None,
                        help="VGGT/DA3 prior height override (default 420; must be a multiple of 14)")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--resume", choices=["auto", "off"], default="auto",
                        help="auto: continue from log/experiments/<run>/model/latest.pth; off: always start fresh")
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
    # --- 消融开关: 全部走 CLI, 这样同一份 commit 可以并行提交多组实验 ---
    parser.add_argument("--num-global", type=int, default=None,
                        help="stage1 全局分支候选数 (num_depths_stage1 自动跟随 = global+local)")
    parser.add_argument("--num-local", type=int, default=None,
                        help="stage1 局部分支候选数")
    parser.add_argument("--gate-local", choices=["on", "off"], default=None,
                        help="conf < gate_hard_conf 时把 local 分支降级为 guard 网格")
    parser.add_argument("--branch-prior", choices=["on", "off"], default=None,
                        help="stage1 logits 上叠 log q / log(1-q) 的分支先验")
    parser.add_argument("--visibility", choices=["on", "off"], default=None,
                        help="逐 (source, pixel) 可见性加权 (仅 stage1)")
    parser.add_argument("--stage1-weight", type=float, default=None,
                        help="stage1 的 loss 权重")
    parser.add_argument("--w-branch", type=float, default=None,
                        help="分支校准 loss 权重 (0 = 关闭)")
    parser.add_argument("--spre-balance-corrupt", choices=["on", "off"], default=None,
                        help="SPRE corruption BCE 是否按 clean/corrupt 两类平衡")
    parser.add_argument("--prior", choices=["on", "off", "affine"], default=None,
                        help="先验通路。off = 严格无先验对照 (prior/conf 置零、meta 中性化、"
                             "branch prior 与 branch loss 关闭、SPRE 不参与前向), 模块仍照常"
                             "构造所以 RNG 流一致; affine = 用逆深度仿射标定过的缓存")
    parser.add_argument("--prior-cache", type=str, default=None,
                        help="覆盖先验缓存目录 (--prior affine 时指向仿射标定的那一份)")
    # ================= W1 / W3 开关 (默认全部等价于旧行为) =================
    parser.add_argument("--axis-space", choices=["legacy_depth", "inverse"], default=None,
                        help="legacy_depth=现有实现(默认); inverse=可学习控制器+逆深度轴")
    parser.add_argument("--axis-blend-steps", type=int, default=None,
                        help="legacy 轴 -> inverse 轴的凸组合迁移步数, 默认 2000")
    parser.add_argument("--tau-stages", type=str, default=None,
                        help="三级 pinball 目标覆盖率, 逗号分隔, 默认 0.98,0.95,0.92")
    parser.add_argument("--rho-max", type=float, default=None,
                        help="半宽相对基准的倍率界 [h0/rho, h0*rho], 默认 8")
    parser.add_argument("--rho-stages", type=str, default=None,
                        help="逐级倍率**安全界**, 逗号分隔如 '8,4,2'。优先于 --rho-max。"
                             "它不负责级联细化 —— 那是 --child-interval-cap 的职责")
    parser.add_argument("--child-interval-cap", choices=["on", "off"], default=None,
                        help="子级最大逆深度候选间隔 <= xi_k * 父级间隔。"
                             "取代半宽单调: D 逐级减半时半宽下降并不蕴含间距下降")
    parser.add_argument("--refine-ratio-init", type=str, default=None,
                        help="xi_k 初值, 三个 (0,1) 内的值, 默认 0.4,0.5142857142857142,0.8")
    parser.add_argument("--refine-cap-p", type=float, default=None,
                        help="间隔上界的平滑锐度, >= 2, 默认 16")
    parser.add_argument("--pinball-scale", choices=["axis", "global"], default=None,
                        help="pinball 分母: axis=上一级局部间距(旧); "
                             "global=全局逆深度间距 g_v, 切断'轴变宽->惩罚变小'的反馈")
    parser.add_argument("--spre-cascade", choices=["on", "off"], default=None,
                        help="SPRE 的 r/e 作为 meta 通道进入 stage2-4 的正则器")
    parser.add_argument("--spre-gate-init", type=str, default=None,
                        help="四级 SPRE 门初值, 第一个必须是 1.0, 默认 1.0,0.6,0.35,0.2")
    parser.add_argument("--stage4-head", choices=["expect", "map"], default=None,
                        help="expect=现有的 mode-centered 期望; map=硬选面+逐候选残差")
    parser.add_argument("--mode-window-stages", type=str, default=None,
                        help="四级各自的 mode_centered_regression 半窗, 逗号分隔")
    parser.add_argument("--geo-valid", choices=["on", "off"], default=None,
                        help="W3-A: 逐假设几何有效 mask + 逐 voxel 归一化 + n_valid 通道")
    # CVPE 只暴露一个开关。d_model / num_planes / n_heads / layer_pattern 一律
    # 走 base.config 的固定值 —— 它们一旦变成训练超参, 就又多出一条按中间指标
    # 逐项调参的支线, 而工单 v5.3 的整个前提就是不再这么干。
    parser.add_argument("--cvpe", choices=["on", "off"], default=None,
                        help="CVPE: 相机感知的跨视位置编码, 注入 FPN 的 1/8 瓶颈")
    parser.add_argument("--vis-mode", choices=["softmax", "sigmoid"], default=None,
                        help="softmax=旧的 source 维竞争; sigmoid=多标签独立可见性")
    parser.add_argument("--w-range", type=float, default=None)
    parser.add_argument("--w-center", type=float, default=None)
    parser.add_argument("--w-residual", type=float, default=None)
    parser.add_argument("--w-oor", type=float, default=None)
    parser.add_argument("--conf-head", choices=["on", "off"], default=None,
                        help="W3-C: 学出来的融合置信度头 (替代 test.cascade_confidence)")
    parser.add_argument("--conf-detach", choices=["on", "off"], default=None,
                        help="置信度头的输入是否 detach, 默认 on (旁路, 不重塑深度特征)")
    parser.add_argument("--w-conf", type=float, default=None)
    parser.add_argument("--conf-tau-mm", type=float, default=None,
                        help="置信度标签阈值 1[|d-gt| < tau], 默认 2.0mm")
    parser.add_argument("--vis-supervise", choices=["on", "off"], default=None,
                        help="W3-B: 用源视图 GT 深度给可见性头接遮挡监督 "
                             "(要求 --vis-mode sigmoid --visibility on; "
                             "dataset 每样本多读 N-1 张 PFM)")
    parser.add_argument("--w-vis", type=float, default=None)
    parser.add_argument("--delta-occ-mm", type=float, default=None,
                        help="遮挡容差 y_vis = 1[z_src <= D_gt(pi(x)) + delta], 默认 2.0mm")
    parser.add_argument("--range-min-gi", type=str, default=None,
                        help="三个转移的窗宽下限, 逗号分隔, 如 '0.66,0.20,0.10'")
    parser.add_argument("--depth-adaptive", choices=["on", "off"], default=None,
                        help="窗宽按 clamp((d/depth_min)^2, 1, max) 随深度放宽 (只放宽远端)")
    parser.add_argument("--depth-adaptive-max", type=float, default=None)
    parser.add_argument("--depth-adaptive-apply", choices=["floor", "both"], default=None,
                        help="floor=只缩放下限; both=连 range_k*winner_interval 一起缩放")
    parser.add_argument("--depth-adaptive-stages", type=str, default=None,
                        help="施加在哪几个转移, 如 'off,on,on'")
    parser.add_argument("--branch-prior-mode", choices=["hard", "soft"], default=None,
                        help="hard=硬指派分支质量 (实测 2k 后有害); soft=计数校正的加性偏置")
    parser.add_argument("--branch-prior-anneal-steps", type=int, default=None,
                        help=">0 时 beta 线性退到 0, 对应'前 1-2k 引导、之后交还 matching'")
    parser.add_argument("--warp-channels-stage4", type=int, default=None,
                        help="stage4 的 warp 通道数 (16 -> 24/32 是最后的容量手段)")
    parser.add_argument("--num-groups", type=int, default=None,
                        help="相关体分组数。stage4 每组只有 warp_ch/groups=2 个通道做 mean, "
                             "抑制最弱, 也是 fp16 溢出的高发点")
    parser.add_argument("--seed", type=int, default=None,
                        help="全局随机种子 (模型初始化 + 采样顺序 + 逐样本增广)")
    parser.add_argument("--deterministic", action="store_true",
                        help="cudnn.deterministic=True, benchmark=False。约慢 5-10%%, "
                             "但同 seed 两次跑的曲线才真的可比")
    parser.add_argument("--expect-sha", type=str, default="",
                        help="提交任务时的 git SHA; 与运行时不符直接退出。"
                             "排队期间主工作树被改动过就会被这道闸门挡下")
    parser.add_argument("--nan-watchdog", choices=["on", "off"], default=None,
                        help="出现第一个非有限 loss/梯度就终止 (默认 on)。off 只用于"
                             "复现历史事故 —— 关掉它 GradScaler 会一路跳步空转")
    parser.add_argument("--allow-fingerprint-mismatch", action="store_true",
                        help="resume 时即使 checkpoint 指纹与当前配置不符也继续。"
                             "只在确认无害时用 (例如仅重建过先验缓存)")
    parser.add_argument("--fit-batch", type=str, default="",
                        help="按真实训练配置扫 per-GPU batch 的显存占用后退出, "
                             "如 '1,2,4,6,8'。量的是多尺度里最大的那个尺度。")
    parser.add_argument("--fit-lr-ref", type=float, default=3e-4,
                        help="--fit-batch 的 lr 基准 (未缩放), 默认 3e-4")
    parser.add_argument("--fit-lr-ref-batch", type=float, default=2,
                        help="--fit-lr-ref 对应的全局 batch, 默认 2")
    parser.add_argument("--fit-hw", type=str, default="",
                        help="--fit-batch 的尺度覆盖, 如 '448x576'。默认用多尺度里最大的。")
    parser.add_argument("--fit-target", type=float, default=0.95,
                        help="--fit-batch 的目标显存占用 (0-1), 默认 0.95")
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
