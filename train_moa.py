"""Train MoAMVSNet (single GPU, epoch-based).

    python train_moa.py --profile umhpc --name MOA_v1 --epochs 15
    python train_moa.py --profile local --name moa_local --smoke          # synthetic batches

Run layout: log/experiments/<name>/{model/latest.pth, model/best.pth, tensorboard/}.
``steps_per_epoch`` is ``len(train_loader)`` (drop_last) and the cosine LR horizon is
``epochs * steps_per_epoch``, both computed here rather than by the launcher.

SIGUSR1 / SIGTERM (slurm's pre-timeout signal) finish the current step, write
latest.pth and exit with code 124, so a launcher can requeue with --resume auto.

This entry point never imports models.network / the prior cache / SPRE.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from base.config_moa import MoAMVSConfig, arch_snapshot, build_moa_config, config_snapshot
from data.augment import PhotometricAug
from data.dtu_moa import MoADTUDataset
from losses.moa_loss import MoALoss
from models.network_moa import MoAMVSNet

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard not installed
    SummaryWriter = None

EXIT_REQUEUE = 124
FROZEN_PREFIX = "dino_sva.dino."        # frozen DINOv3 weights: reloaded from file, not checkpointed


# --------------------------------------------------------------------------- #
# generic helpers (same behaviour as train.py; copied so this entry stays independent)
# --------------------------------------------------------------------------- #
def lr_at(base_lr: float, warmup: int, step: int, horizon: int) -> float:
    """Linear warmup, then cosine decay to 5% of the base LR at ``horizon``."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(horizon - warmup, 1)
    return base_lr * max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0))))


def collate(samples: list[dict]) -> dict:
    out: dict = {}
    for k in samples[0]:
        v = samples[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        elif isinstance(v, np.ndarray):
            out[k] = torch.stack([torch.from_numpy(np.asarray(s[k])) for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


def worker_init(worker_id: int) -> None:
    import cv2
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


class EpochShuffleSampler(torch.utils.data.Sampler):
    """Per-epoch permutation fixed by ``set_epoch`` so the draw order is inspectable
    (multi-scale buckets it into per-batch resolutions)."""

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.seed = int(n), int(seed)
        self.order = list(range(self.n))

    def set_epoch(self, epoch: int) -> None:
        self.order = np.random.default_rng(self.seed + int(epoch)).permutation(self.n).tolist()

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict | None) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    except Exception as exc:
        print(f"[resume] RNG state not restored ({exc}); continuing")


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def load_checkpoint(path, map_location=None) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def trainable_state_dict(model: torch.nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if not k.startswith(FROZEN_PREFIX)}


def load_model_state(model: torch.nn.Module, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith(FROZEN_PREFIX)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")


def depth_errors(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-batch sums [n, sum|e|, <1, <2, <4, <8] (float64 on device)."""
    m = mask.bool() & (gt > 0)
    e = (pred.float() - gt.float()).abs()[m]
    return torch.stack([m.sum().double(), e.double().sum()] +
                       [(e < t).sum().double() for t in (1.0, 2.0, 4.0, 8.0)])


def metrics_from_sums(s: torch.Tensor) -> dict[str, float]:
    s = s.cpu().tolist()
    n = max(s[0], 1.0)
    return {"abs_err": s[1] / n, "acc_1mm": s[2] / n, "acc_2mm": s[3] / n,
            "acc_4mm": s[4] / n, "acc_8mm": s[5] / n, "pixels": int(s[0])}


def metric_mask(batch: dict) -> torch.Tensor:
    gt = batch["depth_gt"].float()
    dv = batch["depth_values"].float()
    return (batch["mask"].float() > 0.5) & (gt >= dv.amin(1).view(-1, 1, 1)) & (gt <= dv.amax(1).view(-1, 1, 1))


# --------------------------------------------------------------------------- #
# config / data / model
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser("MoAMVSNet training")
    p.add_argument("--profile", choices=["local", "umhpc"], default=os.environ.get("UPRMVS_PROFILE", "umhpc"))
    p.add_argument("--name", default="MOA")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="hard stop; 0 = epochs x steps_per_epoch")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--val-batch-size", type=int, default=None)
    p.add_argument("--num-views", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--amp", choices=["on", "off"], default=None)
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--val-interval", type=int, default=None, help="0 = validate only at epoch ends")
    p.add_argument("--ckpt-interval", type=int, default=None)
    p.add_argument("--multi-scale", choices=["on", "off"], default=None)
    p.add_argument("--height", type=int, default=None, help="crop height without multi-scale")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--moa", choices=["on", "off"], default="on",
                   help="off = pure 4-stage MVS cascade (MoA.md stage A baseline); no DA3 needed")
    p.add_argument("--moa-dim", type=int, default=None,
                   help="MoA width: emb/feat/evidence = D, hidden layers = 2D (default 16)")
    p.add_argument("--da3-root", default=None, help="DA3 cache root (default cfg.paths.da3_cache_path)")
    p.add_argument("--da3-missing", choices=["error", "skip"], default=None)
    p.add_argument("--train-list", default=None)
    p.add_argument("--val-list", default=None)
    p.add_argument("--max-val-samples", type=int, default=0, help="0 = whole val split")
    p.add_argument("--resume", default="auto", help="auto / off / path to a .pth")
    p.add_argument("--init-from", default=None,
                   help="load weights only (e.g. a --moa off baseline); modules absent there keep their init")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="MoA.md stage B: train only MoA (DINO/FPN/SVA/cost volume/decoders frozen)")
    p.add_argument("--keep-epoch-ckpts", action="store_true")
    p.add_argument("--smoke", action="store_true", help="synthetic batches, no dataset")
    p.add_argument("--smoke-steps", type=int, default=20)
    p.add_argument("--smoke-hw", type=int, nargs=2, default=(256, 320))
    return p.parse_args(argv)


def build_config(args) -> MoAMVSConfig:
    cfg = build_moa_config(args.profile)
    t = cfg.train
    upd = {}
    for name, attr in [("epochs", "epochs"), ("max_steps", "max_steps"), ("batch_size", "batch_size"),
                       ("val_batch_size", "val_batch_size"), ("num_views", "num_views"),
                       ("num_workers", "num_workers"), ("lr", "lr"), ("warmup_steps", "warmup_steps"),
                       ("weight_decay", "weight_decay"), ("grad_clip", "grad_clip"),
                       ("amp_dtype", "amp_dtype"), ("seed", "seed"), ("log_interval", "log_interval"),
                       ("val_interval", "val_interval"), ("ckpt_interval", "ckpt_interval"),
                       ("height", "height"), ("width", "width"), ("da3_missing", "da3_missing")]:
        v = getattr(args, name)
        if v is not None:
            upd[attr] = v
    if args.amp is not None:
        upd["amp"] = args.amp == "on"
    if args.multi_scale is not None:
        upd["multi_scale"] = args.multi_scale == "on"
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(t, **upd))
    moa = dataclasses.replace(cfg.moa, enabled=args.moa == "on")
    if args.moa_dim:
        d = int(args.moa_dim)
        moa = dataclasses.replace(moa, emb_dim=d, feat_dim=d, evidence_dim=d, evidence_hidden=d,
                                  conf_hidden=2 * d, mix_hidden=2 * d, shape_width=2 * d)
    return dataclasses.replace(cfg, moa=moa)


def build_datasets(cfg: MoAMVSConfig, args):
    t = cfg.train
    da3_root = Path(args.da3_root) if args.da3_root else Path(cfg.paths.da3_cache_path)
    load_mono = cfg.moa.enabled
    aug = cfg.augment
    train_ds = MoADTUDataset(
        cfg.paths.dtu_train_root, args.train_list or str(cfg.paths.train_list_file),
        nviews=t.num_views, mode="train", seed=t.seed,
        aug=PhotometricAug(brightness=aug.brightness, contrast=aug.contrast, saturation=aug.saturation,
                           hue=aug.hue, min_gamma=aug.min_gamma, max_gamma=aug.max_gamma)
        if aug.photometric else None,
        scales=aug.scales if t.multi_scale else (), resize_range=aug.resize_range,
        height=t.height, width=t.width,
        da3_root=da3_root, load_mono=load_mono, da3_missing=t.da3_missing)
    val_ds = MoADTUDataset(
        cfg.paths.dtu_train_root, args.val_list or str(cfg.paths.val_list_file),
        nviews=t.num_views, mode="val", seed=t.seed,
        da3_root=da3_root, load_mono=load_mono, da3_missing=t.da3_missing)
    if args.max_val_samples and len(val_ds.metas) > args.max_val_samples:
        idx = np.linspace(0, len(val_ds.metas) - 1, args.max_val_samples).round().astype(int)
        val_ds.metas = [val_ds.metas[i] for i in idx]
    return train_ds, val_ds


def synthetic_batch(cfg: MoAMVSConfig, device, batch_size: int, hw) -> dict:
    B, V = batch_size, cfg.train.num_views
    H, W = hw
    dmin, interval, nd = 425.0, 2.5, 192
    dv = torch.arange(nd, dtype=torch.float32) * interval + dmin
    yy = torch.linspace(0, 1, H).view(1, H, 1)
    gt = (dmin + 150 + 200 * yy).expand(B, H, W).contiguous()
    batch = {
        "images": torch.rand(B, V, 3, H, W) * 255.0,
        "intrinsics": torch.tensor([[[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]]]).repeat(B, V, 1, 1),
        "extrinsics": torch.eye(4).repeat(B, V, 1, 1),
        "depth_values": dv.unsqueeze(0).repeat(B, 1),
        "depth_gt": gt,
        "mask": (torch.rand(B, H, W) > 0.2).float(),
        "mono_depth": 0.002 * gt + 0.3 + 0.01 * torch.rand(B, H, W),
    }
    batch["extrinsics"][:, 1:, 0, 3] = 5.0
    return {k: v.to(device) for k, v in batch.items()}


def set_train_mode(model: MoAMVSNet, freeze_backbone: bool) -> None:
    model.train()
    if freeze_backbone:
        for m in (model.dino_sva, model.fpn, model.sva_pathway, model.cost_volumes, model.decoders):
            if m is not None:
                m.eval()


# --------------------------------------------------------------------------- #
# logging / checkpoints
# --------------------------------------------------------------------------- #
class RunLogger:
    def __init__(self, name: str, project: Path) -> None:
        self.root = project / "log" / "experiments" / name
        self.model_dir = self.root / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(str(self.root / "tensorboard")) if SummaryWriter else None
        self.best = float("inf")

    def scalars(self, prefix: str, values: dict, step: int) -> None:
        if self.tb is None:
            return
        for k, v in values.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                self.tb.add_scalar(f"{prefix}/{k}", float(v), step)

    def save(self, name: str, payload: dict) -> Path:
        path = self.model_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path


def make_payload(model, optimizer, cfg, step, epoch, steps_per_epoch, max_steps, best, da3_res) -> dict:
    """``optimizer=None`` for weights-only files (best.pth / epoch snapshots)."""
    return {
        "kind": "moa_mvsnet", "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step, "epoch": epoch, "steps_per_epoch": steps_per_epoch, "max_steps": max_steps,
        "best_metric": best, "best_metric_name": "val_abs_err",
        "arch": arch_snapshot(cfg), "config": config_snapshot(cfg), "git": git_state(),
        "da3_process_res": da3_res, "rng": rng_state(),
    }


# --------------------------------------------------------------------------- #
# train / validate
# --------------------------------------------------------------------------- #
class NonFiniteError(RuntimeError):
    pass


def train_step(model, loss_fn, optimizer, scaler, params, batch, cfg, device, diag: bool):
    t = cfg.train
    use_amp = t.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if t.amp_dtype == "bf16" else torch.float16
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        out = model(batch)
    loss, logs = loss_fn(out, batch, diagnostics=diag)
    if not torch.isfinite(loss):
        bad = {k: float(v) for k, v in logs.items() if not torch.isfinite(v)}
        raise NonFiniteError(f"non-finite loss; offending terms: {bad} "
                             f"samples: {list(zip(batch.get('scan', []), batch.get('ref_view', [])))}")
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    # Clipped per group: one global norm would let the (large, early) MoA loss
    # shrink the backbone's update, coupling the two through the optimizer even
    # though no MoA gradient reaches the backbone.
    norms = {name: torch.nn.utils.clip_grad_norm_(ps, t.grad_clip) for name, ps in params.items() if ps}
    ok = all(bool(torch.isfinite(n)) for n in norms.values())
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    elif ok:
        optimizer.step()
    for name, n in norms.items():
        logs[f"grad_norm_{name}"] = n.detach()
    return out, logs, ok


@torch.no_grad()
def validate(model, loader, loss_fn, cfg, device, freeze_backbone: bool) -> dict:
    t = cfg.train
    use_amp = t.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if t.amp_dtype == "bf16" else torch.float16
    model.eval()
    sums = torch.zeros(6, dtype=torch.float64, device=device)
    agg: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch)
        sums += depth_errors(out["depth_full"], batch["depth_gt"], metric_mask(batch))
        _, logs = loss_fn(out, batch, diagnostics=True)
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
    set_train_mode(model, freeze_backbone)
    m = metrics_from_sums(sums)
    m.update({k: v / max(n, 1) for k, v in agg.items()})
    return m


def fmt(m: dict, keys=("abs_err", "acc_2mm", "acc_4mm")) -> str:
    return "  ".join(f"{k}={m[k]:.4f}" for k in keys if k in m)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    t = cfg.train
    project = Path(cfg.paths.project_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(t.seed, args.deterministic)

    model = MoAMVSNet(cfg).to(device)
    if args.init_from:
        ck = load_checkpoint(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        missing = [k for k in missing if not k.startswith(FROZEN_PREFIX)]
        print(f"[init] {args.init_from}: {len(missing)} params kept at init "
              f"(e.g. {missing[:3]}), {len(unexpected)} unexpected")
    if args.freeze_backbone:
        if model.moa is None:
            raise SystemExit("--freeze-backbone with --moa off leaves nothing to train")
        for m in (model.fpn, model.sva_pathway, model.cost_volumes, model.decoders, model.dino_sva):
            if m is not None:
                m.requires_grad_(False)
    moa_ids = {id(p) for p in model.moa.parameters()} if model.moa is not None else set()
    params = {
        "backbone": [p for p in model.parameters() if p.requires_grad and id(p) not in moa_ids],
        "moa": [p for p in model.parameters() if p.requires_grad and id(p) in moa_ids],
    }
    all_params = params["backbone"] + params["moa"]
    print(f"[model] trainable backbone {sum(p.numel() for p in params['backbone']) / 1e6:.2f}M + "
          f"MoA {sum(p.numel() for p in params['moa']) / 1e6:.3f}M params, "
          f"moa={'on' if model.moa is not None else 'off'}, sva_full={cfg.sva.full}")
    optimizer = torch.optim.AdamW(all_params, lr=t.lr, weight_decay=t.weight_decay)
    scaler = (torch.amp.GradScaler("cuda") if (t.amp and t.amp_dtype == "fp16" and device.type == "cuda")
              else None)
    loss_fn = MoALoss(cfg.loss, cfg.cascade.num_depths[0])

    if args.smoke:
        run_smoke(model, loss_fn, optimizer, scaler, params, cfg, device, args)
        return

    train_ds, val_ds = build_datasets(cfg, args)
    sampler = EpochShuffleSampler(len(train_ds), t.seed)
    gen = torch.Generator()
    gen.manual_seed(t.seed)
    loader = DataLoader(train_ds, batch_size=t.batch_size, sampler=sampler, num_workers=t.num_workers,
                        collate_fn=collate, worker_init_fn=worker_init, generator=gen,
                        pin_memory=True, drop_last=True, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=t.val_batch_size, shuffle=False,
                            num_workers=t.num_workers, collate_fn=collate, worker_init_fn=worker_init,
                            pin_memory=True, drop_last=False)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise SystemExit(f"train split has {len(train_ds)} samples < batch {t.batch_size}")
    horizon = t.epochs * steps_per_epoch
    max_steps = t.max_steps if t.max_steps and t.max_steps > 0 else horizon

    logger = RunLogger(args.name, project)
    start_step = 0
    ckpt_path = None
    if args.resume == "auto":
        cand = logger.model_dir / "latest.pth"
        ckpt_path = cand if cand.is_file() else None
    elif args.resume not in ("off", "", None):
        ckpt_path = Path(args.resume)
    if ckpt_path is not None:
        ck = load_checkpoint(ckpt_path, map_location="cpu")
        if ck.get("arch") != arch_snapshot(cfg):
            raise SystemExit(f"[resume] {ckpt_path} was trained with a different architecture; "
                             f"use a new --name or --resume off")
        load_model_state(model, ck["model"])
        if ck.get("optimizer") is None:
            raise SystemExit(f"[resume] {ckpt_path} has no optimizer state (weights-only); "
                             f"use --init-from for weights, or resume from latest.pth")
        optimizer.load_state_dict(ck["optimizer"])
        restore_rng(ck.get("rng"))
        start_step = int(ck["step"])
        logger.best = float(ck.get("best_metric", float("inf")))
        if int(ck.get("steps_per_epoch", steps_per_epoch)) != steps_per_epoch:
            print(f"[resume] WARNING steps_per_epoch {ck.get('steps_per_epoch')} -> {steps_per_epoch} "
                  f"(dataset or batch changed); epoch boundaries shift")
        print(f"[resume] {ckpt_path} at step {start_step}")
    if start_step >= max_steps:
        print(f"[done] {args.name} already at step {start_step} >= {max_steps}; nothing to train")
        return

    (logger.root / "config.json").write_text(json.dumps(config_snapshot(cfg), indent=2))
    print("=" * 72)
    print(f" run={args.name}  epochs={t.epochs}  steps/epoch={steps_per_epoch}  max_steps={max_steps}")
    print(f" batch={t.batch_size} views={t.num_views} lr={t.lr:g} warmup={t.warmup_steps} "
          f"amp={t.amp}/{t.amp_dtype} multi_scale={t.multi_scale}")
    print(f" train={len(train_ds)} samples  val={len(val_ds)} samples  DA3 process_res={train_ds.da3_process_res}")
    print("=" * 72)

    stop = {"sig": None}

    def _on_signal(signum, _frame):
        stop["sig"] = signum
        print(f"[signal] got {signal.Signals(signum).name}; will checkpoint after this step", flush=True)

    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    def save(name: str, step: int, epoch: int) -> None:
        opt = optimizer if name == "latest.pth" else None
        logger.save(name, make_payload(model, opt, cfg, step, epoch, steps_per_epoch, max_steps,
                                       logger.best, train_ds.da3_process_res))

    last_val = {"step": -1}

    def run_val(step: int, epoch: int) -> None:
        if last_val["step"] == step:
            return
        last_val["step"] = step
        m = validate(model, val_loader, loss_fn, cfg, device, args.freeze_backbone)
        logger.scalars("val", m, step)
        improved = m["abs_err"] < logger.best
        if improved:
            logger.best = m["abs_err"]
            save("best.pth", step, epoch)
        print(f"[val step {step} epoch {epoch}] {fmt(m)}{'  *best*' if improved else ''}", flush=True)

    set_train_mode(model, args.freeze_backbone)
    step = start_step
    epoch = step // steps_per_epoch
    skip = step % steps_per_epoch
    bad_grads = 0
    window: dict[str, float] = {}
    n_win = 0
    t_log, s_log = time.time(), step
    while step < max_steps:
        sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        train_ds.reset_scale_plan(list(sampler), t.batch_size)
        if skip:
            print(f"[resume] skipping the first {skip} batches of epoch {epoch}")
        for i, batch in enumerate(loader):
            if i < skip:
                continue
            if step >= max_steps:
                break
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            lr = lr_at(t.lr, t.warmup_steps, step, horizon)
            for g in optimizer.param_groups:
                g["lr"] = lr
            diag = step % t.log_interval == 0
            out, logs, ok = train_step(model, loss_fn, optimizer, scaler, params, batch, cfg, device, diag)
            bad_grads = 0 if ok else bad_grads + 1
            if bad_grads >= t.nan_grad_patience:
                raise NonFiniteError(f"{bad_grads} consecutive non-finite gradients at step {step}")
            if diag:
                with torch.no_grad():
                    logs.update({f"train_{k}": v for k, v in metrics_from_sums(
                        depth_errors(out["depth_full"], batch["depth_gt"], metric_mask(batch))).items()})
                vals = {k: float(v) for k, v in logs.items()}
                logger.scalars("train", vals, step)
                logger.scalars("train", {"lr": lr, "epoch": epoch + i / steps_per_epoch}, step)
                now = time.time()
                sps = (now - t_log) / max(step - s_log, 1)
                t_log, s_log = now, step
                eta_h = sps * (max_steps - step) / 3600.0
                mem = (f" mem={torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f}G"
                       if device.type == "cuda" else "")
                moa = "".join(f" c{s}={vals[f'moa{s}_err_center_mm']:.2f}/{vals[f'moa{s}_err_mvs_mm']:.2f}"
                              for s in (2, 3, 4) if f"moa{s}_err_center_mm" in vals)
                print(f"[step {step} ep {epoch}] loss={vals['loss']:.4f} "
                      f"abs_err={vals.get('train_abs_err', float('nan')):.2f} "
                      f"cover={vals.get('cover_s2', 0):.3f}/{vals.get('cover_s3', 0):.3f}/"
                      f"{vals.get('cover_s4', 0):.3f}{moa} lr={lr:.2e} {sps:.2f}s/step "
                      f"eta={eta_h:.1f}h{mem}", flush=True)
            step += 1
            if stop["sig"] is not None:
                save("latest.pth", step, epoch)
                print(f"[signal] saved latest.pth at step {step}; exiting {EXIT_REQUEUE} for requeue",
                      flush=True)
                sys.exit(EXIT_REQUEUE)
            if t.val_interval > 0 and step % t.val_interval == 0 and step % steps_per_epoch != 0:
                run_val(step, epoch)
            if step % t.ckpt_interval == 0:
                save("latest.pth", step, epoch)
        skip = 0
        if step % steps_per_epoch == 0 and step > start_step:
            done_epoch = step // steps_per_epoch
            save("latest.pth", step, done_epoch)
            if args.keep_epoch_ckpts:
                save(f"epoch_{done_epoch:02d}.pth", step, done_epoch)
            run_val(step, done_epoch)
        epoch += 1

    save("latest.pth", step, step // steps_per_epoch)
    run_val(step, step // steps_per_epoch)
    print(f"[done] {args.name}: {step} steps, best val abs_err {logger.best:.4f}")


def run_smoke(model, loss_fn, optimizer, scaler, params, cfg, device, args) -> None:
    """Synthetic batches: shapes, finiteness, backward, peak memory. No dataset needed."""
    t = cfg.train
    set_train_mode(model, args.freeze_backbone)
    hw = tuple(args.smoke_hw)
    batch = synthetic_batch(cfg, device, t.batch_size, hw)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for step in range(args.smoke_steps):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(t.lr, t.warmup_steps, step, args.smoke_steps)
        out, logs, ok = train_step(model, loss_fn, optimizer, scaler, params, batch, cfg, device, diag=True)
        if step % 5 == 0 or step == args.smoke_steps - 1:
            vals = {k: float(v) for k, v in logs.items()}
            print(f"[smoke {step}] loss={vals['loss']:.4f} grad_ok={ok} "
                  + " ".join(f"{k}={vals[k]:.3f}" for k in ("ce_s1", "ce_s4", "moa2_center", "moa2_pi_mvs",
                                                            "moa2_override_rate") if k in vals), flush=True)
    dt = (time.time() - t0) / max(args.smoke_steps, 1)
    mem = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
    print(f"[smoke] ok: {args.smoke_steps} steps, {dt:.2f}s/step, peak {mem:.2f} GiB at "
          f"batch {t.batch_size} x {t.num_views} views x {hw[0]}x{hw[1]}")


if __name__ == "__main__":
    main()
