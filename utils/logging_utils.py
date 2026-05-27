from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from utils.vis import depth_to_colormap


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class MetricMeter:
    def __init__(self) -> None:
        self._sum: dict[str, float] = defaultdict(float)
        self._cnt: dict[str, int] = defaultdict(int)

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self._sum[k] += float(v)
            self._cnt[k] += 1

    def avg(self) -> dict[str, float]:
        return {k: self._sum[k] / max(self._cnt[k], 1) for k in self._sum}

    def reset(self) -> None:
        self._sum.clear()
        self._cnt.clear()


class StepTimer:
    def __init__(self) -> None:
        self.t = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = now - self.t
        self.t = now
        return dt


def dump_metrics(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def _image_to_chw_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = _to_numpy(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        chw = arr
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        chw = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"unsupported image shape: {arr.shape}")
    if chw.dtype != np.uint8:
        if chw.max() <= 1.0 + 1e-3:
            chw = (chw.clip(0, 1) * 255).astype(np.uint8)
        else:
            chw = chw.clip(0, 255).astype(np.uint8)
    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)
    return chw


def _denormalize_imagenet(image_norm: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = _to_numpy(image_norm)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    if arr.ndim == 3 and arr.shape[0] == 3:
        return (arr * std + mean).clip(0, 1)
    raise ValueError(f"unsupported shape for imagenet denorm: {arr.shape}")


class TensorBoardLogger:
    """Thin wrapper around SummaryWriter with depth-colormap support."""

    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.log_dir = log_dir

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    def add_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.add_scalar(f"{prefix}/{k}" if prefix else k, v, step)

    def add_image_norm(self, tag: str, image_norm: torch.Tensor | np.ndarray, step: int) -> None:
        rgb = _denormalize_imagenet(image_norm)
        self.writer.add_image(tag, _image_to_chw_uint8(rgb), step)

    def add_image_raw(self, tag: str, image: torch.Tensor | np.ndarray, step: int) -> None:
        self.writer.add_image(tag, _image_to_chw_uint8(image), step)

    def add_depth(
        self,
        tag: str,
        depth: torch.Tensor | np.ndarray,
        step: int,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        d = _to_numpy(depth)
        color = depth_to_colormap(d, vmin=vmin, vmax=vmax)
        self.writer.add_image(tag, color.transpose(2, 0, 1), step)

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self.writer.add_histogram(tag, values.detach().float().cpu(), step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
