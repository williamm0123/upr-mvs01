from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; cannot init NCCL distributed training")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("LOCAL_RANK env var missing; launch with torchrun --nproc_per_node=N train.py ...")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return dist.get_rank(), dist.get_world_size(), local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def reduce_scalar_mean(value: float, device: torch.device) -> float:
    if not is_distributed():
        return float(value)
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / get_world_size())


def barrier() -> None:
    if is_distributed():
        dist.barrier()
