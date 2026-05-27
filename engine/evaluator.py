from __future__ import annotations

import torch


def compute_depth_metrics(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    m = mask.bool() & (depth_gt > 0) & torch.isfinite(depth_pred)
    if not m.any():
        return {"abs_rel": 0.0, "rmse": 0.0, "delta_1.25": 0.0, "acc_2mm": 0.0, "comp_2mm": 0.0}
    p = depth_pred[m]
    g = depth_gt[m]
    abs_rel = ((p - g).abs() / g).mean().item()
    rmse = ((p - g) ** 2).mean().sqrt().item()
    ratio = torch.maximum(p / g, g / p)
    delta = (ratio < 1.25).float().mean().item()
    acc = ((p - g).abs() < 2.0).float().mean().item()
    comp = ((g - p).abs() < 2.0).float().mean().item()
    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "delta_1.25": delta,
        "acc_2mm": acc,
        "comp_2mm": comp,
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    cnt = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, device)
        out = model(batch)
        metrics = compute_depth_metrics(out["depth_full"], batch["depth_gt_full"], batch["mask_full"])
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        cnt += 1
    return {k: v / max(cnt, 1) for k, v in sums.items()}


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out
