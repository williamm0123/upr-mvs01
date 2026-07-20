"""检查 train/val prior 的视角数，并用 3 个视角重算 val prior。

默认执行顺序：

1. 从已有 train 缓存的 ``src_weights`` 推断每个 scan 的总视角数；
2. 用相同方法检查已有 val 缓存；
3. 加载一次 VGGT/DA3；
4. 逐 scan 强制覆盖 ``val.txt`` 对应的 prior 缓存。

本脚本不导入或启动训练入口。正常进度只在一个 scan 全部完成后打印一行。

本地运行示例：

    conda run -n uprmvs --no-capture-output \
        python scripts/rebuild_val_priors.py --device cuda:0

只检查 train/val 缓存而不重算 val：

    python scripts/rebuild_val_priors.py --check-only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (
    REPO_ROOT,
    REPO_ROOT / "models",
    REPO_ROOT / "models" / "Depth-Anything-3" / "src",
):
    sys.path.insert(0, str(import_root))

from base.config import ProjectPaths  # noqa: E402


VAL_NUM_VIEWS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check train/val prior views, then rebuild val priors with 3 views."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used for val prior generation (default: cuda:0)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only inspect existing train/val priors; do not load models or rebuild val",
    )
    return parser.parse_args()


def read_scans(list_file: Path) -> list[str]:
    if not list_file.is_file():
        raise FileNotFoundError(f"scan list missing: {list_file}")
    return [line.strip() for line in list_file.read_text().splitlines() if line.strip()]


def cached_num_views(cache_path: Path) -> int:
    """A prior stores one source weight per non-reference input view."""
    with np.load(cache_path, allow_pickle=False) as data:
        if "src_weights" not in data:
            raise KeyError("src_weights")
        return 1 + int(np.asarray(data["src_weights"]).size)


def format_view_counts(counts: Counter[int]) -> str:
    if not counts:
        return "missing"
    if len(counts) == 1:
        return str(next(iter(counts)))
    return "mixed(" + ",".join(f"{views}:{count}" for views, count in sorted(counts.items())) + ")"


def inspect_cached_priors(paths: ProjectPaths, split: str, list_file: Path) -> None:
    """Print one existing-cache view-count result per scan in a split."""
    for scan in read_scans(list_file):
        cache_files = sorted((paths.project_path / "log" / "prior_cache" / scan).glob("prior_*.npz"))
        counts: Counter[int] = Counter()
        unreadable = 0
        for cache_path in cache_files:
            try:
                counts[cached_num_views(cache_path)] += 1
            except (KeyError, OSError, ValueError):
                unreadable += 1

        suffix = f" unreadable={unreadable}" if unreadable else ""
        print(f"[{split} cached] {scan}: views={format_view_counts(counts)}{suffix}", flush=True)


def save_prior_atomically(cache_path: Path, prior: dict) -> None:
    """Replace a cache only after the newly computed npz was written fully."""
    from models.pre_prior import save_prior

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{cache_path.stem}.", suffix=".npz", dir=cache_path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)

    try:
        save_prior(temp_path, prior)
        temp_path.replace(cache_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def rebuild_val_priors(paths: ProjectPaths, device_name: str) -> None:
    """Force-recompute all val caches, while reporting only completed scans."""
    import torch

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT/DA3 val prior generation requires an available CUDA device")
    torch.cuda.set_device(device)

    from data.dtu import DTUMVSDataset
    from models.pre_prior import PriorPrecomputer

    # DTUMVSDataset currently prints a global meta count during construction;
    # keep this script's own progress output at the requested one-line-per-scan level.
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = DTUMVSDataset(
            datapath=paths.dtu_train_root,
            listfile=paths.val_list_file,
            nviews=VAL_NUM_VIEWS,
            ndepths=192,
            mode="val",
            resize_scale=0.5,
        )

    indices_by_scan: dict[str, list[int]] = defaultdict(list)
    for idx, (scan, _light_idx, _ref_view, _src_views) in enumerate(dataset.metas):
        indices_by_scan[scan].append(idx)

    precomputer = PriorPrecomputer(device)
    with torch.inference_mode():
        for scan in read_scans(paths.val_list_file):
            view_counts: Counter[int] = Counter()
            for idx in indices_by_scan[scan]:
                precrop_sample = dataset.precrop_inputs(idx)
                prior = precomputer.compute(precrop_sample)
                num_views = 1 + int(np.asarray(prior["src_weights"]).size)
                if num_views != VAL_NUM_VIEWS:
                    raise RuntimeError(
                        f"{scan} produced a {num_views}-view prior; expected {VAL_NUM_VIEWS}"
                    )
                save_prior_atomically(dataset.prior_cache_path_for(idx), prior)
                view_counts[num_views] += 1

            print(f"[val rebuilt] {scan}: views={format_view_counts(view_counts)}", flush=True)


def main() -> int:
    args = parse_args()
    paths = ProjectPaths()

    # Required ordering: inspect all existing train and val caches before
    # importing the heavy prior models or touching any val cache.
    inspect_cached_priors(paths, "train", paths.train_list_file)
    inspect_cached_priors(paths, "val", paths.val_list_file)
    if not args.check_only:
        rebuild_val_priors(paths, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
