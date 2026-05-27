from __future__ import annotations

from datetime import datetime
from pathlib import Path

from base.config import ProjectPaths


def resolve_project_path(*parts: str | Path) -> Path:
    root = ProjectPaths().project_path
    return root.joinpath(*parts)


def make_run_dir(name: str, root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else ProjectPaths().output_root / "runs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{stamp}_{name}"
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (run_dir / "vis").mkdir(parents=True, exist_ok=True)
    (run_dir / "log").mkdir(parents=True, exist_ok=True)
    return run_dir
