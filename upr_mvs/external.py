"""Import helpers for the upstream UPR-MVS project utilities."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

from .config import ProjectPaths


@lru_cache(maxsize=4)
def load_transformer_utils(project_root: str | Path | None = None):
    root = Path(project_root) if project_root is not None else ProjectPaths().upr_mvs_root
    root = root.expanduser().resolve()
    utils_path = root / "models/transformer/utils.py"
    if not utils_path.is_file():
        raise FileNotFoundError(f"Cannot find upstream transformer utils: {utils_path}")
    module_name = f"_upr_mvs_transformer_utils_{abs(hash(str(utils_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import upstream transformer utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def homo_warping(*args, project_root: str | Path | None = None, **kwargs):
    return load_transformer_utils(project_root).homo_warping(*args, **kwargs)


def intrinsics_to_projection(*args, project_root: str | Path | None = None, **kwargs):
    return load_transformer_utils(project_root).intrinsics_to_projection(*args, **kwargs)


def sample_depth_planes(*args, project_root: str | Path | None = None, **kwargs):
    return load_transformer_utils(project_root).sample_depth_planes(*args, **kwargs)


def scale_intrinsics(*args, project_root: str | Path | None = None, **kwargs):
    return load_transformer_utils(project_root).scale_intrinsics(*args, **kwargs)
