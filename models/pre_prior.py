"""Offline prior precomputation & caching.

For each training sample this computes, **once**, the geometry priors the
network consumes and stores them to disk:

    {depth_prior, conf_prior, norm_depth_fill, src_weights}

Pipeline per sample (all at the *pre-crop* resolution so random-crop stays
valid -- the cached full-frame maps are simply sliced alongside the image at
load time):

    sfm.generate_sparse_depth_from_sample  -> sparse_depth (metric) + source_weights
        └─ sparse_depth fed to norm_fill for metric scale calibration
    norm_fill.generate_priors_from_sample  -> depth_filled / conf_map / normals

Heavy models (VGGT + DA3) are loaded **once** by ``PriorPrecomputer`` and
reused across all samples. This module's top level stays import-light (numpy
only) so ``data.dtu`` can import :func:`load_prior` without pulling in the VGGT
stack; the heavy deps are imported lazily inside the compute path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from base.config import ProjectPaths
import models.norm_fill as norm_fill
# Keys stored in every prior cache file (and expected by the network/loss).
PRIOR_KEYS = ("depth_prior", "conf_prior", "norm_depth_fill", "src_weights")


# --------------------------------------------------------------------------- #
# Cache IO (import-light: numpy only)
# --------------------------------------------------------------------------- #
def save_prior(path: str | Path, prior: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(prior[k], dtype=np.float32) for k in PRIOR_KEYS})


def load_prior(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"prior cache missing: {path}. Run the prior precompute first "
            f"(train.py does this automatically unless --build-priors skip)."
        )
    with np.load(path) as data:
        return {k: data[k] for k in PRIOR_KEYS}


# --------------------------------------------------------------------------- #
# Per-sample computation (heavy deps imported lazily)
# --------------------------------------------------------------------------- #
class PriorPrecomputer:
    """Loads VGGT + DA3 once, computes priors for a pre-crop multi-view sample."""

    def __init__(
        self,
        device,
        image_mode: str = "resize",
        conf_percentile: float = 10.0,
        image_target_wh: tuple[int, int] = (518, 420),
    ) -> None:


        self._nf = norm_fill
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        paths = ProjectPaths()
        self.vggt_model = norm_fill.load_vggt_model(paths.vggt_weights_path, self.device)
        self.da3_model = norm_fill.load_da3_model(paths.da3_weights_file, self.device)
        self.image_mode = image_mode
        self.conf_percentile = conf_percentile
        self.image_target_wh = image_target_wh

    def compute(self, precrop_sample: dict) -> dict:
        """precrop_sample needs ``images`` [V,C,H,W], ``intrinsics`` [V,3,3],
        ``extrinsics`` [V,4,4] at the pre-crop resolution."""
        import models.sfm as sfm

        # 1) SfM: metric sparse depth (for scale) + per-source weights.
        sfm_out = sfm.generate_sparse_depth_from_sample(precrop_sample, ref_idx=0)

        sample = dict(precrop_sample)
        sample["sfm_depth"] = sfm_out["sparse_depth"]  # reused by norm_fill for scale

        # 2) Dense fill + confidence + normals (models reused, not reloaded).
        priors = self._nf.generate_priors_from_sample(
            sample,
            self.device,
            image_mode=self.image_mode,
            conf_percentile=self.conf_percentile,
            image_target_wh=self.image_target_wh,
            vggt_model=self.vggt_model,
            da3_model=self.da3_model,
        )
        return {
            "depth_prior": np.asarray(priors["depth_filled"], np.float32),
            "conf_prior": np.asarray(priors["conf_map"], np.float32),
            "norm_depth_fill": np.asarray(priors["normal"], np.float32),
            "src_weights": np.asarray(sfm_out["source_weights"], np.float32),
        }


def load_or_compute(path: str | Path, precrop_sample: dict, precomputer: PriorPrecomputer,
                    overwrite: bool = False) -> dict:
    path = Path(path)
    if path.exists() and not overwrite:
        return load_prior(path)
    prior = precomputer.compute(precrop_sample)
    save_prior(path, prior)
    return prior


def build_prior_cache(dataset, device, overwrite: bool = False, verbose: bool = True) -> int:
    """Populate the prior cache for every meta in ``dataset`` (run once, main process).

    ``dataset`` must expose ``precrop_inputs(idx)`` (pre-crop multi-view sample)
    and ``prior_cache_path_for(idx)``.
    """
    n = len(dataset)
    # skip loading the heavy models entirely if everything is already cached
    pending = [i for i in range(n) if overwrite or not Path(dataset.prior_cache_path_for(i)).exists()]
    if not pending:
        if verbose:
            print(f"[pre_prior] cache already complete: {n} priors")
        return 0

    if verbose:
        print(f"[pre_prior] building {len(pending)}/{n} priors (loading VGGT + DA3 once) ...")
    precomputer = PriorPrecomputer(device)
    built = 0
    for idx in pending:
        pc = dataset.precrop_inputs(idx)
        prior = precomputer.compute(pc)
        save_prior(dataset.prior_cache_path_for(idx), prior)
        built += 1
        if verbose and built % 20 == 0:
            print(f"[pre_prior]   {built}/{len(pending)} done")
    if verbose:
        print(f"[pre_prior] cache ready: {built} newly built, {n} total")
    return built
