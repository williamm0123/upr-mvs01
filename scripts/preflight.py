"""Pre-flight checkout check + fresh-start guard, shared by the umhpc scripts.

A cluster job that turns out to be running a stale checkout wastes a day before
anyone notices, and the failure mode is usually silent (the model trains, just
not the model you meant). Everything asserted here has burned a run at least
once, so each check names what it is protecting against.

``--fresh`` additionally MOVES any pre-existing checkpoint out of log/model/
before training starts. Deleting the file is not the point; making it
unreadable by this run is. The D4->D8 change does not alter the state_dict
(a Conv3d kernel is [out, in, 3, 3, 3] whatever D is), so an old checkpoint
would load cleanly and train the wrong model — there is no load error to rely
on, which is exactly why the guard has to be physical.

The archive is skipped when ``--run-id`` matches the marker already in the
directory: a requeued SLURM job re-runs this script with the same job id and
must resume its OWN checkpoint rather than throw away ten thousand steps.

Exits non-zero with a one-line reason; prints a summary otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def fail(msg: str) -> None:
    sys.exit(f"[preflight] FAIL: {msg}")


def archive_checkpoints(model_dir: Path, run_id: str) -> None:
    """Move existing .pth out of ``model_dir`` unless this run already owns them."""
    marker = model_dir / ".run_id"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == run_id:
        n = len(list(model_dir.glob("*.pth")))
        print(f"[fresh] log/model/ already belongs to run {run_id} "
              f"({n} checkpoint(s)) — leaving it alone so a requeue can resume")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(model_dir.glob("*.pth"))
    if stale:
        dest = model_dir.parent / f"model.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in stale:
            shutil.move(str(f), str(dest / f.name))
        print(f"[fresh] moved {len(stale)} old checkpoint(s) -> {dest}")
        print("[fresh] they are NOT deleted; move them back if this was a mistake")
    else:
        print("[fresh] log/model/ holds no checkpoints — nothing to archive")
    marker.write_text(run_id, encoding="utf-8")
    print(f"[fresh] this run ({run_id}) starts from step 0 with no pre-existing weights")


def main() -> None:
    ap = argparse.ArgumentParser("uprmvs pre-flight")
    ap.add_argument("--fresh", action="store_true",
                    help="archive any existing log/model/*.pth before training")
    ap.add_argument("--run-id", default="manual",
                    help="skip the archive when log/model/.run_id already matches "
                         "(SLURM requeue keeps the same job id)")
    ap.add_argument("--model-dir", default=str(REPO / "log" / "model"))
    args = ap.parse_args()

    checks: list[str] = []

    # DINOv3 LayerScale: when it was nn.Identity the backbone emitted a constant
    # feature map, so SPRE trained against noise and looked merely "weak".
    from models.dinov3.vision_transformer import vit_base
    if not hasattr(vit_base(patch_size=16, n_storage_tokens=4).blocks[0].ls1, "gamma"):
        fail("DINOv3 LayerScale is still nn.Identity — stale checkout, git pull")
    checks.append("dinov3 LayerScale")

    from models.spre import DinoSVA, SVAFusion  # noqa: F401
    checks.append("SVAFusion")

    # Cascade layout. Both halves matter: the strides decide which FPN levels are
    # consumed, and num_depths decides the terminal bin width. num_depths is NOT
    # in the state_dict, so a wrong value here resumes cleanly and trains a
    # different model (train.py's cascade_signature guard catches the resume, but
    # only if this checkout is the one that wrote the checkpoint).
    from base.config import build_mvs_config
    from models.network import UprMVSNet
    cfg = build_mvs_config()
    strides = UprMVSNet.fpn_stage_strides
    cv = cfg.cost_volume
    depths = (cv.num_depths_stage1, cv.num_depths_stage2,
              cv.num_depths_stage3, cv.num_depths_stage4)
    if strides != (8, 4, 2, 1):
        fail(f"cascade strides are {strides}, expected (8, 4, 2, 1)")
    if depths != (48, 16, 8, 8):
        fail(f"cascade num_depths are {depths}, expected (48, 16, 8, 8) — the D4->D8 "
             "terminal-bin change is missing, so this is a pre-P5 checkout")
    checks.append(f"cascade {strides} x {depths}")

    # Per-stage diagnostics: without them a run produces no oracle/selection
    # split and the next decision has nothing to stand on.
    from utils.stage_metrics import stage_diagnostics  # noqa: F401
    checks.append("stage_metrics")

    # Smoke isolation: --smoke used to write latest.pth/best.pth into the shared
    # log/model, destroying a trained checkpoint with two steps of random weights.
    import inspect

    from train import TrainLogger
    if "smoke" not in inspect.signature(TrainLogger.__init__).parameters:
        fail("TrainLogger has no smoke isolation — --smoke would overwrite log/model")
    checks.append("smoke isolation")

    # Offline fusion path (P0/P1). Cheap to check, and it is the thing the whole
    # attribution plan runs on.
    src = (REPO / "test.py").read_text(encoding="utf-8")
    for flag in ("--fuse-only", "--sweep", "--fusion-src-views", "--geo-abs-mm"):
        if flag not in src:
            fail(f"test.py has no {flag} — stale checkout")
    if "def posterior_confidence" not in src:
        fail("test.py still uses the saturating photometric_confidence")
    checks.append("fuse-only + posterior_confidence")

    print("[preflight] OK: " + ", ".join(checks))
    if args.fresh:
        archive_checkpoints(Path(args.model_dir), args.run_id)


if __name__ == "__main__":
    main()
