"""Verify every dataset file the DTU loaders will touch actually exists.

Run once before submitting a training job so a single missing file surfaces
here (seconds, login node) instead of crashing a DataLoader worker hours into
the run:

    UPRMVS_MACHINE=umhpc python scripts/check_dtu_files.py

Checks, for every scan in train.txt + val.txt and every viewpoint in pair.txt,
exactly the paths data/dtu.py builds: rectified images (all 7 light conditions
for train scans, light 3 for val scans), depth_map .pfm, depth_visual .png,
and the per-view camera files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.config import ProjectPaths


def view_ids_from_pair(datapath: Path) -> list[int]:
    lines = (datapath / "Cameras/pair.txt").read_text().split("\n")
    num = int(lines[0])
    return [int(lines[1 + 2 * i]) for i in range(num)]


def main() -> int:
    paths = ProjectPaths()
    datapath = paths.dtu_train_root
    if not datapath.is_dir():
        print(f"dataset root missing: {datapath}")
        return 1

    views = view_ids_from_pair(datapath)
    missing: list[str] = []
    splits = [(paths.train_list_file, range(7)), (paths.val_list_file, [3])]
    for list_file, lights in splits:
        scans = [s for s in list_file.read_text().split() if s]
        for scan in scans:
            for v in views:
                candidates = [
                    datapath / f"Depths_raw/{scan}/depth_map_{v:0>4}.pfm",
                    datapath / f"Depths_raw/{scan}/depth_visual_{v:0>4}.png",
                    datapath / f"Cameras/{v:0>8}_cam.txt",
                ]
                candidates += [
                    datapath / f"Rectified_raw/{scan}/rect_{v + 1:0>3}_{l}_r5000.png"
                    for l in lights
                ]
                missing += [str(p) for p in candidates if not p.exists()]

    if missing:
        print(f"MISSING {len(missing)} file(s):")
        for p in missing:
            print(" ", p)
        return 1
    n_scans = sum(len(lf.read_text().split()) for lf, _ in splits)
    print(f"OK: all files present for {n_scans} scans x {len(views)} views")
    return 0


if __name__ == "__main__":
    sys.exit(main())
