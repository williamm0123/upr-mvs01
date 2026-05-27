from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera_utils import camera_center_world, view_angle_deg


def parse_dtu_pair_file(pair_path: str | Path) -> list[tuple[int, list[int], list[float]]]:
    pair_path = Path(pair_path)
    pairs: list[tuple[int, list[int], list[float]]] = []
    with pair_path.open("r") as f:
        num_viewpoint = int(f.readline().strip())
        for _ in range(num_viewpoint):
            ref = int(f.readline().strip())
            line = f.readline().strip().split()
            n_src = int(line[0])
            src_ids = [int(line[1 + 2 * i]) for i in range(n_src)]
            src_scores = [float(line[2 + 2 * i]) for i in range(n_src)]
            pairs.append((ref, src_ids, src_scores))
    return pairs


def filter_src_views_by_baseline(
    extrinsics: dict[int, np.ndarray],
    ref_view: int,
    candidate_src: list[int],
    min_deg: float,
    max_deg: float,
    scene_center_world: np.ndarray | None = None,
) -> list[int]:
    kept: list[tuple[int, float]] = []
    e_ref = extrinsics[ref_view]
    for s in candidate_src:
        if s not in extrinsics:
            continue
        ang = view_angle_deg(e_ref, extrinsics[s], scene_center_world)
        if min_deg <= ang <= max_deg:
            kept.append((s, ang))
    kept.sort(key=lambda x: abs(x[1] - 0.5 * (min_deg + max_deg)))
    return [s for s, _ in kept]


def estimate_scene_center(extrinsics_dict: dict[int, np.ndarray]) -> np.ndarray:
    centers = np.stack(
        [camera_center_world(e) for e in extrinsics_dict.values()],
        axis=0,
    )
    return centers.mean(axis=0).astype(np.float32)


def select_src_views(
    extrinsics_dict: dict[int, np.ndarray],
    ref_view: int,
    candidate_src: list[int],
    nviews: int,
    min_deg: float = 5.0,
    max_deg: float = 45.0,
    use_filter: bool = True,
) -> list[int]:
    if not use_filter:
        return candidate_src[: nviews - 1]
    center = estimate_scene_center(extrinsics_dict)
    filtered = filter_src_views_by_baseline(
        extrinsics_dict, ref_view, candidate_src, min_deg, max_deg, center
    )
    if len(filtered) < nviews - 1:
        seen = set(filtered)
        for s in candidate_src:
            if s not in seen and s != ref_view:
                filtered.append(s)
            if len(filtered) >= nviews - 1:
                break
    return filtered[: nviews - 1]
