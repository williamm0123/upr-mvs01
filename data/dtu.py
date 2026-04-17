"""Dataset wrappers used by the notebook experiments."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .io import read_camera_file, read_pfm
from upr_mvs.config import DTUConfig, ProjectPaths


def image_scan_dir_name(scan_name: str, split: str, image_dir: str) -> str:
    if image_dir == "Rectified_raw":
        return scan_name
    return f"{scan_name}_train" if split == "train" else scan_name


def depth_scan_dir_name(scan_name: str, split: str, depth_dir: str) -> str:
    if "raw" in depth_dir.lower():
        return scan_name
    return f"{scan_name}_train" if split == "train" else scan_name


def expected_image_path(root_dir: str | Path, scan_name: str, split: str, image_dir: str, light_id: int, view_id: int) -> Path:
    image_dir_name = image_scan_dir_name(scan_name, split, image_dir)
    image_filename = f"rect_{view_id + 1:03d}_{light_id}_r5000.png"
    return Path(root_dir) / image_dir / image_dir_name / image_filename


def expected_camera_path(root_dir: str | Path, view_id: int) -> Path:
    return Path(root_dir) / "Cameras/train" / f"{view_id:08d}_cam.txt"


def expected_depth_path(root_dir: str | Path, scan_name: str, split: str, depth_dir: str, ref_view: int) -> Path:
    depth_dir_name = depth_scan_dir_name(scan_name, split, depth_dir)
    return Path(root_dir) / depth_dir / depth_dir_name / f"depth_map_{ref_view:04d}.pfm"


class DTUDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        list_file: str | Path,
        n_views: int = 3,
        light_id: int = 3,
        split: str = "train",
        image_dir: str = "Rectified_raw",
        depth_dir: str = "Depths_raw",
        n_depths: int = 192,
    ):
        self.root_dir = Path(root_dir)
        self.n_depths = n_depths
        self.n_views = n_views
        self.light_id = light_id
        self.split = split
        self.image_dir = image_dir
        self.depth_dir = depth_dir

        with open(list_file, "r") as file:
            self.scan_names = [line.strip() for line in file.readlines() if line.strip()]

        self.pair_path = self.root_dir / "Cameras/train/pair.txt"
        self.view_pairs = self._read_pair_file(self.pair_path)

        self.samples: list[tuple[str, int, list[int]]] = []
        for scan_name in self.scan_names:
            for ref_view, src_views in self.view_pairs.items():
                if len(src_views) >= self.n_views - 1:
                    self.samples.append((scan_name, ref_view, src_views[: self.n_views - 1]))

    def _read_pair_file(self, path: str | Path) -> dict[int, list[int]]:
        view_pairs: dict[int, list[int]] = {}
        with open(path, "r") as file:
            num_viewpoints = int(file.readline())
            for _ in range(num_viewpoints):
                ref_view = int(file.readline().rstrip())
                src_info = file.readline().rstrip().split()[1::2]
                view_pairs[ref_view] = [int(x) for x in src_info]
        return view_pairs

    def _image_scan_dir_name(self, scan_name: str) -> str:
        return image_scan_dir_name(scan_name, self.split, self.image_dir)

    def _depth_scan_dir_name(self, scan_name: str) -> str:
        return depth_scan_dir_name(scan_name, self.split, self.depth_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        scan_name, ref_view, src_views = self.samples[idx]
        src_views = list(src_views[: self.n_views - 1])
        view_ids = [ref_view] + src_views
        image_dir_name = self._image_scan_dir_name(scan_name)
        depth_dir_name = self._depth_scan_dir_name(scan_name)

        images_np: list[np.ndarray] = []
        intrinsics_list: list[np.ndarray] = []
        extrinsics_list: list[np.ndarray] = []
        projection_matrices: list[np.ndarray] = []
        ref_depth_min: float | None = None
        ref_depth_interval: float | None = None

        for slot, view_id in enumerate(view_ids):
            image_filename = f"rect_{view_id + 1:03d}_{self.light_id}_r5000.png"
            image_path = self.root_dir / self.image_dir / image_dir_name / image_filename
            if not image_path.exists():
                raise FileNotFoundError(f"Cannot read image: {image_path}")

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"cv2 failed to read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.ascontiguousarray(image)

            camera_path = self.root_dir / "Cameras/train" / f"{view_id:08d}_cam.txt"
            if not camera_path.exists():
                raise FileNotFoundError(f"Cannot read camera file: {camera_path}")

            intrinsics, extrinsics, depth_min, depth_interval = read_camera_file(str(camera_path))
            intrinsics = intrinsics.astype(np.float32)
            extrinsics = extrinsics.astype(np.float32)

            projection_matrix = np.eye(4, dtype=np.float32)
            projection_matrix[:3, :4] = intrinsics @ extrinsics[:3, :4]

            images_np.append(image)
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)
            projection_matrices.append(projection_matrix)

            if slot == 0:
                ref_depth_min = float(depth_min)
                ref_depth_interval = float(depth_interval)

        depth_path = self.root_dir / self.depth_dir / depth_dir_name / f"depth_map_{ref_view:04d}.pfm"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth loading failed for scan {scan_name}. Check path: {depth_path}")

        depth_map = read_pfm(str(depth_path)).astype(np.float32)
        if depth_map.ndim == 3:
            depth_map = depth_map[..., 0]
        depth_map = np.ascontiguousarray(depth_map)

        if ref_depth_min is None or ref_depth_interval is None:
            raise RuntimeError("ref_depth_min / ref_depth_interval is None")

        depth_values = ref_depth_min + np.arange(self.n_depths, dtype=np.float32) * ref_depth_interval
        valid_depth_mask = np.isfinite(depth_map) & (depth_map > 0)

        imgs = torch.from_numpy(np.stack(images_np, axis=0)).permute(0, 3, 1, 2).float()
        intrinsics = torch.from_numpy(np.stack(intrinsics_list, axis=0)).float()
        extrinsics = torch.from_numpy(np.stack(extrinsics_list, axis=0)).float()
        projection_matrices_t = torch.from_numpy(np.stack(projection_matrices, axis=0)).float()

        return {
            "imgs": imgs,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "depth_gt": torch.from_numpy(depth_map).unsqueeze(0).float(),
            "mask": torch.from_numpy(valid_depth_mask.astype(np.float32)).unsqueeze(0),
            "depth_range": torch.tensor([ref_depth_min, float(depth_values[-1])], dtype=torch.float32),
            "depth_values": torch.from_numpy(depth_values),
            "projection_matrices": projection_matrices_t,
            "view_ids": torch.tensor(view_ids, dtype=torch.long),
            "ref_view": torch.tensor(ref_view, dtype=torch.long),
            "src_views": torch.tensor(src_views, dtype=torch.long),
            "light_id": torch.tensor(self.light_id, dtype=torch.long),
            "has_depth_gt": torch.tensor(True, dtype=torch.bool),
            "scan_name": scan_name,
            "sample_name": f"{scan_name}_view{ref_view:03d}_light{self.light_id}",
        }


def build_dtu_dataset(paths: ProjectPaths | None = None, config: DTUConfig | None = None) -> DTUDataset:
    paths = paths or ProjectPaths()
    config = config or DTUConfig()
    return DTUDataset(
        root_dir=paths.dtu_train_root,
        list_file=paths.dtu_list_path,
        n_views=config.n_views,
        light_id=config.light_id,
        split=config.split,
        image_dir=config.image_dir,
        depth_dir=config.depth_dir,
        n_depths=config.n_depths,
    )
