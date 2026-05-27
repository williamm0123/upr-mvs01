from __future__ import annotations
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from .io import read_pfm
from .camera_utils import (
    build_projection_matrix,
    resize_and_crop_depth,
    resize_and_crop_image,
    resize_and_crop_mask,
)
from .pair_utils import select_src_views
from .transforms import make_multiscale_depth, make_multiscale_mask, normalize_image, to_chw

class DTUDataset(Dataset):
    def __init__(
        self, datapath, listfile, nviews = 3, ndepths = 192, mode= "train",
        **kwargs):
        
        self.datapath = Path(datapath)
        self.listfile = listfile
        self.ndepths = ndepths
        self.nviews = nviews
        self.mode = mode
        self.kwargs = kwargs
        self.metas = self.build_list()
        self.resize_scale = kwargs.get('resize_scale', 1)
        # if mode != 'train':
        #     self.random_crop = False
        #     self.augment = False
        # else:
        #     self.random_crop = random_crop
        #     self.augment = augment
        
    def __iter__(self) -> Iterator:
        """
        显式实现迭代器协议，解决 Pylance 无法识别 enumerate 的报错
        """
        for i in range(len(self)):
            yield self[i]
    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    if self.mode == "train":
                        for light_idx in range(7):
                            metas.append((scan, light_idx, ref_view, src_views))
                    else:
                        metas.append((scan, 3, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas
    def read_camera_file(self,filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
        intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * 1.06

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self,filename):
        img=Image.open(filename).convert('RGB')
        return img
    def __len__(self) -> int:
        return len(self.metas)

    def read_mask(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        # np_img = self.prepare_img(np_img)
        return np_img

    # def generate_stage_depth(self, depth):
    #     h, w = depth.shape
    #     depth_ms = {
    #         "stage1": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
    #         "stage2": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
    #         "stage3": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
    #         "stage4": depth
    #     }
    #     return depth_ms
    
    # def center_crop_img(self, img, new_h=None, new_w=None):
    #     h, w = img.shape[:2]

    #     if new_h != h or new_w != w:
    #         start_h = (h - new_h) // 2
    #         start_w = (w - new_w) // 2
    #         finish_h = start_h + new_h
    #         finish_w = start_w + new_w
    #         img = img[start_h:finish_h, start_w:finish_w]
    #     return img
    
    # def center_crop_cam(self, intrinsics, h, w, new_h=None, new_w=None):
    #     if new_h != h or new_w != w:
    #         start_h = (h - new_h) // 2
    #         start_w = (w - new_w) // 2
    #         new_intrinsics = intrinsics.copy()
    #         new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
    #         new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
    #         return new_intrinsics
    #     else:
    #         return intrinsics
    
    def pre_resize(self, img, depth, intrinsic, mask, resize_scale):
        ori_h, ori_w, _ = img.shape
        img = cv2.resize(img, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_AREA)
        h, w, _ = img.shape

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, :] *= resize_scale
        output_intrinsics[1, :] *= resize_scale

        if depth is not None:
            depth = cv2.resize(depth, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

        if mask is not None:
            mask = cv2.resize(mask, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

        return img, depth, output_intrinsics, mask
    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        view_ids = [ref_view] + src_views[:(self.nviews - 1)]
        # view_ids = [ref_view] + src_views
        
        images = []
        image_paths = []
        intrinsics = []
        extrinsics = []
        projection_matrices = []
        depth_gt = None
        depth_map_hr = None
        depth_mask_hr = None
        depth_values = []
        
        
        resize_scale = self.resize_scale if self.resize_scale != 1.0 else 1.0
            
        for i, view_id in enumerate(view_ids):
            # print(f"正在处理第 {i} 个视图，视图 ID 为: {view_id}")
            img_filename = os.path.join(self.datapath, 'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'.format(scan, view_id + 1, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, view_id))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, view_id))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/{:0>8}_cam.txt').format(view_id)
            
            image_paths.append(img_filename)
            
            img = np.asarray(self.read_img(img_filename))
            intrinsic, extrinsic, depth_min, depth_interval = self.read_camera_file(proj_mat_filename)
            
            
            if i == 0:
                depth_map_hr = np.asarray(read_pfm(depth_filename_hr), dtype=np.float32)
                depth_mask_hr = self.read_mask(mask_filename_hr)
                # 只需要在ref视角计算depth interval
                depth_max= depth_min + depth_interval * self.ndepths
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
                depth_gt = depth_map_hr
                
            else:
                depth_map_hr = None
                depth_mask_hr = None

            if resize_scale != 1.0:
                img, depth_map_hr, intrinsic, depth_mask_hr = self.pre_resize(img, depth_map_hr, intrinsic, depth_mask_hr, resize_scale)
            
            projection_matrix = np.eye(4, dtype=np.float32)
            projection_matrix[:3, :4] = intrinsic @ extrinsic[:3, :4]
            
            images.append(img)
            intrinsics.append(np.asarray(intrinsic))
            extrinsics.append(np.asarray(extrinsic))
            projection_matrices.append(projection_matrix)
     

        # 2. 将列表转换为 NumPy 数组 (Stacking)
        # np.stack 会创建一个新维度，将 [H, W, 3] 的图像堆叠成 [V, H, W, 3]
        imgs_np = np.stack(images, axis=0) 
        # 如果你依然需要 PyTorch 训练，通常会在这里进行 permute(0, 3, 1, 2)
        # 但既然你要求返回 np 数组，我们直接 stack 即可
        intrinsics_np = np.stack(intrinsics, axis=0)      # 形状: [V, 3, 3]
        extrinsics_np = np.stack(extrinsics, axis=0)      # 形状: [V, 4, 4]
        proj_matrices_np = np.stack(projection_matrices, axis=0) # 形状: [V, 4, 4]
        
        
        
        # imgs = torch.from_numpy(np.stack(images_np, axis=0)).permute(0, 3, 1, 2).float()
        # intrinsics = torch.from_numpy(np.stack(intrinsics_list, axis=0)).float()
        # extrinsics = torch.from_numpy(np.stack(extrinsics_list, axis=0)).float()
        # projection_matrices_t = torch.from_numpy(np.stack(projection_matrices, axis=0)).float()

        return {
            "images": imgs_np,
            "intrinsics": intrinsics_np,
            "extrinsics": extrinsics_np,
            "depth_gt": depth_gt,
            "mask": depth_mask_hr,
            "depth_values": depth_values,
            "projection_matrices": proj_matrices_np,
            "image_paths":image_paths,
        }


def _read_dtu_cam_file(path: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    with open(path) as f:
        lines = [line.rstrip() for line in f.readlines()]
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1]) * 1.06
    return intrinsics, extrinsics, depth_min, depth_interval


def _read_mask(path: str) -> np.ndarray:
    img = np.array(Image.open(path), dtype=np.float32)
    return (img > 10).astype(np.float32)


def _load_depth_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if suffix == ".npz":
        payload = np.load(path)
        for key in ("depth", "filled_depth", "depth_filled", "arr_0"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)
        first_key = list(payload.keys())[0]
        return np.asarray(payload[first_key], dtype=np.float32)
    if suffix == ".pfm":
        return np.asarray(read_pfm(str(path)), dtype=np.float32)
    raise ValueError(f"Unsupported prior depth format: {path}")


def _resize_prior_depth(
    depth: np.ndarray,
    original_hw: tuple[int, int],
    target_h: int,
    target_w: int,
    resize_info: dict,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    if depth.shape == (target_h, target_w):
        return depth.astype(np.float32)
    if depth.shape == original_hw:
        return resize_and_crop_depth(depth, target_h, target_w, resize_info).astype(np.float32)

    valid_ratio = float((np.isfinite(depth) & (depth > 0)).mean()) if depth.size else 0.0
    interp = cv2.INTER_NEAREST if valid_ratio < 0.5 else cv2.INTER_LINEAR
    safe = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    src_h, src_w = safe.shape[:2]
    original_aspect = float(original_hw[1]) / float(original_hw[0])
    prior_aspect = float(src_w) / float(src_h)
    if abs(prior_aspect - original_aspect) / max(original_aspect, 1e-6) < 0.02:
        scale = max(target_h / src_h, target_w / src_w)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))
        resized = cv2.resize(safe, (new_w, new_h), interpolation=interp)
        crop_x = (new_w - target_w) // 2
        crop_y = (new_h - target_h) // 2
        return resized[crop_y : crop_y + target_h, crop_x : crop_x + target_w].astype(np.float32)
    return cv2.resize(safe, (target_w, target_h), interpolation=interp).astype(np.float32)


class DTUMVSDataset(Dataset):
    """MVS-friendly DTU dataset returning fixed-size tensors with multi-scale GT.

    Output keys:
        imgs:                [V, 3, H, W] float32 normalized
        imgs_raw:            [V, 3, H, W] float32 in [0, 1] (for VGGT / DA3 inputs)
        intrinsics:          [V, 3, 3] float32 (640x512 frame)
        extrinsics:          [V, 4, 4] float32 (world -> cam)
        proj_matrices:       [V, 4, 4] float32 (K @ E)
        depth_gt_full:       [H, W] float32
        depth_gt_multiscale: dict[int stride -> [Hs, Ws] tensor]
        mask_full:           [H, W] float32
        mask_multiscale:     dict[int stride -> [Hs, Ws] tensor]
        depth_min:           float32 scalar
        depth_interval:      float32 scalar
        depth_max:           float32 scalar
        ndepths:             int
        scan / ref_view / src_views / image_paths
    """

    def __init__(
        self,
        datapath: str | Path,
        listfile: str | Path,
        nviews: int = 3,
        ndepths: int = 192,
        target_h: int = 512,
        target_w: int = 640,
        feature_strides: tuple[int, ...] = (4, 8, 16),
        mode: str = "train",
        use_pair_filter: bool = True,
        pair_min_baseline_deg: float = 5.0,
        pair_max_baseline_deg: float = 45.0,
        prior_root: str | Path | None = None,
        prior_confidence: float = 0.9,
        require_prior: bool = False,
    ) -> None:
        self.datapath = Path(datapath)
        self.listfile = Path(listfile)
        self.nviews = nviews
        self.ndepths = ndepths
        self.target_h = target_h
        self.target_w = target_w
        self.feature_strides = feature_strides
        self.mode = mode
        self.use_pair_filter = use_pair_filter
        self.pair_min_baseline_deg = pair_min_baseline_deg
        self.pair_max_baseline_deg = pair_max_baseline_deg
        self.prior_root = Path(prior_root) if prior_root is not None else None
        self.prior_confidence = float(prior_confidence)
        self.require_prior = require_prior
        self._extrinsics_cache: dict[int, np.ndarray] = {}
        self.metas = self._build_metas()

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]

    def _load_extrinsics_cache(self) -> dict[int, np.ndarray]:
        if self._extrinsics_cache:
            return self._extrinsics_cache
        cam_dir = self.datapath / "Cameras"
        for cam_file in sorted(cam_dir.glob("????????_cam.txt")):
            view_id = int(cam_file.stem.split("_")[0])
            _, extrinsic, _, _ = _read_dtu_cam_file(str(cam_file))
            self._extrinsics_cache[view_id] = extrinsic
        return self._extrinsics_cache

    def _build_metas(self) -> list[tuple[str, int, int, list[int]]]:
        metas: list[tuple[str, int, int, list[int]]] = []
        with self.listfile.open() as f:
            scans = [line.strip() for line in f if line.strip()]

        extr_cache = self._load_extrinsics_cache()
        pair_file = self.datapath / "Cameras" / "pair.txt"
        with pair_file.open() as f:
            num_viewpoint = int(f.readline())
            entries: list[tuple[int, list[int]]] = []
            for _ in range(num_viewpoint):
                ref = int(f.readline().strip())
                line = f.readline().strip().split()
                n_src = int(line[0])
                src_ids = [int(line[1 + 2 * i]) for i in range(n_src)]
                entries.append((ref, src_ids))

        for scan in scans:
            for ref, src_ids in entries:
                src_selected = select_src_views(
                    extr_cache,
                    ref,
                    src_ids,
                    self.nviews,
                    self.pair_min_baseline_deg,
                    self.pair_max_baseline_deg,
                    self.use_pair_filter,
                )
                if len(src_selected) < self.nviews - 1:
                    continue
                if self.mode == "train":
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref, src_selected))
                else:
                    metas.append((scan, 3, ref, src_selected))
        print(f"[DTUMVSDataset] mode={self.mode} metas={len(metas)}")
        return metas

    def __len__(self) -> int:
        return len(self.metas)

    def _load_view(
        self,
        scan: str,
        view_id: int,
        light_idx: int,
        load_gt: bool,
    ) -> dict:
        img_path = self.datapath / "Rectified_raw" / scan / f"rect_{view_id + 1:03d}_{light_idx}_r5000.png"
        cam_path = self.datapath / "Cameras" / f"{view_id:08d}_cam.txt"

        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img_pil)
        K, E, depth_min, depth_interval = _read_dtu_cam_file(str(cam_path))

        img_resized, K_resized, info = resize_and_crop_image(img_np, K, self.target_h, self.target_w)

        depth_gt = None
        mask = None
        if load_gt:
            depth_path = self.datapath / "Depths_raw" / scan / f"depth_map_{view_id:04d}.pfm"
            mask_path = self.datapath / "Depths_raw" / scan / f"depth_visual_{view_id:04d}.png"
            if depth_path.is_file():
                depth_full = np.asarray(read_pfm(str(depth_path)), dtype=np.float32)
                if depth_full.shape != img_np.shape[:2]:
                    depth_full = cv2.resize(
                        depth_full,
                        (img_np.shape[1], img_np.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                depth_gt = resize_and_crop_depth(depth_full, self.target_h, self.target_w, info)
            if mask_path.is_file():
                mask_full = _read_mask(str(mask_path))
                if mask_full.shape != img_np.shape[:2]:
                    mask_full = cv2.resize(
                        mask_full,
                        (img_np.shape[1], img_np.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask = resize_and_crop_mask(mask_full, self.target_h, self.target_w, info)

        return {
            "image": img_resized,
            "K": K_resized,
            "E": E.astype(np.float32),
            "depth_min": depth_min,
            "depth_interval": depth_interval,
            "depth_gt": depth_gt,
            "mask": mask,
            "image_path": str(img_path),
            "resize_info": info,
            "original_hw": img_np.shape[:2],
        }

    def _prior_candidates(
        self,
        scan: str,
        light_idx: int,
        ref_view: int,
        view_id: int,
    ) -> list[Path]:
        if self.prior_root is None:
            return []
        root = self.prior_root
        light_candidates = [light_idx] if light_idx == 3 else [light_idx, 3]
        out: list[Path] = []
        for light in light_candidates:
            rect = f"rect_{view_id + 1:03d}_{light}_r5000"
            scene_dirs = [
                root / f"{scan}_light{light}_ref{ref_view:03d}",
                root / f"{scan}_ref{ref_view:03d}_light{light}",
                root / scan,
                root,
            ]
            exact_names = [
                f"{rect}_da3_loggrad_filled_depth.npy",
                f"{rect}_da3_local_affine_filled_depth.npy",
                f"{rect}_normal_constraint_filled_depth.npy",
                f"{rect}_filled_depth.npy",
                f"{rect}_depth.npy",
                f"{rect}.npy",
                f"{rect}.npz",
                f"{rect}.pfm",
            ]
            for scene_dir in scene_dirs:
                for subdir in (scene_dir / "npy", scene_dir / "depths", scene_dir):
                    for name in exact_names:
                        out.append(subdir / name)
                    out.extend(sorted(subdir.glob(f"{rect}*filled_depth.npy")))
                    out.extend(sorted(subdir.glob(f"{rect}*filled_depth.npz")))
                    out.extend(sorted(subdir.glob(f"{rect}*filled_depth.pfm")))
        return out

    def _load_prior_view(
        self,
        scan: str,
        light_idx: int,
        ref_view: int,
        view_id: int,
        view: dict,
    ) -> np.ndarray | None:
        for path in self._prior_candidates(scan, light_idx, ref_view, view_id):
            if path.is_file():
                depth = _load_depth_array(path)
                return _resize_prior_depth(
                    depth,
                    original_hw=view["original_hw"],
                    target_h=self.target_h,
                    target_w=self.target_w,
                    resize_info=view["resize_info"],
                )
        if self.require_prior:
            rect = f"rect_{view_id + 1:03d}_{light_idx}_r5000"
            raise FileNotFoundError(
                f"Missing offline prior for {scan} ref={ref_view} view={view_id}: {rect} under {self.prior_root}"
            )
        return None

    def __getitem__(self, idx: int) -> dict:
        scan, light_idx, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + list(src_views[: self.nviews - 1])

        imgs_norm: list[np.ndarray] = []
        imgs_raw: list[np.ndarray] = []
        intrinsics: list[np.ndarray] = []
        extrinsics: list[np.ndarray] = []
        proj_matrices: list[np.ndarray] = []
        image_paths: list[str] = []
        prior_depths: list[np.ndarray] = []
        prior_confs: list[np.ndarray] = []
        prior_valids: list[np.ndarray] = []

        depth_gt_full: np.ndarray | None = None
        mask_full: np.ndarray | None = None
        depth_min = 0.0
        depth_interval = 0.0

        for i, view_id in enumerate(view_ids):
            view = self._load_view(scan, view_id, light_idx, load_gt=(i == 0))
            img_uint = view["image"]
            imgs_raw.append((img_uint.astype(np.float32) / 255.0).transpose(2, 0, 1))
            imgs_norm.append(to_chw(normalize_image(img_uint)))
            intrinsics.append(view["K"].astype(np.float32))
            extrinsics.append(view["E"])
            proj_matrices.append(build_projection_matrix(view["K"], view["E"]))
            image_paths.append(view["image_path"])
            if i == 0:
                depth_gt_full = view["depth_gt"]
                mask_full = view["mask"]
                depth_min = float(view["depth_min"])
                depth_interval = float(view["depth_interval"])

            if self.prior_root is not None:
                prior_depth = self._load_prior_view(scan, light_idx, ref_view, view_id, view)
                if prior_depth is None:
                    prior_depth = np.zeros((self.target_h, self.target_w), dtype=np.float32)
                    prior_valid = np.zeros((self.target_h, self.target_w), dtype=bool)
                else:
                    prior_depth = prior_depth.astype(np.float32)
                    prior_valid = np.isfinite(prior_depth) & (prior_depth > 0)
                    prior_depth = np.where(prior_valid, prior_depth, 0.0).astype(np.float32)
                prior_conf = np.where(prior_valid, self.prior_confidence, 0.0).astype(np.float32)
                prior_depths.append(prior_depth)
                prior_confs.append(prior_conf)
                prior_valids.append(prior_valid.astype(np.float32))

        imgs_norm_np = np.stack(imgs_norm, axis=0).astype(np.float32)
        imgs_raw_np = np.stack(imgs_raw, axis=0).astype(np.float32)
        intrinsics_np = np.stack(intrinsics, axis=0)
        extrinsics_np = np.stack(extrinsics, axis=0)
        proj_np = np.stack(proj_matrices, axis=0).astype(np.float32)

        depth_max = depth_min + depth_interval * self.ndepths
        depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)[: self.ndepths]

        if depth_gt_full is None:
            depth_gt_full = np.zeros((self.target_h, self.target_w), dtype=np.float32)
        if mask_full is None:
            mask_full = np.zeros((self.target_h, self.target_w), dtype=np.float32)

        depth_ms = make_multiscale_depth(depth_gt_full, self.feature_strides)
        mask_ms = make_multiscale_mask(mask_full, self.feature_strides)

        sample = {
            "imgs": torch.from_numpy(imgs_norm_np),
            "imgs_raw": torch.from_numpy(imgs_raw_np),
            "intrinsics": torch.from_numpy(intrinsics_np),
            "extrinsics": torch.from_numpy(extrinsics_np),
            "proj_matrices": torch.from_numpy(proj_np),
            "depth_gt_full": torch.from_numpy(depth_gt_full),
            "depth_gt_multiscale": {s: torch.from_numpy(d) for s, d in depth_ms.items()},
            "mask_full": torch.from_numpy(mask_full),
            "mask_multiscale": {s: torch.from_numpy(m) for s, m in mask_ms.items()},
            "depth_min": torch.tensor(depth_min, dtype=torch.float32),
            "depth_interval": torch.tensor(depth_interval, dtype=torch.float32),
            "depth_max": torch.tensor(depth_max, dtype=torch.float32),
            "depth_values": torch.from_numpy(depth_values),
            "ndepths": self.ndepths,
            "scan": scan,
            "ref_view": ref_view,
            "src_views": list(src_views),
            "image_paths": image_paths,
        }
        if self.prior_root is not None:
            sample["prior"] = {
                "depth_sparse": torch.from_numpy(np.stack(prior_depths, axis=0).astype(np.float32)),
                "confidence": torch.from_numpy(np.stack(prior_confs, axis=0).astype(np.float32)),
                "valid_mask": torch.from_numpy(np.stack(prior_valids, axis=0).astype(np.float32)),
            }
        return sample
