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
import models.sfm as sfm
from models import pre_prior
from base.config import ProjectPaths

class DTUMVSDataset(Dataset):
    def __init__(
        self, datapath, listfile, nviews = 3, ndepths = 192, mode= "train",random_crop = False,
        resize_scale = 1.0,
        **kwargs):
        
        self.datapath = Path(datapath)
        self.listfile = listfile
        self.ndepths = ndepths
        self.nviews = nviews
        self.mode = mode

        self.metas = self.build_list()
        self.resize_scale = kwargs.get('resize_scale', 0.5)
        self.height = kwargs.get('height', 512)
        self.width  = kwargs.get('width', 640)
        self.sfm_cache_dir = Path(ProjectPaths().sfm_cache_path)
        self.prior_cache_dir = Path(ProjectPaths().project_path) / "log" / "prior_cache"
        # 训练默认随机裁剪做增广, 其余模式居中裁剪保证可复现; 可用 kwarg 覆盖
        self.random_crop = kwargs.get('random_crop', mode == 'train')
        self.kwargs = kwargs
        # self.center_crop_size = kwargs.get('center_crop_size', None)
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

    def pick_crop_origin(self, h, w):
        """选裁剪左上角 (x0, y0): 训练随机, 其余居中。同一 sample 内只调用一次,
        让 ref/src/depth/sfm_depth/mask 共用同一窗口。"""
        # assert h >= self.height and w >= self.width, \
        #     f"裁剪 {self.height}x{self.width} 超出图像 {h}x{w}，请调大 resize_scale"
        max_y, max_x = h - self.height, w - self.width
        if self.random_crop:
            return int(np.random.randint(0, max_x + 1)), int(np.random.randint(0, max_y + 1))
        return max_x // 2, max_y // 2

    def crop_at(self, img, intrinsic, x0, y0, depth=None, mask=None):
        """在给定左上角 (x0, y0) 处裁剪到 (self.width, self.height), 并平移主点。"""
        height, width = self.height, self.width
        img = img[y0:y0 + height, x0:x0 + width]
        K = intrinsic.copy().astype(np.float32)
        K[0, 2] -= x0   # cx：裁剪只平移主点
        K[1, 2] -= y0   # cy：焦距 fx,fy 不变

        if depth is not None:
            depth = depth[y0:y0 + height, x0:x0 + width]
        if mask is not None:
            mask = mask[y0:y0 + height, x0:x0 + width]

        return img, K, depth, mask

    def prior_cache_path_for(self, idx):
        """Prior cache file for a meta (mirrors the SfM cache naming)."""
        scan, light_idx, ref_view, _ = self.metas[idx]
        return self.prior_cache_dir / scan / f"prior_{ref_view:0>4}_{light_idx}.npz"

    def precrop_inputs(self, idx):
        """Pre-crop multi-view inputs (before random crop), shared by both
        __getitem__ and the offline prior precompute so the two stay aligned."""
        scan, light_idx, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:(self.nviews - 1)]
        resize_scale = self.resize_scale

        resized_imgs, resized_intrinsics, extrinsics = [], [], []
        depth_hr = mask_hr = None
        depth_values = []
        for i, view_id in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, 'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'.format(scan, view_id + 1, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, view_id))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, view_id))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/{:0>8}_cam.txt').format(view_id)

            img = np.asarray(self.read_img(img_filename))
            intrinsic, extrinsic, depth_min, depth_interval = self.read_camera_file(proj_mat_filename)

            if i == 0:
                depth_hr = np.asarray(read_pfm(depth_filename_hr), dtype=np.float32)
                mask_hr = self.read_mask(mask_filename_hr)
                depth_max = depth_min + depth_interval * self.ndepths
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
                if resize_scale != 1.0:
                    img, depth_hr, intrinsic, mask_hr = self.pre_resize(img, depth_hr, intrinsic, mask_hr, resize_scale)
            elif resize_scale != 1.0:
                img, _, intrinsic, _ = self.pre_resize(img, None, intrinsic, None, resize_scale)

            resized_imgs.append(img)
            resized_intrinsics.append(np.asarray(intrinsic, dtype=np.float32))
            extrinsics.append(np.asarray(extrinsic, dtype=np.float32))

        imgs = torch.from_numpy(np.stack(resized_imgs, axis=0)).permute(0, 3, 1, 2).float()  # [V,C,H,W]
        return {
            "images": imgs,                                    # for norm_fill / sfm
            "views_np": resized_imgs,                          # HWC list, for cropping
            "intrinsics": np.stack(resized_intrinsics, axis=0),
            "extrinsics": np.stack(extrinsics, axis=0),
            "depth_hr": depth_hr,
            "mask_hr": mask_hr,
            "depth_values": depth_values,
            "scan": scan, "ref_view": ref_view, "light_idx": light_idx,
        }

    def _match_hw(self, arr, hw, is_depth):
        h, w = hw
        if arr.shape[:2] == (h, w):
            return arr
        interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
        return cv2.resize(arr, (w, h), interpolation=interp)

    def __getitem__(self, idx):
        pc = self.precrop_inputs(idx)
        resized_imgs = pc["views_np"]
        resized_intrinsics = pc["intrinsics"]
        extrinsics = pc["extrinsics"]
        depth_hr, mask_hr, depth_values = pc["depth_hr"], pc["mask_hr"], pc["depth_values"]
        # scan, ref_view, light_idx = pc["scan"], pc["ref_view"], pc["light_idx"]
        num_v = len(resized_imgs)
        h0, w0 = resized_imgs[0].shape[:2]

        # --- SfM 稀疏深度 (pre-crop, 带磁盘缓存) ---
        # cache_path = self.sfm_cache_dir / scan / f"sfm_{ref_view:0>4}_{light_idx}.npy"
        # sfm_depth = sfm.load_or_compute_sparse_depth(
        #     images=np.stack(resized_imgs, axis=0),
        #     intrinsics=resized_intrinsics,
        #     extrinsics=extrinsics,
        #     cache_path=cache_path,
        #     ref_idx=0,
        # )

        # --- 预计算好的 prior (pre-crop 全帧, 由 pre_prior 离线缓存) ---
        prior = pre_prior.load_prior(self.prior_cache_path_for(idx))
        depth_prior_full = self._match_hw(prior["depth_prior"], (h0, w0), is_depth=True)
        conf_prior_full = self._match_hw(prior["conf_prior"], (h0, w0), is_depth=False)
        norm_full = self._match_hw(prior["norm_depth_fill"], (h0, w0), is_depth=False)
        src_weights = prior["src_weights"]

        # --- 裁剪到 (640x512): 所有视角/深度/prior 共用同一窗口 ---
        crop_x, crop_y = self.pick_crop_origin(h0, w0)
        y1, x1 = crop_y + self.height, crop_x + self.width

        images, intrinsics, projection_matrices = [], [], []
        # depth_gt = mask_gt = sfm_depth_crop = None
        depth_gt = mask_gt =  None
        for i in range(num_v):
            img, K, depth, mask = self.crop_at(
                resized_imgs[i], resized_intrinsics[i], crop_x, crop_y,
                depth_hr if i == 0 else None, mask_hr if i == 0 else None)
            if i == 0:
                depth_gt, mask_gt = depth, mask
                # _, _, sfm_depth_crop, _ = self.crop_at(
                #     resized_imgs[0], resized_intrinsics[0], crop_x, crop_y, sfm_depth)

            projection_matrix = np.eye(4, dtype=np.float32)
            projection_matrix[:3, :4] = K @ extrinsics[i][:3, :4]
            images.append(img)
            intrinsics.append(K)
            projection_matrices.append(projection_matrix)

        imgs = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float()  # [V, C, H, W]
        sample = {
            "images": imgs,
            "intrinsics": np.stack(intrinsics, axis=0),       # [V, 3, 3]
            "extrinsics": np.stack(extrinsics, axis=0),       # [V, 4, 4]
            "depth_gt": depth_gt,
            "mask": mask_gt,
            "depth_values": depth_values,
            "projection_matrices": np.stack(projection_matrices, axis=0),
            # "sfm_depth": sfm_depth_crop,
            # priors consumed by the network / loss (cropped to the same window)
            "depth_prior": depth_prior_full[crop_y:y1, crop_x:x1],
            "conf_prior": conf_prior_full[crop_y:y1, crop_x:x1],
            "norm_depth_fill": norm_full[crop_y:y1, crop_x:x1],
            "src_weights": src_weights,
        }
        return sample
