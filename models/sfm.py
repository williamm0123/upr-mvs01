"""Reusable fixed-camera SfM utilities for DTU experiments."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import pycolmap
import numpy as np
from PIL import Image


import models.general as G

#忽略pycolmap的日志输出，设置minloglevel为2表示只显示错误信息，忽略警告和信息日志
pycolmap.logging.minloglevel = 2

# 这个常量 COLMAP_PAIR_ID_MULTIPLIER = 2_147_483_647 是 COLMAP 用来生成图像对唯一标识符（Pair ID）的一个计算基数。
# 使用这么大的数字作为乘法因子，可以确保即使你有成千上万张图片，任意两张图片的 ID 组合生成的 pair_id 也绝对不会重复（碰撞）
COLMAP_PAIR_ID_MULTIPLIER = 2_147_483_647


@dataclass
class SFMConfig:
    output_root: Path = Path("outputs/sfm_dtu_fixed_camera")
    max_image_size: int = 1200
    max_num_features: int = 8192
    max_ratio: float = 0.8
    max_view_gap: int = 8
    min_pair_matches: int = 30
    min_depth: float = 1e-6
    max_depth: float = 2000.0
    max_reproj_error: float = 2.0
    min_tri_angle: float = 1.0
    voxel_size: float = 1.0
    # splat_radius: int = 2
    # range_percentiles: tuple[float, float] = (2.0, 98.0)
    overlay_alpha: float = 0.72
    overview_images: int = 8
    gpu: bool = False
    clean: bool = True


def prepare_image_subset(image_paths, image_out_dir: Path, clean: bool = True) -> list[Path]:
    if clean and image_out_dir.exists():
        shutil.rmtree(image_out_dir)

    image_out_dir.mkdir(parents=True, exist_ok=True)

    linked_paths = []

    for source in image_paths:
        source = Path(source)
        target = image_out_dir / source.name

        if not target.exists():
            target.symlink_to(source)

        linked_paths.append(target)

    return linked_paths


def configure_reader_options(intrinsic: np.ndarray) -> pycolmap.ImageReaderOptions:
    intrinsic = intrinsic.astype(np.float64)

    options = pycolmap.ImageReaderOptions()
    options.camera_model = "PINHOLE"
    options.camera_params = (
        f"{float(intrinsic[0, 0])},"
        f"{float(intrinsic[1, 1])},"
        f"{float(intrinsic[0, 2])},"
        f"{float(intrinsic[1, 2])}"
    )
    return options


def project_cloud_to_depth(points: np.ndarray,
                           intrinsic,extrinsic, H, W, 
                           min_depth = 1e-6, max_depth = 2000.0):
    
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera_points = points_h @ extrinsic.T
    z = camera_points[:, 2]
    pixel_h = camera_points[:, :3] @ intrinsic.T
    uv = pixel_h[:, :2] / np.clip(pixel_h[:, 2:3], 1e-12, None)
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    z = z.astype(np.float32)

    valid = (
        np.isfinite(u) &
        np.isfinite(v) &
        np.isfinite(z) &
        (z > min_depth) &
        (z < max_depth) &
        (u >= 0) &
        (u < W) &
        (v >= 0) &
        (v < H)
    )

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]

    print("valid projected points:", valid.sum(), "/", len(valid))

    depth = np.full((H, W), np.inf, dtype=np.float32)

    np.minimum.at(depth, (v_valid, u_valid), z_valid)

    depth[~np.isfinite(depth)] = np.nan

    return np.asarray(depth)
    

def generate_sfm_pointcloud(config, database_path: Path, sample: dict):
    """
    使用已知的精准内外参进行特征点三角化，并通过 Point-Only BA 和重投影误差过滤提升点云质量。
    """
    output_dir = config.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir = output_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    image_dir = colmap_dir / "images"
    database_path = Path(database_path)
    if database_path.parent == output_dir:
        database_path = colmap_dir / database_path.name
    ply_output_path = config.output_root / "sfm_points.ply"

    if config.clean:
        for stale_name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin", "database.db"):
            stale_path = output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        stale_image_dir = output_dir / "images"
        if stale_image_dir.exists() and stale_image_dir != image_dir:
            shutil.rmtree(stale_image_dir)
    
    # 1. 创建重建对象并注入完美相机参数
    reconstruction = pycolmap.Reconstruction()
    image_paths = sample["image_paths"]
    intrinsics = sample["intrinsics"]
    extrinsics = sample["extrinsics"]

    prepare_image_subset(image_paths, image_dir, clean=config.clean)

    if config.clean and database_path.exists():
        database_path.unlink()

    if not database_path.exists():
        reader_options = configure_reader_options(intrinsics[0])

        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.use_gpu = bool(config.gpu)
        extraction_options.max_image_size = int(config.max_image_size)
        extraction_options.sift.max_num_features = int(config.max_num_features)

        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.use_gpu = bool(config.gpu)
        matching_options.sift.max_ratio = float(config.max_ratio)

        device = pycolmap.Device.cuda if config.gpu else pycolmap.Device.cpu
        print(f"Extracting features for {len(image_paths)} images")
        pycolmap.extract_features(
            database_path,
            image_dir,
            camera_mode=pycolmap.CameraMode.PER_IMAGE,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=device,
        )
        print("Running exhaustive matching")
        pycolmap.match_exhaustive(
            database_path,
            matching_options,
            device=device,
        )
    
    imgs_shape = sample["images"].shape
    if imgs_shape[-1] == 3:  
        H, W = imgs_shape[1:3]
    else:                    
        H, W = imgs_shape[2:4]

    sample_by_name = {Path(path).name: i for i, path in enumerate(image_paths)}
    with sqlite3.connect(database_path) as connection:
        database_images = [
            (int(image_id), str(name), int(camera_id))
            for image_id, name, camera_id in connection.execute(
                "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
            )
        ]

    if not database_images:
        raise RuntimeError(f"No images were written to COLMAP database: {database_path}")

    added_camera_ids = set()
    for image_id, image_name, camera_id in database_images:
        basename = Path(image_name).name
        if basename not in sample_by_name:
            raise RuntimeError(
                f"Database image {image_name!r} is not part of the current sample. "
                "Run with clean=True to rebuild the SfM database."
            )

        i = sample_by_name[basename]
        
        # 注入内参
        K = intrinsics[i]
        if camera_id not in added_camera_ids:
            camera = pycolmap.Camera(
                model="PINHOLE",
                width=W,
                height=H,
                params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
            )
            camera.camera_id = camera_id
            reconstruction.add_camera_with_trivial_rig(camera)
            added_camera_ids.add(camera_id)

        
        # 注入外参
        E = extrinsics[i] 
        E_3x4 = np.asarray(E[:3, :4], dtype=np.float64)
        cam_from_world = pycolmap.Rigid3d(E_3x4)
        image = pycolmap.Image(
            image_id=image_id,
            name=image_name,
            camera_id=camera_id,
        )
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)
    # ================= 修正 1：初始三角化 =================
    # print("正在根据真实位姿进行特征三角化...")
    
    # 报错指出：必须使用 IncrementalPipelineOptions
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    # 将三角化的参数设置在 .triangulation 子模块下
    pipeline_options.triangulation.min_angle = getattr(config, 'min_tri_angle', 1.5)
    pipeline_options.triangulation.ignore_two_view_tracks = False 
    
    reconstruction = pycolmap.triangulate_points(
        reconstruction, 
        database_path, 
        image_dir, 
        output_path=colmap_dir, 
        clear_points=True,
        options=pipeline_options  # <--- 传入修正后的 pipeline_options
    )
    print(f"初始三角化获得 {reconstruction.num_points3D()} 个 3D 点。")

    # ==================== 核心改进模块开始 ====================

# ================= 修正 2 & 3：Bundle Adjustment =================
    # print("开始进行 Bundle Adjustment (锁死相机，仅优化 3D 点坐标)...")
    
    # A. 求解器选项 (Options)
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = False
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = False
    ba_options.refine_rig_from_world = False
    ba_options.refine_sensor_from_rig = False
    ba_options.refine_points3D = True

    # B. 问题配置 (Config)
    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        ba_config.add_image(image_id)
    for point3D_id in reconstruction.point3D_ids():
        ba_config.add_variable_point(point3D_id)
    for camera_id in reconstruction.cameras:
        ba_config.set_constant_cam_intrinsics(camera_id)

    # C. 运行优化
    if hasattr(pycolmap, "create_default_bundle_adjuster"):
        bundle_adjuster = pycolmap.create_default_bundle_adjuster(
            ba_options,
            ba_config,
            reconstruction,
        )
        ba_summary = bundle_adjuster.solve()
        if hasattr(ba_summary, "brief_report"):
            print("BA summary:", ba_summary.brief_report())
    elif hasattr(pycolmap, "bundle_adjustment"):
        pycolmap.bundle_adjustment(reconstruction, ba_options)
    else:
        print("提示：当前 pycolmap 版本不支持独立 BA，跳过此步骤 (三角化阶段可能已自带优化)")

    # 4. 剔除劣质噪点 (Filtering)
    print("开始过滤重投影误差过大的噪点...")
    max_reproj_err = getattr(config, 'max_reproj_error', 2.0)
    min_track_length = 2  # 如果你想要更干净的图，可以改成 3 (即至少在3个视角中被看到)
    
    points_to_delete = []
    # 遍历所有生成的 3D 点
    for point3D_id, point3D in reconstruction.points3D.items():
        # 条件A：重投影误差过大（比如大于 2 个像素）
        # 条件B：Track长度不足（视野太少，不可靠）
        if point3D.error > max_reproj_err or point3D.track.length() < min_track_length:
            points_to_delete.append(point3D_id)
            
    # 执行删除
    for point3D_id in points_to_delete:
        reconstruction.delete_point3D(point3D_id)

    # ==================== 核心改进模块结束 ====================

    # 5. 导出结果
    points = np.asarray(
        [point3D.xyz for point3D in reconstruction.points3D.values()],
        dtype=np.float32,
    ).reshape(-1, 3)
    sparse_depth = project_cloud_to_depth(
        points,
        intrinsics[0],
        extrinsics[0],
        H,
        W,
        min_depth=config.min_depth,
        max_depth=config.max_depth,
    )
    num_points = points.shape[0]
    # reconstruction.export_PLY(str(ply_output_path))
    
    print(f"✅ 优化并过滤完成！最终保留 {num_points} 个高质量 3D 点。")
    # print(f"📁 点云已保存至: {ply_output_path}")
   
    return points, sparse_depth
