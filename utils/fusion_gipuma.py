"""Gipuma / fusibile 融合后端 —— 移植自 MVSFormer++ ``misc/gipuma.py``。

为什么需要它: 现有的 ``test.py:fuse_scan`` 是标准的 MVSNet 式几何+光度一致性
过滤 —— **每个 ref 视角各自输出一遍自己的存活像素, 跨视角不去重**。49 个视角、
0.8 缩放下每视角 960x1280 像素、保留约 61%, 于是一个 scan 出 2500-4500 万个点、
ply 600MB。同一个物理表面点在重叠区被重复写了 5-15 次。

MVSFormer++ 的 dypcd (``dynamic_filter_depth``) **同样不去重** —— 它只是把固定
阈值换成随一致视角数递进的动态阈值, 输出仍是 ``views[ref_id]`` 逐视角 concat。
真正做跨视角合并的只有 fusibile: 它把一致的点聚成一个, 输出通常 1-3M 点。

fusibile 是独立的 CUDA C++ 工程 (gipuma 的融合部分), 不在本仓库里, 需要先编译:

    bash scripts/build_fusibile.sh

流程 (与 MVSFormer++ 逐字一致, 便于对齐它们的 DTU 数):

    1. probability filter: conf <= 阈值的像素深度置 0
    2. 导出 gipuma 目录树:
         points_mvsnet/cams/<img>.P          P = K_4x4 @ E 的前三行
         points_mvsnet/images/<img>.png
         points_mvsnet/2333__<prefix>/disp.dmb     深度
         points_mvsnet/2333__<prefix>/normals.dmb  伪法向 (全 1/sqrt(3), 深度为 0 处置 0)
    3. 调 fusibile 二进制
    4. 取 consistencyCheck-*/final3d_model.ply

``normal_thresh=360`` 是 MVSFormer++ 的取值 —— 伪法向处处相同, 这个阈值等于把
法向一致性检查关掉, 只留视差与视角数两个条件。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from struct import pack, unpack

import numpy as np


# --------------------------------------------------------------------------- #
# dmb 读写 (gipuma 的原生深度/法向格式)
# --------------------------------------------------------------------------- #
def write_gipuma_dmb(path, image: np.ndarray) -> None:
    """header = <int type=1, int height, int width, int channels>, 之后是 float32 数据。

    注意通道维要先转到最前再写 —— gipuma 按 (channel, width, height) 的 Fortran
    序读, 顺序错了不会报错, 只会得到一张乱掉的图。
    """
    shape = np.shape(image)
    height, width = shape[0], shape[1]
    channels = shape[2] if len(shape) == 3 else 1
    if len(shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()
    with open(path, "wb") as fid:
        fid.write(pack("<i", 1))
        fid.write(pack("<i", height))
        fid.write(pack("<i", width))
        fid.write(pack("<i", channels))
        np.ascontiguousarray(image, dtype=np.float32).tofile(fid)


def read_gipuma_dmb(path) -> np.ndarray:
    with open(path, "rb") as fid:
        _ = unpack("<i", fid.read(4))[0]
        height = unpack("<i", fid.read(4))[0]
        width = unpack("<i", fid.read(4))[0]
        channel = unpack("<i", fid.read(4))[0]
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def fake_gipuma_normal(depth: np.ndarray) -> np.ndarray:
    """处处相同的伪法向, 深度为 0 的地方置 0。

    fusibile 的接口要求有法向图, 但我们没有。配 normal_thresh=360 (等于关掉法向
    一致性检查) 就退化成纯几何一致性 —— 这正是 MVSFormer++ 的做法。
    """
    n = np.ones(depth.shape + (3,), np.float32) / 1.7320508075688772
    n *= (depth > 0)[..., None].astype(np.float32)
    return n.astype(np.float32)


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def write_gipuma_cam(path, K: np.ndarray, E: np.ndarray) -> None:
    """P = K_4x4 @ E 的前三行, 空格分隔、每行一行、末尾一个空行。"""
    K4 = np.zeros((4, 4), np.float64)
    K4[:3, :3] = np.asarray(K, np.float64)
    P = (K4 @ np.asarray(E, np.float64))[:3, :]
    with open(path, "w") as f:
        for i in range(3):
            f.write(" ".join(str(P[i][j]) for j in range(4)) + " \n")
        f.write("\n")


def export_scan(scan_dir: Path, point_folder: Path, photo_thresh: float) -> int:
    """把一个 scan 的深度缓存导成 fusibile 的目录树, 返回视角数。

    ``scan_dir`` 是 test.py 写的逐视角 npz (depth / conf / K / E / image)。
    """
    import imageio.v2 as imageio

    cams = point_folder / "cams"
    imgs = point_folder / "images"
    for d in (point_folder, cams, imgs):
        d.mkdir(parents=True, exist_ok=True)

    n = 0
    for f in sorted(scan_dir.glob("*.npz")):
        ref_id = int(f.stem)
        z = np.load(f)
        depth = np.asarray(z["depth"], np.float32).copy()
        conf = np.asarray(z["conf"], np.float32)
        # probability filter: 不达阈值的深度置 0, fusibile 把 0 当无效
        depth[conf <= photo_thresh] = 0.0

        name = f"{ref_id:08d}"
        imageio.imwrite(imgs / f"{name}.png", np.asarray(z["image"], np.uint8))
        write_gipuma_cam(cams / f"{name}.png.P", z["K"], z["E"])

        sub = point_folder / f"2333__{name}"
        sub.mkdir(exist_ok=True)
        write_gipuma_dmb(sub / "disp.dmb", depth)
        write_gipuma_dmb(sub / "normals.dmb", fake_gipuma_normal(depth))
        n += 1
    return n


# --------------------------------------------------------------------------- #
# 调用二进制
# --------------------------------------------------------------------------- #
def run_fusibile(point_folder: Path, exe: str, disp_thresh: float,
                 num_consistent: int, color: bool = True) -> Path:
    """跑 fusibile, 返回它写出的 ply。

    depth_min / depth_max / normal_thresh 沿用 MVSFormer++ 的取值。
    """
    cmd = [
        str(exe),
        "-input_folder", f"{point_folder}/",
        "-p_folder", f"{point_folder / 'cams'}/",
        "-images_folder", f"{point_folder / 'images'}/",
        "--depth_min=0.001",
        "--depth_max=100000",
        "--normal_thresh=360",
        f"--disp_thresh={disp_thresh}",
        f"--num_consistent={num_consistent}",
    ]
    if color:
        cmd.append("-color_processing")
    print("[gipuma] " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-2000:]
        raise RuntimeError(f"fusibile 退出码 {r.returncode}\n{tail}")

    outs = sorted(point_folder.glob("consistencyCheck-*/final3d_model.ply"),
                  key=os.path.getmtime)
    if not outs:
        tail = (r.stdout or "")[-2000:]
        raise RuntimeError(f"fusibile 没有产出 final3d_model.ply\n{tail}")
    return outs[-1]


def fuse_scan_gipuma(scan_dir: Path, ply_path: Path, exe: str, photo_thresh: float,
                     disp_thresh: float, num_consistent: int,
                     keep_tmp: bool = False) -> int:
    """一个 scan 的完整流程, 返回点数 (读不出来时返回 -1)。

    中间目录默认跑完就删: 每个 scan 的 dmb 约 1GB (49 视角 x (深度 4.9MB +
    法向 14.7MB)), 22 个 scan 不清理会吃掉几十 GB。
    """
    tmp = scan_dir.parent / f"_gipuma_{scan_dir.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        nv = export_scan(scan_dir, tmp, photo_thresh)
        if nv == 0:
            return 0
        src = run_fusibile(tmp, exe, disp_thresh, num_consistent)
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, ply_path)
        try:
            from plyfile import PlyData
            return int(PlyData.read(str(ply_path))["vertex"].count)
        except Exception:
            return -1
    finally:
        if not keep_tmp and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
