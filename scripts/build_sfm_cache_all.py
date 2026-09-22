#!/usr/bin/env python3
"""离线构建全数据集的 SfM 稀疏点云: 每个 (scan, view) 一个 npz, 只用 light 3。

相机用 DTU 自带的标定 (``Cameras/{view:08d}_cam.txt``, 1200x1600 原生分辨率下的 K 与
world->cam 外参, 单位 mm), **不估计位姿**, 只找对应点再用已知投影矩阵三角化, 所以点天然
是 metric 的。几何与光照无关, 同一个视角的 7 种光照共用一份点云, 这里只在 light 3 (测试
集固定用的那个) 上建, 之后给 ``scripts/calibrate_da3_with_sfm.py`` 标定全部光照的 DA3。

## 相比 models/sfm.py 改了什么, 为什么

[[sfm-scale-is-the-bottleneck]] 的结论是 DA3 标尺的瓶颈是 SfM 点太少 (22 个 test scan
中位 1.1K 点, scan13 只有 47 个)。实测根因是 scan13/48/77 这类高亮/低纹理物体在默认
SIFT 阈值下只有 1~2K 个关键点, 再经过 ratio test + 估计出来的 F 矩阵 RANSAC 就所剩无几。
这里用"相机已知"这个条件把匹配做满:

1. **极线引导匹配**: F 直接由已知 K/R/t 算, 每个 ref 关键点只在 src 图里离它的极线
   ``--epi-thresh`` 像素以内的候选中找最近邻 (+ 双向最近邻), ratio test 也只在这些候选里做,
   不再需要 RANSAC 去估一个本来就已知的 F。代价是 n1 x n2 的距离矩阵, 放在 GPU 上分块算。
   **带内 ratio test 单独用不安全**: 真对应点没被检测到时带里只剩寥寥几个候选, ratio test
   形同虚设, 放进来的错配沿极线滑动, 重投影误差是 0, 三角化守卫拦不住 —— 实测 scan13/77
   这类 scan 上 34~37% 的点误差 >10mm。所以每条匹配另外标一个 "强匹配" (不看极线也成立:
   带内最佳 = 全图最佳, 且全图 ratio < ``--global-ratio``), 见第 4 条的保留规则。
2. **RootSIFT + 更低的对比度阈值** (``--contrast-threshold`` 默认 0.01, OpenCV 默认 0.04):
   高亮 scan 的关键点数翻 2~4 倍。
3. **对称的邻居图**: r 与 s 互为伙伴当且仅当 s 在 r 的 pair.txt 前 ``--neighbors`` 个里,
   或 r 在 s 的前 ``--neighbors`` 个里。每条边只匹配/三角化一次, 结果同时给两端的视角用。
4. **多视角合并 + 保留规则**: 同一个 ref 关键点位置可能和多个伙伴视角都三角化出点, 取中位
   深度, 与中位相差超过 ``--merge-tol`` (相对深度) 的观测视为错配丢掉。一个点保留当且仅当
   有 >= ``--min-obs`` (默认 2) 个不同伙伴视角一致支持, 或者支持它的观测里有强匹配。

2026-09-22 本地 6 个 scan (1/13/24/48/77/114, 294 视角) 端到端对比, 指标是标定后 DA3 对 GT
的逐文件中位误差 (GT 仿射上界 7.41mm):

    保留规则                     每视角点数中位   标定误差中位   p90      >2x 上界的文件
    全部 (--min-obs 1)             7290          8.86mm      87.8mm    67/294
    n_obs>=2 或强匹配 (默认)        5069          8.51mm      38.4mm    41/294
    只要强匹配                      3690          8.53mm      47.3mm    49/294
    (旧 models/sfm.py 的 test scan 点数中位 1.1K, scan13 只有 47)

CLAHE / contrastThreshold 0.005 对点数几乎没帮助 (新增的点和原来在同一片区域), 默认关。
剩下的大误差来自 scan77/48 这类: 白桌面在 7 种光照下都 97~98% 饱和, 任何光照都提不出
特征, 那片深度区间没有 SfM 点, 全局仿射只能外推 —— 这是数据本身的限制, 不是匹配能解决的,
下游用 calibrate_da3_with_sfm.py 输出里的 support_depth / extrap_frac 识别。

守卫: 两个相机前方、两边重投影误差 <= ``--reproj-thresh``、三角化夹角 >=
``--min-tri-angle``、深度在 cam.txt 给的深度范围 (前后各放宽 ``--depth-margin``) 内。

## 输出

``<out>/<scan>/sfm_{view:04d}_{light}.npz``, 每个点对应 ref 图上的一个关键点位置:

    uv          float32 [N,2]  ref 像素坐标 (x 右 y 下, 1600x1200 原生, OpenCV 约定: 整数=像素中心)
    depth       float32 [N]    ref 相机系 z (mm)
    xyz         float32 [N,3]  世界坐标 (mm), = uv+depth 反投影, 与 uv 严格对应
    n_obs       uint8   [N]    支持这个深度的不同伙伴视角数
    strong      bool    [N]    支持它的观测里有没有强匹配
    conf        float32 [N]    [0,1], 与 models/sfm.py 同一公式 (重投影 x 夹角 x 边内点数), 取各观测最大
    reproj_err  float32 [N]    px, 各观测两端重投影误差的最大值
    tri_angle   float32 [N]    度, 各观测三角化夹角的最大值
    rgb         uint8   [N,3]  ref 图 (light 3) 上的颜色
    partners    int32   [P]    这个视角的伙伴视角号; pair_stats int32 [P,3] = 每个伙伴的
                               (极线引导匹配数, 三角化守卫后剩下的点数, 合并后仍被采用的观测数)
    scan / view / light / image_hw / params (json 字符串)

GT (``Depths_raw``) **只用于写到汇总表里的质量统计** (点深度与 GT 的偏差), 不参与任何
筛选, 输出本身不碰 GT。

## 用法

    python scripts/build_sfm_cache_all.py                         # 全量, 断点续跑
    python scripts/build_sfm_cache_all.py --dry-run                # 只统计待办
    python scripts/build_sfm_cache_all.py --scans 1 13 --workers 1  # 调试
    python scripts/build_sfm_cache_all.py --force                  # 全部重建

按 scan 并行 (``--workers`` 个进程, 每个进程自己占一份 CUDA context 做匹配, SIFT 在 CPU
上)。已存在的 (scan, view) 跳过; 一个 scan 只缺几个视角时只补缺的那几个 (以及它们需要的边)。
每轮的汇总写 ``<out>/_sfm_summary_<时间戳>.csv``, 失败写 ``_sfm_failures_<时间戳>.csv``。
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base.config import build_mvs_config  # noqa: E402

NATIVE_H, NATIVE_W = 1200, 1600
NDEPTHS = 192          # cam.txt 第 12 行是 (depth_min, interval), MVSNet 约定 192 档


# --------------------------------------------------------------------------- #
# 参数 / 枚举
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scans", type=int, nargs="*", default=None,
                   help="只建这些 scan 编号 (默认: Rectified_raw/ 下的全部)")
    p.add_argument("--views", type=int, nargs="*", default=None,
                   help="0 起算的视角号 (默认: pair.txt 里的全部)")
    p.add_argument("--light", type=int, default=3, help="用哪个光照的图建点云 (默认 3)")
    p.add_argument("--neighbors", type=int, default=10,
                   help="每个视角取 pair.txt 前几个邻居 (对称化后作为伙伴); pair.txt 每行 10 个")
    # 特征
    p.add_argument("--max-features", type=int, default=40000,
                   help="SIFT 每张图最多保留的关键点 (按响应排序; 0=不限)")
    p.add_argument("--contrast-threshold", type=float, default=0.01,
                   help="SIFT contrastThreshold, OpenCV 默认 0.04; 高亮/低纹理 scan 调低能多 2~4 倍点")
    p.add_argument("--clahe", type=float, default=0.0,
                   help=">0 时先做 CLAHE (clipLimit=该值) 再提特征; 0=关")
    # 匹配
    p.add_argument("--ratio", type=float, default=0.8,
                   help="ratio test 阈值 (只在极线带内的候选之间比)")
    p.add_argument("--max-desc-dist", type=float, default=0.7,
                   help="RootSIFT (单位向量) 最近邻的欧氏距离上限; 极线带里只有一个候选时靠它兜底")
    p.add_argument("--epi-thresh", type=float, default=2.0,
                   help="极线带半宽 (px): 两张图里点到对方极线的距离都要 <= 它")
    p.add_argument("--no-mutual", action="store_true", help="关掉双向最近邻检查")
    p.add_argument("--global-ratio", type=float, default=0.9,
                   help="'强匹配' 判据: 带内最佳同时也是全图最佳, 且全图 ratio < 该值")
    p.add_argument("--min-obs", type=int, default=2,
                   help="保留一个点需要的一致伙伴视角数; 只有一个伙伴支持时, 那条匹配必须是强匹配")
    # 三角化 / 合并
    p.add_argument("--reproj-thresh", type=float, default=2.0, help="两端重投影误差上限 (px)")
    p.add_argument("--min-tri-angle", type=float, default=1.0, help="三角化夹角下限 (度)")
    p.add_argument("--depth-margin", type=float, default=0.25,
                   help="深度守卫: cam.txt 的 [dmin, dmin+192*interval] 前后各放宽这个比例")
    p.add_argument("--merge-tol", type=float, default=0.01,
                   help="同一 ref 关键点多个观测的深度与中位数的相对偏差上限")
    # 运行
    p.add_argument("--workers", type=int, default=4, help="并行的 scan 数 (进程数)")
    p.add_argument("--cv-threads", type=int, default=0,
                   help="每个进程的 OpenCV 线程数 (0=可用 CPU 数 / workers)")
    p.add_argument("--device", default="cuda", help="匹配用的设备; 没有 CUDA 时自动退回 CPU (很慢)")
    p.add_argument("--save-ply", action="store_true", help="每个视角另存一个 .ply 方便看")
    p.add_argument("--no-gt-eval", action="store_true", help="汇总表里不算与 GT 的偏差")
    p.add_argument("--force", action="store_true", help="已存在的文件也重建")
    p.add_argument("--dry-run", action="store_true", help="只统计待办, 不跑")
    p.add_argument("--out", default=None, help="覆盖 cfg.paths.sfm_sparse_cache_path")
    p.add_argument("--dtu-root", default=None, help="覆盖 cfg.paths.dtu_train_root")
    return p.parse_args()


def read_pairs(dtu_root: Path) -> dict[int, list[int]]:
    """pair.txt: 第一行视角数, 之后每个视角两行 (视角号 / "10 v1 s1 v2 s2 ...")。"""
    tok = (dtu_root / "Cameras" / "pair.txt").read_text().split()
    n, k = int(tok[0]), 1
    out: dict[int, list[int]] = {}
    for _ in range(n):
        ref, cnt = int(tok[k]), int(tok[k + 1])
        out[ref] = [int(x) for x in tok[k + 2:k + 2 + 2 * cnt:2]]
        k += 2 + 2 * cnt
    return out


def scan_list(dtu_root: Path) -> list[str]:
    rect = dtu_root / "Rectified_raw"
    return sorted((d.name for d in rect.iterdir() if d.is_dir()), key=lambda s: (len(s), s))


def image_path(dtu_root: Path, scan: str, view: int, light: int) -> Path:
    return dtu_root / "Rectified_raw" / scan / f"rect_{view + 1:03d}_{light}_r5000.png"


def sfm_path(root: Path, scan: str, view: int, light: int) -> Path:
    return root / scan / f"sfm_{view:04d}_{light}.npz"


def read_cam(dtu_root: Path, view: int) -> dict:
    """与 DTUMVSDataset.read_camera_file 同一个格式 (K 是 1200x1600 原生分辨率的)。"""
    lines = (dtu_root / "Cameras" / f"{view:08d}_cam.txt").read_text().splitlines()
    E = np.fromstring(" ".join(lines[1:5]), dtype=np.float64, sep=" ").reshape(4, 4)
    K = np.fromstring(" ".join(lines[7:10]), dtype=np.float64, sep=" ").reshape(3, 3)
    dmin, interval = (float(x) for x in lines[11].split()[:2])
    return {"K": K, "E": E, "P": K @ E[:3, :4], "C": -E[:3, :3].T @ E[:3, 3],
            "dmin": dmin, "dmax": dmin + interval * NDEPTHS}


# --------------------------------------------------------------------------- #
# 特征 / 匹配 / 三角化
# --------------------------------------------------------------------------- #
def extract_features(gray: np.ndarray, sift, clahe) -> tuple[np.ndarray, np.ndarray]:
    """RootSIFT: L1 归一化再开方, 结果自动是单位 L2 向量, 点积即余弦相似度。"""
    g = clahe.apply(gray) if clahe is not None else gray
    kps, des = sift.detectAndCompute(g, None)
    if des is None or not kps:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    xy = np.float32([k.pt for k in kps])
    des = des.astype(np.float32)
    des /= np.maximum(des.sum(axis=1, keepdims=True), 1e-12)
    return xy, np.sqrt(des)


def fundamental(cam1: dict, cam2: dict) -> np.ndarray:
    """x2^T F x1 = 0, 由已知 world->cam 外参直接算, 不估计。"""
    R1, t1 = cam1["E"][:3, :3], cam1["E"][:3, 3]
    R2, t2 = cam2["E"][:3, :3], cam2["E"][:3, 3]
    R = R2 @ R1.T
    t = t2 - R @ t1
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], np.float64)
    return np.linalg.inv(cam2["K"]).T @ (tx @ R) @ np.linalg.inv(cam1["K"])


def guided_match(xy1, d1, xy2, d2, F, *, ratio, max_desc_dist, epi_thresh, mutual, global_ratio,
                 device, max_elems: float = 1.0e8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """极线引导的最近邻匹配, 返回 (idx1, idx2, strong)。

    对每个 i: 候选 = {j : 点 j 到 i 的极线距离 <= thr 且 点 i 到 j 的极线距离 <= thr};
    在候选里取余弦最大的两个做 ratio test (只有一个候选时 ratio 自动通过, 靠
    max_desc_dist 兜底); mutual=True 时还要求 i 也是 j 在它自己候选里的最佳。

    strong = 这条匹配**不看极线**也成立 (带内最佳就是全图最佳, 且全图 ratio < global_ratio)。
    实测 (scan13/77/1/48 的 v0, 对 GT): 带内 ratio 在高亮/低纹理 scan 上会放进 20~40% 沿极线
    滑动的错配 (误差 >10mm) —— 真对应点没被检测到时, 带里剩下的候选很少, ratio test 形同虚设,
    而错配依然满足极线, 重投影误差为 0, 三角化守卫拦不住。强匹配的错配率 0~9%。merge 时只有
    强匹配, 或者被 >=2 个伙伴视角一致支持的点才保留。按 i 分块,
    每块 [c, n2] 的矩阵不超过 max_elems 个元素。TF32 必须关: 极线距离里有 ~2000 的像素坐标,
    TF32 的 10 位尾数会带来 ~px 级误差。"""
    import torch

    n1, n2 = len(xy1), len(xy2)
    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, bool))
    if n1 < 2 or n2 < 2:
        return empty
    torch.backends.cuda.matmul.allow_tf32 = False
    f64 = dict(dtype=torch.float64, device=device)
    Ft = torch.as_tensor(F, **f64)
    x1 = torch.cat([torch.as_tensor(xy1, **f64), torch.ones(n1, 1, **f64)], 1)
    x2 = torch.cat([torch.as_tensor(xy2, **f64), torch.ones(n2, 1, **f64)], 1)
    l2 = x1 @ Ft.T                                   # [n1,3] i 在图 2 里的极线
    l1 = x2 @ Ft                                     # [n2,3] j 在图 1 里的极线 (F^T x2)
    l2 = (l2 / l2[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-12)).float()
    l1 = (l1 / l1[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-12)).float()
    x1f, x2f = x1.float(), x2.float()
    D1 = torch.as_tensor(d1, dtype=torch.float32, device=device)
    D2 = torch.as_tensor(d2, dtype=torch.float32, device=device)

    chunk = int(max(256, min(n1, max_elems // max(n2, 1))))
    best_j = torch.empty(n1, dtype=torch.long, device=device)
    s0 = torch.empty(n1, dtype=torch.float32, device=device)
    s1 = torch.empty(n1, dtype=torch.float32, device=device)
    g_best = torch.empty(n1, dtype=torch.long, device=device)
    g1 = torch.empty(n1, dtype=torch.float32, device=device)
    col_best = torch.full((n2,), -4.0, dtype=torch.float32, device=device)
    col_arg = torch.full((n2,), -1, dtype=torch.long, device=device)
    for a in range(0, n1, chunk):
        b = min(n1, a + chunk)
        epi = (l2[a:b] @ x2f.T).abs_()                          # 图 2 里 j 到 i 的极线
        epi = torch.maximum(epi, (x1f[a:b] @ l1.T).abs_())     # 图 1 里 i 到 j 的极线
        sim = D1[a:b] @ D2.T
        gv, gj = sim.topk(2, dim=1)                             # 不看极线的全图前二
        g_best[a:b], g1[a:b] = gj[:, 0], gv[:, 1]
        sim.masked_fill_(epi > epi_thresh, -4.0)
        del epi
        v, j = sim.topk(2, dim=1)
        best_j[a:b], s0[a:b], s1[a:b] = j[:, 0], v[:, 0], v[:, 1]
        cv_, ci = sim.max(dim=0)
        upd = cv_ > col_best
        col_best = torch.where(upd, cv_, col_best)
        col_arg = torch.where(upd, ci + a, col_arg)
        del sim

    has = s0 > -3.0
    dist0 = (2.0 - 2.0 * s0).clamp_min(0).sqrt()
    dist1 = torch.where(s1 > -3.0, (2.0 - 2.0 * s1).clamp_min(0).sqrt(),
                        torch.full_like(s1, float("inf")))
    keep = has & (dist0 < ratio * dist1) & (dist0 <= max_desc_dist)
    if mutual:
        keep &= col_arg[best_j] == torch.arange(n1, device=device)
    gdist1 = (2.0 - 2.0 * g1).clamp_min(0).sqrt()
    strong = (g_best == best_j) & (dist0 < global_ratio * gdist1)
    i = torch.nonzero(keep).squeeze(1)
    return i.cpu().numpy(), best_j[i].cpu().numpy(), strong[i].cpu().numpy()


def match_with_retry(*args, **kw):
    """显存不够 (多个进程挤一张小卡) 时把分块缩到 1/4 重试, 而不是整个 scan 失败。"""
    import torch

    max_elems = 1.0e8
    while True:
        try:
            return guided_match(*args, **kw, max_elems=max_elems)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() or max_elems < 1.0e6:
                raise
            torch.cuda.empty_cache()
            max_elems /= 4


def triangulate(xy1, xy2, cam1, cam2, opts) -> dict:
    """已知 P 的两视图三角化 + 守卫。返回保留下来的点及每点的 z/误差/夹角。"""
    X4 = cv2.triangulatePoints(cam1["P"], cam2["P"], xy1.T.astype(np.float64),
                               xy2.T.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        X = (X4[:3] / X4[3:4]).T
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
    out = {}
    keep = np.all(np.isfinite(X), axis=1)
    for tag, cam, xy in (("1", cam1, xy1), ("2", cam2, xy2)):
        pr = Xh @ cam["P"].T
        z = (Xh @ cam["E"][:3, :4].T)[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.linalg.norm(pr[:, :2] / pr[:, 2:3] - xy, axis=1)
        m = opts["depth_margin"]
        keep &= (z > cam["dmin"] * (1 - m)) & (z < cam["dmax"] * (1 + m))
        keep &= np.isfinite(err) & (err <= opts["reproj_thresh"])
        out["z" + tag], out["err" + tag] = z, err
    v1, v2 = cam1["C"][None] - X, cam2["C"][None] - X
    cosang = np.sum(v1 * v2, 1) / np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    keep &= ang >= opts["min_tri_angle"]
    out["ang"] = ang
    return {k: v[keep] for k, v in out.items()} | {"X": X[keep], "keep": keep}


def point_conf(err1, err2, ang, n_pair: int) -> np.ndarray:
    """与 models/sfm.py::_point_confidence 同一公式与默认常数 (tau_e=1px, theta_sat=10deg,
    N0=100): exp(-(max_err/tau)^2) * clip(sin(ang)/sin(10deg), 0, 1) * N/(N+N0)。"""
    f_reproj = np.exp(-(np.maximum(err1, err2) / 1.0) ** 2)
    f_angle = np.clip(np.sin(np.radians(ang)) / np.sin(np.radians(10.0)), 0.0, 1.0)
    return (f_reproj * f_angle * (n_pair / (n_pair + 100.0))).astype(np.float32)


# --------------------------------------------------------------------------- #
# 每个 ref 的合并 / 落盘
# --------------------------------------------------------------------------- #
def merge_observations(obs: list[dict], xy_ref: np.ndarray, tol: float,
                       min_obs: int) -> dict | None:
    """obs: 每条边给这个 ref 的观测 {kp, z, err, ang, conf, partner, strong}。按 ref 关键点
    **位置**分组 (SIFT 会在同一位置按多个主方向各出一个关键点, 按 kp 下标分组会重复计数),
    组内取中位深度, |z-med| <= tol*med 的是内点; 深度取内点均值, n_obs 数内点里的不同伙伴。
    保留: n_obs >= min_obs, 或者内点里有强匹配 (见 guided_match)。"""
    if not obs:
        return None
    cat = {k: np.concatenate([o[k] for o in obs])
           for k in ("kp", "z", "err", "ang", "conf", "partner", "strong")}
    if not len(cat["z"]):
        return None
    _, loc_of_kp = np.unique(xy_ref, axis=0, return_inverse=True)
    loc = loc_of_kp.reshape(-1)[cat["kp"]]
    order = np.lexsort((cat["z"], loc))                      # 先按位置, 组内按深度
    loc_s, z_s = loc[order], cat["z"][order]
    uniq, start, cnt = np.unique(loc_s, return_index=True, return_counts=True)
    med = 0.5 * (z_s[start + (cnt - 1) // 2] + z_s[start + cnt // 2])
    g = np.repeat(np.arange(len(uniq)), cnt)                 # 排序后每个观测所属的组号
    inl = np.abs(z_s - med[g]) <= tol * med[g]
    n_in = np.bincount(g, weights=inl, minlength=len(uniq))
    depth = np.bincount(g, weights=z_s * inl, minlength=len(uniq)) / np.maximum(n_in, 1)

    def gmax(v):
        r = np.full(len(uniq), -np.inf)
        np.maximum.at(r, g[inl], v[order][inl])
        return r

    partner_s = cat["partner"][order]
    pairs = np.unique(np.stack([g[inl], partner_s[inl]], 1), axis=0)
    n_obs = np.bincount(pairs[:, 0], minlength=len(uniq)) if len(pairs) else np.zeros(len(uniq), int)
    strong = np.bincount(g, weights=inl & cat["strong"][order], minlength=len(uniq)) > 0
    ok = (n_in > 0) & ((n_obs >= min_obs) | strong)
    # 一个观测是否被最终采用 (内点且组里至少一个内点), 用于每个伙伴的统计
    used_partner = partner_s[inl & ok[g]]
    kp_first = cat["kp"][order][start]                      # 组里任取一个 kp, 用来取 uv
    return {"uv": xy_ref[kp_first][ok], "depth": depth[ok], "n_obs": n_obs[ok], "strong": strong[ok],
            "conf": gmax(cat["conf"])[ok], "reproj_err": gmax(cat["err"])[ok],
            "tri_angle": gmax(cat["ang"])[ok], "used_partner": used_partner}


def backproject(uv: np.ndarray, z: np.ndarray, cam: dict) -> np.ndarray:
    K, E = cam["K"], cam["E"]
    x = (uv[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (uv[:, 1] - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return (pc - E[:3, 3]) @ E[:3, :3]                      # R^T (p - t)


def atomic_savez(path: Path, **arrays) -> None:
    """临时名带 pid+随机串再 os.replace, 与 build_da3_cache_all.save_depth 同一个理由。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as fh:
        fh.write(header.encode())
        fh.write(rec.tobytes())


def gt_stats(dtu_root: Path, scan: str, view: int, uv: np.ndarray, z: np.ndarray) -> dict:
    """只写进汇总表的质量统计: 点深度 vs 最近像素的 GT 深度。GT 缺失时返回空。"""
    f = dtu_root / "Depths_raw" / scan / f"depth_map_{view:04d}.pfm"
    if not f.is_file() or not len(z):
        return {}
    from data.io import read_pfm
    gt = read_pfm(str(f))
    u = np.clip(np.rint(uv[:, 0]).astype(int), 0, gt.shape[1] - 1)
    v = np.clip(np.rint(uv[:, 1]).astype(int), 0, gt.shape[0] - 1)
    g = gt[v, u]
    m = np.isfinite(g) & (g > 0)
    if not m.any():
        return {"gt_n": 0}
    e = np.abs(z[m] - g[m])
    return {"gt_n": int(m.sum()), "gt_med_err": float(np.median(e)),
            "gt_lt2": float((e < 2).mean()), "gt_gt10": float((e > 10).mean())}


# --------------------------------------------------------------------------- #
# 一个 scan
# --------------------------------------------------------------------------- #
def process_scan(job: tuple[str, list[int], dict]) -> list[dict]:
    scan, views_todo, opts = job
    import torch

    cv2.setNumThreads(opts["cv_threads"])
    torch.set_num_threads(max(1, opts["cv_threads"]))
    device = torch.device(opts["device"] if torch.cuda.is_available() else "cpu")
    dtu_root, out_root, light = Path(opts["dtu_root"]), Path(opts["out"]), opts["light"]
    pairs, K = {int(k): v for k, v in opts["pairs"].items()}, opts["neighbors"]
    t_scan = time.time()

    has_img = {v for v in pairs if image_path(dtu_root, scan, v, light).is_file()}
    nb = {v: pairs[v][:K] for v in pairs}
    partners = {r: sorted((set(nb[r]) | {s for s in pairs if r in nb[s]}) & has_img)
                for r in views_todo}
    edges = sorted({(min(r, s), max(r, s)) for r in views_todo for s in partners[r]})
    needed = sorted(set(views_todo) | {v for e in edges for v in e})

    sift = cv2.SIFT_create(nfeatures=opts["max_features"], contrastThreshold=opts["contrast_threshold"])
    clahe = cv2.createCLAHE(clipLimit=opts["clahe"], tileGridSize=(8, 8)) if opts["clahe"] > 0 else None
    cams, feats, rgbs = {}, {}, {}
    for v in needed:
        bgr = cv2.imread(str(image_path(dtu_root, scan, v, light)), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path(dtu_root, scan, v, light))
        feats[v] = extract_features(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), sift, clahe)
        cams[v] = read_cam(dtu_root, v)
        if v in views_todo:
            rgbs[v] = bgr[:, :, ::-1]
    t_feat = time.time() - t_scan

    obs: dict[int, list[dict]] = {r: [] for r in views_todo}
    edge_stat: dict[tuple[int, int], tuple[int, int]] = {}
    todo_set = set(views_todo)
    for i, j in edges:
        (xy1, d1), (xy2, d2) = feats[i], feats[j]
        ii, jj, st = match_with_retry(xy1, d1, xy2, d2, fundamental(cams[i], cams[j]),
                                      ratio=opts["ratio"], max_desc_dist=opts["max_desc_dist"],
                                      epi_thresh=opts["epi_thresh"], mutual=opts["mutual"],
                                      global_ratio=opts["global_ratio"], device=device)
        if len(ii) == 0:
            edge_stat[(i, j)] = (0, 0)
            continue
        tri = triangulate(xy1[ii], xy2[jj], cams[i], cams[j], opts)
        n_tri = len(tri["X"])
        edge_stat[(i, j)] = (len(ii), n_tri)
        if n_tri == 0:
            continue
        conf = point_conf(tri["err1"], tri["err2"], tri["ang"], len(ii))
        strong = st[tri["keep"]]
        err = np.maximum(tri["err1"], tri["err2"])
        for r, s, kp, z in ((i, j, ii[tri["keep"]], tri["z1"]), (j, i, jj[tri["keep"]], tri["z2"])):
            if r in todo_set and s in partners[r]:
                obs[r].append({"kp": kp, "z": z, "err": err, "ang": tri["ang"], "conf": conf,
                               "partner": np.full(n_tri, s, np.int32), "strong": strong})

    rows = []
    for r in views_todo:
        row = {"scan": scan, "view": r, "light": light, "status": "ok"}
        try:
            mg = merge_observations(obs[r], feats[r][0], opts["merge_tol"], opts["min_obs"])
            n = 0 if mg is None else len(mg["depth"])
            if n:
                uv = mg["uv"].astype(np.float32)
                depth = mg["depth"].astype(np.float32)
                xyz = backproject(uv.astype(np.float64), depth.astype(np.float64), cams[r])
                u = np.clip(np.rint(uv[:, 0]).astype(int), 0, NATIVE_W - 1)
                v = np.clip(np.rint(uv[:, 1]).astype(int), 0, NATIVE_H - 1)
                rgb = np.ascontiguousarray(rgbs[r][v, u])
            else:
                uv, depth = np.empty((0, 2), np.float32), np.empty(0, np.float32)
                xyz, rgb = np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
                mg = {k: np.empty(0) for k in ("n_obs", "strong", "conf", "reproj_err",
                                               "tri_angle", "used_partner")}
            ps = partners[r]
            used = np.bincount(mg["used_partner"].astype(int), minlength=max(pairs) + 1)
            pair_stats = np.array([[*edge_stat.get((min(r, s), max(r, s)), (0, 0)), used[s]]
                                   for s in ps], np.int32).reshape(-1, 3)
            dst = sfm_path(out_root, scan, r, light)
            atomic_savez(
                dst, uv=uv, depth=depth, xyz=xyz.astype(np.float32),
                n_obs=np.minimum(mg["n_obs"], 255).astype(np.uint8),
                strong=mg["strong"].astype(bool),
                conf=mg["conf"].astype(np.float32), reproj_err=mg["reproj_err"].astype(np.float32),
                tri_angle=mg["tri_angle"].astype(np.float32), rgb=rgb.astype(np.uint8),
                partners=np.asarray(ps, np.int32), pair_stats=pair_stats,
                scan=np.asarray(scan), view=np.asarray(r, np.int32), light=np.asarray(light, np.int32),
                image_hw=np.asarray([NATIVE_H, NATIVE_W], np.int32),
                params=np.asarray(json.dumps(opts["params"], sort_keys=True)))
            if opts["save_ply"] and n:
                write_ply(dst.with_suffix(".ply"), xyz.astype(np.float32), rgb)
            row.update(n_points=n, n_multi=int((mg["n_obs"] >= 2).sum()),
                       n_partners=len(ps), n_kp=len(feats[r][0]),
                       med_depth=float(np.median(depth)) if n else float("nan"))
            if opts["gt_eval"]:
                row.update(gt_stats(dtu_root, scan, r, uv, depth))
        except Exception as exc:  # noqa: BLE001 - 单个视角坏了不能带崩整个 scan
            row.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    dt = time.time() - t_scan
    for row in rows:
        row.update(scan_time_s=round(dt, 1), feat_time_s=round(t_feat, 1), n_edges=len(edges))
    return rows


def _safe_process_scan(job):
    try:
        return process_scan(job)
    except Exception as exc:  # noqa: BLE001
        scan, views, opts = job
        return [{"scan": scan, "view": v, "light": opts["light"], "status": "fail",
                 "error": f"{type(exc).__name__}: {exc}"} for v in views]


def _fmt(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


SUMMARY_COLS = ["scan", "view", "light", "status", "n_points", "n_multi", "n_kp", "n_partners",
                "med_depth", "gt_n", "gt_med_err", "gt_lt2", "gt_gt10", "n_edges",
                "feat_time_s", "scan_time_s", "error"]


def main() -> None:
    args = parse_args()
    cfg = build_mvs_config()
    dtu_root = Path(args.dtu_root) if args.dtu_root else cfg.paths.dtu_train_root
    out_root = Path(args.out) if args.out else cfg.paths.sfm_sparse_cache_path

    scans = [f"scan{s}" for s in args.scans] if args.scans else scan_list(dtu_root)
    missing = [s for s in scans if not (dtu_root / "Rectified_raw" / s).is_dir()]
    if missing:
        raise SystemExit(f"--scans 里这些在 Rectified_raw/ 下找不到: {missing}")
    pairs = read_pairs(dtu_root)
    views = args.views if args.views is not None else sorted(pairs)

    jobs, n_done, n_noimg = [], 0, 0
    for scan in scans:
        todo = []
        for v in views:
            if not image_path(dtu_root, scan, v, args.light).is_file():
                n_noimg += 1
            elif sfm_path(out_root, scan, v, args.light).is_file() and not args.force:
                n_done += 1
            else:
                todo.append(v)
        if todo:
            jobs.append((scan, todo))
    n_todo = sum(len(v) for _, v in jobs)
    print(f"[sfm-cache] dtu_root={dtu_root}  out={out_root}")
    print(f"[sfm-cache] scans={len(scans)}  views/scan={len(views)}  light={args.light}  "
          f"neighbors={args.neighbors}  已完成 {n_done}  待建 {n_todo} (涉及 {len(jobs)} 个 scan)  "
          f"缺图跳过 {n_noimg}")
    if args.dry_run or not jobs:
        print("[sfm-cache] dry-run 或无待办, 退出。")
        return

    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 1
    workers = max(1, min(args.workers, len(jobs)))
    params = {k: getattr(args, k) for k in ("light", "neighbors", "max_features", "contrast_threshold",
                                            "clahe", "ratio", "max_desc_dist", "epi_thresh",
                                            "reproj_thresh", "min_tri_angle", "depth_margin",
                                            "merge_tol", "global_ratio", "min_obs")}
    params["mutual"] = not args.no_mutual
    opts = {**params, "dtu_root": str(dtu_root), "out": str(out_root), "device": args.device,
            "cv_threads": args.cv_threads or max(1, n_cpu // workers),
            "pairs": {str(k): v for k, v in pairs.items()}, "save_ply": args.save_ply,
            "gt_eval": not args.no_gt_eval, "params": params}
    full_jobs = [(scan, todo, opts) for scan, todo in jobs]
    print(f"[sfm-cache] workers={workers}  cv_threads/worker={opts['cv_threads']}  device={args.device}")

    out_root.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    summary_csv = out_root / f"_sfm_summary_{stamp}.csv"
    fail_csv = out_root / f"_sfm_failures_{stamp}.csv"
    t0 = time.time()
    n_ok = n_fail = done_views = 0
    all_pts = []
    with summary_csv.open("w", newline="") as fs, fail_csv.open("w", newline="") as ff:
        ws = csv.DictWriter(fs, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        wf = csv.writer(ff)
        ws.writeheader()
        wf.writerow(["scan", "view", "light", "error"])
        if workers == 1:
            it = map(_safe_process_scan, full_jobs)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(workers)
            it = pool.imap_unordered(_safe_process_scan, full_jobs)
        try:
            for k, rows in enumerate(it, 1):
                for row in rows:
                    ws.writerow(row)
                    if row["status"] == "ok":
                        n_ok += 1
                        all_pts.append(row.get("n_points", 0))
                    else:
                        n_fail += 1
                        wf.writerow([row["scan"], row["view"], row["light"], row.get("error", "")])
                        print(f"    !! {row['scan']} v{row['view']}: {row.get('error')}", flush=True)
                fs.flush()
                ff.flush()
                done_views += len(rows)
                pts = [r.get("n_points", 0) for r in rows if r["status"] == "ok"]
                gte = [r["gt_med_err"] for r in rows if "gt_med_err" in r]
                el = time.time() - t0
                eta = el / done_views * (n_todo - done_views)
                print(f"[sfm-cache] {k}/{len(jobs)} {rows[0]['scan']:>8s}  "
                      f"点数中位 {np.median(pts) if pts else 0:.0f} (最少 {min(pts) if pts else 0})  "
                      f"GT偏差中位 {np.median(gte) if gte else float('nan'):.2f}mm  "
                      f"本 scan {rows[0].get('scan_time_s', 0)}s  用时 {_fmt(el)}  "
                      f"预计剩余 {_fmt(eta)}", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    print(f"[sfm-cache] 完成: ok={n_ok} fail={n_fail}  总用时 {_fmt(time.time() - t0)}")
    if all_pts:
        a = np.asarray(all_pts)
        print(f"[sfm-cache] 每视角点数: 中位 {np.median(a):.0f}  p10 {np.percentile(a, 10):.0f}  "
              f"最少 {a.min()}  <100 的视角 {int((a < 100).sum())}")
    print(f"[sfm-cache] 汇总表: {summary_csv}")
    if n_fail:
        print(f"[sfm-cache] 失败清单: {fail_csv}")
    else:
        fail_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
