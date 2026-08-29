"""思路三: 低频残差场 prior —— DA3 决定表面形状, VGGT 只提供低频校正。

## 为什么换掉 norm_fill 的 Poisson 填充

旧路径 (``norm_fill.fill_vggt_depth_by_da3_normals``) 把 VGGT 的稀疏深度当硬真值:
``anchor_weight=100`` 加 ``hard_keep_sparse=True`` 意味着 anchor 像素被逐字节写回
输出 —— VGGT 点云在同一个物理表面上的高频抖动 (坑坑洼洼) 因此**没有任何被平滑
的机会**, 而且法向边在 ``edge_ratio_limits`` 之外被丢弃时方程欠定, 只剩
``guide_weight=0.2`` 撑着, 边缘处会反向填充。

这里改成: VGGT 只能通过一个**低频、低自由度**的校正场影响输出。

    D_0 = a_0 * D_A + b_0                       全局鲁棒 affine (depth 域)
    e_i = (1/D_V(i) - 1/D_0(i)) / m_rho         逆深度域归一化残差 (只在锚点上)
    R   = argmin  sum_i w_i huber(B_i R - e_i)  1/8 分辨率控制网格
                + lam_s * 边缘感知平滑
                + lam_c * 各向异性曲率
                + lam_0 * 无支撑处拉回 0
    rho_R = rho_0 + m_rho * R^up                双线性上采样 (R 已带限, 近似无损)
    D_R   = 1 / rho_R                           正性 + 幅度守卫

高频带 100% 来自 DA3, VGGT 的高频噪声在表达能力上就进不来。无锚点区域 R -> 0,
严格退回全局对齐的 DA3, 不会像旧路径那样从远处外推出一个错误平面。

## 域的选择 (为什么全局在 depth 域、残差场在 rho 域)

DA3MONO 直接输出 relative **depth**, 所以全局仿射留在 depth 域 (``affine_domain``
可切到 "rho" 做消融)。残差场则放到归一化逆深度域:

  * 针孔相机下三维平面的 1/D 关于像素坐标是线性的, 在 rho 域加低频量比在 depth
    域更不容易把平面弄弯 (但注意: 任意低频 rho 残差仍会弯曲平面, 所以有 lam_c
    的曲率项);
  * 除以 m_rho = median(rho_0) 之后 e 是无量纲的, huber 阈值和 lam_* 不再依赖
    VGGT 的任意尺度 —— 否则同一组超参在不同 scan 上行为不同。

## 多视角锚点

VGGT 的 ``world_points`` 是 [V,H,W,3], 旧路径只用了 ``points[0]``。这里把全部 V
个视角投到参考相机: 锚点密度翻 V 倍 (低频场方差直接下降), 而且**按来源视角划分
fit / held-out** 之后, held-out 残差才是真正独立于 fit 锚点的证据 —— 同一次前向
同一个视角的随机 70% 与 fit 的 30% 在整片相关错误区域上完全一致, 那种划分只能
测采样过拟合, 测不出正确性。

被参考视角遮挡的其他视角点用 D_0 的第一遍估计做剔除 (见 ``_occlusion_cull``)。

## 已知取舍

  * 上采样用双线性而不是边缘感知滤波: R 由构造就带限到 1/8, 且跨表面耦合已经在
    求解时被 G_qr 挡掉了; 再加一个未经验证的高频算子到"我们正想保持干净的输出"
    上, 风险大于收益。边界处最多 s/2 像素的涂抹。
  * 支撑门控 alpha **不乘到深度上**。lam_0 已经在求解内部把无支撑处拉回 0; alpha
    由锚点密度/局部残差构成, 空间频率和锚点一样高, 乘到 R 上等于把高频从门控
    这条路又放回输出。alpha 只进 confidence。
  * 鲁棒损失杀孤立坑点, 杀不掉整片一致偏移的 VGGT 错误区 —— 那只能靠跨视角
    一致性 (``w_view``) 和 held-out 检查, 两者都在这里, 但都不是完备的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

from models.conf import camera_rays


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResidualFieldConfig:
    # --- 锚点采集 ---
    conf_percentile: float = 10.0     # 逐视角丢掉最低的这个百分比 (与旧路径一致)
    multiview: bool = True            # False = 只用参考视角的点 (旧行为, 做消融用)
    voxel_auto_scale: float = 1.3     # voxel = scale * 最近邻距离中位数
    voxel_auto_sample: int = 20000
    occl_rel_tol: float = 0.06        # 非参考视角点比 D_0 远这个比例就当被遮挡剔掉
    min_anchors: int = 200
    # 控制网格只有 ~7.6k 个未知数, 5 视角 798x602 去重后能剩 100 万个锚点 ——
    # 上百倍的超定对低频场毫无额外信息, 只是让分层 FPS 的 python 内循环跑上百万
    # 次。按 cell 均匀抽到这个预算 (保持空间和视角分布), 单样本约 2s。
    max_anchors: int = 600_000

    # --- 全局 affine ---
    affine_domain: str = "depth"      # "depth" | "rho"
    affine_cell: int = 16             # 权重按这个像素格归一化, 防纹理密集区支配
    affine_huber_k: float = 1.345     # 以 MAD sigma 为单位
    affine_iters: int = 6

    # --- 表面分割 / 边缘 ---
    edge_quantile: float = 0.88       # 高于这个分位算边缘 (分割的屏障)
    seg_min_pixels: int = 64          # 小于这个的连通块并回 "未分割" (用全局统计)
    tau_edge: float = 0.35            # G_qr = exp(-E/tau_edge)

    # --- 锚点鲁棒门 ---
    tukey_c: float = 4.685            # 以 MAD sigma 为单位
    sigma_floor: float = 2e-3         # 归一化 rho 单位; 防止极平坦区 sigma->0

    # --- 残差场 ---
    grid_stride: int = 8              # 控制网格 = 1/8 分辨率
    fit_frac: float = 0.30            # 分层 FPS 取这个比例做拟合, 其余 held-out
    fps_cell: int = 32                # 分层单元 (像素); FPS 在 (segment, cell) 内做
    huber_delta: float = 0.02         # 归一化 rho 单位
    lambda_s: float = 0.10            # 边缘感知平滑 (数据项已按节点归一化)
    lambda_c: float = 0.02            # 各向异性曲率 (用同一套 G 做各向异性!)
    lambda_0: float = 0.05            # 无支撑处拉回 0
    ridge: float = 1e-6
    irls_iters: int = 4
    cg_maxiter: int = 500
    cg_rtol: float = 1e-7
    support_n0: float = 1.0           # S = 1 - exp(-n_eff/n0); 只压真正没锚点的节点

    # --- 输出守卫 ---
    clip_quantile: float = 99.0       # |R| 不超过保留锚点 |e| 的这个分位
    min_rho_ratio: float = 0.25       # rho_R 必须落在 [ratio, 1/ratio] * rho_0


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _robust_norm01(x: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    """按分位数把 x 线性压到 [0,1]。空/退化输入返回全 0.5。"""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.full_like(np.asarray(x, np.float32), 0.5)
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if not np.isfinite(hi - lo) or hi - lo < 1e-12:
        return np.full_like(np.asarray(x, np.float32), 0.5)
    return np.clip((np.asarray(x, np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _mad_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def _voxel_dedup_indices(points: np.ndarray, score: np.ndarray, voxel: float) -> np.ndarray:
    """每个体素保留 score 最大的那个点, 返回它在 ``points`` 里的下标。

    和 norm_fill.voxel_dedup_pointcloud 同一套逻辑, 但返回**下标**, 这样调用方
    能把 conf / view_id / 像素坐标一起带过去。
    """
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    origin = points.min(axis=0)
    vox = np.floor((points - origin) / max(voxel, 1e-8)).astype(np.int64)
    # np.unique(axis=0) 在 100 万点上要 0.7s (它内部走 void view + 排序)。
    # 体素坐标是有界非负整数, 压成一个 int64 key 之后一维 unique 快一个量级。
    ext = vox.max(axis=0) + 1
    if float(ext[0]) * float(ext[1]) * float(ext[2]) < 9.2e18:
        key = (vox[:, 0] * ext[1] + vox[:, 1]) * ext[2] + vox[:, 2]
        _, inverse = np.unique(key, return_inverse=True)
    else:                                   # 极端体素尺度下回退到原路径
        _, inverse = np.unique(vox, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    order = np.lexsort((score, inverse))
    inv_sorted = inverse[order]
    last = np.ones(len(order), dtype=bool)
    last[:-1] = inv_sorted[1:] != inv_sorted[:-1]
    return order[last]


def _estimate_voxel_size(points: np.ndarray, scale: float, sample: int) -> float:
    from scipy.spatial import cKDTree

    n = len(points)
    if n <= 1:
        return 1.0
    idx = np.arange(n) if n <= sample else np.random.default_rng(0).choice(n, sample, replace=False)
    tree = cKDTree(points)
    dists, _ = tree.query(points[idx], k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return max(float(np.median(nn)) * scale, 1e-8) if nn.size else 1.0


def _group_median(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按 labels 分组求中位数和 MAD sigma, 返回与 x 等长的两个数组。

    用 ``ndimage.median`` (C 实现) 而不是 python 层按组切片: 一张图能切出上万个
    表面, 逐组两次 np.median 就是几万次调用。
    """
    uniq, inv = np.unique(labels, return_inverse=True)
    med = np.asarray(ndimage.median(x, labels=inv, index=np.arange(len(uniq))), np.float64)
    med_at = med[inv]
    mad = np.asarray(ndimage.median(np.abs(x - med_at), labels=inv,
                                    index=np.arange(len(uniq))), np.float64)
    return med_at, 1.4826 * mad[inv]


def _fps_2d(xy: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """二维最远点采样, 返回选中的下标。O(k*n), 只在分层单元内部调用。"""
    n = len(xy)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    sel = np.empty(k, dtype=np.int64)
    x = np.ascontiguousarray(xy[:, 0]); y = np.ascontiguousarray(xy[:, 1])
    # 起点取离质心最近的点 —— 用随机起点会让同一份输入两次跑出不同锚点集
    sel[0] = int(np.argmin((x - x.mean()) ** 2 + (y - y.mean()) ** 2))
    dx = x - x[sel[0]]; dy = y - y[sel[0]]
    dist = dx * dx + dy * dy
    tmp = np.empty_like(dist)
    for i in range(1, k):
        j = int(np.argmax(dist))
        sel[i] = j
        np.subtract(x, x[j], out=tmp); np.multiply(tmp, tmp, out=tmp)
        np.subtract(y, y[j], out=dx);  np.multiply(dx, dx, out=dx)
        np.add(tmp, dx, out=tmp)
        np.minimum(dist, tmp, out=dist)
    return sel


# ---------------------------------------------------------------------------
# 边缘与表面分割
# ---------------------------------------------------------------------------
def edge_strength(normal_a: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
    """E in [0,1]: DA3 法向边与 RGB 边取大。用于 G_qr、锚点降权和分割屏障。"""
    n = np.asarray(normal_a, np.float32)
    gx = np.linalg.norm(np.gradient(n, axis=1), axis=-1)
    gy = np.linalg.norm(np.gradient(n, axis=0), axis=-1)
    e_n = _robust_norm01(np.hypot(gx, gy), 50.0, 95.0)
    if rgb is None:
        return e_n
    g = np.asarray(rgb, np.float32).mean(axis=-1) / 255.0
    e_rgb = _robust_norm01(np.hypot(ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0)),
                           50.0, 95.0)
    return np.maximum(e_n, e_rgb).astype(np.float32)


def segment_surfaces(edge: np.ndarray, cfg: ResidualFieldConfig) -> np.ndarray:
    """把非边缘区域的连通块当作近似独立表面。label 0 = 未分割 (走全局统计)。

    这里只用 scipy —— 装 skimage/ximgproc 的超像素会引入一个环境依赖, 而这一步
    的作用仅仅是给 MAD/Tukey 分个组, 精度要求不高。
    """
    thr = float(np.percentile(edge, cfg.edge_quantile * 100.0))
    lab, n = ndimage.label(edge < thr)
    if n == 0:
        return np.zeros_like(lab, dtype=np.int32)
    sizes = np.bincount(lab.reshape(-1), minlength=n + 1)
    small = np.flatnonzero(sizes < cfg.seg_min_pixels)
    if small.size:
        remap = np.arange(n + 1, dtype=np.int32)
        remap[small] = 0
        lab = remap[lab]
    return lab.astype(np.int32)


# ---------------------------------------------------------------------------
# 锚点采集
# ---------------------------------------------------------------------------
def collect_anchors(points_world: np.ndarray, conf: np.ndarray, K_ref: np.ndarray,
                    E_ref: np.ndarray, target_wh: tuple[int, int],
                    cfg: ResidualFieldConfig) -> dict[str, np.ndarray]:
    """把 [V,H,W,3] 的 VGGT 世界点投到参考相机, 返回锚点的稀疏列表。

    每个视角**单独**做 conf 过滤 + 体素去重 (体素尺度也逐视角估), 这样 view_id
    保持完整, 后面才能按来源视角划 fit / held-out。
    """
    W, H = target_wh
    pts = np.asarray(points_world, np.float32)
    cf = np.asarray(conf, np.float32)
    if pts.ndim == 3:                       # [H,W,3] -> [1,H,W,3]
        pts, cf = pts[None], cf[None]
    V = pts.shape[0] if cfg.multiview else 1

    R = np.asarray(E_ref, np.float64)[:3, :3]
    t = np.asarray(E_ref, np.float64)[:3, 3]
    fx, fy = float(K_ref[0, 0]), float(K_ref[1, 1])
    cx, cy = float(K_ref[0, 2]), float(K_ref[1, 2])

    us, vs, ds, cs, vids = [], [], [], [], []
    for v in range(V):
        p = pts[v].reshape(-1, 3)
        c = cf[v].reshape(-1)
        ok = np.isfinite(p).all(axis=1) & np.isfinite(c)
        p, c = p[ok], c[ok]
        if p.shape[0] < 8:
            continue
        keep = c >= float(np.percentile(c, cfg.conf_percentile))
        p, c = p[keep], c[keep]
        if p.shape[0] < 8:
            continue
        vox = _estimate_voxel_size(p, cfg.voxel_auto_scale, cfg.voxel_auto_sample)
        idx = _voxel_dedup_indices(p, c, vox)
        p, c = p[idx], c[idx]

        pc = (R @ p.T.astype(np.float64)).T + t
        Z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * pc[:, 0] / Z + cx
            vv = fy * pc[:, 1] / Z + cy
        m = (Z > 1e-6) & np.isfinite(u) & np.isfinite(vv)
        ui = np.round(u[m]).astype(np.int32)
        vi = np.round(vv[m]).astype(np.int32)
        m2 = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not m2.any():
            continue
        us.append(ui[m2]); vs.append(vi[m2])
        ds.append(Z[m][m2].astype(np.float32))
        cs.append(c[m][m2].astype(np.float32))
        vids.append(np.full(int(m2.sum()), v, dtype=np.int16))

    if not us:
        raise ValueError("collect_anchors: 没有任何 VGGT 点投进参考视角")
    out = {"u": np.concatenate(us), "v": np.concatenate(vs),
           "d": np.concatenate(ds), "conf": np.concatenate(cs),
           "view": np.concatenate(vids)}
    return _cap_anchors(out, (H, W), cfg)


def _cap_anchors(a: dict[str, np.ndarray], shape_hw: tuple[int, int],
                 cfg: ResidualFieldConfig) -> dict[str, np.ndarray]:
    """按 (16px 格, 来源视角) 均匀抽稀到 ``max_anchors``, 保持空间与视角分布。"""
    n = len(a["u"])
    if n <= cfg.max_anchors:
        return a
    h, w = shape_hw
    cell = 16
    key = ((a["v"] // cell).astype(np.int64) * (w // cell + 2)
           + (a["u"] // cell)) * 64 + a["view"].astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    rank = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    frac = cfg.max_anchors / float(n)
    grp_n = np.repeat(np.diff(np.r_[starts, n]), np.diff(np.r_[starts, n]))
    keep_k = np.maximum(1, np.round(frac * grp_n).astype(np.int64))
    sel = order[rank < keep_k]
    return {k: v[sel] for k, v in a.items()}


def _cross_view_weight(a: dict[str, np.ndarray], shape_hw: tuple[int, int],
                       cell: int = 4, tau: float = 0.05) -> np.ndarray:
    """同一个 cell 内不同来源视角的逆深度是否一致 —— 唯一能戳穿"整片一致偏移"的量。

    只有一个视角贡献时返回 1.0 (中性): 系统性地给单视角区域降权会让所有只被参考
    视角看到的地方 (物体边缘之外的大片区域) 一律低权, 那不是我们想表达的意思。
    """
    h, w = shape_hw
    key = (a["v"] // cell).astype(np.int64) * (w // cell + 1) + (a["u"] // cell)
    rho = 1.0 / np.maximum(a["d"].astype(np.float64), 1e-6)
    uniq, inv = np.unique(key, return_inverse=True)
    idx = np.arange(len(uniq))
    med = np.asarray(ndimage.median(rho, labels=inv, index=idx), np.float64)[inv]
    # 只有多个来源视角同时落进同一个 cell 时这条约束才有意义: 单视角 cell 的
    # "离散度"只反映该视角自己的噪声, 拿它降权等于惩罚覆盖稀疏的区域。
    n_view = np.asarray(ndimage.labeled_comprehension(
        a["view"].astype(np.int64), inv, idx,
        lambda z: np.unique(z).size, np.int64, 1))[inv]
    dev = np.abs(rho - med) / np.maximum(np.abs(med), 1e-9)
    out = np.exp(-((dev / tau) ** 2)).astype(np.float32)
    return np.where(n_view >= 2, out, np.float32(1.0))


# ---------------------------------------------------------------------------
# 全局鲁棒 affine
# ---------------------------------------------------------------------------
def robust_global_affine(x: np.ndarray, y: np.ndarray, w: np.ndarray,
                         cell_id: np.ndarray, cfg: ResidualFieldConfig
                         ) -> tuple[float, float, dict[str, Any]]:
    """拟合 y ~ a*x + b, IRLS + Huber, 权重按 cell 归一化。

    cell 归一化是为了防止纹理丰富/近处的区域因为点多而支配这两个参数 —— 这是
    FPS 想解决的真问题, 但用归一化解决不必扔掉 70% 的点。
    """
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    w = np.asarray(w, np.float64)
    cnt = np.bincount(cell_id, weights=w, minlength=int(cell_id.max()) + 1)
    wbar = w / np.maximum(cnt[cell_id], 1e-12)

    a, b = 1.0, 0.0
    A = np.stack([x, np.ones_like(x)], axis=1)
    ww = wbar.copy()
    sigma = 0.0
    for _ in range(cfg.affine_iters):
        AtW = A.T * ww
        try:
            a, b = np.linalg.solve(AtW @ A, AtW @ y)
        except np.linalg.LinAlgError:
            break
        r = y - (a * x + b)
        sigma = _mad_sigma(r)
        if sigma <= 0:
            break
        u = np.abs(r) / (cfg.affine_huber_k * sigma)
        ww = wbar * np.minimum(1.0, 1.0 / np.maximum(u, 1e-9))
    r = y - (a * x + b)
    return float(a), float(b), {"sigma": float(sigma),
                                "n": int(len(x)),
                                "inlier_frac": float(np.mean(np.abs(r) <= 3 * sigma))
                                if sigma > 0 else 0.0}


def _occlusion_cull(a: dict[str, np.ndarray], d0: np.ndarray,
                    cfg: ResidualFieldConfig) -> np.ndarray:
    """非参考视角的点若明显在 D_0 之后, 判为被参考视角遮挡, 剔除。

    用 DA3 的稠密光滑表面当遮挡参考, 而不是 VGGT 自己的 z-buffer —— 后者是逐点
    栅格化的, 一个虚假近点就能把它后面整列真点全判成遮挡。
    """
    ref = d0[a["v"], a["u"]]
    ok = np.isfinite(ref) & (ref > 0)
    keep = np.ones(len(ref), dtype=bool)
    other = a["view"] != 0
    keep[other & ok] = a["d"][other & ok] <= ref[other & ok] * (1.0 + cfg.occl_rel_tol)
    return keep


# ---------------------------------------------------------------------------
# 网格算子
# ---------------------------------------------------------------------------
def _grid_shape(h: int, w: int, s: int) -> tuple[int, int]:
    return (h + s - 1) // s, (w + s - 1) // s


def _bilinear_operator(u: np.ndarray, v: np.ndarray, h: int, w: int, s: int
                       ) -> sparse.csr_matrix:
    """锚点 (u,v) 处对控制网格的双线性采样算子 B, 形状 [n_anchor, Hg*Wg]。

    节点 j 的像素中心取 j*s + (s-1)/2, 上采样用同一套坐标, 两边严格一致。
    """
    hg, wg = _grid_shape(h, w, s)
    gx = (u.astype(np.float64) - (s - 1) / 2.0) / s
    gy = (v.astype(np.float64) - (s - 1) / 2.0) / s
    gx = np.clip(gx, 0, wg - 1); gy = np.clip(gy, 0, hg - 1)
    x0 = np.floor(gx).astype(np.int64); y0 = np.floor(gy).astype(np.int64)
    x1 = np.minimum(x0 + 1, wg - 1);    y1 = np.minimum(y0 + 1, hg - 1)
    tx = gx - x0; ty = gy - y0
    n = len(u)
    rows = np.repeat(np.arange(n, dtype=np.int64), 4)
    cols = np.stack([y0 * wg + x0, y0 * wg + x1, y1 * wg + x0, y1 * wg + x1], 1).reshape(-1)
    vals = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty),
                     (1 - tx) * ty, tx * ty], 1).reshape(-1)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, hg * wg))


def _upsample_grid(R: np.ndarray, h: int, w: int, s: int) -> np.ndarray:
    """和 ``_bilinear_operator`` 同坐标约定的双线性上采样。"""
    hg, wg = R.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64), indexing="ij")
    gy = np.clip((yy - (s - 1) / 2.0) / s, 0, hg - 1)
    gx = np.clip((xx - (s - 1) / 2.0) / s, 0, wg - 1)
    return ndimage.map_coordinates(R, [gy, gx], order=1, mode="nearest").astype(np.float32)


def _grid_edge_terms(edge_g: np.ndarray, cfg: ResidualFieldConfig
                     ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """返回 (平滑算子 S, 各向异性 Laplacian 算子 L)。

    **曲率项必须用同一套 G 做各向异性**: 无差别的 Laplacian 会跨表面边界惩罚
    R 的跳变, 把 lam_s 靠 G_qr 刻意做出的解耦重新抹平 —— 两个正则项在边界上
    直接对打, 跨表面传播会从曲率项漏进来。
    """
    hg, wg = edge_g.shape
    idx = np.arange(hg * wg, dtype=np.int64).reshape(hg, wg)
    pairs = [(idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)),
             (idx[:-1, :].reshape(-1), idx[1:, :].reshape(-1))]
    e_flat = edge_g.reshape(-1)

    rows, cols, vals, row = [], [], [], 0
    g_acc = np.zeros(hg * wg, dtype=np.float64)                 # sum_r G_qr
    lrows, lcols, lvals = [], [], []
    for qa, qb in pairs:
        g = np.exp(-0.5 * (e_flat[qa] + e_flat[qb]) / cfg.tau_edge)
        sw = np.sqrt(g)
        rows.append(np.repeat(np.arange(row, row + len(qa), dtype=np.int64), 2))
        cols.append(np.stack([qa, qb], 1).reshape(-1))
        vals.append(np.stack([sw, -sw], 1).reshape(-1))
        row += len(qa)
        np.add.at(g_acc, qa, g); np.add.at(g_acc, qb, g)
        for a_, b_ in ((qa, qb), (qb, qa)):
            lrows.append(a_); lcols.append(b_); lvals.append(g)
    S = sparse.csr_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(row, hg * wg))

    # L = D^-1 * (W - diag(rowsum))  —— 行归一化的各向异性 Laplacian
    inv = 1.0 / np.maximum(g_acc, 1e-9)
    lr = np.concatenate(lrows); lc = np.concatenate(lcols); lv = np.concatenate(lvals)
    L = sparse.csr_matrix((lv * inv[lr], (lr, lc)), shape=(hg * wg, hg * wg))
    L = L - sparse.diags(np.asarray(L.sum(axis=1)).reshape(-1))
    # 孤立节点 (周围全是强边) 的行整体为 0, 不参与曲率约束
    return S, L.tocsr()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def depth_to_camera_normals(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """相机系法向, 统一朝向相机。与 norm_fill 同一套实现 (放这里避免循环 import)。"""
    rays = camera_rays(depth.shape, intrinsic)
    pts = rays * np.asarray(depth, np.float32)[..., None]
    n = np.cross(np.gradient(pts, axis=1), np.gradient(pts, axis=0))
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    bad = (~np.isfinite(norm).squeeze(-1)) | (norm.squeeze(-1) < 1e-8)
    n = n / np.maximum(norm, 1e-8)
    n[np.sum(n * rays, axis=-1) > 0] *= -1.0
    n[bad] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return n.astype(np.float32)


def build_residual_field_prior(
    points_world: np.ndarray,
    conf: np.ndarray,
    depth_da3: np.ndarray,
    K_ref: np.ndarray,
    E_ref: np.ndarray,
    rgb: np.ndarray | None,
    target_wh: tuple[int, int],
    config: ResidualFieldConfig | None = None,
) -> dict[str, Any]:
    """思路三主入口。返回 ``depth`` (未标尺) 和给 confidence 用的一堆诊断量。"""
    cfg = config or ResidualFieldConfig()
    W, H = target_wh
    dA = np.asarray(depth_da3, np.float32)
    if dA.shape != (H, W):
        raise ValueError(f"depth_da3 shape {dA.shape} != target {(H, W)}")

    # --- 1) 锚点 ---
    a = collect_anchors(points_world, conf, K_ref, E_ref, target_wh, cfg)
    if len(a["d"]) < cfg.min_anchors:
        raise ValueError(f"锚点太少: {len(a['d'])} < {cfg.min_anchors}")

    dA_at = dA[a["v"], a["u"]].astype(np.float64)
    finite = np.isfinite(dA_at) & (dA_at > 0)
    a = {k: val[finite] for k, val in a.items()}
    dA_at = dA_at[finite]

    w_conf = _robust_norm01(a["conf"]).astype(np.float64) * 0.9 + 0.1
    w_view = _cross_view_weight(a, (H, W)).astype(np.float64)
    cell_id = ((a["v"] // cfg.affine_cell).astype(np.int64) * (W // cfg.affine_cell + 1)
               + (a["u"] // cfg.affine_cell)).astype(np.int64)

    # --- 2) 全局 affine, 两遍 ---
    # 第一遍只用参考视角的点、只用 conf/跨视角权重: 此时还没有 D_0, 也就没有可信
    # 的法向边 (DA3 的原始 relative depth 单位任意, 直接求法向的朝向是错的)。第
    # 一遍的 D_0 用来 (a) 算法向 -> 边缘 -> 分割, (b) 当遮挡参考剔除其他视角的点。
    def _fit(mask: np.ndarray, w: np.ndarray) -> tuple[float, float, dict]:
        if cfg.affine_domain == "rho":
            xs = 1.0 / np.maximum(dA_at[mask], 1e-6)
            ys = 1.0 / np.maximum(a["d"][mask].astype(np.float64), 1e-6)
        else:
            xs, ys = dA_at[mask], a["d"][mask].astype(np.float64)
        return robust_global_affine(xs, ys, w[mask], cell_id[mask], cfg)

    w_pass1 = w_conf * w_view
    ref_only = a["view"] == 0
    if int(ref_only.sum()) < cfg.min_anchors:
        ref_only = np.ones(len(dA_at), dtype=bool)
    a1, b1, _ = _fit(ref_only, w_pass1)
    d0_pass1 = _apply_affine(dA, a1, b1, cfg.affine_domain)
    if not np.isfinite(d0_pass1).all() or float(np.nanmin(d0_pass1)) <= 0:
        raise ValueError(f"第一遍 affine 产生非正深度 (a={a1:.4g}, b={b1:.4g})")

    normal_a = depth_to_camera_normals(d0_pass1.astype(np.float32), K_ref)
    edge = edge_strength(normal_a, rgb)
    seg = segment_surfaces(edge, cfg)

    w_edge = (1.0 - edge[a["v"], a["u"]]).astype(np.float64) * 0.9 + 0.1
    w_raw = w_conf * w_view * w_edge

    keep = _occlusion_cull(a, d0_pass1, cfg)
    if int(keep.sum()) >= cfg.min_anchors:
        a = {k: val[keep] for k, val in a.items()}
        dA_at, w_raw, cell_id = dA_at[keep], w_raw[keep], cell_id[keep]
    n_occluded = int((~keep).sum())

    a0, b0, aff_info = _fit(np.ones(len(dA_at), dtype=bool), w_raw)
    D0 = _apply_affine(dA, a0, b0, cfg.affine_domain)
    if not np.isfinite(D0).all() or float(np.nanmin(D0)) <= 0:
        raise ValueError(f"全局 affine 产生非正深度 (a={a0:.4g}, b={b0:.4g})")

    # --- 3) 归一化逆深度残差 + 表面内 Tukey ---
    rho0 = 1.0 / np.maximum(D0, 1e-6)
    m_rho = float(np.median(rho0[np.isfinite(rho0)]))
    rho0_at = rho0[a["v"], a["u"]].astype(np.float64)
    rhoV_at = 1.0 / np.maximum(a["d"].astype(np.float64), 1e-6)
    e = (rhoV_at - rho0_at) / max(m_rho, 1e-12)

    seg_at = seg[a["v"], a["u"]]
    w_tukey = _segment_tukey(e, seg_at, cfg)
    w_hat = w_raw * w_tukey
    alive = w_hat > 1e-6
    if int(alive.sum()) < cfg.min_anchors:
        raise ValueError(f"Tukey 之后存活锚点太少: {int(alive.sum())}")

    # --- 4) 分层 FPS: 30% 拟合, 70% held-out (多视角时按来源视角切) ---
    fit_mask = _stratified_split(a, seg_at, alive, cfg)

    # --- 5) 低频残差场 ---
    R_grid, solve_info = _solve_field(a, e, w_hat, fit_mask, edge, (H, W), cfg)
    R_up = _upsample_grid(R_grid, H, W, cfg.grid_stride)

    lim = float(np.percentile(np.abs(e[fit_mask & alive]), cfg.clip_quantile)) \
        if int((fit_mask & alive).sum()) else 0.0
    R_up = np.clip(R_up, -lim, lim)

    rho_R = rho0 + m_rho * R_up
    lo = cfg.min_rho_ratio * rho0
    hi = rho0 / cfg.min_rho_ratio
    n_clamped = int(np.sum((rho_R < lo) | (rho_R > hi)))
    rho_R = np.clip(rho_R, lo, hi)
    D_R = (1.0 / np.maximum(rho_R, 1e-9)).astype(np.float32)

    # --- 6) 诊断量 (给 confidence) ---
    diag = _diagnostics(a, e, w_hat, fit_mask, alive, R_up, edge, (H, W), cfg)
    diag.update({
        "affine_a": float(a0), "affine_b": float(b0),
        "affine_domain": cfg.affine_domain,
        "affine_sigma": aff_info["sigma"], "affine_inlier": aff_info["inlier_frac"],
        "m_rho": m_rho, "R_limit": lim,
        "n_anchor": int(len(dA_at)), "n_fit": int(fit_mask.sum()),
        "n_holdout": int((~fit_mask & alive).sum()),
        "n_occluded": n_occluded,
        "n_views": int(len(np.unique(a["view"]))),
        "tukey_kept": float(np.mean(w_tukey > 1e-6)),
        "clamped_frac": float(n_clamped) / float(H * W),
        **solve_info,
    })
    return {"depth": D_R, "depth_global": D0.astype(np.float32),
            "residual_grid": R_grid.astype(np.float32),
            "residual_up": R_up.astype(np.float32),
            "normal_da3": normal_a,
            "normal_out": depth_to_camera_normals(D_R, K_ref),
            "edge": edge, "segments": seg, "anchors": a, "info": diag,
            "anchor_depth_map": _rasterize(a, "d", (H, W)),
            "anchor_conf_map": _rasterize(a, "conf", (H, W))}


def _apply_affine(dA: np.ndarray, a: float, b: float, domain: str) -> np.ndarray:
    if domain == "rho":
        rho = a / np.maximum(dA.astype(np.float64), 1e-6) + b
        return 1.0 / np.maximum(rho, 1e-9)
    return a * dA.astype(np.float64) + b


def _segment_tukey(e: np.ndarray, seg_at: np.ndarray, cfg: ResidualFieldConfig) -> np.ndarray:
    """在每个表面内部做 MAD 标准化 + Tukey。label 0 (未分割) 用全局统计。"""
    g_med = float(np.median(e))
    g_sig = max(_mad_sigma(e), cfg.sigma_floor)
    med, sig = _group_median(e, seg_at)
    # label 0 = 未分割, 以及点太少的表面, 退回全局统计
    cnt = np.bincount(np.unique(seg_at, return_inverse=True)[1])
    n_at = cnt[np.unique(seg_at, return_inverse=True)[1]]
    fallback = (seg_at == 0) | (n_at < 12)
    med = np.where(fallback, g_med, med)
    sig = np.maximum(np.where(fallback, g_sig, sig), cfg.sigma_floor)
    u = (e - med) / (cfg.tukey_c * sig)
    return np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)


def _stratified_split(a: dict[str, np.ndarray], seg_at: np.ndarray,
                      alive: np.ndarray, cfg: ResidualFieldConfig) -> np.ndarray:
    """(segment, 粗格) 分层, 层内二维 FPS 取 fit_frac 做拟合。

    先滤离群 (alive) 再 FPS —— 反过来的话 FPS 的"选最远"恰好会优先选中空间上
    极端的离群点。多视角开着时额外保证: held-out 里尽量留下非参考视角的点, 那
    才是独立于 fit 集合的证据。
    """
    n = len(a["u"])
    fit = np.zeros(n, dtype=bool)
    idx_alive = np.flatnonzero(alive)
    if idx_alive.size == 0:
        return fit
    c = cfg.fps_cell
    wcell = int(a["u"].max()) // c + 2
    key = (seg_at[idx_alive].astype(np.int64) << 32) \
        + (a["v"][idx_alive].astype(np.int64) // c) * wcell \
        + (a["u"][idx_alive].astype(np.int64) // c)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for s, t in zip(bounds[:-1], bounds[1:]):
        loc = idx_alive[order[s:t]]
        k = max(1, int(np.ceil(cfg.fit_frac * len(loc))))
        xy = np.stack([a["u"][loc].astype(np.float64), a["v"][loc].astype(np.float64)], 1)
        fit[loc[_fps_2d(xy, k)]] = True
    return fit


def _solve_field(a: dict[str, np.ndarray], e: np.ndarray, w_hat: np.ndarray,
                 fit_mask: np.ndarray, edge: np.ndarray, shape_hw: tuple[int, int],
                 cfg: ResidualFieldConfig) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = shape_hw
    s = cfg.grid_stride
    hg, wg = _grid_shape(h, w, s)
    sel = np.flatnonzero(fit_mask)
    B = _bilinear_operator(a["u"][sel], a["v"][sel], h, w, s)
    ef = e[sel]
    # 数据项按"每个网格节点平均 1 份权重"归一化。不归一化的话 lambda_s / lambda_c
    # / lambda_0 的实际强度会随锚点数漂移 —— 换分辨率、换视角数、动一下 max_anchors
    # 都会让同一组超参表现完全不同 (实测锚点从 100 万降到 20 万, 同一组 lambda 就
    # 从"恰好"变成"过平滑", 误差 0.8mm -> 3.6mm)。
    wf = w_hat[sel]
    n_nodes = float(_grid_shape(h, w, s)[0] * _grid_shape(h, w, s)[1])
    wsum = float(wf.sum())
    if wsum > 0:
        wf = wf * (n_nodes / wsum)

    # 边缘图降到网格: 块内取最大 (只要块里有一条强边就断开耦合)
    pad_h, pad_w = hg * s - h, wg * s - w
    ep = np.pad(edge, ((0, pad_h), (0, pad_w)), mode="edge")
    edge_g = ep.reshape(hg, s, wg, s).max(axis=(1, 3))
    S, L = _grid_edge_terms(edge_g, cfg)

    # 支撑强度 S_q: 有效样本数, 用**未归一化**的权重算 —— 它要回答的是"这个节点
    # 底下到底有几个锚点", 那是个绝对量, 不该随数据项的归一化系数变。
    w_abs = w_hat[sel]
    sw = np.asarray(B.multiply(w_abs[:, None]).sum(axis=0)).reshape(-1)
    sw2 = np.asarray(B.multiply((w_abs ** 2)[:, None]).sum(axis=0)).reshape(-1)
    n_eff = (sw ** 2) / np.maximum(sw2, 1e-12)
    Sq = 1.0 - np.exp(-n_eff / cfg.support_n0)

    reg = (cfg.lambda_s * (S.T @ S)
           + cfg.lambda_c * (L.T @ L)
           + sparse.diags(cfg.lambda_0 * (1.0 - Sq) + cfg.ridge))

    x = np.zeros(hg * wg, dtype=np.float64)
    info: dict[str, Any] = {}
    ww = wf.copy()
    for it in range(cfg.irls_iters):
        Bw = B.multiply(ww[:, None]).tocsr()
        M = (B.T @ Bw + reg).tocsr()
        rhs = Bw.T @ ef
        x, ok = cg(M, rhs, x0=x, rtol=cfg.cg_rtol, atol=0.0, maxiter=cfg.cg_maxiter)
        if not np.isfinite(x).all():
            x = np.zeros(hg * wg, dtype=np.float64)
            info["cg_diverged"] = True
            break
        r = B @ x - ef
        ww = wf * np.minimum(1.0, cfg.huber_delta / np.maximum(np.abs(r), 1e-12))
        info["cg_info"] = int(ok)
        info["fit_mad"] = float(_mad_sigma(r))
    info["support_mean"] = float(Sq.mean())
    return x.reshape(hg, wg), info


def _rasterize(a: dict[str, np.ndarray], key: str, shape_hw: tuple[int, int]) -> np.ndarray:
    """锚点列表 -> 稠密图, 近点优先 (与 camera_utils.project_world_points_to_depth 一致)。"""
    h, w = shape_hw
    out = np.zeros((h, w), dtype=np.float32)
    order = np.argsort(a["d"])[::-1]
    out[a["v"][order], a["u"][order]] = a[key][order]
    return out


def _diagnostics(a, e, w_hat, fit_mask, alive, R_up, edge, shape_hw, cfg) -> dict[str, Any]:
    """支撑 / 拟合残差 / held-out 残差的**稠密图**, 供 confidence 使用。"""
    h, w = shape_hw
    box = max(cfg.grid_stride, 8)

    def splat(mask: np.ndarray, val: np.ndarray | None) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.float64)
        if not mask.any():
            return m.astype(np.float32)
        np.add.at(m, (a["v"][mask], a["u"][mask]),
                  np.ones(int(mask.sum())) if val is None else val[mask])
        return ndimage.uniform_filter(m, box, mode="nearest").astype(np.float32) * box * box

    fit_alive = fit_mask & alive
    hold = (~fit_mask) & alive
    sw = splat(fit_alive, w_hat)
    sw2 = splat(fit_alive, w_hat ** 2)
    n_eff = (sw ** 2) / np.maximum(sw2, 1e-12)

    r_fit = np.abs(e - R_up[a["v"], a["u"]])
    fit_err = splat(fit_alive, w_hat * r_fit) / np.maximum(sw, 1e-9)
    hold_w = splat(hold, w_hat)
    hold_err = splat(hold, w_hat * r_fit) / np.maximum(hold_w, 1e-9)

    anchor_mask = np.zeros((h, w), dtype=bool)
    anchor_mask[a["v"][alive], a["u"][alive]] = True
    return {"n_eff_map": n_eff.astype(np.float32),
            "fit_err_map": fit_err.astype(np.float32),
            "holdout_err_map": hold_err.astype(np.float32),
            "holdout_w_map": hold_w.astype(np.float32),
            "anchor_mask": anchor_mask,
            "edge_map": edge.astype(np.float32)}
