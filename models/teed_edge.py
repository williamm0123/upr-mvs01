"""TEED 边缘提取, 输入输出对齐本仓库的训练张量。

动机: MonoMVSNet 用 TEED (Tiny and Efficient Edge Detector) 取边缘做深度细化的
引导。这里把 ``models/TEED`` 那份官方实现包成一个**冻结的 nn.Module**, 并且只
接受 / 只吐出训练管线里已经在流通的张量形状, 这样它可以原地插进 network.py 或
test.py 而不需要在调用点做任何 numpy 往返:

    images       [B, V, 3, H, W] float, RGB, 0~255   (= sample["images"])
    depth        [B, H, W]       float, 公制 mm      (= sample["depth_prior"] / ["depth_gt"])
    edge         [B, 1, H, W]    float, 概率 [0, 1]

三条约定, 都是为了"接进网络时不会悄悄错位":

1. **通道序是 BGR。** TEED 的 TestDataset 用 ``cv2.imread`` 读图 (BGR) 之后直接
   减 ``mean_bgr``, 那句 RGB->BGR 的转换在官方代码里是注释掉的 (见
   models/TEED/dataset.py:429-433)。我们的 dataset 走 PIL, 拿到的是 RGB, 所以
   这里**必须**翻通道, 否则等于拿训练时没见过的输入去推理。
2. **分辨率补齐到 8 的倍数。** TED 里 block_1 是 stride-2, 上采样块是固定 2x 的
   ConvTranspose, 尺寸不是 8 的倍数时各尺度对不齐。这里 reflect pad 到 8 的倍数,
   出来再切回原尺寸 —— 比 TED 自带的 bicubic ``resize_input`` 少一次重采样,
   边缘位置不会漂。
3. **强制 float32。** 外层 train.py 开着 autocast; smish 里有 log/sigmoid 串联,
   bf16 下边缘图会随 autocast 开关变。边缘是辅助信号, 让它与 autocast 无关。

深度图的边缘: 深度是公制 mm (DTU 上 400~900), 直接喂给在自然图像上训的 TEED
没有意义。``depth_to_pseudo_rgb`` 先做逐图**分位数**归一化 (不是 min-max ——
DA3/先验的极值像素会把整幅压平) 再复制成三通道。sobel 走同一张归一化图, 于是
步骤 2 和步骤 3 比的是同一个输入, 差异只来自算子本身。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TEED_ROOT", "DEFAULT_TEED_CKPT", "TEED_MEAN_BGR",
    "TEEDEdgeExtractor", "depth_to_pseudo_rgb", "sobel_edges",
    "dilate", "binarize", "edge_intersect", "edge_union", "normalize_edge",
    "compute_edge_bundle",
]

TEED_ROOT = Path(__file__).resolve().parent / "TEED"
# main.py 的 --checkpoint_data 默认就是 5/5_model.pth, 跟着它走。
DEFAULT_TEED_CKPT = TEED_ROOT / "checkpoints" / "BIPED" / "5" / "5_model.pth"
# models/TEED/dataset.py:31 BIPED_mean 的前三项 (第四项是 GT 的均值, 不参与输入)。
TEED_MEAN_BGR: tuple[float, float, float] = (103.939, 116.779, 123.68)


# --------------------------------------------------------------------------- #
# 加载官方 TED 结构
# --------------------------------------------------------------------------- #
def _exec_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:      # pragma: no cover
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ted_class():
    """导入 ``models/TEED/ted.py`` 里的 TED, 且不让它的 ``utils`` 顶掉我们的。

    ``ted.py`` 写的是 ``from utils.AF.Fsmish import ...`` —— 一个**绝对**导入,
    包名 ``utils`` 和仓库根的 ``utils/`` 撞名, 而后者有 ``__init__.py`` (常规包),
    按 PEP 420 常规包在 sys.path 扫描中一旦命中就立即胜出, 所以单纯把
    ``models/TEED`` 塞进 sys.path[0] 并**不能**让它解析到 TEED 自己那份。

    这里改成: 按 ted.py 需要的名字把三个模块直接注册进 sys.modules, exec 完
    ted.py 之后原样还原 sys.modules。ted.py 在 exec 时就把 ``Fsmish`` / ``Smish``
    绑进自己的 globals 了, 之后不再触发导入, 所以还原是安全的。

    ``utils.img_processing`` 打的是桩: ted.py 只从它取 ``count_parameters`` 而且
    导入期不调用, 但真模块顶层会拉进 kornia + skimage + sklearn。
    """
    names = ("utils", "utils.AF", "utils.AF.Fsmish", "utils.AF.Xsmish",
             "utils.img_processing")
    saved = {n: sys.modules.get(n) for n in names}
    try:
        pkg_utils = types.ModuleType("utils")
        pkg_utils.__path__ = [str(TEED_ROOT / "utils")]          # type: ignore[attr-defined]
        pkg_af = types.ModuleType("utils.AF")
        pkg_af.__path__ = [str(TEED_ROOT / "utils" / "AF")]      # type: ignore[attr-defined]
        stub_ip = types.ModuleType("utils.img_processing")
        stub_ip.count_parameters = lambda model=None: (          # type: ignore[attr-defined]
            sum(p.numel() for p in model.parameters()) if model is not None else 0)
        sys.modules["utils"] = pkg_utils
        sys.modules["utils.AF"] = pkg_af
        sys.modules["utils.img_processing"] = stub_ip
        pkg_utils.AF = pkg_af                                    # type: ignore[attr-defined]
        pkg_utils.img_processing = stub_ip                       # type: ignore[attr-defined]

        # Xsmish 自己 ``import utils.AF.Fsmish``, 所以 Fsmish 必须先就位。
        pkg_af.Fsmish = _exec_module("utils.AF.Fsmish", TEED_ROOT / "utils" / "AF" / "Fsmish.py")   # type: ignore[attr-defined]
        pkg_af.Xsmish = _exec_module("utils.AF.Xsmish", TEED_ROOT / "utils" / "AF" / "Xsmish.py")   # type: ignore[attr-defined]
        ted = _exec_module("_uprmvs_teed_ted", TEED_ROOT / "ted.py")
    finally:
        for n, old in saved.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old
    return ted.TED


# --------------------------------------------------------------------------- #
# 形状适配: [B,V,3,H,W] / [B,3,H,W] / [3,H,W] 都收, 出去时把前缀维还回去
# --------------------------------------------------------------------------- #
def _flatten_leading(x: torch.Tensor, ndim_core: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """把 ``x`` 压成 [N, *core], 返回 (flat, 被折叠掉的前缀 shape)。"""
    if x.dim() < ndim_core:
        raise ValueError(f"expected at least {ndim_core}D, got {tuple(x.shape)}")
    lead = tuple(x.shape[: x.dim() - ndim_core])
    return x.reshape(-1, *x.shape[x.dim() - ndim_core:]), lead


def _restore_leading(x: torch.Tensor, lead: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(*lead, *x.shape[1:])


def _as_nchw_rgb(images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """RGB 张量 -> ([N,3,H,W], lead)。接受 [3,H,W] / [B,3,H,W] / [B,V,3,H,W]。"""
    if images.dim() < 3 or images.shape[-3] != 3:
        raise ValueError(
            f"images 必须是 (..., 3, H, W) 的 RGB 张量, 收到 {tuple(images.shape)}")
    return _flatten_leading(images, 3)


def _as_nhw_depth(depth: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """深度张量 -> ([N,H,W], lead)。接受 [H,W] / [B,H,W] / [B,1,H,W] / [B,V,H,W]。

    只在 dim>=4 时脱通道维: 三维的 [1,H,W] 在本仓库里恒是 B=1 的一批深度
    (sample["depth_prior"] 就是 [B,H,W]), 当成 [C=1,H,W] 脱掉会把 batch 维弄丢。
    """
    if depth.dim() >= 4 and depth.shape[-3] == 1:
        depth = depth.squeeze(-3)                 # [.., 1, H, W] -> [.., H, W]
    return _flatten_leading(depth, 2)


# --------------------------------------------------------------------------- #
# 深度 -> 伪 RGB / sobel
# --------------------------------------------------------------------------- #
def _robust_lohi(x: torch.Tensor, valid: torch.Tensor,
                 pct: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """逐图分位数。全无效或退化 (lo==hi) 的图返回 (0, 1), 归一化后是常量图。"""
    n = x.shape[0]
    lo = x.new_zeros(n)
    hi = x.new_ones(n)
    q = torch.tensor([pct[0] / 100.0, pct[1] / 100.0], device=x.device, dtype=torch.float32)
    for i in range(n):
        v = x[i][valid[i]].float()
        if v.numel() < 16:
            continue
        a, b = torch.quantile(v, q).tolist()
        if b - a > 1e-6:
            lo[i], hi[i] = a, b
    return lo, hi


def normalize_depth(depth: torch.Tensor, valid: torch.Tensor | None = None,
                    pct: tuple[float, float] = (2.0, 98.0)
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """逐图分位数归一化到 [0,1]。返回 (norm [N,H,W], valid [N,H,W] bool)。

    无效像素 (<=0 或非有限) 填该图**有效像素的中位数** —— 填 0 会在无效区边界上
    造出一圈最强的假边, 填中位数则区内是平的, 只剩边界那一圈, 而边界本来就该由
    调用方用返回的 valid 掩掉。
    """
    d, _ = _as_nhw_depth(depth)
    d = d.float()
    v = (torch.isfinite(d) & (d > 0)) if valid is None else (
        _as_nhw_depth(valid)[0].bool() & torch.isfinite(d))
    filled = d.clone()
    filled[~torch.isfinite(filled)] = 0.0
    for i in range(d.shape[0]):
        vi = filled[i][v[i]]
        if vi.numel():
            filled[i] = torch.where(v[i], filled[i], vi.median())
    lo, hi = _robust_lohi(filled, v, pct)
    norm = ((filled - lo.view(-1, 1, 1)) / (hi - lo).view(-1, 1, 1)).clamp(0.0, 1.0)
    return norm, v


def depth_to_pseudo_rgb(depth: torch.Tensor, valid: torch.Tensor | None = None,
                        pct: tuple[float, float] = (2.0, 98.0)) -> torch.Tensor:
    """[.., H, W] 公制深度 -> [.., 3, H, W] 伪 RGB, 0~255 float (三通道相同)。"""
    _, lead = _as_nhw_depth(depth)
    norm, _ = normalize_depth(depth, valid, pct)
    rgb = (norm * 255.0).unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    return _restore_leading(rgb, lead)


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]]) / 8.0


def sobel_edges(depth: torch.Tensor, valid: torch.Tensor | None = None,
                normalize_input: bool = True,
                pct: tuple[float, float] = (2.0, 98.0),
                mag_pct: float = 99.0) -> torch.Tensor:
    """深度图的 Sobel 梯度幅值, 归一化到 [0,1], 形状 [.., 1, H, W]。

    ``normalize_input=True`` (默认) 时先走 :func:`normalize_depth`, 与
    :meth:`TEEDEdgeExtractor.edges_from_depth` 吃的是**同一张图**, 步骤 2 / 步骤 3
    因此是可比的; 关掉则直接在公制 mm 上求梯度 (幅值量纲是 mm/px)。

    幅值除以逐图的 ``mag_pct`` 分位数而不是最大值: DTU 上背景与前景的那道深度
    断崖比物体表面结构大一到两个量级, 用 max 归一会把表面结构全压到 0 附近。
    """
    d, lead = _as_nhw_depth(depth)
    if normalize_input:
        x, v = normalize_depth(d, valid, pct)
    else:
        x = d.float().clone()
        v = (torch.isfinite(d) & (d > 0)) if valid is None else _as_nhw_depth(valid)[0].bool()
        x[~v] = 0.0
    kx = _SOBEL_X.to(x.device, x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()
    xp = F.pad(x.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, kx)
    gy = F.conv2d(xp, ky)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)                   # [N,1,H,W]
    # 无效像素处的梯度没有意义, 先清零再定标, 免得它抬高分位数
    mag = mag * v.unsqueeze(1).float()
    n = mag.shape[0]
    scale = mag.new_ones(n)
    q = torch.tensor(mag_pct / 100.0, device=mag.device)
    for i in range(n):
        m = mag[i].reshape(-1)
        if m.numel():
            s = torch.quantile(m.float(), q)
            if s > 1e-9:
                scale[i] = s
    out = (mag / scale.view(-1, 1, 1, 1)).clamp(0.0, 1.0)
    return _restore_leading(out, lead)


# --------------------------------------------------------------------------- #
# 边缘图上的形态学 / 集合操作 (全张量, 可微, 可在训练图里用)
# --------------------------------------------------------------------------- #
def dilate(edge: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """膨胀 ``radius`` 像素 (8-邻域, 即 (2r+1) 方形结构元)。形状不变。"""
    if radius <= 0:
        return edge
    x, lead = _flatten_leading(edge, 3)                           # [N,1,H,W]
    k = 2 * radius + 1
    out = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    return _restore_leading(out, lead)


def binarize(edge: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """阈值化成 {0.0, 1.0} 的 float (保持 dtype/device, 便于继续做算术)。"""
    return (edge > thresh).to(edge.dtype)


def edge_intersect(*edges: torch.Tensor) -> torch.Tensor:
    """交集。二值输入 = 逻辑与; 软输入 = 逐像素取小。"""
    if not edges:
        raise ValueError("edge_intersect needs at least one map")
    out = edges[0]
    for e in edges[1:]:
        out = torch.minimum(out, e)
    return out


def edge_union(*edges: torch.Tensor) -> torch.Tensor:
    """并集。二值输入 = 逻辑或; 软输入 = 逐像素取大。"""
    if not edges:
        raise ValueError("edge_union needs at least one map")
    out = edges[0]
    for e in edges[1:]:
        out = torch.maximum(out, e)
    return out


def normalize_edge(edge: torch.Tensor) -> torch.Tensor:
    """逐图 min-max 拉满到 [0,1] —— 只为**可视化**。

    TEED 官方存图前做的就是这一步 (utils/img_processing.image_normalization)。
    注意它是逐图的, 会把"这张图上最强的那条边"永远拉成 1, 所以**不要**拿它去做
    阈值判断或喂给下游网络; 那些地方用原始 sigmoid 概率。
    """
    x, lead = _flatten_leading(edge, 3)
    n = x.shape[0]
    flat = x.reshape(n, -1).float()
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    out = (x - lo) / (hi - lo).clamp_min(1e-6)
    return _restore_leading(out.clamp(0.0, 1.0), lead)


# --------------------------------------------------------------------------- #
# 主模块
# --------------------------------------------------------------------------- #
class TEEDEdgeExtractor(nn.Module):
    """冻结的 TEED 边缘提取器。

    Args:
        ckpt:      BIPED 权重 (默认 ``models/TEED/checkpoints/BIPED/5/5_model.pth``,
                   与 TEED 自己 main.py 的 ``--checkpoint_data`` 默认一致)。
        out_index: 取 TED 四个输出里的哪一个。-1 = DoubleFusion 融合头, 也就是
                   TEED 论文/官方脚本里的 "fused"。
        freeze:    True 时参数 requires_grad=False, 且 ``train()`` 不会把它切回
                   训练模式 —— 它是固定的先验提取器, 不该被主干的 train() 带走。
        context_pad: 前向时先 reflect 补这么多圈上下文, 出来再切掉。卷积在图像
                   最外圈是零填充, TEED 因此在边界 1~2 列上有一条假边: DTU
                   scan34 上第 0/1 列有 47% 的像素越过 0.5 阈值, 而内部只有 8%。
                   接进网络时这条假边会在每一帧的同一位置出现, 是个系统性偏置。
                   补 16 圈上下文后降到与内部同量级。设 0 = 复现 TEED 官方行为。

    形状 (与训练张量一一对应):
        forward(images [.., 3, H, W] float RGB 0~255) -> [.., 1, H, W] 概率 [0,1]
        edges_from_depth(depth [.., H, W] 公制)       -> [.., 1, H, W] 概率 [0,1]
    """

    def __init__(self, ckpt: str | Path | None = None, out_index: int = -1,
                 freeze: bool = True, context_pad: int = 16) -> None:
        super().__init__()
        ckpt_path = Path(ckpt) if ckpt is not None else DEFAULT_TEED_CKPT
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"TEED checkpoint not found: {ckpt_path}\n"
                f"  (仓库自带的在 {TEED_ROOT / 'checkpoints' / 'BIPED'} 下)")
        TED = load_ted_class()
        self.net = TED()
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        self.net.load_state_dict(state, strict=True)
        self.net.eval()

        self.ckpt_path = ckpt_path
        self.out_index = int(out_index)
        self.frozen = bool(freeze)
        self.context_pad = int(context_pad)
        if self.frozen:
            for p in self.net.parameters():
                p.requires_grad_(False)
        # 均值随模块走 device/dtype, 不用每次前向新建张量
        self.register_buffer(
            "mean_bgr", torch.tensor(TEED_MEAN_BGR).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):          # noqa: D102
        if self.frozen:
            return super().train(False)
        return super().train(mode)

    # -- 内部: 一批 [N,3,H,W] RGB 0~255 -> TED 的 4 个 logits ----------------- #
    def _run(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        x = rgb.float()
        x = x.flip(-3)                                   # RGB -> BGR (见模块 docstring)
        x = x - self.mean_bgr.to(x.dtype)
        h, w = x.shape[-2:]
        c = self.context_pad
        # 左/上补 c 圈上下文, 右/下再多补到 8 的倍数, 出来按 (c, c) 起点切回原尺寸
        ph, pw = (-(h + 2 * c)) % 8, (-(w + 2 * c)) % 8
        if c or ph or pw:
            x = F.pad(x, (c, c + pw, c, c + ph), mode="reflect")
            outs = [o[..., c:c + h, c:c + w] for o in self.net(x)]
        else:
            outs = self.net(x)
        return outs

    def forward(self, images: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """RGB -> 边缘概率。``return_all`` 时返回 [.., 4, H, W] (TED 的四个尺度)。"""
        rgb, lead = _as_nchw_rgb(images)
        dev = self.mean_bgr.device
        with torch.autocast(device_type=dev.type, enabled=False):
            outs = self._run(rgb.to(dev))
        if return_all:
            prob = torch.sigmoid(torch.cat(outs, dim=1))          # [N,4,H,W]
        else:
            prob = torch.sigmoid(outs[self.out_index])            # [N,1,H,W]
        return _restore_leading(prob, lead)

    # 语义别名, 调用点读起来对称
    def edges_from_rgb(self, images: torch.Tensor, **kw) -> torch.Tensor:
        return self.forward(images, **kw)

    def edges_from_depth(self, depth: torch.Tensor, valid: torch.Tensor | None = None,
                         pct: tuple[float, float] = (2.0, 98.0), **kw) -> torch.Tensor:
        """公制深度图 -> 边缘概率。深度先做分位数归一化再复制成三通道伪 RGB。"""
        return self.forward(depth_to_pseudo_rgb(depth, valid, pct), **kw)


# --------------------------------------------------------------------------- #
# 五步组合 —— experiments/teed_edge_test.py 和训练侧共用同一段逻辑
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_edge_bundle(images: torch.Tensor, depth: torch.Tensor,
                        extractor: TEEDEdgeExtractor,
                        depth_valid: torch.Tensor | None = None,
                        thresh: float = 0.5, dilate_radius: int = 1,
                        pct: tuple[float, float] = (2.0, 98.0)) -> dict[str, torch.Tensor]:
    """一次算齐五张边缘图。

    Args:
        images: [.., 3, H, W] RGB float 0~255 —— 参考视角那一张 (不是整个 [B,V,...])。
        depth:  [.., H, W] 公制深度 (DA3 或先验缓存的 depth_prior)。

    Returns (全部 [.., 1, H, W] float):
        rgb_teed       步骤1  RGB 的 TEED 概率
        depth_sobel    步骤2  深度图的 Sobel 幅值 [0,1]
        depth_teed     步骤3  深度图的 TEED 概率
        intersection          步骤4  二值(步骤1) ∩ 二值(步骤3)
        intersection_dilated  步骤5  各自膨胀 dilate_radius 像素后再取交
      另外附带 (方便调用方做阈值/统计, 不是那五张图):
        rgb_teed_bin / depth_teed_bin  二值化后的步骤1 / 步骤3
        depth_norm                     喂给 sobel 和 TEED 的那张归一化深度 [.., 1, H, W]
    """
    rgb_teed = extractor.edges_from_rgb(images)
    depth_teed = extractor.edges_from_depth(depth, depth_valid, pct)
    depth_sobel = sobel_edges(depth, depth_valid, normalize_input=True, pct=pct)

    rgb_bin = binarize(rgb_teed, thresh)
    dep_bin = binarize(depth_teed, thresh)
    inter = edge_intersect(rgb_bin, dep_bin)
    # 先各自膨胀再取交 = 带 dilate_radius 像素容差的交集: 两张图上"对得上但差几个
    # 像素"的同一条边, 严格逐像素相交会判成不相交, 膨胀之后才会被留下。
    inter_dil = edge_intersect(dilate(rgb_bin, dilate_radius),
                               dilate(dep_bin, dilate_radius))

    norm, _ = normalize_depth(depth, depth_valid, pct)
    _, lead = _as_nhw_depth(depth)
    return {
        "rgb_teed": rgb_teed,
        "depth_sobel": depth_sobel,
        "depth_teed": depth_teed,
        "intersection": inter,
        "intersection_dilated": inter_dil,
        "rgb_teed_bin": rgb_bin,
        "depth_teed_bin": dep_bin,
        "depth_norm": _restore_leading(norm.unsqueeze(1), lead),
    }
