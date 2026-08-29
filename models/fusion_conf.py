from __future__ import annotations

"""W3-C: 最终重建置信度 —— 连接 val 指标与点云指标的那座桥。

现在的 ``test.cascade_confidence`` 是一个**手工概率乘积**: 把四级 "mode 邻域
质量" 直接相乘当置信度。它有两个问题:

1. 阈值没有物理含义。实测 product 的最小值是 0.019、geomean 的最小值是 0.372,
   同一个 ``--photo-thresh 0.3`` 在两种 mode 下的保留率完全不同 —— 换个 mode
   就得重扫阈值, 这说明这个数不是概率。
2. 它衡量的是 "后验集中不集中", 而融合真正需要的是 "这个像素的深度对不对"。
   后验集中且错 (选错表面) 正是尾部误差的主要来源, 恰恰是它漏掉的那一类。

这里换成一个**有监督、可校准**的头: 训练目标 ``1[|d_final - gt| < tau]``,
训练完在验证集上做一次单参数温度缩放, 于是 ``--photo-thresh 0.5`` 第一次
真的表示 "五成把握"。

关于输入 —— v2 把融合阶段才有的量当成了前向输入, 这是错的
=========================================================
模型一次只预测**一个参考视图**。``n_geo`` (跨视图重投影一致的视图数) 和真正的
前后向重投影一致性, 要等所有参考视图的深度图都存完、在融合阶段才算得出来
(test.py 的 filter_depth 里)。除非额外再跑一遍源视图的深度预测, 前向阶段拿不到
它们。

所以分工是:
  * **可以当输入**: stage4 后验统计、MAP 残差幅度、stage3->4 的中心移动与区间
    宽度、gamma_4*sg(r)、n_valid(W3-A)、source correlation 的均值/方差。
  * **只能当标签**: 基于源视图预测深度的 n_geo、真正的重投影一致性、融合后的
    点云支持数。v2 目标 ``y = 1[|d-gt|<tau] * 1[n_geo_gt >= 3]`` 里的 n_geo 是
    **用 GT 算的**, 它进标签不进输入。

v1 目标 (本实现) 不需要源视图 GT: ``y = 1[|d_final - gt| < tau_mm]``。W3-A 的
n_valid 和相关性统计已经给了头足够的几何代理信息, 不必为了 n_geo 阻塞整个 W3-C。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionConfidenceHead(nn.Module):
    """逐像素的最终重建置信度。输出未校准的 logit。

    ``log_T`` 是 buffer 不是 parameter: 温度缩放是**训练之后**在独立验证集上
    拟合的单参数后处理, 不能和 BCE 一起训 —— 一起训的话温度就被训练集的
    过拟合程度吸收掉了, 校准的意义正好没了。它进 checkpoint, 推理时直接用。
    """

    IN_CH = 12

    def __init__(self, feat_channels: int, hidden: int = 32, feat_proj: int = 8,
                 in_ch: int = 12, prior_pos: float = 0.9) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.proj = nn.Conv2d(feat_channels, feat_proj, 1)
        self.c1 = nn.Conv2d(self.in_ch + feat_proj, hidden, 1)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.head = nn.Conv2d(hidden, 1, 1)
        # 零权重 + 先验 bias。不用零 bias —— 那等于先验 0.5, 而 acc@2mm 实际
        # 在 0.9 附近, 从 logit(0.9) 起步收敛快也稳。
        nn.init.zeros_(self.head.weight)
        p = min(max(float(prior_pos), 1e-3), 1.0 - 1e-3)
        nn.init.constant_(self.head.bias, math.log(p / (1.0 - p)))
        self.register_buffer("log_T", torch.zeros(()))
        # 二参数 Platt 标定的截距。**单参数温度不够**: 置信度损失用的是类别平衡
        # BCE (losses/composite.py 的 conf 分支), 加权会平移最优 logit 的截距,
        # 所以只拟合 T 时 prob=0.5 并不表示 "五成把握"。见 calibrated()。
        self.register_buffer("calib_bias", torch.zeros(()))

    def forward(self, feats: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        """``feats`` [B, IN_CH, H, W]; ``ref_feat`` [B, C, H, W]。返回 logit [B, H, W]。"""
        x = torch.cat([feats.float(), self.proj(ref_feat.float())], dim=1)
        h = F.gelu(self.c1(x))
        h = F.gelu(self.c2(h))
        return self.head(h).squeeze(1)

    def calibrated(self, logit: torch.Tensor) -> torch.Tensor:
        """部署用的概率 ``sigma(z / T + b)``。未标定时 T=1, b=0, 与 sigmoid(z) 相同。

        正斜率仿射 ``z -> z/T + b`` (T > 0) **不改变排序**, 所以 AURC 与
        risk@kappa 在标定前后必须逐位一致 —— scripts/calibrate_conf.py 会断言
        这一点。标定只动 "这个数读作多少概率", 不动 "先信哪个像素"。
        """
        t = self.log_T.exp().clamp_min(1e-3)
        return torch.sigmoid(logit / t + self.calib_bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 旧 checkpoint 没有 calib_bias。补 0 而不是放宽 strict —— strict=False
        # 会把**真正的**形状/命名错误一起放过去, 那比缺一个 buffer 危险得多。
        key = prefix + "calib_bias"
        if key not in state_dict:
            state_dict[key] = torch.zeros((), dtype=self.calib_bias.dtype)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


# ===========================================================================
#  级联置信度。**训练侧 (train.py 的 risk@0.6) 与推理侧 (test.py 的融合门控)
#  必须是同一份实现** —— 两边各写一份的话, "训练时看到的排序" 和 "融合时用的
#  排序" 就是两个东西, 而 risk@0.6 的全部意义正是预测后者。
# ===========================================================================


def _stage_mode_mass(stage: dict, window: int, target_hw) -> torch.Tensor:
    """Posterior mass within +-``window`` bins of this stage's argmax, at ``target_hw``."""
    prob = stage["prob"].float()
    D = prob.shape[1]
    w = min(2 * window + 1, D)
    idx = stage.get("mode_idx")
    if idx is None:
        idx = prob.argmax(dim=1, keepdim=True)
    # Slide the window to stay in bounds rather than clamping indices, which
    # would gather the edge bin repeatedly and double-count its mass.
    start = (idx - (w // 2)).clamp(0, D - w)
    offs = torch.arange(w, device=prob.device).view(1, -1, 1, 1)
    mass = prob.gather(1, start + offs).sum(dim=1).clamp(0.0, 1.0)
    if tuple(mass.shape[-2:]) != tuple(target_hw):
        mass = F.interpolate(mass.unsqueeze(1), size=tuple(target_hw),
                             mode="bilinear", align_corners=False).squeeze(1)
    return mass


def cascade_confidence(outputs: dict, window: int = 1, mode: str = "product",
                       stages=("stage1", "stage2", "stage3", "stage4")) -> torch.Tensor:
    """Fusion confidence combined across cascade stages (ported from test_tt.py).

    Why not the final stage alone: it carries ``num_depths_stage4=4`` hypotheses,
    so the old ``mode_window=2`` window spanned the whole axis and the mass was
    identically 1.0 — which made ``--photo-thresh`` an inert gate and fusion ran
    on geometric consistency alone. That is what invalidated the 2026-08-08 DTU
    numbers (0.3944/0.2482/0.3213); see the note in that run's metrics.

    Stage 1 has 48 bins and is the only level where a +-1 window is genuinely
    selective, so a pixel is trusted when *every* stage concentrated its
    posterior, not just the last.

    ``mode``:
      ``product``  — all stages must agree; the sharpest, and the default.
      ``geomean``  — same ordering, rescaled so a threshold tuned on one stage
                     count still means something.
      ``last``     — final stage only. 注意要复现修复前的失效行为需要
                     ``--conf-mode last --conf-window 2`` (window=2 才覆盖 4 元
                     轴的全部); ``last`` 配 window=1 是另一回事。仅用于 A/B。

    阈值不可跨 mode 迁移: 实测 product 的 min 是 0.019、geomean 的 min 是 0.372,
    同一个 ``--photo-thresh 0.3`` 在 geomean 下仍然一个像素都不过滤。换 mode
    必须重扫 ``--photo-thresh``。
    """
    hw = outputs["depth_full"].shape[-2:]
    masses = [_stage_mode_mass(outputs[s], window, hw) for s in stages if s in outputs]
    if not masses:
        raise KeyError(f"none of {stages} present in outputs")
    if mode == "last":
        return masses[-1]
    stacked = torch.stack(masses, dim=0)
    if mode == "product":
        return stacked.prod(dim=0)
    if mode == "geomean":
        return stacked.clamp_min(1e-8).log().mean(dim=0).exp()
    raise ValueError(f"unknown conf mode {mode!r}")


# ===========================================================================
#  校准与风险-覆盖指标。训练侧 (losses/composite) 与推理侧 (test.py /
#  scripts/calibrate_conf.py) 都从这里取, 避免两边各写一份口径不同的实现。
# ===========================================================================


def expected_calibration_error(prob: torch.Tensor, label: torch.Tensor,
                               n_bins: int = 15) -> float:
    """ECE, **等质量**分箱 (每箱样本数相同), 不是等宽分箱。

    等宽分箱在这里会失真: 校准后的置信度绝大多数挤在 0.85-1.0, 等宽的低置信
    箱几乎是空的, 空箱不贡献误差, ECE 就被系统性低估。
    """
    p = prob.detach().float().flatten()
    y = label.detach().float().flatten()
    n = p.numel()
    if n == 0:
        return 0.0
    order = p.argsort()
    p, y = p[order], y[order]
    edges = torch.linspace(0, n, n_bins + 1, device=p.device).round().long()
    ece = 0.0
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        ece += (hi - lo) / n * float((p[lo:hi].mean() - y[lo:hi].mean()).abs())
    return ece


def brier_score(prob: torch.Tensor, label: torch.Tensor) -> float:
    p = prob.detach().float().flatten()
    y = label.detach().float().flatten()
    return float(((p - y) ** 2).mean()) if p.numel() else 0.0


def risk_coverage(conf: torch.Tensor, risk: torch.Tensor, n_points: int = 101):
    """按置信度降序保留 -> (coverage, risk) 曲线, 外加 AURC。

    ``risk`` 是逐像素的代价 (``|d-gt|`` 或 ``1 - acc@2mm``)。返回
    ``(cov[n], R[n], aurc)``; ``aurc = ∫R(k)dk``, 越低越好。这条曲线才是
    "置信度当过滤器好不好用" 的直接判据 —— ECE 说的是**准不准**, 是另一回事,
    两者可以一好一坏 (AURC 好而 ECE 差 = 只需重新校准, 不必重训)。
    """
    c = conf.detach().float().flatten()
    r = risk.detach().float().flatten()
    n = c.numel()
    if n == 0:
        return (torch.zeros(0), torch.zeros(0), 0.0)
    order = c.argsort(descending=True)
    csum = r[order].cumsum(0)
    ks = torch.linspace(1, n, n_points, device=c.device).round().long().clamp(1, n)
    cov = ks.float() / n
    rk = csum[ks - 1] / ks.float()
    aurc = float(torch.trapz(rk, cov)) if n_points > 1 else float(rk.mean())
    return cov, rk, aurc


def risk_at_coverage(conf: torch.Tensor, risk: torch.Tensor, kappa: float) -> float:
    """保留率固定在 kappa 时的平均风险。现有融合保留率是 0.61-0.65, 报 R(0.6)。"""
    c = conf.detach().float().flatten()
    r = risk.detach().float().flatten()
    n = c.numel()
    if n == 0:
        return 0.0
    k = max(1, min(n, int(round(kappa * n))))
    idx = c.argsort(descending=True)[:k]
    return float(r[idx].mean())


@torch.no_grad()
def fit_temperature(logit: torch.Tensor, label: torch.Tensor,
                    iters: int = 200, lo: float = -3.0, hi: float = 3.0) -> float:
    """单参数温度缩放: 在 log T 上做黄金分割搜索, 最小化 NLL。

    用一维搜索而不是 LBFGS: 只有一个参数、目标在 log T 上是良态单峰的, 搜索
    没有初值和收敛判据的问题, 也不会像 LBFGS 那样偶尔跑到 T<0。
    """
    z = logit.detach().float().flatten()
    y = label.detach().float().flatten()
    if z.numel() == 0:
        return 1.0

    def nll(log_t: float) -> float:
        return float(F.binary_cross_entropy_with_logits(z / math.exp(log_t), y))

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c_, d_ = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(c_), nll(d_)
    for _ in range(iters):
        if b - a < 1e-4:
            break
        if fc < fd:
            b, d_, fd = d_, c_, fc
            c_ = b - phi * (b - a)
            fc = nll(c_)
        else:
            a, c_, fc = c_, d_, fd
            d_ = a + phi * (b - a)
            fd = nll(d_)
    return math.exp(0.5 * (a + b))


def fit_platt(logit: torch.Tensor, label: torch.Tensor,
              max_iter: int = 100) -> tuple[float, float]:
    """二参数 Platt 标定 ``p = sigma(z / T + b)``, 返回 ``(log_T, bias)``。

    为什么不是单参数温度: 训练用的是**类别平衡** BCE, 正负样本被重新加权之后
    最优 logit 的截距被整体平移了, 单靠 T 只能改斜率、改不了截距, 于是
    ``prob=0.5`` 不等于 50% 正确率。加一个 bias 才把两个自由度都补齐。

    确定性: CPU + LBFGS + strong_wolfe + 固定初值 (log_T=0, bias=0)。同一份输入
    必须给出同一组参数, 否则 "标定" 本身就成了一个不可复现的实验变量。

    目标是**未加权**的 BCE —— 校准要对齐的是真实的正类频率, 不是训练时为了
    平衡梯度而人为设定的频率。
    """
    z = logit.detach().float().flatten().cpu()
    y = label.detach().float().flatten().cpu()
    if z.numel() == 0:
        return 0.0, 0.0
    # 非有限输入必须**报错**而不是回退。回退到 (0, 0) 会产出一个"标定过"却其实是
    # 恒等的 checkpoint, 下游 test.py 打印 `learned(T=1.000,b=+0.000)` 看起来完全
    # 正常 —— job 415038 就是这么过去的。
    nz = int((~torch.isfinite(z)).sum())
    if nz:
        raise ValueError(
            f"fit_platt 收到 {nz}/{z.numel()} 个非有限 logit。模型在这个口径下输出了 "
            f"nan/inf, 先修那个, 不要标定。常见原因: 推理的 autocast dtype 与训练不符。")
    log_t = torch.zeros((), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t, bias], max_iter=int(max_iter),
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(z / log_t.clamp(-5.0, 5.0).exp() + bias, y)
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    lt = float(log_t.detach().clamp(-5.0, 5.0))
    if not math.isfinite(lt):
        lt = 0.0
    bb = float(bias.detach())
    if not math.isfinite(bb):
        bb = 0.0
    return lt, bb
