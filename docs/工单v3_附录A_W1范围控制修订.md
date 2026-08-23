# 附录 A · W1 范围控制修订（2026-08-23）

> 本附录追加在《UPRMVS · 实施工单 v3 · 2026-08-22》**尾部**，v3 正文一字未改。
> 触发它的是 W1 的第一次真实训练：单卡 A100-80GB，batch 5，2040 步 / 122.8 分钟。

---

## A.0 一句话

v3 把「窗宽该多大」交给了可学习的控制器，但没有任何东西规定「级联必须**逐级提高采样分辨率**」。首跑的结果是控制器学会了用加宽来满足 pinball，于是四假设的 stage4 变得比八假设的 stage3 还粗。本附录把这条隐含前提写成显式、可验证的结构约束。

---

## A.1 首跑观测

训练本身是健康的：val abs_err 105.8 → 5.51 mm（2000 步），acc@2mm 0.799，acc@8mm 0.914；梯度范数 6–8，无 nan/inf；覆盖率追上了 τ（0.973 / 0.911 / 0.863 对 0.98 / 0.95 / 0.92，如 v3 预期般略低）。

两个 v3 担心的问题**没有**发生：`wbar_fallback_frac` 恒为 0（近重复候选不成问题）；`regress_window_bimodal_frac` 极小（stage3 约 8e-5），「期望窗口跨两个表面」不是当前瓶颈。

问题出在别处 —— **级联倒挂**：

| 指标 @2040 步 | stage2 | stage3 | stage4 |
|---|---|---|---|
| bin 间距 | 2.12 mm | 2.72 mm | **6.88 mm** |
| half_p90 | 123 mm | 165 mm | **194 mm** |
| rho_bind（窗宽超物理域） | 0.046 | 0.079 | **0.150** |

历史 f15 的 stage4 半宽是 2.46 mm（间距约 1.65 mm），现在粗了 4 倍。同期 stage2/stage3 的 CE 反而上升（1.43→1.78、1.31→1.52），与「bin 变粗、分类变难」一致。half_p50 一直很小（stage4 2.35 mm），所以这是**尾部**问题而非整体加宽。

要把两个效应分开：逆深度轴让远端窗口按 d² 变宽，这是 v3 **想要**的；但 `rho_bind = 0.15` 意味着 15% 的 stage4 像素想要一个比整个深度域（425–935 mm）还宽的窗口 —— 再多的 d² 自适应也不该这样。而且加宽随 `ctrl_blend` 0.6→1.0 明显加速，与轴切换强相关。

`sat_frac` 已经是 0，所以 v3 里「sat_frac 高就放宽 rho」那条**不适用**，方向正好相反：顶到的不是 tanh 的界，是物理域。

---

## A.2 归因

**（1）`w̄_v` 一身二职形成正反馈。** 它同时是基准尺度 `A_k = softplus(κ_k)·w̄_v·(1+aH+bE)` 的因子，又是 pinball 损失的归一化分母；而它来自上一级**已经变宽**的轴。两条路都指向「更宽」：轴宽 → A 大 → 更宽；轴宽 → 分母大 → 宽度惩罚显得更小。`w̄_v` 是 detach 的，所以不是单步内可被 game，而是**跨 step** 的慢反馈。

**（2）没有任何约束保证级联细化。** pinball 只关心「区间盖不盖得住 GT」，不关心「盖住之后还剩多少分辨率」。对于 stage1 选错表面的像素，τ=0.92 分位数确实很大，控制器学出巨大窗口在损失意义上是**正确**的 —— 错的是目标函数缺了一项。

**（3）验证漏传 `step`。** `_run_validation` 调用 `model(batch)` 没有传 step，于是 `blend = min(1, step/axis_blend_steps)` 恒取 `step=None → blend=1.0`。训练还在 blend=0.3 的混合轴上，验证已经在纯逆深度轴上 —— 两条曲线量的是两个模型，而且完全静默。这条与窗宽无关，但会污染 v3 全部验收判据。

---

## A.3 为什么不采用半宽单调约束

一个自然的想法是 `h_{k+1} ≤ h_k`。**它约束不到真正要约束的量。**

各级假设数是 D₂=16、D₃=8、D₄=4。父级 D_p、半宽 h 时 bin 间距是 `2h/(D_p−1)`；子级假设数减半、半宽也减半，间距是 `2·(h/2)/(D_p/2−1) = 2h/(D_p−2)`。于是

$$\frac{\Delta v_{\text{child}}}{\Delta v_{\text{parent}}}=\frac{D_p-1}{D_p-2}>1$$

即：**半宽已经减半（满足半宽单调），bin 间距反而更粗**（D_p=16 时 ×1.071，D_p=8 时 ×1.167）。真正要约束的是间距本身：

$$\Delta v_{\text{child}} \le \xi_k\,\Delta v_{\text{parent}},\qquad 0<\xi_k<1$$

此外 `torch.minimum(h, h_prev)` 是一道**硬夹**，而 v3 当初要拆掉的正是 `max()` 那种不可微悬崖（实测把 78–91% 的像素钉在常数上）。用硬夹换硬夹只是把病换个位置。

**决策：不采用半宽单调，不做三个修法的小消融。** 正式 W1 同时采用固定 pinball 归一化、子级候选间隔约束、逐级倍率安全界。

---

## A.4 修改内容

### A.4.1 验证 `step`（`train.py`）

`_run_validation` 增加 `step` 与 `blend_steps` 参数，前向改为 `model(batch, step=step)`，两个调用点都传当前 step。独立的 `test.py` 不传（`step=None → blend=1.0`）是对的：部署用最终的完整逆深度轴。

新增 `val/ctrl_blend` 指标，并在验证首个 batch 断言

$$\bigl|\lambda_{\text{output}}-\min(1,\ \text{step}/\text{axis\_blend\_steps})\bigr|<10^{-6}$$

`legacy_depth` 模式没有 `ctrl_blend`，跳过断言。

### A.4.2 pinball 归一化改用全局尺度（`losses/composite.py`）

**两个尺度不是一回事，必须分开成两个变量**：

| 变量 | 取值 | 用途 |
|---|---|---|
| `w_parent` | 上一级 winner 的局部间距 `w̄_v` | center loss 的定义域 |
| `w_loss` | 全局逆深度间距 `g_v` | pinball 的归一化分母 |

pinball：

$$\mathcal L_{\text{pin}}=Q_\alpha\!\Bigl(\tfrac{v_{gt}-v_L}{w_{\text{loss}}}\Bigr)+Q_{1-\alpha}\!\Bigl(\tfrac{v_{gt}-v_H}{w_{\text{loss}}}\Bigr)$$

三级都用同一个每图全局尺度 `g_v`，不再受上一级窗宽影响 —— 这切断 A.2(1) 的第二条反馈路径。

center loss **继续用 `w_parent`**：

$$\mathcal L_{\text{center}}=M_c\cdot\operatorname{SmoothL1}\!\Bigl(\tfrac{v_c-v_{gt}}{w_{\text{parent}}+\varepsilon}\Bigr),\qquad M_c=\mathbb 1\bigl[|v_{gt}-v_m|\le w_{\text{parent}}\bigr]$$

**不能**把这里也换成 `g_v` —— center head 的职责就是修正「一个父级 bin 以内」的量化误差，换尺度等于改了它的定义域。（这正是首版实现的错误：把 `w̄_v` 整体替换成 `w_fixed`，连 center loss 一起换了。）

开关：`LossConfig.pinball_scale ∈ {axis, global}`，默认 `axis`（旧行为）。

### A.4.3 子级候选间隔约束（`models/range_controller.py`）

新增可学习参数

$$\xi_k=\sigma(\text{refine\_ratio\_raw}_k)\in(0,1)$$

结构性保证后级永远不比父级粗，同时保留可学习性。初值取 legacy 范围策略下的最大 child/parent bin 比：

$$\frac{2\cdot 1.5\cdot 2}{15}=0.4,\qquad \frac{2\cdot 0.9\cdot 2}{7}\approx 0.5143,\qquad \frac{2\cdot 0.6\cdot 2}{3}=0.8$$

其中的 2 来自 `(1 + aH + bE) ≤ 1 + 0.5 + 0.5`。

**计算顺序固定**，不可调换：

1. 计算 `h₀`
2. 计算 controller 原始倍率
3. 得到 `h⁻_raw, h⁺_raw`
4. **子级间隔约束**
5. 物理深度域缩放
6. 夹持中心
7. 生成候选轴

第 4 步必须在第 5 步**之前** —— 否则「级联精度约束」与「场景边界约束」两种 binding 混在一起，诊断上分不开。

约束的是**实际相邻候选的最大逆深度间隔**而非半宽（左右不等宽时，更宽的一侧决定最粗的 bin）：

```
t          = linspace(-1, 1, D).view(1, D, 1, 1)
offset_raw = where(t < 0, t * h_lo_raw, t * h_hi_raw)     # [B,D,H,W]
gap_raw    = (offset_raw[:,1:] - offset_raw[:,:-1]).abs().amax(1)   # [B,H,W]
```

平滑上界（log 域，避免溢出）：

$$\Delta v_{\text{cap}}=\xi_k\,w_{\text{parent}},\qquad q=\exp\!\Bigl(-\tfrac{1}{p}\operatorname{softplus}\bigl(p\log\tfrac{\Delta v_{\text{raw}}}{\Delta v_{\text{cap}}}\bigr)\Bigr)$$

行为：`gap ≪ cap → q ≈ 1`（不干预）；`gap ≫ cap → q ≈ cap/gap`（正好压到上界）。最终 `h⁻ = h⁻_raw·q`，`h⁺ = h⁺_raw·q`，于是

$$\Delta v_{\text{child,max}}\le \xi_k\,w_{\text{parent}}<w_{\text{parent}}$$

关闭时 `q ≡ 1`，完全不干预（W0 与历史 checkpoint 行为不变）。

`w_bar` 在进入控制器前 detach —— 它是上一级轴的几何量，不该从这一级的范围损失往回收梯度。

**与 axis blend 的关系**：interval cap 只作用于完整 inverse 轴；实际匹配轴仍是 `(1−λ)·legacy + λ·inverse`。因此 λ<1 时混合轴不要求严格满足 cap，λ=1 后必须满足；验收 interval ratio 时**只统计 `ctrl_blend == 1` 的阶段**。

### A.4.4 逐级倍率安全界

`rho_stages = (8, 4, 2)`，对应 `m₂∈[1/8,8]`、`m₃∈[1/4,4]`、`m₄∈[1/2,2]`。

它**只**限制 controller 相对 `h₀` 的倍率，**不负责**保证级联细化 —— 那是 child interval cap 的职责。两者职责不同，不要混。优先级：`rho_stages if not None else (rho_max,)*3`。

### A.4.5 诊断

`ctrl_half_p50/p90` 用一阶换算 `h_d ≈ h_v/v_c²`，大窗口下误差明显，标记 **deprecated**（保留仅为历史曲线可比）。改用精确端点：

$$d_{\text{far}}=\frac{1}{v_c-h^-},\quad d_{\text{near}}=\frac{1}{v_c+h^+},\quad h_{\text{depth,eq}}=\frac{d_{\text{far}}-d_{\text{near}}}{2}$$

stage2–4 每级新增：`ctrl_exact_half_mm_p50/p90`、`ctrl_A_over_B_p50/p90`、`ctrl_h0_over_gv_p50/p90`、`ctrl_mult_raw_p50/p90`、`ctrl_refine_ratio`、`ctrl_gap_ratio_raw_p50/p90`、`ctrl_gap_ratio_final_p50/p90/p99`、`ctrl_gap_cap_bind_frac`、`ctrl_physical_bind_frac`、`interval_mm_p50/p90`。

其中

$$r_{\text{raw}}=\frac{\Delta v_{\text{child,raw,max}}}{w_{\text{parent}}},\qquad r_{\text{final}}=\frac{\Delta v_{\text{child,final,max}}}{w_{\text{parent}}}$$

`gap_cap_bind_frac = mean(q < 0.999)`；`physical_bind_frac` 在 cap **之后**、物理缩放**之前**统计 `mean(h⁻+h⁺ > v_max−v_min)`。这样才能分开三件事：controller 想要多宽、cap 拦了多少、最后还有多少触发物理边界。

### A.4.6 保持不变

SPRE 累乘门、stage4 MAP + 逐候选残差、OOR 方向损失，全部不动。

---

## A.5 CLI 与 arm

新增 `--pinball-scale {axis,global}`、`--child-interval-cap {on,off}`、`--refine-ratio-init`、`--refine-cap-p`、`--rho-stages`。

校验规则：`len(refine_ratio_init)==3` 且全部落在 (0,1)；`len(rho_stages)==3` 且全部 >1.0；`refine_cap_p >= 2.0`。

`ARM=w1` / `ARM=w3` 显式带上：

```
--pinball-scale global
--child-interval-cap on
--refine-ratio-init 0.4,0.5142857142857142,0.8
--refine-cap-p 16
--rho-stages 8,4,2
```

`ARM=w0` 不加任何新参数，继续走 legacy_depth。全局默认 `child_interval_cap=False`、`pinball_scale=axis`，保证旧 checkpoint 与 W0 行为不变。

---

## A.6 验收条件

### A.6.1 代码级硬条件

`blend=1` 且 cap 打开时：`stageK/ctrl_gap_ratio_final_p99 <= 1.001`；所有 hypotheses 沿深度维非降序且落在 `[depth_min, depth_max]`；所有 loss/梯度有限；`refine_ratio_raw` 拿到有限且非零梯度。

legacy 模式在相同 checkpoint/batch 下保持原输出：depth / prob / loss 的 max abs diff < 1e-5（BF16 下可按实测放宽到 1e-4，但**必须写进测试**，不能口头判断）。

### A.6.2 训练级健康条件

blend 完成后连续 500 步：`stage4 physical_bind_frac < 0.01`；`stage4 gap_ratio_final_p99 <= 1.001`；stage2/3/4 的 interval 不再逐级上升；`grad_nonfinite_frac == 0`。

**关于 `gap_cap_bind_frac` 的口径修订（实测后补）。** 原判据「`gap_cap_bind_frac > 0.8` 连续 1000 步 = 控制器长期要求违反级联细化」会**误报**。原因是 ξ_k 的初值取的是 legacy 的**最大** child/parent 比，而控制器初值又复现 legacy —— 于是中位数从第一步就恰好贴在界上。6 步实测：`gap_ratio_raw_p50` 分别是 0.4023 / 0.4997 / 0.7958，对应 ξ = 0.4 / 0.5143 / 0.8，`bind_frac ≈ 1.0` 而 p50 的实际压缩只有 3–5%。

所以判据改看**压缩幅度**而不是 bind 比例，新增 `ctrl_gap_cap_q_p50` / `ctrl_gap_cap_q_p10`：

* `q_p50 ≳ 0.95` → 只是贴边，正常，**不是**告警；
* `q_p50 < 0.7` 连续 1000 步 → 控制器中位数都在要求违反级联细化，才是真告警。

真告警时**不放宽 cap**，应优先检查范围中心、W2 双模态与 controller 输入。若 coverage 下降而 cap 大量实压，不能靠加宽 stage4 解决，应把恢复责任前移到 stage2 / W2。

### A.6.3 模型性能验收

与相同 batch / LR / seed 的 W0 30k 比较：val abs_err 改善必须超过已测 seed 波动；q2、q3 必须改善；q0、q1 不得出现超噪声的退化；stage4 interval ≤ stage3 的父级局部 interval；stage4 MAP-nearest-bin 一致率高于当前约 53%；最终 W3 再做固定协议点云终审。

**不能只因 coverage 更高就判定成功。**

---

## A.7 实施与训练顺序

1. 清理 `half_monotonic` 实现
2. 实现 global pinball + child interval cap
3. 修复 validation step
4. legacy golden test
5. synthetic interval-cap 单元测试
6. 200–500 步有限值 / 梯度健康检查（**不做性能排名**）
7. 从头训练 W0 30k
8. 从头训练修订版 W1 30k
9. W0 best 做尾部统计，决定 W2
10. W1 通过后再从头训练 W3

正式训练统一：`PER_GPU_BATCH=4`、同一 LR 缩放规则、同一 seed、同一 30k horizon、BF16、**单卡**。

batch=5 的那个 W1 checkpoint **只用于诊断，不能 resume 成正式 W1** —— batch 变了、LR 变了、范围控制结构变了，resume 会把两个训练协议混在一起。

---

## A.8 结论

控制器仍然可以学习中心、左右非对称范围、SPRE 可靠性和条件分位数；但无论如何不能再让四假设的 stage4 比八假设的 stage3 更粗。这样既保留了可学习性，也把「级联必须逐级提高采样分辨率」从一条隐含前提变成了显式、可验证的结构约束。
