# UprMVS — 网络架构、数据流与公式细节

> 项目: `uprmvs01`
> 主体: 多视图立体 (MVS) 深度估计,基于 DTU + VGGT 几何先验 + FPN/DINOv3 RGB 特征 + 3-stage cascade cost volume

---

## 1. 目录与模块角色

```
uprmvs01/
├── base/config.py                  所有 dataclass 配置 + 路径 + 三个开关 (MACHINE / TRAIN_PROFILE / --ddp)
├── data/                           数据层 (无网络无反传)
│   ├── dtu.py                      DTUMVSDataset (新), DTUDataset (旧,不动)
│   ├── io.py                       read_pfm / write_pfm
│   ├── camera_utils.py             内参 scale/crop, build_projection
│   ├── pair_utils.py               pair.txt 解析, 视角基线过滤
│   └── transforms.py               多尺度 GT, ImageNet 归一化
├── utils/                          纯几何/工具
│   ├── geometry.py                 homography warp, soft-argmin, depth↔normal
│   ├── vis.py                      depth colormap, save_pointcloud
│   ├── logging_utils.py            TensorBoard, metric meter
│   └── path_utils.py
├── models/                         网络模块 (一文件一步骤)
│   ├── fpn.py                      §2 FPN P2/P3/P4
│   ├── dino_adapter.py             §3 DINOv3 + MLP adapter
│   ├── vggt_prior.py               §4 VGGT 推理 + 置信度
│   ├── geo_fusion.py               §5 F_rgb + α·C·F_geo
│   ├── depth_range.py              §6 动态深度区间
│   ├── cost_volume.py              §7 group-wise correlation
│   ├── anchor_pe.py                §8 3D 锚点位置编码
│   ├── points_alignment.py         §9 极线补洞 (离线)
│   ├── decoder.py                  §10 3D UNet + soft-argmin
│   ├── mvsnet.py                   顶层: 串联以上 + 3-stage cascade
│   └── vggt/ dinov3/ DA3/          外部仓库 (不修改)
├── losses/                         §11 六个 loss + composite
├── engine/                         训练/评估编排
│   ├── trainer.py                  Trainer 类 (含 DDP)
│   ├── evaluator.py                Abs-Rel / RMSE / δ<1.25 / Acc@2mm
│   └── ddp_utils.py
├── scripts/
│   ├── eval.py
│   ├── da3_offline_fill.py         DA3 离线补洞 (不进训练)
│   └── train_umhpc.sbatch
└── train.py                        argparse + Trainer
```

---

## 2. 三个独立开关

| 开关 | 位置 | 控制 | 选项 |
|---|---|---|---|
| `MACHINE` | `base/config.py:12` | 数据/权重绝对路径 | `ubuntu` / `umhpc` / `windows` |
| `TRAIN_PROFILE` | `base/config.py:14` | 训练超参规模 | `local` / `umhpc` |
| `--ddp` | `train.py` CLI | 是否拉起 DDP | `on` / `off` / `auto` |

三者互不耦合: 路径走机器,超参走规模,DDP 走启动方式。

---

## 3. 输入数据 (DTUMVSDataset)

### 3.1 输出 batch dict (本地 profile shape: B=1, V=3, H=512, W=640)

| key | shape | dtype | 说明 |
|---|---|---|---|
| `imgs` | `[B, V, 3, 512, 640]` | float32 | ImageNet normalized |
| `imgs_raw` | `[B, V, 3, 512, 640]` | float32 | `[0,1]` 给 VGGT |
| `intrinsics` | `[B, V, 3, 3]` | float32 | 640×512 frame |
| `extrinsics` | `[B, V, 4, 4]` | float32 | world→cam |
| `proj_matrices` | `[B, V, 4, 4]` | float32 | `K @ E` |
| `depth_min/max` | `[B]` | float32 | DTU 相机文件读 |
| `depth_gt_full` | `[B, 512, 640]` | float32 | ref view 的 GT |
| `depth_gt_multiscale` | dict{4→[B,128,160], 8→[B,64,80], 16→[B,32,40]} | float32 | 每 stage 的 GT |
| `mask_full / mask_multiscale` | 同上 | float32 | DTU `depth_visual` |

### 3.2 1600×1200 → 640×512 几何处理

DTU 原图 4:3,目标 5:4。**按高 scale,横向 center-crop**:

$$
s = \max(H_t / H_{src}, W_t / W_{src}) = \max(512/1200, 640/1600) = 0.4267
$$

resize 到 `(H', W') = (round(1200·s), round(1600·s)) = (512, 683)`,然后 `crop_x = (683 - 640)/2 = 21` 横向居中裁。

**内参同步修改** (`data/camera_utils.py`):

$$
K' = \begin{bmatrix} f_x \cdot s_x & 0 & (c_x + 0.5) s_x - 0.5 - \text{crop}_x \\ 0 & f_y \cdot s_y & (c_y + 0.5) s_y - 0.5 - \text{crop}_y \\ 0 & 0 & 1 \end{bmatrix}
$$

(主点用 `(c + 0.5)·s - 0.5` 而非简单 `c·s`,是为了对齐 OpenCV 像素中心约定)

### 3.3 视角对过滤 (`data/pair_utils.py`)

两个相机中心 $C_i = -R_i^T t_i$ (world frame),场景中心 $\bar{C}$ 取所有相机中心均值。两视角对场景中心的夹角:

$$
\theta = \arccos\left( \frac{(C_{ref} - \bar{C}) \cdot (C_{src} - \bar{C})}{\| C_{ref} - \bar{C}\| \cdot \| C_{src} - \bar{C}\|} \right) \cdot \frac{180}{\pi}
$$

保留满足 $5° \le \theta \le 45°$ 的 src,按 $|\theta - 25°|$ 升序取前 $V-1$ 个。

---

## 4. 整体数据流图

```
┌──────────────────────────────────────────────────────────────────────┐
│  batch (imgs, imgs_raw, K, E, depth_min, depth_max)                  │
└───────┬──────────────────────────────────────────────────────────────┘
        │
        ├─→ VGGT (frozen, no_grad) ─→ prior {depth_sparse, conf, world_pts}
        │                                  │
        ├─→ FPN ──────────────→ {4:P2, 8:P3, 16:P4} (RGB feats per view)
        │                       │
        ├─→ DINOv3 + MLP ─────→ dino_feat (1/8)
        │                       │
        │   concat(P3, dino) ──→ merged_p3
        │                       │
        │              ┌────────┴────────┐
        │              │  GeometryEncoder│ ← prior['depth_sparse'] + 法线
        │              │     F_geo       │
        │              └────────┬────────┘
        │                       │
        │   merged_p3 + α·C·F_geo  (α / λ 由 step 控制 warmup)
        │                       │
        │              ┌────────┴────────┐
        │              │ AnchorPositional│ ← prior['world_points']
        │              │   Encoder       │
        │              └────────┬────────┘
        │                       │
        │   ref_feat += λ(step) · PE
        │                       ↓
        │              ┌──────────────────┐
        │              │ Stage1: CV+UNet  │  D=48, 1/8
        │              └────────┬─────────┘
        │                       │ depth1, prob1
        │                       ↓
        │            refine_range_from_prob
        │                       │
        │              ┌────────┴─────────┐
        │              │ Stage2: CV+UNet  │  D=32, 1/4 (P2)
        │              └────────┬─────────┘
        │                       │ depth2, prob2
        │                       ↓
        │            refine_range_from_prob
        │                       │
        │              ┌────────┴─────────┐
        │              │ Stage3: CV+UNet  │  D=16, 1/4 (P2)
        │              └────────┬─────────┘
        │                       │ depth3
        │                       ↓ ↑bilinear upsample to 640×512
        └─→ depth_full ─→ losses (L1, grad, CE, normal, residual, SSIM, feat)
```

---

## 5. 各步骤公式细节

### 5.1 FPN 特征金字塔 (`models/fpn.py`)

ResNet-50 主干,取 `layer1/2/3` 输出 $C_2, C_3, C_4$ (通道 256/512/1024)。每层 1×1 lateral 投到 128 维,自顶向下融合:

$$
P_4 = \text{Lateral}_4(C_4)
$$
$$
P_3 = \text{Lateral}_3(C_3) + \text{Upsample}_{2\times}(P_4)
$$
$$
P_2 = \text{Lateral}_2(C_2) + \text{Upsample}_{2\times}(P_3)
$$

每层再过 3×3 smooth conv。输出 stride = 4/8/16,均 128 通道。

### 5.2 DINOv3 适配器 (`models/dino_adapter.py`)

**主干冻结**,只训 MLP:

$$
F_{\text{dino}} = \text{LN}(\text{GELU}(\text{Linear}_{768 \to 256}(z))) \to \text{Linear}_{256 \to 128} \to F_{2}\text{-norm}
$$

ViT-B/16 patch_size=16,DTU 输入会先 resize 到 `max_side=512` 且能被 16 整除,得到 patch grid `~32×40`。adapter 之后 bilinear upsample 到 FPN P3 的 `(64, 80)`。最后 channel-wise L2 normalize:

$$
\hat F = \frac{F}{\sqrt{\sum_c F^2 + \varepsilon}}, \quad \varepsilon = 10^{-6}
$$

### 5.3 VGGT 先验 (`models/vggt_prior.py`)

VGGT 一次推理输出:
- `depth` $[B,V,H_v,W_v,1]$ — 每视图深度(在 VGGT 内部坐标系)
- `depth_conf` $[B,V,H_v,W_v]$
- `world_points` $[B,V,H_v,W_v,3]$

置信度归一化(去掉极端值后线性缩放到 `[0,1]`):

$$
\hat C = \text{clip}\left( \frac{C - q_{0.02}(C)}{q_{0.98}(C) - q_{0.02}(C)}, 0, 1 \right)
$$

`valid_mask = $\hat C > 0.2$`。

> **⚠️ 当前实现已知问题**: `_align_scale` 把 `world_points` 直接乘 DTU 的 `extrinsics` 取 z 分量。但 VGGT 的 world frame 不是 DTU 的 world frame,且 scale 任意。这是 10 小时训练不收敛的主因。修复方向:中位数尺度对齐到 `(depth_min + depth_max)/2`,或直接使用 VGGT 的 `depth` 头输出做按视图独立的中位数对齐。

### 5.4 几何特征编码 + 门控融合 (`models/geo_fusion.py`)

输入 sparse depth + 法线 (4 通道):

$$
F_{\text{geo}} = \text{Conv3}(\text{GELU}(\text{GN}(\text{Conv3}(\text{GELU}(\text{GN}(\text{Conv3}([d, n_x, n_y, n_z])))))))
$$

法线由 depth 一阶差分计算(见 §5.10)。融合公式:

$$
F_{\text{fused}} = F_{\text{rgb}} + \alpha(t) \cdot C \cdot F_{\text{geo}}
$$

$\alpha$ 是单标量可学习参数,但训练时 **被 step 强制 clamp**:

$$
|\alpha(t)| \le \begin{cases}
\alpha_{\max,warm} = 0.1 & t < t_w = 10\,000 \\
\alpha_{\max,warm} + \frac{t - t_w}{t_r} (1 - \alpha_{\max,warm}) & t_w \le t < t_w + t_r \\
\infty & t \ge t_w + t_r
\end{cases}
$$

其中 $t_r = 30\,000$ (release steps)。前 10k 步几乎冻结 α≈0,30k 步后完全解锁。

### 5.5 动态深度区间 (`models/depth_range.py`)

**Stage 1 初始化**(从 VGGT 先验):

$$
\sigma_{\max} = 0.15 \cdot (d_{\max} - d_{\min})
$$
$$
\sigma(p) = \sigma_{\max} \cdot (1 - C(p))
$$
$$
\text{halfrange}(p) = \min(k_\sigma \cdot \sigma(p), \, 0.5 \cdot (d_{\max} - d_{\min})), \quad k_\sigma = 3
$$
$$
d_{\text{center}}(p) = \text{clip}(d_{\text{prior}}(p), d_{\min}, d_{\max})
$$

第 $i \in [0, D)$ 个深度假设:

$$
d_i(p) = d_{\text{center}}(p) + \text{halfrange}(p) \cdot \frac{2i - (D-1)}{D - 1}
$$

最后整体 clip 到 $[d_{\min}, d_{\max}]$。

**Stage 2/3 细化**(从上一 stage 的概率体):

$$
P_{\max}(p) = \max_d P(p, d)
$$
$$
\text{span}_{\text{prev}}(p) = d_{\text{hypos,prev}}^{\max}(p) - d_{\text{hypos,prev}}^{\min}(p)
$$
$$
\text{halfrange}_{\text{new}}(p) = \begin{cases}
0.5 \cdot \text{span}_{\text{prev}}(p) & \text{if } P_{\max}(p) < 0.3 \;\;\text{(uncertain)} \\
0.5 \cdot \text{span}_{\text{prev}}(p) \cdot r & \text{otherwise}
\end{cases}
$$

其中 $r = 0.5$ (stage2) / $0.25$ (stage3)。新假设以 $d_{\text{pred,prev}}$ 为中心、$\text{halfrange}_{\text{new}}$ 为半径采 $D_i$ 个。

### 5.6 单应性 warp (`utils/geometry.py::homography_warp_features`)

给定参考视像素 $p_{\text{ref}}$ 和深度假设 $d$,求其在源视的像素位置 $p_{\text{src}}(d)$。

**1) Ray un-projection**

将 ref pixel 反投到 ref 相机系:

$$
\mathbf r(p_{\text{ref}}) = K_{\text{ref}}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad \mathbf X_{\text{ref}}(p,d) = d \cdot \mathbf r(p)
$$

**2) 相对位姿**

DTU 外参定义为 world→cam: $\mathbf X_{\text{cam}} = R \mathbf X_w + t$。所以 ref cam → src cam 的相对变换:

$$
R_{\text{rel}} = R_{\text{ref}} R_{\text{src}}^T, \quad t_{\text{rel}} = t_{\text{ref}} - R_{\text{rel}} t_{\text{src}}
$$

逆变换 src→ref 即 $R_{\text{rel}}, t_{\text{rel}}$,所以 ref→src:

$$
\mathbf X_{\text{src}} = R_{\text{rel}}^T (\mathbf X_{\text{ref}} - t_{\text{rel}})
$$

**3) 投影回 src 像素**

$$
\mathbf p_{\text{src,homog}} = K_{\text{src}} \mathbf X_{\text{src}}, \quad z_{\text{src}} = [\mathbf p_{\text{src,homog}}]_3
$$
$$
u_{\text{src}} = \frac{[\mathbf p_{\text{src,homog}}]_1}{\max(z_{\text{src}}, 10^{-6})}, \quad v_{\text{src}} = \frac{[\mathbf p_{\text{src,homog}}]_2}{\max(z_{\text{src}}, 10^{-6})}
$$

**4) 内参缩放**

由于 feature stride 不是 1,内参也要相应缩放:

$$
K^{(s)} = \begin{bmatrix} f_x / s & 0 & c_x / s \\ 0 & f_y / s & c_y / s \\ 0 & 0 & 1 \end{bmatrix}
$$

(此处用简化版,实际代码用 §3.2 的精确公式)

**5) `grid_sample`**

归一化 uv 到 `[-1, 1]` 后用 bilinear 取样:

$$
\tilde u = \frac{u_{\text{src}}}{W - 1} \cdot 2 - 1, \quad \tilde v = \frac{v_{\text{src}}}{H - 1} \cdot 2 - 1
$$

`padding_mode="zeros"`,出界自动填 0。

### 5.7 Group-wise correlation cost volume (`models/cost_volume.py`)

参考特征 $F_{\text{ref}} \in \mathbb R^{B \times C \times H \times W}$,warped 特征 $F_{\text{warp}} \in \mathbb R^{B \times C \times D \times H \times W}$,$C = 128, G = 8, C/G = 16$:

$$
\text{CV}(p, d, g) = \frac{1}{C/G} \sum_{c=g \cdot C/G}^{(g+1) \cdot C/G - 1} F_{\text{ref}}(p, c) \cdot F_{\text{warp}}(p, d, c)
$$

输出 $[B, G, D, H, W]$。多视角加权融合:

$$
\text{CV}_{\text{agg}}(p, d, g) = \frac{\sum_{v} w_v \cdot \text{CV}_v(p, d, g)}{\sum_v w_v}
$$

当前实现 $w_v = 1$,后续可换为视角可靠性权重。

### 5.8 3D 锚点位置编码 (`models/anchor_pe.py`)

**FPS 锚点选取**: 从 valid 且 conf > 0.7 的 world points 里通过 farthest-point sampling 选 $K=24$ 个。FPS 迭代:

$$
\text{anchor}_0 = \arg\max_{p \in \mathcal{P}} \text{rand}, \quad \text{anchor}_k = \arg\max_{p} \min_{j < k} \| p - \text{anchor}_j \|
$$

**可见性**: 锚点 $A_k$ 投到第 $v$ 个 view 是否落在画面内且 $z > 0$:

$$
V(k, v) = \mathbb{1}[0 \le u_{kv} < W \wedge 0 \le v_{kv} < H \wedge z_{kv} > 0]
$$

**相对位置编码**: ref view 每像素 $p$ 的 3D 位置 $X_p$ (由 prior depth 反投),其相对所有锚点的位移拼接:

$$
\mathbf{r}_p = [A_1 - X_p, A_2 - X_p, \ldots, A_K - X_p] \in \mathbb R^{3K} \quad \text{(乘以可见性掩码)}
$$

过 MLP (3K → 64 → 64) 得 PE。最终融到 ref feature:

$$
F_{\text{ref}}' = \text{Conv}_{1\times1}\left( [F_{\text{ref}}, \lambda(t) \cdot \text{PE}] \right)
$$

$\lambda(t)$ 同 α 的 warmup schedule (前 20k 为 0,之后线性升到 1)。

### 5.9 3D UNet decoder (`models/decoder.py`)

输入 cost volume $[B, G, D, H, W]$,经过 3 层下采样 + 3 层上采样:

```
in[G=8]    →  conv3d→GN→GELU   →  base[16]
↓ stride2  →  base*2[32]
↓ stride2  →  base*4[64]
↓ stride2  →  base*8[128]
↑ deconv2  →  base*4 (skip cat) →  64
↑ deconv2  →  base*2 (skip cat) →  32
↑ deconv2  →  base   (skip cat) →  16
→ Conv3d(16, 1)  → logits [B, D, H, W]
```

**Softmax + 数值稳定 shift**:

$$
\tilde \ell(p, d) = \ell(p, d) - \max_{d'} \ell(p, d') \quad \text{(防 fp16 溢出)}
$$
$$
P(p, d) = \frac{\exp(\tilde \ell(p, d))}{\sum_{d'} \exp(\tilde \ell(p, d'))}
$$

**Soft-argmin** (期望深度):

$$
\hat d(p) = \sum_d P(p, d) \cdot d_{\text{hypo}}(p, d)
$$

**Soft-variance** (不确定度):

$$
\hat \sigma(p) = \sqrt{\sum_d P(p, d) \cdot (d_{\text{hypo}}(p, d) - \hat d(p))^2}
$$

### 5.10 深度→法线 (`utils/geometry.py::depth_to_normal`)

把每个像素反投成 3D camera-frame 点,计算横纵向有限差分,叉乘得法向量:

$$
X(u,v) = K^{-1} [u, v, 1]^T \cdot d(u,v)
$$
$$
\mathbf n(u,v) = \frac{\partial X}{\partial u} \times \frac{\partial X}{\partial v}, \quad \hat{\mathbf n} = \frac{\mathbf n}{\| \mathbf n \|}
$$

偏导用对称差分 $\partial / \partial u \approx (X(u+1) - X(u-1)) / 2$。

---

## 6. 多 stage cascade 数据流详细 shape

| Stage | 输入 feature | 内参 | 深度假设 | Cost volume | Output |
|---|---|---|---|---|---|
| 1 | P3 + DINO + GeoFuse + AnchorPE @ 1/8 `[B,V,128,64,80]` | $K^{(8)}$ | `[B, 48, 64, 80]` (来自 prior 或 global) | `[B, 8, 48, 64, 80]` | `depth1: [B, 64, 80]` |
| 2 | P2 @ 1/4 `[B,V,128,128,160]` | $K^{(4)}$ | `[B, 32, 128, 160]` (来自 stage1 refine) | `[B, 8, 32, 128, 160]` | `depth2: [B, 128, 160]` |
| 3 | P2 @ 1/4 (同 stage2) | $K^{(4)}$ | `[B, 16, 128, 160]` (来自 stage2 refine) | `[B, 8, 16, 128, 160]` | `depth3: [B, 128, 160]` |
| Final | — | — | — | — | `depth_full = upsample(depth3, 4×): [B, 512, 640]` |

注: stage2/3 的 cost volume builder **内部对 K 再除 stride** ($K \to K^{(s)}$),所以传给 builder 的 K 始终是原始 640×512 frame 的内参。

---

## 7. 损失函数细节 (`losses/`)

### 7.1 多 stage 深度损失

每个 stage $s \in \{1,2,3\}$,stride $\in \{8, 4, 4\}$:

$$
L_{\text{depth}}^{(s)} = \frac{1}{|\mathcal{M}|} \sum_{p \in \mathcal{M}} |\hat d(p) - d_{\text{gt}}(p)|
$$

$\mathcal{M}$ 是有效像素 $= \text{mask}_p \wedge (d_{\text{gt}}(p) > 0)$。

**梯度损失**(L1):

$$
L_{\text{grad}}^{(s)} = \frac{1}{|\mathcal{M}_x|} \sum |\partial_x \hat d - \partial_x d_{\text{gt}}| + \frac{1}{|\mathcal{M}_y|} \sum |\partial_y \hat d - \partial_y d_{\text{gt}}|
$$

**Cross-entropy** (强制概率峰值在 GT 假设格上):

$$
\ell_{\text{gt}}(p) = \arg\min_d \, |d_{\text{hypo}}(p, d) - d_{\text{gt}}(p)|
$$
$$
L_{\text{CE}}^{(s)} = -\frac{1}{|\mathcal{M}|} \sum_p \log P(p, \ell_{\text{gt}}(p))
$$

### 7.2 全图损失 (用 `depth_full`)

**法线一致性** (cosine):

$$
L_{\text{normal}} = \frac{1}{|\mathcal{M}|} \sum_p (1 - \hat{\mathbf n}_{\text{pred}}(p) \cdot \hat{\mathbf n}_{\text{gt}}(p))
$$

**VGGT 残差 Laplacian** ($b = 0.1$):

$$
L_{\text{res}} = \frac{1}{|\mathcal{M}|} \sum_p \left( \frac{|\hat d(p) - d_{\text{VGGT}}(p)|}{b} + \log(2b) \right)
$$

> 当前 $b=0.1$ 在 DTU mm 量级下使 $L_{\text{res}} \sim 2000$,会主导梯度。**修复方向**: 改为相对尺度 $|\hat d - d_{\text{VGGT}}| / |d_{\text{VGGT}}|$,或 $b = 0.1 \cdot (d_{\max} - d_{\min})$。

**Reprojection SSIM**: 用 $\hat d$ 把 src image warp 回 ref view,与 ref 算 SSIM。窗口 $7 \times 7$,$c_1 = 0.01^2, c_2 = 0.03^2$:

$$
\text{SSIM}(x, y) = \frac{(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
$$
$$
L_{\text{SSIM}} = \frac{1}{V-1} \sum_{v=1}^{V-1} \frac{1 - \text{SSIM}(I_{\text{ref}}, \text{warp}(I_v, \hat d))}{2}
$$

**DINO 特征余弦** (在 1/8 分辨率,需要 $L_{\text{feat}}$ 时 DINO 特征 cache 自外部传入):

$$
L_{\text{feat}} = \frac{1}{V-1} \sum_v \frac{1}{|\mathcal{M}|} \sum_p \left( 1 - \frac{F_{\text{ref}}(p) \cdot \text{warp}(F_v, \hat d)(p)}{\| F_{\text{ref}}(p)\| \cdot \| \text{warp}(F_v, \hat d)(p) \|} \right)
$$

### 7.3 总损失加权 (`losses/composite.py`)

$$
L = \sum_{s=1}^{3} w_s \cdot \left( w_d L_{\text{depth}}^{(s)} + w_g L_{\text{grad}}^{(s)} + w_d L_{\text{CE}}^{(s)} \right) + w_n L_{\text{normal}} + \phi_{\text{res}}(t) \cdot w_r L_{\text{res}} + \phi_{\text{ssim}}(t) \cdot w_s L_{\text{SSIM}} + \phi_{\text{feat}}(t) \cdot w_f L_{\text{feat}}
$$

- stage 权重 $w_s = (0.5, 1.0, 2.0)$
- $w_d = 1.0, w_g = 0.5, w_n = 0.5, w_r = 0.1, w_{\text{ssim}} = 0.1, w_f = 0.05$
- 阶段门控函数 $\phi(t)$ 为硬开关:
  - $\phi_{\text{res}}(t) = \mathbb 1[t \ge 20\,000]$
  - $\phi_{\text{ssim}}(t) = \mathbb 1[t \ge 20\,000]$
  - $\phi_{\text{feat}}(t) = \mathbb 1[t \ge 50\,000]$

---

## 8. 训练阶段调度

| 步数 | $\alpha$ (geo fusion) | $\lambda$ (anchor PE) | $L_{\text{res}}$ | $L_{\text{SSIM}}$ | $L_{\text{feat}}$ |
|---|---|---|---|---|---|
| 0 ~ 10k | 强制 \|·\| ≤ 0.1 | 0 | off | off | off |
| 10k ~ 20k | 线性放宽到 1.0 | 0 | off | off | off |
| 20k ~ 30k | 完全解锁 | 0 | **on** | **on** | off |
| 30k ~ 50k | 完全解锁 | 线性 0 → 1 | on | on | off |
| 50k+ | 完全解锁 | 1.0 | on | on | **on** |

---

## 9. 学习率调度 (`engine/trainer.py::_lr_at`)

**Warmup** (前 `warmup_steps`):

$$
\eta(t) = \eta_{\max} \cdot \frac{t+1}{\text{warmup}}
$$

**Cosine decay** (warmup 之后):

$$
\eta(t) = \eta_{\max} \cdot \max\left( 0.05, \, \frac{1}{2}\left( 1 + \cos\left( \frac{t - \text{warmup}}{\text{max\_steps} - \text{warmup}} \cdot \pi \right) \right) \right)
$$

最小学习率为 $0.05 \cdot \eta_{\max}$,避免末期完全停学。

---

## 10. AMP 数值稳定策略

整体走 fp16 (省显存),但 **cost volume 构建 + 3D UNet + softmax + soft-argmin 强制 fp32**:

```python
with torch.amp.autocast(device_type='cuda', enabled=False):
    feats_f = feats.float()
    K = intrinsics.float()
    cv = cost_builder(...)
    depth, sigma, prob = decoder(cv, depth_hypos.float())
```

理由: softmax 在 fp16 下当 logits 极大值时 `exp(logit)` 溢出 → `inf/inf = NaN`,污染整个网络。混合精度只保留主干 (FPN/DINO/Adapter) fp16,几何/概率敏感路径强制 fp32。

---

## 11. DDP 启动

**umhpc 4 卡**:

```bash
torchrun --standalone --nproc_per_node=4 train.py --profile umhpc --ddp on
```

**本地单卡**:

```bash
python train.py --profile local --ddp off
```

**`--ddp auto`** 检测 `LOCAL_RANK` env: 有则 DDP on,无则随 profile 默认。

DDP wrapper:

```python
model = DDP(model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,   # geo_fusion/anchor_pe 在无 prior 时无梯度
            broadcast_buffers=False)
```

Rank 0 独占: ckpt 保存、TB 写入、validation、文件日志。所有 rank 在 `ckpt_interval` / `val_interval` 处 `barrier()` 同步。

---

## 12. 评测指标 (`engine/evaluator.py`)

仅对有效像素 $\mathcal{M} = \text{mask} \wedge (d_{\text{gt}} > 0) \wedge \text{isfinite}(\hat d)$:

| 指标 | 公式 |
|---|---|
| Abs-Rel | $\frac{1}{\|\mathcal{M}\|} \sum \frac{\|\hat d - d_{\text{gt}}\|}{d_{\text{gt}}}$ |
| RMSE | $\sqrt{\frac{1}{\|\mathcal{M}\|} \sum (\hat d - d_{\text{gt}})^2}$ |
| δ < 1.25 | $\frac{1}{\|\mathcal{M}\|} \sum \mathbb 1[\max(\hat d / d_{\text{gt}}, d_{\text{gt}} / \hat d) < 1.25]$ |
| Acc@2mm | $\frac{1}{\|\mathcal{M}\|} \sum \mathbb 1[\|\hat d - d_{\text{gt}}\| < 2]$ |
| Comp@2mm | $\frac{1}{\|\mathcal{M}\|} \sum \mathbb 1[\|d_{\text{gt}} - \hat d\| < 2]$ |

---

## 13. 已知问题清单

1. **VGGT scale 错位** — `_align_scale` 用错坐标系,导致 prior depth 不在 DTU 的 mm 量级。10h 训练不收敛的主因。
2. **L_residual 量级过大** — 与 1 联动,$b=0.1$ 在 mm 单位下产生 $\sim 2000$ 的 loss,主导梯度。
3. **SSIM 在异常 depth 下 NaN** — `_ssim` 内 $\sigma^2 = E[x^2] - E[x]^2$ 在数值误差下可能为负,需 clamp 到 0。
4. **GPU 利用率稀疏尖峰** — dataloader 喂不饱,需要提高 `num_workers` 或预解码图像。
5. **legacy `test.py` 引用已删除的 `vggt/utils/`** — 不影响训练,但跑那个脚本会失败。

---

## 14. Shape 速记表

```
imgs            [B, V, 3, 512, 640]
intrinsics      [B, V, 3, 3]
extrinsics      [B, V, 4, 4]

FPN feats:      {4: [B,V,128,128,160], 8: [B,V,128,64,80], 16: [B,V,128,32,40]}
DINO feat       [B, V, 128, 64, 80]
Prior depth     [B, V, 512, 640]
Prior conf      [B, V, 512, 640]
Anchor world    [B, 24, 3]
PE feat (ref)   [B, 64, 64, 80]

depth_hypos1    [B, 48, 64, 80]
cost_volume1    [B, 8, 48, 64, 80]
depth1          [B, 64, 80]

depth_hypos2    [B, 32, 128, 160]
cost_volume2    [B, 8, 32, 128, 160]
depth2          [B, 128, 160]

depth_hypos3    [B, 16, 128, 160]
cost_volume3    [B, 8, 16, 128, 160]
depth3          [B, 128, 160]

depth_full      [B, 512, 640]
```