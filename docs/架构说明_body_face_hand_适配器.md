# 人物姿态编辑项目 · 完整架构说明

> 面向读者：需要理解本项目架构的协作者/AI。
> 所有内容均从远端实际代码核验（`ssh sg` → `/home/shangguanrz/project/pic-edit`），非推测。
> 文档生成时间：2026-10-08

---

## 0. 一句话概括

本项目在**两个冻结的扩散基模**上，各自外挂一套**可训练的姿态控制适配器**：给定源人物图像 A 与目标姿态参考 B，生成"保持 A 的身份、服装、体型与背景，但姿态换成 B"的图像。适配器**不修改基模任何权重**，只通过 **block 输出的残差注入** 干预去噪轨迹。

**两套并行体系**：

| | 体系一 | 体系二 |
|---|---|---|
| 基模 | DeepGen-1.0-diffusers | Qwen-Image-2.1 |
| 适配器 | V6.6（5 条分支） | qwen21（body / face / dual） |
| 代码 | `src/pose_control/v6/` | `src/pose_control/qwen21/` |
| 注入点 | block 0–22（block 23 排除） | block 3/8/13/18/23/28 |
| 状态 | 已完成，作为对照基线 | **当前主力**，效果更优 |

---

## 1. 基模规格（均冻结，不训练）

### 1.1 DeepGen-1.0-diffusers

| 项 | 值 |
|---|---|
| Transformer | `SD3Transformer2DModel`（MMDiT 架构） |
| 层数 | 24 |
| 参数量 | ≈ 2.47 B |
| VAE 下采样 | **8×** |
| 最后一层 | `context_pre_only=True`（**block 23 不注入残差**） |
| 注入接口 | 接受 `block_controlnet_hidden_states` 关键字参数 |
| 文本/VLM | Qwen2.5-VL-3B |

### 1.2 Qwen-Image-2.1

| 项 | 值 |
|---|---|
| Transformer | 单流 MMDiT（**非**双流，这是与 DeepGen 最本质的差异） |
| 层数 | **32**（硬校验，见 §5.3） |
| hidden size | **4096** |
| VAE 下采样 | **16×**（1024² 输入 → 64×64 token 网格） |
| VAE latent 通道 | **64** |
| 模型权重 revision | 钉死 `790c92633540aa0cb11d9abf19eb46d861714758` |
| 支持分辨率 | 512 或 1024（其余报错） |
| 环境 | 独立 conda env `nihaochen/envs/qwen21`（diffusers 0.41.0.dev0、transformers 5.17.0） |

> ⚠️ **环境隔离铁律**：DeepGen 所在的 `deepgen` 环境（diffusers 0.35.2）**无法**加载 Qwen-Image-2.1，
> 但**严禁就地升级**（会波及 SD3 全链路）。两套体系必须用两个独立环境。

---

## 2. 条件契约（适配器的输入张量）

### 2.1 几何条件（Body / 主体分支用）

| 名称 | 形状 | 取值范围 | 含义 |
|---|---|---|---|
| `normal` | `[B, 3, H, W]` | `[-1, 1]` | 表面法线图 |
| `pose_heatmap` | `[B, 25, H, W]` | `[0, 1]` | 25 关键点高斯热力图 |
| `part_onehot` | `[B, 14, H, W]` | 二值 one-hot | 14 类人体部位掩码（背景全零） |
| `smplx_global` | `[B, 26]` | 浮点 | **betas[10] + rot6d[6] + transl[3] + cam[7]** |
| `human_mask` | `[B, 1, H, W]` | `[0, 1]` | 人体前景掩码 |
| `depth` | `[B, 1, H, W]` | 浮点 | 深度图（**qwen21 未启用**，`use_depth=False`） |
| `task_id` | `[B]` | int64 | 任务枚举（见下） |

**14 类部位的确切名称**（源码 `PART_NAMES`，顺序即通道顺序）：

```
0: head          1: torso
2: left_upper_arm      3: left_lower_arm      4: left_hand
5: right_upper_arm     6: right_lower_arm     7: right_hand
8: left_upper_leg      9: left_lower_leg     10: left_foot
11: right_upper_leg   12: right_lower_leg    13: right_foot
```

**任务枚举 `TaskType`（IntEnum）**：

```
SINGLE = 0    DUAL = 1    HUG = 2        HANDSHAKE = 3
SHOULDER_HUG = 4    HOLD_HAND = 5    UNKNOWN = 6
```

**配对约束**（`condition_injector` 强制）：
- 单人行（person_count==1）必须用 `SINGLE` 或 `UNKNOWN`
- 双人行（person_count==2）**不得**使用 `SINGLE`

**硬性校验**（`SMPLXConditionInjector` 内，违反即抛错）：
- `H % 8 == 0 and W % 8 == 0`
- `normal` 必须在 `[-1, 1]`
- `pose_heatmap` 必须在 `[0, 1]`
- `part_onehot` 必须严格二值且 one-hot（`sum(dim=1) ≤ 1`）
- 所有张量必须 finite、同 device、同 dtype
- person-B 的 4 项核心输入（normal/pose_heatmap/part_onehot/smplx_global）**要么全给要么全不给**

### 2.2 ⚠️ 关键事实：`smplx_global[26]` 不含手脸

构造源码（`scripts/build_tiktok_v6_conditions.py:349`）：

```python
smplx_global_26d = torch.cat([betas_all[:, :10], rot6d_all, transl_all, cam_7d], dim=-1)  # [N,26]
```

即 **只有体型 + 全局朝向 + 位移 + 相机**，**没有手指姿态、没有表情**。
手脸信息由**独立分支**（face_adapter / hand_adapter）以各自的条件通路提供，不走这 26 维。

---

## 3. V6 体系（DeepGen 主干）

### 3.1 五条分支

| 分支 | 引入版本 | 作用 |
|---|---|---|
| `geometry` | V6.3 | 人体几何（法线/部位/热力图）驱动姿态 |
| `interaction` | V6.3 | 双人相对几何与接触 |
| `detail` | V6.4 | 细节增强 |
| `face` | V6.5 | 面部身份保持（ArcFace + DINOv2） |
| `hand` | V6.6 | 手部精细控制（ROI + 逆映射 + 深度加权） |

架构版本权威来源：`src/pose_control/v6/checkpoint.py` 的 `ARCHITECTURE_VERSION = "v6.6"`。
（`configs/adapter_v6_architecture.json` 与 `docs/adapter_v6_architecture.md` 已滞后，**勿照抄**。）

### 3.2 六组 × 24 层的映射（核心机制）

`DeepGenControlInterface.active_block_mapping` 的实现逻辑：

```python
interval = num_transformer_layers / num_stages   # 24 / 6 = 4
for block in range(24):
    if block in context_pre_only_blocks:          # 排除 block 23
        continue
    groups[int(block / interval)].append(block)
```

结果：

| 组 | 覆盖 block |
|---|---|
| 0 | 0, 1, 2, 3 |
| 1 | 4, 5, 6, 7 |
| 2 | 8, 9, 10, 11 |
| 3 | 12, 13, 14, 15 |
| 4 | 16, 17, 18, 19 |
| 5 | 20, 21, 22（**23 被排除**） |

> **为什么排除 block 23**：该层 `context_pre_only=True`，其输出语义与其余层不同，注入会破坏契约。

### 3.3 残差如何进入基模

适配器产出 6 组残差 → 通过 `block_controlnet_hidden_states` 关键字参数传给 DeepGen transformer →
由 transblock 在对应层加上去。入口在 `src/pose_control/v6/controlled_pipeline.py:201`：

```python
kwargs["block_controlnet_hidden_states"] = output.block_controlnet_hidden_states
```

**防重复校验**：若外部已传入该参数，直接抛错（`would duplicate control`）。

### 3.4 强度控制（`ControlStrengthController`）

这是**调控生成效果的主要旋钮**，共三层机制：

**(a) 每分支 × 每组 的可学习正增益**（log 空间，初始化全零 → 初始等价于 1.0）：

```python
self.geometry_log_group_scale    = nn.Parameter(torch.zeros(6))
self.interaction_log_group_scale = nn.Parameter(torch.zeros(6))
self.detail_log_group_scale      = nn.Parameter(torch.zeros(6))
self.face_log_group_scale        = nn.Parameter(torch.zeros(6))   # 仅 enable_face
self.hand_log_group_scale        = nn.Parameter(torch.zeros(6))   # 仅 enable_hand
```

**(b) 去噪时间步调度**（`StrengthScheduleConfig`，三种曲线 `constant` / `linear` / `cosine`）。
默认值（即"何时开始施加控制"的设计意图）：

| 分支 | 曲线 | 生效进度区间 | 起始倍率 → 结束倍率 |
|---|---|---|---|
| geometry | constant | 0.0 – 1.0 | 1.0 → 1.0 |
| interaction | constant | 0.0 – 1.0 | 1.0 → 1.0 |
| **detail** | cosine | **0.35 – 1.0** | 0.0 → 1.0 |
| **face** | cosine | **0.35 – 1.0** | 0.0 → 1.0 |
| **face_texture** | cosine | **0.65 – 1.0** | 0.0 → 1.0 |
| **hand** | cosine | **0.0 – 1.0** | **0.15** → 1.0 |
| **hand_texture** | cosine | **0.65 – 1.0** | 0.0 → 1.0 |

> **解读**：`progress` 是去噪进度（0 = 纯噪声，1 = 完全去噪）。
> 几何条件从头就全程生效；**细节/面部等到 35% 才开始介入**，纹理更晚（65%）——
> 这是"先摆好姿态，再补细节"的显式设计，避免早期噪声阶段被细节条件带偏。

**(c) 面部组先验**（`face_group_prior`，固定 buffer，非可学习）：

```python
torch.tensor([0.1, 0.2, 0.5, 0.8, 1.0, 1.0])
```

> **解读**：面部控制**浅层几乎不施加、深层全开**。浅层管全局结构（姿态），
> 深层管细节（五官），面部是细节所以放在深层。

### 3.5 Face 分支（V6.5）

`FaceControlAdapter(face_dim=512, deepgen_dim=1536, num_stages=6, rank=64, heads=8)`

**输入**（`PreparedFaceConditioning`）：
- `identity_tokens`、`texture_tokens`、`geometry_tokens`：各 16 token
- `spatial_features`：`[B, 2, 512, 16, 16]`
- `face_masks`、`target_boxes` `[B,2,4]`、`face_valid` `[B,2]`

**核心块 `FaceAttentionBlock`**（继承 `third_party.v65_face.visual_persona.IPAttnProcessor2_0`）：
1. ROI 内自注意力（`self_attention`）
2. **独立的 face-token K/V 通路**（`face_attention`）——不是简单拼接，而是独立 KV
3. FFN

**残差落位 `scatter_face_roi_residuals`**：
把 `[B,2,Hroi,Wroi,D]` 的双人 ROI 残差用 `grid_sample` **逆映射**回目标 token 网格 `[B,2,16,16,D]`，
两人重叠区用"加权平均 × 最大掩码"混合 —— **重叠不会让强度翻倍**（代码注释明确写了这条不变量）。

### 3.6 Hand 分支（V6.6）

`HandControlAdapter(dim=512, deepgen_dim=1536, rank=64, heads=8)`

**输入**（`HandFineCondition`）：
- `mesh_bps` `[B,2,2,1024]`（每只手 1024 个基姿态系数）
- `depth` `[B,2,2,1,16,16]`、`visibility` `[B,2,2]`、`mesh_valid` `[B,2,2]`

**处理**：
1. 两轮 `HandFusionBlock`（几何 query + 外观 KV 的独立注意力）
2. 6 个 stage，每 stage：stage embedding + timestep FiLM + 自注意力 + LoRA 风格 `_StageAdapter`
3. 输出 `[B,2,2,16,16,1536]` 局部残差

**落位 `scatter_hand_roi_residuals`**（比 face 更复杂）：
- 4 个 ROI（2 人 × 2 手）逆映射到目标网格
- `weight = sampled_mask × valid × visibility`
- **深度加权**：`blend_weight = weight × exp(-4 × depth)` —— 深度改变的是"相对归属"，
  单只手仍保留自身掩码强度
- 归一化：`sum / clamp_min(1e-6)` ⇒ 重叠不叠加

---

## 4. Qwen21 体系（当前主力）

### 4.1 设计难点与解法

**难点**：Qwen-Image-2.1 是**单流** MMDiT —— 文本 token、参考图 token、目标图 token
在同一个序列里一起做注意力。若像 DeepGen 那样对整段 hidden states 加残差，
会污染文本与参考图的注意力表示。

**解法（本项目核心创新点）**：**只对 target token 段加残差**。

```python
prefix = output.shape[1] - self.target_tokens
target = output[:, prefix:]                    # 只取目标段
controlled = target + strength * delta         # 只在目标段加
return torch.cat((output[:, :prefix], controlled), dim=1)   # 前缀原样返回
```

**text + 参考图 prefix 完全不动**，从而化解了"单流架构无法简单接 ControlNet"的难题。

### 4.2 Body 适配器（`Qwen21BodyAdapter`）

```python
Qwen21BodyAdapter(width=512, hidden_size=4096)
```

**复用 V6 的推理内核**（不重写）：
- `SMPLXConditionInjector(use_depth=False)`
- `SMPLXAdapterReasoner`
- `ReasonerControlBridge(reasoning_dim=512, geometry_channels=128, token_dim=1536)`

并提供权重重用接口：

```python
def load_v6_geometry(self, checkpoint):
    prefixes = ("condition_injector.", "reasoner.", "condition_bridge.")
    # 只加载 shape 完全匹配的键，strict=False
```

**注入层**：`LAYER_INDICES = (3, 8, 13, 18, 23, 28)`（32 层中取 6 层，均匀分布）

**前向**：

```
encode_geometry:
  injector → bundle → reasoner → bridge → geometry_scene [B,128,h,w]
  → F.interpolate → (h,w) → Conv2d(128→512) → flatten(2).transpose(1,2) → [B, h*w, 512]

residual(layer_index, target_hidden, geometry, sigma):
  state = hidden_in(target_hidden)      # Linear(4096→512)
        + geometry                       # [B,h*w,512]
        + time_in(sigma)                 # 时间步嵌入
        + stage_embedding[stage]         # 层身份嵌入
  delta = zero_heads[str(layer_index)](fusion(state))   # → [B,h*w,4096]
```

**零初始化不变量**：

```python
for head in self.zero_heads.values():
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
```

⇒ 训练开始时 **控制 ON ≡ OFF**（输出完全相同）。这是可验证的接口不变量，
训练脚本里专门有 `verify_zero_control()` 做检查。

### 4.3 Face 适配器（`Qwen21FaceAdapter`）

```python
FACE_LAYERS = (3, 8, 13, 18, 23, 28)     # 与 body 相同
Qwen21FaceAdapter(width=512, hidden_size=4096)
```

**输入仅 4 项**（比 V6 的 face 分支大幅简化）：
- `embedding` `[B,512]` —— **ArcFace 身份特征（来自 A）**
- `crop` `[B,3,128,128]` —— **A 的人脸裁剪图**
- `target_box` `[B,4]` —— 目标脸在画面中的归一化框（**投影几何得到，非真值**）
- `valid` `[B]` —— bool

**⚠️ 严格约束：目标 RGB 与目标身份特征从不作为输入**，只作监督/离线评测。

**处理流程**：

```python
code = identity_projection(embedding) + texture_encoder(crop)   # 身份 + 纹理
code = code * valid[:, None]

# 由 target_box 在 token 网格上生成 ROI 掩码
mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & shape_valid & valid

# 4 维位置编码：(local_x, local_y, global_x, global_y)
position = position_projection(features) * mask[..., None]

residual:
  x = hidden_projection(target_hidden) + code + position + time(sigma) + layer_emb
  return zero_heads[layer](fusion(x)) * mask      # ← 掩码乘在最外层
```

⇒ **残差只在目标脸 ROI 内非零**，ROI 外严格为 0。

### 4.4 Dual 适配器（Hi4D 双人交互，`Qwen21DualAdapter`）

继承 `Qwen21BodyAdapter`，**保留 iPER 单人路径不变**，新增双人路径。

**新增组件**（全部零初始化以免破坏已训权重）：
- `dual_global_encoder` —— **独立**的全局编码器
  （注释明确：Hi4D 用 SMPL + 不同相机向量约定，**不能覆盖**已训的 iPER 全局编码器）
- `source_projection` —— 64 通道 source latent → 512
- `person_binder`（`PersonTokenBinder`）—— 把人体几何与外观绑定
- `interaction_heads` + `appearance_heads` —— 两路残差，各 6 个零初始化头

**新增条件**：
- `normal_a/b`、`pose_heatmap_a/b`、`part_onehot_a/b`、`smplx_global_a/b`、`human_mask_a/b`、`depth_a/b`
- `relative_geometry` `[B,11]` —— 双人相对几何
- `contact_raster` `[B,2,H,W]` —— 接触栅格
- `ContactRelationBatch` —— 结构化接触关系
- `source_mask_a/b` —— 双人源掩码
- `task_id ∈ {HUG, DUAL}`

**几何池化**：`_pool_geometry` 把每人 `[N,512]` 的 token 按 2×2 自适应池化（要求 N 是完全平方数）。

**残差合成**：

```python
delta = super().residual(...) * state.target_mask                 # body 基础
if state.appearance_strength:
    delta += appearance_strength * appearance_heads[layer](...) * target_mask
if state.interaction_strength:
    active = interaction_valid × target_mask
    delta += interaction_strength * interaction_heads[layer](...) * active
```

**提供降级接口**：

```python
def geometry_only(self):  # interaction_strength=0, appearance_strength=0
```

---

## 5. 生成如何被调控（控制接口汇总）

### 5.1 上下文管理器（`Qwen21ControlContext`）

这是 Qwen21 侧的**唯一控制入口**，用 context manager 管理 hook 生命周期：

```python
with Qwen21ControlContext(transformer, adapter, geometry,
                          sigma=sigma, target_tokens=tokens,
                          strength=1.0,              # body 强度
                          face_adapter=..., face_state=...,
                          face_strength=0.0,         # face 强度
                          body_face_scale=1.0) as control:
    prediction = transformer(**inputs)[0][:, -tokens:]
```

**挂载的 hook**：

| hook | 挂载点 | 作用 |
|---|---|---|
| `register_forward_pre_hook` | `transformer` 本体 | 从 kwargs 抓 `timestep`（→ sigma）与 `img_shapes`（→ target_tokens） |
| `register_forward_hook` | `transformer_blocks[i]`（i ∈ LAYER_INDICES）**输出** | 加残差 |

**可调参数（生成侧旋钮）**：

| 参数 | 含义 | 默认 |
|---|---|---|
| `strength` | body 残差总强度 | 1.0 |
| `face_strength` | face 残差总强度（0 = 关闭） | 0.0 |
| `body_face_scale` | **脸区内的 body 残差缩放**（∈[0,1]） | 1.0 |

**`body_face_scale` 的巧妙设计**：

```python
if body_face_scale != 1 and face_state is not None and layer_index >= 18:
    delta = delta * (1 - (1 - body_face_scale) * face_state.mask)
```

⇒ 在**深层（≥18）的脸部 ROI 内**主动衰减 body 残差，
让位给 face 分支 —— 防止"body 把脸拽走"。

### 5.2 硬性护栏（运行时断言）

| 位置 | 断言 | 目的 |
|---|---|---|
| `__enter__` | `len(transformer.transformer_blocks) == 32` | 防止换基模 |
| `__enter__` | `_pic_edit_qwen21_control_active` 标志 | 防止重复挂载 |
| `_after_block` | `output.shape[1] >= target_tokens` | 防止 token 段不匹配 |
| `residual` | `target_hidden.shape[-1] == hidden_size` | 防止 hidden 宽度不符 |
| `residual` | `geometry.shape[:2] == target_hidden.shape[:2]` | 防止网格不匹配 |
| `runtime` | `RESOLUTION ∈ {512, 1024}` | 防止非法分辨率 |
| `runtime` | `tokens == (resolution//16)**2 and tokens % 4` | 防止 token 数异常 |

### 5.3 训练时的调控（`train_iper_qwen21_body.py`）

**加噪 / 目标**（rectified flow 风格）：

```python
sigma = torch.rand(B)                                   # 均匀采样 [0,1)
noisy = (1 - sigma) * target + sigma * noise
velocity = noise - target                               # ← 预测目标
```

**损失**（人体区域加权 MSE）：

```python
mask = interpolate(human_mask, (grid, grid))
weight = (1 + 2 * mask).clamp(1, 3)                     # 人体区权重 3×，背景 1×
loss = ((prediction - velocity)**2 * weight).sum() / (weight.sum() * C)
```

**双学习率**（区分"迁移来的权重"与"新增的零初始化头"）：

| 参数 | 值 | 适用 |
|---|---|---|
| `--transferred-lr` | `5e-5` | 从 V6 迁移的 geometry 权重 |
| `--new-lr` | `1e-4` | 新增的零初始化头 |
| `--max-steps` | 3000 | 优化器步数（非 micro-step） |
| `--micro-batch-size` | 1 | 每对 |

**零控制验证**：`verify_zero_control()` 在训练前确认 `sigma=0.5` 时控制 ON 与 OFF 输出一致。

---

## 6. 数据流水线

### 6.1 iPER（主训练集）

```
datasets/iPER/  (18 GB)
├── iper_raw_1024/                    原始 1024 分辨率帧
├── iper_sampled_6src64tgt/          采样后的配对
│   ├── <appearance>/                101 个 appearance
│   ├── samples.jsonl
│   └── splits/
│       ├── train_pairs.jsonl         30,336 对
│       ├── test_pairs.jsonl           7,680 对
│       ├── dev_train_pairs.jsonl     26,496 对
│       └── dev_val_pairs.jsonl        3,840 对
└── iper_assets_512_nvdiffrast/      离线渲染的几何资产
    └── <appearance>/
        ├── normal_512/{source_motion,source_standard,target}/
        ├── depth_npy_512/  depth_vis_512/
        ├── dwpose_512/  mask_512/  part_512/  rgb_512/
        └── v6_conditions.pt          smplx_global[70,26] + kps_25_coords[70,25,3]
```

**配对语义**：`appearance` ≈ **同一人同一套服装**；同一 appearance 内不同帧 = **同人同衣的跨姿态配对**
⇒ 天然满足"ID=ID_A、Cloth=Cloth_A、Shape=Shape_A、Pose=Pose_B"的 C 定义。

几何资产生成方式：**SMPLest-X 离线拟合 → nvdiffrast 渲染**（这套管线可复用于新数据集）。

### 6.2 Hi4D（双人交互）

`datasets/Hi4D-origin/`（359 GB，**仍是未解压的 tar.gz**）

### 6.3 TikTok（身份多样性）

`datasets/TikTokDataset/`（105 GB，含 `TikTok_3d_assets_native`）
配套 v3 流水线：三道质量闸（模糊 / 姿态冗余 / 视觉质量）+ 严格配对 + 缺失几何回填。

---

## 7. 不变量与铁律（改代码前必读）

| # | 不变量 | 后果 |
|---|---|---|
| 1 | **残差头零初始化** → 初始化时 OFF ≡ ON | 破坏则预训练权重被随机扰动 |
| 2 | **只对 target token 段加残差**，text+参考图 prefix 不动 | 破坏则污染文本/参考注意力 |
| 3 | **block 23 不注入**（DeepGen），32 层硬校验（Qwen21） | 破坏则语义错位或直接报错 |
| 4 | **ROI 残差重叠不叠加**（face/hand 的 scatter 都做了归一化） | 破坏则重叠区强度翻倍 |
| 5 | **目标 RGB / 目标身份从不作为条件** | 破坏则训练作弊、推理不可用 |
| 6 | 两个基模环境**严格隔离**，禁止就地升级 deepgen | 破坏则 SD3 全链路失效 |
| 7 | Qwen21 `MODEL_REVISION` 与 `RESOLUTION` 钉死 | 换基模权重会直接踩断言 |

---

## 8. 关键文件索引

```
src/pose_control/v6/                         V6.6（DeepGen 侧）
├── checkpoint.py            ARCHITECTURE_VERSION = "v6.6"
├── conditions.py            TaskType / ConditionBundle / PART_NAMES(14)
├── condition_injector.py    SMPLXConditionInjector（条件校验+编码）
├── geometry_encoder.py      SpatialConditionEncoder / normalize_scene_depth
├── token_encoders.py        GlobalConditionTokenEncoder 等
├── control_core.py          SharedRecurrentControlCore（6 stage）
├── deepgen_adapter.py       DeepGen 侧适配器
├── controlled_pipeline.py   残差注入 pipeline
├── face/adapter.py          FaceControlAdapter
├── face/conditions.py       PreparedFaceConditioning
├── hand/adapter.py          HandControlAdapter
├── hand/conditions.py       HandFineCondition
└── interface/
    ├── deepgen_interface.py DeepGenControlInterface（6组×24层映射）
    ├── strength.py          ControlStrengthController（强度调度）
    └── outputs.py           BranchControlResiduals

src/pose_control/qwen21/                     Qwen21（当前主力）
├── body_adapter.py          Qwen21BodyAdapter + Qwen21ControlContext
├── face_adapter.py          Qwen21FaceAdapter + Qwen21FaceState
├── dual_adapter.py          Qwen21DualAdapter + Qwen21DualState
├── dual_inference.py        双人推理
├── runtime.py               Qwen21IPERCache / transformer_inputs（契约钉死）
└── action_prompts.py        动作提示词

src/data/
├── iper_dataset.py          IPERPoseDataset
└── hi4d_qwen21_dataset.py   Hi4D 双人数据集

训练/评测入口（项目根）
├── train_iper_qwen21_body.py   / eval_iper_qwen21_body.py
├── train_iper_qwen21_face.py   / eval_iper_qwen21_face.py
├── train_hi4d_qwen21_dual.py   / eval_hi4d_qwen21_dual.py
└── eval_iper_four_arm.py       四臂对照评测
```

---

## 9. 已实测的量化结论（供理解设计取舍）

**四臂对照**（iPER **test** 50 对，seed42，同集）
⚠️ 步数不同：DeepGen 40 步 cfg4.0 / Qwen21 30 步 cfg1.0 ⇒ 跨模型只能看趋势

| 指标 | DG off | DG on | Q21 off | Q21 on |
|---|---|---|---|---|
| PCK@0.02 | 0.174 | 0.955 | 0.205 | **0.979** |
| NME | 0.0704 | 0.0068 | 0.0784 | **0.0047** |
| PSNR | 17.35 | 24.54 | 17.55 | **25.91** |
| 人体 MAE | 58.85 | 19.95 | 60.73 | **17.45** |
| 背景 MAE | 9.78 | 6.48 | **2.92** | 5.28 |
| **源脸余弦** | 0.258 | 0.297 | **0.822** | 0.353 |
| 峰值 GiB | 14.51 | 14.53 | 32.76 | 32.76 |

**两个核心洞察**：
1. Qwen21 + 控制比 DeepGen + 控制 PSNR **+1.37 dB**，姿态精度全面更优；
2. **控制使源脸余弦从 0.822 掉到 0.353**（−0.47），跌破 source-target 基线 0.613
   ⇒ **身份保持是当前主矛盾**，这正是 face 分支存在的原因。

**面部四臂**（50 对未见外观，严格可见脸 n=30）：
源脸余弦 0.395 → **+Face+A脸图 0.480**；PCK@0.02 0.979→0.981。
增益主要来自**额外送 A 人脸裁剪图**（+0.085）；Face 分支在双图之上的独立增量仅 **+0.008，且 CI 跨 0**（不稳固）。

**分辨率杠杆**（1024 工作线）：
源脸余弦 strict(n=21) 0.508 → **0.660**，PCK@0.02 0.990。
⇒ **分辨率是迄今最有效的单一杠杆**（身份 +0.15 且姿态不掉），代价是推理 18.9 s/张、显存 39.19 GiB。

---

## 10. 已知瓶颈（设计时需记住）

1. **VAE 下采样是根本约束**：DeepGen 8×、Qwen21 16×（64 通道 VAE）。
   1024² 下仍是 64×64 token 网格 —— **与 SD3 的可寻址分辨率相同**，
   换基模**不会**缓解手/脸落到亚像素级 latent 的问题。
   detail / face / hand 三条分支都是对这个瓶颈的正面解法。
2. **`smplx_global[26]` 不含手脸**：手脸参数目前未进入 body 条件通路，
   face/hand 分支走独立条件。若要强化，需扩展条件维度。
3. **跨分辨率/跨模型对比必须同集同口径**（1024 只有 val、无 test）。
