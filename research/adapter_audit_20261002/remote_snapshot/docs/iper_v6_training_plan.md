# DeepGen Unified SMPL-X Adapter V6.3: iPER 训练实施计划（方案 A）

本文档详细记录基于 **DeepGen V6.3 原生架构** 的 iPER 单人姿态控制训练规范、数据隔离机制、分阶段推进方案及运行手册。

---

## 1. 架构定位与方案背景

### 1.1 为什么废弃旧 `train_iper_adapter.py`
旧训练入口存在根本性架构偏差与数据泄漏隐患：
1. **模型架构未对齐**：旧入口使用旧版 8 通道 `PoseConditionAdapter(in_channels=8)`，只读取 `normal + depth + dwpose + mask`，完全没有调用 `SMPLXConditionInjector`、`ConditionBundle`、`SMPLXAdapterReasoner` 和 `SharedRecurrentControlCore`。
2. **验证集污染**：旧脚本直接加载官方测试集作为 validation，导致测试集参与早停与超参调优。
3. **数据增强不匹配**：水平翻转未同步翻转 26D SMPL-X Global 参数，破坏了几何一致性。

### 1.2 方案 A：V6.3 单人分支分阶段训练（核心定义）
iPER 数据集均为单人序列，不包含双人交互、身体接触和相对几何关系。为避免在缺乏双人监督的情况下破坏 Interaction 分支的先验，方案 A 实施以下隔离与训练策略：
- **训练分支**：Geometry 空间编码、SMPL-X Global 编码、Appearance 身份绑定（PersonTokenBinder）、Dynamic Control Core（共享 Transformer block、时空投影、几何分支自适应层及 6 组 Zero Heads）。
- **冻结分支**：Interaction 分支全模块、Contact 关系推理、相对几何编码器、双人特征融合层（DualFeatureFusion）、桥接层交互投影（interaction_projection / interaction_high_downsample）。
- **主干冻结**：DeepGen DiT 主干（24 层）、SD-VAE、Qwen2.5-VL（14B VLM）及特征连接器（Connector）全部处于 `eval().requires_grad_(False)` 状态。

---

## 2. 数据划分与严格防泄漏机制

### 2.1 数据集分布与隔离规范
iPER 数据集包含 99 个有效 appearance，预处理产物包括 6,930 帧 512 分辨率法线、深度、Part Map（14 分类）及 `v6_conditions.pt`。数据划分如下：

```text
官方全集 (99 身份)
├── 官方训练身份 (79 身份, 30,336 pairs)
│   ├── 开发训练集 (dev_train_pairs.jsonl): 69 身份, 26,496 pairs
│   └── 开发验证集 (dev_val_pairs.jsonl):   10 身份,  3,840 pairs
└── 官方测试身份 (test_pairs.jsonl):        20 身份,  7,680 pairs (调参期间完全隔离)
```

### 2.2 防泄漏硬性约束（自动化测试已验证）
- **身份交集检验**：
  $$\text{dev\_train} \cap \text{dev\_val} = \emptyset$$
  $$\text{dev\_train} \cap \text{test} = \emptyset$$
  $$\text{dev\_val} \cap \text{test} = \emptyset$$
- **增强策略**：新入口强制固定 `augment=False`，保证 26D SMPL-X Global 与 2D Keypoints/Part Map 的严格对齐。

---

## 3. 模型参数拓扑与分层学习率体系

经过严密的参数剖析，`UnifiedSMPLXAdapterV6` 总参数量为 **227.98 M**，在方案 A 下分配如下：
- **可训练参数**：**169.89 M**
- **严格冻结参数**：**58.09 M**

### 3.1 4 层分层优化器配置

| 优化器分组 | 包含核心模块 | 参数量 | 学习率 | 权重衰减 |
| :--- | :--- | :--- | :--- | :--- |
| **Group 1: Zero Heads & 门控** | `control_core.geometry_zero_heads` (6组)<br>`control_interface.strength_controller.geometry_log_group_scale` | 14.165 M | **`1e-4`** | 0.0 |
| **Group 2: 条件编码与推理** | `condition_injector.spatial_encoder`<br>`condition_injector.global_encoder`<br>`condition_injector.task_encoder`<br>`reasoner.person_reasoner`<br>`reasoner.role_embedding`<br>`reasoner.single_geometry_block`<br>`reasoner.single_high_projection`<br>`reasoner.single_high_block` | 31.84 M | **`5e-5`** | 1e-2 |
| **Group 3: 桥接与身份绑定** | `condition_bridge.geometry_projection`<br>`condition_bridge.geometry_high_downsample`<br>`geometry_token_projection`<br>`appearance_token_encoder`<br>`person_token_binder` | 5.63 M | **`5e-5`** | 1e-2 |
| **Group 4: 共享控制核心** | `control_core.shared_block` (DeepGen block 0)<br>`control_core.pos_embed`<br>`control_core.time_text_embed`<br>`control_core.context_embedder`<br>`control_core.geometry_branch.condition_embed`<br>`control_core.geometry_branch.stage_embeddings`<br>`control_core.geometry_branch.stage_adapters` | 118.26 M | **`1e-5`** | 1e-2 |

### 3.2 冻结模块清单（梯度严格为 None）
1. `condition_injector.relative_encoder`
2. `condition_injector.contact_raster_encoder`
3. `condition_injector.contact_relation_encoder`
4. `reasoner.cross_person_reasoner`
5. `reasoner.contact_reasoner`
6. `reasoner.dual_fusion`
7. `condition_bridge.interaction_projection`
8. `condition_bridge.interaction_high_downsample`
9. `control_core.interaction_branch` (全部嵌入层、LoRA适配器及 6 组 Zero Heads)
10. `control_interface.strength_controller.interaction_log_group_scale`

---

## 4. 损失函数与目标加权设计

### 4.1 掩码加权 Flow-Matching 速度损失
DeepGen 采用整流流（Rectified Flow / Flow Matching）建模：
- 目标潜在表示：$z_0 = \text{VAE}(x_{\text{target}}) \in \mathbb{R}^{B \times 16 \times 64 \times 64}$
- 加噪过程：$z_t = (1 - \sigma) z_0 + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad \sigma = \frac{t}{1000}$
- 目标速度向量：$v_{\text{target}} = \epsilon - z_0$

由于背景在姿态编辑中仅需保持静止，过高的背景梯度会稀释人体结构的学习。我们在潜在空间引入人体掩码空间自适应权重矩阵 $W \in \mathbb{R}^{B \times 16 \times 64 \times 64}$：
$$M_{\text{lat}} = \text{Downsample}(\text{human\_mask}) \in [0, 1]$$
$$W = 1.0 + (w_{\text{human}} - 1.0) \times M_{\text{lat}}, \quad w_{\text{human}} = 3.0$$

标准化加权损失：
$$\mathcal{L}_{\text{flow}} = \frac{\sum_{b,c,h,w} W_{b,c,h,w} \cdot (v_{\text{pred}} - v_{\text{target}})^2_{b,c,h,w}}{\sum_{b,c,h,w} W_{b,c,h,w}}$$

该计算方式保证当 $w_{\text{human}}=1.0$ 时完全退化为标准 MSE，且无论人体区域占比如何变化，损失整体数值尺度始终平稳。

---

## 5. 显存与存储安全预算（RTX 6000D / 161GB 磁盘）

### 5.1 显存分配（实测值）
- **RTX 6000D 总显存**：约 85 GiB
- **BF16 混合精度显存实测**：
  - DeepGen Transformer (24层) + VAE: ~12.5 GiB
  - V6.3 Adapter (169.89M 可训练) + AdamW 状态: ~3.0 GiB
  - 激活值（Gradient Checkpointing 开启，Batch=1, Resolution=512）: ~1.0 GiB
  - **训练峰值占用**：**16.50 GiB**
  - **显存余量**：> 68 GiB（极大安全裕度，完全无 OOM 风险）

### 5.2 磁盘存储配额与自动轮换
服务器可用磁盘空间仅余约 161 GB，每个含 Adam 状态的 Checkpoint 约占用 2.3 GB。`CheckpointManager` 实行硬性保留策略：
1. `checkpoint_best.pt`：仅保存 1 份验证集 Loss 最低模型。
2. `checkpoint_last.pt`：始终同步最新 step。
3. `checkpoint_step_{N}.pt`：滚动保存，上限固定为 **2 份**。
4. **总 Checkpoint 磁盘占用**：$\le 4 \times 2.3\,\text{GB} \approx 9.2\,\text{GB}$。
5. **条件缓存共享**：直接复用 `/home/shangguanrz/project/pic-edit/experiments/iper_adapter_finetune_v1/condition_cache` 中的 578 个预计算 VLM 特征文件（共 1.9 GB），避免重复占用磁盘且节省每步调用 14B VLM 的耗时。

---

## 6. 四阶段推进实施计划

### 阶段 1：反向传播冒烟测试（已通过验证）
- **目标**：验证计算图连通性、梯度回传完整度、12 个 Residual Heads 冻结状态（6 个 Geometry Head 有效梯度，6 个 Interaction Head 冻结为 None）、显存占用及权重保存恢复。
- **命令**：
  ```bash
  python train_iper_v6_native.py --smoke_test
  ```
- **实测指标记录**：
  - 显存占用：16.50 GiB
  - 梯度状态：169.89M 参数 100% 获得有效有限梯度，冻结模块 0 梯度
  - 12 Residual Heads 校验：6 个 Geometry Heads 全部接收梯度更新，6 个 Interaction Heads 梯度严格为 None
  - Residual RMS：从 0.0000 稳定启动至 0.0647，无 NaN/Inf
  - Denoise Progress 方向：严格与推理对齐（纯噪声 t=1000 时 progress=0.0，纯净图像 t=0 时 progress=1.0）

### 阶段 2：小样本过拟合验证（Overfit Gate）
- **目标**：在 32 个固定 pair 上训练 500–1,000 optimizer steps（每 step 积累 4 个 micro-batches），确证模型具备对单人条件的强拟合能力。
- **启动命令**：
  ```bash
  python train_iper_v6_native.py \
    --overfit_samples 32 \
    --max_train_steps 800 \
    --gradient_accumulation_steps 4 \
    --log_steps 10 \
    --save_steps 200 \
    --val_every 100 \
    --output_dir experiments/iper_v6_overfit_32
  ```
- **通过准则（Gate Criteria）**：
  1. 训练 Loss 从初始 ~0.18 下降到 0.05 以下。
  2. Residual RMS 稳定维持在 0.10–0.50 区间。
  3. Adapter ON 预测损失显著低于 Adapter OFF（差异 > 30%）。
  4. 验证对比（ON vs OFF vs Perturbed）在过拟合子集上呈现明确分化。

### 阶段 3：小规模 Pilot 训练
- **目标**：在 10 个身份（约 3,800 pairs）上训练 2,000–3,000 optimizer steps，微调学习率并测量验证集泛化收敛曲线。
- **启动命令**：
  ```bash
  python train_iper_v6_native.py \
    --max_train_steps 3000 \
    --gradient_accumulation_steps 4 \
    --val_every 300 \
    --save_steps 500 \
    --output_dir experiments/iper_v6_pilot_v1
  ```

### 阶段 4：全量 iPER 正式训练
- **目标**：在 69 个 dev-train 身份（26,496 pairs）上训练约 3 个 epochs（约 20,000–25,000 optimizer steps，等价于 80,000–100,000 micro-batches）。
- **建议正式配置**：
  ```bash
  python train_iper_v6_native.py \
    --train_pairs datasets/iPER/iper_sampled_6src64tgt/splits/dev_train_pairs.jsonl \
    --val_pairs datasets/iPER/iper_sampled_6src64tgt/splits/dev_val_pairs.jsonl \
    --resolution 512 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lr_zero_heads 1e-4 \
    --lr_condition 5e-5 \
    --lr_bridge 5e-5 \
    --lr_control_block 1e-5 \
    --warmup_heads_steps 500 \
    --max_train_steps 20000 \
    --val_every 500 \
    --save_steps 1000 \
    --max_rolling_checkpoints 2 \
    --output_dir experiments/iper_v6_native_full_run
  ```

---

## 7. 验证与诊断协议

### 7.1 训练中验证协议
在每次 `validate()` 期间，脚本自动固定同一批验证帧与随机种子，对比以下三组指标：
1. **Adapter ON（正确条件）**：评估当前几何引导质量。
2. **Adapter OFF（无控制基线）**：评估模型是否过度依赖先验或退化。
3. **Adapter ON + 扰动条件（Perturbed）**：对法线、部位、姿态与 Global 参数施加完整随机扰动，评估模型对姿态信号的真实敏感度。
4. **可视化对比预览（可选）**：通过 `--eval_generate_samples 2` 参数，可在验证时自动调用 `ControlledDeepGenPipeline` 生成 `[Source | Ground Truth | Adapter OFF | Adapter ON | Adapter Perturbed]` 的 5 栏横向拼图，直接观察人物身份、衣物纹理与姿态控制。

### 7.2 独立评测脚本（eval_iper_v6.py）
训练完成后，使用专门适配 V6 架构的评测脚本对验证集或测试集进行客观量化评测与拼图生成：
```bash
python eval_iper_v6.py \
  --adapter_checkpoint experiments/iper_v6_native_full_run/checkpoint_best.pt \
  --pairs_jsonl datasets/iPER/iper_sampled_6src64tgt/splits/dev_val_pairs.jsonl \
  --max_samples 50 \
  --inference_steps 28 \
  --output_dir experiments/iper_v6_val_eval
```
该脚本输出：
- `eval_metrics.json`：包含 Adapter ON 与 Adapter OFF 的 PSNR、SSIM、MAE 对比。
- `images/`：每一对测试样本的 4 栏对比高清切片（`Source`, `Ground Truth`, `Adapter OFF`, `Adapter ON`）。

### 7.3 全量自动化测试回归
测试套件可通过以下命令一键全量回归验证：
```bash
python -m pytest tests/test_iper_v6_training_pipeline.py \
                 tests/test_iper_v6_data_integrity.py \
                 tests/test_condition_injector_v1.py \
                 tests/test_adapter_reasoning_v1.py \
                 tests/test_deepgen_control_interface_v1.py \
                 tests/test_adapter_v6_architecture.py
```
