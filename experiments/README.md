> ⚠️ **本地是部分镜像**：本目录在本地是平铺结构；权威结构是 sg 远端 `experiments/` 的 6 个分组（2026-10-17 设立 `6.qwen-controlnet union`），以远端 `experiments/README.md` 为准。

# experiments/ —— 项目全量实验运行数据、定量评测与技术演进全景

> **更新时间**：2026年10月  
> **工作区目录**：`/home/shangguanrz/project/pic-edit/experiments`  
> **核心职责**：记录从早期显存压测、DeepGen MMDiT V6 系列、Qwen-Image-2.1 单流 DiT 身体/面部/双人交互适配器，到开源 BFS 方案对比的**全部 34 个实验运行（Run）**。  
> **数据保管规范**：本目录包含模型权重检查点（`.pt` / `.safetensors`）、离线条件特征缓存（`.pt`）、逐步诊断日志（`.log`）与生成图像（`.png`）。大张量文件默认不纳入 Git，但在服务器持久化盘与归档盘完好保留，确保每一项科学结论均有不可篡改的量化证据支持。
>
> **中文快速索引**： [`实验中文索引.md`](实验中文索引.md) —— 36 个目录名的中文对照与一句话说明（目录名暂保持英文，原因见该文件末尾）。

---

## 一、 实验全景分类索引 (Experiment Classification Matrix)

根据模型基座、任务目标与技术演进脉络，本目录下的 34 个实验划分为 **7 大核心实验族**：

| 实验族 | 包含实验目录 | 核心基座与架构 | 核心关注点与科学问题 | 关键里程碑结论 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 显存与架构极限压测族** | `review_20260908_v2`<br>`review_20260908` | DeepGen-1.0 (SD3 MMDiT 2.47B) | 512/768/1024 分辨率显存墙测试、Direct vs Letterbox 边界、梯度与零等价性 | 111 个运行矩阵 100% PASS；确立 Letterbox 为显存安全标准；建立 80G GPU 并发容量模型 |
| **2. DeepGen V6 单人长程训练族** | `iper_v6_native_20k_v1`<br>`iper_v6_comparative_3000`<br>`iper_adapter_finetune_v1`<br>`iper_adapter_v1`<br>`iper_v6_native_finetune_v1` | DeepGen-1.0 + V6/V6.3 Unified SMPL-X Adapter | iPER 数据集上的长程训练动态、步数演变规律（0~20,000 steps） | 4k 步出现关键收敛拐点（PSNR跃升 +9.8dB）；16k 步达保真度峰值（PSNR 28.20dB）；保留 580+ 核心条件缓存 |
| **3. DeepGen 定量评测与消融族** | `iper_v6_20k_eval_final`<br>`iper_v6_20k_eval_step4000`<br>`iper_v6_testdata_eval`<br>`deepgen_no_control_ablation_v1`<br>`champ_deepgen_formal_v1`<br>`iper_test` / `iper_test_1024` | DeepGen-1.0 + V6 系列适配器 | 50 样本 Held-out 定量评估、受控 vs 无控制消融对照、外推泛化度 | 受控模型相比无控制提升 +7.07 dB PSNR，误差降低 57.2%；发现并记录 4k 步评估时的瞬时显存悬崖 (OOM) |
| **4. 面部适配器与过拟合门禁族** | `tiktok_overfit_sanity_full`<br>`tiktok_overfit_sanity_smoke`<br>`iper_v65_face_overfit_024_6`<br>`iper_v65_face_overfit_smoke`<br>`iper_v64_detail_overfit_024_6`<br>`tiktok_v65_face_overfit_00001`<br>`tiktok_00001_condition_repair`<br>`tiktok_objective_eval`<br>`tiktok_smoke_00001`<br>`tiktok_v65_pure_inference`<br>`tiktok_v65_test_2steps`<br>`test_smoke_detail`<br>`v6_condition_inspection` | DeepGen-1.0 + V6.4 细节 / V6.5 面部适配器 | 确定性记忆能力（Phase A）与全扩散轨迹生成（Phase B）门禁验证（C1~C6）、SMPL-X 条件修复 | 严格证明 V6.5 单层特征注入在全轨迹去噪中晚期破坏背景与服装全局流（门禁FAIL），推动技术路线转向解耦方案 |
| **5. Qwen 2.1 单流 DiT 身体控制族** | `qwen21_iper_body`<br>`qwen21_tiktok_v3_smoke`<br>`qwen21_union_migration_v1` | Qwen-Image-2.1 (7B Single-Stream DiT) + Qwen21BodyAdapter；官方 Union ControlNet 直迁 | 7B 单流架构下长文本/图像 Token 融合机制、50 样本四臂（Four-Arm）全面定量对比；官方预训练控制分支不训练直用的可行性 | **全面超越 DeepGen**：PSNR 25.86dB (+1.33dB)，**PCK@0.02 达 98.00%**；同时揭示姿态形变导致面部 ID 下降的核心矛盾。官方 Union 直迁（`qwen21_union_migration_v1`）判定为**需要进一步输入适配**：接入版控制强度 0 与现工作流逐字节相同，单人严格 PCK@0.2 0.750→0.964，但强度 1.0 身份分 0.916→0.585 且骨架线漏进画面、双人检出人数并成 1 人 |
| **6. Qwen 2.1 双人交互与物理接触族** | `qwen21_hi4d_dual`<br>`hi4d_overfit_archive`<br>`qwen-2.1-test`<br>`qwen_edit_results` | Qwen-Image-2.1 + DualAdapter (SMPL 接触特征与 Replay Rehearsal) | 双人复杂肢体纠缠、物理拥抱、防身份互换（Identity Swap）、接触距离误差 | 双人检测率 100%，无身份互换；拥抱过拟合 PSNR 达 29.85dB (+17.5dB)；两阶段提示词下 Body ID 保持率超 86% |
| **7. 综合评测与开源基准对比族** | `16-test` | DeepGen vs Qwen21 vs BFS LoRA vs PEI2I | 跨架构、跨条件、复杂长提示词（托篮球/雨夜西装）全场景 16 类难度对照 | 确立当前多任务极限：纯提示词无法精确控姿，两阶段解耦（Adapter 控姿 + BFS/Face Adapter 精修面部）是最佳落地路线 |
| **8. Qwen ControlNet Union 控制族** | `6.qwen-controlnet union`<br>（含 `qwen21_union_migration_v1`<br>`qwen21_union_dataset_adaptation_v1`） | Qwen-Image-2.1 + ControlNet Union 官方权重 (7.1GB) | 多合一控制网络（OpenPose/Depth/Inpaint）直接迁移验证、多数据集 LoRA 适配微调与动作编辑流水线接入 | 官方权重 180 键 100% 审计通过；文生图姿态与深度完全受控；建立纯代码管理与模型软链接机制；全套 117 项契约测试全部 PASS |

---

## 二、 逐项实验详细解析 (In-Depth Experiment Breakdown)

### 1. 显存与架构极限压测族 (VRAM Lab Benchmarks)

#### 1.1 `review_20260908_v2/`（正式复核运行库）
* **实验目的**：对 `DeepGen-1.0` (SD3 MMDiT 2.47B) 开展严苛的显存基准测绘，验证不同分辨率、预处理策略与训练迭代下的显存安全性与数值不变量。
* **实验条件**：
  - 硬件平台：NVIDIA RTX 6000 Ada / A100 (80GB VRAM)，CUDA 12.8，PyTorch 2.8.0。
  - 测试矩阵：覆盖 111 个主运行用例，包括 512×512、768×768、1024×1024 分辨率；Direct 缩放 vs Letterbox 填充；条件缓存命中 vs 实时计算；50ms NVML 显存高频采样。
* **核心结果与结论**：
  - **验收结论**：`ACCEPTANCE.json` 记录 111 个 primary runs 全部判定为 `PASS`，零异常退出。
  - **显存规律**：Letterbox 保持纵横比可显著平抑显存尖峰；Direct 模式在 1024 分辨率下发生显存陡增，确立了 Letterbox 在全项目预处理中的标准地位。
  - **不变量契约**：零初始化门槛校验（Zero-head equivalence）与梯度阈值校验 100% 达标。
* **产物与位置**：
  - 主权重与运行记录已归档至 `/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/review_20260908_v2`，当前路径以软链接无缝保留。
  - 轻量化复核总结与 HTML Demo 副本位于 `results/review_20260908_v2/`。

#### 1.2 `review_20260908/`（第 1 轮早期实验）
* **实验目的**：显存基准测试初版，包含 6 个原生质量运行用例，已被 `review_20260908_v2` 完整演进并取代。

---

### 2. DeepGen V6 系列单人姿态适配器长程训练族 (iPER Long-Horizon)

#### 2.1 `iper_v6_native_20k_v1/`（20,000 步全量训练）
* **实验目的**：验证 `Unified SMPL-X Adapter V6.3` 在大规模步数（20,000 steps）下的长程动态、收敛曲线与图像保真度上限。
* **实验条件**：
  - 骨干模型：DeepGen-1.0（SD3 MMDiT）。
  - 数据集：iPER 配对训练集，原生分辨率。
  - 学习率调度：Warmup + Cosine 衰减，固定每 4000 步采样评测。
* **核心定量结果（基于 `three_row_showcase/metrics_summary.json`）**：
  - **Sample 1 (`001_3`)**：
    - Step 0：PSNR 16.75 dB, SSIM 0.8521, MAE 14.99
    - Step 4000：PSNR 26.59 dB, SSIM 0.9262, MAE 4.86（**关键质量跃迁点**）
    - Step 8000：PSNR 27.68 dB, SSIM 0.9327, MAE 4.87
    - Step 16000：**PSNR 28.20 dB**, **SSIM 0.9361**, MAE 4.94（**全过程质量峰值**）
    - Step 20000：PSNR 27.82 dB, SSIM 0.9345, MAE 5.00（进入轻度饱和平台期）
  - **Sample 2 (`001_4`)**：
    - Step 0 (18.69 dB) $\to$ Step 4000 (26.46 dB) $\to$ Step 12000 (27.01 dB) $\to$ Step 20000 (26.95 dB)
* **结论**：4,000 步即可完成骨架姿态的粗对齐，12,000~16,000 步可使服装纹理与身体边缘达到最自然状态，过长的训练步数不会显著提升画质。
* **产物与位置**：
  - 检查点：`checkpoint_best.pt`, `checkpoint_last.pt`, `adapter_step_12000.pt`
  - 评估图像与三行演变对比拼图：`three_row_showcase/` (148 张图像)。

#### 2.2 `iper_v6_comparative_3000/`（3,000 步多阶段对比展示）
* **实验条件**：每 500 步保存并生成对比网格（Step 500, 1000, 1500, 2000, 2500, 3000）。
* **定量指标**：Step 500 时 PSNR 20.58 dB / SSIM 0.9612；Step 3000 时平均 PSNR 22.30 dB / SSIM 0.9582。
* **位置**：归档于 `/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/iper_v6_comparative_3000`（软链映射）。

#### 2.3 `iper_adapter_finetune_v1/`（V6 共享条件特征库与 3,000 步微调）
* **重要职责**：本目录下的 `condition_cache/` 存有 **584 个 iPER 样本的 SMPL-X 几何特征张量**，是当前 V6 系列训练与推理的共享底层依赖！
* **定量指标**：Step 3000 验证集损失 0.1981，50 样本评估平均 PSNR 22.61 dB，SSIM 0.9608，单图耗时 3.98s。
* **产物**：`adapter_step_3000.pt`, `eval_results/eval_metrics.json`。

#### 2.4 `iper_adapter_v1/` & `iper_v6_native_finetune_v1/`
* 早期初始版本运行，分别产出 100 步冒烟权重 `adapter_step_100.pt` 与原生微调对照基线。

---

### 3. DeepGen 定量评测与消融对照族 (Evaluation & Ablation)

#### 3.1 `iper_v6_20k_eval_final/`（V6.3 最终权威 50 样本打分）
* **实验目的**：在 50 个未参与训练的 iPER 验证对上，严格对比开启 Adapter 与关闭 Adapter（基座直出）的定量质量。
* **核心对比数据（记录于 `eval_metrics.json` 与 `iper_v6_20k_eval_final.log`）**：
  - **Adapter ON（开启姿态适配器）**：
    - **PSNR: 24.00 dB**
    - **SSIM: 0.9811** (Skimage SSIM: 0.9133)
    - **MAE: 6.15** (MSE: 270.10)
  - **Adapter OFF（基座直出/无姿态注入）**：
    - **PSNR: 16.93 dB**
    - **SSIM: 0.9005**
    - **MAE: 14.36**
  - **结论**：受控适配器带来了 **+7.07 dB 的 PSNR 绝对提升**，像素级平均绝对误差（MAE）降低了 **57.2%**，为论文与技术评审提供了决定性证据。
* **产物**：50 张高质量测试生成图保存在 `images/`。

#### 3.2 `iper_v6_20k_eval_step4000/`（显存悬崖复现分析）
* **实验现象**：加载 Step 4000 检查点进行评测时，在 80GB 显存设备上尝试分配 432 MiB 显存时偶发触发 `torch.OutOfMemoryError`。
* **结论**：揭示了 Diffusers Pipeline 在连续大批量生成且未及时清除 PyTorch 缓存上下文时的显存碎片化风险，推动了后续评估脚本中增加强制 `torch.cuda.empty_cache()` 的调度策略。

#### 3.3 `deepgen_no_control_ablation_v1/`（无控制消融）
* **实验目的**：验证姿态控制对齐是由 SMPL-X 适配器注入产生，而非基座先验或 Prompt 运气。
* **定量指标**：无控制时 PSNR 仅 14.99 dB，MAE 28.06；对比实验证明控制信号具有强因果性。

#### 3.4 `champ_deepgen_formal_v1/`
* **实验条件**：在 Champ 独立样本上执行手势重定向（胸前双手比心手势），运行 120 步。
* **指标**：Loss 由 0.1659 持续下降至 0.0798，梯度范数稳定在 0.28~0.43。产出 `adapter_step_120.pt`。

#### 3.5 `iper_test/` & `iper_test_1024/`
* 分别测试 512×512 与 1024×1024 分辨率下 DeepGen 适配器直出效果，留存生成质量样本图。

---

### 4. 面部适配器分支与过拟合门禁诊断族 (Face Adapter & Sanity Gating)

#### 4.1 `tiktok_overfit_sanity_full/`（V6.5 面部适配器两阶段门禁测试）
* **实验目的**：设计两阶段严苛过拟合协议，对 V6.5 Face Adapter 在 TikTok 样本上的记忆能力、梯度传播与生成正向贡献进行科学裁决。
* **协议设计**：
  - **Phase A（前 200 步）**：固定时间步 $t=150$，严格测试确定性记忆拟合能力。
  - **Phase B（201~1500 步）**：全扩散轨迹随机时间步采样，测试生成泛化。
* **六大门禁准则（C1~C6 审计结果，记录于 `summary.json`）**：
  1. `C1 接口不变量`：初始 ON $\equiv$ OFF（零初始化头，$\max|\Delta\text{pred}| \le 10^{-5}$） $\to$ **PASS** (误差 0.0)
  2. `C2 Phase A 记忆收敛`：尾部 flow 损失 $\le 0.7 \times$ 初始损失 $\to$ **PASS** (0.4196 $\le$ 0.455)
  3. `C3 Phase B 全轨迹下降`：尾部 flow 损失持续显著下降 $\to$ **FAIL** (0.2383，未能达标)
  4. `C4 显著优于初始`：最终 ON PSNR $\ge$ step0 ON + 5.0 dB $\to$ **PASS** (+7.95 dB)
  5. `C5 绝对拟合拉伸指标`：最终 ON PSNR $\ge 20.0$ dB $\to$ **PASS** (20.04 dB)
  6. `C6 适配器正向贡献`：最终 ON $\ge$ OFF $\to$ **FAIL** (ON 比 OFF 略低 -0.044 dB)
* **核心科学发现与结论**：
  - **裁决结论：`FAIL`**。
  - **技术根因**：V6.5 尝试在 MMDiT 中间层直接注入面部局部特征，但在全轨迹扩散去噪的“中晚期时间步”中，强行覆盖的特征扰乱了全局背景与服装特征流，导致整体生成质量未能超越原生基座。这一关键发现**直接终结了在 MMDiT 单模型内强行单层揉合面部的方案**，确立了转向 Qwen-Image-2.1 或采用后置两阶段面部精修（BFS/Face Retargeting）的技术决策。
* **产物**：`summary.json`, `checkpoints/`, `gen_evals/`。

#### 4.2 `iper_v65_face_overfit_024_6/`（iPER 样本面部适配器 1,500 步训练）
* **损失项细粒度追踪**：
  - Step 1：Total Loss 0.4569 (Flow 0.4428, Face ID 0.1193, Face Perceptual 0.0442)
  - Step 1500：Total Loss 0.3495 (Flow 0.3478, Face ID 0.0000, Face Perceptual 0.0000, Outside Mask 0.0840)
* **产物**：留存 step 250 至 1500 共 6 个阶段检查点，保存在 `checkpoints/`。

#### 4.3 `tiktok_00001_condition_repair/`
* **实验内容**：对比修复不同 SMPL-X 条件通道对 TikTok 姿态漂移的影响：
  - `pose_only.pt` vs `pose_part.pt` vs `inference_trained_1500` vs `face025`。
  - 产出 44 张各阶段修复对比图像。

#### 4.4 `tiktok_objective_eval/`
* **实验内容**：对 TikTok 00001 执行单步单样本的时间步长扫描（$t=0 \sim 1000$）。
* **量化基准**：源图 vs 目标图 ROI PSNR 11.42 dB，全图 PSNR 12.48 dB，序列平均余弦相似度 0.7826。

#### 4.5 其余辅助冒烟与对齐实验
- `iper_v65_face_overfit_smoke/` / `iper_v64_detail_overfit_024_6/` / `tiktok_overfit_sanity_smoke/` / `tiktok_v65_face_overfit_00001/` / `tiktok_v65_pure_inference/` / `tiktok_v65_test_2steps/` / `tiktok_smoke_00001/` / `test_smoke_detail/` / `v6_condition_inspection/`：
  分别留存了 2 步冒烟、条件渲染投影检查、推理前向校验与细节流水线快照。

---

### 5. Qwen-Image-2.1 单流 DiT 身体姿态控制族 (Qwen21 Body Adapter)

#### 5.1 `qwen21_iper_body/`（项目里程碑：单流 7B 架构突破与四大实验臂评测）
* **实验目的**：将姿态控制迁移到更强大的单流 DiT 基座 `Qwen-Image-2.1`（7B 参数量），验证其长文本多模态理解与姿态生成能力。
* **训练历程**：
  - `pilot_3k/` & `pilot_3k_acc4_resumed/`：3,000 步探索性先导试验。
  - `full_20k/`：20,000 步全量微调训练，产出核心权重 `checkpoint_best.pt` 与 `checkpoint_last.pt`。
* **50 样本 Held-out 四大实验臂（Four-Arm）最终决战数据（记录于 `four_arm_eval_50_s40/summary.json`）**：

| 评估指标 (50 样本均值) | 臂 1: `deepgen_off` (基座直出) | 臂 2: `deepgen_on` (受控适配) | 臂 3: `qwen21_off` (Qwen基座) | 臂 4: `qwen21_on` (Qwen受控适配) |
| :--- | :---: | :---: | :---: | :---: |
| **全图 PSNR (dB)** | 17.24 | 24.53 (+7.29) | 17.53 | **25.86 (+8.33, 史上最高)** |
| **全图 SSIM** | 0.8017 | 0.8756 | 0.8363 | **0.8969 (最保真)** |
| **像素绝对误差 (MAE)** | 16.26 | 6.58 | 13.61 | **4.93 (误差最低)** |
| **姿态对齐率 PCK@0.02** | 17.44% | 95.50% | 20.31% | **98.00% (几乎完美对齐)** |
| **姿态对齐率 PCK@0.05** | 48.48% | 98.12% | 47.05% | **99.18%** |
| **人体前景 MAE (Human)** | 60.01 | 20.03 | 61.16 | **17.57 (前景重构最佳)** |
| **身份余弦相似度 (Face ID)** | 0.2612 | 0.2917 | **0.8187 (超强先验)** | 0.3510 |
| **单图生成耗时 (秒)** | **5.08s** | 5.94s | 5.84s | 5.87s |
| **峰值显存 (Peak VRAM)** | **14.51 GiB** | 14.53 GiB | 32.76 GiB | 32.76 GiB |

* **关键科学发现与技术结论**：
  1. **姿态控制力碾压性领先**：`qwen21_on` 的 PCK@0.02 达到 **98.00%**，超越了 DeepGen 的 95.50%；全图 PSNR 达到 **25.86 dB**（比 DeepGen 高出 1.33 dB），全图 MAE 压缩到 4.93。
  2. **揭示核心技术痛点（当前主攻方向）**：Qwen 基座在未开启 Adapter 时（`qwen21_off`），由于其强大的文本上下文与图像注意力，Face ID 余弦相似度高达 **0.8187**；然而一旦开启 Body Adapter 强行将躯干与四肢形变到目标姿态，Face ID 陡降至 **0.3510**。虽然该值仍显著高于 DeepGen 的 0.2917，但这直接证实了：**全身姿态适配器的几何变换在潜空间中对抗了面部特征的无损保留**，这构成了本项目当前最重要的技术攻坚目标。
* **产物与位置**：
  - 核心权重：`full_20k/checkpoint_best.pt`
  - 评测明细：`four_arm_eval_50_s40/summary.json`，包含了 50 组各臂生成图像共 779 张。

#### 5.2 `qwen21_tiktok_v3_smoke/`
* **实验内容**：Qwen-Image-2.1 在 TikTok 数据集上的 Face/Body 双适配器训练与特征预缓存测试。包含 24 个检查点与离线潜变量特征缓存（`qwen21_cache/`）。

---

### 6. Qwen-Image-2.1 双人交互与物理接触姿态控制族 (Hi4D Dual Adapter)

#### 6.1 `qwen21_hi4d_dual/`（双人复杂交互与物理接触）
* **实验目的**：突破单人编辑边界，在 Hi4D 数据集上实现双人复杂姿态（如拥抱、推拉、太极）的精确控制，解决双人空间重叠、物理接触与身份互换（Identity Swap）难题。
* **架构配置（记录于 `run_manifest.json`）**：
  - 基座模型：Qwen-Image-2.1。
  - 初始化：继承自单人最佳权重 `qwen21_iper_body/full_20k/checkpoint_best.pt`。
  - 防遗忘策略：在 1,312 对 Hi4D 双人交互训练中，混入 iPER 单人样本进行**回放排练（Rehearsal Replay）**。
  - 控制网络：`src/pose_control/qwen21/dual_adapter.py`，注入两人 SMPL-X 网格接触图与双人几何信号。
* **核心评估数据（基于 `single_hug_eval/summary.json` 与 `single_pair14_action_v2_overfit/eval_seed42/summary.json`）**：
  - **身份防混淆**：检测人数准确率 100%（平均 2.0 人），**Identity Swap = 0.0**（完全杜绝两人面部或服装互换）。
  - **物理接触精度**：接触距离误差（Contact Distance Error）控制在 0.113~0.117 以内。
  - **双人身份保持率**：在 `Full Detailed` 两阶段提示词驱动下，Person A 身份保持率达 **0.8689**，Person B 身份保持率达 **0.7409**，背景 MAE 仅 3.16。
* **产物**：`single_hug_overfit/checkpoint_best.pt`, `single_pair14_action_v2_overfit/checkpoint_best.pt`, 多个测试阶段评测总结 `summary.json`。

#### 6.2 `hi4d_overfit_archive/`
* 早期 Hi4D 拥抱姿态训练归档，保留 `hug_120steps` 与 `hug_1000steps` 两个阶段的运行记录与双人条件缓存 `hi4d_dual_source.pt`。

#### 6.3 `qwen-2.1-test/`
* **实验内容**：双人太极、拥抱及单人泛化评测。
  - `overfit_inter01`：双人交互样本 500 步过拟合，PSNR 从初始 12.35 dB 攀升至 **29.85 dB**，SSIM 从 0.4009 提升至 **0.9245**（大幅提升 +17.50 dB）。
  - 保存有 `gen_v1`, `interaction_v1`, `smoke` 共 309 张评测生成图。

#### 6.4 `qwen_edit_results/`
* Qwen-Image-Edit 原生指令编辑输出对照样本。

---

### 7. 综合评测与开源基准对比套件族 (Benchmark & Challenge Suite)

#### 7.1 `16-test/`（16 类高难度复杂多条件编辑综合基准）
* **实验目的**：设置极高挑战性场景，测试模型在面对长复合提示词、极端姿态形变、道具生成（手托篮球）以及防止参考图背景泄漏时的综合表现。
  > **测试 Case 典型提示词**：“让图一中的男士摆出图二中人物的姿势，手中托起篮球。保持外貌、西装、雨夜城市街道背景完全不变；不要引入图二的白色影棚背景或背带裤。”
* **全流程 6 大 Stage 跨架构对比链**：
  - `stage1_baselines`：Qwen 基座直接生成（纯靠 Prompt 理解）。
  - `stage2_adapters`：挂载 `deepgen_adapter` vs `qwen_adapter`（对比仅输入源图条件 vs 双向条件注入）。
  - `stage3_detailed_prompt`：输入结构化拆解长提示词（明确锁定身份、服装、背景负向排除）。
  - `stage4_bfs_body`：挂载开源 `BFS-Best-Face-Swap` 的 Body LoRA (`bfs_body_swap_v1.0_qwen_2.1.safetensors`)。
  - `stage5_bfs_bracket_norm`：使用 BFS 官方归一化括号提示词。
  - `stage6_pei2i`：使用 PEI2I 指令重写流。
* **核心科学发现与结论**：
  1. **纯 Prompt 的局限性**：基座模型（Stage 1）完全无法按图二精确摆出肢体动作，容易遗忘篮球或直接复制图二的背景。
  2. **Adapter 的动作控制力**：Adapter（Stage 2）能强制肢体严密贴合图二姿势，但在要求“托起篮球”这一局部物理交互时，若无细粒度手部/物体条件，手掌与球体容易融合畸变。
  3. **BFS LoRA 的表现**：BFS 在身体重定向时表现出较好的背景保持，但对于复杂几何约束仍依赖于 Prompt 引导。
  4. **全项目技术启示**：证明了目前行业内单靠单阶段端到端网络无法完美兼顾“大幅度骨骼重定向 + 细粒度道具交互 + 面部 100% 细节冻结”，**必须采用两阶段解耦管线（Stage 1: Qwen21BodyAdapter 确立主体姿态与构图 $\to$ Stage 2: BFS/Face Retargeting 局部高分辨率精修贴回）**。
* **产物与位置**：
  - 缓存与条件：`stage2_cache/16test_pA_pA.pt`, `conditions_ab/`
  - 96 张各阶段高清对比图分布在各 `stage*/` 目录下，附带各阶段评测结果 `results.json`。

---

## 三、 快速复现与关键资产调用路径

如果你需要在后续实验或代码中调用上述产出的核心模型权重或条件缓存，请参考以下权威路径：

| 资产类型 | 推荐检查点 / 数据路径 | 对应实验与说明 |
| :--- | :--- | :--- |
| **Qwen 2.1 最佳单人身体适配器** | `experiments/qwen21_iper_body/full_20k/checkpoint_best.pt` | PCK@0.02=98.00%, PSNR=25.86dB 的最强身体控制权重 |
| **Qwen 2.1 双人交互适配器** | `experiments/qwen21_hi4d_dual/single_hug_overfit/checkpoint_best.pt` | Hi4D 双人拥抱与接触控制权重，无身份互换 |
| **DeepGen V6.3 最佳单人适配器** | `experiments/iper_v6_native_20k_v1/checkpoint_best.pt` | DeepGen 20k 训练最佳权重 (PSNR=28.20dB) |
| **DeepGen V6 共享条件特征库** | `experiments/iper_adapter_finetune_v1/condition_cache/` | 包含 584 个 iPER 样本离线 SMPL-X 特征缓存（不可删除） |
| **显存复核实验全量数据（归档）** | `/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/` | 包含 111 个 run 的 NVML 采样、大权重与环境镜像 |

---

## 四、 规范与注意事项

1. **不可篡改性**：所有已完成并记录了 `summary.json` / `metrics.json` 的实验目录，严禁覆盖、二次运行或改名，以确保学术与研发证据链的完整可信。
2. **新实验创建规范**：启动新实验请在 `scripts/train/` 中指定新的输出子目录（例如 `experiments/<backbone>_<dataset>_<variant>_v<N>/`），并生成对应的 `run_manifest.json` 记录代码版本哈希与超参数。
3. **大张量清理与归档**：若根分区空间不足，应严格按照冷数据归档规范，将历史检查点同步至 `/data/shangguanrz/pic-edit-archive/` 并在原路径建立软链接，切勿直接 `rm -rf`。
