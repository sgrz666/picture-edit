# tests/ —— 自动化测试套件

本目录是人物图像姿态编辑系统的回归与契约测试层，由仓库根的 `pytest.ini` 驱动（`testpaths = tests`，并排除 `add/datasets/models/experiments/results/data/vram_lab/.git` 等非源码树）。测试守护三条技术线的**架构契约、零初始化行为、条件注入契约、数据完整性与训练/评测可复现性**——多数测试是 CPU-only 的合同断言（不跑完整推理、不训练），少数会做有界 GPU 冒烟。实际测试文件共 **64 个** `test_*.py`，按主题分为九组。

## 分组一览

### 1. V6 架构契约与适配器（DeepGen 线）
守护 Unified SMPL-X Adapter V6 的架构边界、条件注入与零初始化不变量。

- `test_adapter_v6_architecture.py` —— V6 adapter 架构契约（smoke report、build_tiny、runtime 等）
- `test_adapter_reasoning_v1.py` —— V6.2 独立 reasoner 的公共契约
- `test_condition_injector_v1.py` —— 条件注入器模块可用性、person/relations 行为
- `test_condition_bridge_contract.py` —— reasoner bridge 的几何/交互投影无偏、消耗双高分辨率控制路径
- `test_deepgen_control_interface_v1.py` —— V6.3 动态控制接口：分支残差 token 形状、strength 调度曲线
- `test_deepgen_pose_patch.py` —— DeepGen pose patch 契约
- `test_deepgen_no_control_ablation.py` —— 无控制消融 runner 契约

### 2. Qwen21 控制契约（迁移线）
守护 Qwen-Image-2.1 上 Body/Face/Full controller 的 token 边界、双人与推理隔离。

- `test_qwen21_native_geometry.py` —— Full方案C第一步：原生几何特征与旧projection前输出一致、独立权重继承、梯度、网格拒绝与CPU实权重审计工具。
- `test_qwen21_native_body.py` —— Full方案C第二步：主干实时hidden、零输出/FP32/target范围、原生32层结构和跨条件checkpoint重算。

- `test_qwen21_control_contract.py` —— 控制钩子只能改 target token、保留 prefix
- `test_qwen21_dual_control.py` —— 双控必须绑定两个 source 人且不改 native prefix
- `test_qwen21_dual_inference_boundary.py` —— target 像素/评测 mask 不得进入生成输入
- `test_qwen21_face_control.py` —— Face 分支只改 target face ROI、保留 Body 行为
- `test_qwen21_frozen_bfs_lora.py` —— 冻结 BFS LoRA 加载（真实 safetensors/torch 层）
- `test_qwen21_full_bridge.py` —— 完整 controller bridge 契约
- `test_qwen21_full_face_diagnostics.py` —— Face-only 消融与 flow 阶段对齐
- `test_qwen21_full_face_letterbox.py` —— Face 跟随 RGB、不拉伸/不重变换
- `test_qwen21_full_preservation.py` —— 独立审计复制的控制架构（与 Qwen bridge 解耦）
- `test_qwen21_full_training_runtime.py` —— 多 pair 诊断与严格训练续跑
- `test_qwen21_action_prompts.py` —— 双人动作文本与 source 缓存契约

### 3. 面部 / 细节 / 手部分支
守护 V6.5 人脸、V6.6 手部与 V6 detail 分支的训练接线、梯度与参考网格边界。

- `test_v65_face_foundation.py` / `test_v65_face_integration.py` / `test_v65_face_training_gradients.py` —— V6.5 人脸基础、集成、训练梯度
- `test_v65_iper_face_pipeline.py` —— V6.5 iPER 人脸管线
- `test_v65_upstream_migrations.py` —— V6.5 上游迁移
- `test_v66_hand_core.py` —— V6.6 手部核心
- `test_v6_detail_branch.py` / `test_v6_detail_contracts.py` / `test_v6_detail_preparer.py` —— V6 detail 分支、契约、准备器
- `test_qwen21_face_loss_alignment.py` —— Face 损失对齐（与 OpenCV 几何比对）
- `test_qwen21_face_multisample_data.py` —— Face 多样本实验 split/采样 fail-closed
- `test_qwen21_face_objective.py` —— Face 目标函数不可在画布里藏损
- `test_qwen21_face_reference_grid.py` —— 可逆训练网格 Face scatter 边界

### 4. 数据完整性
守护 iPER / TikTok 的 manifest、几何、letterbox 与数据链契约。

- `test_tiktok_data_contract.py` —— TikTok 数据契约（大文件，24867 B）
- `test_iper_v6_data_integrity.py` —— iPER V6 数据集加载与 injector 通过
- `test_iper_v6_eval_parity.py` —— iPER V6.3 评测口径与 130.28M 基线可比
- `test_tiktok_letterbox.py` —— TikTok letterbox 契约
- `test_tiktok_qwen21_face_cache.py` / `test_tiktok_qwen21_full_chain.py` —— TikTok→Qwen21 人脸缓存、全链
- `test_tiktok_qwen21_learning_rates.py` / `test_tiktok_qwen21_readiness.py` —— 学习率、就绪度

### 5. 训练 / 评测回归
守护训练流水线、joint pilot 续跑、权重继承与可审计结果的回归。

- `test_iper_v6_training_pipeline.py` —— iPER V6.3 Scheme A 训练流水线单测
- `test_qwen21_joint_training.py` —— joint Body+Face 训练契约
- `test_qwen21_joint_pilot_resume.py` / `test_qwen21_joint_pilot_summary.py` / `test_qwen21_joint_pilot_visuals.py` —— joint pilot 续跑闸门、JSON 契约、已保存图像视觉契约
- `test_qwen21_native_resolution.py` —— native 分辨率契约
- `test_qwen21_native_value_audit.py` —— native value 审计
- `test_qwen21_resume_reconciliation.py` —— 续跑对账
- `test_qwen21_weight_audit.py` / `test_qwen21_weight_inheritance.py` —— 权重审计 / 继承
- `test_train_tiktok_v65_face_smoke.py` —— TikTok V6.5 人脸训练冒烟（大文件，21467 B）

### 6. BFS 评测契约
守护四臂平衡对照的输入库存、参考路由与 resume 检查。

- `test_qwen21_bfs_rgb.py` —— 保护 RGB 审计不误用 skeleton 输入契约
- `test_qwen21_bfs_skeleton.py` —— 固定实验输入库存与 resume 检查
- `test_qwen21_condition_combinations.py` —— 补充因子审计的归属/路由/续跑

### 7. Hi4D 双人接触
守护 Hi4D 人对姿态对应实验的坐标/身份防混淆与接触打分。

- `test_hi4d_native_binding.py` —— 防坐标与人物分配混淆（48 个唯一 job）
- `test_hi4d_binding_scoring.py` —— 解剖顺序/有效帧比较、缺失 identity 不静默重分配
- `test_hi4d_geometry_registration.py` —— 几何语义区朝向、对齐栅格不变

### 8. 集成 / 杂项
- `test_champ_deepgen_integration.py` —— CHAMP 样本 DeepGen 集成
- `test_analysis.py` / `test_lab.py` / `test_report.py` —— 分析/实验/报告契约的一般性验收
- `test_cli.py` —— CLI 入口契约

### 9. 无 GT 评测 / 手部合理性 / 提示词银行契约（workflow 侧）
守护 workflow 评测流水线的无参考指标、合同级手部结构判定与 TikTok 提示词改写的输出契约。

- `test_no_gt_metrics.py` —— `no_gt_metrics` 合同：守住三类静默失真——关键点布局重映射与渲染器脱节、出框记账静默丢弃 miss、保护区域误把人体纳入；全程 CPU、不加载 ONNX/InsightFace/torch。
- `test_hand_plausibility.py` —— `hand_plausibility` 单元夹具：合成 96px 手掌理想手，逐一验证手指长度比/指尖融合等失败模式（每个 mutation 对应姿态条件扩散的一种真实缺陷）；运行依赖 `nihaochen/envs/qwen21/bin/python`。
- `test_tiktok_prompt_bank_contract.py` —— TikTok 每对 PE 提示词改写的契约（改写脚本在 `experiments/`，被 `pytest.ini` 排除收集，故规则在此 pin 住）：两种编译形式、短语清理、`compile_words` 不得保留图像编号、不得引用 target 照片；缺 `transformers`（PE 栈）时整体 skip。

## 与上下层的关系

- **被测对象**：`src/adapter/`、`src/pose_control/`（v6 / qwen21 / qwen21_full）、`src/controls/`、`src/evaluation/`、`src/pose_edit/`，以及 `scripts/train/`、`scripts/eval/`、`scripts/data/` 的契约。
- **运行入口**：仓库根 `pytest.ini`；CI/本地执行 `pytest`（受 `norecursedirs` 约束，不会误跑大数据/模型目录）；`test_hand_plausibility.py` 需用 `nihaochen/envs/qwen21/bin/python -m pytest tests/test_hand_plausibility.py -q`。
- **关联文档**：`scripts/audit/verify_project_organization.py` 做目录组织验收；评测口径见 [`scripts/eval/README.md`](../scripts/eval/README.md)；数据契约上游见 [`scripts/data/README.md`](../scripts/data/README.md)。

## 注意与坑

- 多数是 CPU-only 合同测试，但 `test_train_tiktok_v65_face_smoke.py`、`test_champ_deepgen_integration.py` 等会做有界 GPU 冒烟，跑全套前确认 GPU 空闲。
- 测试刻意「fail-closed」：身份缺失、appearance 泄漏、参考路由错乱等都会被拒绝而非静默通过——改数据/训练逻辑时要同步更新这些断言，否则会红。
- `pytest.ini` 已排除 `experiments/`、`datasets/` 等重目录；新增测试若需临时读大资产，应避免被误递归。
- `test_tiktok_prompt_bank_contract.py` 通过 glob 在 `experiments/` 下定位改写脚本，仓库布局变动（脚本路径改名）会使其整体 skip 而非失败，重命名脚本时同步更新本测试的 `PATTERNS`。

## Native Depth / CHAMP (2026-10-23)

`test_qwen21_depth_condition.py` / `test_qwen21_native_depth.py` / `test_qwen21_native_depth_runtime.py` 覆盖Depth方向/valid/canvas、CHAMP权重、512残差、pair绑定、资产漂移、checkpoint重建与原生32层反向。
