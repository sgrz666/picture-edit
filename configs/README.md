# configs/ —— 架构契约与资产登记

本目录是人物图像姿态编辑系统的**静态配置中枢**，只放「机器可读的契约与登记」，不含可执行代码。它把两件事固化下来：① Unified SMPL-X Adapter V6 的架构契约（不训练、只描述 forward 形态）；② 项目实体路径的人工可读登记（模型、缓存、受保护 run/data 等）。下游的 `src/pose_control/` 实现、`scripts/train/` 训练与 `scripts/audit/` 验收都对照这里的契约。父级整理说明把 `configs/` 定位为架构与资产登记目录。

## 里面有什么

### 1. `adapter_v6_architecture.json` —— V6 架构契约
声明 V6 adapter 的 forward-only 架构形态（`scope: architecture-and-forward-only-no-training`，`version: 6.3-dynamic-deepgen-interface-v1`）。顶层键：

| 顶层键 | 内容指向 |
|---|---|
| `version` / `scope` | 架构版本与范围（仅架构与 forward，不含训练）|
| `max_people` / `person_axis_is_fixed` | 最多 2 人；person 轴是否固定（当前为 false，支持双人动态）|
| `condition_injection` | 条件注入规格：输入/输出分辨率、normal 3 通道、pose 25 通道、part 14 通道、global 参数 26D、relative 11D、spatial 通道、token 维度、每 person 全局 token 数、contact token、role_binding（film）、缺失人体掩码策略 |
| `tokens` | token 布局：hidden_dim 1536、每 person geometry 4 / appearance 8、relative 2、contact 8、task 1，tokenizer 是否修改 |
| `reasoning` | reasoner 规格：condition/hidden dim、下采样、各注意力头数与层数、FFN 比、contact_gate 初值、输出集合（geometry-low/high、interaction-low/high）、mixed_batch_routing（gather-dual-scatter）、appearance 是否独立 |
| `dual_person_reasoner` | 双人 reasoner 配置 |
| `control_core` / `control_strength` | 控制核心与控制强度调度 |
| `inference_cache` | 推理缓存策略 |
| `deepgen_injection` | 注入 DeepGen 的方式（block 0-22 残差注入等）|
| `parameter_policy` | 参数策略（哪些可训练/冻结）|

### 2. `asset_registry.json` —— 资产路径登记
人工可读的项目实体路径登记（`schema_version: 1`，`as_of: 2026-10-03`），明确「不覆盖现有训练配置与默认值，实体切换状态以 `archive/cleanup_20261003/assets.json` 为准」。顶层键：

| 顶层键 | 内容指向 |
|---|---|
| `project_root` / `purpose` | 项目根与登记用途说明 |
| `models` | 模型登记：deepgen（`models/DeepGen-1.0-diffusers`，含 `protected_runtime`）、qwen21（`models/Qwen-Image-2.1`，存储于 `/data/.../Qwen-Image-2.1`）、legacy_qwen_edit_20b（兼容 `model-qwen-edit`）|
| `cold_runs` | 冷归档 run：`review_20260908_v2`、`iper_v6_comparative_3000` 等（含 entry 与归档 storage）|
| `caches` | 缓存登记：hi4d_qwen21、iper_v6_shared（condition_cache，训练/评测默认依赖）、iper_qwen21 等 |
| `raw_3dpw_archive` | 原始 3DPW 归档位置 |
| `reports_qwen21` | Qwen21 报告相关路径 |
| `environments` | 运行环境登记（各 conda 环境）|
| `protected_runs` / `protected_data` | 受保护 run 与数据（整理时不迁移、不清理）|


## 与上下层的关系

- **被消费方**：`src/pose_control/v6/`（按 `adapter_v6_architecture.json` 实现与校验架构）、`scripts/train/`（训练入口对照参数策略）、`scripts/audit/`（按契约做架构保全/权重审计）、`scripts/data/`（条件通道数等来自 `condition_injection`）。
- **资产落地**：`asset_registry.json` 登记的模型/缓存/受保护项，对应 `models/`、`experiments/`、`datasets/`、`caches/` 等真实目录；与 `archive/cleanup_20261003/` 的 `organization.json`、`assets.json` 互补（前者是整理映射，后者是实体切换真相源）。
- 相关索引：训练线见 [`scripts/train/README.md`](../scripts/train/README.md)；目录组织验收见 [`scripts/audit/README.md`](../scripts/audit/README.md) 的 `verify_project_organization.py`。

## 注意与坑

- `adapter_v6_architecture.json` 是**架构契约而非超参配置**：`scope` 明确「no-training」，改它等于改 forward 形态，会牵连 `src/pose_control/` 实现与一众 V6 架构测试（`test_adapter_v6_architecture.py` 等），需同步更新测试。
- `asset_registry.json` 自身声明「不覆盖现有训练配置与默认值」，且实体切换以 `archive/cleanup_20261003/assets.json` 为准——它只是可读登记，不是权威真相源，不要把它当成路径的唯一裁决。
- 两个 json 都是整理（2026-10-03）时落定的，若后续模型存储路径迁移，应同步更新 `asset_registry.json` 与归档 `assets.json`，否则 `scripts/audit/` 验收会报资产缺失。
