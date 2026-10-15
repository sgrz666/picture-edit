# scripts/train/ —— 训练入口

本目录是人物图像姿态编辑系统的**训练层**，集中所有训练入口。项目有三条技术线，训练脚本也据此分组：DeepGen-1.0-diffusers（2.47B MMDiT 双流，frozen）+ 自研 Unified SMPL-X Adapter 的 V6 系列；Qwen-Image-2.1（7B 单流 DiT）迁移线（简化 / full）；以及 Hi4D 双人与 TikTok 过拟合 sanity 等工程验证入口。各入口把可训练 adapter / 分支接到冻结基座上，把 checkpoint 落到 `experiments/<run>/`。父级 [`../README.md`](../README.md) 把 `train/` 定位为「保留 V6、Qwen 简化/Full 和数据集前缀」的入口目录（例：`scripts/train/train_iper_qwen21_full.py --help`）。注意：本目录的大部分训练脚本通过 `Path(__file__).resolve()` 解析到原文件，从分类入口执行不改变 ROOT、源码哈希或模块身份。

## 里面有什么

### DeepGen V6 / Native 训练

| 文件 | 训练什么 | 数据 | 权重落盘 |
|---|---|---|---|
| `train_adapter.py` | 在配对图（8 通道 SMPL-X 控制图）上训练 `PoseConditionAdapter`（in_channels=8，注入全部 transformer block）；冻结整个 DeepGen 基座（VAE/transformer/VLM/connector） | 默认 `data/debug_samples`（含 `pairs.json`、`images/`、`controls/`）| 默认 `experiments/adapter_v1`，存 `adapter_step_{n}.pt`（含 adapter/optimizer 状态与 args）|
| `train_iper_v6_native.py` | V6.3 Native 单人称几何（Scheme A）：训练 Geometry 分支、Appearance identity 绑定、动态控制核心、Zero heads；冻结 Interaction/Contact/Person-B/DeepGen | iPER 配对 + 条件缓存（带 mask 加权 flow-matching 速度损失、分层学习率、滚动 checkpoint）| 默认 `experiments/iper_v6_native_finetune_v1`，支持 `--resume_from`、`--save_steps`、`--max_rolling_checkpoints` |

### Qwen21 迁移线（简化 / Body / Face / Full）

| 文件 | 训练什么 | 数据 | 权重落盘 |
|---|---|---|---|
| `train_iper_qwen21_body.py` | 在冻结 Qwen-Image-2.1 主干上训练 target-token body adapter（默认从 V6 checkpoint 起）| iPER；默认从 `experiments/iper_v6_native_20k_v1/checkpoint_last.pt` 起 | 默认 `experiments/qwen21_iper_body` |
| `train_iper_qwen21_face.py` | 在冻结 Body 20k adapter 之上训练 source-identity 的 Qwen21 Face ROI 分支 | iPER 人脸；默认 Body checkpoint `experiments/qwen21_iper_body/full_20k/checkpoint_last.pt` | 默认 `FACE_ROOT/face_train_3k` |
| `train_iper_qwen21_full.py` | 完整 V6 架构在 Qwen21 上的有界 Body/Face 联合训练 | iPER | 默认由脚本内/参数指定 `experiments/` 下 run 目录 |
| `train_qwen21_joint_pilot.py` | 有界 joint Body+Face pilot，带固定可选 BFS 与严格 resume | 锁定子集（与 `scripts/data/prepare_qwen21_joint_pilot.py` 对齐）| 由参数指定 `experiments/` 下 pilot 目录 |

### V6.5 / V6.6 分支（Face / Hand）

| 文件 | 训练什么 | 数据 | 权重落盘 |
|---|---|---|---|
| `train_iper_v65_face.py` | V6.5 独立 Face Adapter 单样例 iPER 训练（轻量 helper 可无 Diffusers/GPU 导入）| iPER 单样例 overfit loader | 默认 `experiments/iper_v65_face_overfit`，`--checkpoint-every 250` |
| `train_tiktok_v65_face.py` | V6.5 独立 Face Adapter 单样例 TikTok 训练（镜像 iPER 流程，用 `TikTokV65FaceOverfitLoader`）| TikTok 单样例 | 默认 `experiments/tiktok_v65_face_overfit`，`--checkpoint-every 250` |
| `train_iper_v66_hand.py` | V6.6 Hand 固定 pair iPER overfit；legacy/off/v66 共享随机状态（从 V65 checkpoint 起，冻结供 hand 训练）| iPER 固定 pair | 默认 `experiments/iper_v66_hand_overfit`，`--checkpoint-every 250` |

### Hi4D 双人与 TikTok 工程验证

| 文件 | 训练什么 | 数据 | 权重落盘 |
|---|---|---|---|
| `train_hi4d_qwen21_dual.py` | 从 Hi4D N2C pair 训练 Qwen21 双人控制（默认从 body 20k checkpoint 起）| Hi4D N2C pairs | 默认 `experiments/qwen21_hi4d_dual/action_prompt_v2`；`--single-checkpoint` 默认 `experiments/qwen21_iper_body/full_20k/checkpoint_best.pt` |
| `train_tiktok_overfit_sanity.py` | V6.5 TikTok 管线两阶段单样例过拟合 sanity：Phase A 固定噪声+固定 t 必须让 loss 塌到 0（抓数据/VAE/loss 接线/静默冻结/checkpoint 错误），Phase B 验证从 100% 噪声的多步生成也能泛化 | TikTok 单样例 | 默认 `experiments/tiktok_overfit_sanity`，`--checkpoint-every 500` |

## 与上下层的关系

- **上游输入**：`src/adapter/pose_adapter.py`（PoseConditionAdapter）、`src/pose_control/`（v6 / qwen21 / qwen21_full 控制实现）、`models/`（DeepGen-1.0-diffusers、Qwen-Image-2.1）、`scripts/data/` 产出的 manifest 与条件缓存。
- **产出物**：`experiments/<run>/` 下的 adapter/controller checkpoint（`.pt`）、优化器状态、训练日志；这些产物被 `scripts/eval/` 评测、`scripts/run/` 编排续跑、`scripts/reports/` 写成报告。
- **验收**：`scripts/audit/` 的 `verify_qwen21_full_resume.py`、`verify_qwen21_full_runtime.py`、`check_qwen21_full_preservation.py` 等验收 checkpoint 可复现性与运行时正确性；训练数据契约由 `tests/`（`test_tiktok_data_contract.py`、`test_iper_v6_training_pipeline.py` 等）守护。
- 相关索引：父级 [`../README.md`](../README.md)；运行编排见 [`../run/README.md`](../run/README.md)；评测对照见 [`../eval/README.md`](../eval/README.md)。

## 注意与坑

- 多数 Qwen21 训练入口默认从某个已有 checkpoint 起（V6 native 或 body 20k），改默认值前确认上游 run 存在，否则会直接报文件缺失。
- `train_adapter.py` 是 DeepGen 线的「原始」训练入口，使用 `DiffusionPipeline.from_pretrained` 加载 DeepGen 并冻结全部组件；它的 `data_dir` 默认指向 `data/debug_samples`，正式训练需改 `--data_dir` 与 `--output_dir`。
- V6.5/V6.6 的 overfit 入口 checkpoint 默认落在 `experiments/*_overfit`，属于工程验证，不是正式训练产出；不要把它们误当发布权重。
- 训练脚本质疑「是否真的在学」时，优先用 `train_tiktok_overfit_sanity.py` 的两阶段 sanity 验证接线正确性。
