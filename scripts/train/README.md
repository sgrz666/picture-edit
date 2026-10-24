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
| `train_qwen21_native_full.py` | **等比画布正式入口（方案 §4）**：只接受 `--canvas-policy native-pad / aspect-preserving`，`resize-square`、`square-crop` 直接拒；加载模型前先在 CPU 上跑 `--preflight-only`（`assert_pair_contract` 逐对验画布、token 网格与条件同栅格），latent 缓存按 `canvas_policy` 绑定，改造前的 checkpoint 只能以 `--warm-start-ok` 显式继承 | TikTok v3（`dataset_type=tiktok_v3`，`resolution='native'`）| `--output` 指定；落 `canvas_preflight.json`、`cache/`、`checkpoint_last.pt`、`run_manifest.json` |
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

- 多数 Qwen21 训练入口默认从某个已有 checkpoint 起（V6 native 或 body 20k），改默认值前确认上游 run 存在，否则会直接报文件缺失。**2026-10-22 复核**：`execution.BODY_CHECKPOINT` 指向的 `experiments/iper_v6_native_20k_v1/checkpoint_last.pt` 在本机已不存在，三批清理清单 `/data/shangguanrz/_trash_ckpt_2026101{3,6,7}/MANIFEST.json` 里也不含任何 `iper_v6_native` 条目（不是这次整理挪走的）。但盘上**确实有**可用的改造前权重，别误记成"一个都不剩"：`nihaochen/qwen21_full/body_512_100/checkpoint_last.pt`（4.0 GB）、`qwen21_full/face_512_100/checkpoint_last.pt`（2.8 GB）、`nihaochen/iper1024/body_3k_final/checkpoint_best.pt`（772 MB）、`nihaochen/qwen21_face/face_train_3k/checkpoint_best_loss.pt`（216 MB）。它们 `canvas_policy=None`，所以 `require_canvas_compatible` 默认拒、`--warm-start-ok` 显式放行 —— 两个方向都在真实文件上验过。`train_qwen21_native_full.py --adapter-init scratch` 用 DeepGen config 随机初始化 adapter，只为验证画布/网格/反向传播接线，**不可当正式训练**。
- 竖图训练一律走 `train_qwen21_native_full.py`；旧 `train_iper_qwen21_*` 的 `--align-mode` 已无默认值，复现历史 run 要显式 `--align-mode resize --legacy-policy-ok`。
- `train_adapter.py` 是 DeepGen 线的「原始」训练入口，使用 `DiffusionPipeline.from_pretrained` 加载 DeepGen 并冻结全部组件；它的 `data_dir` 默认指向 `data/debug_samples`，正式训练需改 `--data_dir` 与 `--output_dir`。
- V6.5/V6.6 的 overfit 入口 checkpoint 默认落在 `experiments/*_overfit`，属于工程验证，不是正式训练产出；不要把它们误当发布权重。
- padding 参与注意力的问题已可**按 run 开关**：`--exclude-padding-tokens`
  （实现在 `src/pose_control/qwen21_full/padding_attention.py`，不改 diffusers）。
  实测在真实 20B 主干上：不屏蔽时只扰动 41 个纯 padding token，有效 token 输出位移 4.037（= 信号的 501%）；
  屏蔽后位移 **0.000000**；但屏蔽本身让有效输出变化 4.197（521%），
  所以**已按旧语义训练的 checkpoint 不能打开它**。默认关闭，且必须显式 `allow_semantic_change=True`。
  注意 `aspect-preserving --max-pixels 65536` 这类画布的 margin 只有 1~10 像素，凑不满一个 16×16 token 块，
  此时开关是空操作（`padding_tokens: 0`）—— 这正是"先最小化 padding"应有的表现。
- `train_qwen21_native_full.py` 实测边界（2026-10-22，RTX 6000D 83 GiB，qwen21 环境）：
  `aspect-preserving --max-pixels 65536`（画布 352×192、网格 **22×12=264 token**）2 步跑通，
  `grad_norm 16.09 → 22.58`，第 2 步 `residual_rms` 已到 **0.056–0.068**（第 1 步为 0 是 zero-head 初始化的正常表现），
  峰值显存 48.6 → 54.2 GiB，落 `checkpoint_last.pt` + `run_manifest.json`，checkpoint 带
  `canvas_schema_version=canvas_v2` 且 resume 时被 `require_canvas_compatible` 认下。
  同日又跑了一次 `--steps 4 --warmup-steps 1`（真实 `train_pairs.jsonl` 11042 对 / `val_pairs.jsonl` 1364 对，
  产物 `/data/shangguanrz/canvas_audit/native_full_train_ap_v1/`，ckpt 3.7 GB + cache 902 MB）：
  `grad_norm = 0.0 / 3.27 / 55.7 / 65.2`。**第 1 步的 0.0 是 warmup 内**（`--warmup-steps 1` 使 step 索引 0 落在
  trainable 相位为 0 的区间），不是接线坏了 —— 越过 warmup 后非零才是判据，这条断言已经封住假绿。
  两次 loss（0.226/0.484/4.05/1.99 与前一组的 16.09 那次）都**不能读成学习曲线**：每步 `sigma` 不同，
  样本也不同，4 步只证明管脚通。
  同一张卡上 `native-pad`（1088×608 → **68×38=2584 token**）**直接 OOM**：入口会把它翻译成
  「需要 gradient checkpointing，或降 `--max-pixels`」而不是丢一句 `Allocation on device`。
  另有两个自陷坑已封：`--steps <= --warmup-steps` 直接拒（warmup 期相位被置零，grad_norm 恒为 0，
  跑完看着绿其实没学）；step 越过 warmup 后 `grad_norm == 0` 抛 `AssertionError`。

## Native Depth / CHAMP (2026-10-23)

`train_qwen21_native_depth.py` 使用实际Native Geometry/Body与可选Depth；显式off/relative_png/camera_z、pair/canvas CPU预检、原生cache与flow训练。用法见 `src/pose_control/qwen21_full/NATIVE_DEPTH.md`。

## Native Geometry Body21

`train_qwen21_native_depth.py` 默认 `--geometry-version body21`，必须给出 `--pose-manifest`，接通 TikTok/iPER 的同一目标帧体型、动作与相机。新增 Pose 参数独立开放，`--freeze-pose` 可关闭；`--geometry-version legacy_v1` 保留旧路径。只改条件/配置/checkpoint 接线，原 loss、optimizer 与训练算法保持原样。本轮仅执行 `--preflight-only`。复查见 `src/pose_control/qwen21_full/GLOBAL_GEOMETRY_V2.md`。

## Native Depth V2 架构接线（2026-10-24，未训练）

新建启用Depth时默认 `--depth-version v2`，须Body21及V2 manifest；旧Depth使用 `--depth-version v1`。`--depth-mode auto`支持逐条明确声明的Camera-Z/PNG混合数据。七组参数可单独freeze，或显式 `--train-native-all`；旧loss/AdamW算法保留，没有分阶段调度。本轮没有运行preflight、测试、模型或训练。详见 `src/pose_control/qwen21_full/NATIVE_DEPTH_V2.md`。
