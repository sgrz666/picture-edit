# src/pose_control/ —— 姿态控制器的三条实现线

`pose_control/` 是 pic-edit 的控制器核心，所有"把 B 的姿态重定向到 A"的可训练逻辑都在这里。它包含三条同构但不同基座的实现：

- `v6/` —— 挂到 DeepGen-1.0-diffusers（2.47B MMDiT 双流，frozen）上的 **Unified SMPL-X Adapter V6**，目前的主线实现，分 V6.4 detail / V6.5 face / V6.6 hand 三个分支演进。
- `qwen21/` —— 挂到 Qwen-Image-2.1（7B 单流 DiT）的**简化版**：body / face / dual 三个独立控制头，扁平结构，注入 6 个固定层（3,8,13,18,23,28），无 hand 分支。
- `qwen21_full/` —— **V6 的全量 Qwen 迁移**：把 `v6/` 的全部控制架构原样复制过来（内部仍保持 1536 维表示），再把残差注入扩展到 Qwen 的 32 层（6 组），并补上面部监督、双人 Token 绑定（`PersonTokenBinder`）与训练基础设施（`execution.py`、`joint_training.py`、`frozen_bfs_lora.py`）。

## 里面有什么

- `v6/` —— 见 [v6/README.md](v6/README.md)。顶层 12 个文件 + 5 个分支子目录（detail/face/hand/interface/reasoning）。
- `qwen21/` —— 见 [qwen21/README.md](qwen21/README.md)。扁平 7 个 .py：body/face/dual 适配器、dual 推理、运行时、动作提示词。
- `qwen21_full/` —— 见 [qwen21_full/README.md](qwen21_full/README.md)。在 v6 同名文件基础上新增 qwen_adapter.py、qwen_runtime.py 及训练件。

三者共用同一套条件语义：`ConditionBundle`（normal / part_onehot / smplx_global / depth / contact 等）、`TaskType`（SINGLE/DUAL/HUG/HANDSHAKE/SHOULDER_HUG）、14 段身体分区 `PART_NAMES`。

## 与上下层的关系

- 上游：`src/data/` 产出的 `ConditionBundle` 与参考特征；`third_party/` 的面部/身份依赖；基座模型权重（DeepGen、Qwen-Image-2.1）。
- 下游：被训练脚本（qwen21_full/execution.py、joint_training.py）与推理入口通过 `UnifiedSMPLXAdapterV6`、`ControlledDeepGenPipeline`（`v6/`）或 `Qwen21FullAdapter`（`qwen21_full/`）调用。
- 更上层导航见 [../README.md](../README.md)，本目录各线入口见各自子 README。

## 注意与坑

- `v6/` 与 `qwen21_full/` 的顶层文件名高度对应（conditions.py / control_core.py / condition_injector.py / checkpoint.py / deepgen_adapter.py / controlled_pipeline.py / experts.py / geometry_encoder.py / token_encoders.py），但 qwen21_full 的对应文件是**复制后迁移**的版本，改动集中在注入坐标与 Qwen 目标流；修改时需确认你改的是哪一份。
- `qwen21_full/` 的 `detail/face/hand/interface/reasoning` 五个分支与 `v6/` 的五个分支**逐文件字节一致**（原样复制），不是独立实现。
- `smplx_global` 字段名不副实：实际是 26 维 = betas[10] + root rot6d[6] + transl[3] + cam[7]，**不含任何关节姿态**，仅用于全局形状/朝向/位移/相机条件。
