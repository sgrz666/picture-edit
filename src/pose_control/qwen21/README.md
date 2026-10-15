# src/pose_control/qwen21/ —— Qwen-Image-2.1 简化姿态控制器

`qwen21/` 是把姿态控制迁移到 Qwen-Image-2.1（7B 单流 DiT）的**简化版**，扁平结构、无子目录。它复用 V6 的几何推理（`src/pose_control/v6/condition_bridge`、`condition_injector`、`reasoning`）产出几何/交互状态，再在 Qwen 的 6 个固定层（3,8,13,18,23,28）注入 target-only 残差。

## 里面有什么（逐个文件）

- `body_adapter.py` —— `Qwen21BodyAdapter`：复用 V6 几何推理，产出 target-only 残差；`LAYER_INDICES=(3,8,13,18,23,28)`、`zero_heads` 各层线性头（零初始化）、`Qwen21ControlContext`（注册 forward hook 注入）。`face_strength` 默认 0.0。
- `face_adapter.py` —— `Qwen21FaceAdapter` / `Qwen21FaceState`：仅在目标头部 ROI 内加源脸身份/纹理；`FACE_LAYERS` 同为 6 层。依赖 `third_party.v65_face.visual_persona.IPAttnProcessor2_0`。
- `dual_adapter.py` —— `Qwen21DualAdapter`（继承 body）、`Qwen21DualState`（geometry/interaction/appearance + `interaction_strength`/`appearance_strength`）：在 iPER 单路径上追加可训练的双人路径，`geometry_only()` 关掉交互与外观。
- `dual_inference.py` —— `NativeGenerationInput`/`DualGenerationInput`、`generation_inputs_from_sample`：Hi4D 双人编辑的**无目标 RGB**生成边界；明确 `tgt_image`/`target_rgb`/`gt_mask` 不越界。
- `runtime.py` —— 锁定的 Qwen-Image-2.1 图像/latent 契约（`MODEL_REVISION`、`PROMPT`、`RESOLUTION=512`、`Qwen21IPERCache`、`transformer_inputs`）。
- `action_prompts.py` —— `build_dual_action_prompts` 与确定性动作文案（highfive/fight/talk/...，版本 `hi4d_dual_action_v2`）。
- `__init__.py` —— 模块 docstring：Qwen-Image-2.1 target-token controls for iPER Body and Face editing。

## 与上下层的关系

- 上游：复用 `src/pose_control/v6/` 的 `SMPLXConditionInjector`、`SMPLXAdapterReasoner`、`ReasonerControlBridge`；数据集侧来自 `src/data/`（iPER/TikTok/Hi4D）。
- 下游：由 `runtime.py` 的契约与 `dual_inference.py` 的边界驱动 Qwen-Image-2.1 推理；训练在 iPER 上进行。
- 完整 32 层迁移版见 [../qwen21_full/README.md](../qwen21_full/README.md)；V6 DeepGen 版见 [../v6/README.md](../v6/README.md)。

## 注意与坑

- **无 hand 控制分支**：本目录没有 hand adapter、全仓 grep 也不存在 `hand_strength`（该参数仅出现在 `qwen21_full/`），手部细节完全依赖基座模型，未做重定向增强。
- `Qwen21BodyAdapter` 默认 `face_strength=0.0`，要启用脸部需显式传 `face_adapter` 与非零 `face_strength`。
- 注入层固定为 `LAYER_INDICES` 的 6 层，与 qwen21_full 的 32 层全量注入不同；改层需同时改 body/face 两处常量。
