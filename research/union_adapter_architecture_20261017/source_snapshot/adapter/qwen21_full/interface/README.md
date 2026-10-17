# src/pose_control/qwen21_full/interface/ —— 控制器对外接口与强度调度（全量迁移副本）

本目录是从 `v6/interface/` **原样迁移（逐文件字节一致）** 到 qwen21_full 的副本，是 V6 控制器与基座模型之间的边界层：把内部推理状态整理成每步去噪都消费的静态条件，并负责把各分支残差对齐到 Qwen 的目标/源/填充 token 序列、按调度施加强度。迁移后 `latent_token_count` / `align_target_residuals` 面向 Qwen 的 patch/token 布局。

## 里面有什么

- `outputs.py` —— `PreparedControlConditioning`（几何/交互条件、adapter token、人数、交互有效性、`InternalControlState`、可选 detail/face/hand 分支）、`DeepGenControlOutput`、`BranchControlResiduals`（再导出）；含 `validate()`。
- `deepgen_interface.py` —— `DeepGenControlInterface`：`latent_token_count`、`align_target_residuals`（把仅目标的残差对齐到目标/源/padding 序列）。
- `strength.py` —— `ControlStrengthController` 与 `StrengthScheduleConfig`（curve ∈ {constant,linear,cosine}，progress 窗口 [start,end]，乘子起止）；`add_optional_face_residual` 由 `third_party.v65_face.xdyna` 提供。
- `__init__.py` —— 导出上述类与 `align_target_residuals`、`latent_token_count`。

## 与上下层的关系

- 上游：接收 `../reasoning/` 的 `InternalControlState` 与各分支 preparer 的产物。
- 下游：被 `../deepgen_adapter.py` 的 `UnifiedSMPLXAdapterV6` 装配；`PreparedControlConditioning` 喂给 Qwen 注入上下文。
- 源目录见 [../../v6/interface/README.md](../../v6/interface/README.md)；上层见 [../../qwen21_full/README.md](../../qwen21_full/README.md)。

## 注意与坑

- `latent_token_count` 要求 latent 尺寸能被 `patch_size` 整除，否则直接报错。
- 强度调度窗口必须 `0 ≤ start < end ≤ 1` 且宽度为正。
- 本目录与 `v6/interface/` 逐文件一致；改通用接口逻辑要确认改的是哪一份。
