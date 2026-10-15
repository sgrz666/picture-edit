# src/pose_control/v6/interface/ —— 控制器对外接口与强度调度

`interface/` 是 V6 控制器与基座模型之间的边界层：把内部推理状态整理成"每步去噪都消费"的静态条件，并负责把各分支残差对齐到 DeepGen 的目标/源/填充 token 序列、按调度施加强度。

## 里面有什么

- `outputs.py` —— `PreparedControlConditioning`（几何/交互条件、adapter token、人数、交互有效性、`InternalControlState`、可选 detail/face/hand 分支）、`DeepGenControlOutput`、`BranchControlResiduals`（再导出）；含 `validate()` 形状校验。
- `deepgen_interface.py` —— `DeepGenControlInterface`：`latent_token_count`（按 patch_size 算 token 数）、`align_target_residuals`（把仅目标的残差对齐到 DeepGen 的 target/source/padding 序列）。
- `strength.py` —— `ControlStrengthController` 与 `StrengthScheduleConfig`（curve ∈ {constant,linear,cosine}，progress 窗口 [start,end]，乘子起止）；`add_optional_face_residual` 由 `third_party.v65_face.xdyna` 提供。
- `__init__.py` —— 导出上述类与 `align_target_residuals`、`latent_token_count`。

## 与上下层的关系

- 上游：接收 `reasoning/` 的 `InternalControlState` 与各分支 preparer 的产物。
- 下游：被 `deepgen_adapter.py` 的 `UnifiedSMPLXAdapterV6` 装配；`PreparedControlConditioning` 喂给 `ControlledDeepGenPipeline`。
- 同构迁移版见 [../../qwen21_full/interface/README.md](../../qwen21_full/interface/README.md)。

## 注意与坑

- `latent_token_count` 要求 latent 尺寸能被 `patch_size` 整除，否则直接报错。
- 强度调度窗口必须 `0 ≤ start < end ≤ 1` 且宽度为正，配置 `StrengthScheduleConfig` 时注意。
- 对齐逻辑区分 target/source/padding 三类 token，合并分支残差时不能把源帧 token 当成目标 token 注入。
