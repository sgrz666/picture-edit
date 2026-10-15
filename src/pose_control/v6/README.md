# src/pose_control/v6/ —— Unified SMPL-X Adapter V6（DeepGen 主线）

`v6/` 是挂在 DeepGen-1.0-diffusers（2.47B MMDiT 双流，推理 frozen）上的自研可训练姿态控制器，也是当前主线。数据流分三步：**条件编码**（`condition_injector.py` + `geometry_encoder.py` + `token_encoders.py`）→ **循环控制核心**（`control_core.py` 的 `SharedRecurrentControlCore`，由 DeepGen 的某个 block 克隆而来，含独立的 geometry/interaction 动态状态）→ **残差注入**（`deepgen_adapter.py` 经 `block_controlnet_hidden_states` 把残差送到 DeepGen，block 0–22 注入、context_pre_only 的第 23 层不注入）。版本演进：V6.4 detail 分支、V6.5 face 分支、V6.6 hand 分支（`checkpoint.py` 中 `ARCHITECTURE_VERSION="v6.6"`，`V64/V65/V66_BRANCH_NAMES` 记录各版本分支集合）。

## 里面有什么（顶层文件）

- `conditions.py` —— 公共条件数据结构：`ConditionBundle`、`AdapterIdentityCondition`、`TaskType`、`PART_NAMES`（14 段身体）、`ContactRelationBatch`，以及保持离散 dtype 的 `_to_preserving_discrete_dtype` 工具。
- `condition_injector.py` —— `SMPLXConditionInjector`：把准备好的 normal/part_onehot/smplx_global/depth/contact 编码为稳定 bundle；**止于跨人推理之前**，不估 SMPL-X 也不渲染图。固定公开契约（spatial_dim=256、token_dim=256/1536、contact_channels=128、max_contact_relations=8）。
- `geometry_encoder.py` —— `SpatialConditionEncoder`（`normalize_scene_depth`）与 `NativeConditionStem`/`ConditionResBlock`，逐模态编码空间条件。
- `token_encoders.py` —— 各类 token 编码器：`AppearanceTokenEncoder`（IP-Adapter 式、每人 8 个解耦身份 token）、`PersonTokenBinder`（把几何/外观 token 绑到同一源/角色槽）、`GlobalConditionTokenEncoder`、`RelativeGeometryTokenEncoder`、`ContactRasterEncoder`、`StructuredContactRelationEncoder`。
- `reasoning/` —— `SMPLXAdapterReasoner` 把 `ConditionBundle` 推理为几何/交互状态，见 [reasoning/README.md](reasoning/README.md)。
- `control_core.py` —— `SharedRecurrentControlCore`（克隆 DeepGen block、独立 geometry/interaction 状态）、`BranchControlResiduals`、`CrossNorm`、`LowRankStageAdapter`、`_TinyPatchEmbed` 等。
- `condition_bridge.py` —— `ReasonerControlBridge`：让 geometry 与 interaction 路径在 DeepGen 控制侧保持独立。
- `experts.py` —— `TaskRouter`：仅依据已校验的人数（1 或 2）做 SINGLE/DUAL 路由。
- `deepgen_adapter.py` —— `UnifiedSMPLXAdapterV6`：装配上述全部组件，对外以 `forward_deepgen(...)` 产出 `block_controlnet_hidden_states`。
- `controlled_pipeline.py` —— `ControlledDeepGenPipeline`：在不改 DeepGen 运行时代码的前提下注入控制器（带 `_state_lock` 防止嵌套/并发调用）。
- `interface/` —— 控制器对外接口：`DeepGenControlInterface`、`PreparedControlConditioning`、`StrengthScheduleConfig` 与强度调度，见 [interface/README.md](interface/README.md)。
- `checkpoint.py` —— 版本/分支常量与 v6.3–v6.6 的 load/build 函数，以及 `detail_only`/`face_only`/`hand_only` 的 `*_ONLY_PREFIXES` 冻结前缀。
- `detail/ face/ hand/` —— V6.4/6.5/6.6 三个分支模块，见各自 README。
- `__init__.py` —— 统一导出 `UnifiedSMPLXAdapterV6`、`SMPLXConditionInjector`、`SMPLXAdapterReasoner`、各分支 preparer/adapter 等。

## 与上下层的关系

- 上游：由 `src/data/` 或 `qwen21_full/data_runtime.py` 提供 `ConditionBundle` 与参考特征；基座为 DeepGen-1.0-diffusers（frozen）。
- 下游：训练经 `v6/` 内部 forward；推理经 `ControlledDeepGenPipeline` 包裹 DeepGen pipeline。
- 同构迁移版见 [../qwen21_full/README.md](../qwen21_full/README.md)，简化版见 [../qwen21/README.md](../qwen21/README.md)。

## 注意与坑

- 第 23 层（context_pre_only block）不接收残差注入，注入只覆盖 block 0–22。
- `condition_injector` 显式禁用 `use_full_pose_tokens`（保留但未启用），不要误以为全局 token 含完整关节姿态。
- `smplx_global` 仅 26 维（betas10 + root rot6d6 + transl3 + cam7），与"全局 SMPL-X 参数"的字面含义不符，不含任何关节角。
- `UnifiedSMPLXAdapterV6` 构造时会调用 `cls._freeze_backbone(deepgen_transformer)` 冻结基座参数，只训练本目录内的控制结构。
