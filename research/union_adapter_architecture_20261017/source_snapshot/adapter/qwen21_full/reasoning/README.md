# src/pose_control/qwen21_full/reasoning/ —— 几何与交互推理网络（全量迁移副本）

本目录是从 `v6/reasoning/` **原样迁移（逐文件字节一致）** 到 qwen21_full 的副本，是 V6 控制核心的"静态推理"部分：把 `ConditionBundle` 转换为 batch 安全的几何/交互状态（`InternalControlState`）。迁移到 Qwen-Image-2.1 后，推理产出的状态仍走 `condition_bridge` 与 `interface` 注入，本目录的推理逻辑未改。

## 里面有什么

- `config.py` —— `AdapterReasoningConfig`（frozen dataclass）：condition_dim=256、hidden_dim=512、reasoning_downsample_factor=2、各注意力头/层数与 ffn_ratio；多处硬约束。
- `state.py` —— `InternalControlState`：几何/交互特征与高分辨率版、person token/mask、有效性、人数、交互有效性；含 `validate()`。
- `person_geometry.py` —— `PersonGeometryReasoner`：A、B 共用的局部/全局推理路径。
- `cross_person.py` —— `BidirectionalCrossPersonReasoner`：A↔B 同时更新、各方向共享权重。
- `contact.py` —— `ContactConditionReasoner`：用门控把 contact 栅格与类型化关系 token 注入双人状态（`contact_gate_init=-4.0`）。
- `fusion.py` —— `DualFeatureFusion`：产出两个尺度上独立的几何与交互控制。
- `blocks.py` —— `ReasoningResBlock`、`CrossAttentionFFN`、`group_count` 等基础构件。
- `reasoner.py` —— `SMPLXAdapterReasoner`：装配上述，`_role_bind` 把角色嵌入加到 person token。
- `__init__.py` —— 导出 `SMPLXAdapterReasoner`、`InternalControlState`、`AdapterReasoningConfig`。

## 与上下层的关系

- 上游：接收 `../condition_injector.SMPLXConditionInjector` 产出的 bundle。
- 下游：状态经 `../condition_bridge.ReasonerControlBridge` 与 `../interface/` 进入 `../control_core` 注入 Qwen 的 32 层。
- 源目录见 [../../v6/reasoning/README.md](../../v6/reasoning/README.md)；上层见 [../../qwen21_full/README.md](../../qwen21_full/README.md)。

## 注意与坑

- `AdapterReasoningConfig` 的 condition_dim 固定 256、reasoning_downsample_factor 固定 2（`__post_init__` 强校验）。
- `InternalControlState.validate()` 对 shape 严格校验，运行时不符会直接抛错。
- 本目录与 `v6/reasoning/` 逐文件一致；改通用推理逻辑要确认改的是哪一份。
