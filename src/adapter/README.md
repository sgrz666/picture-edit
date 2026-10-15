# src/adapter/ —— 独立姿态条件适配器（早期实现）

`adapter/` 是项目早期/独立路线的姿态条件适配器，与 `pose_control/` 的 V6 主线并行存在。当前主线控制器在 `pose_control/`，本目录更多是 ControlNet 式条件编码的雏形实现。

## 里面有什么

- `pose_adapter.py` —— `PoseConditionAdapter`（ControlNet 式）：把单张控制图（normal/depth/pose/semantic 等多通道）编码为各层级特征列表 `List[Tensor]`；基础构件 `ResnetBlock2D`、`Downsample2D`（GroupNorm + SiLU + conv，含 shortcut）。
- `__init__.py` —— 空。

## 与上下层的关系

- 上游：外部拼好的控制图，通道约定与 `integration/champ_deepgen.py` 的 `CONTROL_CHANNELS` 一致（normal_r/g/b、depth、pose_r/g/b、semantic）。
- 下游：可接支持 `block_controlnet_hidden_states` 风格的基座；与 `pose_control/` 主线无直接 import 关系。

## 注意与坑

- 本目录与 `pose_control/` 是两条独立演进的线，新增功能优先在 `pose_control/`，不要在本目录重复造轮子。
- `PoseConditionAdapter.forward` 返回的是**列表**而非单一张量，下游消费时注意按层级索引。
