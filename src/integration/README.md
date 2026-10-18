# src/integration/ —— CHAMP / DeepGen 样本校验与残差对齐

`integration/` 提供基座流水线的样本校验工具：把 CHAMP 源/目标样本与其结构控制（normal/depth/pose/semantic）对齐，并在不加载控制器的情况下调用 DeepGen 基线，产出误差与诊断。

## 里面有什么

- `champ_deepgen.py` —— `ChampSample`（对齐的源/目标对 + 控制张量 + 路径/元数据，`CONTROL_CHANNELS` 为 normal_r/g/b、depth、pose_r/g/b、semantic）；`load_champ_sample`、`align_control_residuals`、`call_deepgen_without_control`（无控制器基线调用）、`image_error_metrics`、`image_diagnostics`。
- `__init__.py` —— 空。

## 与上下层的关系

- 上游：CHAMP 数据集样本与 `adapter/pose_adapter.py` 同款控制通道约定。
- 下游：被基座流水线（workflow/）的样本校验与无控制基线对比调用。

## 注意与坑

- `CONTROL_PIPELINE_ARGUMENTS` 固定为 `{"block_controlnet_hidden_states","control_scale"}`，与 `pose_control/v6` 的注入接口一致；改控制通道时需同步。
- `call_deepgen_without_control` 走基线（无 adapter），用于隔离"控制器贡献"的消融。
- `align_control_residuals` 假设控制图与目标图已按同一分辨率/letterbox 对齐，跨分辨率样本需先统一。
