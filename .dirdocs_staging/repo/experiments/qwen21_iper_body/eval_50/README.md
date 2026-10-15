# experiments/qwen21_iper_body/eval_50/ —— Qwen21 单臂 50 样本评测（40 步，四臂协议的前身）

> 路径：`…/qwen21_iper_body/eval_50`　|　`summary.json` + `per_sample.json` + `images/`（152 文件 / 45.4 MB）　|　数据源：`summary.json`，2026-10-12

## 一、在干什么
在引入跨架构对照之前，先对 Qwen21 身体适配器单臂在 50 对 held-out iPER 上做定量评测（`checkpoint=full_20k/checkpoint_last.pt`，steps=40，seed=42，`kv_cache=true`）。后被 `four_arm_eval_50*`（同时评 DeepGen）取代，作为单基座快速回归保留。

## 二、内容实证
- `summary.json` 顶层键：`checkpoint / model / model_revision / pairs / steps(=40) / seed(=42) / prompt / kv_cache / num_samples(=50) / timing_includes_control_geometry / mean`。
- 其 `mean` 对应四臂表里的 `qwen21_on` 列（PSNR 25.86、PCK@0.02 98.00%、Face ID 0.3510）。
- `images/` 存 50 张 Qwen21 受控生成图。

## 三、关系与注意
`eval_50_s30/` 是其 30 步变体；`four_arm_eval_50_s40/` 是在同协议上扩到四臂的最终版。引用总数值以四臂目录为准。
