# experiments/16-test/stage1_baselines/ —— Stage1：Qwen2.1 基座纯 Prompt 直出（最弱基线）

> 路径：`…/16-test/stage1_baselines`　|　`results.json` + `images/`(22 图) + `qwen_base_grid.png`　|　数据源：`results.json`，2026-10-12

## 干什么
不给任何几何 Adapter，仅让 Qwen-Image-2.1 基座读 prompt（含 zh/en、simple/complex 变体）直出，测纯语言/图像先验的极限。

## 实证
`results.json` 4 例，`config=qwen_base`，单例耗时 **160.14s**、输出 640×1088、steps=40、cfg=1.0、seed=42。结论：基座无法按图二精确摆姿，易遗篮球/复制图二白影棚背景。

## 关系
流水线第 1 站；受控改进见 `../stage2_adapters/`。冒烟 `_smoke/` 对应本 runner。
