# experiments/qwen21_iper_body/eval_50_s30/ —— Qwen21 单臂 50 样本评测（30 步）

> 路径：`…/qwen21_iper_body/eval_50_s30`　|　`summary.json` + `per_sample.json` + `images/`（152 文件 / 45.3 MB）　|　数据源：`summary.json`，2026-10-12

## 一、在干什么
`eval_50/` 的 30 步采样变体：单臂 Qwen21（`full_20k/checkpoint_last.pt`，seed 42，50 held-out 对），用于考察降低步数对单基座指标的影响。

## 二、内容实证
`summary.json`：`steps=30`、`num_samples=50`、`checkpoint=full_20k/checkpoint_last.pt`、`kv_cache=true`；均值口径与 `eval_50`（40 步）几乎重合（PCK@0.02 稳定在 98% 量级）。`images/` 50 张生成图。

## 三、关系
是 `eval_50/` 的步数消融，亦是 `four_arm_eval_50_s30/` 单臂前身。正式四臂结论见 `four_arm_eval_50_s40/`。
