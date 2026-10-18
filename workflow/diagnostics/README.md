# diagnostics/ —— 诊断性调查的归档

diagnostics/ 是 workflow 的「诊断档案室」：每当需要对某次失败或异常做深入排查、或验证一个修复假设时，把调查过程与证据留在按日期分组的 campaign 子目录里。它只做调查与归档，**不改生产代码（`src/pose_edit/`、`workflow/run.py` 等），也不改写历史 run**；结论若需落地，回到正式流水线或 `docs/` 去做。

## 里面有什么

按日期分的 campaign，每个 campaign 自带说明 + 脚本 + 证据：

- `2026-10-13-scale/`：A 人物缩放与骨架定标诊断（主报告 `REPORT.md`）。结论：当前流程把 B 的完整骨架等比缩放、居中塞进 A 的可见检测框，并未在 A 体型上重定向动作，导致近景/截断人物被重画成骨架大小。四坏例分两种失败模式（下身截断构图被当成全身身高；`hard_05` 长裙撑宽框使缩放被宽度主导）。证据含 `probe_source_pose.py`、`inspect_scale.py`、`metrics.json`、`integrity_checks.json`、`counterfactual_geometry.json`、`source_pose_measurements.json`、7 对 `visible_pose.npz`、`good_cases.jpg`/`bad_cases.jpg`、`hard05_variants.jpg`/`a_scale_anchor_preview.jpg`、`evidence.tar`。
- `2026-10-13-scale-generation/`：source-anchor 生成实验（计划 `PLAN.md`、报告 `README.md`）。用 A 实测尺度（肩中→髋中）+ 骨盆锚定重生成 4 坏例 + `medium_02` 对照 + `hard_02` 提示词修正版，产物在 [`../runs/scale_anchor_20261013/`](../runs/README.md)。脚本 `geometry_ablation.py`、`run_experiment.py`、`run_prompt_variant.py`、`analyze_results.py`、`test_geometry_ablation.py`；产物 `measurements_summary.json`、多张 `comparison*.jpg`、`microphone_prompt_comparison.jpg`、`recommended_comparison_1.jpg`。

## 与上下层的关系

- 上游证据来自正式 run：骨架数组与图像读自 [`../runs/`](../runs/README.md)（本次基线 7 对、hard_05 变体），诊断脚本不修改它们。
- 结论去往：若确认为流水线缺陷，修复落到 `../src/pose_edit/geometry.py`（如 2026-10-14 的 `source_torso` 实测躯干改动，见 [`../README.md`](../README.md) 对应节）；过程与整改对应见 `../docs/`。
- 流水线总说明见 [`../README.md`](../README.md)；运行产物见 [`../runs/README.md`](../runs/README.md)。

## 注意与坑

- 每个 campaign 是一次性调查，**不是可重复的实验基础设施**：`__pycache__/` 与 `.pyc` 是残留，不进证据链；重跑请用 campaign 自带 `run_experiment.py --verify` 等入口，不要直接 import 散脚本。
- 证据里的 `.npz`/`.json` 截图坐标基于当时画布分辨率（本次 768），换分辨率后数值不可直接比。
- `evidence.tar` 是 `2026-10-13-scale` 的打包证据，需解包才有内部文件；优先用目录内的散文件。
- 诊断结论只是「这次测量的现象」：例如 scale 诊断只覆盖 7 对混合条件的单次历史运行，不是控制实验，引用时别当成普适结论。
