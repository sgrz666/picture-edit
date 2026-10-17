# 2026-10-13-scale/ —— A 人物缩放与骨架定标诊断（第一次 campaign）

本目录是 `workflow/diagnostics/` 的第一次诊断 campaign（2026-10-13）：排查用户指定的七组生成结果为何出现"人物被重画成骨架大小"的现象，并补看 `hard_05_torso`、`hard_05_manual` 两个变体。调查只用现有工具做 CPU 测量（DWPose ONNX），**未改动生产代码与任何历史 run**。

## 里面有什么

- `REPORT.md` —— 主报告：结论（B 骨架被等比塞进 A 可见检测框、未按 A 体型重定向）、四坏例的两种失败模式（截断构图被当成全身身高 / hard_05 宽度主导）、实际数值与证据。**先读这个**。
- `probe_source_pose.py`、`inspect_scale.py` —— 测量脚本：对 A 与生成图跑 DWPose，量躯干/框高比与骨架尺度。
- `metrics.json`、`integrity_checks.json`、`counterfactual_geometry.json`、`source_pose_measurements.json` —— 量化结果与反事实几何推演。
- 7 对 `*_visible_pose.npz`（hard_01~05、medium_01/02）+ `hard_05_manual_visible_pose.npz` —— 各对可见关键点测量原始数据。
- `good_cases.jpg` / `bad_cases.jpg` / `hard05_variants.jpg` / `a_scale_anchor_preview.jpg` —— 好坏例与变体对比图。
- `evidence.tar` —— 打包证据（解包后用；优先用目录内散文件）。

## 与上下层的关系

- 索引与背景见 [`../README.md`](../README.md)；后续的生成对照实验在同级 [`../2026-10-13-scale-generation/`](../2026-10-13-scale-generation/README.md)。
- 诊断结论落地为 `src/pose_edit/geometry.py` 的 `source_torso` 定标改动（2026-10-14，见 [`../../README.md`](../../README.md) 对应节）；整改前的整树运行备份在 `/data/shangguanrz/pic-edit-archive/workflow_runs_backup_20261012/`。

## 注意与坑

- 证据坐标基于当时画布（768），换分辨率后数值不可直接比。
- 本次只覆盖 7 对混合条件的单次历史运行，是诊断而非控制实验，引用时别当成普适结论。
- `__pycache__/` 为残留，不进证据链。
