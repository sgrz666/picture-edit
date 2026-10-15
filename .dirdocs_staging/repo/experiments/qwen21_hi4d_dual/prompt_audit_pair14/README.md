# experiments/qwen21_hi4d_dual/prompt_audit_pair14/ —— pair14 双人提示词与目标真值泄漏审计 v1

> 路径：`…/qwen21_hi4d_dual/prompt_audit_pair14`　|　子目录 `seed42/ seed43/ seed44/`（各含 `summary.json`、`per_sample.json`、`images/`）+ `leak_and_geometry/`（`audit.json`、`frame34_geometry.png`、`frame62_baseline.png`、`frame62_replaced_gt.png`、`geometry_switch_panel.png`）　|　2026-10-12

## 一、干什么
把 `pair01` 训出来的 300 步双人过拟合检查点，套用在同源不同条的 `pair14`（源 `talk14/cam4/frame32` → 目标 `hug14/cam4/frame62`）上，**排查两件事**：(1) 提示词/控制注入是否真的改变结果（控制敏感性）；(2) 目标几何/接触是否被当作评测真值泄漏进生成（`leak_and_geometry`）。这是数据边界审计，**不作泛化或质量验收**。

## 二、实证
- 三个 seed 目录：每组相同源图、40 步，比较 `geometry_short`（接收目标姿态/部位/法线/深度/相对几何，不含接触）与 `full_short`（额外接收目标接触及 A/B 外观绑定）——后两组是"已知目标几何的控制任务"。
- `leak_and_geometry/audit.json` + `frame62_replaced_gt.png`：把渲染人物区域（来自 SMPL/geometry，非真值掩码）与目标真值 frame62 对照，量化泄漏嫌疑。

## 三、关系
**v2 已取代本 v1**（见 `prompt_audit_pair14_v2/`，含手写 `report.md`）。保留本目录作 v1 历史留档。
