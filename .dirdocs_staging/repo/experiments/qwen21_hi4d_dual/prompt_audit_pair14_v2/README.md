# experiments/qwen21_hi4d_dual/prompt_audit_pair14_v2/ —— pair14 提示词/数据边界审计 v2（取代 v1，含手写结论）

> 路径：`…/qwen21_hi4d_dual/prompt_audit_pair14_v2`　|　子目录 `seed42/ seed43/ seed44/` + 手写 `report.md`　|　数据源：`report.md`，2026-10-12

## 一、干什么
`prompt_audit_pair14/` 的修订版：同样用 `pair01` 训练、`pair14` 未见过的 300 步过拟合检查点，系统排查提示词、控制敏感性与目标真值泄漏；三 seed（42/43/44）、40 步、512×512。

## 二、实证（`report.md` 关键界定，人工手写）
- 明确样本与边界：源 `talk14/cam4/frame32`、目标 `hug14/cam4/frame62`；检查点训练样本是 `pair01`，**没训过本条 pair14**。
- 三档控制：原生 Qwen（仅源图+文本）、`geometry_short`（目标姿态/部位/法线/深度/相对几何，无接触）、`full_short`（+目标接触 + A/B 外观绑定）；后两档是"目标几何已知"的控制任务，渲染人物区域来自 SMPL 几何而非评测真值掩码。
- 附原生详细提示词与 Adapter 简短提示词原文，逐档对比生成对提示词的响应。
- 结论定性为"排查"，**不作为泛化/质量验收**。

## 三、关系
取代 v1（`prompt_audit_pair14/`）；下游真正带权重的样本审计是 `single_pair14_action_v2_overfit/`。本目录提供双人评测的"可观测性与数据边界"论证背景。
