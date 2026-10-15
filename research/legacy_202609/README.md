# research/legacy_202609/ —— 2026 年 9 月历史调研与实验证据归档

本目录是 2026 年 9 月的**历史调研与实验证据归档**，集中保存 DeepGen 适配性评估与「第三阶段」人物姿态图像编辑调研/显存预算文档。作为早期路线存档，它记录了项目在 Qwen-Edit 优先路线确立前的选型思考，引用时须注意其已被后续演进覆盖。

## 里面有什么

| 文件 | 内容 |
|---|---|
| `DeepGen人物姿态图像编辑适配性评估.docx` | DeepGen 人物姿态图像编辑**适配性评估正文**（52 KB Word，含渲染样本与结论） |
| `第三阶段图像编辑调研与显存预算.md` | 2026-09-08 的「第三阶段」调研：依据任务书与方案材料核算论文/官方仓/模型配置/权重头；明确 6B 限制针对生图主干、设备 RTX 6000D 84GB。补查指出 DeepGen/Klein 采用 Qwen 编码器≠Qwen-Image-Edit 压缩学生，尚未找到同时满足「源自 Qwen-Image-Edit、主干≤6B、公开权重、具编辑能力」的现成模型 |
| `audit_checkpoint_headers.py` | 检查点头审计脚本（docstring：仅用公开 HF safetensors 头统计张量元素数，不 import torch、不执行模型） |
| `checkpoint_audit.json` | 仓库级检查点审计结果：顶层键 `accessed_at`/`method`/`caveats`/`repositories`/`errors`，按仓库记录张量元素数与 bf16 等效 GiB 预算 |

## 与上下层的关系

- 同源主题渲染：`research/deepgen_assessment/`（DeepGen 纯基座渲染样本）。
- 选型被取代：本目录的 DeepGen 主路线推荐在 2026-09-08 补查后**不再作为最终选型**，后续以 Qwen-Edit 家族优先，详见 `research/qwen21_dual_upgrade_20261011/`。
- 显存预算与设备约束（RTX 6000D 84GB）仍对当前训练线（如 1024 原生画幅、qwen21_full）有参考价值。

## 注意与坑

- **明确为历史归档**：其中的 DeepGen 选型结论、6B 预算核算均基于 2026-09 的任务书与当时公开权重，部分口径（如「48GB/80GB 预算仅适用原训练配置」）已被后续架构改动推翻，勿当现状。
- `checkpoint_audit.json` 顶部 `repositories` 是**按公开 HF 头统计**的只读审计，不含本地私有权重；`errors` 字段记录哪些仓头无法读取，引用预算前先看 `errors` 是否为空。
- docx 与 md 的渲染图不在本目录，DeepGen 渲染样本见 `research/deepgen_assessment/`。
