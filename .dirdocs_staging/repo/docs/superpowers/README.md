# docs/superpowers/ —— superpowers 工作流的任务计划与设计档案
> 工作区路径：/home/shangguanrz/project/pic-edit/docs/superpowers　|　本级：2 个子目录（`plans/`、`specs/`）/ 0 份本级文档　|　另含生成物 README.md、目录说明.md（`ls -la` 2026-10-12 实测）

## 一、这一层在干什么
本仓库用 superpowers 技能工作流（brainstorming → writing-plans → executing-plans/subagent-driven-development）驱动大任务时留下的纸面过程：`specs/` 存设计（回答"为什么这样做、边界是什么"），`plans/` 存逐任务实施计划（回答"具体按什么顺序改哪些文件、每步怎么验证"）。它们与 `docs/` 根级结论文档的分工：这里的文件常带执行状态（勾选框/[x] 或"已停止"标注），是历史工作记录而非现行规范。

## 二、层级地图
| 子目录 | 装什么 | 关系 |
|---|---|---|
| `plans/` | 8 份带日期计划：V6.2/V6.3 三个架构里程碑、CHAMP/消融两件套、10 月三次治理与实验 | 时间线序列，多数已完成并勾选，1 份标注被用户叫停 |
| `specs/` | 3 份设计文档 | 与 plans 同名日期配对（champ、no-control-ablation、joint-pilot），plans 引用 specs 为"Approved design" |

## 三、注意事项
读这些文件时注意状态标注：`plans/2026-10-08-…-joint-pilot.md` 与 `specs/2026-10-08-…-design.md` 均自述"历史 512 方案、已停止"，现行结论在 `docs/experiments/2026-10-08-bfs-body-face-joint-handoff.md`。
