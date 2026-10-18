# docs/superpowers/ —— 工程方法论层：阶段计划与接口规格

`docs/superpowers/` 是 `docs/` 下的**工程方法论目录**，收纳面向实现的「计划（plans）」与「规格（specs）」两层文档。计划描述要做什么、按什么顺序做；规格描述模块/接口的契约与设计约束。两者的内容是 `docs/` 顶层那些结论性文档（如 BFS 对照、iper 训练计划）的上游依据，多数与 `src/pose_control/`、`scripts/train` 的具体实现一一对应。

## 里面有什么

子目录（本目录自身不含散落文件）：

- `plans/` —— **阶段计划**：架构重构与特性落地的执行计划，共 8 篇（2026-09 至 2026-10）。
- `specs/` —— **接口规格**：系统设计规格说明书，共 3 篇（CHAMP–DeepGen 单样本联合、DeepGen 无约束消融、BFS Body/Face 联合 pilot）。


## 与上下层的关系

- `plans/` 里的 `2026-10-03-project-organization.md` 对应 `docs/目录清理执行记录_20261003.md`；`2026-10-04-bfs-expanded-comparison.md` 与 `2026-10-08-bfs-body-face-joint-pilot.md` 对应 `docs/BFS_*` 两篇与 `docs/experiments/` 的 runbook。
- `specs/` 的 `2026-10-08-bfs-body-face-joint-pilot-design.md` 是 `docs/experiments/2026-10-08-bfs-body-face-joint-runbook.md` 的设计源头。
- 实现落点：`src/pose_control/{v6,qwen21,qwen21_full}/`、`src/adapter/`（旧 CNN，保留）、`scripts/train/`、`scripts/eval/`。
- 同级相关：`docs/README.md`（顶层设计文档索引）。

## 注意与坑

- 这里的 plan/spec 多为**历史决策快照**：2026-09 的 CHAMP/DeepGen 单样本、V6.3 动态控制接口等，部分已被后续 V6.5/V6.6 分支与 Qwen 2.1 迁移线取代，引用前先对照 `docs/架构审查_20261012.md` 确认是否仍生效。
- `plans/` 与 `specs/` 文件同名成对（如 `*-design.md` ↔ `*-plan.md`）：design 是契约，plan 是排期，不要混读。
