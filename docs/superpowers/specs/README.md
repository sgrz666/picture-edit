# docs/superpowers/specs/ —— 系统设计规格说明书

`docs/superpowers/specs/` 收纳 pic-edit 的**接口/设计规格（spec）**：约定模块边界、控制条件契约与设计约束，是 `plans/` 执行计划的上游契约。本目录是 `docs/superpowers/` 的方法论层之一，与 `plans/` 同名成对。

## 里面有什么（3 篇）

| 文件 | 主题 | 日期 |
|---|---|---|
| `2026-09-12-champ-deepgen-single-sample-design.md` | CHAMP 控制条件与 DeepGen PoseAdapter **单样本联合设计**：normal/depth/openpose/semantic 四图契约与注入点 | 9-12 |
| `2026-09-12-deepgen-no-control-ablation-design.md` | DeepGen **无动作约束消融实验设计**：去除姿态控制后基座能力的对照基线设定 | 9-12 |
| `2026-10-08-bfs-body-face-joint-pilot-design.md` | TikTok Body+Face 联合训练与 BFS 小规模对照**方案设计**：两阶段（Adapter 控姿 + BFS 768 ROI 精修面部）的接口与对照设置 | 10-08 |


## 与上下层的关系

- 与 `../plans/` 成对：`2026-09-12-champ-deepgen-single-sample-design.md` ↔ `../plans/2026-09-12-champ-deepgen-single-sample.md`；BFS pilot 设计 ↔ `../plans/2026-10-08-bfs-body-face-joint-pilot.md`。
- BFS 方案设计直接约束 `docs/experiments/2026-10-08-bfs-body-face-joint-*.md` 的实验配置，并对应 `research/bfs_transfer_audit_20261004/`（移用性审计）。
- 实现层：`src/pose_control/v6/`、`src/pose_control/qwen21_full/`。

## 注意与坑

- `2026-09-12-*` 两篇属**早期基线规格**，仅对当时单样本/消融实验有效；后续 V6.5/V6.6 分支已改变条件注入结构，引用需对照 `docs/adapter_v6_architecture.md` 与 `docs/架构审查_20261012.md`。
- 规格中的模块/CLI 名称若标注「计划接口」，代表撰写时尚不可运行，不要当成已实现入口。
