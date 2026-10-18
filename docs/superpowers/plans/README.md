# docs/superpowers/plans/ —— 架构重构与特性落地的执行计划

`docs/superpowers/plans/` 收纳 pic-edit 各阶段**执行计划（plan）**：明确要做什么、按什么顺序做、对应哪份设计规格（specs）与哪段 `src/` 实现。本目录是 `docs/superpowers/` 的方法论层之一，与 `specs/`（接口契约）成对出现。

## 里面有什么（8 篇，按时间）

| 文件 | 主题 | 日期 |
|---|---|---|
| `2026-09-12-champ-deepgen-single-sample.md` | CHAMP–DeepGen 单样本集成实施计划：把 CHAMP 控制条件接到 DeepGen PoseAdapter 的单样本跑通路径 | 9-12 |
| `2026-09-12-deepgen-no-control-ablation.md` | DeepGen 无动作约束消融实验实施计划：剥离姿态控制检验基座能力下限 | 9-12 |
| `2026-09-15-adapter-reasoning-v1.md` | Adapter 内部推理 V1 实现记录（短文，仅 730 B） | 9-15 |
| `2026-09-15-smplx-condition-injection-v1.md` | SMPL-X 条件注入 V1 实施计划：几何/交互条件如何进入 adapter | 9-15 |
| `2026-09-16-deepgen-dynamic-control-interface-v1.md` | DeepGen V6.3 动态控制接口 V1 实施计划：逐去噪步的动态双分支接口 | 9-16 |
| `2026-10-03-project-organization.md` | 项目清理归类执行计划：远端脚本/HOME 工具/本地报告的分流入 `scripts/`、`archive/`、`reports/` | 10-03 |
| `2026-10-04-bfs-expanded-comparison.md` | BFS 扩大样本对照实施计划（30 对 / 8 人 / 10 衣着组） | 10-04 |
| `2026-10-08-bfs-body-face-joint-pilot.md` | BFS Body/Face 联合 pilot 执行计划：原生画幅下 Body+Face 联合训练与冻结底座对照 | 10-08 |


## 与上下层的关系

- 与 `../specs/` 成对：`2026-09-12-champ-deepgen-single-sample.md` ↔ `../specs/2026-09-12-champ-deepgen-single-sample-design.md`；BFS pilot 同理对应 `../specs/2026-10-08-bfs-body-face-joint-pilot-design.md`。
- 落地产物：`docs/BFS_扩大样本对照_20261004.md`、`docs/目录清理执行记录_20261003.md`、`docs/experiments/`。
- 实现层：`src/pose_control/v6/`（V6.3 动态控制接口）、`src/pose_control/qwen21_full/`（BFS pilot）。

## 注意与坑

- 2026-09 的 CHAMP/DeepGen 单样本与 V6.3 动态控制接口属**早期路线**，部分已被 V6.5/V6.6 分支和 Qwen 2.1 迁移线覆盖；作为历史依据引用即可，不要当成当前默认方案。
- `2026-09-15-adapter-reasoning-v1.md` 篇幅极短（仅实现记录），不要期待其中含完整设计——设计在 `../specs/`。
