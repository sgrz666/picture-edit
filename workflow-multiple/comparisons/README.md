# comparisons：成对比较结果

双人线批量评测的成对比较结果目录，由 `compare_dual.py` 产出（口径见 ../README.md「批量进度」与「②的成对读数」）。它是 `../runs/` 里 68 个 DWPose 向导格 + 17 个 fit 孪生 的**聚合汇总**，本身不存图，与 `runs/`（每格一个实例子目录）是汇总/实例关系。

## 里面有什么

三份 JSON，结构一致，顶层都是 `by` / `rows` / `missing` / `summary`：

- `compare_arm.json` —— `--by arm`：同一对上 `skeleton_only`（不给 B）与 `with_b`（给 B）两臂的成对差，left 恒为 skeleton_only。
- `compare_framing.json` —— `--by framing`：`b_canvas`（整对 letterbox 保 B 布局）与 `source_fit`（统一缩放贴两人尺度）两档定标的成对差，left 恒为 b_canvas。
- `compare_source.json` —— `--by source`：同一格 GT(DWPose) 向导孪生 vs multi-HMR 拟合向导孪生的成对差（n=17，left 恒为 dwpose）。

`rows` 是成对投票表（每对 left−right 差与判定）；`missing` 列只有一边跑成的对；`summary` 给分组胜/平/负与均值。

## 与上下层的关系

- 上游：`../runs/`、`../logs_batch/matrix_audit.txt`（覆盖判据）。
- 互补：`../report_dual.py` 给按臂均值，回答不了"同一对上另一臂/档更好吗"——这正是本目录职责。
- 权威口径：../README.md 第 168–210 行；设计依据 ../../docs/workflow_multiple_设计_20261013.md。

## 注意与坑

- 只取"同一对两边都 complete 且能打分"的 run；单侧跑成的对进 `missing`，不能当证据缺失。
- PCK/MPJPE/身份只在能打分者上比较，人数/换人覆盖全组；`rows.metrics.<m>.verdict` 为 `not measured` 即该臂产不出可计分两人图（with_b 常见）。
- 覆盖判据以 `matrix_audit.py` 为准（run.json complete 且有 eval_dual.json），光看 runs/ 会把无人能打分的 run 算成证据。
- 三份是 2026-10-14 11:48 finalize 统一重算后的快照，重跑批量后需重新生成。
