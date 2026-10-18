# fits_recheck：拟合臂复核产物

拟合臂（WP7）复核产物目录，对应 ../README.md「门的参照口径修正」一节。是 2026-10-14 10 时按修正后的参照口径（投影几何而非 DWPose）重跑 `fit_dual_report.py` 的结果，用来把"髖距门的判定项"从 DWPose 换成投影几何。

## 里面有什么

文件清单与 `../fits/` 同名：

- `<pair>.npz`（17+5，与 `../fits/` 逐位相同，md5 一致）。
- `fit_dual_report.json`、`fit_manifest.json`、`fit_request.json`、`fit_worker.log`。

## 与上下层的关系

- 是 `../fits/` 的复核副本；复核目的见 ../README.md 第 259–263 行。
- 门禁契约：../tests/test_fit_gate.py（要求判定项含 `hip_gap_error_norm_gt`、DWPose 项只在 `DIAGNOSTIC`）。
- 共享包见 [`../../src/pose_edit/README.md`](../../src/pose_edit/README.md)；双人线总说明见 [`../README.md`](../README.md)。

## 注意与坑

- npz 与 `../fits/` 完全一致，差别只在 JSON/LOG：复核报告改用投影几何，`hip_gap_error_norm_gt` 均值 0.021 ≤ 0.10 → `VERDICT PASS`，DWPose 项降级 `DIAGNOSTIC` 并注明"分歧可能是向导错"。两个方向数值同时印出，不是放宽阈值。
- README 写作的复核输出目录是 `fits_recheck2`，但磁盘上**实际只有 `fits_recheck/`，没有 `fits_recheck2`**——以磁盘为准。
- 这是复核快照（10-14 10:20）。若要用今天代码重跑，命令见 ../README.md：`fit_dual_report.py --out <新目录>`，写新目录、不覆盖 blessed 的 `../fits/`。
