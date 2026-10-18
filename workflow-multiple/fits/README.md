# fits：拟合臂（WP7）multi-HMR 产物

双人线拟合臂的 single-image dual-person 3D 拟合产物目录，对应 ../README.md 拟合臂（WP7）节。一期用 Hi4D 预计算 DWPose 向导、不做拟合；multi-HMR 拟合是二期度量参照，本目录是它的 **blessed 报告**（2026-10-13 23 时落盘）与每对 B 原图的拟合骨架。

## 里面有什么

- `<pair>.npz`（17 同名对 + 5 跨身份孪生，命名含 `cross_*`/`*_from_pair*`）——multi-HMR 对每对 B 原图拟合的 3D 人体；生成链读 `j2d_native`，按髖心 x 排 slot1/2，人数≠2 或缺髖报错。
- `fit_dual_report.json` —— 与 DWPose 向导、Hi4D SMPL 投影两套参照的误差汇总（顶层 `provenance`/`skipped`/`summary`/`results`）；结论 `VERDICT PASS`（门见 ../tests/test_fit_gate.py）。
- `fit_manifest.json` —— 拟合任务清单（`provenance` + `records`，17 条）。
- `fit_request.json` —— 本次请求（`project_root`/`out_dir`/`det_thresh`/`images`）。
- `fit_worker.log` —— worker 逐对 `person_count` 与 `reproject_median_px`。

## 与上下层的关系

- 输入：`../pairs/`、`../pairs_cross/` 的 `B_target.png`。
- 下游接入：`dual_fitting.py` / `run_fit_guide.sh` / `run_fit_after_batch.sh`。
- 门禁契约：[`../tests/test_fit_gate.py`](../tests/README.md)；共享包见 [`../../src/pose_edit/README.md`](../../src/pose_edit/README.md)。

## 注意与坑

- 10-14 的复核另落 `../fits_recheck/`（npz 逐位相同，仅 JSON/LOG 不同）。README 提的输出目录是 `fits_recheck2`，但磁盘上**没有** `fits_recheck2`，实际复核目录是 `fits_recheck/`。
- 髖距门最初拿 DWPose 当参照判 FAIL（0.120>0.10），已改为拿投影几何当参照（均值 0.021 PASS），DWPose 项降为 `DIAGNOSTIC`；读报告看 `hip_gap_error_norm_gt` 而非 `hip_gap_error_norm`。
- 三对 backhug 的 DWPose 向导本身错（两髖距报 ~4px，真值 ~72px），缠绕姿态下逐人 mask 框污染关键点——这类对以拟合 3D 为准，评测须同时报 `projected_gt_px`。
