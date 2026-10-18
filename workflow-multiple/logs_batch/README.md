# logs_batch：批处理日志与汇总

双人线批量跑（`batch_dual.sh` 链路）的全部日志与汇总落点，对应 ../README.md「批量进度」与各 gpu_gate/finalize 节。108 个文件，分三类。

## 里面有什么

- **批量驱动日志**：`batch.log`（第二轮 09:39 RESUMING 起，每行 START/END/SKIP 一格）、`batch_round1.log`（首轮）；`finalize.log`/`finalize2.out`/`finalize_round2.out`（统一重算评测）；`fit_after.log`/`fit_after_round2.out`（fit 孪生）；`fit.log`/`fit_run.log`/`fit_report.txt`（拟合链）；`gate.log`（gpu_gate 显存闸门，RESUME/HOLD/PAUSE）；`nohup.out`/`nohup_round2.out`、`report.txt`、`matrix_audit.txt`（覆盖判据，退出码 0=满覆盖）。
- **逐 run 日志**：`<pair>.<arm>.<framing>.{source_fit,b_canvas,with_b}.log`（每格几何/PE/生成/校验逐阶段）与 `fit.<pair>...fit.log`（fit 孪生）；`retry1/2/3.cross_backhug_*.log` 是首轮补跑重试。
- **探针/汇总 JSON**：`import_crop_probe.json`（裁剪口径探针）、`size_compliance.json`（出图尺寸合规）、`index.tsv`（批量进度索引，batch.log 头引用）。

## 与上下层的关系

- 上游：`../runs/`（逐格产物，日志里记绝对路径）、`../comparisons/`（成对结果）、`../fits/` 与 `../fits_recheck/`（拟合报告）。
- 驱动脚本都在 `../`（batch_dual.sh、watch_and_retry.sh、finalize_dual.sh、run_fit_after_batch.sh、gpu_gate.sh）。

## 注意与坑

- 看实时覆盖别数目录，用 `matrix_audit.txt`（判据=run.json complete 且有 eval_dual.json）；退出码非 0 表示还有空格。
- 显存紧张时看 `gate.log` 判断处于 RESUME/HOLD/PAUSE；闸门 TERM 掉批量循环但保护正在跑的 run，重启后已 ok 的 run 命中缓存被跳过。
- 逐 run 日志里 `person_count_ok=0` 在纠缠姿态下是检测器把相拥两人并成一框，不代表图错（见 ../README.md 人数修正）；读 `finalize.log` 的 `EVALUATE_DUAL_COMPLETE` 行取真实指标。
