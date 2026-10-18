# logs/ —— 运行日志目录

logs/ 存放 workflow 每次 `batch.sh` 全测试集批处理、以及 `run.sh` 单对运行的终端日志副本。单对失败不中断批处理，整批跑完后再在日志尾部汇总每对状态。

## 里面有什么（5 个，均为 2026-10-12~13 产出）

按「日期 + 主题」命名，便于排障时翻：

- `batch_testset_768_20261012.log`：全测试集 768 批处理（7 对）。尾部有 `pairs: 7 known, 7 with a run` 汇总；可见 `medium_01` 失败、`testset_hard05` abandoned(legacy) 但批处理未中断。
- `hard05_manual_20261012.log`：hard_05 手工提示词（`--prompt-file`）单对运行，尾部 `MANUAL_RUN_DONE` + `EVALUATE_COMPLETE body_pck@0.2=0.667`。
- `hard05_torso_20261012.log`：hard_05 躯干定标（`--framing source_torso`）单对，尾部 `TORSO_RUN_DONE` + `body_pck@0.2=0.389`。
- `subject_largest_20261012.log`：`--subject largest` 复跑（hard_01、medium_01 多人图选主体），尾部 `SUBJECT_RERUNS_DONE`。
- `wp1_accept_20261013.log`：WP1 验收对照 run（hard_02 配对），尾部 `WP1_ACCEPT_DONE` + `body_pck@0.2=0.944`。

## 与上下层的关系

- 由 `../run.sh`、`../batch.sh` 写出：每阶段 `RUN <worker>` + `WORKFLOW_FINISHED`/`_FAILED`/`_DONE` 标记；阶段推理明细在对应 [`../runs/`](../runs/README.md)`/<name>/<stage>/run.log`。
- 与 `../runs/` 一一对应：日志里 `runs/<name>/...` 的路径，在该批处理之后这些 run 已被整体迁到 `../runs/test/` 下，按当前结构要看 `../runs/test/<name>/`。
- 横向指标汇总以 `../report.py` 为准，本目录只保留原始终端输出。

## 注意与坑

- 日志是「终端副本」，单对失败也照写不遗漏；排障先翻尾部状态行（failed / abandoned / validated），再向上找对应 `RUN <worker>` 的 traceback。
- 日志路径指向产出当时的 run 位置（顶层 `runs/<name>`）；2026-10-14 起的整批迁移后，run 实际在 `runs/test/` 下，路径字符串不再直接对应磁盘。
- 没有按 run 名建子目录，所有日志平铺；排障靠文件名里的日期+主题，别按内容猜。
