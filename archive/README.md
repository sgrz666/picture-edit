# archive/ —— 历史归档根目录（只读）

历史归档根目录，集中保存历次目录清理的快照、旧家目录工具封存，以及外部工作区（`add/`）的救援副本。大资产实体通常落在 `/data/shangguanrz/`，此处只保留小文件原稿、审查证据与索引 JSON。与 `results/` 一样，**archive/ 是只读证据区，删除前必须审计，不得随手清理**。

## 里面有什么

- **`cleanup_20261003/`** —— 2026-10-03 第一轮整理的完整快照：搬迁前源码原稿（`before/`）、待删暂存（`holds/`）、清理工具（`tooling/`）、`add/` 区备份（`add_backups/`、`backup_holds/`），以及 `organization.json`、`assets.json`、`finalization.json` 等审查证据。其顶层说明 [`cleanup_20261003/README.md`](cleanup_20261003/README.md) 是人工正稿，请勿改动。
- **`cleanup_20261008/`** —— 第二轮清理产出与冒烟结果暂存；其 `removed_links.json` 记录了撤除的旧位置兼容层软链清单（source→target 映射），可用于按需重建。顶层说明是人工正稿。
- **`cleanup_20261011/`** —— 仅一个文件的安全快照：目录文档生成器 `gen_folder_docs.py` 在 2026-10-11 重新生成前的版本。见 [cleanup_20261011/README.md](cleanup_20261011/README.md)。
- **`add_picture_edit_rescue_20260921/`** —— 2026-09-21 `add_picture_edit` 工作区救援快照与冒烟日志。见 [add_picture_edit_rescue_20260921/README.md](add_picture_edit_rescue_20260921/README.md)。
- **`legacy_docs_202610/`** —— 2026-10-17 归档的 14 份历史阶段性目录整理方案、执行记录与过渡性部署清单（见 [legacy_docs_202610/README.md](legacy_docs_202610/README.md)）。
- **`home_tools_202609/`** —— 2026 年 9 月家目录散落脚本归档（历史 Shell 与一次性诊断 Python），家目录以符号链接回指。其顶层说明是人工正稿，请勿改动。

## 与上下层的关系

- 这些归档对应 live 目录的「历史状态」：`before/` 对应 `scripts/` 迁移前的根级脚本原稿；`add_backups/`、`backup_holds/` 对应外部 `add/` 工作区；`home_tools_202609` 对应 `$HOME` 下的工具。
- 日常只读复查入口在 live 侧：`python3 scripts/audit/verify_project_organization.py`（加 `--full-assets` 做完整大资产复查）。
- 完整执行记录在 `docs/目录清理执行记录_20261003.md`，收尾证据另在 `research/cleanup_20261003/`。

## 注意与坑

- 严格只读。需要恢复旧实体时，先查对应 JSON 的源路径、目标路径与 SHA-256，确认没有训练/评测/写入进程占用，再按人工说明里的流程操作；不能用原稿直接覆盖已有后续改动。
- `cleanup_20261003/` 的 `before/git-index.before` 仅用于历史取证，**不要**为撤销整理而把它写回 `.git/index`。
- 注意：`removed_links.json` 实际位于 `cleanup_20261008/`，并非 `cleanup_20261003/`。
