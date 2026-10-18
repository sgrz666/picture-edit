# research/cleanup_20261003/ —— 2026-10-03 目录大清理现场快照

本目录是 2026-10-03 项目目录大清理的**只读证据存档**：保存了迁移对照表、备份清单、各阶段验证 JSON 与日志，以及一份独立审查最终结论。它证明清理「未修改业务代码/数据、未 git 提交或推送、所有操作为同盘 mv / 删除可再生文件」。对应 `docs/目录清理执行记录_20261003.md` 的执行叙述。Reviewer `/root/cleanup_review` 最终判定无 Critical/Important 问题，验收可接受。

## 里面有什么

关键文件（共 23 个，含 14 JSON / 4 日志 / 2 脚本 / 1 md / 1 csv / 1 txt）：

| 文件 | 内容 |
|---|---|
| `independent_review_final.md` | 独立清理审查最终结论：无 Critical/Important，唯一非阻塞 Minor（HOME 目标路径唯一性）已补齐 |
| `relocation_map.csv` | 迁移对照总表（33 KB，表头 `kind,source,target,backup`），逐条记录移动/备份去向 |
| `final_verification.json` | 完整验收：`passed`、`moved_entries`、`classified_aliases`、`home_tools_archived`、`home_runner_aliases`、`navigation_links`、`asset_files_hashed`、`retired_holds_confirmed` |
| `home_classification.json` | HOME 旧工具分类清单（397 条：name/sha256/referenced_by/group） |
| `data_path_verification.json` | 数据路径与清单校验（manifests、hardlink 资产、problems、manifest_sha256） |
| `deployed_documents_sha256.json` / `document_verification.json` | 已部署文档 SHA256 与校验（PROJECT_STRUCTURE、清理记录、scripts/reports/add README、asset_registry） |
| `archive_home_tools.py` | 归档脚本（docstring：逐字节归档 HOME 未变旧工具，不改字节） |
| `run_full_check.py` | 完整校验启动脚本（pathlib/subprocess） |
| `apply.log` / `dry_run.log` / `full_verification.log` / `full_job_launcher.log` | 各阶段操作日志（runbook 阶段为 0 字节空日志） |
| `dry_run_result.json` / `full_verification_result.json` / `checkpoint_load_verification.json` / `structure_verification.json` 等 | 各阶段机器可读验收结果 |
| `remote_git_status_after.txt` | 清理后远端 git 状态文本（确认未提交/未推送） |

## 与上下层的关系

- 执行记录：`docs/目录清理执行记录_20261003.md`；后续二轮：`docs/目录清理执行记录_20261009.md`、计划 `docs/superpowers/plans/2026-10-03-project-organization.md`。
- 产物归位：`scripts/`（分类入口）、`archive/home_tools_202609/`（HOME 旧工具）、`reports/`（本地报告）、`experiments/tiktok_legacy_evidence_202609/`（TikTok 历史）。

## 注意与坑

- **本目录是证据存档，不是操作手册**：不要据此反向执行删除或重新迁移；真实清理动作已结束，复现前先读 `docs/目录清理执行记录_20261003.md`。
- `full_job_launcher.log` 为空（0 字节），属正常——完整校验以 `full_verification_result.json` 的 `passed` 为准。
- `relocation_map.csv` 含 `backup` 列，备份实体在归档目录，清理未破坏原始字节（脚本仅做同盘 mv / 删可再生文件）。
