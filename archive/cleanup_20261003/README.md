# 2026-10-03 整理记录与恢复说明

本文件部署在远端 `/home/shangguanrz/project/pic-edit/archive/cleanup_20261003/README.md`。大资产实体位于 `/data/shangguanrz/`；这里保存小文件原稿和审查证据。

| 文件或目录 | 内容 |
|---|---|
| `organization.json` | 62 个脚本的原/现路径、修改前后 SHA-256、31 个分类入口、Git 基线 |
| `before/` | 搬迁前源码原稿及 Git 暂存索引备份 |
| `before_docs/` | README 更新前内容 |
| `small_operations.json` | 演示包、日志、三组历史脚本备份与缓存清理 |
| `assets.json` | 七项大资产逐文件哈希、属性、路径和撤除时间 |
| `finalization.json` | 撤除原副本的操作清单与前后磁盘测量 |
| `verification_pre_finalize.json` | 实际验收的源码绑定、11 个 CLI、133 份脚本 AST、246 项相关测试 |
| `safety_fixture_final.txt` | 七种迁移与恢复边界的审查用例结果 |
| `tooling/` | 当时的迁移、验收与收尾工具，含固定允许操作清单 |

日常只读复查：在项目根运行 `python3 scripts/audit/verify_project_organization.py`。完整归档复查加 `--full-assets`。其他收尾证据位于 `research/cleanup_20261003/`；HOME 归档操作位于 `archive/home_tools_202609/operations.json`。总执行记录见 `docs/目录清理执行记录_20261003.md`。

已完成的收尾不能再次作为新迁移执行。`tooling/finalize_migration.py` 默认只读检查，只有 `--apply` 才撤除允许清单中的旧实体；它拒绝覆盖已经完成的收尾日志。

## 需要恢复旧目录时

恢复前先查对应 JSON 的源路径、目标路径和哈希，并确认没有训练、评测或写入进程使用该对象。

小脚本原稿在 `before/`。先复制到临时目录核对 `before_sha256`，再检查原位置仍是本次创建的链接，分类实体也未被后续开发改变。满足这些条件才替换原位置；分类实体保留供比对。存在后续改动时应人工合并，不能直接用原稿覆盖。

HOME 工具字节未修改，可从 `archive/home_tools_202609/operations.json` 查来源。检查 HOME 原位置仍是对应链接且归档文件哈希一致后，才能移回。正式 runner 本来就在 HOME，不需要恢复。

大资产需反向复制到原分区的临时同级路径：使用 `rsync -aHAXS --numeric-ids`，预留足够空间，再按 `assets.json` 核对全部文件哈希及属性。验证完成且原位置仍是预期链接、没有并发访问后，才用已验证的实体替换链接；`/data` 完整副本保留。不能只删除链接，否则旧路径会不可用。

Git 暂存索引备份只用于历史取证。清理未修改暂存索引；不要为撤销目录整理而把旧 `.git/index` 写回，以免覆盖后来暂存的工作。本次没有提交、推送或重置工作树。
