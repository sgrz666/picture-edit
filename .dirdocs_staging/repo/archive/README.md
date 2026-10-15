# archive/ —— 冷归档批次仓库：判断"这里的东西还要不要、能不能恢复"

> 工作区路径：`/home/shangguanrz/project/pic-edit/archive/`　|　本级：5 个子目录 / 0 个直属文件　|　子树规模：约 164 个文件、4.6 MB（`du -sh --si` 实测 4.6M，2026-10-12）
数据来源：本文件数字来自各批次 `README.md`、`*.json` 清单、`df`/`du` 实测，读取时间 2026-10-12。

## 一、这一层在干什么

本目录是全仓库的**冷归档层**。归档的直接动因是根分区长期高水位：`df -h /` 实测 `/dev/sda2 1.8T，已用 1.6T，余 76G，96%`（2026-10-12），项目要求大资产与不再活跃的小文件必须离开根分区。每个批次对应一次有记录的目录清理行动，批次内容不是"垃圾堆积"，而是**清理时的证据链**：清单 JSON（删了什么、从哪搬到哪、哈希）、搬迁前源码原稿、验收结果、当时用的迁移工具。上游是 `scripts/audit/verify_project_organization.py` 与各批次 README 所引用的执行记录（`docs/目录清理执行记录_20261003.md`、`docs/目录清理执行记录_20261009.md`），下游消费者是需要考证"某文件历史上在哪、为何消失、如何恢复"的人。

## 二、层级地图（批次命名规范：`cleanup_<YYYYMMDD>` = 清理批次；其余为 `<来源>_<年月/日期>`）

| 子目录 | 装什么 | 内容实证 | 与本级其他目录的关系 |
|---|---|---|---|
| `add_picture_edit_rescue_20260921/` | 2026-09-21 从 AutoDL 容器工作区（日志内路径 `/root/autodl-tmp/projects/picture-edit`）抢救回的 3 份冒烟日志 | 3 个 `.log`，共 5.3 KB | 单事件快照，与 cleanup 批次无关 |
| `cleanup_20261003/` | 10-03 大清理的证据链与工具（62 脚本迁移清单、7 项大资产清单、收尾删除记录） | 11 个直属文件 + 6 子目录，4.0 MB | 本层核心批次，`home_tools_202609` 是它同日执行的配套 |
| `cleanup_20261008/` | 10-09 二轮整理：撤除 54 个兼容软链的登记表 + 9 个 smoke json | `removed_links.json`（removed 54 / kept 8），91 KB | 对 `cleanup_20261003` 遗留链接的二轮收编 |
| `cleanup_20261011/` | 目录文档生成器 `gen_folder_docs.py` 在 2026-10-11 改版前的源码快照 | 仅 1 个文件 `gen_folder_docs.py.before-20261011`（65,971 B） | 与 `scripts/gen_folder_docs.py` 现版本对照 |
| `home_tools_202609/` | 家目录散落脚本归档：60 个一次性诊断 Python + 2 个历史 Shell | `operations.json` 登记 62 条，394 KB | `cleanup_20261003` 同日（10-03 22:46）执行的家目录分支 |

## 三、项目惯例（务必先读）

1. **冷归档后原路径留软链，切勿 `rm -rf`**。迁移是两阶段的：`cleanup_20261003/tooling/migrate_assets.py`（拷到 `/data/shangguanrz/` → SHA-256 逐文件核对 → 原路径改为软链）→ 独立验收通过后 `finalize_migration.py --apply` 才撤除旧实体。仓库根现存实例：`model-qwen-edit -> /data/shangguanrz/pic-edit-archive/cleanup_20261003/models/Qwen-Image-Edit-20B`、`experiments/review_20260908_v2 -> /data/.../cleanup_20261003/experiments/review_20260908_v2`、`demooutput -> results/legacy_demo_202609`（仓内移，同机制）。删软链 = 旧路径直接失效，恢复需按 `cleanup_20261003/README.md` 的 rsync 流程反向复制。
2. **归档实体大文件在 `/data`，本目录只放小文件原稿和证据**。`/data/shangguanrz/pic-edit-archive/cleanup_20261003/` 实测：experiments 39 GB、models 58 GB、raw 5.1 GB。
3. **归档清单纯磁盘存在**：`cleanup_20261003/` 与 `home_tools_202609/` 下有 `.gitignore`（内容 `*`，仅放行 `README.md`），清单 JSON 不入库，备份本目录等于备份恢复能力。
4. 复查入口：项目根运行 `python3 scripts/audit/verify_project_organization.py`（日常只读），加 `--full-assets` 做完整归档复查。

## 四、注意事项

- `cleanup_*` 目录的 README 是当次操作人**手写**的恢复说明（不是 `gen_folder_docs.py` 产物），信息优先级最高，改删前先读。
- 判断"还要不要"的次序：先看批次 README → 再看清单 JSON 的 `state`/`retired` 字段确认该对象当前是否已被撤除 → 最后抽样核对哈希。历史脚本的参数与数据时点不代表当前训练规范。
