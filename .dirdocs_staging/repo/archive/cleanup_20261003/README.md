# archive/cleanup_20261003/ —— 10-03 大清理批次的完整证据链（本层最重要的归档）

> 工作区路径：`/home/shangguanrz/project/pic-edit/archive/cleanup_20261003/`　|　本级：6 个子目录 / 11 个直属文件　|　子树体积：4.0 MB（`du -sh --si`，2026-10-12）
数据来源：本目录手写 `README.md`（2026-10-03，即"那份清理记录"）及 `organization.json`、`assets.json`、`finalization.json` 等清单，读取时间 2026-10-12。

## 一、这一层在干什么

记录 2026-10-03 执行的一次系统性目录整理：**62 个根目录散落脚本迁入 `scripts/` 分类目录、7 项大资产（合计约 169 GB）冷归档到 `/data/shangguanrz/` 并在原路径留软链、演示包与家目录工具分别归位**。它"删"的不是内容而是位置——所有实体要么改链、要么保留原稿；真正被撤除的是 76 个迁移中间态占位对象。判断这批东西要不要，以本目录清单 + 手写 README 的恢复流程为准。

## 二、层级地图

| 子目录 | 装什么 | 内容实证 |
|---|---|---|
| `before/` | 62 个迁移脚本中**修改过**的那部分之搬迁前源码原稿 | 30 个 `.py`（如 `audit_tiktok_v3_visual_quality_v3.py` 50 KB、`build_architecture_docx.py` 41 KB），1.1 MB |
| `before_docs/` | README 更新前的旧版本文档 | 根 `README.md` 旧稿（4,520 B，2026-09-17）+ `experiments/README.md` 旧稿 |
| `holds/`（含 `scripts/`） | 迁移时"新旧并存"阶段的旧实体暂存位 | **现已被 finalization 清空**（34 个 `.py` 在 `finalization.json.retired` 名单里，如 `holds/build_tiktok_v3_pairs.py`、`holds/scripts/smoke_adapter_v6.py`），只剩目录骨架 |
| `backup_holds/add/`（`datasets/`、`tools/`） | `add/` 下 7 个带时间戳备份脚本的暂存位 | **同样已清空**（`..._before_use_dwpose_render_20260930_090016.py` 等在 retired 名单） |
| `add_backups/group_1..3/` | 尚存的 3 个 `add/` 数据加载/可视化脚本备份实体 | 每组 1 个 `.py`（`visualize_dpw_single_pose_edit_dataset.py` 等），共 42 KB |
| `tooling/` | 当时执行迁移的 6 个工具脚本 | `migrate_assets.py`（copy→SHA-256→建链→保留源）、`finalize_migration.py`（默认只读，`--apply` 才撤旧，拒绝重复收尾）、`organize_remote.py`、`verify_runtime.py`、`audit_dry_run.py`、`test_migration_safety.py` |

## 三、关键文件与实证内容（本级 11 个直属文件）

| 文件 | 是什么 | 读出的真实结论 |
|---|---|---|
| `README.md`（手写） | 整理记录 + 恢复手册 | 恢复三分法：小脚本查 `before/`+`before_sha256`；大资产用 `rsync -aHAXS --numeric-ids` 从 `/data` 反向复制并按 `assets.json` 核对；禁止用 `git-index.before` 回写暂存索引 |
| `organization.json`（131 KB） | 62 个脚本迁移清单 | 字段 `moves[]`：source/target/backup/`before_sha256`/`after_sha256`/state（值如 `linked_with_original_retained`），62 条全部去向 `scripts/` 各分类；`aliases[]` 31 条兼容软链；Git 基线 head `b1a00c2a`，`index_before_sha256 == index_after_sha256`（暂存索引未动）；`protected_changed` 为空（`src/` 受护文件零改动） |
| `assets.json`（1.8 MB） | 7 项大资产逐文件哈希/属性/时间 | 全部 state=completed：Qwen-Image-Edit-20B 模型 57.7 GB/110 文件 → `/data/.../models/`；`experiments/review_20260908_v2` 27.3 GB/3,106 文件；`iper_v6_comparative_3000` 12.8 GB/156 文件；`qwen21_hi4d_controls` 66.3 GB/1,389 文件 → `/data/shangguanrz/pic-edit-caches/`；3DPW 三个 zip（`imageFiles` 4.6 GB、`sequenceFiles` 278 MB、`camera_tram` 150 MB）→ `/data/.../raw/3dpw/`。合计约 169 GB |
| `finalization.json`（17 KB） | 撤除原副本的操作清单 | `apply: true`；`retired` 76 项（7 个 `*.cleanup-hold-20261003` 标记 + `holds/`、`backup_holds/` 下 69 个旧实体）；根分区 used 1,697.6 GB → 1,419.0 GB，**净释放约 278.6 GB** |
| `small_operations.json`（9.4 KB） | 演示包/日志/备份组/缓存清理 | `moves` 含 `demooutput → results/legacy_demo_202609`（逐文件 sha256）、`inference.log → logs/legacy/`；另有 `backup_groups`、`cache_removed` |
| `verification_pre_finalize.json` / `_early.json` | 收尾前独立验收 | 源码绑定校验、11 个 CLI、133 份脚本 AST、246 项相关测试（手写 README 载明） |
| `entry_verification.json` / `pytest_verification.json` | 入口与 pytest 复核 | 迁移后各分类入口可运行、pytest 通过记录 |
| `safety_fixture_final.txt` | 迁移安全夹具结果 | `passed: true`，7 个用例：历史脚本/测试删除拒绝、目标篡改拒绝、开放写者拒绝、回滚冲突保双 inode 等 |
| `git-index.before`（385 KB） | 清理前 Git 暂存索引备份 | **仅历史取证用**，不得回写（会覆盖后来的暂存） |
| `.gitignore` | 内容 `*`，只放行 `README.md` | 上述清单全部不入库，需单独备份 |

## 四、注意事项 / 调用路径

- 实体去向：`/data/shangguanrz/pic-edit-archive/cleanup_20261003/{models,experiments,raw}`（du 实测 58 GB / 39 GB / 5.1 GB）与 `/data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls`。`experiments/review_20260908_v2`、`model-qwen-edit` 等原路径现为软链，指向这些实体。
- 总执行记录：`docs/目录清理执行记录_20261003.md`；收尾证据副本：`research/cleanup_20261003/`（含 `apply.log`、`dry_run.log` 等）；家目录分支：`archive/home_tools_202609/operations.json`。
- 已完成的收尾**不可当作新迁移重放**：`tooling/finalize_migration.py` 默认只读检查，且拒绝覆盖既有收尾日志。
- 恢复前置检查（手写 README 原文要求）：确认无训练/评测进程占用该对象、原位置仍是本次创建的链接、分类实体未被后续开发改变；有后续改动时人工合并，不得用原稿直接覆盖。
