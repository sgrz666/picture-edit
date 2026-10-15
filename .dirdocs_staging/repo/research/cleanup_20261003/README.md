# research/cleanup_20261003/ —— 2026-10-03 目录大清理的现场快照与验收证据链

> 工作区路径：`/home/shangguanrz/project/pic-edit/research/cleanup_20261003`　|　本级：叶子目录，23 个实证文件　|　体积：189 KB（du 实测）　|　本轮清理释放根分区 278.6 GB（`apply.log` 末行 `root_freed_bytes=278574833664`）

## 一、这一层在干什么

记录 2026-10-03 那次"把根目录脚本迁进 `scripts/`、把散落工具归档、给 `/data` 资产建软链、退役旧副本"的大清理**全过程与验收结论**。它不参与训练，但它是全仓库路径的"权威变更台账"：想知道某个旧路径为什么没了、现在指向哪里，先查这里的 `relocation_map.csv` 和 `navigation_operations.json`。

## 二、层级地图

叶子目录。23 个文件按流水线阶段分四类：预演（dry_run）、执行（apply/relocation）、验收（*verification*、`final_*`）、辅助工具（2 个 `.py`）。

## 三、关键文件与实证内容

- `relocation_map.csv`（176 行）：迁移总账。按 kind 列计数：`remote_script_moves=62`、`remote_script_aliases=31`、`home_moved=62`、`remote_asset=7`、`home_runner_links=4`、`remote_navigation=4`、`remote_evidence=2`、`local_evidence_tree=3`。每行 source→target→backup 三列，备份统一落在 `archive/cleanup_20261003/before/`。
- `navigation_operations.json`：**4 条关键软链的去向**——`reports/qwen21 → nihaochen/docs`、`models/Qwen-Image-Edit-20B → model-qwen-edit`、`caches/qwen21_hi4d_controls → /data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls`、`reports/legacy_demo_202609 → results/legacy_demo_202609`。文档备份在 `archive/cleanup_20261003/before_docs`。
- `final_verification.json`（17 KB，`passed=true`）：结构验收。moved_entries=62、classified_aliases=31、home_tools_archived=62、retired_holds_confirmed=76、全量资产哈希 `full_asset_files_hashed=4764`、`source_bindings_verified=252`；并如实记录了清理后 5 个又改动的源码文件（`src/pose_control/qwen21_full/*` 等）。
- `home_classification.json`：HOME 下 66 个工具的分类，`scratch=60 / run=4 / historical_destructive=2`，每项带 sha256 与 `referenced_by`。
- `data_path_verification.json`：8 份数据清单（3DPW `*_pairs.csv`、Hi4D_pilot_v1 manifest、iPER `samples.jsonl`、TikTok `candidate_pool_geometry_valid.jsonl`）；sample_path_checks=750、unique_paths=577、硬链资产计数 33,315。
- `checkpoint_load_verification.json`：从 `experiments/iper_v6_comparative_3000/checkpoint_last.pt` 实际加载成功，1,588,989,994 B、1,413 张量、5 个键，device=cpu。
- `dry_run_result.json` / `dry_run.log` / `apply.log` / `apply_job.json`：预演与执行日志，pid 2159017，命令 `finalize_migration.py --apply`（脚本本体在 `archive/cleanup_20261003/tooling/`）。
- `independent_review_final.md`：独立复查结论——无 Critical/Important，验收可接受；唯一 Minor 为 HOME 目标路径唯一性，已补齐。

## 四、注意事项 / 调用路径

- **本目录不可删除**：它是软链台账。若根分区紧张需冷归档，按规范先同步到 `/data/shangguanrz/pic-edit-archive/` 再建软链，**切勿 `rm -rf`**，否则 `navigation_operations.json` 记录的 4 条软链语义将失去出处。
- 清理中退役的 76 个 `*.cleanup-hold-20261003` 为固定删除范围，`RETIRED` 行见 `apply.log` 尾部。
- 复现/核验入口：`run_full_check.py`、`archive_home_tools.py`；完整资产哈希由后台 job（pid 2179938）跑完，结果落 `full_verification_result.json`。
