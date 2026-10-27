# research/ —— 专题调研与审计证据层

`research/` 是 pic-edit 的**调研与审计证据层**：每个子目录对应一个专题，且都带来料佐证——JSON 结论、日志、脚本、第三方仓库副本或渲染样本，可独立核验。它与 `docs/` 的分工很明确：`docs/` 写「项目自己的设计规范、训练计划与清理记录」，`research/` 写「带着证据材料的专题调研/审计」（数据集选型、适配器审计、条件质量核查、双人升级调研等），每个子目录自成一个可审计的证据包。

## 里面有什么（9 个子目录）

| 子目录 | 专题 | 主要证据形态 |
|---|---|---|
| `bfs_transfer_audit_20261004/` | BFS（Body-Face-Skeleton）控制**移用性审计**：768 探针、LoRA 有无对照、CPU 兼容性核验 | 6 份 JSON（results/corrected_768_pose/existing_probe_summary…）+ 对比面板 PNG |
| `cleanup_20261003/` | 2026-10-03 目录大清理的**现场快照**：迁移对照、备份清单、独立审查结论 | 14 JSON + 4 日志 + `relocation_map.csv` + `independent_review_final.md` |
| `deepgen_assessment/` | DeepGen-1.0-diffusers **纯基座生成能力评估**：不同提示词/采样下的原始渲染 | `build_report.py`、`deepgen_pipeline_audited.py.txt` + `render*/` 渲染图与 PDF |
| `legacy_202609/` | 2026 年 9 月**历史调研与实验证据归档**（含 DeepGen 适配性评估 docx、第三阶段显存预算调研） | docx + `checkpoint_audit.json` + 审计脚本 |
| `qwen21_dual_upgrade_20261011/` | Qwen-Image-2.1 **双人分支升级调研**：设计稿、论文/代码审查、第三方仓库副本、审计证据 | 2 设计稿 md + 7 源码 + `evidence.json`/`readiness.json` + `repositories/`（8 个第三方仓） |
| `qwen21_dual_prompt_pilot_20261013/` | 双人 Qwen-Image-2.1 **提示词 pilot 调研证据包**（对应 `experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1`）：设计/数据/训练/交付快照/人工核验 | `setup.json`/`baseline_integrity.json`/`selection_lock.json` + `data/`（`dataset.json`/`train|val|test` 等）+ `training_*.json`+`training_evidence_final_v1.zip` + `delivery_snapshot_v1/v2.json`+`.zip` + `test_evidence_final_v1.zip` + `implementation/` 代码镜像 + `verify_delivery_from_manual.py` 核验脚本 |
| `smplx_condition_audit_20261008/` | SMPL-X **条件输入质量审计**：身体条件张量与渲染输入核查 | 单份 `body_inputs.json` |
| `tiktok_training_20261003/` | TikTok 数据集**接入与训练方案调研**：基线代码快照、适配器增量对比、交付报告 | `REPORT.zh-CN.md` + `verification.json`/`readiness.json` + `comparison.png` |
| `hi4d_smplx_pilot_20261011/` | SMPL(-X) 网格与参数产物目录（人工正稿，请勿改动） | — |


## 与上下层的关系

- `research/` 的证据支撑 `docs/` 的决策：`bfs_transfer_audit_20261004/` ↔ `docs/BFS_*` 与 `docs/experiments/`；`tiktok_training_20261003/` ↔ `docs/tiktok_qwen21_full_training.md`；`qwen21_dual_upgrade_20261011/` ↔ `docs/qwen21_hi4d_dual.md` 与后续双人升级；`qwen21_dual_prompt_pilot_20261013/` ↔ `experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1`（其是 run 的影子证据包，`evaluation_process.json` 的 `--training-root` 指向该 run）。
- 实现落点：`src/pose_control/{v6,qwen21,qwen21_full}/`、`src/data/`、`src/pose_edit/`、`scripts/`。
- 同级相关：`docs/README.md`（设计规范层）、`docs/superpowers/`（计划与规格）。

## 注意与坑

- **别把 `research/` 当可执行代码库**：其中 `src/`、`scripts/` 副本是调研快照（如 `qwen21_dual_upgrade_20261011/implementation/`、`source_snapshot/`），真实运行代码在远端 `ssh sg:/home/shangguanrz/project/pic-edit` 主树，副本可能滞后；`qwen21_dual_prompt_pilot_20261013/implementation/` 同理只是镜像，真正执行在 `experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1` 与对应 `scripts/`。
- `cleanup_20261003/` 是清理操作的**只读证据存档**，不要据此反向执行删除/迁移。
- `legacy_202609/` 明确为历史归档，其中的 DeepGen 选型结论（2026-09）已被后续 Qwen-Edit 优先路线取代，引用时看 `research/qwen21_dual_upgrade_20261011/` 的当前状态。
- `qwen21_dual_prompt_pilot_20261013/` 的 `delivery_snapshot_v2` 显式 `supersedes` v1（仅背景分母措辞修正），引用交付物以 v2 为准；其 `*.zip` 均附同名 json 记录 `sha256` 与文件数，解包或引用前务必比对，别凭文件名当真。
- 各 `*.zip`（如 `audit_bundle.zip`、`source_snapshot.zip`）是快照压缩包，解包前先核对文内 `source_sha256`/`code_sha256`。
