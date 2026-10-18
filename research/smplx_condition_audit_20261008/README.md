# research/smplx_condition_audit_20261008/ —— SMPL-X 条件输入质量审计

本目录是 2026-10-08 对 **SMPL-X 身体条件张量**质量的一次专项审计：核查由 `scripts/build_tiktok_v6_conditions.py` 生成、供 `src/data/tiktok_dataset.py` 读取的身体控制条件（betas / 6D 根旋转 / 平移 / 相机等）是否与渲染输入、全局 schema 一致。它是技术线 1（DeepGen + Unified SMPL-X Adapter）训练数据正确性的上游把关证据，整包仅一份 JSON。

## 里面有什么

- `body_inputs.json` —— 审计结果，顶层键：`audit_date`、`repo`、`baseline`、`method`、`code_sha256`、`scripts/build_tiktok_v6_conditions.py`、`src/data/tiktok_dataset.py`、`global_schema`、`betas`、`root_rotation_6d`、`translation`、`camera`、`samples`、`manifest`。其中 `code_sha256` 锁定被审计脚本版本，`global_schema` 描述条件张量全局结构，`samples`/`manifest` 给出抽样配对与清单路径，`camera` 记录相机参数取法。

## 与上下层的关系

- 被审计代码：主树 `scripts/build_tiktok_v6_conditions.py` 与 `src/data/tiktok_dataset.py`（TikTok v6 条件构建链路）。
- 关联调研：`research/tiktok_training_20261003/`（TikTok 接入与训练方案）、`docs/tiktok_qwen21_full_training.md`。
- 条件定义源头：`docs/adapter_v6_architecture.md`（SMPL-X 条件注入管线）。

## 注意与坑

- 本目录**只有一份 JSON、无渲染图或日志**：它是条件张量结构的静态核查，不代表端到端生成质量；要看法本身效果需去 `research/bfs_transfer_audit_20261004/` 或 `docs/experiments/`。
- 引用结论前先核对 `code_sha256` 是否仍等于当前主树脚本——条件构建逻辑若改动，本审计即过期。
- `betas`/`root_rotation_6d` 等字段是 SMPL-X 参数而非像素，跨目录比较时别和 `bfs_transfer_audit` 里的 PCK/身份均值混为一谈。
