# archive/cleanup_20261008/ —— 二轮整理的撤链登记 + 冒烟 json 收容

> 工作区路径：`/home/shangguanrz/project/pic-edit/archive/cleanup_20261008/`　|　本级：1 个子目录 / 3 个直属文件　|　子树体积：91 KB（`du -sh --si`，2026-10-12）
数据来源：本目录手写 `README.md`（2026-10-09）与 `removed_links.json`，读取时间 2026-10-12。

## 一、这一层在干什么

登记 2026-10-09 执行的第二轮整理（方案日期 2026-10-08，目录名沿用方案标识）：把 `cleanup_20261003` 迁移后留在根目录的**兼容软链**退役掉——活动工作流已改用 `scripts/` 实体路径，旧链接不再必要。本目录因此保留两样东西：撤链登记表（可随时重建链接）与二轮期间收容的冒烟结果。

## 二、层级地图

| 项 | 装什么 | 内容实证 |
|---|---|---|
| `removed_links.json`（7.8 KB） | 撤除链接登记表，字段 `generated/reason/root/removed[]/kept[]` | removed **54** 条（source→target，如 `audit_hi4d_qwen21_prompt_leak.py → scripts/audit/...`），kept **8** 条 |
| `verify_project_organization.py.before`（14.2 KB，2026-10-09） | 验收脚本更新前的原版备份 | 与 `scripts/audit/verify_project_organization.py` 现版对照 |
| `outputs_smoke_202609/` | 原根目录 `outputs/` 的 9 个冒烟 json（2026-09） | 如 `adapter_v6_smoke_single.json`：status PASS，`adapter_trainable_parameters: 227,980,664`，residual 形状 `[1,32,1536]`；另含 `adapter_reasoner_smoke_{single,mixed,dual}.json`、`deepgen_control_interface_smoke_*.json` |

## 三、注意事项 / 恢复路径

- 需要旧根路径时按 `removed_links.json` 的映射逐条 `ln -s <target绝对路径> <source>` 即可复原，无实体丢失风险（目标全部在 `scripts/` 分类目录）。
- `outputs_smoke_202609/` 的 9 个 json 是 v6/reasoner 适配器与 DeepGen 控制接口的历史冒烟凭证，属 vram_lab 早期线，被 `cleanup_20261003` 的 AST/入口验收引用过；纯文本 KB 级，可长期保留。
- 执行记录：`docs/目录清理执行记录_20261009.md`。
