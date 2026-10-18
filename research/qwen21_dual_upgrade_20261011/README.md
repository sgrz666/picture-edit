# research/qwen21_dual_upgrade_20261011/ —— Qwen-Image-2.1 双人分支升级调研与审计证据包

本目录是 2026-10-11 的**双人交互（dual）控制升级调研证据包**：围绕 Qwen-Image-2.1 把单人 adapter 升级为可同时控制 A/B 双人的分支（`dual_v8` 候选设计）。它不是运行代码库，而是「设计稿 + 论文/代码审查 + 第三方仓库调研副本 + 可复现审计」的合集——用于支撑后续实施决策。对应 `docs/qwen21_hi4d_dual.md`（双人主体控制说明）与 `docs/架构审查_20261012.md`。

## 里面有什么

顶层（15 文件）：

| 文件 | 内容 |
|---|---|
| `2026-10-11-qwen21-dual-v8-design.md` | 双人分支升级**设计稿**（32 KB）：版本 `dual_v8`，状态「已完成源码审查的设计，待用户审阅后实施」；模块/CLI 为计划接口暂不可运行 |
| `Qwen21_双人Adapter论文与代码审查_20261011.md` | 双人 Adapter 的**论文、代码与接口审查**（24 KB） |
| `evidence.json` | 本地源码审计证据（32 KB）：`scope`/`audit_date`/`remote_project`/`remote_git_head_at_snapshot`/`transformer_config`/`vae_config`/`full_injection_groups`/`legacy_lite_injection_indices`/`videox_control_indices`/`repositories`/`cosh_files_identical_to_intergen`/`checks`/`source_sha256` |
| `readiness.json` | 就绪度（14 KB）：`scope`/`date`/`checkpoints`/`cuda`/`smplx_tests`/`source_sha256` |
| `audit_evidence.py` | 复现本地源码审计的脚本（docstring：不 import/运行外部模型） |
| `body_adapter.py` `conditions.py` `control_core.py` `data_runtime.py` `pipeline_qwenimage21.py` `transformer_qwenimage21.py` | 升级涉及的 adapter/条件/控制核/数据运行时/管线/transformer 源码草稿（含 `Qwen21BodyAdapter`、`ConditionBundle`、`BranchControlResiduals`、`QwenFullDataset`、`QwenImage21TemporalTimesteps` 等类） |
| `audit_bundle.zip` `implementation_base.zip` `repo_metadata.zip` `source_snapshot.zip` | 审计包 / 实现基座 / 仓库元数据 / 源码快照压缩包（解包前核对 `source_sha256`） |

子目录：

- `implementation/`（288 文件）—— 实现快照：`scripts/{audit,data,eval,precompute,reports,run,train,weights}`、`src/{adapter,controls,data,evaluation,integration,pose_control}`、`tests/`。
- `repositories/`（8 个第三方开源仓库调研副本）—— `aigc-apps__VideoX-Fun`、`bytedance__XVerse`、`tr3e__InterGen`、`MS-Diffusion__MS-Diffusion`、`yisuanwang__DanceTog`、`AliManjotho__coshmdm-code`、`AnnLin0628__mpie-bench`、`PeopleComposer`（另含 `PeopleComposer_tree.json` 与 `README.md`）。
- `source_snapshot/`（116 文件）—— 远端源码快照。
- `pilot_controls_v1/`（13 文件）—— 小规模验证性实验（v1）。
- `pilot_controls_v2/` `pilot_controls_v3/` —— 后续小规模验证批次（v2/v3）。


## 与上下层的关系

- 设计落点：`src/pose_control/qwen21/`（简化双人）与 `src/pose_control/qwen21_full/`（全量迁移），双人入口 `scripts/train/train_hi4d_qwen21_dual.py`。
- 关联文档：`docs/qwen21_hi4d_dual.md`（双人主体控制说明）、`docs/架构审查_20261012.md`。
- 对比线：技术线 2（Qwen 2.1 迁移）相对技术线 1（DeepGen+Adapter）的 PSNR/PCK 对照见 `docs/` 训练横评与 `docs/experiments/`。

## 注意与坑

- **这是调研/审计快照，不是主树可运行代码**：`implementation/`、`source_snapshot/`、`*.py` 顶层草稿可能滞后于远端 `ssh sg:/home/shangguanrz/project/pic-edit` 主树；真要实施以主树 + `dual_v8-design.md` 为准，引用前核对 `evidence.json`/`readiness.json` 的 `source_sha256` 与 `remote_git_head_at_snapshot`。
- `repositories/` 下的 8 个第三方仓是**调研副本**（VideoX-Fun/XVerse/InterGen/MS-Diffusion/DanceTog/coshmdm/mpie-bench/PeopleComposer），用于接口与方案对照，非本项目依赖；改项目代码勿直接引用其中文件。
- `audit_evidence.py` 明确「不 import/运行外部模型」，只能复现源码层面的审计，不能复现生成结果。
- 各 `pilot_controls_v1/v2/v3` 是不同时间的小规模验证，结论口径可能不同，比较时别跨版本混读。
- 解任何 `*.zip` 前先校验文内 SHA256，避免用过期的快照覆盖主树。
