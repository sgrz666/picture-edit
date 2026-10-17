# docs/ —— 项目设计规范、训练计划与演进记录层

`docs/` 是 pic-edit 的**设计文档与演进记录层**，收纳面向项目自身的架构说明、训练规范、对照实验结论、清理执行记录，以及 `superpowers/`（工程方法论：计划与规格）和 `experiments/`（实验专项说明与交接 runbook）。它记录的是「项目自己的决策与现状」，与 `research/`（带来料证据包的专题调研/审计）分工明确。远端访问：`ssh sg` → `/home/shangguanrz/project/pic-edit/docs`。

## 里面有什么

内容 Markdown 文件（不含本文）：

| 文件 | 主题 | 日期 |
|---|---|---|
| `Qwen2.1_项目框架总览_20261015.md` | 项目现行核心框架、各模块契约与端到端流水线总览 | 2026-10-15 |
| `Qwen21_双人Adapter适配性与实验调查报告_20261017.md` | 远端 `4.two man ` 实验深度调查，Adapter 架构适配性与 SOTA 双人方案对比调研 | 2026-10-17 |
| `空间清理审计与清单_20261017.md` | 2026-10-17 远端根盘空间释放（+126GB）与权重迁移对账审计 | 2026-10-17 |
| `workflow_multiple_设计_20261013.md` | `workflow-multiple/`（双人姿态编辑与交互迁移基线）的设计记录与阶段状态 | 2026-10-13 |
| `workflow_PE蒸馏到Qwen3VL任务分支_20261015.md` | workflow PE-I2I 蒸馏与小模型选型调研 | 2026-10-15 |
| `workflow_PE_I2I量化选型_GGUF社区权重评估_20261014.md` | GGUF 社区量化评估报告 | 2026-10-14 |
| `workflow_PE重写成本_量化与蒸馏调研_20261014.md` | 提示词重写低延迟优化与成本测算 | 2026-10-14 |
| `workflow_无GT定量评测调研_20261013.md` | 单人 workflow 无 GT 定量评测多轴指标体系 | 2026-10-13 |
| `workflow_大方向优化调研_20261014.md` | workflow 单人/双人方向层缺口与优化方案调研 | 2026-10-14 |
| `workflow审查与整改_20261012.md` | workflow 流水线规范审查与整改记录 | 2026-10-12 |
| `BFS_扩大样本对照_20261004.md` | BFS 扩大样本对照评估 | 2026-10-04 |
| `BFS_面部保持移用评估_20261004.md` | BFS 面部保持移用评估报告 | 2026-10-04 |
| `tiktok_qwen21_full_training.md` | TikTok 数据集及 Qwen 2.1 全量训练规范 | 2026-10-03 |
| `qwen21_hi4d_dual.md` | Qwen-Image-2.1 双人主体控制说明 | 2026-09-29 |

> **历史记录归档**：2026 年 9 月至 10 月初早期的阶段性清理对账、临时目录整理方案已统一归档至 [`archive/legacy_docs_202610/`](../archive/legacy_docs_202610/)。

子目录：

- `experiments/` —— 实验专项说明与交接 runbook（BFS Body+Face 联合实验的最终总结与历史 runbook）。
- `superpowers/` —— 工程方法论目录，下含 `plans/`（阶段计划）与 `specs/`（接口规格）。

## 与上下层的关系

- **技术线 1（DeepGen + Unified SMPL-X Adapter）**：`adapter_v6_architecture.md`、`iper_v6_training_plan.md` 对应 `src/pose_control/v6/`，训练入口在 `scripts/train/train_iper_v6_native.py`、`train_iper_v65_face.py`、`train_iper_v66_hand.py`。
- **技术线 2（Qwen-Image-2.1 迁移）**：`qwen21_hi4d_dual.md`、`tiktok_qwen21_full_training.md` 对应 `src/pose_control/qwen21/`（简化）与 `src/pose_control/qwen21_full/`（全量），入口见 `scripts/train/train_iper_qwen21_body.py`、`train_iper_qwen21_full.py`、`train_hi4d_qwen21_dual.py`。
- **BFS 落地路线**：`BFS_*` 两篇 + `experiments/` 是「Adapter 控姿 + BFS 768 ROI 精修面部」两阶段级联的落地记录，对接 `research/bfs_transfer_audit_20261004/`。
- 同级相关：`research/README.md`（调研/审计证据层）、`research/qwen21_dual_upgrade_20261011/`（双人升级设计稿与审计证据）。

## 注意与坑

- **`PROJECT_STRUCTURE_20261003.md` 是 2026-10-03 的旧结构快照**，已非现状（它未含后续 V6.5/V6.6 hand 分支、qwen21 双人升级等演进），新人勿当现状引用，以 `架构审查_20261012.md` 的实况统计为准。
- `架构审查_20261012.md` 内所有数字的唯一来源是文件内的统计快照（2026-10-12 00:2x 实况），带时间截面性，与后续改动不自动同步。
- `tiktok_qwen21_full_training.md` 明确：TikTok **测试集 1,156 对全程未用于训练 / Face 特征提取 / 采样**，只用于检查分组交叉，引用该数据线评测时勿把 test 当 train。
