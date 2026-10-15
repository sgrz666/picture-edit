# 当前目录与工作线（2026-10-03）

远端运行树：`ssh sg` → `/home/shangguanrz/project/pic-edit`。本地成果树：`D:\codeplus\pictureedit`。两边有各自未提交改动；本次整理不执行 pull、push、reset 或全量同步。

## 工作线与边界

| 路线 | 核心实现 | 入口 | 当前证据与限制 |
|---|---|---|---|
| DeepGen V6/V6.5/V6.6 | `src/pose_control/v6/`，独立 Face/Hand | `scripts/train/train_iper_v6_native.py`、`train_iper_v65_face.py`、`train_iper_v66_hand.py` | 已有 iPER 训练与横评；Hand 架构存在仍需生成质量验收 |
| Qwen 2.1 简化控制 | `src/pose_control/qwen21/` | `scripts/train/train_iper_qwen21_body.py`、`train_iper_qwen21_face.py`、`train_hi4d_qwen21_dual.py` | Body/Face 的 512/1024 实验及 Hi4D 单例；单例不代表双人全量验收 |
| Qwen 完整迁移 | `src/pose_control/qwen21_full/` | `scripts/train/train_iper_qwen21_full.py`、`scripts/eval/eval_iper_qwen21_full.py` | 继承工程、单例及 Face 修复；10 月 3 日继续 TikTok 接入和 Face 多样本评测，源码与运行目录保留 |
| 数据工程 | `src/data/`、`add/datasets/`、`add/tools/` | `scripts/data/`，`add/tools/` | iPER、TikTok v3、Hi4D、3DPW、HumanVid；不同 manifest/条件版本并存 |

旧 CNN `src/adapter/` 与三条控制架构分别保留。Full 内的 47 份 V6 核心副本是迁移隔离的一部分。检查点不能跨架构互换。

## 目录职责

| 目录 | 放什么 | 使用方式 |
|---|---|---|
| `src/` | 可复用模块、数据读取、模型接口 | 由入口导入；本次不改核心实现 |
| `scripts/train/`、`scripts/eval/` | 正式训练与评测入口 | 当前训练源码仍在根目录，分类入口指向原文件，旧命令继续可用 |
| `scripts/data/` | 配对、筛选、划分、预计算 | 构建脚本可能直接写数据，先看脚本参数，不用批量执行 `--help` 验证固定流程脚本 |
| `scripts/audit/` | 契约、质量、权重、兼容性核验 | 与正式训练区分 |
| `scripts/reports/` | 图表、报告、面板生成 | 生成物进入 reports/results，源码留 scripts |
| `scripts/weights/`、`scripts/run/` | 权重转换与编排 | 旧 scripts 路径保留兼容链接 |
| `tests/`、`configs/` | 回归测试与显式资产登记 | `configs/asset_registry.json` 是路径索引，不改变训练默认配置 |
| `docs/` | 当前规范、runbook、带日期计划 | 本文作当前导航；历史计划保留原日期和结论 |
| `reports/` | Word/HTML 成果、架构图、评价材料 | 本地原 reportdoc；远端 qwen21 链接到 nihaochen/docs |
| `research/` | 调研与审计证据 | 不当作正式运行入口 |
| `experiments/` | 按 run 保存权重、配置、日志、输出 | 每个 run 保持完整上下文；旧 run 允许链接到冷存储 |
| `results/` | 精选展示结果 | `legacy_demo_202609/` 保留早期演示包的内部结构 |
| `data/`、`inputs/` | 小样本与 fixture | 不与大数据原料混放 |
| `datasets/`、`models/`、`caches/` | 大资产的项目入口 | 实体可在 /data，按模型/分辨率/条件版本隔离缓存 |
| `add/` | 独立数据工程工作区 | 不整体改名；自有工具、环境、上游实现和中间资产职责见其 README |
| `vram_lab/`、`deepgen/`、`third_party/` | 实验包、上游实现与许可证 | 保持模块名、上游 Git 与许可证 |
| `archive/` | 历史工具/证据、整理回退记录 | 大件实体在 /data；小文件迁移前副本在 cleanup_20261003 |

远端 HOME 下 62 份旧诊断与临时脚本已归入 `archive/home_tools_202609/`，HOME 原位置保留链接。四个正式 runner 保留 HOME 实体，并提供 `scripts/run/home/` 分类入口。归档旧脚本仍包含历史固定路径，仅供查阅或明确知道用途时运行。

兼容软链接是本轮过渡措施，目录中出现同名链接不代表又复制了一份权重或源码。历史文档中的旧命令和缓存路径仍可解析。

## 大资产位置

冷归档根：`/data/shangguanrz/pic-edit-archive/cleanup_20261003/`。

- 旧 20B 模型：`models/Qwen-Image-Edit-20B/`；原 `model-qwen-edit/` 保留链接。当前 Qwen 2.1 使用 `models/Qwen-Image-2.1/`，二者不同。
- 早期正式验收与 V6 阶段比较：冷归档根 `experiments/review_20260908_v2/` 和 `experiments/iper_v6_comparative_3000/`；项目原路径保留链接，阶段检查点和报告一起保存。
- 3DPW 三份原始 ZIP：冷归档根 `raw/3dpw/`；`add/` 原路径保留链接。
- Hi4D Qwen 控制缓存：`/data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls/`；nihaochen 原路径与项目 caches 入口均可访问。

七项大资产均已完成复制、内容及属性比对、入口验收和原实体撤除。逐文件 SHA-256、原/目标路径以 `archive/cleanup_20261003/assets.json` 为准，最后撤除和空间测量见 `finalization.json`。

共享 `experiments/iper_adapter_finetune_v1/condition_cache`、当前 Full/TikTok 检查点、环境、HumanVid/3DPW 中间资产、原始数据均保留。数据最终包的硬链接不意味着底层原料与 QC 已无用途。

## 查找与使用

```bash
ssh sg
cd /home/shangguanrz/project/pic-edit
ls scripts/{train,eval,data,audit,reports,weights,run}
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/train/train_iper_qwen21_full.py --help
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/eval/eval_iper_qwen21_full.py --help
python3 scripts/audit/verify_project_organization.py
```

最后一个命令只读检查整理日志、入口链接、受保护源码和资产路径，不训练、不改数据。完整内容哈希复查使用该命令的 `--full-assets`，会读取全部归档资产，耗时较长。

本地使用 `reports/` 浏览成果；旧 TikTok 图表位于 `experiments/tiktok_legacy_evidence_202609/one-piece test/`，`scripts/polish_figures.py` 与 `scripts/make_tiktok_overfit_comparisons.py` 已指向此处。历史清理报告移至 `archive/cleanup_history_202609/`。

整理的操作与回退方式见 `目录清理执行记录_20261003.md`。此前只读摸底报告 `项目进展与目录整理建议_20261003.md` 保留原始时点，不把它当作整理后的目录现状。
