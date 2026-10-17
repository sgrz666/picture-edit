# scripts/ —— 仓级跨实验脚本入口

`scripts/` 是 pic-edit 仓级的跨实验脚本入口，集中所有不隶属于单一流水线、需要跨 `experiments/` 反复调用的训练、评测、数据、审计、报告与权重工具。2026-10-11 重组后，稳定工具按职责拆进七个分类子目录，每个子目录各有独立、详尽的 README（见下方索引表）。本文件因此只做三件事：列清 `scripts/` 根还保留哪些**顶层脚本**、给出七个分类子目录的**索引**、并标注已**退役的文档生成器**与 `workflow/scripts/` 的分工边界。

## 顶层脚本（scripts/ 根）

根目录只保留少数“横跨分类、不适合塞进某一子类”的脚本；其余入口以**软链接**指回对应的分类子目录（软链接不复制源码，运行时 `Path(__file__).resolve()` 仍落到原文件）。

### 工具 / 巡检类（直接运行）

| 文件 | 一句话职责 |
|---|---|
| `inspect_deep_metrics.py` | 临时核对脚本：读取 `experiments/` 下各 run 的 `summary.json`，打印 qwen21_iper_body、qwen21_hi4d_dual 等评测的均值与指标采样 |
| `inspect_experiments.py` | 扫描 `experiments/` 根，逐目录清点子目录 / checkpoint / config / log / metric / 图像数，做实验清单巡检 |
| `parse_audit.py` | 读取 `experiments_audit.json`，逐实验打印子目录、checkpoint、配置样例与指标样例 |
| `dataset_notes.py` | 数据盘（`/data/shangguanrz`）各目录的**人工核定说明字典**；被退役的 `gen_folder_docs.py` 注入目录说明，本身不单独运行 |
| `build_tiktok_v6_conditions.py` | 为 TikTok 数据集生成 DeepGen V6.3 Native Adapter 条件（14 类部位图 PNG + SMPL-X 全局 26D 参数与 25 关键点） |
| `precompute_tiktok_face_features_v65.py` | 预计算 V6.5 TikTok 人脸身份 / 外观 / 几何特征，产出 `v65_face_features.pt`（严格遵循 V6.5 Face Cache Schema v1） |
| `prepare_tiktok_qwen21_face.py` | 为 full Qwen TikTok 训练准备 source-only 的 ArcFace / DINO 人脸缓存 |
| `prepare_tiktok_qwen21_training.py` | 审计严格 TikTok 划分（组 / 序列不重叠）并写出独立的合格训练 manifest |
| `run_tiktok_qwen21_full.sh` | TikTok Qwen full 训练的 bash 驱动，支持 `prepare / smoke / body / face-prepare / face / resume-body / resume-face / eval` 等模式 |

### 入口软链接（指向分类子目录）

根目录另含 8 个指向子目录脚本的软链接，方便从仓根直接调用分类入口，例如：

- `eval_qwen21_face_repair.py` → `eval/`
- `run_qwen21_face_single_sample.py` → `run/`
- `probe_qwen21_face_single_sample.py`、`verify_qwen21_face_repair.py`、`verify_qwen21_face_single_sample.py`、`verify_qwen21_full_resume.py` → `audit/`
- `finalize_qwen21_face_single_sample.py` → `run/`
- `write_qwen21_face_single_sample_report.py` → `reports/`

（`data/` 子目录里也有指向根层 `build_tiktok_v6_conditions.py`、`precompute_tiktok_face_features_v65.py`、`prepare_tiktok_qwen21_*.py` 的软链接，实质同一份源码。）

## 七个分类子目录（索引）

每个子目录都有独立 README，下表是一句话定位与链接：

| 目录 | 用途 | 子 README |
|---|---|---|
| `train/` | 训练入口：DeepGen V6 Native、Qwen21 Body/Face/Full、joint pilot、Hi4D 双人、TikTok sanity | [train/README.md](train/README.md) |
| `eval/` | 正式评测与阶段对照：iPER 四臂 / V6 / Qwen21 主干、BFS、Face、native-value、Hi4D 双人、joint pilot | [eval/README.md](eval/README.md) |
| `run/` | 实验编排与收尾：single-overfit、demo 姿态编辑、条件组合、complete / finalize 收尾 | [run/README.md](run/README.md) |
| `data/` | 数据构建、过滤、划分、预计算：TikTok V3 配对与条件、iPER / Qwen21 特征与 manifest | [data/README.md](data/README.md) |
| `audit/` | 数据 / 权重 / 兼容性检查与整理验收：视觉质量审计、权重兼容、项目组织复核 | [audit/README.md](audit/README.md) |
| `reports/` | 成果物化：Word 报告、图表、对照面板、结果摘要与可视化 | [reports/README.md](reports/README.md) |
| `weights/` | 权重继承与转换 | [weights/README.md](weights/README.md) |

## 退役的文档生成器（保留但不要再运行）

以下三个脚本是 10-12 / 10-13 已整体删除的「仓级文档生成层」产物，脚本文件保留在根层但**已退役**——不要再次运行它们去批量生成或覆盖 README：

| 文件 | 原职责（已停用） |
|---|---|
| `gen_folder_docs.py` | 为仓内每个非隐藏目录生成《目录说明.md》（基于 Python docstring / Markdown 首标题 / JSON 顶层键等实际读取内容）；不覆盖已有 README |
| `sync_folder_readmes.py` | 确保每个目录都有 README：若仅有《目录说明.md》而无 README，则从层级生成 README |
| `generate_master_readme.py` | 依据完整目录层级生成仓级权威 `README.md` 模板 |

它们依赖 `dataset_notes.py` 中的数据盘人工说明字典。当前各目录 README 由人工维护，生成层退出后这些脚本仅作历史残留保留。

## 与 workflow/scripts/ 的分工

- **`scripts/`**（本目录）：仓级、跨实验的**通用**脚本分类入口。训练 / 评测 / 数据 / 审计 / 报告 / 权重 / 编排都面向 `experiments/` 下任意 run，与具体流水线解耦。
- **`workflow/scripts/`**：`workflow/` 流水线**内部私有**的工具（如 `six-picture-create/`、流水线专属的诊断 / 生成 worker），服务于单一流水线的内部步骤，不进入 `scripts/` 的跨实验分类。

两者边界：凡是“被多个实验、多个流水线复用”的通用能力进 `scripts/`；只服务于 `workflow/` 某一流水线步骤的实现进 `workflow/scripts/`。

## 运行约定

- 多数脚本通过 `Path(__file__).resolve()` 解析到原文件，从分类入口执行不改变 ROOT、源码哈希或模块身份。
- 部分数据 / 旧报告脚本无通用 `--help`、运行时直接写输出，先检查脚本的 `main` / 参数，不要批量探测所有脚本。
- 本地与远端不是全量镜像；验收脚本默认只检查结构、链接、小文件哈希与资产是否存在，`--full-assets` 另读七项资产并核对清单、属性与内容哈希。
