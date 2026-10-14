# scripts/run/ —— 目录层级与功能说明

> 本文件由 `scripts/gen_folder_docs.py` 依据目录实际内容自动生成（生成时间：2026-10-11 16:23）；如需补充人工说明，请写在本目录 `README.md` 中（生成器不会覆盖 README）。

- **路径**：`scripts/run`　- **层级**：第 3 级　- **上级**：[../目录说明.md](scripts)

## 一、目录定位

全流程实验编排与一键运行脚本目录。

- **本级内容**：文件 7 个，76.8 KB
- **子树规模**：文件 7 个，逻辑体积 76.8 KB

## 二、层级目录

本子目录（1 个，其中符号链接 0 个）：

- [`home/`](home/目录说明.md) — 全流程实验编排与一键运行脚本目录。 下按内容自然分组形成的子目录。（子树 0 个文件、0 B）

## 三、文件清单

共 **7** 个文件（不含隐藏文件与自动生成的说明文件）。类型分布：Python 源码×6、PS1 文件×1

| 文件 | 类型 | 大小 | 内容摘要 |
|---|---|---|---|
| `finalize_qwen21_face_repair.py` | Python 源码 | 9.4 KB | 模块 docstring：Consolidate saved Face-repair checks and the manually inspected single case.；顶层定义 3 个（`read`, `sha`, `main`） |
| `finalize_qwen21_face_single_sample.py` | Python 源码 | 7.4 KB | 模块 docstring：Collect bounded-run results and verify the explicit single-case criteria.；顶层定义 4 个（`read`, `write`, `csv_write`, `main`） |
| `run_qwen21_condition_combinations.py` | Python 源码 | 2.4 KB | 模块 docstring：Run the two locked formal inference batches, with durable stage status.；顶层定义 1 个（`main`） |
| `run_qwen21_face_single_sample.py` | Python 源码 | 12.6 KB | 模块 docstring：Bounded single-pair diagnosis. Every GPU action uses the complete controller.；顶层定义 8 个（`sha`, `command`, `generate`, `Metrics`…） |
| `run_qwen21_joint_pilot.py` | Python 源码 | 27.0 KB | 模块 docstring：Run a bounded pilot phase. Inspect saved images before approving a mid-stage continuation.；顶层定义 9 个（`invoke`, `train`, `evaluate`, `_sha256`…） |
| `run_qwen21_weight_inheritance.py` | Python 源码 | 16.7 KB | 模块 docstring：Verify and generate the fixed 60-image inheritance comparison, without training.；顶层定义 4 个（`source_pair`, `verify_gpu`, `check_sample_boundaries`, `main`） |
| `sync_qwen21_joint_pilot.ps1` | PS1 文件 | 1.4 KB | PS1 文件 |

## 四、构成画像

- **直属类型分布**：Python 源码×6、PS1 文件×1
- **子树类型构成**：Python 源码×6、PS1 文件×1

## 五、导航

- 根目录总览：[`目录总览.md`](../../目录总览.md)
- 根 README：[README.md](../../README.md)
