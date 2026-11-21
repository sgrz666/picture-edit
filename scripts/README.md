# scripts/ —— 仓级跨实验脚本入口

`scripts/` 是 pic-edit 仓级的跨实验脚本入口，集中所有不隶属于单一流水线、需要跨 `experiments/` 反复调用的训练、评测、数据预处理、审计、报告与权重工具。

---

## 一、分类子目录导航与职责总表

| 目录 | 核心职责 | 代表脚本 | 子 README |
|---|---|---|---|
| [`train/`](train/README.md) | 正式训练入口 | `run_qwen21_v5_fixedpair.py`（最新 V5 固定样本训练）<br>`train_iper_qwen21_full.py`<br>`train_hi4d_qwen21_dual.py`<br>`train_iper_v6_native.py` | [train/README.md](train/README.md) |
| [`eval/`](eval/README.md) | 正式量化评测 | `eval_iper_qwen21_full.py`（30 步全量采样与打分）<br>`eval_iper_four_arm.py`<br>`eval_hi4d_qwen21_dual.py` | [eval/README.md](eval/README.md) |
| [`run/`](run/README.md) | 推理编排与单例验证 | `run_qwen21_native_full_v5.py`（Native V5 推理入口）<br>`run_qwen21_face_single_sample.py`<br>`run_demo_pose_edit.py` | [run/README.md](run/README.md) |
| [`data/`](data/README.md) | 数据构建、划分与几何预计算 | `assemble_qwen21_fixedpair_v5.py`<br>`prepare_qwen21_fixedpair_v5.py`<br>`split_iper_train_val.py`<br>`prepare_qwen21_full_data.py` | [data/README.md](data/README.md) |
| [`audit/`](audit/README.md) | 架构、权重与训练一致性审计 | `summarize_qwen21_fixedpair_training.py`（300 步训练谱系核验）<br>`status_qwen21_fixedpair_v5.py`<br>`audit_qwen21_fixedpair_source.py`<br>`audit_qwen21_fixedpair_workspace_drift.py` | [audit/README.md](audit/README.md) |
| [`reports/`](reports/README.md) | 成果物化与报告批量构建 | `export_qwen21_fixedpair_report.py`（只读证据包导出）<br>`build_qwen21_fixedpair_visual_report.py`（单文件 HTML 报告）<br>`render_qwen21_fixedpair_report.cjs`（离线 PDF 导出） | [reports/README.md](reports/README.md) |
| [`weights/`](weights/README.md) | 权重转换与继承映射 | `inherit_qwen21_full_weights.py`<br>`convert_control_lora.py` | [weights/README.md](weights/README.md) |

---

## 二、最新重点工具：Native V5 固定样本工具链

2026-10-25 至 10-27 建立的全套 Native V5 固定样本审计与报告工具链：
1. **数据准备与配置**：`scripts/data/prepare_qwen21_fixedpair_v5.py` 与 `assemble_qwen21_fixedpair_v5.py`，负责锁定样本配准几何与张量缓存。
2. **训练执行**：`scripts/train/run_qwen21_v5_fixedpair.py`，支持单人（TikTok）与双人（Hi4D）300 步过程记录。
3. **审计与状态**：`scripts/audit/summarize_qwen21_fixedpair_training.py`，CPU 端核对 AdamW 状态连续性与 30 步过程图完整性。
4. **报告导出**：`scripts/reports/export_qwen21_fixedpair_report.py` 与 `build_qwen21_fixedpair_visual_report.py`，导出包含各阶段 ROI 的离线 HTML/PDF 成果。

---

## 三、与 workflow/scripts/ 的分工边界

- **`scripts/`**（本目录）：仓级、跨实验的**通用**工具，面向 `experiments/` 下任意 run，与特定流水线解耦。
- **`workflow/scripts/`**：`workflow/` 流水线**内部私有**的工序（如 `six-picture-create/`），服务于单一流水线步骤，不进入仓级通用分类。

---

## 四、2026-11-21 根目录清理归类

根目录散置的 57 个脚本已完成归类（完整映射见 `/data/shangguanrz/pic-edit-cleanup-20261120/scripts_moves.tsv` 与 `cleanup_manifest.json`）：

- **39 个 `git mv` 归入子目录**：audit 10 个（审计巡检与全仓 README 工具）、data 5 个（条件/提示词预计算与数据说明）、eval 7 个（TikTok 评测系列与评测面板）、run 2 个（`run_tiktok_sample_00001_inference.py`、`run_tiktok_qwen21_full.sh`）、reports 14 个（对比条图、Word 报告、可视化与同步）、weights 1 个（`download_models.py`）。另有 `code/qwen21/edit.py` 归入 `run/qwen21_edit.py`，`code/` 目录撤销。
- **14 个根目录副本删除**：8 个与子目录逐字节一致（`eval_qwen21_face_repair.py`、`finalize_qwen21_face_single_sample.py`、`probe_qwen21_face_single_sample.py`、`run_qwen21_face_single_sample.py`、`verify_qwen21_face_repair.py`、`verify_qwen21_face_single_sample.py`、`verify_qwen21_full_resume.py`、`write_qwen21_face_single_sample_report.py`）；6 个子目录版含 `parents[2]` 路径适配且功能更新（`precompute_iper_face_features_v65.py`、`precompute_qwen21_face_features.py`、`split_iper_train_val.py`、`smoke_adapter_reasoner.py`、`smoke_adapter_v6.py`、`smoke_deepgen_control_interface.py`）。一律以子目录为准。
- **根目录仅保留 4 个软链目标**：`build_tiktok_v6_conditions.py`、`precompute_tiktok_face_features_v65.py`、`prepare_tiktok_qwen21_face.py`、`prepare_tiktok_qwen21_training.py`——它们是 `data/` 内 4 个相对软链接的目标，按原架构「改原文件即可」保留在根。
- 同批修复 13 个文件里的旧路径引用（含 `run/run_qwen21_pei2i_study.py` 运行时拼接的 `scripts/precompute_qwen21_pei2i_prompts.py` → `scripts/data/...`），被改的 .py 全部通过 `py_compile`。
