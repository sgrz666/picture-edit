# scripts/reports/ —— 目录层级与功能说明

> 本文件由 `scripts/gen_folder_docs.py` 依据目录实际内容自动生成（生成时间：2026-10-11 16:23）；如需补充人工说明，请写在本目录 `README.md` 中（生成器不会覆盖 README）。

- **路径**：`scripts/reports`　- **层级**：第 3 级　- **上级**：[../目录说明.md](scripts)

## 一、目录定位

实验报告汇总、图表美化与对比面板合成脚本目录。

- **本级内容**：文件 30 个，514.3 KB
- **子树规模**：文件 30 个，逻辑体积 514.3 KB

## 二、层级目录

本目录为叶子目录，不含子目录。

## 三、文件清单

共 **30** 个文件（不含隐藏文件与自动生成的说明文件）。类型分布：Python 源码×30

| 文件 | 类型 | 大小 | 内容摘要 |
|---|---|---|---|
| `build_20b_memory_docx.py` | Python 源码 | 17.2 KB | 模块 docstring：生成《Qwen-Image-Edit 20B 显存占用预算与单卡可行性分析简报》Word文档 专门分析 20B 模型静态权重、各分辨率与模式下的显存占用区间、单卡 84GB 可行性；顶层定义 10 个（`set_run_font`, `set_cell_background`, `set_cell_margins`, `set_table_borders`…） |
| `build_architecture_docx.py` | Python 源码 | 41.8 KB | 模块 docstring：生成《人物图像姿态编辑系统架构设计与核心技术方案》Word文档 涵盖：项目背景、5B DeepGen 1.0 模型架构、多模态连接器 SCB、 接触感知几何适配器 (Contact；顶层定义 13 个（`set_run_font`, `set_cell_background`, `set_cell_margins`, `set_table_borders`…） |
| `build_brief_architecture_docx.py` | Python 源码 | 19.1 KB | 模块 docstring：生成《人物图像姿态编辑项目技术架构与进展简报（科研同步版）》Word文档 专为课题组项目汇报、月报、双周同步设计： 内容精炼、结构直观、图文并茂，涵盖模型选型、系统架构、Adapt；顶层定义 12 个（`set_run_font`, `set_cell_background`, `set_cell_margins`, `set_table_borders`…） |
| `build_hi4d_native_binding_report.py` | Python 源码 | 14.1 KB | 模块 docstring：Embedded-image Word report for the fixed native Hi4D binding experiment.；顶层定义 6 个（`crop`, `panel`, `configure`, `table`…） |
| `build_iper1024_comparison_panels.py` | Python 源码 | 4.1 KB | 模块 docstring：Make readable iPER comparison sheets with face crops from saved RGB outputs.；顶层定义 4 个（`arguments`, `face_crop`, `draw_sheet`, `main`） |
| `build_latest_experiment_report_docx.py` | Python 源码 | 22.3 KB | 模块 docstring：生成《DeepGen 1.0 图像姿态编辑最新实验结果与显存分析简报》Word文档 基于 120 次实测运行、60 个画质复核、显存全矩阵采样与 100 次更新稳定性测试。 多图呈；顶层定义 12 个（`set_run_font`, `set_cell_background`, `set_cell_margins`, `set_table_borders`…） |
| `build_qwen21_bfs_rgb_report.py` | Python 源码 | 9.0 KB | 模块 docstring：Concise Word report for the fixed 1024 A-RGB + B-RGB BFS comparison.；顶层定义 1 个（`main`） |
| `build_qwen21_bfs_skeleton_report.py` | Python 源码 | 9.0 KB | 模块 docstring：Short, embedded-image Word report for the fixed three-arm BFS audit.；顶层定义 3 个（`value`, `table`, `main`） |
| `build_qwen21_combination_report.py` | Python 源码 | 17.1 KB | 模块 docstring：Revision of the Word value audit with the locked native combination grid.；顶层定义 7 个（`quality_table`, `shared_nodes`, `increment_table`, `factor_table`…） |
| `build_qwen21_face_panels.py` | Python 源码 | 3.9 KB | 模块 docstring：Create paired full-image and face-detail panels for a fixed Qwen21 Face evaluation.；顶层定义 4 个（`arguments`, `image_for`, `portrait`, `main`） |
| `build_qwen21_joint_pilot_report.py` | Python 源码 | 25.9 KB | 模块 docstring：Build portable, saved-image stage/variant reports for the joint BFS pilot. No GPU, detecto；顶层定义 7 个（`artifact_path`, `link`, `open_image`, `discover_runs`…） |
| `build_qwen21_joint_pilot_review_sheets.py` | Python 源码 | 3.6 KB | 模块 docstring：Render all twelve locked monitor cases in four-case baseline/candidate sheets.；顶层定义 1 个（`main`） |
| `build_qwen21_native_value_report.py` | Python 源码 | 28.7 KB | 模块 docstring：Build a readable Word report from the locked Qwen native/control audit. Images are embedde；顶层定义 8 个（`font`, `para`, `number`, `summary_value`…） |
| `plot_vram_requirement_dashboard.py` | Python 源码 | 8.8 KB | 模块 docstring：生成《DeepGen 1.0 姿态编辑实验显存需求全景看板》高清可视化图表 包含： 1. 训练模式显存实测与消费级/专业级显卡承载线 (16G/24G/48G/84G) 2. 全生 |
| `render_3dpw_60_pairs_report_style_v1.py` | Python 源码 | 9.9 KB | 模块 docstring：Render already-selected 3DPW 60 showcase pairs in report style. Layout: SOURCE RGB / TARGE；顶层定义 7 个（`load_rgb`, `load_depth`, `dwpose_black_background`, `normal_black_background`…） |
| `summarize_qwen21_face_groups.py` | Python 源码 | 6.8 KB | 模块 docstring：Paired Face variant comparisons on common visible faces and pose groups.；顶层定义 4 个（`arguments`, `mean`, `paired`, `main`） |
| `summarize_qwen21_joint_pilot.py` | Python 源码 | 26.9 KB | 模块 docstring：Summarize the actual joint-pilot logs and saved-image scores on locked cohorts. Full36 ini；顶层定义 10 个（`error`, `read_json`, `read_jsonl`, `finite`…） |
| `test_qwen21_joint_pilot_visuals.py` | Python 源码 | 14.9 KB | 模块 docstring：CPU contract tests using saved real RGB images; no pipeline simulation.；顶层定义 18 个（`module`, `portrait`, `test_index_selection_preserves_requested_order`, `test_bad_selection_rejected_without_substitution`…） |
| `visualize_3dpw_20_pairs_assets.py` | Python 源码 | 14.7 KB | 模块 docstring：Visualize 20 representative pairs from 3DPW single-person dataset. Each pair is saved as o；顶层定义 6 个（`load_rgb`, `load_gray`, `load_depth`, `depth_for_display`…） |
| `visualize_3dpw_20_pairs_assets_diverse.py` | Python 源码 | 13.8 KB | 顶层定义 5 个（`load_rgb`, `load_gray`, `load_depth`, `depth_vis`…） |
| `visualize_3dpw_60_pairs_diverse_dinov2_v2.py` | Python 源码 | 40.8 KB | 模块 docstring：3DPW 60-pair diverse visualization set v2 Goal ---- Build a visually diverse 60-pair showc；顶层定义 3 个（`sequence_family`, `load_dinov2`, `RGBFrameDataset`） |
| `visualize_3dpw_60_pairs_large_pose.py` | Python 源码 | 16.6 KB | 模块 docstring：3DPW final dataset visualization Selection: - all 30 final sequences - exactly 2 pairs per；顶层定义 6 个（`choose_two`, `load_rgb`, `load_gray`, `load_depth`…） |
| `visualize_tiktok_v3_final_global_blur.py` | Python 源码 | 6.2 KB | 顶层定义 8 个（`read_jsonl`, `read_bgr`, `build_index`, `resize_h`…） |
| `write_iper1024_result.py` | Python 源码 | 6.5 KB | 模块 docstring：Summarize paired 512/1024 Qwen21 iPER results without mixing pixel scales.；顶层定义 3 个（`arguments`, `table`, `main`） |
| `write_qwen21_face_repair_report.py` | Python 源码 | 27.3 KB | 模块 docstring：Write the bounded Face repair report from saved evidence; no model execution.；顶层定义 11 个（`read`, `finite`, `number`, `by_step`…） |
| `write_qwen21_face_report.py` | Python 源码 | 13.7 KB | 模块 docstring：Write a compact, image-rich Word report from fixed Face validation/test outputs.；顶层定义 5 个（`arguments`, `metric`, `wrap_panel`, `add_table`…） |
| `write_qwen21_face_single_sample_report.py` | Python 源码 | 14.3 KB | 模块 docstring：Word report with embedded full-pixel panels and explicit training-data labels.；顶层定义 5 个（`read`, `number`, `panel`, `metric_table`…） |
| `write_qwen21_full_report.py` | Python 源码 | 40.2 KB | 模块 docstring：Build a Word acceptance report and full-resolution two-row comparison PNGs. Reads recorded；顶层定义 12 个（`arguments`, `read_json`, `read_jsonl`, `finite`…） |
| `write_qwen21_inheritance_report.py` | Python 源码 | 19.8 KB | 模块 docstring：Embed measured tables and original-pixel comparison panels in a Word report.；顶层定义 3 个（`make_panel`, `comparison_table`, `main`） |
| `write_qwen21_weight_audit_report.py` | Python 源码 | 17.4 KB | 模块 docstring：Add provenance/semantic evidence and write the standalone Word audit report.；顶层定义 3 个（`table`, `supplemental_evidence`, `main`） |

## 四、构成画像

- **直属类型分布**：Python 源码×30
- **子树类型构成**：Python 源码×30

## 五、导航

- 根目录总览：[`目录总览.md`](../../目录总览.md)
- 根 README：[README.md](../../README.md)
