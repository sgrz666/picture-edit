# scripts/reports/ —— 成果报告与可视化面板生成器（只读已存结果，不做推理）
> 工作区路径：/home/shangguanrz/project/pic-edit/scripts/reports　|　本级：0 个子目录 / 30 个实体文件（约 0.8 MB，`du -sh --si` 2026-10-12 实测）　|　另含生成物 README.md、目录说明.md、`__pycache__`

## 一、这一层在干什么
把 `scripts/eval/`、`scripts/train/` 留在 `experiments/` 与 `/home/shangguanrz/nihaochen/` 的日志、指标 JSON 和图片，转成可交付的 Word 简报、HTML/面板图和汇总统计。几乎所有 docstring 都声明"Reads recorded results only / No GPU, detector or metric model"，即报告生成绝不重新跑模型，缺失证据会显式标记 incomplete 而不是静默补数。产出物历史上归入 `reports/`（本地）与 `docs/`。

## 二、层级地图（按产物类型分类）
| 类别 | 文件 | 实证 |
|---|---|---|
| 中文 Word 简报（Docx） | `build_20b_memory_docx.py`（Qwen-Image-Edit-20B 显存预算与单卡 84GB 可行性）、`build_architecture_docx.py`（系统架构与核心技术方案）、`build_brief_architecture_docx.py`（课题组月报版）、`build_latest_experiment_report_docx.py`（基于 120 次实测运行、60 画质复核、100 次更新稳定性测试）、`plot_vram_requirement_dashboard.py`（显存看板，含"缓存+检查点较在线无检查点降 61.2%"的瀑布分解、5B vs 20B 对比） | 输出 .docx/.png 到 `reports/`、`docs/` 或调用参数 |
| Qwen21 系列英文报告 | `write_qwen21_full_report.py`（验收 Word + 两行对比图，`--require-complete` 门禁）、`write_qwen21_inheritance_report.py`、`write_qwen21_weight_audit_report.py`、`write_qwen21_face_report.py`、`write_qwen21_face_repair_report.py`、`write_qwen21_face_single_sample_report.py`、`write_iper1024_result.py`（512/1024 配对结果不混尺度）、`build_qwen21_native_value_report.py`（插图来自预选集合、不按结果好坏排序）、`build_qwen21_combination_report.py`、`build_qwen21_joint_pilot_report.py` + `build_qwen21_joint_pilot_review_sheets.py`（12 个固定监看案例四列基线/候选表） | 与 `scripts/eval/` 同名族一一对应 |
| Hi4D / BFS 报告 | `build_hi4d_native_binding_report.py`（嵌图 Word）、`build_qwen21_bfs_rgb_report.py`（1024 固定 A/B RGB 参照）、`build_qwen21_bfs_skeleton_report.py` | — |
| 面板/网格可视化 | `build_iper1024_comparison_panels.py`、`build_qwen21_face_panels.py`（整图+脸部细节配对面板）、`build_20b...`（见上） | — |
| 3DPW 展示选样 | `visualize_3dpw_20_pairs_assets.py`（2×5 montage：RGB/DWPose/Mask/Depth/Normal）、`_diverse.py`、`visualize_3dpw_60_pairs_large_pose.py`（30 序列×2 对、大姿态优先）、`visualize_3dpw_60_pairs_diverse_dinov2_v2.py`（不改正式 manifest 的多样性选样）、`render_3dpw_60_pairs_report_style_v1.py`（报告版式） | 输入是 3DPW 资产目录（`add/outputs/3dpw_assets_all_v1` 一线），输出 montage PNG |
| 汇总统计 | `summarize_qwen21_face_groups.py`（公共可见脸与姿态组上的配对比较）、`summarize_qwen21_joint_pilot.py`（Full36 初/J/终 与 fixed12 分离，配对估计用案例交集而非均值差）、`test_qwen21_joint_pilot_visuals.py`（用已存真图的 CPU 契约测试）、`visualize_tiktok_v3_final_global_blur.py`（输出 `TikTok_v3/qc/`） | — |

## 三、关键文件与实证内容
- `summarize_qwen21_joint_pilot.py`（26.9 KB）：`docs/experiments/2026-10-08-bfs-body-face-joint-handoff.md` 中的指标表（J 底座 ID 0.4602、BFSHead PCK@0.05 0.9206）即由它汇总。
- `build_latest_experiment_report_docx.py`：docstring 的"120 次实测运行、显存全矩阵采样、100 次更新稳定性测试"对应 V6 阶段显存与训练审计成果，正文内嵌 2 张高清显存图表。
- 报告纪律实例：`build_qwen21_joint_pilot_report.py` 逐案例裁切统一使用 B 几何 ROI，"不做输出检测或讨好式重新居中"。

## 四、注意事项 / 调用路径
- 全部脚本依赖结果已存在；跑报告前确认对应 `--run/--root` 指向完成的实验目录。
- 根目录兼容链接 `scripts/write_qwen21_face_single_sample_report.py → reports/…`，实体在本目录。
- Word 生成需 python-docx（qwen21 环境自带）；历史产物在本地 `reports/*.docx`，远端 `reports/` 只有指向他人目录的链接（见 `reports/README.md`）。
