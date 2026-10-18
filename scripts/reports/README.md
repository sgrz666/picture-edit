# scripts/reports/ —— Word 报告、图表与面板生成

本目录是系统的「成果物化」层：把 `experiments/`、`scripts/eval/` 与 `scripts/audit/` 产出的数值、日志与已保存图像，沉淀为可汇报的 Word 文档、可视化图表与对照面板。它位于流水线末端，几乎全部是**只读消费**既有产物（docstring 普遍标注「不生成图像、不跑推理、不更新权重」），因此可在把评测树拷到本地后离线运行。多数 Word 报告用 `python-docx` 内嵌图像，并偏好从预提交的固定选择中挑图（而非按结果质量排序），以保证汇报客观。

## 里面有什么

### 架构 / 实验简报（Word）

| 文件 | 一句话职责 |
|---|---|
| `build_architecture_docx.py` | 生成《人物图像姿态编辑系统架构设计与核心技术方案》Word：背景、DeepGen 1.0 架构、SCB 连接器、接触感知适配器、SMPL-X 控制注入、双人交互、训练损失与实测对比 |
| `build_brief_architecture_docx.py` | 科研同步版精简架构与进展简报（月报/双周同步用） |
| `build_latest_experiment_report_docx.py` | 生成《DeepGen 1.0 最新实验结果与显存分析简报》Word（多图内嵌） |
| `build_20b_memory_docx.py` | 生成《Qwen-Image-Edit 20B 显存预算与单卡可行性分析》简报 |

### Qwen21 各线报告（Word / 面板）

| 文件 | 一句话职责 |
|---|---|
| `build_qwen21_bfs_rgb_report.py` | 固定 1024 A-RGB + B-RGB BFS 对照的简明 Word 报告 |
| `build_qwen21_bfs_skeleton_report.py` | 固定三臂 BFS 审计的短报告（内嵌图） |
| `build_qwen21_combination_report.py` | 带锁定 native 组合网格的 value 审计 Word 修订版 |
| `build_qwen21_face_panels.py` | 给固定 Qwen21 Face 评测做全图 + 人脸细节配对面板 |
| `build_qwen21_joint_pilot_report.py` | joint BFS pilot 的可移植、已保存图像阶段/变体报告（无需 GPU/检测器） |
| `build_qwen21_joint_pilot_review_sheets.py` | 把 12 个锁定监测样例渲染成四案 baseline/candidate 对照表 |
| `build_qwen21_native_value_report.py` | 从锁定 Qwen native/control 审计构建可读 Word 报告（全图 + 实测人脸行） |
| `build_hi4d_native_binding_report.py` | 固定 native Hi4D 绑定实验的内嵌图 Word 报告 |
| `write_qwen21_face_repair_report.py` | 从已保存证据写 Face 修复报告（不执行模型） |
| `write_qwen21_face_report.py` | 从固定 Face val/test 输出写紧凑、图像丰富的 Word 报告 |
| `write_qwen21_face_single_sample_report.py` | 内嵌全像素面板、显式标注训练数据归属的 Word 报告 |
| `write_qwen21_full_report.py` | 构建 Word 验收报告与全分辨率双行对照 PNG；`--require-complete` 为交付闸门 |
| `write_qwen21_inheritance_report.py` | 内嵌实测表与原像素对照面板的权重继承 Word 报告 |
| `write_qwen21_weight_audit_report.py` | 追加溯源/语义证据并写独立权重审计 Word 报告 |

### Hi4D 双人深度 / 骨架对比报告（Word，10-12 ~ 10-13 新增）

| 文件 | 一句话职责 |
|---|---|
| `build_hi4d_separate_depth_bilingual_report.py` | 内嵌式 Word 报告：Hi4D 双人「骨架与深度」**中英文提示词对比**，含精确像素输出面板与目标骨架叠加；消费 `eval_hi4d_separate_depth_bilingual` 与 `score_hi4d_separate_depth_bilingual` 的结果（输出《Hi4D双人骨架与深度_中英文提示词对比_20261012.docx》） |
| `build_hi4d_depth_combinations_report.py` | 内嵌式 Word 报告：Hi4D 双人 5 种锁定 native 深度输入组合（skeleton / skeleton_joint_depth / joint_depth / separate_depth / joint_separate_depth）对比，含精确像素面板与骨架叠加；消费 `eval_hi4d_depth_combinations` 与 `score_hi4d_depth_combinations` 的结果（输出《Hi4D双人_深度输入组合对比_20261013.docx》） |

### 图表 / 数据集可视化

| 文件 | 一句话职责 |
|---|---|
| `plot_vram_requirement_dashboard.py` | 生成《DeepGen 1.0 显存需求全景看板》高清图（训练/推理/缓存三阶段、优化瀑布、5B vs 20B） |
| `visualize_3dpw_20_pairs_assets.py` | 3DPW 单人 20 对代表样本 montage（RGB/DWPose/Mask/Depth/Normal） |
| `visualize_3dpw_20_pairs_assets_diverse.py` | 3DPW 20 对资产多样可视化 |
| `visualize_3dpw_60_pairs_diverse_dinov2_v2.py` | 3DPW 60 对多样展示集 v2（不改官方/最终训练 manifest） |
| `visualize_3dpw_60_pairs_large_pose.py` | 3DPW 最终可视化：30 序列 × 2 对 = 60 对 |
| `render_3dpw_60_pairs_report_style_v1.py` | 3DPW 60 展示对按报告样式渲染（SOURCE/TARGET RGB/DWPose/Normal/Depth） |
| `visualize_tiktok_v3_final_global_blur.py` | TikTok V3 终稿全局模糊可视化 |
| `build_iper1024_comparison_panels.py` | 从已保存 RGB 输出做带人脸裁切的可读 iPER 对照表 |

### 汇总 / 合同测试

| 文件 | 一句话职责 |
|---|---|
| `summarize_qwen21_face_groups.py` | 在常见可见人脸与姿态组上做配对 Face 变体对照 |
| `summarize_qwen21_joint_pilot.py` | 汇总 joint-pilot 真实日志与已保存图像分数（Full36 与 fixed12 分开，配对估计用交集） |
| `test_qwen21_joint_pilot_visuals.py` | 用已保存真实 RGB 图像的 CPU 合同测试（不模拟 pipeline） |
| `write_iper1024_result.py` | 在不混像素尺度下汇总配对的 512/1024 Qwen21 iPER 结果 |

## 与上下层的关系

- **上游输入**：`experiments/<run>/`（训练/采样产物）、`scripts/eval/` 的结果 JSON/CSV 与对照图、`scripts/audit/` 的审计证据（如 `write_qwen21_weight_audit_report.py` 直接消费权重审计）。
- **产出物**：`reports/` 相关目录与 `results/` 下的 `.docx`、面板 PNG、看板图；少量脚本（如 `visualize_3dpw_*`）也服务于 `datasets/` 展示。
- 相关索引：父级 [`../README.md`](../README.md)；评测契约见 [`../eval/README.md`](../eval/README.md)，资产路径登记在 [`../../configs/asset_registry.json`](../../configs/asset_registry.json)。

## 注意与坑

- 本目录脚本普遍**只读产物、不重跑模型**；但 `build_*` 系列若依赖 `scripts/eval/` 尚未生成的对照图，需先跑对应 `eval_*`。
- 多个报告脚本强调「图从预提交选择中挑、不按质量排序」，改动选图逻辑会破坏汇报的客观性，需谨慎。
- `write_qwen21_full_report.py` 的 `--require-complete` 是最终交付闸门，中间报告可无它先出。
