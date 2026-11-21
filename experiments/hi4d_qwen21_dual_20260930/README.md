# hi4d_qwen21_dual_20260930 目录说明

新增 [Native V7 单配对真实训练](native-v7-singlepair-overfit-20261117/README.md)：原生尺寸300步与四阶段真实门禁全部完成。训练／同人物观察帧 PCK@0.2 从62.50%／58.33%升至95.83%／95.83%，旧手臂残影消失；检测对齐的训练身份分数反而下降，背景保真也有退步，暂不建议直接扩大训练。完整固定噪声、SDPose和两种身份对齐审计见实验 README 及服务器报告。

后续 [Source身份／背景保持改进](native-v7-source-preservation-20261121/README.md) 保留V7架构，加入Source绝对身份与背景x0目标。相同180步起点的两组12步控制完成，四个检测对齐身份分数均提高，背景小幅改善；训练姿态误差约+0.06px、观察PCK@0.2为100%。这是短程同人物证据，未启动大规模训练。

**路径**: `experiments/hi4d_qwen21_dual_20260930`

## 1. 目录定位与职责

Hi4D Qwen 2.1 双人交互姿态编辑实验子目录 `hi4d_qwen21_dual_20260930`。

**工程定位**: 包含针对 Hi4D 双人交互数据集（搭肩、握手、拥抱）的专用数据管道、控制器扩展与测试用例。

## 2. 目录结构与内容汇总

### 子文件夹

- **`docs/`**: 该子目录下的专门功能模块或产物。
- **`src/`**: 该子目录下的专门功能模块或产物。
- **`tests/`**: 该子目录下的专门功能模块或产物。

### 文件概览

- **文件总数**: 23 个文件
- **文件类型统计**: .json: 4个, .md: 4个, .png: 8个, .py: 7个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `audit_hi4d_qwen21_prompt_leak.py` | 该模块下实现/记录文件 |
| `backhug_geometry_573.png` | 该模块下实现/记录文件 |
| `complete_hi4d_qwen21_dual.py` | 该模块下实现/记录文件 |
| `diagnose_backhug_geometry.py` | 该模块下实现/记录文件 |
| `eval_hi4d_qwen21_dual.py` | 该模块下实现/记录文件 |
| `full_run_stopped_status.json` | 该模块下实现/记录文件 |
| `inspect_single_hug.py` | 该模块下实现/记录文件 |
| `pair14_geometry_switch.png` | 该模块下实现/记录文件 |
| `pair14_new_overfit_seed42_panel.png` | 该模块下实现/记录文件 |
| `pair14_prompt_audit_report.md` | 该模块下实现/记录文件 |
| `pair14_seed42_panel.png` | 该模块下实现/记录文件 |
| `pair14_seed42_panel_v2.png` | 该模块下实现/记录文件 |
| `pair14_seed42_summary_v2.json` | 该模块下实现/记录文件 |
| `pair14_seed43_summary_v2.json` | 该模块下实现/记录文件 |
| `pair14_seed44_summary_v2.json` | 该模块下实现/记录文件 |
| `reconcile_overfit_resume.py` | 该模块下实现/记录文件 |
| `single_hug_panel.png` | 该模块下实现/记录文件 |
| `single_hug_panel_seed43.png` | 该模块下实现/记录文件 |
| `single_hug_report.md` | 该模块下实现/记录文件 |
| `smoke_panel.png` | 该模块下实现/记录文件 |
| `test_control_audit.md` | 该模块下实现/记录文件 |
| `train_hi4d_qwen21_dual.py` | 该模块下实现/记录文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
