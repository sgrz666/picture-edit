# baseline 目录说明

**路径**: `research/tiktok_training_20261003/baseline`

## 1. 目录定位与职责

TikTok 数据集接入与训练方案技术调研子目录 `baseline`。

**工程定位**: 对比训练基线代码快照 (`baseline`) 与适配器增量变更 (`changes`)。

## 2. 目录结构与内容汇总

### 子文件夹

- **`scripts/`**: 该子目录下的专门功能模块或产物。
- **`src/`**: 该子目录下的专门功能模块或产物。
- **`tests/`**: 该子目录下的专门功能模块或产物。

### 文件概览

- **文件总数**: 26 个文件
- **文件类型统计**: .md: 1个, .py: 24个, .sh: 1个

由于文件数量较多（共 26 个），下表列出部分代表性文件：

| 文件名（样例） | 说明 |
|---|---|
| `README.md` | 代表性数据/代码/图像文件 |
| `audit_tiktok_v3_blur_sample.py` | 代表性数据/代码/图像文件 |
| `audit_tiktok_v3_final_pairs.py` | 代表性数据/代码/图像文件 |
| `audit_tiktok_v3_pose_redundancy.py` | 代表性数据/代码/图像文件 |
| `audit_tiktok_v3_strict_final.py` | 代表性数据/代码/图像文件 |
| `audit_tiktok_v3_visual_quality_v2.py` | 代表性数据/代码/图像文件 |
| `audit_tiktok_v3_visual_quality_v3.py` | 代表性数据/代码/图像文件 |
| `build_tiktok_v3_candidate_pool.py` | 代表性数据/代码/图像文件 |
| `build_tiktok_v3_condition_inventory.py` | 代表性数据/代码/图像文件 |
| `build_tiktok_v3_pairs.py` | 代表性数据/代码/图像文件 |
| `build_tiktok_v3_pairs_strict.py` | 代表性数据/代码/图像文件 |
| `build_tiktok_v3_strict_visual_pool.py` | 代表性数据/代码/图像文件 |
| `build_tiktok_v3_usable_pool.py` | 代表性数据/代码/图像文件 |
| `eval_iper_qwen21_full.py` | Qwen 2.1 Full 模型在 iPER 数据集上的评测与采样入口 |
| `fill_tiktok_v3_missing_geometry.py` | 代表性数据/代码/图像文件 |
| *...（其余 11 个文件）* | 同类型输出或支持文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
