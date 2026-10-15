# hand 目录说明

**路径**: `research/tiktok_training_20261003/baseline/src/pose_control/qwen21_full/hand`

## 1. 目录定位与职责

TikTok 数据集接入与训练方案技术调研子目录 `hand`。

**工程定位**: 对比训练基线代码快照 (`baseline`) 与适配器增量变更 (`changes`)。

## 2. 目录结构与内容汇总

### 文件概览

- **文件总数**: 9 个文件
- **文件类型统计**: .md: 1个, .py: 8个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `__init__.py` | 该模块下实现/记录文件 |
| `adapter.py` | 该模块下实现/记录文件 |
| `conditions.py` | 条件数据结构定义（包含目标法线、姿态热图、部位掩码与任务 Token） |
| `geometry.py` | 该模块下实现/记录文件 |
| `preparer.py` | 该模块下实现/记录文件 |
| `quality.py` | 该模块下实现/记录文件 |
| `refiner_export.py` | 该模块下实现/记录文件 |
| `vton_layers.py` | 该模块下实现/记录文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../../../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../../../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
