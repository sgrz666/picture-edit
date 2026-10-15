# data 目录说明

**路径**: `research/tiktok_training_20261003/baseline/src/data`

## 1. 目录定位与职责

TikTok 数据集接入与训练方案技术调研子目录 `data`。

**工程定位**: 对比训练基线代码快照 (`baseline`) 与适配器增量变更 (`changes`)。

## 2. 目录结构与内容汇总

### 文件概览

- **文件总数**: 2 个文件
- **文件类型统计**: .md: 1个, .py: 1个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `tiktok_dataset.py` | TikTok 数据集专用读取管道，支持单人舞蹈视频帧对与关键点坐标对齐 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
