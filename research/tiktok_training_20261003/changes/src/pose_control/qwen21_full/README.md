# qwen21_full 目录说明

**路径**: `research/tiktok_training_20261003/changes/src/pose_control/qwen21_full`

## 1. 目录定位与职责

TikTok 数据集接入与训练方案技术调研子目录 `qwen21_full`。

**工程定位**: 对比训练基线代码快照 (`baseline`) 与适配器增量变更 (`changes`)。

## 2. 目录结构与内容汇总

### 文件概览

- **文件总数**: 4 个文件
- **文件类型统计**: .md: 1个, .py: 3个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `data_runtime.py` | 运行时数据构建、Batch 打包及 Condition Bundle 动态生成 |
| `execution.py` | 执行上下文管理与前向执行流调度逻辑 |
| `face_supervision.py` | 人脸区域局部监督与梯度反向传播控制策略 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
