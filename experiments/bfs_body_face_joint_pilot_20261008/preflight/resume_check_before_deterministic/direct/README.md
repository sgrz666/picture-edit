# direct 目录说明

**路径**: `experiments/bfs_body_face_joint_pilot_20261008/preflight/resume_check_before_deterministic/direct`

## 1. 目录定位与职责

模型起训前 Preflight 快速前向、梯度与断点续训确定性验证目录。

**工程定位**: 确保在提交大规模训练前消除随机性偏差，验证 checkpoint 恢复前后 Loss 严格一致。

## 2. 目录结构与内容汇总

### 文件概览

- **文件总数**: 5 个文件
- **文件类型统计**: .json: 3个, .jsonl: 1个, .md: 1个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `lora_loading.json` | 该模块下实现/记录文件 |
| `metrics.jsonl` | 该模块下实现/记录文件 |
| `run_manifest.json` | 该模块下实现/记录文件 |
| `summary.json` | 该模块下实现/记录文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
