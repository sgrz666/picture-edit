# Qwen 目录说明

**路径**: `experiments/bfs_body_face_joint_pilot_20261008/training/Qwen`

## 1. 目录定位与职责

联合训练执行过程日志与检查点配置目录。

**工程定位**: 分别记录 Joint 联合模型 (J) 与基线微调模型 (Qwen) 的训练损失、显存占用与状态。

## 2. 目录结构与内容汇总

### 文件概览

- **文件总数**: 8 个文件
- **文件类型统计**: .json: 6个, .jsonl: 1个, .md: 1个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `lora_loading.json` | 该模块下实现/记录文件 |
| `metrics.jsonl` | 该模块下实现/记录文件 |
| `resume_manifest.json` | 该模块下实现/记录文件 |
| `resume_reconciliation.json` | 该模块下实现/记录文件 |
| `resume_state_step_000128.json` | 该模块下实现/记录文件 |
| `run_manifest.json` | 该模块下实现/记录文件 |
| `summary.json` | 该模块下实现/记录文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
