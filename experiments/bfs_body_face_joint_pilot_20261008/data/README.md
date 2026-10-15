# data 目录说明

**路径**: `experiments/bfs_body_face_joint_pilot_20261008/data`

## 1. 目录定位与职责

Pilot 实验数据索引、预处理日志与配置目录。

**工程定位**: 记录训练集/验证集 manifest 划分及特征抽取处理日志。

## 2. 目录结构与内容汇总

### 子文件夹

- **`process_logs/`**: 该子目录下的专门功能模块或产物。

### 文件概览

- **文件总数**: 12 个文件
- **文件类型统计**: .gz: 2个, .json: 4个, .jsonl: 5个, .md: 1个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `branch256_pairs.jsonl` | 该模块下实现/记录文件 |
| `common256_pairs.jsonl` | 该模块下实现/记录文件 |
| `handoff_integration.json` | 该模块下实现/记录文件 |
| `iper12_pairs.jsonl` | 该模块下实现/记录文件 |
| `lock.json` | 该模块下实现/记录文件 |
| `preprocessing_review_required.json` | 该模块下实现/记录文件 |
| `remote_before_handoff.tar.gz` | 该模块下实现/记录文件 |
| `runtime_source_20261008.tar.gz` | 该模块下实现/记录文件 |
| `runtime_source_hashes.json` | 该模块下实现/记录文件 |
| `train512_pairs.jsonl` | 该模块下实现/记录文件 |
| `val24_pairs.jsonl` | 该模块下实现/记录文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
