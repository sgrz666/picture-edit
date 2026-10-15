# remote_snapshot 目录说明

**路径**: `research/adapter_audit_20261002/remote_snapshot`

## 1. 目录定位与职责

2026-10-02 适配器架构审计与远端快照子目录 `remote_snapshot`。

**工程定位**: 保存当时远端服务器的完整代码快照、命名空间迁移验收测试与原生运行时诊断记录。

## 2. 目录结构与内容汇总

### 子文件夹

- **`configs/`**: 该子目录下的专门功能模块或产物。
- **`docs/`**: 该子目录下的专门功能模块或产物。
- **`scripts/`**: 该子目录下的专门功能模块或产物。
- **`src/`**: 该子目录下的专门功能模块或产物。
- **`tests/`**: 该子目录下的专门功能模块或产物。

### 文件概览

- **文件总数**: 9 个文件
- **文件类型统计**: .md: 1个, .py: 7个, .txt: 1个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `eval_hi4d_qwen21_dual.py` | 该模块下实现/记录文件 |
| `eval_iper_qwen21_full.py` | Qwen 2.1 Full 模型在 iPER 数据集上的评测与采样入口 |
| `requirements_qwen21.txt` | 该模块下实现/记录文件 |
| `train_hi4d_qwen21_dual.py` | 该模块下实现/记录文件 |
| `train_iper_qwen21_body.py` | 该模块下实现/记录文件 |
| `train_iper_qwen21_face.py` | 该模块下实现/记录文件 |
| `train_iper_qwen21_full.py` | Qwen 2.1 Full 模型在 iPER 数据集上的主训练入口 |
| `train_iper_v6_native.py` | 该模块下实现/记录文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
