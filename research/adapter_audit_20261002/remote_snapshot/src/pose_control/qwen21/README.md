# qwen21 目录说明

**路径**: `research/adapter_audit_20261002/remote_snapshot/src/pose_control/qwen21`

## 1. 目录定位与职责

2026-10-02 适配器架构审计与远端快照子目录 `qwen21`。

**工程定位**: 保存当时远端服务器的完整代码快照、命名空间迁移验收测试与原生运行时诊断记录。

## 2. 目录结构与内容汇总

### 文件概览

- **文件总数**: 8 个文件
- **文件类型统计**: .md: 1个, .py: 7个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `__init__.py` | 该模块下实现/记录文件 |
| `action_prompts.py` | 该模块下实现/记录文件 |
| `body_adapter.py` | Qwen 2.1 简化版人体姿态控制适配器（`Qwen21BodyAdapter`） |
| `dual_adapter.py` | Qwen 2.1 简化版双人交互姿态控制适配器（`Qwen21DualAdapter`） |
| `dual_inference.py` | 该模块下实现/记录文件 |
| `face_adapter.py` | Qwen 2.1 简化版人脸局部控制适配器（`Qwen21FaceAdapter`） |
| `runtime.py` | 该模块下实现/记录文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
