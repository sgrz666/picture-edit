# v6 目录说明

**路径**: `research/adapter_audit_20261002/remote_snapshot/src/pose_control/v6`

## 1. 目录定位与职责

2026-10-02 适配器架构审计与远端快照子目录 `v6`。

**工程定位**: 保存当时远端服务器的完整代码快照、命名空间迁移验收测试与原生运行时诊断记录。

## 2. 目录结构与内容汇总

### 子文件夹

- **`detail/`**: 该子目录下的专门功能模块或产物。
- **`face/`**: 该子目录下的专门功能模块或产物。
- **`hand/`**: 该子目录下的专门功能模块或产物。
- **`interface/`**: 该子目录下的专门功能模块或产物。
- **`reasoning/`**: 该子目录下的专门功能模块或产物。

### 文件概览

- **文件总数**: 12 个文件
- **文件类型统计**: .md: 1个, .py: 11个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `__init__.py` | 该模块下实现/记录文件 |
| `checkpoint.py` | 检查点保存与加载逻辑，要求 `architecture == qwen21_full_v1` 并严格校验权重形状 |
| `condition_bridge.py` | 条件特征转换与桥接器，将 1536 维几何特征投影并重排为 Qwen DiT token |
| `condition_injector.py` | `SMPLXConditionInjector` 多模态几何条件注入器，统一编码法线、姿态与全局参数 |
| `conditions.py` | 条件数据结构定义（包含目标法线、姿态热图、部位掩码与任务 Token） |
| `control_core.py` | `SharedRecurrentControlCore` 共享循环控制核心，负责 6 阶段控制残差循环计算 |
| `controlled_pipeline.py` | 集成 Qwen 2.1 控制器的端到端受控图像采样生成 Pipeline |
| `deepgen_adapter.py` | 保留的 DeepGen 原始适配器基类与兼容层 |
| `experts.py` | 多专家网络（MoE）路由与身体/局部细节控制专家实现 |
| `geometry_encoder.py` | SMPL-X 空间几何结构编码器 |
| `token_encoders.py` | 针对目标图像 token 的零初始化投影与残差映射头 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
