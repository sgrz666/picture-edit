# remote_source 目录说明

**路径**: `research/adapter_value_diagnosis_20261011/remote_source`

## 1. 目录定位与职责

研究与调研子目录 `remote_source`。

**工程定位**: 保存专项调研文档、测试数据或审计记录。

## 2. 目录结构与内容汇总

### 文件概览

- **文件总数**: 24 个文件
- **文件类型统计**: .md: 1个, .py: 23个

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
| `data_runtime.py` | 运行时数据构建、Batch 打包及 Condition Bundle 动态生成 |
| `deepgen_adapter.py` | 保留的 DeepGen 原始适配器基类与兼容层 |
| `eval_qwen21_joint_pilot.py` | 该模块下实现/记录文件 |
| `eval_qwen21_native_value.py` | 该模块下实现/记录文件 |
| `execution.py` | 执行上下文管理与前向执行流调度逻辑 |
| `experts.py` | 多专家网络（MoE）路由与身体/局部细节控制专家实现 |
| `face_coordinates.py` | 面部关键点检测与局部 16x16 ROI 坐标映射计算 |
| `face_loss_alignment.py` | 面部特征对齐、表情重建与身份一致性损失函数 |
| `face_supervision.py` | 人脸区域局部监督与梯度反向传播控制策略 |
| `frozen_bfs_lora.py` | 冻结 BFS 骨干网络并集成 LoRA 微调权重逻辑 |
| `geometry_encoder.py` | SMPL-X 空间几何结构编码器 |
| `joint_training.py` | Body-Face-Hand 联合多任务训练管理与损失加权 |
| `qwen_adapter.py` | `Qwen21FullControlContext`，通过 PyTorch forward hook 向 Qwen 32 个 DiT Block 注入 4096 维控制残差 |
| `qwen_runtime.py` | Qwen 2.1 运行时环境管理、Prompt Embedding 缓存与 VAE 解码 |
| `token_encoders.py` | 针对目标图像 token 的零初始化投影与残差映射头 |
| `train_qwen21_joint_pilot.py` | 该模块下实现/记录文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
