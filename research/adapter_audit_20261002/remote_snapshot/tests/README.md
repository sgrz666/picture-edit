# tests 目录说明

**路径**: `research/adapter_audit_20261002/remote_snapshot/tests`

## 1. 目录定位与职责

2026-10-02 适配器架构审计与远端快照子目录 `tests`。

**工程定位**: 保存当时远端服务器的完整代码快照、命名空间迁移验收测试与原生运行时诊断记录。

## 2. 目录结构与内容汇总

### 文件概览

- **文件总数**: 39 个文件
- **文件类型统计**: .md: 1个, .py: 38个

由于文件数量较多（共 39 个），下表列出部分代表性文件：

| 文件名（样例） | 说明 |
|---|---|
| `README.md` | 代表性数据/代码/图像文件 |
| `test_adapter_reasoning_v1.py` | 代表性数据/代码/图像文件 |
| `test_adapter_v6_architecture.py` | 验证 DeepGen V6 架构完整前向传播与各分支残差张量形状 |
| `test_analysis.py` | 代表性数据/代码/图像文件 |
| `test_champ_deepgen_integration.py` | 代表性数据/代码/图像文件 |
| `test_condition_bridge_contract.py` | 验证条件桥接协议（Condition Bridge）输入输出契约与张量对齐 |
| `test_condition_injector_v1.py` | 代表性数据/代码/图像文件 |
| `test_deepgen_control_interface_v1.py` | 验证适配器零卷积初始化行为，确保未训练时生成基座输出严格不变 |
| `test_deepgen_no_control_ablation.py` | 验证在无控制信号输入时的消融基线表现与退化安全性 |
| `test_deepgen_pose_patch.py` | 代表性数据/代码/图像文件 |
| `test_hi4d_geometry_registration.py` | 代表性数据/代码/图像文件 |
| `test_iper_v6_data_integrity.py` | 代表性数据/代码/图像文件 |
| `test_iper_v6_eval_parity.py` | 代表性数据/代码/图像文件 |
| `test_iper_v6_training_pipeline.py` | 代表性数据/代码/图像文件 |
| `test_lab.py` | 代表性数据/代码/图像文件 |
| *...（其余 24 个文件）* | 同类型输出或支持文件 |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
