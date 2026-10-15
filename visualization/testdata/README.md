# testdata 目录说明

**路径**: `visualization/testdata`

## 1. 目录定位与职责

标准化评测测试集基准切片目录。

**工程定位**: 提供评测脚本标准输入图像与对应的 SMPL-X 控制条件对。

## 2. 目录结构与内容汇总

### 子文件夹

- **`conditions/`**: 该子目录下的专门功能模块或产物。
- **`images/`**: 该子目录下的专门功能模块或产物。

### 文件概览

- **文件总数**: 3 个文件
- **文件类型统计**: .json: 1个, .md: 2个

| 文件名 | 说明 |
|---|---|
| `README.md` | 该模块下实现/记录文件 |
| `README_test_evaluation.md` | 评测基准构建规范、样本选取原则与指标计算说明文档 |
| `eval_metrics.json` | 标准化评测集上的多模型定量评价指标对比数据 (PSNR, SSIM, LPIPS) |

## 3. 使用与关联说明

- 更多全局架构说明与模型演进路线请参阅根目录 [README.md](../../README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md](../../docs/PROJECT_STRUCTURE_20261003.md)。
- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。
