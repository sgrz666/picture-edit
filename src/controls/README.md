# src/controls/ —— 骨架可视化与生成辅助

`controls/` 提供姿态骨架的可视化与基础图像处理工具，供调试、样本检查与报告使用，不是训练/推理的核心路径。

## 里面有什么

- `generator.py` —— 骨架绘制与主流程辅助：`SMPLX_BODY_PAIRS`（24 关节父子连接，用于连线）、`BODY_PART_NAMES`（14 段身体命名）、骨架渲染与图像生成相关函数。文件较大（约 12KB），含多类绘制/合成工具。
- `__init__.py` —— 空。

## 与上下层的关系

- 上游：data 层或推理产出的关键点（COCO/Body25/SMPL-X 布局）。
- 下游：被评测/可视化脚本与 `evaluation/` 的辅助展示调用。

## 注意与坑

- `generator.py` 直接 `import cv2` 与 `numpy`，属于 CPU/本地可视化依赖，无需 GPU。
- `SMPLX_BODY_PAIRS` 是 SMPL-X 24 体关节连接表，与 V6 内部使用的 14 段 `PART_NAMES` 是两套不同粒度的约定，互转时需小心。
