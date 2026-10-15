# src/adapter/ —— 第一代 CNN 姿态条件适配器（历史基线）

> 工作区路径：`/home/shangguanrz/project/pic-edit/src/adapter`　|　本级：1 个实现文件 / 3 个文件 / 约 4 KB　|　父级：`src/`

## 一、这一层在干什么

项目最早的姿态注入方案：`PoseConditionAdapter` 用零卷积把 8 通道控制图（骨架/法线等）残差注入 UNet/MMDiT 特征。它只在早期 DeepGen 对照实验（`experiments/iper_adapter_v1/`、`iper_adapter_finetune_v1/`）和显存测绘中被使用，现已被 `src/pose_control/v6/` 的统一控制架构取代，保留目的是复现历史数字与 `vram_lab` 的规模敏感性实验。

## 二、层级地图

| 文件 | 是什么 | 实证 |
|---|---|---|
| `pose_adapter.py`（165 行） | `ResnetBlock2D`(L7)、`Downsample2D`(L32)、`PoseConditionAdapter`(L44) 三个类 | 无训练入口，纯 nn.Module |
| `__init__.py` | 空文件 | — |

## 三、注意事项

- 检查点与 `pose_control/*` 各线**互不兼容**（`docs/PROJECT_STRUCTURE_20261003.md`：检查点不能跨架构互换）。
- 消费方：`scripts/train/train_adapter.py`、`vram_lab/`。新人请勿从这里了解现役架构。
