# src/ —— 核心源码架构与模块导航

`src/` 是 pic-edit 的核心源码根目录，承载人物图像姿态编辑系统的全部可训练模型、几何条件构建、推理管线与评测逻辑。

---

## 一、核心子目录导航

| 目录 | 职责与定位 | 核心模块 / 关键代码 |
|---|---|---|
| [`pose_control/`](pose_control/README.md) | **姿态控制器核心**（训练与受控推理） | 包含 DeepGen V6、Qwen21 简化版、Qwen21 Full (含 `native_v5`)、Qwen21 Pro 与 Union 架构 |
| [`pose_edit/`](pose_edit/README.md) | **推理流水线共享层**（单/双人共用） | 缓存指纹（`fingerprint.py`）、SMPLest-X 封装、DWPose-134、geometry 定标与外观计算 |
| [`data/`](data/) | **数据集读取与几何预处理** | `tiktok_dataset.py`, `base_dataset.py`, `augmentation.py`，负责空间对齐与流式加载 |
| [`evaluation/`](evaluation/) | **定量评测与关键点回估** | `dwpose_onnx/` (133/134 点转换)、ArcFace 人脸特征提取、PSNR/SSIM/PCK/NME 计算 |
| [`controls/`](controls/) | **SMPL-X 控制信号生成器** | `generator.py`，负责从 3D 参数离线/在线渲染骨架、法线、深度、接触图 |
| [`integration/`](integration/) | **生成管线挂接与推理桥接** | `deepgen_pipeline.py`，将控制器残差无缝插入 Diffusers Pipeline 采样循环 |
| [`adapter/`](adapter/) | **早期 CNN 姿态适配器** | `pose_adapter.py`，早期零卷积基线实验代码 |

---

## 二、姿态控制技术线划分

1. **DeepGen V6 主线** ([`pose_control/v6/`](pose_control/v6/README.md))：
   - 基于 DeepGen-1.0 (2.47B MMDiT 双流基座)，挂载可训练的 Unified SMPL-X Adapter V6（V6.4 detail / V6.5 face / V6.6 hand）。
2. **Qwen 2.1 迁移与 Native V5 主线** ([`pose_control/qwen21_full/`](pose_control/qwen21_full/README.md))：
   - 基于 Qwen-Image-2.1 (7B 单流 DiT)，32 层全量注入。
   - 最新包含 [`native_v5/`](pose_control/qwen21_full/native_v5/)：模块化训练架构、显式损失权重、解耦 source_bank 与无检测器 ROI 监督。
3. **Qwen 2.1 Pro 适配器主线** ([`pose_control/qwen21_pro/`](pose_control/qwen21_pro/))：
   - 继承官方 16 个原生控制块 (4096维) + 14.6M FP32 ProTaskAdapter，零初始化安全相加，独立外观 Bank (`visible_owner`)，显式双人接触关系与两阶段局部精修工厂。
4. **官方 ControlNet Union 适配** ([`pose_control/qwen21_union/`](pose_control/qwen21_union/))：
   - 官方 Qwen-Image-2.1-Fun-Controlnet-Union 权重加载、契约审计与 pipeline 封装。

---

## 三、与外部数据盘及上下层关系

- **模型依赖**：基座与提取器实体位于独立数据盘 `/data/shangguanrz/pic-edit-models/`，通过 `models/` 软链透明读取。
- **训练输出**：权重检查点与训练日志直接落盘于 `/data/shangguanrz/pic-edit-experiments/`。
- **流水线复用**：`workflow/` 与 `workflow-multiple/` 统一通过 `import src.pose_edit.*` 复用几何、投影与提示词契约。
