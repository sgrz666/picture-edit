# src/pose_control/qwen21_full/ —— Qwen-Image-2.1 全量控制与 Native V5 体系

`qwen21_full/` 是面向 Qwen-Image-2.1（7B 单流 DiT）的全量 32 层控制架构体系。

---

## 一、核心结构与模块导航

| 目录 / 模块 | 定位与职责 | 关键特性 / 说明 |
|---|---|---|
| [`native_v5/`](native_v5/README.md) | **最新 Native Full V5 模块化训练与推理包** | 显式 Geometry / Interaction 分流、Source 局部 Detail、Face / Hand 独立模态、六组强度与 sigma 调度、模块化训练引擎（`training/`）与无检测器 ROI 监督 |
| `qwen_adapter.py` | 32 层单流 DiT 控制上下文管理器 | `Qwen21FullAdapter`，将 32 层分为 6 组残差注入，支持 geometry, interaction, detail, face, hand 五分支 |
| `qwen_runtime.py` | 原生 Qwen I/O 与画布坐标管理 | 负责原生分辨率及 `resize_v1` 预处理与 Token 空间坐标映射 |
| `execution.py` | 早期全量训练循环与参数分组 | 支持多种优化器与学习率配置，管理分支启用与检查点保存 |
| `joint_training.py` | 联合训练调度 | 在保持继承架构前提下执行 Body/Face 联合优化与状态断言 |
| `data_runtime.py` | 数据与特征运行时缓存接口 | `FaceConditionStore`、`HandConditionStore`、`QwenFullDataset`，构建统一条件包 |
| `face_supervision.py` | 目标人脸离线合格性门控 | SCRFD 检测框对齐与布尔损失掩膜生成（不读取目标人脸外观） |
| `face_loss_alignment.py` | 微观人脸对齐损失 | 5 点仿射变换与可微空间旋转对齐损失 |
| `token_encoders.py` | 双人 Token 绑定器 | `PersonTokenBinder`，双人空间解耦与 slot 顺序绑定 |

---

## 二、Native V5 子系统重点概览 ([`native_v5/`](native_v5/README.md))

Native V5 是目前 32 层控制架构的主推演进版本：
- **`training/`**：包含独立解耦的训练组件（`trainer.py`, `backend.py`, `config.py`, `data.py`, `observations.py`），提供显式损失权重配置。
- **`source_bank.py`**：多参考图独立外观池，彻底避免目标人物与参考人物的外观特征混合。
- **`conditioning.py` & `frontend.py`**：几何与部位特征编码。
- **`detail.py` & `local.py`**：局部高频服饰纹理与面部/手部稀疏 Query 注入。
- **检查点与训练产物**：落地于 `/data/shangguanrz/pic-edit-experiments/` 下的 `tiktok-fixedpair-v5_20261025_v1` 与 `hi4d-fixedpair-v5_20261025_v1`。
