# src/pose_control/ —— 姿态控制器架构导航

`src/pose_control/` 是 pic-edit 的核心控制网络目录，承载将参考人物姿态与表情重定向到目标人物的所有可训练网络与受控推理逻辑。

---

## 一、控制器技术线导航

| 目录 | 对应基座 | 架构形态 | 核心职责与关键特性 | 检查点落点 |
|---|---|---|---|---|
| [`v6/`](v6/README.md) | DeepGen-1.0 (2.47B MMDiT) | Unified SMPL-X Adapter V6 | 循环控制核 + 残差注入（block 0–22）；演进为 V6.4 detail / V6.5 face / V6.6 hand 三分支 | `experiments/1.deepgen experiment/` |
| [`qwen21/`](qwen21/README.md) | Qwen-Image-2.1 (7B DiT) | 扁平简化适配器 | 独立 body / face / dual 控制头，注入 6 个固定层（3, 8, 13, 18, 23, 28），轻量验证 | `experiments/3.singel man/` |
| [`qwen21_full/`](qwen21_full/README.md) | Qwen-Image-2.1 (7B DiT) | 32 层全量单流 DiT 控制 | 32 层残差注入、目标面部对齐损失、双人 `PersonTokenBinder`。<br>**包含最新 [`native_v5/`](qwen21_full/native_v5/) 模块化训练包** | `/data/shangguanrz/pic-edit-experiments/` (tiktok & hi4d fixedpair-v5) |
| [`qwen21_pro/`](qwen21_pro/) | Qwen-Image-2.1 (7B DiT) + 官方 Union | 16 块官方 Union + 零初始化 Adapter | 继承 16 个原生 4096 维控制块；在 stage 0,4,8,12 注入 14.6M FP32 任务分支；`visible_owner` 硬隔离防串脸；两阶段局部精修 | `/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1/` (`checkpoint_000200`) |
| [`qwen21_union/`](qwen21_union/) | Qwen-Image-2.1 + 官方 Union | 官方 ControlNet 封装 | 官方 7.1GB 单模型多条件控制权重加载、Pipeline 桥接与 180 键完整性校验 | `/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/` |

---

## 二、最新架构重点：Native V5 与 Qwen21 Pro

### 1. `qwen21_full/native_v5/` (Native V5 体系)
- **定位**：解决旧版 32 层全量训练中损失不可控与单双人耦合问题。
- **模块结构**：
  - `training/`：模块化训练引擎（`trainer.py`, `backend.py`, `config.py`, `data.py`, `observations.py`），支持显式损失权重与可配置评测器。
  - `source_bank.py`：解耦人物外观池，支持独立 Source64 编码与多参考图协议。
  - `conditioning.py` & `frontend.py`：几何条件渲染与特征编码。
  - `generation.py` & `pipeline.py`：受控推理管线与上下文管理。

### 2. `qwen21_pro/` (Qwen21 Pro 体系)
- **定位**：解决官方 Union Inpaint 擦人脸与朴素 Hybrid 裸残差崩面的两难矛盾。
- **模块结构**：
  - `contracts.py`：强制数据契约与 Provenance 审计，禁止目标真实外观泄露进条件。
  - `adapter.py`：`ProTaskAdapter`（14.6M 参数，全量 FP32），零初始化保证 step 0 数学等价于官方发布模型。
  - `banks.py`：`PersonBankAttention`，结合 `visible_owner` 掩码截断跨人物注意力。
  - `encoders.py`：6 模态类型化几何编码 (`normal`, `depth`, `mask`, `visibility`, `contact`, `part`) 与显式接触编码。
  - `losses.py`：`regional_flow_loss`（软边界互斥分割为 face, contact, hand, body）。
  - `refinement.py`：Stage 2 局部 768px 高清面部与接触手部精修工厂。
  - `runner.py`：统一 CLI 入口（`prepare`, `cache`, `audit`, `smoke`, `train`, `infer`, `refine`, `status`）。
