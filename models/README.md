# models/ —— 预训练基座与外挂权重库

> **核心原则**：
> 1. 本目录存放项目依赖的生成基座与预训练提取器，总计约 **88 GB**，属于大体积二进制资产，不入 Git 仓库。
> 2. **物理实体全部位于独立数据盘 `/data/shangguanrz/`**，本目录下的条目均为软链接，透明支持绝对路径与相对路径读取。
> 3. 训练产出的 checkpoint 权重一律保存在各实验对应的目录中，不放在本目录。

---

## 一、模型资产清单与物理落点导航

| 模型条目 | 真实物理落点 (`/data/shangguanrz/`) | 实体体积 | 类别与作用 | 关键调用模块 |
|---|---|---|---|---|
| `Qwen-Image-2.1/` | `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1` | 31 GB | 阿里 7B 单流 DiT 图像编辑与生成基座 | `src/pose_control/qwen21*`, `workflow/` |
| `DeepGen-1.0-diffusers/` | `/data/shangguanrz/pic-edit-models/repo-models/DeepGen-1.0-diffusers` | 14.4 GB | 2.47B MMDiT 双流生成基座（10-25 迁入） | `src/pose_control/v6/`, `scripts/train/train_iper_v6*` |
| `antelopev2/` | `/data/shangguanrz/pic-edit-models/repo-models/antelopev2` | 4.97 GB | SCRFD 人脸检测与 ArcFace 512D 提取器（10-25 迁入） | `src/evaluation/`, `src/pose_control/*/face*` |
| `DWPose/` | `/data/shangguanrz/pic-edit-models/repo-models/DWPose` | 351 MB | ONNX 姿态估计器 (`yolox_l`, `dw-ll_ucoco_384`)（10-25 迁入） | `src/evaluation/dwpose_onnx/`, `workflow/` |
| `face_eval/` | `/data/shangguanrz/pic-edit-models/repo-models/face_eval` | 356 MB | 面部相似度评测专用模型（10-25 迁入） | `src/evaluation/` |
| `smpl/` | `/data/shangguanrz/pic-edit-models/repo-models/smpl` | 78 MB | 官方人体拓扑与网格模型 (`SMPLX_NEUTRAL.pkl`)（10-25 迁入） | `workflow/pose_transfer/`, `src/controls/` |
| `Qwen-Image-2.1-PE-I2I/` | `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I` | 18 GB | 官方 9B 图像到图像提示词重写增强模型 (Prompt Enhancement) | `workflow/pe_worker.py` |
| `Qwen-Image-2.1-PE-I2I-GGUF/` | `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I-GGUF` | 10 GB | PE-I2I 的 Q8_0 高精度量化版 + bf16 视觉投影头 | `workflow/pe_worker.py` (llama.cpp 引擎) |
| `Qwen-Image-2.1-PE-MTP-Q4_K_M/` | `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-MTP-Q4_K_M` | 6.5 GB | PE-I2I 的 Q4_K_M 中精度低显存量化版 | 低显存环境备用 |
| `Qwen3-VL-4B-Instruct-GGUF/` | `/data/shangguanrz/pic-edit-models/Qwen3-VL-4B-Instruct-GGUF` | 3.2 GB | 4B 多模态视觉-语言模型 GGUF 版本（快速图像理解） | 提示词规划与轻量视觉审计 |
| `home-models/` | `/data/shangguanrz/models/home-models` | 15.3 GB | 10-25 迁移自 `~/models` 的家目录模型集合 | 离线环境与开发调用 |

---

## 二、2026-10-25 跨盘迁移说明

为解决系统根盘 `/`（1.8TB 占用率 99%）空间紧缺问题，2026-10-25 将仓库内 5 组模型（`DeepGen-1.0-diffusers`、`antelopev2`、`DWPose`、`face_eval`、`smpl`）整体迁移至数据盘 `/data/shangguanrz/pic-edit-models/repo-models/`：
- **一致性**：源端与目标端逐文件 SHA256 **8/8 组全等**（详见 `docs/模型跨盘迁移执行记录_20261025.md`）。
- **零代码改动**：在 `models/` 原位建立同名软链接回填，所有代码直接访问 `models/<name>` 保持完全兼容。

---

## 三、基座加载验证示例

```python
import torch
from diffusers import DiffusionPipeline

# 加载 DeepGen-1.0（软链直达 /data 实体）
pipe = DiffusionPipeline.from_pretrained(
    "models/DeepGen-1.0-diffusers",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

# 加载 Qwen-Image-2.1（位于 /data/shangguanrz/pic-edit-models/Qwen-Image-2.1）
# 可直接通过绝对路径或软链调用
```
