# data/ —— 外部数据盘 `/data/shangguanrz` 映射与资产导航

> **核心原则**：
> 1. 本项目所有大型数据集（463GB+）、基座/外挂模型（88GB+）及大体积实验产物均持久化存储于宿主机独立数据盘 `/data/shangguanrz`（22TB ext4 分区，挂载于 `/dev/sdb1`）。
> 2. 项目工作区目录（`data/`、`models/`、`datasets/` 等）通过**同名软链接直达数据盘**，实现代码路径零修改。
> 3. 本目录下的所有软链被 `.gitignore`（`/data/*`, `!/data/README.md`）排除，不入 Git 仓库。

---

## 一、数据盘顶层目录导航与物理落点总表

实测数据更新至 2026-10-27（含 10-25 跨盘迁移的 54GB 权重与最新工作流）：

| 仓库/引用路径 | 数据盘物理落点 (`/data/shangguanrz/`) | 实体体积 | 类别性质 | 核心功能与内容描述 |
|---|---|---|---|---|
| `datasets/` (软链入口) | `/data/shangguanrz/pic-edit-datasets` | 463 GB | 数据集本体 | 包含 Hi4D-origin (359G 原始压缩包) 与 TikTokDataset (105G 全构建层) |
| `models/` (软链入口) | `/data/shangguanrz/pic-edit-models` | 88 GB | 核心模型库 | Qwen 2.1 基座、PE-I2I 提示词模型、GGUF 量化版及 10-25 迁入的 5 组 repo-models |
| `experiments/` (部分软链) | `/data/shangguanrz/pic-edit-experiments` | 38 GB | 核心实验区 | Qwen21 Pro 适配器 (`qwen21_pro_adapter_v1`)、Native V5 固定样本实验、Union 迁移、8-arm 实验 |
| `workflow-multiple/` (产物) | `/data/shangguanrz/pic-edit-workflow-multiple` | 动态 | 流水线产物 | A+B+C 双源双人交互姿态迁移流水线产物与中间数据 |
| `workflow-v2/` (产物) | `/data/shangguanrz/pic-edit-workflow-v2` | 动态 | 流水线产物 | 双人交互动作迁移二期流水线工作区 |
| `HIA/` | `/data/shangguanrz/HIA` | 动态 | 交互资产 | Human Interaction Assets，高精度双人交互切片、face/hand 质检子集 |
| `pic-edit-caches/` | `/data/shangguanrz/pic-edit-caches` | 9.6 GB | 特征缓存 | 包含 1390 个 Hi4D 双人骨架与交互特征缓存 (`.pt`)，免重复提取 |
| `pic-edit-archive/` | `/data/shangguanrz/pic-edit-archive` | 95 GB | 历史归档 | 冷备归档（含旧 20B 模型、历史 runs 备份、Git bundle 快照、清理日志） |
| `pic-edit-audits/` | `/data/shangguanrz/pic-edit-audits` | 150 MB | 审计归档 | 跨盘迁移基线双端 sha256 清单及空间清理现场留档 |
| `pic-edit-runtime/` | `/data/shangguanrz/pic-edit-runtime` | 动态 | 运行时中转 | 运行期临时张量挂载与进程间中转 |
| `dataset_samples/` | `/data/shangguanrz/dataset_samples` | 7.0 GB | 调研样例 | 候选数据集选型切片 (DNA-Rendering, Freeman, MVHumanNet 等) |
| `route_c_prompt_planner/` | `/data/shangguanrz/route_c_prompt_planner` | 19 GB | 推理服务 | 提示词规划器实验区与 Qwen3.5-9B 全量备份模型权重 |
| `tools/` | `/data/shangguanrz/tools` | 1.8 GB | 推理引擎 | llama.cpp (b11617 CUDA 12.8) 高性能推理二进制工具链 |
| `models/home-models` | `/data/shangguanrz/models/home-models` | 15.3 GB | 家目录模型 | 10-25 迁移自 `~/models` 的模型文件（原位保留软链） |
| `nihaochen/` | `/data/shangguanrz/nihaochen` | 22.3 GB | 训练权重/切片 | 10-25 迁移自 `~/nihaochen/` 的 tiktok_qwen21_full (10.9G) 与 iper1024 (11.4G) |

---

## 二、模型资产详细导航 (Models)

### 1. 基座与提示词模型 (`pic-edit-models/`)
- `Qwen-Image-2.1/` (31 GB)：阿里官方 7B 单流 DiT 图像编辑基座（`transformer/`, `text_encoder/`, `vae/`, `scheduler/`）。
- `Qwen-Image-2.1-PE-I2I/` (18 GB)：官方 9B 图像到图像提示词重写增强模型（4 个 safetensors 分片）。
- `Qwen-Image-2.1-PE-I2I-GGUF/` (10 GB)：PE-I2I 的 Q8_0 量化版 + bf16 mmproj 投影头。
- `Qwen-Image-2.1-PE-MTP-Q4_K_M/` (6.5 GB)：PE-I2I 的 Q4_K_M 低显存量化版。
- `Qwen3-VL-4B-Instruct-GGUF/` (3.2 GB)：4B 多模态指令模型（快速图像理解）。

### 2. 10-25 迁入的仓库工具模型 (`pic-edit-models/repo-models/`)
原位于 `project/models/`，为释放根盘空间迁入 `/data`，原位留软链，双端哈希逐文件全等：
- `DeepGen-1.0-diffusers/` (14.4 GB)：2.47B MMDiT 基座模型。
- `antelopev2/` (4.97 GB)：InsightFace 人脸检测与 ArcFace 512 维特征提取模型。
- `DWPose/` (351 MB)：`yolox_l.onnx` 与 `dw-ll_ucoco_384.onnx` 骨骼关键点提取权重。
- `face_eval/` (356 MB)：面部评估检测器与对齐权重。
- `smpl/` (78 MB)：SMPL/SMPL-X 官方人体网格模型文件（`SMPLX_NEUTRAL.pkl`）。

---

## 三、数据集资产详细导航 (Datasets)

- `pic-edit-datasets/Hi4D-origin/` (359 GB)：Hi4D 官方 32 个原始 `tar.gz` 分包（保持压缩，按需切片至 `datasets/Hi4D_pilot_v1/`）。
- `pic-edit-datasets/TikTokDataset/` (105 GB)：
  - `TikTok_dataset/` (48 GB)：340 序列原始高清晰抽帧。
  - `TikTok_3d_assets_native/` (39 GB)：Native 分辨率下 3D SMPL-X 几何资产与条件包。
  - `TikTok_v3/` (761 MB)：v3 正式构建层（train 8574 对 / val 1120 对 / test 1156 对清单、DWPose 关键点、人脸特征缓存）。
- `HIA/`：双人高难度接触交互切片库与质量过滤集。
- `dataset_samples/abc_showcase_20261003/` (7.0 GB)：DNA-Rendering、Freeman、HumanVid 等选型评测样本。

---

## 四、核心实验与产物落点导航 (Experiments)

位于 `/data/shangguanrz/pic-edit-experiments/`：
- `qwen21_pro_adapter_v1/`：Qwen21 Pro 适配器训练与验证区。
  - 核心架构：16 个原生控制块 (4096维) + 14.6M FP32 ProTaskAdapter。
  - 关键检查点：`runs/pilot_main200_v4_20261018T161055/checkpoint_000200/`。
  - 评测证据：Face flow loss -31.86%, Contact -64.37%, Hand -29.44%。
- `qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1/`：Native V5 单人固定样本 300 步训练区（含过程帧与消融）。
- `qwen21_hi4d_dual/hi4d-fixedpair-v5_20261025_v1/`：Native V5 双人固定样本 300 步训练区（含接触几何与消融）。
- `qwen21_union_migration_v1/` (7.3 GB)：Qwen-Image-2.1-Fun-Controlnet-Union 官方权重 (7.1 GB) 与 180 键验证流水线。
- `pei2i_body_face_8arm_20261009/` (18 GB)：8-arm 联合提示词与多分支控制消融实验。
