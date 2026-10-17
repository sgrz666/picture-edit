# data/ —— 外部数据盘 `/data/shangguanrz` 映射与数据资产目录

> **核心原则**：
> 1. 本项目的所有大型数据集（463G+）与基座/微调大模型（68G+）均持久化存储于宿主机独立数据盘 `/data/shangguanrz`（22TB 分区，挂载于 `/dev/sdb1`）。
> 2. 本目录（`/home/shangguanrz/project/pic-edit/data/`）仅作为**文件夹名字搬运与符号链接索引**，**绝对不进行实体大文件拷贝与迁移**。
> 3. 本目录下的所有软链被 `.gitignore` 规则（`/data/*`, `!/data/README.md`）严格忽略，不计入 Git 仓库追踪。

---

## 一、数据盘顶层文件夹搬运与映射总表

以下文件夹名字已直接搬运并建立符号链接直达 `/data/shangguanrz`，实测数据如下（2026-10-17 磁盘核对）：

| 搬运文件夹名 (软链名称) | 真实物理落点 (`/data/shangguanrz/`) | 实体体积 | 类别性质 | 核心功能与内容描述 |
|---|---|---|---|---|
| `pic-edit-datasets` | `/data/shangguanrz/pic-edit-datasets` | 463 GB | 数据集 | 训练与评测核心数据集本体（Hi4D 原始包 + TikTok 全量构建层） |
| `pic-edit-models` | `/data/shangguanrz/pic-edit-models` | 68 GB | 核心模型 | Qwen-Image-2.1 7B 基座、PE-I2I 原生与 GGUF 量化系列模型 |
| `pic-edit-experiments` | `/data/shangguanrz/pic-edit-experiments` | 29 GB | 独立实验 | 独立闭环实验区（含 ControlNet Union 权重、8-arm 实验、低延迟探针） |
| `pic-edit-caches` | `/data/shangguanrz/pic-edit-caches` | 9.6 GB | 运行缓存 | 双人 Hi4D 姿态控制特征张量缓存（1390 个 `.pt`，免重复提取） |
| `pic-edit-archive` | `/data/shangguanrz/pic-edit-archive` | 95 GB | 历史归档 | 冷备归档（含旧 20B 模型、历史 runs 备份、Git bundle 快照） |
| `dataset_samples` | `/data/shangguanrz/dataset_samples` | 7.0 GB | 调研样例 | 候选数据集选型样例（DNA-Rendering, Freeman, MVHumanNet 等） |
| `route_c_prompt_planner`| `/data/shangguanrz/route_c_prompt_planner` | 19 GB | 推理服务 | 提示词规划器实验区与 Qwen3.5-9B 全量备份模型权重 |
| `tools` | `/data/shangguanrz/tools` | 1.8 GB | 推理引擎 | llama.cpp (b11617 CUDA 12.8) 高性能推理二进制工具链 |
| `Hi4D-origin` | `/data/shangguanrz/pic-edit-datasets/Hi4D-origin` | 359 GB | 数据集 | 【直达软链】Hi4D 官方 32 个未解压原始分包（双人交互核心数据） |
| `TikTokDataset` | `/data/shangguanrz/pic-edit-datasets/TikTokDataset` | 105 GB | 数据集 | 【直达软链】单人舞蹈动作全流程数据集（抽帧+3D资产+v3构建清单） |
| `Qwen-Image-2.1` | `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1` | 31 GB | 基座模型 | 【直达软链】阿里 7B 单流 DiT 图像编辑与生成基座模型 |
| `Qwen-Image-2.1-PE-I2I`| `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I` | 18 GB | 提示词模型 | 【直达软链】官方 9B 图像到图像提示词重写增强模型 (Prompt Enhancement) |
| `Qwen-Image-2.1-PE-I2I-GGUF` | `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I-GGUF` | 10 GB | 量化模型 | 【直达软链】PE-I2I 的 Q8_0 GGUF 高精度量化模型（含 bf16 视觉投影视角） |
| `Qwen-Image-2.1-PE-MTP-Q4_K_M` | `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-MTP-Q4_K_M` | 6.5 GB | 量化模型 | 【直达软链】PE-I2I 的 Q4_K_M 中精度低显存模型 |
| `Qwen3-VL-4B-Instruct-GGUF` | `/data/shangguanrz/pic-edit-models/Qwen3-VL-4B-Instruct-GGUF` | 3.2 GB | 多模态模型 | 【直达软链】超轻量 4B 多模态指令模型 GGUF 版本（用于低延迟图文理解） |
| `Qwen-Image-2.1-Fun-Controlnet-Union` | `/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/models/Qwen-Image-2.1-Fun-Controlnet-Union` | 7.1 GB | 控制网络 | 【直达软链】官方 ControlNet Union 单模型多条件控制权重 |

---

## 二、数据集资产详细说明 (Datasets - 463 GB + 7.0 GB)

### 1. `Hi4D-origin/` (359 GB)
- **定位**：双人高精度交互姿态与动作数据集的官方原始数据源。
- **内容构成**：
  - 包含 32 个官方原始 `tar.gz` 压缩包（如 `pair00_1.tar.gz` ~ `pair28.tar.gz`），每个包约 11~15 GB。
  - 附带官方下载清单 `hi4d_files.txt` 与 `hi4d_urls.txt`。
- **使用说明**：
  - 原始包保持未解压状态（避免展开后占用 445GiB 且视频帧占比超 80%）。
  - 训练与评测所用的配准几何标注和切片位于项目工作区 `datasets/Hi4D_pilot_v1/`。
  - 数据模型采用 SMPL 标准（不含手脸精细网格），与主线 SMPL-X 口径在接入时通过桥接适配器转换。

### 2. `TikTokDataset/` (105 GB)
- **定位**：单人全身舞蹈动作姿态迁移的核心基准数据集，提供完整的各处理阶段切片。
- **内部各层结构**：
  - `TikTok_dataset/` (48 GB)：按编号抽取的视频序列高清晰度图像帧（`00001` ~ `00340`）。
  - `TikTok_3d_assets_native/` (39 GB)：Native 原生分辨率下的 3D 人体几何与姿态资产包（每序列包含对应的 SMPL-X 参数与条件特征）。
  - `showcase_single/` (16 GB)：单人姿态迁移评测展示切片（如 `sample_01_00001_0014_to_0074`）。
  - `TikTok_Raw_Videos/` (2.5 GB)：原始采集的 mp4 视频文件。
  - `TikTok_v3/` (761 MB)：**v3 正式构建层**，包括：
    - `manifests/` (401 MB)：严格划分的配对清单（train 8574 对 / val 1120 对 / test 1156 对）；
    - `dwpose_v1/` (275 MB)：提取的 DWPose 人体与手部关键点；
    - `qwen21_face_feature_cache/` (58 MB)：面部特征向量缓存；
    - `qc/` (29 MB)：质检指标分布；
    - `stats.json`：340 序列 / 92961 帧详细统计数据。
  - `model/` (665 MB)：TikTok 预处理自用的辅助检测与关键点提取模型。
  - `TikTok_v2/` (252 MB)：历史 v2 构建层留存备查。

### 3. `dataset_samples/` (7.0 GB)
- **定位**：多数据集调研选型 showcase 与官方代码样例库。
- **内部构成**：
  - `abc_showcase_20261003/` (7.0 GB)：候选数据集样例切片，涵盖 `dna_rendering` (18M)、`freeman` (6.8M)、`humanvid` (34M)、`mvhumannet`、`pku_dymvhumans`，并附有详尽的选型报告。
  - `MVHumanNet_repo/` (3.0 MB) & `MVHumanNet++_repo/` (8.1 MB)：多视角人体数据集的官方数据格式说明与可视化脚本。

---

## 三、模型资产详细说明 (Models - 68 GB + 19 GB + 7.1 GB)

### 1. `Qwen-Image-2.1/` (31 GB)
- **定位**：通义千问 2.1 图像生成与编辑基座大模型（7B 单流 DiT 架构）。
- **包含组件**：
  - `transformer/` (14 GB)：DiT 主干网络权重；
  - `text_encoder/` (17 GB)：文本多模态理解与 Prompt 编码器；
  - `vae/` (1.3 GB)：图像编解码变分自编码器；
  - `processor/` & `scheduler/`：图像前处理与 DDIM/Flow 调度器。

### 2. `Qwen-Image-2.1-PE-I2I/` (18 GB)
- **定位**：官方发布的 Prompt Enhancement Image-to-Image（图像到图像提示词重写增强模型，9B 规模）。
- **组件**：包含 4 个 `safetensors` 分片与索引，用于在 workflow 第二阶段根据参考图与动作指令重写高质量正负提示词。

### 3. `Qwen-Image-2.1-PE-I2I-GGUF/` (10 GB) 与 `Qwen-Image-2.1-PE-MTP-Q4_K_M/` (6.5 GB)
- **定位**：PE-I2I 模型的量化部署版本，配合 `tools/llama-cpp` 实现极致低延迟和低显存占用。
- **包含文件**：
  - `qwen3.5_9b_qwen_image_2.1_pe_i2i.mtp.Q8_0.gguf` (9.1 GB)：Q8_0 高精度量化模型；
  - `qwen3.5_9b_qwen_image_2.1_pe_i2i.mtp.Q4_K_M.gguf` (5.6 GB)：Q4_K_M 中精度极速模型；
  - `qwen3.5_9b_qwen_image_2.1_pe_i2i.mmproj.bf16.gguf` (879 MB)：多模态视觉投影头。

### 4. `Qwen3-VL-4B-Instruct-GGUF/` (3.2 GB)
- **定位**：超轻量 4B 多模态视觉-语言模型 GGUF，支持低显存推理与快速图像语义理解。
- **包含文件**：
  - `Qwen3VL-4B-Instruct-Q4_K_M.gguf` (2.3 GB)；
  - `mmproj-Qwen3VL-4B-Instruct-F16.gguf` (797 MB)。

### 5. `Qwen-Image-2.1-Fun-Controlnet-Union` (7.1 GB)
- **物理路径**：`/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/models/Qwen-Image-2.1-Fun-Controlnet-Union/`
- **定位**：阿里官方发布的 Qwen 2.1 ControlNet Union 多合一多条件控制权重。
- **状态**：`Qwen-Image-2.1-Fun-Controlnet-Union.safetensors` (7.1 GB，180 个键全部完整通过审计)。

### 6. `Qwen3.5-9B_modelscope` (19 GB)
- **物理路径**：`/data/shangguanrz/route_c_prompt_planner/models/Qwen3.5-9B_modelscope`
- **定位**：Route C 提示词重写规划器的完整全精度模型权重。

---

## 四、实验区、推理环境与特征缓存

### 1. `pic-edit-experiments/` (29 GB)
- `qwen21_union_migration_v1/` (7.3 GB)：ControlNet Union 官方迁移与验证区，内置官方权重与评测流水线。
- `qwen21_union_dataset_adaptation_v1/` (3.1 GB)：ControlNet Union 与多动作数据集结合的 LoRA 训练与适配实验。
- `pei2i_body_face_8arm_20261009/` (18 GB)：8-arm PE-I2I 联合提示词与多分支控制消融实验区（含独立运行缓存）。
- `pe_i2i_gguf_selection_20261014/` (4.1 MB)：GGUF 量化精度与时延量化对比报告。
- `lowlatency_pe_probe_20261016/` (576 KB)：低延迟推理探针实测日志。

### 2. `pic-edit-caches/` (9.6 GB)
- `qwen21_hi4d_controls/` (9.6 GB)：包含 1390 个已提取的 Hi4D 双人骨架与交互特征缓存文件（`control_<hash>.pt`），大幅节省训练期前处理时间。

### 3. `tools/` (1.8 GB)
- `llama-cpp/` (1.8 GB)：编译完成的 `llama-b11617` 与适配 CUDA 12.8 的高性能推理二进制包，负责驱动所有 GGUF 模型。

---

## 五、历史归档与安全隔离区

### 1. `pic-edit-archive/` (95 GB)
- 只读历史归档库：
  - `cleanup_20261003/` (95 GB)：包含早期 20B 模型（`Qwen-Image-Edit-20B` 54 GB）、历史长程实验冷备（37 GB）与 3DPW 原始切片（4.7 GB）。
  - `workflow_runs_backup_20261012/` (64 MB)：workflow 重构前的基线产物备份。
  - `pic-edit_sg-main_20261012.bundle` & `20261013.bundle` (各约 123 MB)：主仓库完整镜像。

### 2. 回收站隔离区（待确认后可直接清空，共释放 ~258 GB）
- `_trash_ckpt_20261017/` (118 GB)：2026-10-17 空间清理时从 `/home` 盘迁移出的早期多余 checkpoint 权重（含 nihaochen 99G 与 project 20G）。
- `_trash_ckptB_20261016/` (116 GB)：2026-10-16 隔离的冗余 checkpoint 历史文件。
- `_trash_ckpt_20261013/` (24 GB)：2026-10-13 隔离的历史临时检查点。

---

## 六、本地合成调试样本 (Debug Samples - 兼容说明)

如果开发环境处于无 GPU 或无网络连接的纯本地环境，可通过以下方式生成轻量调试样本（129 MB）：
- 路径：`data/debug_samples/`
- 生成命令：`python create_debug_samples.py`
- 用途：仅用于验证代码接口调用、张量形状与梯度链路，不可用于正式效果评测。
