# datasets/ —— 训练与评测数据集工作区

> **核心原则**：
> 1. 本目录为人物姿态编辑系统的数据集工作区，实体与软链接并存。
> 2. 大体积、不可再生的官方原始数据（463GB+）存放于独立数据盘 `/data/shangguanrz/pic-edit-datasets/`，以软链接接入本目录。
> 3. 项目自产的处理切片、几何条件包与配对清单留在仓库实体目录，被训练脚本直接引用。

---

## 一、数据集资产目录导航与物理落点

| 数据集条目 | 形态 | 物理落点 (`/data/shangguanrz/` 或本地) | 体积 | 任务定位与内容描述 | 关联模块 |
|---|---|---|---|---|---|
| `TikTokDataset/` | 软链 | `/data/shangguanrz/pic-edit-datasets/TikTokDataset` | 105 GB | 单人全身舞蹈动作主线数据集（抽帧 48G + 3D 资产 39G + v3 构建清单 761M） | `src/data/tiktok_dataset.py`, `scripts/train/run_qwen21_v5_fixedpair.py` |
| `Hi4D-origin/` | 软链 | `/data/shangguanrz/pic-edit-datasets/Hi4D-origin` | 359 GB | Hi4D 官方原始 32 个压缩分包（`pair00_1.tar.gz` ~ `pair28.tar.gz`），保持未解压 | 原始备份 |
| `Hi4D_pilot_v1/` | 实体 | `datasets/Hi4D_pilot_v1/` | 2.5 GB | Hi4D 双人交互 Pilot 定稿切片：SMPL-X 配准几何标注、512 条件资产与配对清单 | `src/data/`, `workflow-multiple/` |
| `iPER/` | 实体 | `datasets/iPER/` | 12 GB | 国际动作驱动基准数据集：包含 `raw_1024`、`sampled_6src64tgt` 评测集与 `assets_512` 条件包 | `src/data/base_dataset.py`, `scripts/train/train_iper_*` |
| `3dpw_single_final_v2/` | 实体 | `datasets/3dpw_single_final_v2/` | 1.1 GB | 户外极限大角度旋转与遮挡评测集：60 组极限动作评测对与五路对齐条件 | `scripts/eval/` |
| `champ_motions_example/`| 实体 | `datasets/champ_motions_example/`| 180 MB | CHAMP 官方动作驱动序列库（只含姿态运动，不含图像配对） | `third_party/champ_guidance/` |
| `iPER_sample/` | 实体 | `datasets/iPER_sample/` | 45 MB | iPER 最小样例切片，专供本地/CI 冒烟与格式契约核对 | 冒烟测试 |
| `HIA/` (外部) | 数据盘 | `/data/shangguanrz/HIA/` | 动态 | Human Interaction Assets：高精度双人交互紧密接触切片与手脸质检集 | `docs/superpowers/plans/2026-10-26-hia-*` |
| `dataset_samples/` (外部) | 数据盘 | `/data/shangguanrz/dataset_samples/` | 7.0 GB | 调研候选数据集切片（DNA-Rendering, Freeman, MVHumanNet, HumanVid 等） | 调研选型 |

---

## 二、数据集层级与调用关系

1. **训练与数据加载**：
   - 数据加载器位于 [`src/data/`](../src/data/)（`tiktok_dataset.py`, `base_dataset.py`, `augmentation.py`）。
   - 离线特征提取与清洗脚本位于 [`scripts/data/`](../scripts/data/)（如 `prepare_qwen21_fixedpair_v5.py`, `assemble_qwen21_fixedpair_v5.py`）。
2. **软链一致性保证**：
   - 换机器或环境重置时，仅需更新本目录指向 `/data/shangguanrz/` 的软链接，清单文件中的路径将自动无缝映射，禁止直接改写清单内的绝对路径。
