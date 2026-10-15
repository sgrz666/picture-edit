# experiments/qwen-2.1-test/data/ —— Qwen2.1 编辑验证台的自备测试/候选数据

> 路径：`…/qwen-2.1-test/data`　|　子目录 `testset/`、`interaction/`、`candidates/`、`experiment/`　|　2026-10-12

## 一、干什么
`experiment/` 消费的**小规模本地测试集与候选素材**，与 Hi4D/iPER 全量数据集相互独立；用人工挑帧/配对的方式构造单人姿态迁移对与双人交互对。

## 二、层级地图
| 子目录 | 装什么 | 内容实证 |
| :--- | :--- | :--- |
| `testset/` | 单人姿态迁移测试集 | `manifest.json` + `pairs/hard_01…05`、`medium_01/02` + `A_heavy`/`B_slim`（体型对照源图） |
| `interaction/testset/` | 双人交互测试集 | 12 条 `inter_01_pair00_taichi`…`inter_12_pair02_pose`（taichi/hug/dance/backhug/pose，来自 pair00/02） |
| `candidates/` | 素材挑选中间态 | `heavy/`、`slim/`、`_rejected/`（淘汰）、`contact_sheets/`（挑选拼版） |
| `experiment/` | 镜像的占位/复制子树 | `code/`、`results/`（与上层同名，早期放置，内容以 `../experiment/` 为准） |

## 三、注意
- 编号叶子目录（`hard_01/`、`inter_01_*/`）内部为成对源/目标图 + 掩码 + 提示词的固定布局，不逐个展开。
- `data/` 是私有小样本，不代表 Hi4D/iPER 全量；正式训练数据在 `datasets/Hi4D_pilot_v1/`、`datasets/iPER/`。
- `data/experiment/` 与顶层 `experiment/` 命名重复，疑为早期搬迁遗留副本，以 `../experiment/` 为权威。
