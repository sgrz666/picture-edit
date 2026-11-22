# AGENTS.md — 协作智能体须知

这个仓库有明确的目录约定和历史包袱。改代码、开实验、建目录之前先读这篇。
权威层级说明在根 [`README.md`](README.md) 第 3 节；本文只讲**你动手时要遵守什么**。

---

## 1. 动手前

- **默认直接在 `main` 修改、验证和提交**（用户 2026-11-12 明确要求）。先核对并更新主线；不为普通任务另开 feature 分支或 worktree。现有 worktree 不删除，其他人的未提交改动不覆盖。
- **先读后写**。要改的文件先打开看一遍，不要凭目录名猜内容。
- **不信旧记录，以磁盘实测为准**。README、记忆、日志都可能过期；结论要有当前磁盘上的证据。
- **常规训练／测试默认使用当前架构和训练分支中最新、已完整提交且通过相应验收的 checkpoint**（用户 2026-11-21 要求）。开跑前报告所选路径、SHA256、来源／父链与训练阶段；不能只按文件mtime或跨分支的最大step号选权重。选定后整轮绑定同一SHA，不在运行中追随可变`latest`指针。
- **历史权重只用于明确标注的对照、消融、回归或显式初始化**。开始前说明锁定的历史起点及原因；对照各臂必须同父权重，评价使用各臂实际新产物。不能把最新代码等同于最新权重，或加载旧权重后称为最新。
- **迁移／扩任务前核对完整预训练祖先**，同时审计上游已完成主线（例如Bilibili20000步），列明哪些能力已继承、哪些未继承。支线时间最新不等于已接入全项目最新训练成果；不能跳过上游权重审计后直接扩大训练。继承结论须有实际SHA、checkpoint parent／initialization及迁移张量记录。
- **Native V7 系训练初始化统一使用 Bilibili20k→V7 迁移权重（用户 2026-11-22 指定）**：`/data/shangguanrz/pic-edit-experiments/two/qwen21_hi4d_dual/native-v7-singlepair-overfit-20261117-v1/initialization/bilibili20k-v7-dual192new.pt`（SHA256 `17b23faec7c39c1c8e0b99f4bd05b82bbfe646eb3695a9336459d44796c72331`；1054 键继承 Bilibili20k `d1a655a8…`，89 个 V7 新模块键取自双人 192 分支最终权重 `eca8f5ec…` 的训练值，迁移报告与脚本同目录）。此后新建 native v7 系训练默认从该权重初始化；双人 192 步分支（`eca8f5ec…`）不再直接作为扩大训练的整包初始化，只作为对照与其新模块训练来源。备选变体 `bilibili20k-v7.pt`（`f53ec451…`，89 键为 v7.pt 未训练初始态）仅用于标注的对照。产出新迁移产物时必须逐张量回读校验并通过 `NativeFullRuntimeV7.from_checkpoint` 严格重建后方可使用；每次使用前仍须核对磁盘上是否有更新的合格迁移产物。
- **Qwen基座、冻结evaluator和条件缓存继续按实验manifest固定版本／SHA**，不因Adapter更新而自动替换；更换时重新绑定缓存与门禁。Native V7近期交付权重与复用命令见[`Source身份／背景保持实验`](experiments/hi4d_qwen21_dual_20260930/native-v7-source-preservation-20261121/README.md)，每次仍须核对磁盘上是否有更新的合格产物。
- 任何删除/移动先做**只读审计 + dry-run**，列清单给人确认后再落地。
- 远端与本地会双向漂移。动远端前先 `git log` 核对，快照带时间戳。

---

## 2. 开一个新实验

### 2.1 先归族，再建目录

实验统一放 `experiments/` 下，**先看有没有对应的实验族目录，有就进去建，没有才新建族**。



> **⚠️ 两侧结构不一致（2026-11-21 实测）**：本地 `experiments/` 是「平铺每个实验目录 + README 索引」；
> 服务器 `sg:/home/shangguanrz/project/pic-edit/experiments/` 按技术线分 9 族 ——
> `1.deepgen experiment` / `2.qwen-test` / `3.singel man` / `4.two man ` / `5.workflow` / `6.qwen-controlnet union` / `7.tiktok` / `8.HIA` / `9.hi4d`（部分名字带空格与拼写错误，未清理），族外另有根级 `README.md` 与 `实验中文索引.md` 两个索引文件。
> **动手前先 `ls` 确认自己在哪一侧**，不要假设结构。族内归类的原则两边一致：按任务目标，不按时间或人名。
> **2026-11-21 远端清理**：跨族重复残根已删（`7.tiktok`/`9.hi4d` 下的残根；权威落点见各族 README「去重说明」）；`hia_person_pair_v3_20261103` 归入 `8.HIA/`；`code/` 目录撤销（`edit.py` 移入 `scripts/run/qwen21_edit.py`）；`scripts/` 根仅保留 4 个 `data/` 软链目标文件；根目录 README.md 各节标注了远端不落盘的目录。

**命名**：命名遵守易读容易理解,比如中文名称是"tiktok 脸手训练"英文名字就是"tiktok-face-hand-joint-train"`

### 2.2 实验目录内部结构

一个实验是**自包含**的：别人只靠这个目录 + README 就能复现。按需建，不要一次建全套空目录：

```
experiments/<你的实验>/
├── README.md            # 必须有，见第 4 节
├── input/               # 原始输入（参考图、待处理素材、参数文件）
├── source/              # 从别处拷来的源素材，保留原样不改写
├── output/              # 全部产物：生成图、指标、报告图
│   ├── images/
│   └── results.json
├── scripts/             # 只服务本实验的脚本
├── data/                # 本实验的数据集切片、pairs、缓存索引
├── logs/                # 运行日志
└── report/              # 汇总报告、面板图,docx
```

参照样本：`experiments/bfs_body_face_joint_pilot_20261008/`（大实验，子目录切得细）、
`experiments/qwen-2.1-test/`（中等实验，`data/` + `experiment/{code,results}` 两层）。

### 2.3 脚本放哪

- **只服务本实验** → 实验目录内的 `scripts/`
- **跨实验可复用** → 提升到仓级 `scripts/`，按职责进子目录：

```
scripts/
├── train/    # 正式训练入口
test/
├── eval/     # 量化评测入口
├── run/      # 单例推理与编排
├── data/     # 数据预处理与几何准备
└── audit/    # 架构与权重一致性审计
```

- **不要往仓库根目录散落脚本**。根目录只保留被 `tests/` import 的少数几个（`conftest.py`、`bfs_explore.py` 等）。临时探针脚本用完删掉，不要留 `_probe_*.sh` 这种痕迹。

### 2.4 如果你在改 workflow

`workflow/` 是独立的单人动作迁移流水线，有**自己的两套目录约定**，不要和 `experiments/` 混：

```
workflow/
├── run.sh / batch.sh          # 入口
├── run.py                     # 编排
├── smplx_worker.py / pe_worker.py / generate_worker.py   # 四阶段中的三个阶段
├── pose_transfer/             # 几何与提示词核心逻辑
├── tests/
├── runs/<run_name>/           # 每个 run 四阶段产物
│   ├── inputs/                #   prepare
│   ├── geometry/              #   SMPLest-X → DWPose 骨架
│   ├── prompt/                #   PE-I2I 重写
│   └── generation/            #   Qwen 基座出图
└── scripts/<工具名>/
    ├── script/                #   可执行脚本
    ├── output/                #   产物
    └── README.md
```

`runs/` 下每个 run 的 `stage.json` 记录输入 SHA256 与产物 SHA256，**改产物会被 `--verify` 发现**。
`workflow/scripts/six-picture-create/` 是工具目录的标准样板。

---

## 3. 目录职责速查

| 目录 | 放什么 | 不放什么 |
|---|---|---|
| `src/` | 核心源码（适配器、pipeline、条件构建） | 实验脚本、一次性探针 |
| `scripts/` | 跨实验的标准入口 | 某个实验私有的逻辑 |训练代码,测试代码,评估代码,放代码逻辑
| `experiments/` | 每个实验自包含的目录 | 共享代码、模型权重本体 |
| `workflow/` | 动作迁移流水线及其 runs | 训练代码、数据集 |
| `docs/` | 设计规范、审查与整改记录、调研报告 | 运行日志、指标 json |
| `reports/` | Word/PPT 汇报成果 | 中间产物 |
| `research/` | 专题调研与审计证据 | 正在跑的代码 |
| `tests/` | 契约与回归测试 | 手动跑的实验 |
| `datasets/` `models/` | 数据与权重本体（服务器路径，本地多为软链或说明） | 任何新写的文件 |


`datasets/` 与 `models/` 在服务器上，本地这份仓库里通常是软链或说明文档。**新产物默认落服务器的 `/data/shangguanrz/`**（`/` 盘常年紧张）。

---

## 4. README 怎么写

### 4.1 什么时候必须写

- 新建实验目录 → 写 `<实验>/README.md`
- 新建工具目录 → 写 `<工具>/README.md`
- 在已有目录里新增一类产物 → 更新该目录 README，不要新开一个

### 4.2 要写成什么样

**参考样板**：`experiments/qwen-2.1-test/README.md`（实验 README）、
[`workflow/README.md`](workflow/README.md)（流水线 README，两侧都有）。

这两个的共同点：**具体、有数字、能照着做**。四段结构：

1. **实验目的** —— 一两句说清要验证什么，不要写"本研究旨在…"
2. **里面有什么** —— 逐个文件/子目录说，写清楚它是什么、怎么用
3. **关键结论** —— 带具体数字（PSNR 从多少到多少、多少样本、什么阈值）
4. **与其他实验的关系** —— 上游是谁、喂给谁、取代了谁

### 4.3 不要写成什么样

❌ **生成层空话**。10-12 已经全仓清掉过一批，典型症状：

> - **`data/`**: 该子目录下的专门功能模块或产物。
> - **文件总数**: 3 个文件
> - 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。

这种句子没有任何信息量，等于没写。

❌ **正确的废话**。"提升了效果"、"具有重要价值"、"为后续研究奠定基础"——全部删掉，换成数字。

❌ **罗列用不到的文件**。README 是地图，不是 `ls` 输出。只写别人需要知道的。

❌ **过度结构化**。三四层嵌套标题 + 满页表格，不如三段实在话。

### 4.4 语言

中文。术语保留英文原文（如 `Adapter`、`PCK`、`stage.json`），不要自造译名。

---

## 5. 硬性红线

1. **图片/视频永不删移**（见第 1 节）。
2. **删除前先审计**：按「文件名 + 尺寸」查唯一性，先 dry-run，列清单，人确认后执行。
3. **沙箱/权限被拒即终止**，不绕。
4. **不静默改实验变量**。拷贝 run 目录做实验时，`request.json` 里的路径是**绝对路径**，不会随复制改写——不手动改的话，worker 会读写源目录，实验白做。检测标准：副本产物与对照的 SHA256 完全相同 = 副本没生效。
5. **不覆盖既有产物**。重跑加 `--force` 或换 `--name`，别把上一轮结果冲掉。
6. **大文件不进 Git**。`.pt` / `.safetensors` / 条件缓存默认忽略，服务器持久盘与归档盘保留。
7. **不向 `qwen21` 环境塞不兼容的包**。`diffusers 0.41.0.dev0` 是为 Qwen-Image-2.1 专门保留的版本，装包前先想清楚会不会破坏它。
8. **姿态评分统一使用 SDPose-Wholebody + YOLO11x**。范围包括在线 RL 的候选图 Pose/Hand Reward、训练门禁、留出集评测、单人/双人 workflow 的生成图评分，以及姿态结果汇总。先沿入口追到实际读图打分的 worker，再确认随数据记录的模型来源。复用 [`src/evaluation/sdpose.py`](src/evaluation/sdpose.py) 的 `load_scoring_runtime()` 和 `score_people()`，按任务选 geometry 134 槽位或由 `coco17()` 映射 Body17。默认关节置信度 `>= 0.30`，检查坐标有限且在有效画布内。RL 的目标可见关节和 Reward 资格在 rollout 前固定，生成漏检计作失分，评测器异常写明失败并停止该批更新。运行用本节下方的 SDPose 环境，保留 `pose_backend=sdpose`、槽位序、阈值、模型或代码 SHA。依赖或权重缺失时直接报错，禁止静默回退到 DWPose。DWPose 可用于明确标注的历史对照及既有生成条件、标注和缓存读取。更换评分器须用新结果名和新实验签名，已锁定实验沿用原快照。槽位、阈值及评测细则见 [SDPose 规范](docs/SDPOSE_POSE_EVAL_SPEC.md)。

---

## 6. 环境与路径

| 项 | 值 |
|---|---|
| 本地 | `D:\codeplus\pictureedit` |
| 远端 | `ssh sg` → node01（用户 `shangguanrz`，RTX 6000D 85GB） |
| 远端主项目 | `/home/shangguanrz/project/pic-edit` |
| DeepGen 环境 | `miniconda3/envs/deepgen/bin/python` |
| Qwen21 环境 | `nihaochen/envs/qwen21/bin/python` |
| SMPLest-X 环境 | `add/envs/smplestx_bw/bin/python`（`pyrender` 需 EGL） |
| SDPose 评测环境 | `/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python` |
| SDPose 权重与代码 | `/data/shangguanrz/models/SDpose/{SDPose-Wholebody,SDPose-OOD}` |
| 新产物落盘 | `/data/shangguanrz/` |

**SSH 投递纪律**：向远端送含转义或重格式的脚本/文档，**一律 `scp` 直传**。本机 bash 的 heredoc 经 `ssh` 传送时曾把 `\n` 污染成 `/n`，改过的远端文件必须 `py_compile` 或等价校验。

---

## 7. 单双人配对数据口径（2026-11-22 起）

双人姿态对产出**双口径**，目录与 manifest 分开，来源平台（小红书 / B 站 `bil/dual` 等）口径一致：

- **SFT 对**：背景相似度必须达标——同机位同背景，DINOv2 `bg_sim >= 0.83`、Lab L1 `<= 26`、人物服饰色 `person_col >= 0.52`，姿态变化走标准门。放 `pairs/sft/`。
- **RL 对**：**不要求背景完全一致**，只要求同一双人、姿态有明显变化（宽松门，如 `bg_sim >= 0.55`、`L1 <= 45`、每视频配额提高），供强化学习扩大覆盖。放 `pairs/rl/`。
- 两档的身份门一致（`face_sim >= 0.38`；未核到脸时仍锁严格背景，防止错人进对）。
- 每条 pair 的 manifest 记录所属档位与实际门限参数；看护/挖掘脚本参考 `/data/shangguanrz/bil/scripts/bil_dual_pair_watch.py`（增量、GPU 空闲门、断点续跑）。
- **RL 背景奖励固定取 Source A**。单人和同源双人用源图 A，独立双源用有序参考的第一张图 A（Qwen index 0）。只比较 A 到输出画布的可靠对应区域，排除 A 的原前景与目标所有人物的前景并集及边界。A/B 背景差异不限制这项奖励的资格；B 的 RGB 不参与背景评分。前景未通过资格门槛或画布无法对齐时记录不适用。已锁定的运行沿用原奖励签名。

---
