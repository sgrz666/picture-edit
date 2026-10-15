# AGENTS.md — 协作智能体须知

这个仓库有明确的目录约定和历史包袱。改代码、开实验、建目录之前先读这篇。
权威层级说明在根 [`README.md`](README.md) 第 3 节；本文只讲**你动手时要遵守什么**。

---

## 1. 动手前

- **先读后写**。要改的文件先打开看一遍，不要凭目录名猜内容。
- **不信旧记录，以磁盘实测为准**。README、记忆、日志都可能过期；结论要有当前磁盘上的证据。
- **沙箱或权限被拒就停下**，不要绕，回来问人。
- **图片与视频是受保护资源**（`.png` `.jpg` `.jpeg` `.mp4` 等）。任何整理、清理、迁移都不得把它们纳入删移范围。
- 任何删除/移动先做**只读审计 + dry-run**，列清单给人确认后再落地。
- 远端与本地会双向漂移。动远端前先 `git log` 核对，快照带时间戳。

---

## 2. 开一个新实验

### 2.1 先归族，再建目录

实验统一放 `experiments/` 下，**先看有没有对应的实验族目录，有就进去建，没有才新建族**。

按**任务目标**分 7 个族（本地目录名与对应关系见 `experiments/README.md` 与 `实验中文索引.md`）：

| 族 | 本地代表目录 |
|---|---|
| 显存与架构压测 | `review_20260908_v2` |
| DeepGen V6 单人长程训练 | `iper_v6_native_20k_v1` 等 |
| DeepGen 定量评测与消融 | `iper_v6_20k_eval_final` 等 |
| 面部/细节适配器与过拟合门禁 | `tiktok_00001_condition_repair` 等 |
| Qwen 2.1 身体控制 | `qwen21_iper_body` |
| Qwen 2.1 双人交互 | `qwen-2.1-test`、`qwen21_hi4d_dual` |
| 综合评测与开源对比 | `16-test` |

> **⚠️ 两侧结构不一致（2026-10-13 实测）**：本地 `experiments/` 已重构成「平铺每个实验目录 + README 索引」；
> 服务器 `sg:/home/shangguanrz/project/pic-edit/experiments/` 仍是旧的 5 族分组 ——
> `1.deepgen experiment` / `2.qwen-test` / `3.singel man` / `4.two man ` / `5.workflow`（名字带空格与拼写错误，未清理）。
> **动手前先 `ls` 确认自己在哪一侧**，不要假设结构。族内归类的原则两边一致：按任务目标，不按时间或人名。

**命名**：`<主题>_<YYYYMMDD>`（如 `bfs_body_face_joint_pilot_20261008`）或 `<主题>_v<N>`。不要用 `test`、`tmp`、`new` 这种名字。

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
└── report/              # 汇总报告、面板图
```

参照样本：`experiments/bfs_body_face_joint_pilot_20261008/`（大实验，子目录切得细）、
`experiments/qwen-2.1-test/`（中等实验，`data/` + `experiment/{code,results}` 两层）。

### 2.3 脚本放哪

- **只服务本实验** → 实验目录内的 `scripts/`
- **跨实验可复用** → 提升到仓级 `scripts/`，按职责进子目录：

```
scripts/
├── train/    # 正式训练入口
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
| `scripts/` | 跨实验的标准入口 | 某个实验私有的逻辑 |
| `experiments/` | 每个实验自包含的目录 | 共享代码、模型权重本体 |
| `workflow/` | 动作迁移流水线及其 runs | 训练代码、数据集 |
| `docs/` | 设计规范、审查与整改记录、调研报告 | 运行日志、指标 json |
| `reports/` | Word/PPT 汇报成果 | 中间产物 |
| `research/` | 专题调研与审计证据 | 正在跑的代码 |
| `tests/` | 契约与回归测试 | 手动跑的实验 |
| `datasets/` `models/` | 数据与权重本体（服务器路径，本地多为软链或说明） | 任何新写的文件 |
| `archive/` | 历史归档与清理快照 | 活跃代码 |

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
| 新产物落盘 | `/data/shangguanrz/` |

**SSH 投递纪律**：向远端送含转义或重格式的脚本/文档，**一律 `scp` 直传**。本机 bash 的 heredoc 经 `ssh` 传送时曾把 `\n` 污染成 `/n`，改过的远端文件必须 `py_compile` 或等价校验。

**`sgrzhh` 别名已失效**（`Connection closed` 持续失败），用 `sg`。

---

## 7. 提交前自查

- [ ] 新目录有 README，且不是空话
- [ ] 实验目录自包含，别人能照着复现
- [ ] 没有往根目录散落新脚本
- [ ] 没有图片/视频被删移
- [ ] 远端改过的文件过了语法/编译校验
- [ ] 大文件没有进 Git
- [ ] 结论都有磁盘上的证据，没有靠记忆
