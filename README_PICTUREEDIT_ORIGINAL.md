# 🥋 人物图像姿态编辑系统（Picture Edit）项目全仓层级目录与架构指南

> **更新时间**：2026年10月（已核验至 2026-10-27 最新代码、实验与数据盘事实）  
> **仓库根路径**：`/home/shangguanrz/project/pic-edit`  
> **外部持久数据盘**：`/data/shangguanrz`（22TB 分区，持久挂载于 `/dev/sdb1`）  
> **核心任务**：基于三维人体表示（SMPL-X / DWPose）的单人与双人参考姿态图像编辑。在严格保持源人物 A 的身份（Face ID）、服饰材质（Clothing）和背景一致性（Background Consistency）的前提下，实现身高体型解耦的动作姿态与面部表情重定向（目标姿态与表情来自 B）。

---

## 目录索引 (Table of Contents)

1. [项目定位与核心技术路线](#1-项目定位与核心技术路线)
2. [全局层级目录总览 (Global Tree)](#2-全局层级目录总览)
3. [核心代码与工作区层级详解](#3-核心代码与工作区层级详解)
   - [3.1 src/ —— 核心源码架构](#31-src--核心源码架构)
   - [3.2 scripts/ —— 标准分类运行入口](#32-scripts--标准分类运行入口)
   - [3.3 experiments/ —— 实验运行工作区与全量权重库](#33-experiments--实验运行工作区与全量权重库)
   - [3.4 datasets/ —— 训练与评测数据集工作区](#34-datasets--训练与评测数据集工作区)
   - [3.5 models/ —— 预训练基座与外挂权重库](#35-models--预训练基座与外挂权重库)
   - [3.6 configs/ —— 全局配置与资产注册表](#36-configs--全局配置与资产注册表)
   - [3.7 docs/ —— 项目设计规范、迁移记录与规划索引](#37-docs--项目设计规范迁移记录与规划索引)
   - [3.8 reports/ —— 成果报告与交付物导航](#38-reports--成果报告与交付物导航)
   - [3.9 research/ —— 专题技术调研与审计证据导航](#39-research--专题技术调研与审计证据导航)
   - [3.10 results/ —— 评审结果与可视化 Demo 库](#310-results--评审结果与可视化-demo-库)
   - [3.11 data/ —— 外部数据盘映射与调试数据](#311-data--外部数据盘映射与调试数据)
   - [3.12 add/ —— 外部工具链、扩展流水线与离线环境](#312-add--外部工具链扩展流水线与离线环境)
   - [3.13 archive/ —— 历史阶段归档与清理快照](#313-archive--历史阶段归档与清理快照)
   - [3.14 tests/ —— 自动化契约与回归测试套件](#314-tests--自动化契约与回归测试套件)
   - [3.15 third_party/ —— 第三方开源依赖与组件](#315-third_party--第三方开源依赖与组件)
   - [3.16 vram_lab/ —— 显存实验与测绘工具包](#316-vram_lab--显存实验与测绘工具包)
   - [3.17 辅助与运行缓存目录 (inputs, outputs, logs, caches, patches, deepgen)](#317-辅助与运行缓存目录)
   - [3.18 workflow/ —— 单人动作迁移基座流水线](#318-workflow--单人动作迁移基座流水线)
   - [3.19 workflow-multiple/ —— 双人姿态编辑流水线（一期）](#319-workflow-multiple--双人姿态编辑流水线一期)
   - [3.20 workflow-v2/ —— A+B+C 双源双人交互迁移流水线（二期）](#320-workflow-v2--abc-双源双人交互迁移流水线二期)
   - [3.21 task/ —— 合同任务书](#321-task--合同任务书)
4. [核心工作流运行命令矩阵 (Workflows)](#4-核心工作流运行命令矩阵)
5. [环境配置与外部数据盘 `/data/shangguanrz` 资产路径规范](#5-环境配置与外部数据盘-datashangguanrz-资产路径规范)
6. [文档体系：人工 README](#6-文档体系人工-readme)

---

## 1. 项目定位与核心技术路线

本项目聚焦于高质量人物姿态图像生成与编辑，致力于攻克姿态驱动下的**肢体接触交互、衣物几何保真、面部高频微纹理重定向**三大行业瓶颈。

### 1.1 关键考核指标 (OKR Guardrails)
- **姿态几何对齐**：$\text{PCK@0.02} \ge 0.98$，$\text{NME} \le 0.005$（毫米级骨骼对齐，零肢体穿模与悬空）。
- **人脸身份保持**：ArcFace 深度特征余弦相似度 $\ge 0.65$（突破 512 全身图 VAE 空间下采样瓶颈）。
- **交互动作合理率**：双人肢体接触（握手、搭肩、拥抱）合理率 $\ge 90\%$，主观生成质量优于开源通用基线 $5\%$ 以上。

### 1.2 四大技术演进分支
```
                          ┌────────────────────────┐
                          │   picture-edit 项目    │
                          └───────────┬────────────┘
                                      │
    ┌──────────────────┬──────────────┴─────┬──────────────────┐
    ▼                  ▼                    ▼                  ▼
【分支一：DeepGen-1.0】 【分支二：Qwen-Image Full】 【分支三：Qwen-2.1 Pro】  【分支四：Workflow 流水线】
• 2.47B MMDiT 双流    • 7B 单流 DiT 全尺寸控制   • 16块官方 Union 骨架   • 单人 4 阶段无 LoRA
• V6 统一控制适配器   • Native V5 模块化训练包   • 14.6M FP32 ProAdapter • 双人 A+B+C (workflow-v2)
• V6.5 面部/手部适配器 • 双人 Token 绑定与解耦   • visible_owner 硬隔离  • 官方 Hi4D 接触与重定向
• 门禁驱动转向 Qwen   • 300步固定样本训练体系     • 2阶段 768px 局部精修  • 骨架对齐 PCK=0.976 打平
```

---

## 2. 全局层级目录总览

```
pic-edit/
├── src/                    # [核心源码] 姿态适配器网络、SMPL-X 几何提取、数据集读取与模型集成
│   ├── adapter/            #   早期 CNN 姿态适配器基线 (零卷积初始化)
│   ├── controls/           #   SMPL-X 渲染器 (骨架、法线、深度、接触图)
│   ├── data/               #   数据集加载器 (TikTok, iPER, Hi4D, 数据增强)
│   ├── evaluation/         #   评测工具包 (DWPose ONNX, ArcFace, PSNR/SSIM)
│   ├── integration/        #   生成管线挂接与推理 Pipeline Bridge
│   ├── pose_edit/          #   推理侧共享层：单/双人 workflow 共用（缓存指纹、SMPLest-X、DWPose-134、geometry 定标）
│   └── pose_control/       #   姿态控制核心演进架构
│       ├── v6/             #     DeepGen V6 架构 (detail, face, hand, interface, reasoning)
│       ├── qwen21/         #     Qwen 2.1 简化版控制适配器 (body, face, dual, runtime)
│       ├── qwen21_full/    #     Qwen 2.1 完整迁移架构 (32层全尺寸控制、native_v5 模块化训练包)
│       ├── qwen21_pro/     #     Qwen 2.1 Pro 适配器 (16块官方Union + 14.6M FP32 ProTaskAdapter)
│       └── qwen21_union/   #     官方 ControlNet Union pipeline 与 180 键权重封装
├── scripts/                # [运行入口] 标准化分类运行入口 (train, eval, run, data, audit, reports, weights)
│   ├── train/              #   正式训练脚本 (V6, Qwen21 Full, V5 fixedpair, Hi4D Dual)
│   ├── eval/               #   量化评测与阶段对照脚本
│   ├── run/                #   快速单例验证与推理编排 (含 run/home 统一运行入口与 V5 推理)
│   ├── data/               #   数据清洗、划分、SMPL-X 离线预计算与固定样本构建
│   ├── audit/              #   权重审计、架构一致性与运行时契约检查 (含 V5 状态审计)
│   ├── reports/            #   Word 文档、指标图表与对比 Panel 生成 (含 V5 报告导出)
│   └── weights/            #   权重继承、格式转换与兼容性映射
├── experiments/            # [实验工作区] 全部训练/评测/消融 run，按技术线分 6 组，大检查点落盘 /data
│   ├── 1.deepgen experiment/  # DeepGen 基座全部实验（champ/ + iper/ 子分组 + 散放 run）
│   ├── 2.qwen-test/        #   Qwen 基座综合测试（16-test、qwen-2.1-test 等）
│   ├── 3.singel man/       #   Qwen 迁移线单人姿态控制（里程碑 qwen21_iper_body）
│   ├── 4.two man /         #   Qwen 迁移线双人交互（qwen21_hi4d_dual 等）
│   ├── 5.workflow/         #   workflow 相关实验预留区
│   └── 6.qwen-controlnet union/ # 官方 ControlNet Union 控制与迁移适配区
├── workflow/               # [单人动作迁移流水线] 无 adapter/LoRA：A 保外观、B 供动作，四阶段出图
│   ├── runs/               #   每次运行的四阶段产物与证据链（run.json + stage.json 指纹，含 tiktok/union-test）
│   ├── diagnostics/        #   诊断性调查归档（按日期分 campaign：scale 诊断等）
│   ├── human_eval/         #   主观成对盲评（出题/盲打/tally 统计）
│   ├── scripts/            #   流水线私有工具（six-picture-create 为样板）
│   ├── logs/ patches/      #   批处理与单跑日志 / SMPLest-X 本地补丁
│   └── pose_transfer/ tests/  #  几何与提示词核心逻辑 / 单元测试
├── workflow-multiple/      # [双人姿态编辑流水线] Hi4D 双人交互迁移一期，与 workflow/ 共用 src/pose_edit/
├── workflow-v2/            # [双人 A+B+C 迁移流水线] A/B 双源独立外观 + C 双人交互动作参考迁移二期
├── task/                   # [合同任务书] 人物图像姿态编辑任务书_v3.docx（验收条款权威来源）
├── datasets/               # [数据集工作区] 图像、视频帧与 3D 几何标注资产（大包软链直达 /data）
├── models/                 # [模型权重库] 预训练基座与外挂权重（10-25 迁入 5 组至 /data，原位软链回填）
├── configs/                # [全局配置] 架构超参定义与全局资产集中注册表 (asset_registry.json)
├── docs/                   # [技术文档] 架构白皮书、训练计划、对照报告、空间清理与跨盘迁移执行记录
├── reports/                # [成果汇报] 面向汇报的 Word 简报、量化图表、V5 报告与对比 Panel
├── research/               # [专题调研] BFS 移用评估、Hi4D SMPL-X 拟合、Pro 架构取舍、CHAMP QC 审计
├── results/                # [可视化Demo] 早期评审结果、显存对比图表与精选展示图板
├── data/                   # [数据盘导航与调试] 外部数据盘 /data/shangguanrz/ 映射导航与轻量调试样本
├── add/                    # [扩展工具链] SAM2, DinoV2, Multi-HMR, SMPLest-X 离线环境与批处理产物
├── archive/                # [历史归档] 历史整理批次快照 (cleanup_20261003/08/11) 与旧工具封存
├── tests/                  # [自动化测试] 80+ 自动化测试与回归套件 (覆盖架构契约、适配器梯度与数据协议)
├── third_party/            # [第三方依赖] CHAMP 引导模块与 V6.5 Face 外部组件
├── vram_lab/               # [显存测绘包] 显存开销自动化测量、极限压测与验收基准
├── inputs/ outputs/ logs/  # 样图输入 / 临时输出 / 训练与运行日志
├── caches/ patches/ deepgen/ # 运行时缓存 / Diffusers 侵入补丁 / DeepGen 上游副本
```

---

## 3. 核心代码与工作区层级详解

### 3.1 `src/` —— 核心源码架构
`src/` 是项目的算法心脏，承载所有的适配器网络设计、3D 控制信号生成、扩散管线挂接与评测实现。

```
src/
├── adapter/                # 早期 CNN 姿态适配器基线 (零卷积初始化)
├── controls/               # SMPL-X 控制信号生成器 (骨架、法线、深度、接触图)
├── data/                   # PyTorch 数据集协议实现 (TikTok, iPER, Hi4D, 数据增强)
├── evaluation/             # 定量评测模块 (DWPose 133/134 ONNX, ArcFace, PSNR/SSIM)
├── integration/            # 扩散模型集成桥接 (DeepGenPipeline)
├── pose_edit/              # 推理侧共享层：单/双人 workflow 共用（缓存指纹、SMPLest-X、DWPose、定标）
└── pose_control/           # 姿态控制演进核心
    ├── v6/                 #   【DeepGen-1.0 V6 架构】(detail, face, hand, interface, reasoning)
    ├── qwen21/             #   【Qwen-Image-2.1 简化适配版】(body, face, dual, runtime)
    ├── qwen21_full/        #   【Qwen-Image-2.1 Full 全量迁移架构】
    │   ├── native_v5/      #     最新 V5 模块化训练包 (conditioning, generation, source_bank, training)
    │   ├── qwen_adapter.py #     32 层单流 DiT 控制上下文管理器
    │   └── execution.py    #     全量训练循环与混合精度优化器
    ├── qwen21_pro/         #   【Qwen-Image-2.1 Pro 适配器架构】
    │   ├── adapter.py      #     ProTaskAdapter (14.6M FP32 参数，零初始化安全增量)
    │   ├── banks.py        #     PersonBankAttention (visible_owner 硬隔离门控，杜绝串脸)
    │   ├── encoders.py     #     6 模态类型化几何编码与显式双人接触编码 (ContactRelations)
    │   ├── losses.py       #     区域化互斥 Flow Loss (face, contact, hand, body) 与免检测器特征损失
    │   └── refinement.py   #     Stage 2: 768px 高清面部与接触手部局部精修工厂
    └── qwen21_union/       #   【官方 ControlNet Union 适配层】(180 键权重校验与 Pipeline 包装)
```

---

### 3.2 `scripts/` —— 标准分类运行入口
所有日常开发、训练与评测工具按工程职责统一归类于 7 大子目录：
- `train/`：训练入口（`run_qwen21_v5_fixedpair.py`、`train_iper_qwen21_full.py`、`train_hi4d_qwen21_dual.py`）。
- `eval/`：正式量化评测（`eval_iper_qwen21_full.py`、`eval_hi4d_qwen21_dual.py`）。
- `run/`：单例验证与推理编排（`run_qwen21_native_full_v5.py`、`run_demo_pose_edit.py`）。
- `data/`：数据清洗、划分与几何预计算（`assemble_qwen21_fixedpair_v5.py`、`prepare_qwen21_fixedpair_v5.py`）。
- `audit/`：权重审计、状态核验与一致性检查（`summarize_qwen21_fixedpair_training.py`、`status_qwen21_fixedpair_v5.py`）。
- `reports/`：报告与交付物导出（`export_qwen21_fixedpair_report.py`、`build_qwen21_fixedpair_visual_report.py`、`render_qwen21_fixedpair_report.cjs`）。
- `weights/`：权重继承与格式转换（`inherit_qwen21_full_weights.py`）。

---

### 3.3 `experiments/` —— 实验运行工作区与全量权重库
全部训练、评测、消融实验的落点，遵循自包含原则。大检查点落盘于 `/data/shangguanrz/pic-edit-experiments/`：
1. **显存压测族** (`review_20260908_v2`)：111 矩阵 PASS，确立 Letterbox 规范。
2. **DeepGen V6 单人长程** (`iper_v6_native_20k_v1`)：16k 步达保真峰值（PSNR 28.20dB）。
3. **DeepGen 定量与消融** (`iper_v6_20k_eval_final`)：受控相比无控制 PSNR +7.07dB。
4. **面部适配器门禁** (`tiktok_overfit_sanity_full`)：证明 MMDiT 单层特征强行注入破坏全局流。
5. **Qwen 2.1 身体控制** (`qwen21_iper_body`)：PSNR 25.86dB，PCK@0.02 达 98.00%。
6. **Qwen 2.1 双人交互** (`qwen21_hi4d_dual`)：双人检出率 100%，零身份互换。
7. **综合评测与开源对比** (`16-test`)：16 组全场景对照，确立两阶段解耦路线。
8. **ControlNet Union 官方实测** (`6.qwen-controlnet union`, `workflow/runs/tiktok/union-test`)：TikTok 8 对实测：对齐布局下 PCK@0.2=0.976 打平基线；纯控制模式 ID 降至 0.01~0.08。
9. **Qwen 2.1 Pro 适配器** (`/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1/`)：16 块原生控制块 + 14.6M FP32 ProTaskAdapter，零初始化安全，200 步过拟合验证 (Face loss -31.86%, Contact -64.37%, Hand -29.44%)。
10. **Native V5 固定样本训练** (`/data/shangguanrz/pic-edit-experiments/`)：TikTok 单人 300 步与 Hi4D 双人 300 步固定样本训练，配齐 30 步过程记录与无检测器 ROI 监督。

---

### 3.4 `datasets/` —— 训练与评测数据集工作区
- `TikTokDataset/`（软链 → `/data/shangguanrz/pic-edit-datasets/TikTokDataset`，105 GB）：单人主线。
- `Hi4D-origin/`（软链 → `/data/shangguanrz/pic-edit-datasets/Hi4D-origin`，359 GB）：双人原始 32 个未解压分包。
- `Hi4D_pilot_v1/`（实体，2.5 GB）：配准几何标注与 512 条件资产。
- `iPER/`（实体，12 GB）：`raw_1024`、`sampled_6src64tgt` 基准集与 `assets_512` 条件包。
- `3dpw_single_final_v2/`（实体，1.1 GB）：户外极限姿态评测集。
- `HIA/`（外部，`/data/shangguanrz/HIA/`）：Human Interaction Assets 双人紧密交互高质切片。

---

### 3.5 `models/` —— 预训练基座与外挂权重库
存放项目依赖的生成基座与预训练模型。实体全部位于独立数据盘 `/data/shangguanrz/`：
- `Qwen-Image-2.1/`（31 GB）：7B 单流 DiT 图像编辑与生成基座（`/data/shangguanrz/pic-edit-models/Qwen-Image-2.1`）。
- **10-25 迁入的 5 组工具模型**（位于 `/data/shangguanrz/pic-edit-models/repo-models/`，原位软链回填）：
  - `DeepGen-1.0-diffusers/` (14.4 GB)、`antelopev2/` (4.97 GB)、`DWPose/` (351 MB)、`face_eval/` (356 MB)、`smpl/` (78 MB)。
- `Qwen-Image-2.1-PE-I2I/`（18 GB）及 GGUF 量化系列（10 GB / 6.5 GB）。
- `models/home-models/`（15.3 GB）：10-25 迁移自 `~/models`。

---

### 3.6 `configs/` —— 全局配置与资产注册表
- `asset_registry.json`：全仓资产单一事实源（登记权重、数据集、缓存、冷归档路径与版本 SHA）。
- `adapter_v6_architecture.json`：V6 适配器各层通道、头数与注入层配置。

---

### 3.7 `docs/` —— 项目设计规范、迁移记录与规划索引
- [`模型跨盘迁移执行记录_20261025.md`](docs/模型跨盘迁移执行记录_20261025.md)：8 组大模型/数据（54GB）迁至 `/data` 执行记录。
- [`远端空间清理执行记录_20261025.md`](docs/远端空间清理执行记录_20261025.md)：远端根盘空间释放（+15GB）执行记录。
- [`Qwen21_V5固定样本结果报告工具_20261026.md`](docs/Qwen21_V5固定样本结果报告工具_20261026.md)：V5 报告工具与指南。
- [`Qwen21_V5双人双源训练与使用说明_20261026.md`](docs/Qwen21_V5双人双源训练与使用说明_20261026.md)：V5 双人双源训练说明。
- [`superpowers/plans/`](docs/superpowers/)：HIA 双人手脸数据集计划。

---

### 3.8 `reports/` —— 成果报告与交付物导航
- [`native_v5_fixedpair_20261026_v1/`](reports/native_v5_fixedpair_20261026_v1/README.md)：V5 固定样本实验成果报告（HTML/PDF/证据包）。
- `qwen_union_controlnet_20261027/`：Qwen ControlNet Union 控制能力报告。
- `architecture/`：系统架构设计图册。
- `evaluation/`：TikTok 与 Hi4D 定量评测指标库。

---

### 3.9 `research/` —— 专题技术调研与审计证据导航
- [`champ_detail_quality_20261025/`](research/champ_detail_quality_20261025/README.md)：CHAMP 细节质量、模糊度门禁与 7173 行质检记录。
- [`union_adapter_architecture_20261017/`](research/union_adapter_architecture_20261017/README.md)：Qwen21 Pro 与 Union 架构取舍与设计证据。
- `canvas_native_portrait_acceptance_20261022/`：竖屏原生画布与尺度适配验收。
- `qwen21_depth_champ_20261023/`：Depth 与 CHAMP 空间注意力编码融合。
- `bfs_transfer_audit_20261004/`：BFS 控制移用性与两阶段级联审计。
- `hi4d_smplx_pilot_20261011/`：Hi4D 双人交互 SMPL-X 拟合与接触图修复。

---

### 3.10 `results/` —— 评审结果与可视化 Demo 库
早期重要评审的图像面板与显存测绘曲线（`review_20260908_v2/`、显存测绘仪表板）。

---

### 3.11 `data/` —— 外部数据盘映射与调试数据
- 承担外部数据盘 `/data/shangguanrz/` 映射索引功能（详见 [`data/README.md`](data/README.md)）。
- `debug_samples/`：16 对合成调试样本，供快速打通端到端训练与推理链路。

---

### 3.12 `add/` —— 外部工具链、扩展流水线与离线环境
- `tools/`：SAM2, DinoV2, Multi-HMR, SMPLest-X, nvdiffrast。
- `envs/`：独立 Python 环境（`smplestx_bw`, `multihmr`）。

---

### 3.13 `archive/` —— 历史阶段归档与清理快照
封存历次架构重构淘汰的代码与备份（`cleanup_20261003/08/11/`）。

---

### 3.14 `tests/` —— 自动化契约与回归测试套件
80+ 自动化测试用例，覆盖零初始化契约、几何注入维度、Qwen21 权重继承、损失函数数学一致性与数据加载协议。

---

### 3.15 `third_party/` —— 第三方开源依赖与组件
- `champ_guidance/`：CHAMP 3D 骨架渲染算法。
- `v65_face/`：V6.5 适配器人脸对齐组件。

---

### 3.16 `vram_lab/` —— 显存实验与测绘工具包
自动化测量 PyTorch 实际占用与分配峰值，输出分析报告与仪表板。

---

### 3.17 辅助与运行缓存目录
`inputs/`、`outputs/`、`logs/`、`caches/`、`patches/`、`deepgen/`。

---

### 3.18 `workflow/` —— 单人动作迁移基座流水线
**无 adapter/LoRA** 的单人动作迁移：A 提供外观场景，B 提供动作；SMPLest-X 估计 SMPL-X $\to$ 投影 DWPose-134 骨架 $\to$ PE-I2I 编译提示词 $\to$ Qwen 基座读 A+骨架出图。
- 关键实验：`runs/tiktok/union-test/` 证实骨架对齐到 A 时 PCK@0.2=0.976 打平基线。
- 入口：`bash workflow/run.sh --check`，批处理 `bash batch.sh`。

---

### 3.19 `workflow-multiple/` —— 双人姿态编辑流水线（一期）
Hi4D 双人姿态编辑与交互迁移流水线，共用 `src/pose_edit/`。
- 测试集：17 对样本（12 对 same_identity + 5 对 cross_pair）。
- 入口：`bash workflow-multiple/batch_dual.sh`。

---

### 3.20 `workflow-v2/` —— A+B+C 双源双人交互迁移流水线（二期）
A/B 提供两名独立人物的外观、服装与体型，C 仅提供双人动作交互参考（不进入 Qwen 参考图）。
- 调度与模式：`abc_runner.py`（支持 `geometry_only`、`qwen_baseline`、`native_v5`）。
- 官方 Hi4D 接触路径：输出 body22 骨架与接触点，SMPL 表面附件转移。
- 产物根目录：`/data/shangguanrz/pic-edit-workflow-v2/runs/`。
- 入口：`bash workflow-v2/run.sh`。

---

### 3.21 `task/` —— 合同任务书
人物图像姿态编辑任务书 v3（`人物图像姿态编辑任务书_v3.docx`），验收条款权威来源。

---

## 4. 核心工作流运行命令矩阵

所有任务均支持在单卡 RTX 6000 Ada (85.6 GiB) 上运行：

```bash
# 激活推荐训练环境
conda activate /home/shangguanrz/nihaochen/envs/qwen21
cd /home/shangguanrz/project/pic-edit
```

| 任务场景 | 推荐命令 | 对应输入 / 输出 |
|---|---|---|
| **Qwen 2.1 Native V5 固定样本训练** | `python scripts/train/run_qwen21_v5_fixedpair.py --dataset tiktok --steps 300` | 单人 TikTok 0374/0126 300步训练，产出过程图 |
| **Qwen 2.1 Native V5 固定样本报告导出** | `python scripts/reports/export_qwen21_fixedpair_report.py --require-complete` | 导出只读证据包并构建 HTML/PDF 报告 |
| **Qwen 2.1 Pro 适配器推理/训练** | `python src/pose_control/qwen21_pro/runner.py infer --arm full4 --steps 40` | 官方 16 块 Union + 14.6M ProAdapter 受控出图 |
| **单人动作迁移流水线 (workflow)** | `cd workflow && bash run.sh --pair ../experiments/2.qwen-test/qwen-2.1-test/data/testset/pairs/hard_05 --resolution 768` | 输出 `workflow/runs/<name>/` 四阶段产物 |
| **双人 A+B+C 迁移流水线 (workflow-v2)** | `cd workflow-v2 && bash run.sh --source-a /path/A.jpg --source-b /path/B.jpg --interaction /path/C.jpg --mode native_v5` | 输出 `/data/shangguanrz/pic-edit-workflow-v2/runs/` |
| **自动化测试全量执行** | `pytest tests/test_qwen21_*.py workflow-v2/tests -v` | 验证控制契约与数学正确性 |

---

## 5. 环境配置与外部数据盘 `/data/shangguanrz` 资产路径规范

### 5.1 Python 环境规范
- **Qwen 2.1 主线推荐**：`/home/shangguanrz/nihaochen/envs/qwen21/bin/python`（PyTorch 2.8.0+cu128, Diffusers 0.41.0.dev0, Transformers 5.17.0）。
- **DeepGen V6 主线**：`/home/shangguanrz/miniconda3/envs/deepgen/bin/python`。
- **SMPLest-X 几何与渲染环境**：`/home/shangguanrz/project/pic-edit/add/envs/smplestx_bw/bin/python`。

### 5.2 外部数据盘 `/data/shangguanrz` 映射总表

| 数据盘物理路径 | 映射/软链入口 | 实体体积 | 功能与内容 |
|---|---|---|---|
| `/data/shangguanrz/pic-edit-datasets` | `datasets/` | 463 GB | Hi4D-origin (359G 原始包) + TikTokDataset (105G 全构建层) |
| `/data/shangguanrz/pic-edit-models` | `models/` | 88 GB | Qwen 2.1 基座 (31G)、PE-I2I 提示词模型 (18G/10G)、repo-models 5组 (20G) |
| `/data/shangguanrz/pic-edit-experiments` | `experiments/` 部分映射 | 38 GB | Qwen21 Pro 适配器、Native V5 固定样本实验、ControlNet Union (7.1G) |
| `/data/shangguanrz/pic-edit-workflow-multiple` | `workflow-multiple/runs/` | 动态 | 双人动作迁移一期 runs 产物 |
| `/data/shangguanrz/pic-edit-workflow-v2` | `workflow-v2/runs/` | 动态 | 双人 A+B+C 动作迁移二期 runs 产物 |
| `/data/shangguanrz/HIA` | 外部引用 | 动态 | Human Interaction Assets 高精度双人手脸交互切片集 |
| `/data/shangguanrz/pic-edit-caches` | `caches/` | 9.6 GB | 1390 个 Hi4D 双人骨架与交互特征缓存 (`.pt`) |
| `/data/shangguanrz/pic-edit-archive` | `archive/` | 95 GB | 历史 20B 模型、历史 runs 备份与 Git bundle 快照 |
| `/data/shangguanrz/pic-edit-audits` | 外部引用 | 150 MB | 跨盘迁移双端 sha256 清单及空间清理现场留档 |
| `/data/shangguanrz/models/home-models` | `~/models` 软链 | 15.3 GB | 10-25 迁移自 `~/models` 的家目录模型集合 |
| `/data/shangguanrz/nihaochen` | `~/nihaochen` 软链 | 22.3 GB | 10-25 迁移自 `~/nihaochen/` 的训练切片与模型 |

---

## 6. 文档体系：人工 README

本仓库每个目录的说明文档均为**人工逐目录撰写**的 `README.md`（目录职责、与上下层的关系、关键事实、坑与纪律）：
- 截至 2026-10-27，全仓主要业务与技术线目录均已同步人工维护的最新版本。
- 两个 Qwen 权重目录（`models/Qwen-Image-2.1/`、`models/Qwen-Image-Edit-20B/`）为保护上游官方模型卡，项目侧说明以同目录 `README.pic-edit.md` 承载，详见 [`models/README.md`](models/README.md)。
