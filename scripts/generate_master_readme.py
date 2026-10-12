#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the master authoritative README.md for the picture-edit repository,
structured strictly according to the repository's full directory hierarchy.
"""
import os
import sys

def build_master_readme(repo_root="/home/shangguanrz/project/pic-edit"):
    template = r"""# 🥋 人物图像姿态编辑系统（Picture Edit）项目全仓层级目录与架构指南

> **更新时间**：2026年10月  
> **仓库根路径**：`__REPO_ROOT__`  
> **核心任务**：基于三维人体表示（SMPL-X / DWPose）的单人与双人参考姿态图像编辑。在严格保持源人物 A 的身份（Face ID）、服饰材质（Clothing）和背景一致性（Background Consistency）的前提下，实现身高体型解耦的动作姿态与面部表情重定向（目标姿态与表情来自 B）。

---

## 目录索引 (Table of Contents)

1. [项目定位与核心技术路线](#1-项目定位与核心技术路线)
2. [全局层级目录总览 (Global Tree)](#2-全局层级目录总览)
3. [核心代码与工作区层级详解](#3-核心代码与工作区层级详解)
   - [3.1 src/ —— 核心源码架构](#31-src--核心源码架构)
   - [3.2 scripts/ —— 标准分类运行入口](#32-scripts--标准分类运行入口)
   - [3.3 experiments/ —— 实验运行工作区与权重库](#33-experiments--实验运行工作区与权重库)
   - [3.4 datasets/ —— 训练与评测数据集本体](#34-datasets--训练与评测数据集本体)
   - [3.5 models/ —— 预训练基座与外挂权重库](#35-models--预训练基座与外挂权重库)
   - [3.6 configs/ —— 全局配置与资产注册表](#36-configs--全局配置与资产注册表)
   - [3.7 docs/ —— 项目设计规范与演进记录](#37-docs--项目设计规范与演进记录)
   - [3.8 reports/ —— 成果汇报、Word简报与量化指标](#38-reports--成果汇报word简报与量化指标)
   - [3.9 research/ —— 专题技术调研与审计证据](#39-research--专题技术调研与审计证据)
   - [3.10 results/ —— 评审结果与可视化 Demo 库](#310-results--评审结果与可视化-demo-库)
   - [3.11 data/ —— 小型调试样本数据](#311-data--小型调试样本数据)
   - [3.12 add/ —— 外部工具链、扩展流水线与离线环境](#312-add--外部工具链扩展流水线与离线环境)
   - [3.13 archive/ —— 历史阶段归档与清理快照](#313-archive--历史阶段归档与清理快照)
   - [3.14 tests/ —— 自动化契约与回归测试套件](#314-tests--自动化契约与回归测试套件)
   - [3.15 third_party/ —— 第三方开源依赖与组件](#315-third_party--第三方开源依赖与组件)
   - [3.16 vram_lab/ —— 显存实验与测绘工具包](#316-vram_lab--显存实验与测绘工具包)
   - [3.17 辅助与运行缓存目录 (inputs, outputs, logs, caches, patches, deepgen)](#317-辅助与运行缓存目录)
4. [核心工作流运行命令矩阵 (Workflows)](#4-核心工作流运行命令矩阵)
5. [环境配置与资产路径规范](#5-环境配置与资产路径规范)

---

## 1. 项目定位与核心技术路线

本项目聚焦于高质量人物姿态图像生成与编辑，致力于攻克姿态驱动下的**肢体接触交互、衣物几何保真、面部高频微纹理重定向**三大行业瓶颈。

### 1.1 关键考核指标 (OKR Guardrails)
- **姿态几何对齐**：$\text{PCK@0.02} \ge 0.98$，$\text{NME} \le 0.005$（毫米级骨骼对齐，零肢体穿模与悬空）。
- **人脸身份保持**：ArcFace 深度特征余弦相似度 $\ge 0.65$（突破 512 全身图 VAE 空间下采样瓶颈）。
- **交互动作合理率**：双人肢体接触（握手、搭肩、拥抱）合理率 $\ge 90\\%$，主观生成质量优于开源通用基线 $5\\%$ 以上。

### 1.2 三大技术演进分支
```
                          ┌────────────────────────┐
                          │   picture-edit 项目    │
                          └───────────┬────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
【分支一：DeepGen-1.0】       【分支二：Qwen-Image-2.1】     【分支三：两阶段高保真级联】
• 2.47B MMDiT 双流架构         • 7B 单流 DiT 统一序列架构       • Stage 1: BodyAdapter 强姿态驱动
• V6 统一控制适配器            • Qwen21 简化 Body/Face/Dual     (锁定 PCK=1.0 与衣服背景)
• V6.5 面部特征适配器          • Qwen21 Full 全量 32 层迁移   • Stage 2: BFS 768 ROI 局部精修
• V6.6 手部接触图适配器        • PersonTokenBinder 双人绑定     (前16层保留表情，后16层剪枝防磨皮)
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
│   └── pose_control/       #   姿态控制核心演进架构
│       ├── v6/             #     DeepGen V6 架构 (detail, face, hand, interface, reasoning)
│       ├── qwen21/         #     Qwen 2.1 简化版控制适配器 (body, face, dual, runtime)
│       └── qwen21_full/    #     Qwen 2.1 完整迁移架构 (32层全尺寸控制、面部监督、Token绑定)
├── scripts/                # [运行入口] 标准化分类运行入口 (train, eval, run, data, audit, reports, weights)
│   ├── train/              #   正式训练脚本 (V6, Qwen21 Body/Face/Full, Hi4D Dual)
│   ├── eval/               #   量化评测与阶段对照脚本
│   ├── run/                #   快速单例验证与推理编排 (含 run/home 统一运行入口)
│   ├── data/               #   数据清洗、划分、SMPL-X 离线预计算
│   ├── audit/              #   权重审计、架构一致性与运行时契约检查
│   ├── reports/            #   Word 文档、指标图表与对比 Panel 生成
│   └── weights/            #   权重继承、格式转换与兼容性映射
├── experiments/            # [实验工作区] 保存所有正式训练 runs、检查点、日志与评测生成图
│   ├── qwen21_iper_body/   #   Qwen21 iPER 身体姿态控制训练运行
│   ├── qwen21_hi4d_dual/   #   Qwen21 Hi4D 双人交互训练与动作 Prompt 实验
│   ├── iper_v6_native_20k/ #   DeepGen V6 20k 步正式基准 run
│   ├── iper_v65_face_*/    #   V6.5 面部适配器过拟合与泛化 run
│   └── tiktok_overfit_*/   #   TikTok 动作数据集适配性实验
├── datasets/               # [数据集本体] 图像、视频帧与 3D 几何标注资产
│   ├── iPER/               #   iPER 视频基准 (raw_1024, sampled_6src64tgt, assets_512)
│   ├── TikTokDataset/      #   TikTok 舞蹈姿态数据与 native 3D assets
│   ├── 3dpw_single_final_v2#   3DPW 单人高难度动作对齐测试集
│   ├── Hi4D-origin/        #   Hi4D 双人交互原始数据与 pilot 配准切片
│   └── champ_motions_*/    #   CHAMP 动作驱动样例
├── models/                 # [模型权重库] 基座模型、特征提取器与 3D 参数模型
│   ├── Qwen-Image-2.1/     #   Qwen 2.1 7B 单流 DiT 官方模型权重 (软链接至 /data)
│   ├── DeepGen-1.0-diffusers#  DeepGen-1.0 2.47B MMDiT 基座模型
│   ├── DWPose/             #   DWPose ONNX 骨骼关键点提取权重
│   ├── antelopev2/         #   InsightFace 人脸检测与 ArcFace 提取模型
│   ├── smpl/               #   SMPL / SMPL-X 官方三维人体拓扑模型文件
│   └── face_eval/          #   面部评估专用检测与对齐权重
├── configs/                # [全局配置] 架构超参定义与全局资产集中注册表 (asset_registry.json)
├── docs/                   # [技术文档] 架构白皮书、训练计划、对照报告、清理记录
├── reports/                # [成果汇报] 面向汇报的 Word (.docx) 简报、量化图表与 Panel 结果
├── research/               # [专题调研] BFS 移用评估、Hi4D SMPL-X 拟合、SynBody 审计等深度技术证据
├── results/                # [可视化Demo] 早期评审结果、显存对比图表与精选展示图板
├── data/                   # [调试数据] 合成调试样本、单例过拟合数据与小型切片
├── add/                    # [扩展工具链] SAM2, DinoV2, Multi-HMR, SMPLest-X 离线环境与批处理产物
├── archive/                # [历史归档] 历史整理批次快照 (cleanup_20261003/08/11) 与旧工具封存
├── tests/                  # [自动化测试] 60+ 单元测试与回归套件 (覆盖架构契约、适配器梯度与数据协议)
├── third_party/            # [第三方依赖] CHAMP 引导模块与 V6.5 Face 外部组件
├── vram_lab/               # [显存测绘包] 显存开销自动化测量、极限压测与验收基准
├── inputs/                 # [输入样图] 单人 person1.jpg 与双人 dual_person.jpg 测试用例
├── outputs/                # [临时输出] 快速推理与临时脚本产物
├── logs/                   # [运行日志] 训练与跨实验通用日志
├── caches/                 # [运行时缓存] 特征缓存与潜变量缓存
├── patches/                # [代码补丁] 针对上游 DeepGen 源码的微侵入式补丁 (deepgen_pose_control.patch)
└── deepgen/                # [上游副本] DeepGen-1.0-diffusers 原版官方代码库镜像
```

---

## 3. 核心代码与工作区层级详解

### 3.1 `src/` —— 核心源码架构
`src/` 是整个项目的算法心脏，承载了所有的适配器网络设计、3D 控制信号生成、扩散管线挂接与评测实现。

```
src/
├── adapter/                # 早期 CNN 姿态适配器
│   └── pose_adapter.py     #   PoseConditionAdapter：零卷积初始化，支持 8 通道特征残差注入
├── controls/               # SMPL-X 控制信号生成器
│   └── generator.py        #   从 SMPL-X 参数离线/在线渲染骨架、法线、深度、接触图
├── data/                   # PyTorch 数据集协议实现
│   ├── base_dataset.py     #   基类抽象与多模态数据对齐契约
│   ├── augmentation.py     #   刚体变换与尺度增强策略
│   └── tiktok_dataset.py   #   TikTok / iPER 序列数据集流式加载
├── evaluation/             # 定量评测模块
│   ├── metrics.py          #   PSNR, SSIM, LPIPS, ArcFace 余弦计算
│   └── dwpose_onnx/        #   自包含的 DWPose ONNX 推理器与关键点坐标归一化
├── integration/            # 扩散模型集成桥接
│   └── deepgen_pipeline.py #   将 Adapter 残差无缝插入 Diffusers Pipeline 采样循环
└── pose_control/           # 姿态控制演进核心
    ├── v6/                 #   【DeepGen-1.0 V6 架构】
    │   ├── interface/      #     MMDiT 控制接口与残差注入协议
    │   ├── reasoning/      #     Cross-Frame 关系推理与 Transformer 块
    │   ├── detail/         #     高频边缘与服装纹理特征注入
    │   ├── face/           #     V6.5 面部适配器 (ArcFace 512 + 128x128 纹理编码器)
    │   └── hand/           #     V6.6 手部接触网格与局部几何先验
    ├── qwen21/             #   【Qwen-Image-2.1 简化适配版】
    │   ├── body_adapter.py #     轻量级 BodyAdapter (SMPL-X 注入，PCK 1.0)
    │   ├── face_adapter.py #     轻量级 FaceAdapter (目标 ROI 脸部残差注入)
    │   ├── dual_adapter.py #     双人交互控制与两组几何拼接注入
    │   └── runtime.py      #     Qwen21 原生 Latent / Token 编码与缓存
    └── qwen21_full/        #   【Qwen-Image-2.1 Full 全量迁移架构】
        ├── qwen_adapter.py #     32 层单流 DiT 控制上下文管理器 (Qwen21FullControlContext)
        ├── qwen_runtime.py #     纯原生 I/O、True CFG 与 RoPE 空间坐标映射
        ├── execution.py    #     全量训练循环、参数分组与混合精度优化器
        ├── face_supervision.py # 目标脸部离线损失门控 (SCRFD 几何对齐)
        ├── face_loss_alignment.py # 5点对齐空间旋转变换与微观人脸对齐损失
        └── token_encoders.py # PersonTokenBinder：双人 Token 空间解耦绑定器
```

---

### 3.2 `scripts/` —— 标准分类运行入口
为了防止根目录脚本混乱，2026年10月全仓工具已按工程职责划分为 7 大类别。**所有日常开发、训练与评测均应从对应子目录调用**。

```
scripts/
├── train/                  # 正式训练入口
│   ├── train_iper_qwen21_full.py   # Qwen21 Full 全量 32 层身体与面部训练入口
│   ├── train_iper_qwen21_body.py   # Qwen21 简化身体控制 20k 训练入口
│   ├── train_hi4d_qwen21_dual.py   # Qwen21 双人 Hi4D 动作交互训练入口
│   ├── train_iper_v6_native.py     # DeepGen V6 原生 20k 步基准训练
│   ├── train_iper_v65_face.py      # DeepGen V6.5 面部保持分支训练
│   └── train_tiktok_overfit_sanity.py # TikTok 姿态适配性过拟合验证
├── eval/                   # 正式量化评测入口
│   ├── eval_iper_qwen21_full.py    # Qwen21 Full 30步全量采样评测 (输出 PCK, ArcFace, MAE)
│   ├── eval_iper_four_arm.py       # 四臂基准对照评测 (Body, Face, Dual, 无控制)
│   ├── eval_qwen21_face_repair.py  # 面部修复检查点逐样本量化评分
│   ├── eval_hi4d_qwen21_dual.py    # Hi4D 双人交互生成与接触姿态评分
│   ├── multi_stage_eval_v6.py      # DeepGen V6 多阶段演进评测
│   └── summarize_bfs_comparison.py # BFS 两阶段对比量化汇总
├── run/                    # 快速推理与单例编排
│   ├── run_qwen21_face_single_sample.py # 单样本极速训练与诊断探针
│   ├── run_demo_pose_edit.py       # 交互式 Demo 推理
│   ├── run_v6_comparative_3000.sh  # 3000 步横向对照批量推理流程
│   └── home/                       # HOME 运行器链接统一入口
├── data/                   # 数据预处理与离线几何准备
│   ├── split_iper_train_val.py     # iPER 训练集/验证集无重叠划分
│   ├── precompute_iper_face_features_v65.py # 离线提取 ArcFace 与 128x128 纹理
│   ├── prepare_qwen21_full_data.py # 准备 Qwen21 Full 训练对与张量缓存
│   └── build_tiktok_v6_conditions.py # TikTok 视频提取 SMPL-X 与 DWPose 几何
├── audit/                  # 架构与权重一致性审计
│   ├── check_qwen21_full_preservation.py # 验证 47 个核心架构文件是否被非预期修改
│   ├── audit_qwen21_face_alignment.py # 校验人脸检测框与旋转仿射变换矩阵
│   ├── verify_qwen21_full_resume.py   # 断点续训 RNG 种子与状态一致性校验
│   └── inventory_qwen21_full_weights.py # 统计权重层级与未初始化的残差层
├── reports/                # 文档与图表批量构建
│   ├── generate_bfs_word_report.py # 自动构建 BFS 组会分析报告 Word (.docx)
│   ├── build_iper1024_comparison_panels.py # 拼接 1024 高清三栏对比图板
│   ├── build_architecture_docx.py  # 导出系统架构设计规范 Word 文档
│   └── write_qwen21_face_report.py # 汇总 Face 实验数据并自动更新 Markdown
└── weights/                # 权重继承与格式转换
    └── inherit_qwen21_full_weights.py # 将简化版 Body 权重安全移植到 Full 架构中
```

---

### 3.3 `experiments/` —— 实验运行工作区与权重库
每个实验独立成目录，严格遵循**自包含原则**：包含运行配置文件、检查点（`.pt`）、逐步训练日志（`.jsonl`）、采样预览图板与评估指标。

```
experiments/
├── qwen21_iper_body/       # Qwen21 身体控制主线
│   └── full_20k/           #   20,000 步正式检查点 (checkpoint_last.pt)，PCK@0.02=1.0
├── qwen21_hi4d_dual/       # Qwen21 双人 Hi4D 动作交互实验
│   ├── single_pair14_action_v2_overfit/ # 单样本过拟合完美重构检查点
│   └── single_hug_eval/    #   双人拥抱动作测试指标与图板
├── iper_v6_native_20k_v1/  # DeepGen V6 原生架构 20k 训练
│   ├── checkpoints/        #   每 2500 步检查点与最优模型
│   └── three_row_showcase/ #   三栏可视化对比成果图
├── iper_v65_face_overfit_024_6/ # V6.5 Face Adapter 单例过拟合基准
│   └── checkpoints/        #   step 1500 检查点，验证 ArcFace 梯度流通性
├── tiktok_overfit_sanity_full/ # TikTok 数据集上的姿态对齐与光影泛化运行
└── 16-test/                # 16 组系统化对照流水线 (涵盖 Stage 1~6 的全套基线)
```

---

### 3.4 `datasets/` —— 训练与评测数据集本体
存放项目训练所需的真实图像帧、SMPL-X 三维网格参数与几何条件渲染产物。
- `iPER/`：国际动作驱动权威公开数据集，包含 206 段多视角视频。本仓划分了 `iper_sampled_6src64tgt`（6段源视频，64段目标动作）基准评测集，并离线提取了 `iper_assets_512_nvdiffrast` 条件包。
- `TikTokDataset/`：单人高动态舞蹈动作集，包含 340 个高质量全身视频片段，提取了完整 3D 骨架与部位掩膜。
- `3dpw_single_final_v2/`：户外复杂光照与大角度躯干旋转评测集，包含 60 组极限动作评测对。
- `Hi4D-origin/` & `Hi4D_pilot_v1/`：双人紧密肢体交互（握手、拥抱、摔跤、背负）数据集，提供精确配准的双人 SMPL-X 交互模型。

---

### 3.5 `models/` —— 预训练基座与外挂权重库
存放所有受控模型、开源基座以及下游提取器（大文件不入 Git，通过集中挂载管理）：
- `Qwen-Image-2.1/`：单流 7B DiT 官方完整权重，链接至 `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1`。
- `DeepGen-1.0-diffusers/`：2.47B MMDiT 官方权重，包含 VLM 编码器、Connector、DiT 主干与 VAE。
- `DWPose/`：`yolox_l.onnx` 与 `dw-ll_ucoco_384.onnx`，用于极速提取 133 个人体骨骼关键点。
- `antelopev2/`：SCRFD 人脸检测与 ResNet-100 ArcFace 512 维特征提取引擎。
- `smpl/`：SMPL-X 官方中性身体模型与网格拓扑先验（`SMPLX_NEUTRAL.pkl`）。
- `face_eval/`：面部评估专用检测器与评估模型。

---

### 3.6 `configs/` —— 全局配置与资产注册表
- `asset_registry.json`：**全仓资产单一事实源（Single Source of Truth）**。登记了基座权重、数据集实体、缓存路径、冷归档挂载点与版本哈希。
- `adapter_v6_architecture.json`：严格定义 V6 适配器各层的通道数、注意力头数、注入层索引（`[3, 8, 13, 18, 23, 28]`）及零卷积初始化参数。

---

### 3.7 `docs/` —— 项目设计规范与演进记录
- `PROJECT_STRUCTURE_20261003.md`：核心工作线与目录职责规范。
- `目录清理执行记录_20261003.md` & `目录清理执行记录_20261009.md`：历次仓库规范化整理的历史审计变更记录。
- `BFS_面部保持移用评估_20261004.md` & `BFS_扩大样本对照_20261004.md`：开源基准 BFS 的底层机理逆向与跨样本量化测评。
- `adapter_v6_architecture.md`：V6 统一适配器数学原理与条件桥接规范。

---

### 3.8 `reports/` —— 成果汇报、Word简报与量化指标
存放面向阶段交付、课题组汇报与技术评审的正式交付物：
- `BFS_组会技术报告_20261004.docx`：针对最新 In-Context 换脸与姿态重定向的 25 MB 学术级 Word 图文简报。
- `qwen21/`：Qwen2.1 阶段性推进成果报告与量化表格。
- `legacy_demo_202609/`：前期里程碑成果存档。

---

### 3.9 `research/` —— 专题技术调研与审计证据
存放针对技术难点开展的独立深入实验证据与探索记录：
- `bfs_transfer_audit_20261004/`：BFS LoRA 权重在 CPU/GPU 上的数学切分重投影验证与 768 局部采样探针。
- `hi4d_smplx_pilot_20261011/`：双人交互动作 SMPL-X 几何配准与接触图修复。
- `qwen21_dual_upgrade_20261011/`：Qwen21 双人控制架构升级与 Token 绑定验证。
- `smplx_condition_audit_20261008/`：条件渲染精度与视锥裁剪误差审计。

---

### 3.10 `results/` —— 评审结果与可视化 Demo 库
保存早期重要评审的图像面板与显存测绘曲线：
- `review_20260908/` & `review_20260908_v2/`：第一轮与第二轮全功能对照生成图板。
- `deepgen_vram_requirement_dashboard.png`：显存开销与批次缩放对比仪表板。

---

### 3.11 `data/` —— 小型调试样本数据
不进入训练集的小型轻量数据，专供快速打通端到端链路：
- `debug_samples/`：16 对 512×512 合成图像与对应模拟控制图，供单步 Forward/Backward 冒烟测试。
- `champ_overfit/`：单帧 CHAMP 快速调试几何。
- `iper_debug_001/`：单人极小调试切片。

---

### 3.12 `add/` —— 外部工具链、扩展流水线与离线环境
为保障主干环境纯净，外挂的模型预处理与动捕算法统一部署于此：
- `tools/`：集成最新 2D/3D 工具（`sam2/`, `dinov2/`, `multi-hmr/`, `SMPLest-X/`, `nvdiffrast-main/`, `dwpose_mimicmotion_official/`）。
- `envs/`：外挂独立 Python 环境定义（`smpl_clean`, `sapiens2_hf`, `multihmr` 等）。
- `outputs/`：离线批量处理生成的海量特征张量。

---

### 3.13 `archive/` —— 历史阶段归档与清理快照
封存历次架构重构被淘汰的废弃代码与备份：
- `cleanup_20261003/`、`cleanup_20261008/`、`cleanup_20261011/`：历史整理快照与原始结构备份。
- `home_tools_202609/`：早期家目录调试脚本归档。

---

### 3.14 `tests/` —— 自动化契约与回归测试套件
项目配备了严密的自动化测试工程，包含 60 多个自动化测试用例，覆盖：
- 零初始化行为契约（`test_deepgen_control_interface_v1.py`）
- 几何条件注入维度匹配（`test_condition_injector_v1.py`）
- Qwen21 权重继承与未破坏性回归（`test_qwen21_full_preservation.py`）
- 损失函数与五点对齐数学一致性（`test_qwen21_face_loss_alignment.py`）
- 数据集加载与随机增强鲁棒性（`test_tiktok_data_contract.py`）

---

### 3.15 `third_party/` —— 第三方开源依赖与组件
- `champ_guidance/`：CHAMP 3D 骨架渲染核心算法。
- `v65_face/`：V6.5 适配器中使用的开源人脸对齐组件。

---

### 3.16 `vram_lab/` —— 显存实验与测绘工具包
专用于显存峰值测量与模型吞吐分析的基准套件：
- `worker.py`：隔离子进程显存探针，捕捉 PyTorch 实际占用与分配峰值。
- `analysis.py` & `report.py`：自动输出图表与 Markdown 分析结论。

---

### 3.17 辅助与运行缓存目录
- `inputs/`：单人 `person1.jpg`、双人 `dual_person.jpg` 快速推理用例。
- `outputs/`：日常调试与交互式出图临时目录。
- `logs/`：训练中断恢复与守护进程运行日志。
- `caches/`：动态生成的条件潜码特征缓存。
- `patches/`：`deepgen_pose_control.patch` 记录对上游 Diffusers 的零侵入补丁。
- `deepgen/`：官方原版代码镜像库。

---

## 4. 核心工作流运行命令矩阵

所有任务均支持在单卡 RTX 6000D (85.6 GiB) 上直接运行：

```bash
# 激活推荐训练环境
source /home/shangguanrz/miniconda3/etc/profile.d/conda.sh
conda activate /home/shangguanrz/nihaochen/envs/qwen21
cd /home/shangguanrz/project/pic-edit
```

| 任务场景 | 推荐命令 | 对应输入 / 输出 |
|---|---|---|
| **Qwen21 Full 身体控制训练** | `python3 scripts/train/train_iper_qwen21_full.py --branch body --resolution 512` | 输入 iPER 样本对，输出 `experiments/` 检查点 |
| **Qwen21 Full 面部保持训练** | `python3 scripts/train/train_iper_qwen21_full.py --branch face --resolution 512 --face-aux-alignment five_point` | 继承 Body 权重，训练面部残差 |
| **Qwen21 Full 30步量化评测** | `python3 scripts/eval/eval_iper_qwen21_full.py --modes body body_face --resolution 512` | 输出 PCK, NME, ArcFace 余弦与对比图 |
| **Qwen21 Hi4D 双人动作训练** | `python3 scripts/train/train_hi4d_qwen21_dual.py --steps 500` | 输入 Hi4D 交互对，训练双人解耦 Token |
| **两阶段 BFS 高清换脸精修** | `python3 bfs_explore.py --pairs <pairs.jsonl> --resolution 768 --steps 30` | Stage 1 锁姿态 + Stage 2 BFS 局部修脸 |
| **自动化单元测试全量执行** | `pytest tests/test_qwen21_*.py -v` | 验证控制契约与数学正确性 |

---

## 5. 环境配置与资产路径规范

1. **Python 环境划分**：
   - **Qwen 2.1 主线推荐**：`/home/shangguanrz/nihaochen/envs/qwen21/bin/python`（PyTorch 2.8.0+cu128, Diffusers 0.41.0.dev0, Transformers 5.17.0）。
   - **DeepGen V6 主线**：`/home/shangguanrz/miniconda3/envs/deepgen/bin/python`。
2. **大文件挂载原则**：
   - 超过 500 MB 的预训练权重与海量数据集统一存放于 `/data/shangguanrz/`，通过软链接映射入 `models/` 和 `datasets/`，Git 提交时默认忽略，保证仓库轻量化。
"""
    return template.replace("__REPO_ROOT__", repo_root)

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "/home/shangguanrz/project/pic-edit/README.md"
    readme_text = build_master_readme()
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(readme_text)
    print(f"Master README.md generated successfully at: {out_file}")
