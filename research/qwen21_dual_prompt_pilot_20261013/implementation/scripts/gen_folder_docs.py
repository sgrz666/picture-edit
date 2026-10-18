#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 pic-edit 仓库的每一个非隐藏目录生成《目录说明.md》：目录定位 + 层级目录 + 文件内容摘要。

文件说明来自实际读取文件内容（Python 模块 docstring、Markdown 首个标题、JSON 顶层键、
Shell 注释等），二进制文件按扩展名与体积归类说明。已存在的 README.md 一律不覆盖。
"""

import os
import sys
import re
import json
import argparse
from collections import Counter

DOC_NAME = "目录说明.md"
INDEX_NAME = "目录总览.md"
SKIP_DIR_NAMES = {"__pycache__", "node_modules"}
MAX_LIST = 40
MAX_TEXT_READ = 60
HEAD_BYTES = 1500

ROOT = os.environ.get("PICEDIT_ROOT", "/home/shangguanrz/project/pic-edit")
DEFAULT_ROOT = ROOT

# ---------------------------------------------------------------- 扩展名分类
EXT_KIND = {
    ".py": "Python 源码", ".pyi": "Python 类型存根", ".sh": "Shell 脚本",
    ".md": "Markdown 文档", ".rst": "reST 文档", ".txt": "纯文本",
    ".json": "JSON 数据", ".jsonl": "JSONL 逐行数据",
    ".yaml": "YAML 配置", ".yml": "YAML 配置", ".toml": "TOML 配置",
    ".ini": "INI 配置", ".cfg": "配置文件", ".conf": "配置文件",
    ".csv": "CSV 表格", ".tsv": "TSV 表格",
    ".log": "运行日志", ".out": "标准输出记录", ".err": "错误输出记录",
    ".html": "HTML 页面", ".css": "CSS 样式", ".js": "JavaScript 脚本",
    ".ipynb": "Jupyter Notebook",
    ".png": "PNG 图像", ".jpg": "JPEG 图像", ".jpeg": "JPEG 图像",
    ".webp": "WebP 图像", ".gif": "GIF 图像", ".bmp": "BMP 图像", ".tif": "TIFF 图像",
    ".mp4": "MP4 视频", ".avi": "AVI 视频", ".mov": "MOV 视频", ".mkv": "MKV 视频",
    ".webm": "WebM 视频", ".zip": "ZIP 压缩包", ".tar": "tar 归档",
    ".gz": "gzip 压缩档", ".7z": "7z 压缩包", ".rar": "RAR 压缩包",
    ".pt": "PyTorch 张量/权重", ".pth": "PyTorch 权重", ".ckpt": "训练检查点",
    ".safetensors": "Safetensors 权重", ".bin": "二进制权重/数据块",
    ".onnx": "ONNX 模型", ".npy": "NumPy 数组", ".npz": "NumPy 压缩数组",
    ".pkl": "Pickle 序列化对象", ".pickle": "Pickle 序列化对象",
    ".joblib": "Joblib 序列化对象", ".h5": "HDF5 数据", ".hdf5": "HDF5 数据",
    ".obj": "Wavefront 三维网格", ".ply": "PLY 三维点云", ".glb": "glTF 三维模型",
    ".gltf": "glTF 三维模型", ".fbx": "FBX 三维模型", ".stl": "STL 网格",
    ".docx": "Word 文档", ".doc": "Word 文档", ".pptx": "PowerPoint 演示",
    ".xlsx": "Excel 表格", ".xls": "Excel 表格", ".pdf": "PDF 文档",
    ".svg": "SVG 矢量图", ".xml": "XML 数据", ".pickle": "Pickle 序列化对象",
    ".smpl": "SMPL 人体模型参数", ".smplx": "SMPL-X 人体模型参数",
    ".mha": "医学图像体积", ".nrrd": "NRRD 体积数据",
}

TEXT_EXTS = {
    ".py", ".pyi", ".sh", ".md", ".rst", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".csv", ".tsv", ".log", ".html", ".css", ".js",
    ".xml", ".ipynb",
}

# ---------------------------------------------------------------- 目录定位（人工核定）
PURPOSE = {
    ".": "项目根目录：基于 SMPL-X 三维人体表示的单/双人参考姿态图像编辑研究工程，收录取代方案、训练源码、实验产物、数据集与归档的全部工作线。",

    "src": "核心源码包：姿态控制适配器网络、SMPL-X 几何条件生成器、数据集读取管道与扩散管线集成代码。",
    "src/adapter": "早期 CNN 姿态适配器实现目录，`PoseConditionAdapter` 以零卷积初始化向 DiT 注入 8 通道控制特征。",
    "src/controls": "SMPL-X 控制信号生成器目录，从人体网格与姿态渲染姿态热图、法线图、深度图与接触图。",
    "src/data": "PyTorch 数据集与数据加载器目录，覆盖 iPER、TikTok、Hi4D 的配对样本读取与条件打包。",
    "src/evaluation": "定量评测工具目录：PSNR/SSIM/LPIPS 指标与 DWPose 关键点一致性评估。",
    "src/evaluation/dwpose_onnx": "DWPose 的 ONNX 推理实现目录，提供轻量 2D 全身关键点检测。",
    "src/integration": "扩散模型与 DiT 生成管线集成目录，负责外挂控制器与 Diffusers Pipeline 的挂接。",
    "src/pose_control": "姿态控制网络架构总目录，按 DeepGen V6、Qwen 2.1 简化版、Qwen 2.1 Full 三条演进线组织。",
    "src/pose_control/qwen21": "Qwen 2.1 简化姿态控制器：Body、Face、Dual 独立控制头与运行时封装。",
    "src/pose_control/qwen21_full": "Qwen 2.1 完整迁移控制器：把 DeepGen V6 全部组件迁移到 Qwen DiT，含 32 层 Hook 注入。",
    "src/pose_control/qwen21_full/detail": "Qwen 2.1 Full 细节控制分支：局部高频特征与残差补偿。",
    "src/pose_control/qwen21_full/face": "Qwen 2.1 Full 独立人脸 ROI 控制器：ArcFace + DINO 特征引导。",
    "src/pose_control/qwen21_full/hand": "Qwen 2.1 Full 独立手部 ROI 控制器：四手几何与可见性注意力融合。",
    "src/pose_control/qwen21_full/interface": "Qwen 2.1 Full 抽象接口与协议目录：控制器契约与输出结构定义。",
    "src/pose_control/qwen21_full/reasoning": "Qwen 2.1 Full 空间几何推理核心：单人几何、双人交叉注意力与接触推理。",
    "src/pose_control/v6": "DeepGen V6 Unified SMPL-X 适配器基准实现：条件编码、循环控制核心与残差投影。",
    "src/pose_control/v6/detail": "DeepGen V6 细节控制分支模块。",
    "src/pose_control/v6/face": "DeepGen V6.5 人脸增强模块：面部关键点编码与身份保持。",
    "src/pose_control/v6/hand": "DeepGen V6.6 手部控制模块：手指几何与可见性建模。",
    "src/pose_control/v6/interface": "DeepGen V6 控制器基础接口目录。",
    "src/pose_control/v6/reasoning": "DeepGen V6 几何与交互推理网络。",

    "scripts": "分类运行入口目录：train / eval / data / audit / reports / weights / run 七类可执行脚本。",
    "scripts/audit": "工程规范、权重结构与环境依赖审计脚本目录。",
    "scripts/data": "数据划分、清洗与 SMPL-X 条件离线预计算脚本目录。",
    "scripts/eval": "模型评测与推理验证脚本目录，含多模型消融对照。",
    "scripts/reports": "实验报告汇总、图表美化与对比面板合成脚本目录。",
    "scripts/run": "全流程实验编排与一键运行脚本目录。",
    "scripts/train": "训练启动包装脚本目录，含多卡配置与阶段切换。",
    "scripts/weights": "权重登记、校验与迁移脚本目录。",

    "configs": "全局配置目录：`asset_registry.json` 集中登记模型、数据、缓存与冷归档路径；`adapter_v6_architecture.json` 描述 V6 结构超参。",
    "tests": "自动化测试套件：适配器架构、零初始化行为、条件契约、数据完整性与训练/评测回归测试。",
    "docs": "项目文档目录：结构说明、架构文档、训练计划、对照实验结论与目录整理执行记录。",
    "docs/experiments": "实验专项说明与交接 runbook 目录。",
    "docs/superpowers": "工程方法论目录，收纳 plans（阶段计划）与 specs（接口规格）。",
    "docs/superpowers/plans": "架构重构与特性落地计划文档目录。",
    "docs/superpowers/specs": "系统设计规格说明书目录。",
    "reports": "成果报告目录：面向汇报的架构说明、量化结论与历史演示报告。",
    "research": "调研与审计证据目录：数据集选型、适配器审计、条件质量核查的历史快照。",
    "third_party": "第三方依赖与外部参考代码目录。",
    "third_party/champ_guidance": "Champ 姿态引导特征编码器封装。",
    "third_party/v65_face": "V6.5 人脸模块的外部依赖封装（StableAnimator、X-Dyna 等）。",
    "vram_lab": "显存实验包：preflight、显存测量、分析与验收报告的一体化 runner。",
    "patches": "对外部仓库的补丁文件目录。",
    "archive": "历史归档根目录：清理批次快照、旧家目录脚本与救援副本。",
    "deepgen": "上游 DeepGen-1.0-diffusers 官方仓库副本，未做项目化改动，仅作参考与训练实现来源。",

    "data": "小型调试样本数据目录（合成样本、单例过拟合数据、iPER 调试切片）。",
    "datasets": "训练与评测数据集本体目录：iPER、Hi4D、TikTok、3DPW、CHAMP 及各自的条件产物版本。",
    "models": "模型权重与依赖模型目录：DeepGen、Qwen-Image、DWPose、antelopev2、dinov2、SMPL 等。",
    "caches": "运行期条件缓存目录（如 Qwen2.1 Hi4D 控制信号缓存）。",
    "experiments": "实验工作区，每个实验一个独立目录，保存训练日志、检查点、条件缓存与评测产物。",
    "results": "早期评审结果与可视化 Demo 目录（显存分析图、review_20260908 系列）。",
    "outputs": "运行输出目录。",
    "logs": "跨实验共享的运行日志与失败诊断记录目录。",
    "inputs": "演示输入样图目录（单人/双人参考图）。",
    "add": "数据接入与外部资产工作区：新增数据集、环境包、模型、辅助工具与批量产出快照，体量大且多为不可再生的原始资产。",
}

# ---------------------------------------------------------------- 目录名关键词启发
NAME_HINTS = [
    (r"(^|/)site-packages($|/)", "Conda/虚拟环境中已安装的第三方 Python 包目录（非本项目代码）"),
    (r"(^|/)conda-meta($|/)", "Conda 环境包清单与历史元数据目录"),
    (r"(^|/)(bin|sbin)($|/)", "环境可执行程序目录"),
    (r"(^|/)include($|/)", "环境头文件目录"),
    (r"(^|/)lib(64)?/python[\d.]+", "环境 Python 运行库目录"),
    (r"(^|/)lib(64)?($|/)", "环境动态库目录"),
    (r"(^|/)share($|/)", "环境共享资源目录"),
    (r"site-packages", "第三方依赖包内部目录"),
    (r"(^|/)qc($|/)", "人工质检（QC）结果与抽查样本目录"),
    (r"pretrained_models", "预训练权重存放目录"),
    (r"(^|/)(img|images?)($|/)", "图像文件目录"),
    (r"(^|/)(vid|videos?)($|/)", "视频文件目录"),
    (r"(^|/)rgb($|/)", "RGB 原图帧目录"),
    (r"depth", "深度图产物目录"),
    (r"normal", "法线图产物目录"),
    (r"(^|/)(mask|segm|segmentation|semantic)", "分割/掩码产物目录"),
    (r"(dwpose|openpose|kps|keypoint|skeleton)", "人体关键点/骨架可视化产物目录"),
    (r"(smpl|body_model|mesh)", "SMPL(-X) 网格与参数产物目录"),
    (r"(camera|camparam|intrinsic|extrinsic)", "相机参数目录"),
    (r"checkpoint", "训练检查点目录"),
    (r"(^|/)ckpt", "训练检查点目录"),
    (r"cache", "预计算条件缓存目录"),
    (r"(^|/)logs?$", "运行日志目录"),
    (r"(^|/)log(s)?(_|$)", "运行日志目录"),
    (r"eval", "评测输出目录"),
    (r"(metric|result)", "评测指标与结果目录"),
    (r"(panel|grid|contact_sheet|sheet)", "对比面板拼图目录"),
    (r"(face|facial)", "人脸区域专项产物目录"),
    (r"(hand|finger)", "手部区域专项产物目录"),
    (r"(visual|vis_)", "可视化结果目录"),
    (r"preview", "预览图目录"),
    (r"sample", "样本切片目录"),
    (r"(^|/)raw", "原始未处理数据目录"),
    (r"processed", "经处理的数据目录"),
    (r"manifest", "数据清单与索引目录"),
    (r"(video|frames)", "视频帧数据目录"),
    (r"(report|summary)", "报告与总结目录"),
    (r"(train(ing)?)$", "训练过程产物目录"),
    (r"(^|/)data", "数据存放目录"),
    (r"(output|out)$", "输出结果目录"),
    (r"images?$", "图像文件目录"),
    (r"weights?$", "权重文件目录"),
    (r"(env|conda)", "运行环境配置目录"),
    (r"(tool|third)", "工具/第三方代码目录"),
    (r"archive", "归档目录"),
    (r"(cleanup|clean)", "目录清理记录归档"),
    (r"(audit|inspect)", "审计核查记录目录"),
    (r"(pilot|smoke|sanity)", "小规模验证性实验目录"),
    (r"(overfit)", "过拟合健壮性实验目录"),
    (r"(rescue)", "历史救援快照目录"),
]

OVERRIDE_BY_EXACT = {}

# ---------------------------------------------------------------- 补充核定（远端实测内容）
PURPOSE.update({
    # 数据集
    "datasets/iPER": "iPER 单人姿态数据集工作区：`iper_raw_1024` 为原生分辨率视频帧，`iper_assets_512_nvdiffrast` 为 nvdiffrast 渲染的 512 条件资产，`iper_sampled_6src64tgt` 为 6 源图 × 64 目标姿态的训练采样切片。",
    "datasets/iPER/iper_raw_1024": "iPER 原生 1024 分辨率源视频与解码帧，作为条件渲染与评测的高清基准。",
    "datasets/iPER/iper_assets_512_nvdiffrast": "iPER 在 512 分辨率下由 nvdiffrast 渲染的 SMPL-X 几何条件资产（法线/深度/姿态/掩码）。",
    "datasets/iPER/iper_sampled_6src64tgt": "iPER 训练用配对采样：每个源人物 6 张源图配 64 个目标姿态，是 Qwen2.1 Full 主线训练数据来源。",
    "datasets/TikTokDataset": "TikTok 单人舞蹈数据集工作区：原始视频、v2/v3 构建版本、3D 几何资产、划分清单与展示样本。",
    "datasets/TikTokDataset/TikTok_Raw_Videos": "TikTok 原始舞蹈视频文件。",
    "datasets/TikTokDataset/TikTok_v2": "TikTok 第二版构建结果（抽帧、DWPose 与几何补全的中间版本）。",
    "datasets/TikTokDataset/TikTok_v3": "TikTok 第三版构建结果，补齐缺失几何后的当前训练版本。",
    "datasets/TikTokDataset/TikTok_3d_assets_native": "TikTok 原生分辨率三维几何资产（SMPL-X 拟合、深度与法线）。",
    "datasets/TikTokDataset/splits": "TikTok 训练/验证/测试划分清单。",
    "datasets/Hi4D_pilot_v1": "Hi4D 双人交互 Pilot 数据工作区：`native_interaction_v1..v4_2` 记录双人配对与接触关系的多轮迭代，`final_assets_*` 为定稿资产与清单。",
    "datasets/Hi4D_pilot_v1/final_manifests": "Hi4D Pilot 定稿数据清单（配对索引与统计摘要 JSON/JSONL）。",
    "datasets/Hi4D_pilot_v1/processed_512_1500": "Hi4D 512 分辨率、1500 帧规模的处理结果。",
    "datasets/3dpw_single_final_v2": "3DPW 定稿数据集 v2：`rgb/depth/normal/mask/dwpose` 五路对齐条件、train/validation/test 配对 CSV 清单，以及多种抽样可视化目录。",
    "datasets/champ_motions_example": "CHAMP 官方示例动作序列与参考图，用于姿态引导的接口验证。",
    "datasets/iPER_sample": "iPER 小规模样例切片，用于快速冒烟测试与格式核对。",

    # 模型资产
    "models/DeepGen-1.0-diffusers": "DeepGen-1.0-diffusers 基座权重仓库（transformer/vae/text_encoder/scheduler/tokenizer 组件目录），并含项目改造后的 `deepgen_pipeline.py` 及控制注入前向的备份。",
    "models/Qwen-Image-2.1": "Qwen-Image-2.1 基座权重仓库（DiT transformer、VAE、text encoder 与 processor 分片）。",
    "models/Qwen-Image-Edit-20B": "Qwen-Image-Edit-20B 官方权重，作为质量上界与教师模型，仅做显存可行性分析与对照采样。",
    "models/DWPose": "DWPose 全身 133 关键点检测模型（ONNX 权重）。",
    "models/antelopev2": "InsightFace antelopev2 人脸识别/检测模型包，用于人脸身份保真度评估。",
    "models/dinov2-giant": "DINOv2-Giant 视觉特征模型，用于外观与服饰一致性特征比对。",
    "models/smpl": "SMPL/SMPL-X 人体统计形状模型与模板网格资产。",
    "models/face_eval": "人脸评测专用的辅助模型与配置集合。",

    # 数据接入工作区
    "add/data": "接入期原始数据暂存区（3DPW 源数据与 iPER 视频统计）。",
    "add/data/3DPW": "3DPW 原始下载数据与序列文件。",
    "add/datasets": "3DPW/HumanVid 接入期的数据读取脚本与数据集副本。",
    "add/models": "接入期使用的三维人体与分割模型（multiHMR、Sapiens 分割、SMPL 清理版、YOLO11 姿态/分割权重）。",
    "add/envs": "为接入工具链单独建立的 Conda 环境定义与依赖记录（deepgen_pose、multihmr、sapiens2_hf、smpl_clean、smplestx_bw）。",
    "add/tools": "数据接入工具链：HumanVid 处理流水线（编号 01–18 的阶段脚本）与第三方几何/姿态库副本。",
    "add/outputs": "数据接入的批量产出快照：3DPW 与 HumanVid 各版本的资产、审计、配对与质检结果。",
    "add/logs": "接入作业的远端运行日志留存目录。",
    "add/migration": "目录迁移与整理期的映射记录、搬迁清单。",
    "add/cleanup_records": "接入区清理过程的核查记录与保留决定。",
    "add/envs/deepgen_pose": "项目主训练环境的 Conda 副本（PyTorch + Diffusers + SMPL-X 工具链），随数据盘一同迁移以便复现实验。",
    "add/envs/multihmr": "MultiHMR 多人体三维姿态估计推理环境的 Conda 副本。",
    "add/envs/sapiens2_hf": "Sapiens2 人体分割/关键点模型的 HuggingFace 版运行环境副本。",
    "add/envs/smpl_clean": "SMPL 网格清洗与拟合工具运行环境的 Conda 副本。",
    "add/envs/smplestx_bw": "SMPLest-X 人体网格拟合环境的 Conda 副本。",
    "add/tools/sam2": "Segment Anything Model 2 官方仓库副本，用于人体与部位分割。",
    "add/tools/nvdiffrast-main": "nvdiffrast 可微光栅化库源码副本，SMPL-X 法线/深度条件渲染依赖它。",
    "add/tools/multi-hmr": "MultiHMR 官方仓库副本（多人体带形状三维姿态估计）。",
    "add/tools/dinov2_e1277af": "DINOv2 官方仓库副本，用于外观与服饰一致性特征提取。",
    "add/tools/dwpose_mimicmotion_official": "DWPose 官方推理代码与权重副本（全身 133 关键点检测）。",
    "add/tools/SMPLest-X": "SMPLest-X 官方仓库副本与预训练权重（人体网格拟合）。",
    "add/tools/3dpw_official_readme_demo": "3DPW 官方 README 示例代码副本，用于核对数据格式。",

    # 调研与审计
    "research/bfs_transfer_audit_20261004": "BFS（Body-Face-Skeleton）控制移用性审计：768 分辨率探针、LoRA 有无对照与 CPU 兼容性核验的 JSON 结论与对比面板。",
    "research/smplx_condition_audit_20261008": "SMPL-X 条件输入质量审计：身体条件张量与渲染输入的核查记录。",
    "research/deepgen_assessment": "DeepGen-1.0-diffusers 纯基座生成能力评估：不同提示词与采样配置下的原始渲染样本。",
    "research/cleanup_20261003": "2026-10-03 目录大清理的现场快照、迁移对照与备份清单。",
    "research/tiktok_training_20261003": "TikTok 数据集接入与训练方案调研：基线代码快照与适配器增量对比。",
    "research/legacy_202609": "2026 年 9 月历史调研与实验证据归档。",

    # 报告与日志
    "reports/qwen21": "Qwen 2.1 线成果报告目录：面向组会与交付的 docx/md 对比结果、Lark 汇报指标 JSON 与配图资产。",
    "reports/qwen21/assets": "Qwen 2.1 报告引用的图表与对比图像资产。",
    "reports/legacy_demo_202609": "2026 年 9 月演示阶段报告与对比图归档。",
    "logs/legacy": "历史数据处理与审计作业的命令行日志留存（Hi4D 构建、TikTok 抽帧质检、DWPose 失败诊断等）。",
    "archive/home_tools_202609": "2026 年 9 月家目录散落脚本归档：历史 Shell 与一次性诊断 Python 脚本，家目录以符号链接回指。",
    "archive/home_tools_202609/scratch": "一次性诊断与调试脚本集合（DWPose、网格、掩码、指标、边框等问题排查）。",
    "archive/home_tools_202609/historical_shell": "历史批处理与训练/评测启动 Shell 脚本。",
    "archive/cleanup_20261003": "2026-10-03 清理批次的备份暂存、待判清单与清理工具。",
    "archive/cleanup_20261008": "2026-10-08 清理批次的产出与冒烟结果暂存。",
    "archive/add_picture_edit_rescue_20260921": "2026-09-21 `add_picture_edit` 工作区救援快照与冒烟输出。",

    # 结果与实验
    "results/review_20260908": "2026-09-08 初版 CNN 适配器验收评审：运行记录、显存实测与 Demo 输出。",
    "results/review_20260908_v2": "2026-09-08 正式版验收评审：111 项核心验收、17 项自动化测试与精确复现哈希记录。",
    "results/legacy_demo_202609": "演示阶段的对比图、控制图与演示报告归档。",
    "experiments/review_20260908_v2": "vram_lab 验收实验目录：显存测量、preflight 与 acceptance 全流程产物。",
    "experiments/champ_deepgen_formal_v1": "CHAMP 姿态引导 + DeepGen 基座的正式单样本过拟合实验。",
    "experiments/deepgen_no_control_ablation_v1": "无控制信号消融基线实验，用于确认适配器接入前后基座输出一致性。",
    "experiments/iper_v6_native_20k_v1": "DeepGen V6 原生分辨率 20k 步主线训练实验目录。",
    "experiments/iper_v6_comparative_3000": "DeepGen V6 3000 步横向对照实验。",
    "experiments/iper_v64_detail_overfit_024_6": "V6.4 细节控制分支过拟合实验（24 源 × 6 目标）。",
    "experiments/iper_v65_face_overfit_024_6": "V6.5 人脸保持过拟合实验（24 源 × 6 目标）。",
    "experiments/qwen21_iper_body": "Qwen 2.1 简化版 Body 控制器在 iPER 上的训练实验。",
    "experiments/qwen21_hi4d_dual": "Qwen 2.1 简化版 Dual 控制器在 Hi4D 双人交互上的训练实验。",
    "experiments/qwen21_tiktok_v3_smoke": "Qwen 2.1 在 TikTok v3 数据上的冒烟训练实验。",
    "experiments/qwen-2.1-test": "原生 Qwen-Image-2.1 能力评测实验：提示词遵循、单/双人编辑与显存特征。",
    "experiments/tiktok_overfit_sanity_full": "TikTok 全量过拟合健壮性实验（两阶段训练），含检查点、条件缓存与阶段预览。",
    "experiments/tiktok_overfit_sanity_smoke": "TikTok 过拟合健壮性冒烟版实验。",
    "experiments/tiktok_00001_condition_repair": "TikTok 00001 单样本几何条件错位的修复验证与多条件消融推理。",
    "experiments/tiktok_objective_eval": "TikTok 客观指标评测实验目录。",
    "experiments/v6_condition_inspection": "V6 控制条件可视化检查实验：核对姿态/法线/深度/掩码渲染是否对齐。",
    "experiments/hi4d_overfit_archive": "Hi4D 早期过拟合实验归档。",
    "experiments/qwen_edit_results": "Qwen-Edit 基线对照结果目录。",
    "caches/qwen21_hi4d_controls": "Qwen 2.1 在 Hi4D 上的预计算控制条件张量缓存。",
    "data/champ_overfit": "CHAMP 单样本过拟合调试数据（深度/DWPose/法线/语义图条件）。",
    "data/debug_samples": "合成调试样本：控制图与图像对，由 `create_debug_samples.py` 生成。",
    "data/iper_debug_001": "iPER 单条调试样本的控制条件与图像切片。",
    "patches": "对上游 DeepGen 仓库的补丁文件（`deepgen_pose_control.patch`），记录控制注入的最小改动。",
    "inputs": "演示输入样图目录：单人 `person1.jpg` 与双人 `dual_person.jpg` 参考图。",
    "outputs": "临时运行输出目录。",
    "configs": "全局配置目录：`asset_registry.json` 集中登记基座模型、数据、缓存与冷归档路径；`adapter_v6_architecture.json` 描述 V6 适配器结构超参。",

    # 仓库内经符号链接接入的数据盘目录
    "datasets/Hi4D-origin": "Hi4D 官方原始数据（实体在 `/data` 数据盘，仓库以符号链接接入）：多视角双人交互序列原样存放，是 Pilot 配对资产的来源。",
    "datasets/TikTokDataset/model": "TikTok 接入流程所用的检测/分割模型副本（YOLO、DWPose 等）。",

    # /data/shangguanrz 数据盘视角
    "pic-edit-datasets": "数据集实体存放区：TikTok 与 Hi4D 原始数据体积最大，仓库 `datasets/` 下的同名目录是以符号链接指向这里的入口。",
    "pic-edit-datasets/TikTokDataset": "TikTok 单人舞蹈数据集实体：原始视频、逐 clip 的抽帧/DWPose/DensePose/三维资产、v2/v3 构建版本与划分清单。",
    "pic-edit-datasets/Hi4D-origin": "Hi4D 官方原始数据实体：多视角双人交互序列、标定与注释文件。",
    "pic-edit-models": "大体积模型权重实体区：Qwen-Image-2.1 正式版与 PE-I2I 变体，供仓库 `models/` 链接使用。",
    "pic-edit-models/Qwen-Image-2.1": "Qwen-Image-2.1 基座权重实体（DiT transformer、VAE、text encoder 分片）。",
    "pic-edit-models/Qwen-Image-2.1-PE-I2I": "PE-I2I 报告流水线所用的 Qwen-Image-2.1 权重副本，与主线训练分开存放。",
    "pic-edit-caches": "预计算控制条件缓存实体区：训练时直接读取的 SMPL-X 几何条件张量缓存，可重复生成因此单独存放。",
    "pic-edit-caches/qwen21_hi4d_controls": "Qwen 2.1 在 Hi4D 上的预计算控制条件缓存实体。",
    "pic-edit-experiments": "进行中的实验产物区：一次训练/评测运行的代码快照、输入清单、日志与审计记录，内容与规模随运行持续变化，因此只按目录实际内容自动画像，不做人工核定描述。",
    "pic-edit-archive": "2026-10-03 目录整理时移出的冷归档：旧实验、旧模型与原始数据快照，仓库内原路径以符号链接保持可访问。",
    "pic-edit-archive/cleanup_20261003": "10-03 清理移出内容：`experiments/` 为旧实验产物，`models/` 为 Qwen-Image-Edit-20B 权重，`raw/` 为原始数据。",
    "pic-edit-archive/SynBody_inspection_20261003": "SynBody 合成人体数据的视觉质量检查留档（纹理真实感与多视角一致性评估）。",
    "dataset_samples": "公开人体数据集抽样展示区：ABC、MVHumanNet、HumanVid 等数据集的样本切片与清单，用于选型调研。",
    "dataset_samples/MANIFEST.txt": "抽样展示区各数据集样本的来源、规模与许可说明清单。",
    "add_migration": "`add/` 数据从 AutoDL 实例迁移到本数据盘的脚本与传输日志。",
})


FOLLOW_UNDER = []
EXCLUDE_UNDER = []

# ---------------------------------------------------------------- 数据集构成画像（--dataset-profile）
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset_notes import NOTES as DATA_NOTES, CHILD_DESC, FIELD_HINTS
except Exception:                                        # 允许单文件部署
    DATA_NOTES, CHILD_DESC, FIELD_HINTS = {}, {}, {}

PROFILE = False
OWN = {}         # rel -> (直属文件数, 直属字节, Counter 扩展名)
ROLL = {}        # rel -> [子树文件数, 子树字节, Counter 扩展名]
CN_NUM = "一二三四五六七八九十"

TEXT_NOTE_FILES = (".json", ".jsonl", ".csv", ".tsv", ".txt")


def build_profile_index(dirs):
    OWN.clear(); ROLL.clear()
    for rel, absdir, dnames, fnames in dirs:
        cnt = byt = alloc = 0
        ext = Counter()
        for f in fnames:
            if f in (DOC_NAME, INDEX_NAME):
                continue
            try:
                st = os.stat(os.path.join(absdir, f))
            except OSError:
                st = None
            if st is None:
                continue
            cnt += 1; byt += st.st_size
            alloc += getattr(st, "st_blocks", 0) * 512
            ext[os.path.splitext(f)[1].lower() or "(无后缀)"] += 1
        OWN[rel] = (cnt, byt, ext, alloc)
    for rel, (c, b, e, a) in OWN.items():
        cur = rel
        while True:
            t = ROLL.setdefault(cur, [0, 0, Counter(), 0])
            t[0] += c; t[1] += b; t[3] += a
            for k, v in e.items():
                t[2][k] += v
            if cur == ".":
                break
            cur = os.path.dirname(cur) or "."


def child_census(absdir, dnames, max_items=25):
    """统计兄弟项内部常见的子目录名与特殊文件名，用于自动描述该层的数据布局。"""
    dirs_c = Counter(); files_c = Counter(); sampled = 0
    for d in dnames[:max_items]:
        try:
            cd, cf, _cl = entries(os.path.join(absdir, d))
        except OSError:
            continue
        sampled += 1
        for x in cd:
            dirs_c[x] += 1
        for x in cf:
            if x in (DOC_NAME, INDEX_NAME):
                continue
            if re.match(r"^\d+", x):
                continue
            files_c[x] += 1
    return dirs_c, files_c, sampled


def numeric_pattern(names):
    """识别 NNNN.png / 00001 这类连续编号，返回 (数量, 起, 止, 缺号数)。"""
    nums = []
    for n in names:
        base = os.path.splitext(n)[0]
        m = re.match(r"^(?:[A-Za-z]+[-_])?0*(\d+)$", base)
        if m:
            nums.append(int(m.group(1)))
    if len(nums) < 3:
        return None
    lo, hi = min(nums), max(nums)
    missing = (hi - lo + 1) - len(set(nums))
    return len(nums), lo, hi, missing


def read_records(path, ext, exact_max=16_000_000, probe=16_000_000):
    """返回 (记录数 或 None, 字段列表, 备注)。小文件精确统计，大文件按首段行均字节估算。"""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None, [], "不可读"
    if size == 0:
        return 0, [], "空文件"
    if ext not in (".jsonl", ".csv", ".tsv", ".json"):
        return None, [], None
    first = b""; n = 0; tail = b"\n"; est = None
    try:
        with open(path, "rb") as fh:
            if size > exact_max and ext in (".jsonl", ".csv", ".tsv"):
                buf = fh.read(probe)
                cut = buf.rfind(b"\n")
                if cut <= 0:
                    return None, [], "首段无完整记录"
                first = buf[:4096]
                lines = buf.count(b"\n", 0, cut)
                avg = cut / max(1, lines)
                est = int(round(size / avg))
                n = est
                note_kind = "记录数为估算值（按首 %d 行均 %.0f 字节推算）" % (lines, avg)
                if est < 1000:
                    note_kind = None
                    n = lines
            else:
                note_kind = None
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    if not first:
                        first = chunk[:4096]
                    n += chunk.count(b"\n")
                    tail = chunk[-1:]
    except OSError:
        return None, [], "读取失败"
    if note_kind is None and tail not in b"\n\r" and ext in (".jsonl", ".csv", ".tsv"):
        n += 1                      # 末行无换行符时不计入 count(b"\n")
    text = first.decode("utf-8", "replace")
    if ext in (".jsonl",):
        line = next((l for l in text.splitlines() if l.strip()), "")
        keys = []
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                keys = list(obj.keys())
        except Exception:
            keys = re.findall(r'"([^"]{1,40})"\s*:', line)
        return n, list(dict.fromkeys(keys))[:24], note_kind
    if ext in (".csv", ".tsv"):
        line = next((l for l in text.splitlines() if l.strip()), "")
        sep = "\t" if ext == ".tsv" else ","
        cols = [c.strip().strip('"') for c in line.split(sep) if c.strip()]
        header = bool(cols) and not re.match(r"^\s*[{[]", line)
        rec = (n - 1 if header and n else n)
        extra = ("表头 %d 列" % len(cols) if header else None)
        if note_kind:
            extra = "；".join(x for x in (extra, note_kind) if x)
        return rec, cols[:24], extra
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = fh.read(2_000_000).lstrip()
    except OSError:
        return None, [], "读取失败"
    if head.startswith("["):
        keys = re.findall(r'"([^"]{1,40})"\s*:(?=[^"]*$)', head[:2000], re.M)
        keys += re.findall(r'^\s{2,8}"([^"]{1,40})"\s*:', head, re.M)
        return n, list(dict.fromkeys(keys))[:24], "JSON 数组"
    keys = re.findall(r'^\s{0,8}"([^"]{1,40})"\s*:', head, re.M)
    return None, list(dict.fromkeys(keys))[:24], "JSON 对象"


def scalar_markers(absdir, fnames):
    """读取 _ASSETS_DONE.json / summary.json 之类的小元数据，摘出标量字段。"""
    out = []
    cands = [f for f in fnames if f.startswith("_") and f.endswith(".json")]
    cands += [f for f in fnames if f in ("README.json", "stats.json", "summary.json")
              or f.endswith("_summary.json")]
    for f in dict.fromkeys(cands):
        p = os.path.join(absdir, f)
        try:
            if os.path.getsize(p) > 200_000:
                continue
            with open(p, encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        pairs = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                pairs.append("%s(%d 项)" % (k, len(v)))
            else:
                pairs.append("%s=%s" % (k, str(v)[:60]))
        out.append((f, pairs[:12]))
    return out


def render_profile(rel, absdir, dnames, fnames):
    lines = []
    own = OWN.get(rel)
    roll = ROLL.get(rel)
    if own:
        c, b, ext, _a = own
        if c:
            lines.append("- **直属类型分布**：%s" % (
                "、".join("%s×%d" % (kind(e), n) for e, n in ext.most_common(6))))
    if roll:
        c, b, ext, _a = roll
        if c:
            lines.append("- **子树类型构成**：%s" % (
                "、".join("%s×%s" % (kind(e), "{:,}".format(n)) for e, n in ext.most_common(8))))
    pat = numeric_pattern(fnames)
    if pat:
        n, lo, hi, miss = pat
        lines.append("- **命名与编号**：数值编号文件 %d 个，覆盖 `%04d`–`%04d`，%s" % (
            n, lo, hi, "编号连续无缺号" if miss == 0 else "缺号 %d 个（多为构建期判定剔除）" % miss))
    if len(dnames) >= 3:
        dc, fc, sampled = child_census(absdir, dnames)
        dc = Counter({k: v for k, v in dc.items() if v >= 2})
        fc = Counter({k: v for k, v in fc.items() if v >= 2})
        if dc or fc:
            lines.append("- **兄弟项内部布局**（抽样 %d/%d 个子目录，统计每项共有内容的出现次数）：" % (
                sampled, len(dnames)))
            lines.append("")
            lines.append("  | 项内条目 | 出现于 | 含义 |")
            lines.append("  |---|---|---|")
            for name, cnt in dc.most_common(12):
                lines.append("  | `%s/` | %d/%d | %s |" % (
                    name, cnt, sampled, CHILD_DESC.get(name, "—")))
            for name, cnt in fc.most_common(8):
                lines.append("  | `%s` | %d/%d | %s |" % (
                    name, cnt, sampled, CHILD_DESC.get(name, "每项自带的元数据/索引文件")))
            lines.append("")
    meta = []
    for f in fnames[:600]:
        ext = os.path.splitext(f)[1].lower()
        if ext in (".jsonl", ".csv", ".tsv", ".json"):
            n, cols, note = read_records(os.path.join(absdir, f), ext)
            if cols or n is not None:
                meta.append((f, ext, n, cols, note))
        if len(meta) >= 24:
            break
    if meta:
        lines.append("- **清单/元数据文件字段**：")
        lines.append("")
        lines.append("  | 文件 | 类型 | 记录数 | 字段 |")
        lines.append("  |---|---|---|---|")
        for f, ext, n, cols, note in meta[:10]:
            if note and not cols:
                lines.append("  | `%s` | %s | — | %s |" % (f, ext.lstrip(".").upper(), note))
                continue
            shown = "、".join("`%s`" % c for c in cols[:14]) + ("…" if len(cols) > 14 else "")
            rec = "—" if n is None else "{:,}".format(n)
            if note:
                rec = "%s%s" % ("≈" if "估算" in note else "", rec or "—")
                shown = (shown + "　｜" + note) if shown else note
            lines.append("  | `%s` | %s | %s | %s |" % (f, ext.lstrip(".").upper(), rec, shown or "—"))
            if cols:
                gloss = ["%s=%s" % (c, FIELD_HINTS[c]) for c in cols if c in FIELD_HINTS][:10]
                if gloss:
                    lines.append("  | 　↳ 字段释义 | | | %s |" % "；".join(gloss))
        if len(meta) > 10:
            lines.append("  | *……其余 %d 个清单文件* | | | |" % (len(meta) - 10))
        lines.append("")
    marks = scalar_markers(absdir, fnames)
    if marks:
        lines.append("- **构建状态/摘要标记**：")
        for f, pairs in marks[:4]:
            lines.append("  - `%s`：%s" % (f, "；".join(pairs)))
    if not lines:
        return []
    return lines


def r(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def is_skip(name):
    return name.startswith(".") or name in SKIP_DIR_NAMES


def under_any(path, prefixes):
    """path 是否等于或位于 prefixes 之内（prefixes 元素以 / 结尾，按路径边界匹配）。"""
    for p in prefixes:
        if path == p.rstrip("/") or path.startswith(p):
            return True
    return False


def allowed_link(path):
    """符号链接是否允许进入：仅当目标落在 --follow-under 指定的前缀之内。"""
    if not FOLLOW_UNDER:
        return False
    try:
        tgt = os.path.realpath(path)
    except OSError:
        return False
    return under_any(tgt, FOLLOW_UNDER) and not under_any(tgt, EXCLUDE_UNDER)


def entries(d):
    """返回 (可见子目录, 可见文件, 仓库外符号链接)。允许进入的链接按普通目录对待。"""
    dirs, files, links = [], [], []
    try:
        with os.scandir(d) as it:
            for e in it:
                if is_skip(e.name):
                    continue
                try:
                    if e.is_symlink():
                        if e.is_dir(follow_symlinks=True):
                            if allowed_link(e.path):
                                dirs.append(e.name)
                            else:
                                links.append(e.name)
                        continue
                    if e.is_dir(follow_symlinks=False):
                        dirs.append(e.name)
                    elif e.is_file(follow_symlinks=False):
                        files.append(e.name)
                except OSError:
                    continue
    except OSError:
        pass
    return sorted(dirs), sorted(files), sorted(links)


def link_note(absdir, name):
    try:
        tgt = os.path.realpath(os.path.join(absdir, name))
    except OSError:
        tgt = "?"
    return "符号链接 → `%s`（实体在本仓库之外，随挂载位置变化；未在其内部生成说明）" % tgt


def kind(ext):
    return EXT_KIND.get(ext, (ext.upper().lstrip(".") or "无后缀") + " 文件")


def human(nbytes):
    # 十进制单位，与 df / du --si 以及《数据集构成详解》中的容量口径保持一致
    for unit, div in (("TB", 1000**4), ("GB", 1000**3), ("MB", 1000**2), ("KB", 1000)):
        if nbytes >= div:
            return "%.1f %s" % (nbytes / div, unit)
    return "%d B" % nbytes


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("|", "/").strip("#*:> ").strip()


def summarize_file(path, name):
    """实际读取文件内容头部，产出一句话内容说明。"""
    ext = os.path.splitext(name)[1].lower()
    if ext not in TEXT_EXTS:
        return ""
    try:
        limit = 9000 if ext in (".py", ".pyi") else HEAD_BYTES
        with open(path, "rb") as f:
            raw = f.read(limit)
    except OSError:
        return "（不可读）"
    text = raw.decode("utf-8", "replace")
    if ext in (".py", ".pyi"):
        body = text
        while body.lstrip().startswith("#"):
            body = body.lstrip()
            nl = body.find("\n")
            body = "" if nl < 0 else body[nl + 1:]
        body = body.lstrip()
        doc = ""
        m = re.match(r'(?:[rRbBuU]{0,2})("""|\'\'\'|")', body)
        if m:
            q = m.group(1)
            rest = body[m.end():]
            end = rest.find(q)
            doc = clean(rest if end < 0 else rest[:end])[:90]
        names = re.findall(r'^(?:class|def)\s+([A-Za-z_]\w*)', text, re.M)
        bits = []
        if doc:
            bits.append("模块 docstring：" + doc)
        if names:
            uniq = list(dict.fromkeys(names))
            bits.append("顶层定义 %d 个（%s%s）" % (
                len(names), ", ".join("`%s`" % n for n in uniq[:4]),
                "…" if len(uniq) > 4 else ""))
        if not bits:
            imports = re.findall(r'^(?:from|import)\s+([\w.]+)', text, re.M)
            bits.append("依赖/导入：" + ", ".join(imports[:4]) if imports
                        else "空模块或仅注释/常量模块")
        return "；".join(bits)
    if ext in (".md", ".rst"):
        for l in text.splitlines():
            l = l.strip()
            if not l:
                continue
            if l.startswith("<!--"):
                continue
            return "首个标题/正文：" + clean(l)[:90]
        return "空文档"
    if ext in (".json", ".jsonl"):
        if ext == ".jsonl":
            first = next((l for l in text.splitlines() if l.strip()), "")
            return "逐行 JSON 记录，首行：" + clean(first)[:80] if first else "空 JSONL 文件"
        try:
            keys = re.findall(r'^\s{0,8}"([^"]{1,40})"\s*:', text, re.M)
            if keys:
                return "顶层键：" + ", ".join(list(dict.fromkeys(keys))[:8])
        except Exception:
            pass
        return "JSON 数据"
    if ext in (".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"):
        for l in text.splitlines():
            l = l.strip()
            if l and not l.startswith("#"):
                return "首个配置项：" + clean(l)[:80]
        return "配置文件"
    if ext == ".sh":
        for l in text.splitlines():
            l = l.strip()
            if l.startswith("#") and not l.startswith("#!"):
                return "脚本注释：" + clean(l)[:80]
            if l and not l.startswith("#"):
                return "起始命令：" + clean(l)[:80]
        return "Shell 脚本"
    if ext == ".log":
        lines = [l for l in text.splitlines() if l.strip()]
        return "日志，首行：" + (clean(lines[0])[:80] if lines else "空日志")
    if ext in (".csv", ".tsv"):
        lines = [l for l in text.splitlines() if l.strip()]
        return "表头：" + (clean(lines[0])[:80] if lines else "空表")
    if ext == ".txt":
        lines = [l for l in text.splitlines() if l.strip()]
        return "首行：" + (clean(lines[0])[:80] if lines else "空文本")
    if ext == ".html":
        m = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
        if m:
            return "页面标题：" + clean(m.group(1))[:80]
        return "HTML 页面"
    if ext == ".ipynb":
        m = re.search(r'"cell_type"\s*:\s*"([^"]+)"', text)
        return "Jupyter Notebook" + ("，首单元 %s" % m.group(1) if m else "")
    return kind(ext)


def folder_purpose(relpath, dnames, fnames):
    if relpath in PURPOSE:
        return PURPOSE[relpath]
    leaf = relpath.split("/")[-1]
    parts = [p for p in relpath.split("/") if p]
    for i in range(1, len(parts) - 1):
        suffix = "/".join(parts[i:])
        for key in sorted(PURPOSE, key=lambda k: -k.count("/")):
            if key.endswith("/" + suffix):
                return PURPOSE[key]
    if PROFILE and leaf and re.fullmatch(r"[0-9]{2,8}", leaf):
        return ("编号 `%s` 的样本/序列数据项目录，内部子目录与文件即该样本的条件资产与元数据；"
                "逐项构成见本文件「构成画像」，整体口径见上层《数据集构成详解》。" % leaf)
    low = relpath.lower()
    for pat, desc in NAME_HINTS:
        if re.search(pat, low):
            return desc
    parent = os.path.dirname(relpath)
    if parent in PURPOSE:
        return "%s 下按内容自然分组形成的子目录。" % PURPOSE[parent][:40].split("：")[0]
    return "项目工作目录，收纳 %d 个子目录与 %d 个文件。" % (len(dnames), len(fnames))


def link_target(relpath):
    depth = len([p for p in relpath.split("/") if p])
    return "../" * depth


STATUS_FILES = ("_ASSETS_DONE.json", "_FAILED.json", "_DONE.json", "_STATUS.json", "_COMPLETE.json")


def kid_build_status(absdir, name):
    """读取编号子项自带的构建标记文件，摘要成一行，便于在上层目录纵览整批构建结果。"""
    for fn in STATUS_FILES:
        p = os.path.join(absdir, name, fn)
        try:
            if os.path.getsize(p) > 200_000:
                continue
            with open(p, encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        st = data.get("status") or data.get("state")
        if not st:
            continue
        bits = ["构建标记 `%s`" % clean(str(st))[:20]]
        frames = next((data[k] for k in ("frame_count", "frames", "expected_frames")
                       if isinstance(data.get(k), int)), None)
        if frames is not None:
            bits.append("帧数 %d" % frames)
        res = data.get("resolution")
        if isinstance(res, str) and len(res) <= 40:
            bits.append("分辨率 %s" % clean(res))
        err = data.get("error") or data.get("reason")
        if isinstance(err, str) and err.strip():
            bits.append("判定依据 %s" % clean(err)[:70])
        return "，".join(bits)
    return ""


def overview_profile(rel):
    """目录定位段落的规模概览：直属/子树逻辑体积与真实占盘，用于暴露稀疏文件差异。"""
    out = []
    own = OWN.get(rel)
    roll = ROLL.get(rel)
    if own:
        c, b, ext, a = own
        tail = "（仅子目录）" if c == 0 else ""
        out.append("- **本级内容**：文件 %s 个，%s%s" % ("{:,}".format(c), human(b), tail))
    if roll:
        c, b, ext, a = roll
        out.append("- **子树规模**：文件 %s 个，逻辑体积 %s" % ("{:,}".format(c), human(b)))
        if a and b and a < b * 0.9:
            out.append("- **稀疏性**：子树 %s 个文件，逻辑体积 %s，实占仅 %s（低 %.1f%%）"
                        "——大量内容为稀疏/共享块文件；`目录说明.md` 中的文件数按逻辑内容统计，磁盘规划请以实占为准" % (
                            "{:,}".format(c), human(b), human(a), (1 - a / b) * 100))
    return out


def build_doc(relpath, dnames, fnames, absdir):
    dnames, fnames, lnames = entries(absdir)
    fnames = [f for f in fnames if f not in (DOC_NAME, INDEX_NAME)]
    name = relpath.split("/")[-1] if relpath != "." else os.path.basename(ROOT)
    purpose = folder_purpose(relpath, dnames + lnames, fnames)
    depth = len(relpath.split("/")) if relpath != "." else 0
    up = link_target(relpath)

    lines = ["# `%s` 目录说明" % (relpath if relpath != "." else "/"), ""]
    lines.append("> 本文件由 `scripts/gen_folder_docs.py` 依据目录实际内容自动生成；"
                 "如需补充人工说明，请写在本目录 `README.md` 中（生成器不会覆盖 README）。")
    lines.append("")
    crumb = "- **路径**：`%s`" % (relpath if relpath != "." else "./")
    crumb += "　- **层级**：第 %d 级" % (depth + 1)
    if relpath != ".":
        crumb += "　- **上级**：[%s](%s)" % (
            ("../" + DOC_NAME) if up else DOC_NAME, os.path.dirname(relpath) or "./")
    lines.append(crumb)
    lines.append("")
    lines.append("## 一、目录定位")
    lines.append("")
    lines.append(purpose)
    if PROFILE:
        ov = overview_profile(relpath)
        if ov:
            lines.append("")
            lines.extend(ov)
    lines.append("")

    lines.append("## 二、层级目录")
    lines.append("")
    cap_dirs = 400 if PROFILE else MAX_LIST
    kids = len(dnames) + len(lnames)
    if kids:
        lines.append("本子目录（%d 个，其中符号链接 %d 个）：" % (kids, len(lnames)))
        lines.append("")
        shown = 0
        num_kids = [d for d in dnames if re.fullmatch(r"[0-9]{2,8}", d)]
        compact = PROFILE and len(num_kids) >= 3 and len(num_kids) >= len(dnames) * 0.6
        num_status = {}
        if compact:
            num_status = {d: kid_build_status(absdir, d) for d in num_kids}
            got = {d: s for d, s in num_status.items() if s}
            if got:
                tally = Counter(re.search(r"`([^`]+)`", s).group(1) for s in got.values())
                lines.append("其中 %d 个编号项自带构建标记，取值分布：%s。" % (
                    len(got), "、".join("`%s`×%d" % (k, v) for k, v in tally.most_common())))
                lines.append("")
        for d in dnames[:cap_dirs]:
            child_rel = ("%s/%s" % (relpath, d)) if relpath != "." else d
            cd, cf, cl = entries(os.path.join(absdir, d))
            scale = ""
            if PROFILE and child_rel in ROLL:
                c_, b_, _e, a_ = ROLL[child_rel]
                scale = "（子树 %s 个文件、%s" % ("{:,}".format(c_), human(b_))
                if a_ and b_ and a_ < b_ * 0.85:
                    scale += "，稀疏存储，实占 %s" % human(a_)
                scale += "）"
            if compact and d in num_status:
                st = num_status[d]
                if st:
                    scale = scale[:-1] + "，%s）" % st
                lines.append("- [`%s/`](%s/%s) — 编号项%s" % (d, d, DOC_NAME, scale))
            else:
                lines.append("- [`%s/`](%s/%s) — %s%s" % (
                    d, d, DOC_NAME, folder_purpose(child_rel, cd + cl, cf), scale))
            shown += 1
        for d in lnames[:max(0, cap_dirs - shown)]:
            lines.append("- `%s/` — %s" % (d, link_note(absdir, d)))
            shown += 1
        rest = kids - shown
        if rest > 0:
            lines.append("- *……其余 %d 个子目录同理，各自均有 `目录说明.md`。*" % rest)
    else:
        lines.append("本目录为叶子目录，不含子目录。")
    lines.append("")

    lines.append("## 三、文件清单")
    lines.append("")
    cap_files = 200 if PROFILE else MAX_LIST
    if fnames:
        exts = Counter(os.path.splitext(f)[1].lower() for f in fnames)
        lines.append("共 **%d** 个文件。类型分布：%s" % (
            len(fnames), "、".join("%s×%d" % (kind(e), n) for e, n in exts.most_common())))
        lines.append("")
        readable = [f for f in fnames if os.path.splitext(f)[1].lower() in TEXT_EXTS][:MAX_TEXT_READ]
        listed = fnames if len(fnames) <= cap_files else fnames[:cap_files]
        lines.append("| 文件 | 类型 | 大小 | 内容摘要 |")
        lines.append("|---|---|---|---|")
        summary_map = {}
        for f in readable:
            summary_map[f] = summarize_file(os.path.join(absdir, f), f)
        for f in listed:
            ext = os.path.splitext(f)[1].lower()
            try:
                size = human(os.path.getsize(os.path.join(absdir, f)))
            except OSError:
                size = "?"
            desc = summary_map.get(f) or kind(ext)
            lines.append("| `%s` | %s | %s | %s |" % (f, kind(ext), size, desc))
        if len(fnames) > cap_files:
            lines.append("| *……其余 %d 个文件* | 同类型资产/输出 | — | 数量较多，未逐项列出 |" % (len(fnames) - cap_files))
    else:
        lines.append("本目录当前不含文件（仅子目录）。")
    lines.append("")

    sec = 4
    if PROFILE:
        prof = render_profile(relpath, absdir, dnames, fnames)
        if prof:
            lines.append("## %s、构成画像" % CN_NUM[sec - 1])
            sec += 1
            lines.append("")
            lines.extend(prof)
            lines.append("")

    own_note = None
    try:
        rp = os.path.realpath(absdir)
    except OSError:
        rp = absdir
    if rp in DATA_NOTES:
        own_note = DATA_NOTES[rp]
    if own_note:
        lines.append("## %s、数据集构成详解（人工核定）" % CN_NUM[sec - 1])
        sec += 1
        lines.append("")
        lines.append(own_note.strip())
        lines.append("")

    lines.append("## %s、导航" % CN_NUM[sec - 1])
    lines.append("")
    lines.append("- 根目录总览：[`%s`](%s%s)" % (INDEX_NAME, up, INDEX_NAME))
    lines.append("- 根 README：[%s](%sREADME.md)" % ("README.md", up))
    note_anc = nearest_note_ancestor(absdir)
    if note_anc:
        anc_dir = os.path.relpath(note_anc, absdir).replace(os.sep, "/")
        lines.append("- 上层数据集构成详解：[%s](%s/%s)" % (note_anc.split("/")[-1], anc_dir, DOC_NAME))
    if relpath == ".":
        lines.append("- 结构与工作线说明：[docs/PROJECT_STRUCTURE_20261003.md](docs/PROJECT_STRUCTURE_20261003.md)")
    lines.append("")
    return "\n".join(lines) + "\n"


def nearest_note_ancestor(absdir):
    """返回最近的、带人工核定数据集说明的上层目录绝对路径（不含自身）。"""
    try:
        cur = os.path.realpath(absdir)
    except OSError:
        return None
    parent = os.path.dirname(cur)
    while parent and parent != cur:
        if parent in DATA_NOTES:
            return parent
        cur = parent
        parent = os.path.dirname(cur)
    return None


def build_index(root, tree_lines, n_docs):
    top_dirs, _top_files, top_links = entries(root)
    out = [
        "# %s 层级目录总览" % os.path.basename(root.rstrip("/")),
        "",
        "> 由 `scripts/gen_folder_docs.py` 依据目录实际内容生成。仓库中每一个非隐藏目录都有一份",
        "> `目录说明.md`，内含该目录的定位说明、层级子目录清单，以及逐个文件读取内容后得到的摘要。",
        "",
        "- 仓库路径：`%s`" % root,
        "- 目录说明数量：**%d** 个" % n_docs,
        "- 隐藏目录、`__pycache__` 不生成说明；指向仓库外存储的符号链接不在其目标内生成说明，"
        "但会在上层目录中列出并标注链接去向。",
        "",
        "## 全局层级目录（前 3 级）",
        "",
        "```",
        os.path.basename(root) + "/",
    ] + tree_lines + [
        "```",
        "",
        "## 顶级目录职责速览",
        "",
        "| 目录 | 说明 |",
        "|---|---|",
    ] + [
        "| [`%s/`](%s/%s) | %s |" % (d, d, DOC_NAME, folder_purpose(d, *entries(os.path.join(root, d))[:2]))
        for d in top_dirs
    ] + [
        "| `%s/` | %s |" % (d, link_note(root, d)) for d in top_links
    ] + [
        "",
        "## 阅读顺序建议",
        "",
        "1. 先看根 `README.md` 与 `docs/PROJECT_STRUCTURE_20261003.md` 了解工作线；",
        "2. 算法实现从 `src/` 进入，运行入口从 `scripts/` 进入；",
        "3. 每个实验的输入、条件缓存、检查点与评测产物见对应 `experiments/*/目录说明.md`；",
        "4. 数据集与权重的组织方式见 `datasets/`、`models/`、`add/` 下逐级说明。",
        "",
    ]
    return "\n".join(out) + "\n"


def walk_dirs(root, skip_top=()):
    """自研递归遍历：只进入真实目录与 allowed_link 允许的符号链接，跳过隐藏目录，防环。

    skip_top 在进入前就剪掉，避免被剪掉的子树污染防环集合（同一实体可能有多个链接入口）。
    """
    out = []
    seen = {os.path.realpath(root)}

    def rec(cur, rel):
        dnames, fnames, lnames = entries(cur)
        kids = []
        for name in dnames:
            child_abs = os.path.join(cur, name)
            child_rel = ("%s/%s" % (rel, name)) if rel != "." else name
            if rel == "." and name in skip_top:
                continue
            rp = os.path.realpath(child_abs)
            if rp in seen:
                continue
            if under_any(rp, EXCLUDE_UNDER):
                continue
            seen.add(rp)
            kids.append((name, child_abs, child_rel))
        out.append((rel, cur, sorted(n for n, _, _ in kids), fnames))
        for name, child_abs, child_rel in kids:
            rec(child_abs, child_rel)

    rec(root, ".")
    return out


def make_tree(root, dirs, max_depth=3, max_children=25):
    children = {}
    nfiles = {}
    for rel, absdir, dnames, fnames in dirs:
        children[rel] = dnames
        nfiles[rel] = len([f for f in fnames if f not in (DOC_NAME, INDEX_NAME)])
    lines = []

    def emit(rel, prefix, depth):
        kids = children.get(rel, [])
        for i, d in enumerate(kids[:max_children]):
            child = ("%s/%s" % (rel, d)) if rel != "." else d
            last = (i == len(kids) - 1)
            if os.path.islink(os.path.join(root, child)):
                lines.append("%s%s/  (仓库外符号链接)" % (prefix, d))
                continue
            cdirs = children.get(child, [])
            lines.append("%s%s%s/  (%d 子目录, %d 文件)" % (
                prefix, "└── " if last else "├── ", d, len(cdirs), nfiles.get(child, 0)))
            if depth < max_depth and cdirs:
                emit(child, prefix + ("    " if last else "│   "), depth + 1)
        extra = len(kids) - max_children
        if extra > 0:
            lines.append("%s… 其余 %d 个子目录" % (prefix, extra))

    emit(".", "", 1)
    return lines


def main():
    global ROOT, FOLLOW_UNDER, EXCLUDE_UNDER, PROFILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--only", default=None, help="只为该相对子树生成（调试用）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset-profile", action="store_true",
                    help="附加数据集构成画像与人工核定的数据集详解章节")
    ap.add_argument("--print-sample", type=int, default=0)
    ap.add_argument("--follow-under", default="",
                    help="逗号分隔的绝对路径前缀：仅进入目标位于这些前缀之下的符号链接目录")
    ap.add_argument("--exclude-under", default="",
                    help="逗号分隔的绝对路径前缀：跳过目标位于这些前缀之下的目录")
    ap.add_argument("--no-index", action="store_true", help="不生成根级 目录总览.md")
    ap.add_argument("--skip-top", default="model-qwen-edit", help="逗号分隔的顶级目录名，跳过整棵子树")
    args = ap.parse_args()

    def norm_prefixes(vals):
        out = []
        for v in vals:
            v = os.path.realpath(v.rstrip("/")) + "/"
            out.append(v)
        return tuple(out)

    FOLLOW_UNDER = norm_prefixes([p.strip() for p in args.follow_under.split(",") if p.strip()])
    EXCLUDE_UNDER = norm_prefixes([p.strip() for p in args.exclude_under.split(",") if p.strip()])

    global ROOT
    root = os.path.abspath(args.root)
    ROOT = root
    skip_top = {s for s in args.skip_top.split(",") if s}

    dirs = walk_dirs(root, skip_top)
    PROFILE = bool(args.dataset_profile)
    if PROFILE:
        build_profile_index(dirs)
    if args.only:
        dirs = [t for t in dirs if t[0] == args.only or t[0].startswith(args.only + "/")]

    written = preserved = skipped_manual = failed = 0
    for rel, absdir, dnames, fnames in dirs:
        target = os.path.join(absdir, DOC_NAME)
        if os.path.islink(target):
            skipped_manual += 1
            continue
        if os.path.exists(target):
            try:
                with open(target, encoding="utf-8", errors="ignore") as fh:
                    head = fh.read(400)
            except OSError:
                head = ""
            if "gen_folder_docs.py" not in head:
                skipped_manual += 1
                continue
            preserved += 1
        try:
            content = build_doc(rel, dnames, fnames, absdir)
        except Exception as exc:
            print("FAIL %s: %r" % (rel, exc), file=sys.stderr)
            failed += 1
            continue
        if args.dry_run:
            if written < args.print_sample:
                print("=" * 70 + "\n[%s]\n" % rel + content)
            written += 1
            continue
        try:
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            written += 1
        except OSError as exc:
            print("WRITE-FAIL %s: %r" % (rel, exc), file=sys.stderr)
            failed += 1

    if not args.dry_run and not args.only and not args.no_index:
        try:
            tree = make_tree(root, dirs)
            with open(os.path.join(root, INDEX_NAME), "w", encoding="utf-8") as f:
                f.write(build_index(root, tree, written))
        except Exception as exc:
            print("INDEX-FAIL: %r" % (exc), file=sys.stderr)

    print("dirs=%d generated=%d regenerated=%d skipped_manual_or_link=%d failed=%d dry_run=%s"
          % (len(dirs), written, preserved, skipped_manual, failed, args.dry_run))


if __name__ == "__main__":
    main()
