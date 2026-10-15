# -*- coding: utf-8 -*-
"""
Generate detailed README.md for every folder in the pictureedit repository.
Preserves existing custom README.md files while providing rich, domain-aware
descriptions for code modules, scripts, experiments, research, and data folders.
"""

import os
import sys

IGNORED_NAMES = {'.git', '.pytest_cache', '__pycache__', '.mimosa', '.workbuddy'}

FILE_DESCRIPTIONS = {
    # configs
    'asset_registry.json': '全局核心资产注册表，集中管理基座模型路径、数据缓存、冷归档位置与 Conda 运行环境映射',
    # src/adapter
    'pose_adapter.py': '早期 CNN 姿态适配器 `PoseConditionAdapter` 实现，利用零卷积初始化向 DiT 注入 8 通道控制特征',
    # src/controls
    'generator.py': '`SMPLXConditionGenerator` 控制信号生成器，从 SMPL-X 网格与姿态渲染 25 通道姿态热图、法线图、深度图和身体接触图',
    # src/data
    'dataset.py': '通用配对姿态编辑数据集抽象基类与基础加载管道',
    'iper_dataset.py': 'iPER 数据集专用读取管道，支持源图/目标图配对与几何条件加载',
    'tiktok_dataset.py': 'TikTok 数据集专用读取管道，支持单人舞蹈视频帧对与关键点坐标对齐',
    'hi4d_dataset.py': 'Hi4D 双人交互数据集专用读取管道，支持双人骨架与空间接触关系配对',
    'cache.py': '条件张量离线缓存与内存加速读取模块',
    # src/evaluation
    'metrics.py': '图像质量定量评价指标实现（PSNR、SSIM、LPIPS、FID 等）',
    'dwpose_detector.py': '基于 ONNX 的 DWPose 全身人体 133 关键点检测器',
    'wholebody.py': '全人体关键点解析与姿态骨架可视化辅助模块',
    # src/integration
    'pipeline.py': 'Diffusers / DiT 受控图像生成端到端 Pipeline 封装',
    'hooks.py': 'PyTorch 前向传播 Hook 管理工具，负责动态截获并注入 DiT token 残差',
    # src/pose_control/qwen21
    'body_adapter.py': 'Qwen 2.1 简化版人体姿态控制适配器（`Qwen21BodyAdapter`）',
    'face_adapter.py': 'Qwen 2.1 简化版人脸局部控制适配器（`Qwen21FaceAdapter`）',
    'dual_adapter.py': 'Qwen 2.1 简化版双人交互姿态控制适配器（`Qwen21DualAdapter`）',
    'qwen21_adapter.py': 'Qwen 2.1 基础适配器挂载与 Hook 上下文管理',
    # src/pose_control/qwen21_full
    'checkpoint.py': '检查点保存与加载逻辑，要求 `architecture == qwen21_full_v1` 并严格校验权重形状',
    'condition_bridge.py': '条件特征转换与桥接器，将 1536 维几何特征投影并重排为 Qwen DiT token',
    'condition_injector.py': '`SMPLXConditionInjector` 多模态几何条件注入器，统一编码法线、姿态与全局参数',
    'conditions.py': '条件数据结构定义（包含目标法线、姿态热图、部位掩码与任务 Token）',
    'control_core.py': '`SharedRecurrentControlCore` 共享循环控制核心，负责 6 阶段控制残差循环计算',
    'controlled_pipeline.py': '集成 Qwen 2.1 控制器的端到端受控图像采样生成 Pipeline',
    'data_runtime.py': '运行时数据构建、Batch 打包及 Condition Bundle 动态生成',
    'deepgen_adapter.py': '保留的 DeepGen 原始适配器基类与兼容层',
    'execution.py': '执行上下文管理与前向执行流调度逻辑',
    'experts.py': '多专家网络（MoE）路由与身体/局部细节控制专家实现',
    'face_coordinates.py': '面部关键点检测与局部 16x16 ROI 坐标映射计算',
    'face_loss_alignment.py': '面部特征对齐、表情重建与身份一致性损失函数',
    'face_supervision.py': '人脸区域局部监督与梯度反向传播控制策略',
    'frozen_bfs_lora.py': '冻结 BFS 骨干网络并集成 LoRA 微调权重逻辑',
    'geometry_encoder.py': 'SMPL-X 空间几何结构编码器',
    'joint_training.py': 'Body-Face-Hand 联合多任务训练管理与损失加权',
    'qwen_adapter.py': '`Qwen21FullControlContext`，通过 PyTorch forward hook 向 Qwen 32 个 DiT Block 注入 4096 维控制残差',
    'qwen_runtime.py': 'Qwen 2.1 运行时环境管理、Prompt Embedding 缓存与 VAE 解码',
    'token_encoders.py': '针对目标图像 token 的零初始化投影与残差映射头',
    # scripts
    'verify_project_organization.py': '项目组织规范、资产登记表与文件软链接合规性自动化核验脚本',
    'prepare_iper_dataset.py': 'iPER 数据集切片划分、元数据生成与清理脚本',
    'build_condition_cache.py': '批量预计算并缓存 SMPL-X 几何条件特征张量脚本',
    'multi_stage_eval_v6.py': 'DeepGen V6 多阶段模型基准评测主脚本',
    'eval_runner.py': '多卡/多阶段批量评测自动化调度程序',
    'polish_figures.py': '图表与可视化成果样式美化与高清渲染脚本',
    'make_tiktok_overfit_comparisons.py': 'TikTok 过拟合实验四联对比面板自动拼接生成脚本',
    'train_iper_qwen21_full.py': 'Qwen 2.1 Full 模型在 iPER 数据集上的主训练入口',
    'eval_iper_qwen21_full.py': 'Qwen 2.1 Full 模型在 iPER 数据集上的评测与采样入口',
    # tests
    'test_adapter_v6_architecture.py': '验证 DeepGen V6 架构完整前向传播与各分支残差张量形状',
    'test_condition_bridge_contract.py': '验证条件桥接协议（Condition Bridge）输入输出契约与张量对齐',
    'test_deepgen_control_interface_v1.py': '验证适配器零卷积初始化行为，确保未训练时生成基座输出严格不变',
    'test_deepgen_no_control_ablation.py': '验证在无控制信号输入时的消融基线表现与退化安全性',
    'test_iper_adapter_pipeline.py': '端到端集成测试：iPER 数据输入到受控生成 Pipeline 采样',
    # visualization
    'eval_metrics.json': '标准化评测集上的多模型定量评价指标对比数据 (PSNR, SSIM, LPIPS)',
    'README_test_evaluation.md': '评测基准构建规范、样本选取原则与指标计算说明文档',
    'pipeline.log': '流水线运行跟踪与执行日志',
}

def is_ignored(path):
    parts = set(os.path.normpath(path).replace('\\', '/').split('/'))
    return any(p in IGNORED_NAMES or p.startswith('.') for p in parts if p != '.')

def generate_readme_content(norm_path, subdirs, files):
    parts = norm_path.split('/')
    folder_name = parts[-1] if norm_path != '.' else '项目根目录'
    top = parts[0] if norm_path != '.' else '.'

    title = f"# {folder_name} 目录说明"
    path_info = f"**路径**: `{norm_path}`\n"

    # Analyze file types and count
    file_types = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower() or '(无后缀)'
        file_types[ext] = file_types.get(ext, 0) + 1
    type_summary = ", ".join([f"{ext}: {count}个" for ext, count in sorted(file_types.items())])

    # Contextual description based on path
    purpose = ""
    pipeline_role = ""

    # 1. ROOT
    if norm_path == '.':
        purpose = "本项目核心根目录，专注于基于三维人体表示（SMPL-X）的单人与双人参考姿态图像编辑。"
        pipeline_role = "包含核心配置、主运行脚本、环境依赖与各子模块入口。"

    # 2. SRC
    elif norm_path == 'src':
        purpose = "核心源代码根目录，包含姿态控制适配器网络、SMPL-X几何条件生成器、数据集加载管道与推理集成模块。"
        pipeline_role = "承载 DeepGen V6 系列与 Qwen-Image-2.1 系列外挂控制网络的核心算法实现。"
    elif norm_path == 'src/adapter':
        purpose = "早期 CNN 姿态适配器模块目录，包含 `PoseConditionAdapter`。"
        pipeline_role = "实现零卷积初始化的 8 通道姿态条件注入网络，作为早期显存基线与轻量控制原型。"
    elif norm_path == 'src/controls':
        purpose = "SMPL-X 控制信号生成器目录，包含 `generator.py` 等核心生成逻辑。"
        pipeline_role = "负责从 3D 人体网格与关节点提取 25 通道姿态热图、法线图、深度图、部位分割图与身体接触图。"
    elif norm_path == 'src/data':
        purpose = "PyTorch 数据集与数据加载器目录，支持 iPER、TikTok、Hi4D 等数据集。"
        pipeline_role = "负责配对样本读取、几何条件动态打包（Condition Bundle）及多分辨率图像张量预处理。"
    elif norm_path == 'src/evaluation':
        purpose = "自动化评测与指标度量工具目录。"
        pipeline_role = "提供生成质量计算接口，包括 PSNR、SSIM、LPIPS 及姿态估计一致性评估。"
    elif norm_path == 'src/evaluation/dwpose_onnx':
        purpose = "DWPose ONNX 推理实现目录。"
        pipeline_role = "提供轻量化、无需复杂 GPU 依赖的 2D 全身关节点检测与姿态估计推理能力。"
    elif norm_path == 'src/integration':
        purpose = "扩散模型与 DiT 生成管线集成目录。"
        pipeline_role = "负责将外挂姿态控制器与 Diffusers / DiT Pipeline 挂接，实现端到端采样与去噪控制。"
    elif norm_path == 'src/pose_control':
        purpose = "姿态控制网络架构根目录，包含 Qwen 2.1 简化版、Qwen 2.1 Full 完整版及 DeepGen V6 基准版。"
        pipeline_role = "汇聚所有针对不同基座模型与不同演进阶段的姿态注入网络定义。"
    elif norm_path == 'src/pose_control/qwen21':
        purpose = "Qwen 2.1 简化姿态控制器目录，包含 Body、Face 与 Dual 独立控制器。"
        pipeline_role = "针对 Qwen-Image-2.1 架构精简设计的几何注入头，支撑单人身体、面部与双人 Hi4D 实验。"
    elif norm_path == 'src/pose_control/qwen21_full':
        purpose = "Qwen 2.1 完整迁移控制器目录，将 DeepGen V6 47 个完整组件迁移至 Qwen 2.1 DiT。"
        pipeline_role = "具备 32 层 Hook 残差注入、1536 到 4096 维输出投影桥、共享循环控制核心与全功能几何推理机制。"
    elif norm_path == 'src/pose_control/qwen21_full/detail':
        purpose = "Qwen 2.1 Full 局部细节控制分支模块目录。"
        pipeline_role = "负责高频细节特征提取与局部残差补偿，提升服装褶皱与纹理一致性。"
    elif norm_path == 'src/pose_control/qwen21_full/face':
        purpose = "Qwen 2.1 Full 独立人脸 ROI 控制器目录。"
        pipeline_role = "融合 ArcFace 512 维向量与 DINO patch 特征，在目标人脸区域进行独立 ROI 局部引导与残差散射。"
    elif norm_path == 'src/pose_control/qwen21_full/hand':
        purpose = "Qwen 2.1 Full 独立手部 ROI 控制器目录。"
        pipeline_role = "提取四只手的局部空间几何与可见性特征，通过注意力融合实现高精细度手指交互姿态控制。"
    elif norm_path == 'src/pose_control/qwen21_full/interface':
        purpose = "Qwen 2.1 Full 抽象接口与协议定义目录。"
        pipeline_role = "定义控制器上下文、输入输出规范及与生成 Pipeline 的挂载契约。"
    elif norm_path == 'src/pose_control/qwen21_full/reasoning':
        purpose = "Qwen 2.1 Full 空间几何推理核心模块目录。"
        pipeline_role = "包含单人几何推理、双人空间关系注意力机制、接触推理网络及外观特征空间桥接。"
    elif norm_path == 'src/pose_control/v6':
        purpose = "DeepGen V6 Unified SMPL-X 适配器基准实现目录。"
        pipeline_role = "完整控制结构的原始实现，包含条件编码、多阶段循环控制核心与残差投影。"
    elif norm_path == 'src/pose_control/v6/detail':
        purpose = "DeepGen V6 细节控制分支模块目录。"
        pipeline_role = "提供局部细节特征处理与辅助残差计算。"
    elif norm_path == 'src/pose_control/v6/face':
        purpose = "DeepGen V6.5 人脸增强模块目录。"
        pipeline_role = "负责面部关键点编码、表情注入及人脸身份保持特征融合。"
    elif norm_path == 'src/pose_control/v6/interface':
        purpose = "DeepGen V6 控制器基础接口目录。"
        pipeline_role = "规范适配器状态字典、前向传参与输出特征结构。"
    elif norm_path == 'src/pose_control/v6/reasoning':
        purpose = "DeepGen V6 几何与交互推理网络目录。"
        pipeline_role = "执行 SMPL-X 几何特征的空间自注意力与跨人物交叉注意力推理。"

    # 3. SCRIPTS
    elif norm_path == 'scripts/audit':
        purpose = "工程规范、权重结构与环境依赖审计脚本目录。"
        pipeline_role = "提供项目组织结构校验、检查点元数据检查与数据完整性验收工具。"
    elif norm_path == 'scripts/data':
        purpose = "数据处理、预计算与缓存构建脚本目录。"
        pipeline_role = "包含 iPER/TikTok 划分清洗、SMPL-X 条件离线预计算与缓存生成脚本。"
    elif norm_path == 'scripts/eval':
        purpose = "模型评测与推理验证脚本目录。"
        pipeline_role = "包含单/双阶段模型推理、多模型消融对照与定量指标自动评估脚本。"
    elif norm_path == 'scripts/reports':
        purpose = "实验报告汇总、图表美化与面板合成脚本目录。"
        pipeline_role = "负责生成四联图对比面板、指标柱状图以及汇报用 Word/HTML 文档配图。"
    elif norm_path == 'scripts/run':
        purpose = "全流程实验运行编排与执行管理目录。"
        pipeline_role = "提供一键运行、阶段收尾及自动化实验流程管理。"
    elif norm_path == 'scripts/train':
        purpose = "模型训练启动脚本目录。"
        pipeline_role = "提供适配器训练入口包装、多卡加速器配置与微调阶段切换。"

    # 4. CONFIGS
    elif norm_path == 'configs':
        purpose = "项目配置与资产注册目录。"
        pipeline_role = "包含 `asset_registry.json`，集中管理基座模型路径、数据缓存、冷归档位置与 Conda 运行环境映射。"

    # 5. TESTS
    elif norm_path == 'tests':
        purpose = "自动化测试套件目录。"
        pipeline_role = "包含针对适配器架构、零卷积初始化、输入张量契约、梯度传递与推理输出的单元测试与回归测试。"

    # 6. DOCS
    elif norm_path == 'docs':
        purpose = "项目核心设计文档、架构说明与清理审计记录目录。"
        pipeline_role = "记录项目工作线、模型演进路径、环境配置与多次整理的详细执行记录。"
    elif norm_path == 'docs/experiments':
        purpose = "实验专项说明与交接操作指南目录。"
        pipeline_role = "记录 BFS 联合训练 runbook、交接备忘及几何-Prompt 协同调研报告。"
    elif norm_path == 'docs/qwen21_architecture':
        purpose = "Qwen 2.1 架构设计图解与绘图脚本目录。"
        pipeline_role = "包含适配器内部数据流、网络拓扑图生成脚本及高分辨率架构图。"
    elif norm_path == 'docs/superpowers':
        purpose = "工程方法论与结构化设计规范根目录。"
        pipeline_role = "管理开发计划（plans）与接口设计规格（specs）。"
    elif norm_path == 'docs/superpowers/plans':
        purpose = "架构重构与特性落地计划目录。"
        pipeline_role = "保存各阶段任务的拆解计划、执行步骤与验收标准（如 TikTok 条件修复、BFS 扩大对照等）。"
    elif norm_path == 'docs/superpowers/specs':
        purpose = "系统设计规格说明书目录。"
        pipeline_role = "定义模块接口契约、张量形状与前后端挂载标准。"

    # 7. ARCHIVE
    elif norm_path == 'archive':
        purpose = "历史产物与早期清理归档根目录。"
        pipeline_role = "存放不再活跃执行的历史 runner 脚本与过往清理核查报告。"
    elif norm_path == 'archive/cleanup_history_202609':
        purpose = "2026年9月历史目录清理清单与核查记录。"
        pipeline_role = "留存早期清理决策依据、文件迁移对照表与历史状态快照。"
    elif norm_path == 'archive/legacy_runners_202609':
        purpose = "2026年9月远端早期训练与评测启动脚本归档。"
        pipeline_role = "归档已完成阶段性使命的历史 Shell 脚本（如 `eval_runner.sh`, `run_overfit_sanity.sh`）。"

    # 8. THIRD PARTY
    elif norm_path == 'third_party':
        purpose = "第三方依赖库与外部参考代码根目录。"
        pipeline_role = "提供外部人体姿态引导及面部特征处理的基础代码支持。"
    elif norm_path == 'third_party/champ_guidance':
        purpose = "Champ 姿态引导特征编码器封装目录。"
        pipeline_role = "提取人体骨骼与姿态结构引导信号，提供与 DiT 集成的特征提取工具。"
    elif norm_path == 'third_party/v65_face':
        purpose = "V6.5 人脸模块外部依赖与接口包装目录。"
        pipeline_role = "集成 MCLD、StableAnimator、Visual Persona 及 X-Dyna 等开源人脸保持技术。"
    elif norm_path == 'third_party/v65_face/licenses':
        purpose = "第三方人脸模块开源许可证目录。"
        pipeline_role = "保存 StableAnimator (MIT) 及 X-Dyna (Apache 2.0) 许可证声明。"

    # 9. OUTPUT
    elif norm_path == 'output':
        purpose = "任务流水线执行产物与自动报告生成根目录。"
        pipeline_role = "存放多阶段自动生成流水线的中间产物与最终交付文档。"
    elif norm_path == 'output/pei2i_20261008':
        purpose = "2026-10-08 PE-I2I (Pose-Enabled Image-to-Image) 自动生成报告流水线执行目录。"
        pipeline_role = "记录多阶段文本润色、实验数据提取、配图生成与 Word 文档编译的全流程。"
    elif norm_path == 'output/pei2i_20261008/draft':
        purpose = "PE-I2I 报告初稿存储目录。"
        pipeline_role = "保存由大模型根据实验日志与指标初步撰写的 Markdown 草稿。"
    elif norm_path == 'output/pei2i_20261008/params':
        purpose = "流水线执行参数与主题配置文件目录。"
        pipeline_role = "包含报告评价配置 `critic_config.yaml` 与选题定义 `topic.yaml`。"
    elif norm_path == 'output/pei2i_20261008/research':
        purpose = "PE-I2I 背景资料与调研结果整理目录。"
        pipeline_role = "存放自动搜集的文献、方法对比与技术背景材料。"
    elif norm_path == 'output/pei2i_20261008/stage1':
        purpose = "第一阶段文本产物目录。"
        pipeline_role = "保存经初轮审阅与修改后的终稿 Markdown 文件。"
    elif norm_path == 'output/pei2i_20261008/stage2':
        purpose = "第二阶段排版与 HTML 渲染结果目录。"
        pipeline_role = "保存包含样式设计与图表排版的 HTML 报告及设计 Token。"
    elif norm_path == 'output/pei2i_20261008/stage2/assets':
        purpose = "报告排版所需的架构与流程图表资源目录。"
        pipeline_role = "存放流程图、重写过程图及结果汇总矢量/位图资产。"
    elif norm_path == 'output/pei2i_20261008/stage2/images':
        purpose = "报告正文中引用的实验效果对比配图目录。"
        pipeline_role = "存放图 1 至图 5 的最终高清对比图像。"
    elif norm_path == 'output/pei2i_20261008/stage3':
        purpose = "第三阶段最终成果输出目录。"
        pipeline_role = "包含最终生成的 Word 格式排版报告 (`PE-I2I自动编写实验报告.docx`)。"
    elif norm_path == 'output/pei2i_20261008/trace':
        purpose = "流水线执行追踪与调试日志目录。"
        pipeline_role = "保存完整的流水线执行日志 (`pipeline.log`)。"

    # 10. REPORTS
    elif norm_path == 'reports/architecture':
        purpose = "系统架构设计说明、展示页面与工程证据汇总目录。"
        pipeline_role = "包含架构图、交互数据流设计以及架构合规性证明。"
    elif norm_path == 'reports/architecture/diagrams_v64':
        purpose = "V6.4 架构设计图表资产目录。"
        pipeline_role = "存放全套 V6.4 适配器多阶段控制流程的高清架构示意图。"
    elif norm_path == 'reports/architecture/evidence':
        purpose = "架构有效性与维度匹配核验证据目录。"
        pipeline_role = "留存残差维度匹配、零初始化权重状态及 Hook 挂载正确性的核验证据。"
    elif norm_path == 'reports/architecture/mermaid_flows':
        purpose = "架构数据流与控制时序 Mermaid 源码目录。"
        pipeline_role = "提供可动态渲染的 Mermaid 格式流程图文件。"
    elif norm_path == 'reports/evaluation':
        purpose = "定量评测报告与性能指标分析目录。"
        pipeline_role = "汇总 TikTok 与 iPER 数据集上的定量评估数据及局部 ROI 效果分析。"

    # 11. VISUALIZATION
    elif norm_path == 'visualization':
        purpose = "项目可视化成果与评测对比画廊总目录。"
        pipeline_role = "汇集条件控制信号图、训练阶段演化图、四联对比面板及 3D 可视化页面。"
    elif '00_' in norm_path or '总览' in norm_path:
        purpose = "可视化总览说明与核心指标汇总目录。"
        pipeline_role = "提供生成质量指标 JSON、挑选展示对列表及全流程说明文档。"
    elif '01_' in norm_path or '条件控制信号' in norm_path:
        purpose = "SMPL-X 几何条件控制信号展示目录。"
        pipeline_role = "直观呈现 25 通道姿态热图、法线图、深度图与身体部位分割图。"
    elif '02_' in norm_path or '训练阶段演化' in norm_path:
        purpose = "训练不同步数下生成质量演化对比大图目录。"
        pipeline_role = "展示从 Step 0 到 Step 3000 模型在姿态对齐与衣着保持上的渐进提升过程。"
    elif norm_path == 'visualization/03_最终评测四联图':
        purpose = "最终评测四联图对比面板集合目录。"
        pipeline_role = "将原图、目标姿态条件、模型生成图与真实目标图四图并排展示，便于人工目测验收。"
    elif '模板版_带条件控制' in norm_path:
        purpose = "统一排版模板的带几何条件四联图展示目录。"
        pipeline_role = "采用统一高分辨率模板展示源图、条件图、生成图及目标真值。"
    elif '04_' in norm_path or '编辑效果条带' in norm_path:
        purpose = "多姿态连续编辑条带效果图目录。"
        pipeline_role = "展示单张人物源图迁移至多个连续目标动作下的横向条带对比图。"
    elif norm_path == 'visualization/anigra':
        purpose = "交互式 3D 人体网格与动画可视化工具目录。"
        pipeline_role = "支持 3D 人体几何结构在浏览器端的交互式旋转与审查。"
    elif norm_path == 'visualization/anigra/three':
        purpose = "Three.js 交互式渲染页面与姿态样例目录。"
        pipeline_role = "提供 WebGL 3D 渲染环境及对应生成的姿态编辑效果图。"
    elif norm_path == 'visualization/testdata':
        purpose = "标准化评测测试集基准切片目录。"
        pipeline_role = "提供评测脚本标准输入图像与对应的 SMPL-X 控制条件对。"
    elif norm_path == 'visualization/testdata/conditions':
        purpose = "评测集 SMPL-X 条件控制图目录（200 个样本）。"
        pipeline_role = "包含法线图、DWPose 骨架图、深度图与部位分割图。"
    elif norm_path == 'visualization/testdata/images':
        purpose = "评测集原始输入图像与目标真实图像目录（200 个样本）。"
        pipeline_role = "提供评估生成模型图像保真度的标准基准图像对。"

    # 12. EXPERIMENTS
    elif top == 'experiments':
        # Sub-branches of experiments
        if 'bfs_body_face_joint_pilot_20261008' in norm_path:
            exp_sub = norm_path.split('bfs_body_face_joint_pilot_20261008')[-1].strip('/')
            if not exp_sub:
                purpose = "BFS Body+Face 联合训练 Pilot 核心实验主目录（2026-10-08）。"
                pipeline_role = "验证身体几何控制与独立人脸 ROI 联合微调在 iPER 和 TikTok 上的收敛性与保真度。"
            elif exp_sub.startswith('data'):
                purpose = "Pilot 实验数据索引、预处理日志与配置目录。"
                pipeline_role = "记录训练集/验证集 manifest 划分及特征抽取处理日志。"
            elif exp_sub.startswith('diagnostic_failures'):
                purpose = "实验过程中的诊断性失败样本留存目录。"
                pipeline_role = "记录条件缓存缺失或初始化异常时的故障图像与面板，用于排查根因。"
            elif exp_sub.startswith('handoff'):
                purpose = "Pilot 实验阶段成果交接备忘目录。"
                pipeline_role = "提供交接记录、关键指标对比总结与后续全量训练建议。"
            elif exp_sub.startswith('preflight'):
                purpose = "模型起训前 Preflight 快速前向、梯度与断点续训确定性验证目录。"
                pipeline_role = "确保在提交大规模训练前消除随机性偏差，验证 checkpoint 恢复前后 Loss 严格一致。"
            elif exp_sub.startswith('quality_reviews'):
                purpose = "生成图像人工与自动化质量审查评分记录目录。"
                pipeline_role = "包含逐样本缺陷诊断、人脸畸变分析与服饰保真度评估明细。"
            elif exp_sub.startswith('report'):
                if 'face_sheets' in exp_sub:
                    purpose = "面部细节特写对比汇总表格图目录。"
                    pipeline_role = "横向对比不同模型阶段（基线、Body、Joint）下人脸身份与表情的保持质量。"
                elif 'panels' in exp_sub:
                    purpose = "全景四联评测面板汇总目录。"
                    pipeline_role = "集中汇集各阶段评估生成的高清四联对比图。"
                else:
                    purpose = "Pilot 阶段评测成果与汇报材料汇总目录。"
                    pipeline_role = "提供总体质量报告与结构化展示图表。"
            elif exp_sub.startswith('training'):
                purpose = "联合训练执行过程日志与检查点配置目录。"
                pipeline_role = "分别记录 Joint 联合模型 (J) 与基线微调模型 (Qwen) 的训练损失、显存占用与状态。"
            elif exp_sub.startswith('evaluation'):
                eval_parts = exp_sub.split('/')
                run_name = eval_parts[1] if len(eval_parts) > 1 else 'evaluation'
                leaf = eval_parts[-1] if len(eval_parts) > 2 else ''
                
                if leaf == 'faces':
                    purpose = f"评测 Run `{run_name}` 的人脸局部特写切片目录。"
                    pipeline_role = "高分辨率裁剪人脸区域，用于精准审查面部微表情、五官结构与身份保真度。"
                elif leaf == 'images':
                    purpose = f"评测 Run `{run_name}` 的全身完整生成图像目录。"
                    pipeline_role = "保存模型完整推理输出图像，用于评估整体构图、背景融合与服装结构。"
                elif leaf == 'panels':
                    purpose = f"评测 Run `{run_name}` 的四联对比验收面板目录。"
                    pipeline_role = "并排拼接展示源图、目标姿态条件图、模型生成图与真实目标图。"
                else:
                    purpose = f"评测 Run `{run_name}` 的输出结果目录。"
                    pipeline_role = f"包含该轮评测生成的人脸特写、全图结果与对比面板。"
            else:
                purpose = f"BFS Body+Face 联合训练 Pilot 实验子目录 `{exp_sub}`。"
                pipeline_role = "承载该实验模块对应的数据与结果。"
        elif 'bfs_compare_30_20261004' in norm_path:
            if 'appearance_sheets' in norm_path:
                purpose = "30 组扩大样本对照实验的外观保真度分析表目录。"
                pipeline_role = "分析服饰纹理、身型比例与颜色在姿态迁移过程中的一致性保持情况。"
            elif 'panels' in norm_path:
                purpose = "30 组扩大样本对照实验的四联对比面板目录。"
                pipeline_role = "提供 30 组代表性样本在基线与新适配器下的横向直观对比。"
            else:
                purpose = "BFS 扩大至 30 组样本的横向对照评测实验主目录。"
                pipeline_role = "评估模型在更大测试集上的泛化能力与肢体结构合理性。"
        elif 'hi4d_qwen21_dual_20260930' in norm_path:
            purpose = f"Hi4D Qwen 2.1 双人交互姿态编辑实验子目录 `{folder_name}`。"
            pipeline_role = "包含针对 Hi4D 双人交互数据集（搭肩、握手、拥抱）的专用数据管道、控制器扩展与测试用例。"
        elif 'qwen-2.1-test' in norm_path:
            purpose = f"Qwen 2.1 原生能力评测与微调实验子目录 `{folder_name}`。"
            pipeline_role = "测试原生 Qwen 模型的提示词遵循、生成质量、单/双人交互能力与显存特征。"
        elif 'tiktok_00001_condition_repair' in norm_path:
            purpose = f"TikTok 00001 单样本几何条件修复实验子目录 `{folder_name}`。"
            pipeline_role = "针对 TikTok 早期单样本条件错位问题进行修复验证与多条件消融推理。"
        elif 'tiktok_legacy_evidence_202609' in norm_path:
            purpose = f"TikTok 2026年9月历史实验留存与证据子目录 `{folder_name}`。"
            pipeline_role = "保存早期单段视频（one-piece test）过拟合实验的条件缓存与预览图。"
        elif 'tiktok_overfit_sanity_full' in norm_path:
            purpose = f"TikTok 全量过拟合健壮性实验（两阶段训练）子目录 `{folder_name}`。"
            pipeline_role = "包含训练检查点（约 7.4G）、预计算条件张量缓存、生成采样与阶段预览图。"
        else:
            purpose = f"实验目录 `{norm_path}`。"
            pipeline_role = "记录特定实验配置下的模型输入、中间缓存或评测成果。"

    # 13. RESEARCH
    elif top == 'research':
        if 'abc_showcase_20261003' in norm_path:
            purpose = f"多源开源人体数据集调研与样本展示子目录 `{folder_name}`。"
            pipeline_role = "针对 DNA-Rendering、Freeman、HumanVid、MVHumanNet、PKU-DyMVHumans 等数据集进行格式与质量调研。"
        elif 'adapter_audit_20261002' in norm_path:
            purpose = f"2026-10-02 适配器架构审计与远端快照子目录 `{folder_name}`。"
            pipeline_role = "保存当时远端服务器的完整代码快照、命名空间迁移验收测试与原生运行时诊断记录。"
        elif 'bfs_body_face_joint_pilot_20261008' in norm_path:
            sub = norm_path.split('bfs_body_face_joint_pilot_20261008')[-1].strip('/')
            if sub.endswith('faces'):
                purpose = "Research 归档：评测生成的人脸特写对比图目录。"
                pipeline_role = "用于审查人脸细节、五官结构与身份保持质量。"
            elif sub.endswith('images'):
                purpose = "Research 归档：评测生成的全身图像目录。"
                pipeline_role = "用于审查全身姿态迁移与整体视觉表现。"
            elif sub.endswith('panels'):
                purpose = "Research 归档：评测对比面板目录。"
                pipeline_role = "汇总源图、条件、生成图与真值的并排面板。"
            elif 'report' in sub:
                purpose = "Research 归档：BFS Joint Pilot 阶段成果图表与报告汇总。"
                pipeline_role = "保存人脸保持表格图与全景评测面板。"
            elif 'training' in sub:
                purpose = "Research 归档：BFH 联合训练过程记录目录。"
                pipeline_role = "记录 Body-Face-Hand 联合训练脚本与参数。"
            else:
                purpose = f"BFS Body+Face 联合训练 Pilot 调研证据与评测存档 `{sub}`。"
                pipeline_role = "保存该阶段的评测面板、人脸特写与调研记录。"
        elif 'cleanup_20261003' in norm_path:
            purpose = f"2026-10-03 目录大清理专项留存与备份子目录 `{folder_name}`。"
            pipeline_role = "留存历史清理前现场快照、远端旧记录、备份分组与清理工具。"
        elif 'deepgen_assessment' in norm_path:
            purpose = f"DeepGen-1.0-diffusers DiT 基座生成能力评估子目录 `{folder_name}`。"
            pipeline_role = "保存纯基座在不同提示词与采样配置下的原始渲染图与效果评估。"
        elif 'dfm_preview_20261008' in norm_path:
            purpose = f"DFM (Deep Facial Model) 人脸先验与分割采样预览子目录 `{folder_name}`。"
            pipeline_role = "包含人脸关键特征提取样例 (`face_samples`) 与精细部位分割图 (`segm_samples`)。"
        elif 'dna_rendering_20261008' in norm_path:
            purpose = f"DNA-Rendering 高精度三维人体数据集接入与签署子目录 `{folder_name}`。"
            pipeline_role = "保存待签署协议、往来沟通记录与官方参考渲染样本。"
        elif 'geometry_prompt_coordination_20261009' in norm_path:
            purpose = f"几何条件信号与语言提示词协同机制调研子目录 `{folder_name}`。"
            pipeline_role = "探索 SMPL-X 几何特征与 Prompt 语义在 DiT 跨注意力层中的竞争与协同关系。"
        elif 'synbody_audit_20261003' in norm_path:
            purpose = f"SynBody 合成三维人体数据集视觉质量审计子目录 `{folder_name}`。"
            pipeline_role = "评估合成人体的纹理真实感、边缘光影缺陷及多视角一致性。"
        elif 'tiktok_training_20261003' in norm_path:
            purpose = f"TikTok 数据集接入与训练方案技术调研子目录 `{folder_name}`。"
            pipeline_role = "对比训练基线代码快照 (`baseline`) 与适配器增量变更 (`changes`)。"
        else:
            purpose = f"研究与调研子目录 `{folder_name}`。"
            pipeline_role = "保存专项调研文档、测试数据或审计记录。"
    else:
        purpose = f"`{folder_name}` 模块目录。"
        pipeline_role = "承载项目专属子功能或中间产物。"

    # Build Markdown Content
    lines = [
        title,
        "",
        path_info,
        "## 1. 目录定位与职责",
        "",
        purpose,
        "",
        f"**工程定位**: {pipeline_role}",
        "",
        "## 2. 目录结构与内容汇总",
        "",
    ]

    if subdirs:
        lines.append("### 子文件夹")
        lines.append("")
        for d in sorted(subdirs):
            lines.append(f"- **`{d}/`**: 该子目录下的专门功能模块或产物。")
        lines.append("")

    if files:
        lines.append("### 文件概览")
        lines.append("")
        lines.append(f"- **文件总数**: {len(files)} 个文件")
        lines.append(f"- **文件类型统计**: {type_summary}")
        lines.append("")
        
        # Display sample files or all if <= 20
        if len(files) <= 25:
            lines.append("| 文件名 | 说明 |")
            lines.append("|---|---|")
            for f in sorted(files):
                desc = FILE_DESCRIPTIONS.get(f, "该模块下实现/记录文件")
                lines.append(f"| `{f}` | {desc} |")
        else:
            lines.append(f"由于文件数量较多（共 {len(files)} 个），下表列出部分代表性文件：")
            lines.append("")
            lines.append("| 文件名（样例） | 说明 |")
            lines.append("|---|---|")
            for f in sorted(files)[:15]:
                desc = FILE_DESCRIPTIONS.get(f, "代表性数据/代码/图像文件")
                lines.append(f"| `{f}` | {desc} |")
            lines.append(f"| *...（其余 {len(files) - 15} 个文件）* | 同类型输出或支持文件 |")
        lines.append("")

    # Relative path to root
    depth = len([p for p in parts if p and p != '.'])
    root_rel = "../" * depth if depth > 0 else "./"

    lines.extend([
        "## 3. 使用与关联说明",
        "",
        f"- 更多全局架构说明与模型演进路线请参阅根目录 [README.md]({root_rel}README.md) 或 [docs/PROJECT_STRUCTURE_20261003.md]({root_rel}docs/PROJECT_STRUCTURE_20261003.md)。",
        "- 本目录下的所有文件均遵循项目统一的代码风格与数据管理规范。",
        ""
    ])

    return "\n".join(lines)


def main():
    root_dir = '.'
    created_count = 0
    updated_count = 0
    preserved_roots = {
        '.', 'add', 'experiments', 'experiments/qwen-2.1-test', 'reports',
        'research/abc_showcase_20261003', 'research/cleanup_20261003/before',
        'research/cleanup_20261003/remote_records/cleanup_20261003',
        'research/cleanup_20261003/remote_records/cleanup_20261003/before_docs',
        'research/cleanup_20261003/remote_records/cleanup_20261003/before_docs/experiments',
        'research/cleanup_20261003/remote_records/home_tools_202609', 'scripts'
    }

    for root, dirs, files in os.walk(root_dir):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_NAMES and not d.startswith('.')]
        
        rel = os.path.relpath(root, root_dir)
        if rel != '.':
            parts = rel.split(os.sep)
            if any(p in IGNORED_NAMES or p.startswith('.') for p in parts):
                continue
            norm_path = rel.replace('\\', '/')
        else:
            norm_path = '.'

        # Check existing custom READMEs that were authored prior to this
        if norm_path in preserved_roots:
            continue

        readme_text = generate_readme_content(norm_path, dirs, files)
        target_path = os.path.join(root, 'README.md')
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(readme_text)
            
        updated_count += 1

    print(f"Execution complete.")
    print(f"Preserved custom authored READMEs: {len(preserved_roots)}")
    print(f"Generated/Updated detailed READMEs: {updated_count}")

if __name__ == '__main__':
    main()
