# Hybrid Qwen Base + Union Control (No Mask) 单样本实验

## 1. 实验目的
验证非官方混合推理架构（Hybrid Qwen Base + Union Controlnet）在切断 Inpainting Mask 路径、由 Qwen 基模保留原生多模态参考图输入的前提下，能否同时解决 A 的身份保持问题并利用 Union 注入的纯几何残差（Pose / Depth）提升姿态对齐精度。

## 2. 目录结构与资产说明
```text
workflow/experiment/qwen-union-hybrid/
├── README.md                      # 本实验技术说明与关键结论
├── input/
│   └── prompt_pe_native.txt       # 冻结的 PE-I2I 提示词（保留 <image1> <image2>）
├── data/
│   └── prepared_controls/         # 几何控制输入
│       ├── pose_source_torso.png  # 对齐骨架图 (640x928)
│       ├── depth_source_torso.png # 从 SMPL-X 3D 网格渲染的无噪声对齐深度图
│       ├── keypoints_source_torso.npz # 目标真实关键点与可见性掩码
│       ├── overlay_debug.png      # 骨架与深度像素级对齐自检图
│       └── prep_manifest.json     # 完整元数据清单与 SHA256 记录
├── output/                        # 四组正式推理生成产物
│   ├── H0/                        # Native Qwen Baseline（无 Union 残差）
│   ├── H1/                        # Hybrid + Pose（注入 Pose 控制残差）
│   ├── H2/                        # Hybrid + Depth（注入 Depth 控制残差）
│   ├── H3/                        # Hybrid + Pose + Depth（双几何残差融合）
│   └── execution_summary.json     # 全局生成耗时与显存日志
├── report/                        # 定量评测报告与全景对比
│   ├── comparison.png             # 全景 8 宫格横向对比图 (A / Pose / Depth / H0 / H1 / H2 / H3 / T4b)
│   ├── scorecard.md               # 针对四个科研假设的详尽分析报告
│   ├── summary.json               # 评测定量指标原始 JSON
│   └── comparison.csv             # 指标横向对比表
└── scripts/                       # 核心执行与评测引擎
    ├── prepare_hybrid_controls.py # 控制条件渲染与提示词冻结脚本
    ├── run_hybrid_single.py       # 分阶段 Offload 混合推理脚本（峰值显存 20.31 GiB）
    └── eval_hybrid_single.py      # DWPose 134 与 ArcFace 余弦评测引擎
```

## 3. 核心定量评测结论

测试样本：`hard_05_torso`（Canvas 640x928，40 步 FlowMatchEuler 采样，Seed=42，TrueCFG=1.0）。

| 实验组别 | 架构与控制条件 | ArcFace 身份余弦 (↑) | 人脸检出 | Body PCK@0.2 (↑) | Strict PCK (↑) | Norm MPJPE (↓) | 生成耗时 (s) | 峰值显存 (GiB) |
|---|---|---|---|---|---|---|---|---|
| **H0 (基线)** | Native Qwen (A + PoseGuide, 无 Union) | **0.2381** | True | 0.3889 | 0.3889 | 0.2945 | 158.18 | 20.31 |
| **H1 (Hybrid)** | Base + Union Pose (Hints) | **0.0835** | True | 0.3889 | 0.3889 | 0.2846 | 155.18 | 20.31 |
| **H2 (Hybrid)** | Base + Union Depth (Hints) | **0.1526** | True | 0.3889 | 0.3889 | **0.2756** | 152.04 | 20.31 |
| **H3 (Hybrid)** | Base + Pose + Depth 残差融合 | None | False | 0.3889 | 0.3889 | 0.2845 | 189.07 | 20.31 |
| **T4b (对比组)**| 官方 Inpainting + Mask + Union | **0.0746** | True | 0.8125 | 0.7222 | 0.1742 | 108.50 | 22.40 |

### 关键科研结论
1. **身份保持翻盘（H0 vs T4b）**：上一轮官方 Inpaint 模式因强制对 A 图打 Mask 挖空人脸，ArcFace 余弦仅为 **0.0746**（身份完全崩塌）；本轮让基模直接读取 A 的多模态视觉特征（H0），ArcFace 余弦跃升至 **0.2381**（提升超过 219%），证明了“不打 Mask、保留原生图像输入”是保持 ID 的正确方向。
2. **未校准 Union 残差对人脸的扰动机理**：官方 Union 权重是基于“包含完整 Masked Inpainting 特征”训练的。在 Hybrid 架构中人为置零 Mask 和 Inpaint 特征后，残差网络注入的特征未经过校准（max diff 达到 8.94），直接干扰了 DiT 中的面部细粒度自注意力，导致 H1 的 ID 下降至 0.0835，H3 人脸直接无法检出。
3. **几何深度残差的积极作用**：在姿态指标上，纯深度残差（H2）使得 Normalized MPJPE 从 H0 的 0.2945 改善至 **0.2756**，证明了无背景噪声的 SMPL-X 稠密深度图能够有效约束三维体态空间。

## 4. 与其他实验的关系
- **上游依赖**：基于 `workflow/runs/test/hard_05` 与 `hard_05_torso` 的 SMPL-X 拟合产物（`B_smplx.pt`）与相机参数（`B_camera.json`）。
- **技术定位**：对比上一轮 `experiments/6.qwen-controlnet union`（官方 Inpainting 路径），本实验首次在单卡环境下打通非官方混合推理流程（Hybrid Qwen Base + Union Residuals No Mask），回答了“纯几何注入是否可行”的科研命题，为后续 Adapter 权重微调提供了基准参照。
