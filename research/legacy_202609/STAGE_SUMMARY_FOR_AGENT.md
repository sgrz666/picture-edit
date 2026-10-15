# 人体姿态编辑项目阶段性技术总结与 Agent 交接文档
> **文档用途**：供后续接手或协作的 AI Agent / 开发者快速同步项目全貌、技术架构、资产路径、实验结论及待办清单。
> **更新时间**：2026-09-12
> **当前阶段**：单人姿态编辑（SMPL 3D 几何提取 ➔ DeepGen DiT 注入 ➔ 单样本验证通过）

---

## 1. 项目核心目标与技术路线

### 1.1 核心任务
实现**单人（未来扩展至双人）人体图像姿态迁移与精准编辑**：
* **输入原图 A**：提供人物身份外貌（五官、发型、体型）、服装款式/材质纹理、场景背景；
* **几何控制条件**：由目标动作 B 提取的 3D SMPL 人体网格（渲染出 3D 表面法向 Normal 图、2D 骨骼图、深度图）；
* **输出编辑图**：人物的动作变更为目标姿态，同时 100% 保持 A 的身份、衣着和背景。

### 1.2 三阶段技术架构
```
[阶段一: 3D参数解析与SMPL网格]
  · 视频/图片 B ──▶ 提取 72维 Pose + 10维 Shape + 相机参数 [scale, tx, ty]
  · 加载 SMPL 模型 ──▶ 生成 6890 个三维顶点与面片网格 (Vertices & Faces)
       │
       ▼
[阶段二: 多模态几何控制图渲染 (Control Signals)]
  · 弱透视相机到针孔坐标系转换 ──▶ 渲染对齐的 512×512 8通道张量:
    - 3ch 表面法向图 (Normal Map): 身体部位空间曲率与朝向
    - 3ch 2D骨骼图 (Skeleton Map): 关节点走向与肢体拓扑
    - 1ch 深度图 (Depth Map): 空间前后层级与遮挡关系
    - 1ch 接触图 (Contact/Alpha)
       │
       ▼
[阶段三: DeepGen 1.0 DiT + PoseConditionAdapter 生成]
  · 冻结基础大模型 (Qwen2.5-VL 视觉大模型 + 2.47B SD3 DiT 主干)
  · 仅训练轻量 PoseConditionAdapter (172.77M 参数):
    - 卷积下采样提取 8ch 几何张量空间特征
    - 匹配 DiT 潜空间 Patch (32×32=1024 tokens)
    - 残差注入 DiT 各 Transformer Block (block_controlnet_hidden_states)
  · 流匹配逆向去噪 (Flow Matching Denoising) ──▶ 高清生成目标图
```

---

## 2. 硬件环境与服务器资产清单

### 2.1 主机与环境
* **本地工作区**：Windows 11，路径 `d:\codeplus\pictureedit`
* **远程计算节点**：
  * **SSH 别名**：`sg`（Host: `node01`, IP: `172.24.247.154`, User: `shangguanrz`）
  * **GPU**：1 × NVIDIA RTX 6000D（**84 GB 超大显存**，CUDA 12.4）
  * **Conda 虚拟环境**：`/home/shangguanrz/miniconda3/envs/deepgen`
  * **Python 版本**：Python 3.12
  * **核心依赖**：`torch==2.5.1`, `diffusers==0.32.1`, `transformers`, `smplx`, `trimesh`, `opencv-python`, `accelerate`
* **远程项目根目录**：`/home/shangguanrz/project/pic-edit/`

### 2.2 核心模型与权重资产路径
| 资产名称 | 路径 (远程服务器) | 状态 / 说明 |
|---|---|---|
| **DeepGen-1.0 扩散模型** | `/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/` | 完整 Diffusers 格式（包含 DiT、Qwen2.5-VL、SCB Connector、VAE） |
| **SMPL 中性人体模型** | `/home/shangguanrz/project/pic-edit/models/smpl/smpl/SMPL_NEUTRAL.pkl` | **已解耦 chumpy**，转换为纯 NumPy/PyTorch 格式，完美兼容 Python 3.12 |
| **iPER 官方样本数据集** | `/home/shangguanrz/project/pic-edit/datasets/iPER_sample/` | 包含 `smpls.zip`（参数）与 `iPER_256_video_release.zip`（视频帧） |
| **提取的调试切片样本** | `/home/shangguanrz/project/pic-edit/datasets/iPER_sample/sample_001_32_2/` | 原图 A (`frame_0000.jpg`)、目标 B (`frame_0100.jpg`)、3D Normal、骨骼、深度图 |
| **训练好的 Adapter 检查点** | `/home/shangguanrz/project/pic-edit/experiments/iper_adapter_v1/adapter_step_100.pt` | 100 步过拟合训练权重 (172.77M) |

---

## 3. 关键核心代码与脚本说明

| 脚本文件 | 位置 | 核心职责 |
|---|---|---|
| `clean_smpl_pickle.py` | 项目根目录 | 清洗 SMPL 官方旧 pickle，移除 Python 2 时代的 chumpy 依赖，生成现代化 SMPL 权重 |
| `src/adapter/pose_adapter.py` | 项目根目录 | `PoseConditionAdapter` 架构实现（8通道输入，分层提取并投影至 DiT 隐层维度 1536） |
| `train_adapter.py` | 项目根目录 | Adapter 训练脚本：冻结 DiT 和 VAE，仅优化 Adapter，支持 Flow Matching 速度预测与残差注入 |
| `test_iper_deepgen.py` | 项目根目录 | DeepGen Pipeline 原生图像编辑与文本/图像条件联合推理脚本 |
| `build_visual_report_docx.py` | 本地 scratch | 自动编译生成排版精美、以图为主的 Word 实验报告（docx） |

---

## 4. 关键技术突破与避坑经验 (CRITICAL INSIGHTS)

### 4.1 Token 序列长度拼接匹配机制
* **现象**：当向 DeepGen 输入参考图 A 时，DeepGen 将目标潜在 Token（1024）与参考图潜在 Token（1024）拼接为 **2048 个长 Token 序列**。
* **解决方案**：Adapter 针对目标 512×512 控制图提取出的残差张量形状为 `[B, 1024, 1536]`。向 DiT 注入时，**必须在参考图维度补零**：
  ```python
  residuals = [torch.cat([r, torch.zeros_like(r)], dim=1) for r in residuals] # 变为 2048 长度
  ```
  这样既能对目标动作精准施加几何控制，又不会污染参考图 A 的特征表征。

### 4.2 iPER 弱透视相机的绝对精准转换
* iPER 数据集提供的相机参数为 `[scale, trans_x, trans_y]`。
* 转换为 512×512 针孔投影的标准公式：
  $$\text{focal\_length} = \text{scale} \times \frac{W}{2} = \text{scale} \times 256.0$$
  $$c_x = 256.0, \quad c_y = 256.0, \quad T = [trans\_x, trans\_y, 1.0]$$
* **实测验证**：渲染出的 3D 骨架与视频真值（Ground Truth）重叠对齐（Overlay Check），误差小于 1 个像素。

### 4.3 为什么必须引入 3D 控制？（左右手反向实验印证）
* **大模型纯文本编辑的缺陷**：在早期纯文本提示驱动下，虽然指定了动作，但由于扩散大模型在语言理解上天生存在“人物右手 vs 观察者视角右侧”的混淆，大模型生成了相反的手臂动作（镜像反向）。
* **结论**：文本大模型缺乏三维刚性几何常识，**必须由 3D Normal / 骨骼控制网络强行在空间像素层面指定左右与关节自由度**。

### 4.4 人脸特征漂移（Identity Drift）的根本原因
* **原图极低分辨率瓶颈**：iPER 数据集单帧分辨率仅为 **256×256**，全身镜头下，人脸区域**仅有 28×40 像素**（马赛克级别）。
* **大模型的脑补机制**：视觉大模型无法捕捉细粒度五官，在 512×512 生成时依照先验“自作主张”脑补了一张标准亚洲微笑脸。
* **落地解决建议**：真实工业界应用需输入 1024 原图，或者接入 **InsightFace (512维 FaceID 向量)** / **CodeFormer 面部超分贴回**模块。

---

## 5. 当前关键实验产物

所有实验图像与报告均已保存在工作区 `d:\codeplus\pictureedit\research\`：
1. **[单人姿态编辑效果对比_左右修正版.png](file:///d:/codeplus/pictureedit/research/单人姿态编辑效果对比_左右修正版.png)**：
   4 联排对比图（原图 A ➔ 3D Normal 举手向右 ➔ 模型生成图向右精准举手 ➔ 真实目标 B 真值对照）。
2. **[人脸低分辨率马赛克与生成脑补对比.png](file:///d:/codeplus/pictureedit/research/人脸低分辨率马赛克与生成脑补对比.png)**：
   28 像素原脸 vs 模型脑补脸 vs 真值脸的放大分析图。
3. **[单人姿态编辑实验报告_图文简报_高清修正版.docx](file:///d:/codeplus/pictureedit/research/单人姿态编辑实验报告_图文简报_高清修正版.docx)**：
   排版完整的 Word 实验总结。

---

## 6. 下一步工作待办清单 (TODO FOR NEXT AGENT)

1. [ ] **深度打通 Adapter 注入与官方 DeepGenPipeline**：
   在 `models/DeepGen-1.0-diffusers/deepgen_pipeline.py` 的 `_SD3Pipeline.__call__` 中开放 `block_controlnet_hidden_states` 参数传递，使得 Adapter 的 8 通道控制特征能直接享受 Qwen2.5-VL 多模态文本/图像条件以及 CFG（Guidance Scale = 4.5）的高画质去噪红利。
2. [ ] **构建多样本/批量微调数据集**：
   当前仅在 1 个样本切片（`001_32_2`）上做了过拟合验证。接下来需编写脚本自动解压并处理 iPER 其余 30 个视频序列，构建含有 500~2000 对数据的训练集，进行 1000~3000 步的通用泛化训练。
3. [ ] **引入 FaceID 身份锁机制**：
   评估接入 InsightFace 或 ArcFace 提取的面部嵌入向量，与 DiT Cross-Attention 层对接，在改变姿态的同时完全锁死面部 ID，消除人脸漂移。
4. [ ] **推进至双人互动姿态编辑**：
   基于现有的双人交互切片（握手、搭肩），引入双方 SMPL 重叠碰撞/接触图（Contact Map），验证复杂多人物交互控制。
