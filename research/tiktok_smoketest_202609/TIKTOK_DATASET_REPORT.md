# TikTok 数据集调研与适配方案深度报告

## 一、 数据集存储与物理规模

在服务器 `sg` 上，TikTok 数据集完整存储于：
**`/home/shangguanrz/project/pic-edit/datasets/TikTokDataset`**

总磁盘占用约为 **~89 GB**，主要由以下核心子目录构成：

| 目录名称 | 磁盘占用 | 内容与说明 | 状态 |
| :--- | :---: | :--- | :---: |
| `TikTok_dataset/` | **48 GB** | 原始解压帧序列（包含 340 个视频目录，每个目录含 `images/`、`masks/`、`densepose/`） | 完整可用 |
| `TikTok_3d_assets_native/` | **38 GB** | 基于 NVIDIA nvdiffrast 预渲染的 3D 几何资产与 SMPL-X 参数 | 核心资产已完成 |
| `TikTok_Raw_Videos/` | **2.5 GB** | 68 个原始 MP4 舞蹈视频与原链接信息 | 归档参考 |
| `model/` | **665 MB** | 官方深度与法线预测 TensorFlow 模型权重 | 原数据集提供 |
| `splits/` | **20 KB** | 官方/标准数据划分配置 (`train`, `val`, `test` 序列列表与统计) | 配置就绪 |
| `showcase_single/` | **9.6 MB** | 单样本多模态展示 Panel (Source, Target, Mask, DensePose, Depth, Normal, SMPL-X Mask) | 抽样验证通过 |

---

## 二、 序列规模与划分（Splits）情况

依据 `splits/split_summary.json` 及全库扫描统计：

- **总视频序列数**：`340` 个视频序列；
- **总帧数**：**92,961 帧**（每帧均具有 100% 对应的 Person Mask，无空缺）；
- **已渲染 3D 资产的序列**：**296** 个序列（共 **80,541** 帧通过 `_ASSETS_DONE.json` 校验）；
- **标准 Split 与 3D 资产覆盖率**：
  - **Train Split**：**179 个序列**（49,202 帧）$\to$ **100% 拥有完整 3D 资产**；
  - **Val Split**：**20 个序列**（5,357 帧）$\to$ **100% 拥有完整 3D 资产**；
  - **Test Split**：**20 个序列**（5,887 帧）$\to$ **100% 拥有完整 3D 资产**；
  - **合计 219 个标准评估序列全部已具备 3D 资产！**

---

## 三、 数据模态与现存资产清单

在 `TikTok_dataset` 与 `TikTok_3d_assets_native` 中，每一个有效序列（例如 `00001`）包含以下模态：

1. **RGB 原图 (`images/`)**：
   - 分辨率：**604 × 1080**（标准 9:16 竖屏手机录制画面）；
   - 特点：真实室内/卧室背景，服装多样（卫衣、紧身衣、亮色外套等），动作动态大；
2. **前景人像掩膜 (`masks/`)**：
   - 分辨率：`604 × 1080`，单通道二值 Mask，覆盖全部 92,961 帧；
3. **DensePose 姿态映射 (`densepose/`)**：
   - 分辨率：`604 × 1080`，RGB 编码的 IUV 身体表面参数化分布；
4. **SMPL-X 3D 参数 (`smplestx/params.pt`)**：
   - 包含完整的 SMPL-X 人体模型拟合参数字典：
     - `global_orient`: `[N, 3]`
     - `body_pose`: `[N, 63]`（21 个身体关节旋转）
     - `left_hand_pose`: `[N, 45]`（15 个手部关节旋转）
     - `right_hand_pose`: `[N, 45]`（15 个手部关节旋转）
     - `jaw_pose`: `[N, 3]`（下颌开合旋转）
     - `betas`: `[N, 10]`（体型参数）
     - `expression`: `[N, 10]`（面部表情参数）
     - `mesh_cam`: `[N, 10475, 3]`（相机空间 10,475 顶点绝对三维网格）
     - `transl`: `[N, 3]`（相机平移）
     - `focal_length_xy`, `princpt`, `width`, `height`（相机内参）
5. **nvdiffrast 渲染几何图像**：
   - `normal/`: 相机空间 SMPL-X 表面法线图（RGB，`604 × 1080`，+X右, +Y上, +Z向前）
   - `depth_vis/`: 相机空间 Z 深度可视化图（单通道灰度，`604 × 1080`）
   - `smplx_mask/`: SMPL-X 网格光栅化轮廓掩膜（单通道灰度，`604 × 1080`）

---

## 四、 TikTok 与 iPER 数据集全方位对比

| 对比维度 | iPER 数据集 | TikTok 数据集 | 对当前模型架构的影响与意义 |
| :--- | :--- | :--- | :--- |
| **画面构图与画幅** | 1024×1024 正方形，全身直立，脚部往往完整 | **604×1080 竖屏 (9:16)**，以半身/大半身跳舞动作为主 | 视野更聚焦于上半身，面部与手部像素占比显著高于 iPER |
| **背景与光照环境** | 单一室内/纯色白墙背景，光照均匀固定 | **真实家庭/生活环境**，光照变化复杂，包含阴影与反光 | 更考验 DiT 的背景保持与解耦迁移能力 |
| **肢体动作与动态** | A-pose 舒展、缓慢全身旋转、简单挥臂 | **高动态律动舞蹈、手势舞、快速手部交叠** | 对手部与脸部局部细节分支（V6.4 / V6.5）提出更高要求 |
| **面部表情** | 表情极其平淡单一（几乎无张嘴或大幅表情） | **极丰富**（大笑、搞怪、对口型张嘴、眨眼等） | **极度适合验证 V6.5 Face Adapter 的表情控制力** |
| **现有资产就绪度** | 512×512 Normal/Part/Pose/DWPose 全部就绪 | 3D Normal/Depth/Mask/SMPL-X 就绪，缺 Part 与 Keypoints | 需进行轻量化前置补充处理 |

---

## 五、 接入当前 V6/V6.5 训练链路的技术路线 (Action Plan)

要将 TikTok 投入实际训练或单样本评测，推荐以下 4 步流程：

### 1. 分辨率对齐策略（推荐方案）
- **方案 A（等比例人像居中裁剪到 512×512，推荐首选）**：
  - 由于 TikTok 人物主要集中在中部（宽 604，高 1080），可根据 SMPL-X 投影包围盒或中心裁剪为 $604 \times 604$ 并双三次下采样至 $512 \times 512$；
  - **最大优势**：可 100% 无缝复用现有的全部 V6.3 Geometry Injector、V6.4 Hand/Detail Branch 和 V6.5 Face Adapter 预训练权重，无需改动 DiT 分辨率。
- **方案 B（保持 9:16 竖屏，如 576×1024）**：
  - DeepGen 原生支持多画幅，但需注意下采样对齐倍率（须满足 64 的整倍数）。

### 2. 补齐两个缺失的输入条件
1. **14-Class Part Map (`part_onehot`)**：
   - 现有 `params.pt` 已保存了相机空间 `mesh_cam`（10,475 顶点）；
   - 使用项目内已有的 `SMPLXPartRenderer`（基于 `nvdiffrast`），每个序列仅需数秒即可直接批量生成 `part_512`（耗时短、精度高）。
2. **2D 姿态骨骼点 (`pose_heatmap`)**：
   - 方式：调用项目内已配置的 DWPose 模型批处理生成 25 关键点，或直接从 SMPL-X 模型回归骨骼点并渲染高斯热图。

### 3. V6.5 Face 特征预计算
- 针对要训练的 TikTok 序列，直接运行已验证的 `scripts/precompute_iper_face_features_v65.py`：
  - 自动提取 ArcFace 512D 身份向量；
  - 自动提取 DINOv2-Giant 256×1536 特征图；
  - 提取 72 维 2D 面部关键点与面部 ROI 包围盒；
  - 保存为 `v65_face_features.pt`。

### 4. 构建 `src/data/tiktok_dataset.py`
- 参考现有的 `IPERV65FaceOverfitLoader` 与 `IPERDetailDataset`；
- 支持**同视频自驱动**（例如选取第 14 帧作为 Source，第 74 帧作为 Target）以及**跨视频姿态迁移**；
- 统一输出项目标准的 `batch` 字典，直通 `train_iper_v65_face.py` 或多样本训练脚本。
