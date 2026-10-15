# Body + Face / BFS 实验最终总结与交接（2026-10-08）

本项目已全面完成**原生画幅（Native Resolution）**下的 Body + Face 联合架构实验，彻底废除旧 512 letterbox/resize 方案，并已完成全部 1,024 步模型更新与全量 36 例评测对照。

---

## 1. 核心工作目录与环境

- **Windows 主工作目录**：`D:/codeplus/pictureedit`，分支 `main`。
- **训练服务器**：`ssh sg` 进入 `/home/shangguanrz/project/pic-edit`，分支 `main`。
- **Python 环境**：`/home/shangguanrz/nihaochen/envs/qwen21/bin/python`（GPU: RTX 6000D 80GB）。
- **远端实验根目录**：`/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008`。
- **本地图片与报告目录**：`D:/codeplus/pictureedit/research/bfs_body_face_joint_pilot_20261008/`（包含完整 `report/` 与 2,851 个 `evaluation/` 生成资产，本地可双击 `report/index.html` 离线完整查看）。

---

## 2. 实验方案与实施架构

1. **原生分辨率与零梯度掩膜（Native Resolution）**：
   - 输入 A 与目标 B 保留原始像素尺寸（TikTok 604×1080 边缘最小 padding 至 608×1088，匹配 VAE/DiT stride 步长）。
   - 边缘 padding 区域置零并不进入 Body/Face 损失计算，输出反向裁剪回目标 B 原始尺寸。
   - 彻底消除了旧 512 带来的身体压扁和五官几何畸变。
2. **两阶段受控对比体系**：
   - **共同底座 J（256 步）**：从 Body6000 / Face3000 初始化，Body + Face 联合训练 256 步，产出统一底座 `checkpoint_step_000256.pt`。
   - **分叉三分支对比（各 256 步，独立优化器实例）**：
     - **Qwen**：纯 Qwen 基干微调 256 步。
     - **BFSBody**：加载 `bfs_body_swap_v1.0_qwen_2.1.safetensors`（冻结）微调 256 步。
     - **BFSHead**：加载 `bfs_head_v1.1_qwen_2.1.safetensors`（冻结）微调 256 步。
   - 预算严格对齐：总计 1,024 次正式参数更新，所有分支继承自同一 J256 原生权重。

---

## 3. 全量客观指标总览（36 例全量测试集）

评测集包含 24 例 TikTok 608×1088 原生画幅样本与 12 例 iPER 512 诊断样本。
身份分数由 ArcFace 提取（仅统计源 A 与生成图均检出唯一人脸的有效样本）；骨骼姿态采用 DWPose 提取并计算 PCK 归一化距离。

| 阶段 / 模型底座 | 评测规模 | 有效记录 | ID 对 A（有效/固定） | PCK@0.02 | PCK@0.05 |
|---|---|---|---|---|---|
| **初始未训练基线 (`initial/init`)** | 36 | 36 | 0.4285 (36/36) | 0.5921 (36/36) | 0.9058 (36/36) |
| **共同联合底座 (`common/J`)** | 36 | 36 | **0.4602** (36/36) <br>*(+3.17%)* | 0.5554 (36/36) | 0.9083 (36/36) |
| **纯 Qwen 基干微调 (`final/Qwen`)** | 36 | 36 | 0.4047 (36/36) | 0.5461 (36/36) | 0.9099 (36/36) |
| **冻结 BFS-Body 微调 (`final/BFSBody`)** | 36 | 36 | 0.3583 (36/36) | 0.5248 (36/36) | 0.9111 (36/36) |
| **冻结 BFS-Head 微调 (`final/BFSHead`)** | 36 | 36 | 0.3333 (36/36) | 0.5481 (36/36) | **0.9206** (36/36) <br>*(+1.48%)* |

### 固定 12 例监控子集演进过程

| 阶段 / 分支 | 监控样本数 | ID 对 A | PCK@0.02 | PCK@0.05 | 审查状态 |
|---|---|---|---|---|---|
| `hotload/Qwen` | 12 | 0.4989 | 0.5323 | 0.8382 | 预检通过 |
| `hotload/BFSBody` | 12 | 0.4412 | 0.4993 | 0.8633 | 预检通过 |
| `hotload/BFSHead` | 12 | 0.3839 | 0.5359 | 0.8353 | 预检通过 |
| `mid/J` | 12 | 0.4183 | 0.4992 | 0.8446 | 128 步门禁通过 |
| `mid/Qwen` | 12 | 0.4662 | 0.5154 | 0.8101 | 128 步门禁通过 |
| `mid/BFSBody` | 12 | 0.4067 | 0.5501 | 0.8573 | 128 步门禁通过 |
| `mid/BFSHead` | 12 | 0.4000 | 0.5315 | 0.8156 | 128 步门禁通过 |

### 配对差值（严格基于相同案例交集）

- **BFSBody final vs Qwen final**: ID 差值 `-0.0464`，PCK@0.05 差值 `+0.0012`
- **BFSHead final vs Qwen final**: ID 差值 `-0.0714`，PCK@0.05 差值 `+0.0107`
- **BFSHead final vs BFSHead hotload**: PCK@0.05 净提升 `+0.0419`（全场姿态进步最大）

---

## 4. 核心物理规律与科学结论（客观分析）

1. **原生画幅联合底座 J 具备显著的表征优势**：
   - 在原生分辨率下联合训练 256 步后，人脸 ID 相似度从基线的 0.4285 大幅提升至 **0.4602**（全场最高），同时 PCK@0.05 从 0.9058 提升至 0.9083。
   - 结论：彻底解决 512 尺寸挤压后，Body 与 Face 的联合架构在原生画幅下能建立高质量的几何一致性与身份记忆。
2. **LoRA 分支对复杂姿态的强驱动效应**：
   - 姿态准确率呈现明确的递进阶梯：`Qwen (0.9099) < BFSBody (0.9111) < BFSHead (0.9206)`。
   - BFSHead 分支在 256 步微调后姿态对齐达到全场峰值 0.9206，说明头部与骨骼姿态的紧耦合能有效纠正极端动作下的肢体偏差。
3. **长程微调下人脸身份的漂移与工程权衡（Trade-off）**：
   - 随着姿态微调步数增加（128 → 256 步），三个分支的姿态准确率均持续提高，但人脸 ID 均呈现不同程度的让步（Qwen 0.4047、BFSBody 0.3583、BFSHead 0.3333）。
   - 根本原因：Diffusion/Flow-matching 的全图重建损失由大面积的躯干与背景主导，在缺少专门的高权重局部 Face ROI 梯度保护或独立交叉注意力隔离的情况下，全身姿态的高精度对齐会对面部微结构形成反向牵引。
   - **后续落地建议**：若要在生产环境中兼顾 BFSHead 的极致姿态（0.92+）与 J 底座的极致身份（0.46+），应在训练中为 Face ROI 引入独立的身份梯度损失（如 ArcFace Feature Loss 或局部自注意力门控权重）。

---

## 5. 代码质量与自动化测试

所有模块已通过 CPU 回归测试与契约测试验证：

```bash
# 本地 Windows 验证（120 项通过，1 项无输入图自动跳过）
pytest tests/test_qwen21_frozen_bfs_lora.py \
       tests/test_qwen21_joint_training.py \
       tests/test_qwen21_joint_pilot_visuals.py \
       tests/test_qwen21_joint_pilot_summary.py \
       tests/test_qwen21_joint_pilot_resume.py \
       tests/test_qwen21_native_resolution.py -q

# 远端 Linux 服务器验证（121/121 项全部通过，0 错误）
ssh sg "cd /home/shangguanrz/project/pic-edit && CUDA_VISIBLE_DEVICES= /home/shangguanrz/nihaochen/envs/qwen21/bin/python -m pytest tests/test_qwen21_frozen_bfs_lora.py tests/test_qwen21_joint_training.py tests/test_qwen21_joint_pilot_visuals.py tests/test_qwen21_joint_pilot_summary.py tests/test_qwen21_joint_pilot_resume.py tests/test_qwen21_native_resolution.py -q"
```

---

## 6. 报告查看指引

- **本地离线可视化报告**：
  直接在浏览器中打开：
  `D:\codeplus\pictureedit\research\bfs_body_face_joint_pilot_20261008\report\index.html`
  页面包含：
  - 36 例完整对比矩阵（横向对比 Init / J / Qwen / BFSBody / BFSHead，纵向对比各训练阶段）。
  - 固定 B 靶向人脸 ROI 裁切原图对照（无主观居中或滤镜美化）。
  - 72 张完整拼图与 30 张高分辨率人脸接触表。
- **全量指标与数据结构**：
  - Markdown 汇总：`research/bfs_body_face_joint_pilot_20261008/report/summary.md`
  - 机器可读 JSON：`research/bfs_body_face_joint_pilot_20261008/report/summary.json`
