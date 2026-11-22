# SDPose 姿态对齐评测细则（项目统一口径）

> **生效范围**：新运行的 RL Pose/Hand Reward、训练门禁、留出集和 workflow 生成评测统一采用 **SDPose-Wholebody + YOLO11x**。CPU 评测也使用 SDPose。DWPose 用于明确标注的历史对照与已有生成条件资产。

---

## 1. 为什么切换为 SDPose

依据同库实测对比（`/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/`）：
- **单人 50 样本（Bilibili）**：LOO 3 方法中位数下，`SDPose` 取得最高 `pose_reward = 0.8693`、`PCK@0.02 = 98.24%`、`NME = 0.00390`，且全 134 点平均有效检出 `132.1 / 134`（手部 `Hands42 PCK@0.02 = 98.30%`，显著优于 DWPose 的 `91.97%`）。
- **双人 10 样本（Hi4D 含遮挡/拥抱）**：`YOLO11x` 在双人紧密交互下的双人检出率（`90%`）与目标召回率（`95%`）优于 YOLOX（`80%` / `90%`），`SDPose` 的 `GT Pose Reward = 0.6449`（DWPose `0.5414`）、`WholeBody134 PCK@0.05 = 91.83%`。

---

## 2. 标准模型、代码与运行环境

| 项 | 标准路径 / 配置 |
|---|---|
| 共享代码入口 | `/home/shangguanrz/project/pic-edit/src/evaluation/sdpose.py` |
| 姿态模型权重 | `/data/shangguanrz/models/SDpose/SDPose-Wholebody`（133 COCO-WholeBody 点） |
| 人体检测器权重 | `/data/shangguanrz/models/SDpose/SDPose-Wholebody/yolo11x.pt` |
| 上游推理代码 | `/data/shangguanrz/models/SDpose/SDPose-OOD` |
| 推荐 Python 环境 | `/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python`（同时支持 `SDPose-OOD` 与 `DINOv2` CUDA 推理；禁止往 `nihaochen/envs/qwen21` 混装冲突包） |

> ⚠️ **必须设置 `CUDA_HOME`**（2026-11-17 实测）：本机没有 `/usr/local/cuda`，而 `mmengine._strategy` 会 import `deepspeed`，其版本探测在找不到 CUDA toolkit 时直接抛 `MissingCUDAException`，导致 `from src.evaluation import sdpose` 之前就崩在 `pipelines.SDPose_D_Pipeline` 的 import 链上。统一用 qwen21 环境自带的 nvcc：
>
> ```bash
> export CUDA_HOME=/home/shangguanrz/nihaochen/envs/qwen21   # 该环境含 nvcc 12.8.93，不编译任何扩展
> ```
>
> 另外 `nvidia-smi` 在本机因内核/用户态 NVML 版本不一致会报 `Driver/library version mismatch`，属已知无害项；`torch.cuda` 正常，不要为此重装驱动。

---

## 3. 推理与前处理固定参数

为保证跨实验可比，`src/evaluation/sdpose.py` 默认锁定以下参数：

1. **人体框检测（`YOLO11x`）**：
   - `classes=[0]`（仅 person）
   - `conf = 0.25`、`iou = 0.65`
   - 若存在远景背景小人干扰（如户外/海滩场景），需设置前景面积阈值过滤（例如 `min_area = 5000.0 px²`），并**同时落盘** `fg_persons_detected`（过滤后）与 `raw_persons_detected`（原始 YOLO 检出数），禁止静默吞掉背景人检出计数。
2. **单框仿射与姿态推理（`SDPose_D_Pipeline`）**：
   - `GetBBoxCenterScale(padding=1.25)`
   - `TopdownAffine(input_size=(768, 1024), use_udp=True)`
   - `weight_dtype = torch.float32`
   - `timesteps = [999]`、`mode = "predict"`、`test_cfg = {"flip_test": True}`
3. **关键点置信度阈值与画布裁剪**：
   - 姿态对齐主评测默认 **`score_threshold = 0.30`**（`DEFAULT_SCORE_THRESHOLD = 0.30`，与 SDPose 官方渲染/有效点阈值一致；在极端遮挡补充敏感度分析时可额外加报 `0.10` 口径，但需显式标注）。
   - 必须同时校验 `np.isfinite(xy).all()`、坐标落在 `[0, W) × [0, H)` 画布内（`valid_points`），以及 `score >= threshold`。

评分入口调用 `load_scoring_runtime()` 和 `score_people()`。评分随机种子固定为 42，调用退出后恢复训练侧 RNG。RL 用目标侧条件固定关节分母和奖励资格，候选漏点计作失分，模型或评测器异常按失败记录。每次新运行将评分模型和代码的 SHA 固定在清单；从旧评分器切换时使用新的清单和独立结果路径。

---

## 4. 关键契约：133 → 134 关节槽位序约定（必读）

> ⚠️ **历史高发 Bug 警告**：仓库内存在两套 Body-18 眼/耳顺序，混用会造成约 **10.7 个百分点** 的虚假 `PCK` 下跌！

| 顺序名 | 调用方式 | 索引 `14..17` 语义 | 适用场景 |
|---|---|---|---|
| **`geometry` 序（默认）** | `order="geometry"`（调用 `src.pose_edit.geometry.cocowholebody133_to_dwpose134`） | `14=R_eye, 15=R_ear, 16=L_eye, 17=L_ear` | 与 `render_poseguide`、`maps_3d.npz['dwpose134_xy']`、`src/pose_edit/dwpose.py`、`workflow-multiple/evaluate_dual.py` 逐槽对齐 |
| **`openpose` 序** | `order="openpose"`（在 `geometry` 序基础上 `swap_eye_ear_slots` 交换 `15 <-> 16`） | `14=R_eye, 15=L_eye, 16=R_ear, 17=L_ear` | 对齐 OpenPose JSON、`Sapiens2` 映射、`src/rl/metrics.py::DWPOSE134_TO_COCO17` |

- **Neck（槽位 `1`）派生规则**：两肩均值 `(kpts133[5] + kpts133[6]) / 2`，其分数定义为两肩较弱者 `min(sc133[5], sc133[6])`（任一肩点无效则置 `0.0`）。
- **严禁**：不经 `order` 检查就把 `openpose` 序点直接送进按 `geometry` 序导出的 `dwpose134_xy` 评测。
- **本数据集实测（2026-11-17，1,960 帧逐点验证）**：`/data/shangguanrz/datasets/bilibili/frames/poses/<fid>.json`（DWPose）是 **`openpose` 序**，而 `geometry/skeleton/sdpose_2d_kp/*_keypoints.json`（SDPose）是 **`geometry` 序**。逐槽直比时槽位 15/16 误差 61–132 px，交换后降到 1.3–8.7 px；其余 16 个槽位不受影响。跨后端比对请直接用 `scripts/audit/check_sdpose_vs_dwpose_agreement.py`（默认按 openpose 解析 DWPose 侧）。

---

## 5. 双人/多人与单人评测细则

### 5.1 单人任务
- 若同帧存在多框，按**主体 IoU** 或 **最大前景框** 选取目标，不得盲目取 `boxes[0]`。
- 关键点坐标统一转至**去留边后的有效内容画布**（`final_image.png`）或**同尺度 GT 坐标系**后再逐点求欧氏距离。

### 5.2 双人任务（沿用 `workflow-multiple` 交互协议 + SDPose 后端）
1. **人数校验**：先查 `fg_persons_detected == 2`；同时保留 `region_count`（按各目标请求框扩大 20% 裁切后逐框调用 SDPose `detect_people` 统计是否“每个请求位都确有身体”）。
2. **人物-槽位分配**：
   - 全图前景刚好 2 人时，尝试两种排列并按**髋心总距离最小**分配（差值 `< 40 px` 标 `ambiguous = true`）。
   - 全图合并/多检（`fg_persons_detected != 2`）时，退回逐请求区域 SDPose 检出（`pose_basis = "regions"`）完成可评测关节打分，同时明确保留人数未通过事实。
3. **必备姿态指标（基于 Body-18 共同有效点，按请求 `torso_length` 归一化）**：
   - `body_pck@0.1`、`body_pck@0.2`、`body_pck@0.5`（含左右分人 `@0.2`）
   - `body_mpjpe_px`、`body_mpjpe_norm`
   - `torso_scale_ratio`（检出躯干 / 请求躯干，理想值 `1.0`）
   - `hip_gap_error_px` & `hip_gap_error_rel`（双人相对站位间距误差）
4. **配套外观与背景指标（与姿态解耦）**：
   - **外观与防串人/防 `B` 泄漏**：DINOv2 (`dinov2_vits14`) 裁切相似度矩阵，输出 `identity_similarity_{mean,left,right}`、`identity_swap_suspected`（margin `0.02`）及 `bodies_liking_B_more` / `target_minus_source`。
   - **背景保真**：在源前景框与目标请求框并集以外区域计算 `bg_mae` 与 `bg_psnr_db`。
5. **可视化叠加（Overlay）**：每个评测结果必须产出叠图（白圈 = 请求目标 `0..17` 关节，绿点/蓝点 = SDPose 在生成图上检出的左/右人物关节），用于肉眼复核“算分是否真的落在肉体对应关节上”。

---

## 6. 数值与落盘纪律

- 不可测指标（如缺髋心、无共同有效关键点、`mse == 0` 时的 PSNR）一律写 `null`（经 `finite()` 转换），严禁用 `0` 或 `0.0` 替代“测不出”。
- 新增/重跑 SDPose 评测不得直接覆盖历史 DWPose 对照结果文件；优先独立命名为 `eval_sdpose.json` / `evaluation_sdpose_*.json` 或记录 `pose_backend: "sdpose"`。
- `AGENTS.md` 中的 `smplestx_bw` / `qwen21` 既有环境不得擅自升级或降级以迁就 SDPose；统一通过 `/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python` 执行 SDPose 评测。
