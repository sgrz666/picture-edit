#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据盘数据集构成说明（人工核定，供 gen_folder_docs.py 注入目录说明）。

键为目录的绝对路径（realpath），因此仓库内的符号链接视图与数据盘实体共用同一份说明。
"""

NOTES = {}

NOTES["/data/shangguanrz"] = r"""
**磁盘与角色**：本目录位于独立数据盘 `/dev/sdb1`（容量 22 TB，使用率约 5%），专门存放无法纳入 git 仓库的大体积实体：数据集原始数据、基座模型权重、可重生成的条件张量缓存、冷归档与选型样片。

**与代码仓库的挂接方式**：仓库 `/home/shangguanrz/project/pic-edit` 内以符号链接指回这里，路径映射的权威登记是仓库 `configs/asset_registry.json`（`as_of` 2026-10-03）：

| 仓库入口 | 数据盘实体 | 内容 |
|---|---|---|
| `datasets/TikTokDataset` | `pic-edit-datasets/TikTokDataset` | TikTok 单人舞蹈数据集全部构建层 |
| `datasets/Hi4D-origin` | `pic-edit-datasets/Hi4D-origin` | Hi4D 官方原始 tar.gz |
| `models/Qwen-Image-2.1` | `pic-edit-models/Qwen-Image-2.1` | Qwen-Image-2.1 基座权重 |
| `caches/qwen21_hi4d_controls` | `pic-edit-caches/qwen21_hi4d_controls` | Hi4D 双人控制条件张量缓存 |
| `experiments/review_20260908_v2`、`experiments/iper_v6_comparative_3000`、`models/Qwen-Image-Edit-20B`、`archive/cleanup_20261003` | `pic-edit-archive/cleanup_20261003/*` | 2026-10-03 清理移出的冷归档 |

**容量分布（子树累计）**：

| 区域 | 文件数 | 体积 | 性质 |
|---|---:|---:|---|
| `pic-edit-datasets/` | 549,557 | 495.54 GB | 数据集实体，不可再生，需长期保留 |
| `pic-edit-archive/` | 3,575 | 102.86 GB | 冷归档，只读 |
| `pic-edit-caches/` | 1,391 | 66.28 GB | 条件缓存，可由脚本重生成 |
| `pic-edit-models/` | 52 | 51.97 GB | 官方权重，可重新下载 |
| `dataset_samples/` | 489 | 7.46 GB | 选型调研样片 |
| `add_migration/` | 4 | <0.01 GB | 迁移脚本与传输日志 |

**写入约定**：本目录所有 `目录说明.md` 由 `scripts/gen_folder_docs.py` 生成；人工补充请写入同目录 `README.md`，生成器不会覆盖 README，也不会覆盖不含生成标记的说明文件。
"""

NOTES["/data/shangguanrz/pic-edit-datasets"] = r"""
数据集实体区，只含两个数据集：**TikTok**（单人舞蹈，多版本构建产物）与 **Hi4D**（双人交互，仅官方原始压缩包，未在此解压）。

- 合计 549,557 个文件、495.54 GB；其中 PNG 帧图 535,679 个，构成绝对主体。
- Hi4D 的解压与配对资产不落在数据盘，而在仓库内 `datasets/Hi4D_pilot_v1/`（`raw/` 为按需解包的单视角序列，`final_assets_v4_2/` 为指向这些原始文件的符号链接资产农场）。
- iPER、3DPW、CHAMP 等其他数据集也不在这里：iPER 与 3DPW 定稿数据位于仓库 `datasets/` 实体目录，`add/outputs`、`add/data` 保存接入期快照。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset"] = r"""
## TikTok 单人舞蹈姿态编辑数据集（项目内代号 TikTok Dataset）

**来源与许可**：公开数据集 TikTok Dataset（Kaggle/HF 路径 `datasets/yasaminjafarian/tiktokdataset`，见 `.complete/datasets/yasaminjafarian/tiktokdataset/` 完成标记目录），随附 `License.pdf`。原始形态是「舞蹈教学视频 → 逐序列抽帧 + 人像 mask + DensePose」，本项目在此基础上自建 SMPL-X 几何条件层与多版本配对清单。

**总体规模**：340 个序列目录（`00001`–`00340`），合计 **92,961 帧**（`TikTok_v3/stats.json` 与 `TikTok_v2`/`TikTok_v3` 清单一致）；分辨率保持原生（典型 604×1080，逐序列不同），全程未做几何裁剪。

**分层结构（自下而上）**：

| 层 | 目录 | 内容 | 体积 |
|---|---|---|---|
| L0 原始视频 | `TikTok_Raw_Videos/` | 26 个 `seq_<起>_<止>/` 分组，每组 `YouTube.mp4` + `dance_name.txt` + `video_link.txt` | 2.60 GB |
| L1 官方释放 | `TikTok_dataset/TikTok_dataset/` | 每序列 `images/`（RGB 帧）、`masks/`（人像二值 mask）、`densepose/`（官方 DensePose 渲染） | 50.00 GB |
| L2 三维资产 | `TikTok_3d_assets_native/` | 每序列 SMPLest-X 拟合参数 + 原生分辨率法线/深度/网格 mask/部位分割 + V6 条件索引 | 40.34 GB |
| L3a 构建版本 v2 | `TikTok_v2/` | 全帧 QC 打分表与候选配对（未冻结） | 263 MB |
| L3b 构建版本 v3 | `TikTok_v3/` | DWPose 结果、冻结候选池、训练/验证/测试配对、八类质检、人脸特征缓存 | 745 MB |
| L4 划分 | `splits/` | 以序列为单位的 train/val/test 清单与摘要 | 24 KB |
| 辅助 | `model/`、`showcase_single/`、`logs/` | 官方 TF 深度/法线预测权重、展示样本与过拟合运行、构建日志 | 0.7 GB / 16.26 GB / 3.9 MB |

**版本谱系与取舍**：v2 是「全量候选 + 未冻结」的探索版（`pair_status=CANDIDATE_NOT_FROZEN`，90,942 对候选）；v3 是训练实际使用的定稿版，改为「先按序列做候选池冻结，再用 DWPose 几何有效性筛选配对」，并新增严格版配对 `pairs_strict_v1`。**训练与评测一律读取 v3 清单 + L1 图像 + L2 几何 + v3 DWPose**，v2 仅作为质量审计与回溯证据保留。

**消费方（仓库脚本）**：数据加载 `src/data/tiktok_dataset.py`；构建链 `scripts/data/build_tiktok_v3_{candidate_pool,condition_inventory,pairs,pairs_strict,strict_visual_pool,usable_pool}.py`、`run_tiktok_v3_dwpose.py`、`fill_tiktok_v3_missing_geometry.py`、`build_tiktok_v6_conditions.py`、`finalize_tiktok_v3_visual_quality_v1.py`；人脸特征 `scripts/precompute_tiktok_face_features_v65.py`、`prepare_tiktok_qwen21_face.py`；训练/评测入口 `scripts/prepare_tiktok_qwen21_training.py`、`scripts/audit/audit_tiktok_v3_*.py`。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_Raw_Videos"] = r"""
**L0 原始视频层**。`TikTok_Raw_Videos/TikTok_Raw_Videos/`（同名双层嵌套是官方释放包的原始结构）下 26 个分组目录，目录名 `seq_<起始序列号>_<结束序列号>` 表明该视频对应 L1 中哪一段序列区间，例如 `seq_00001_00009/` 覆盖 `00001`–`00009`。

每个分组三件文件：`YouTube.mp4`（原始舞蹈教学视频，54–133 MB）、`dance_name.txt`（舞蹈名，官方包中普遍为空）、`video_link.txt`（原视频链接，官方包中普遍为空）。

用途：仅在需要重新抽帧、核对运动模糊或复核 SMPLest-X 拟合失败（帧数不匹配）时回溯；训练不直接读取本层。总计 96 个文件、2.60 GB。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_dataset"] = r"""
**L1 官方释放层**：`TikTok_dataset/TikTok_dataset/<序列号>/{images,masks,densepose}`，340 个序列目录，每个序列固定含三份子目录，帧数完全一致（如 `00001`=127 帧、`00002`=223 帧、`00292`=267 帧），总帧数 92,961。

- `images/`：RGB 帧 PNG，原生分辨率（例 `00001` 为 604×1080），帧名从 `0001.png` 起连续编号，无跳号。
- `masks/`：官方提供的人像二值 mask（同尺寸，单通道，约 5 KB/帧），是候选池几何/遮挡判定（`touch_*`、`mask_area_ratio`）的来源。
- `densepose/`：官方 DensePose UV 渲染 RGB 图，本项目仅用作身体部位语义的参考与展示，几何条件优先使用 L2 的 SMPL-X 渲染结果。

280,224 个文件、50.00 GB。本层是不可再生的官方数据，任何重建脚本都只写 `TikTok_3d_assets_native/`、`TikTok_v2/`、`TikTok_v3/`，不回写这里。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_3d_assets_native"] = r"""
**L2 SMPL-X 三维几何资产层（原生分辨率）**，由 `smplest_x_h` 拟合 + 与 iPER 主流程同一套 NVIDIA nvdiffrast 渲染器生成，是本项目自产、可重生成但耗时极长的核心资产。340 个序列目录，252,446 个文件、40.34 GB。

**单个序列目录的标准布局**（以 `00292`，267 帧为例）：

```
<序列号>/
├── smplestx/params.pt        # SMPLest-X 逐帧拟合参数（见下表）
├── smplestx/frame_map.json   # 帧名 → params.pt 行索引映射
├── normal/0001.png…          # 法线图，RGB，原生分辨率，全帧
├── depth_vis/0001.png…       # 相机空间 Z 深度可视化，单通道，全帧（原始 float 深度不落盘）
├── smplx_mask/0001.png…      # SMPL-X 网格投影二值 mask，单通道，全帧
├── part_512/0004.png…        # 512×512 部位分割标签图（像素值即部位编号，取值 0–14）
├── v6_conditions.pt          # V6/V6.5 加载器需要的全局条件（smplx_global、kps_25_coords、stems、stem_to_idx、width、height）
└── _ASSETS_DONE.json         # 该序列的构建验收记录（status/frame_count/分辨率/渲染器/相机约定）
```

**`smplestx/params.pt` 字段**（首维 F = 该序列帧数）：

| 键 | 形状 | 含义 |
|---|---|---|
| `global_orient` | (F, 3) | 全局朝向轴角 |
| `body_pose` | (F, 63) | 21 个身体关节轴角 |
| `left_hand_pose` / `right_hand_pose` | (F, 45) | 每手 15 关节轴角 |
| `jaw_pose` | (F, 3) | 下颌轴角 |
| `betas` | (F, 10) | 形状系数（身份/体型） |
| `expression` | (F, 10) | 表情系数 |
| `transl` | (F, 3) | 平移 |
| `mesh_cam` | (F, 10475, 3) | SMPL-X 10475 顶点相机坐标系坐标，法线/深度/mask 渲染的几何来源 |
| `joint_proj_model` | (F, 137, 2) | 137 个 SMPL-X 关节的模型坐标系投影 |
| `focal_length` / `focal_length_xy` / `princpt` / `width` / `height` | (F,1)/(F,2) | 官方逐帧相机参数；渲染优先用 `focal_length_xy_raw + princpt_raw`，缺失时回退 |

**`v6_conditions.pt` 字段**：`smplx_global` (F, 26)、`kps_25_coords` (F, 25, 3)（25 点 COCO+脚部关键点的像素坐标与可见性）、`stems`（帧名列表）、`stem_to_idx`（帧名 → 行索引）、`width`/`height`（原生宽高）。由 `scripts/data/build_tiktok_v6_conditions.py` 生成，供 `src/data/tiktok_dataset.py` 直接索引。

**构建验收统计**：296 个序列有 `_ASSETS_DONE.json` 且 `status=PASS`（具备 `normal/depth_vis/smplx_mask` 全帧）；295 个序列同时具备 `part_512/` 与 `v6_conditions.pt`；44 个序列带 `_SMPLestX_FAILED.json` + `_FAILED.json`，失败原因均为拟合帧数与图像帧数不一致（例 `"mesh_cam rows=268, images=271"`），这些序列不进入训练清单。`part_512` 合计 7,598 张，仅覆盖被采样到的帧（与 `TikTok_v3/dwpose_v1` 的帧集合一致）。

**渲染约定（摘自 `_ASSETS_DONE.json`）**：不做几何裁剪（方形草稿画布渲染后裁回原始 H×W）；法线基为 +X 右、+Y 上、+Z 朝向相机；深度为相机空间 Z 光栅化，仅保存可视化 PNG；官方 `images/masks/densepose` 保持原样不改写。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v2"] = r"""
**L3a 第二代构建（探索版，未冻结）**：`manifests/` + `qc/`（300 张抽查 JPEG），312 个文件、263 MB。

- `manifests/frames_all_qc.csv`（85 MB）：全帧逐帧 QC 打分明细。
- `manifests/sources.jsonl`（1,771 条）/ `targets.jsonl`（15,157 条）/ `selected_unique.jsonl`（16,928 条）：逐帧档案，字段覆盖解码、mask 面积/包围盒、四边触碰、清晰度（`lap_var`/`tenengrad`/`sharpness`）、`smplx_iou`、各条件存在性、`completeness`、`quality_score`、`reject_reasons` 与 `role`。
- `manifests/pairs_candidate.jsonl`（90,942 条）：源帧→目标帧候选对，`pair_status` 恒为 `CANDIDATE_NOT_FROZEN`，即该版本只完成打分与候选生成，未产出可训练清单。

**为什么被 v3 取代**：v2 以全帧为候选单位、配对数量过大且未控制姿态冗余；v3 改为先按序列冻结候选池、再做几何有效性与视觉质量双重筛选。v2 目录保留作为阈值与打分口径的审计证据。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3"] = r"""
**L3b 第三代构建（训练实际使用版）**：四个子模块 + `stats.json`，16,255 个文件、745 MB。

| 子目录 | 内容 | 规模 |
|---|---|---|
| `dwpose_v1/` | 每序列 `dwpose/`（骨架可视化 PNG）与 `dwpose_json/`（134 关键点检测记录），仅覆盖被采样的帧 | 295 序列、7,481 帧 |
| `manifests/` | 帧库存、候选池、条件库存、DWPose 汇总、冻结池、配对清单（含严格版） | 28 个文件、311 MB |
| `qc/` | 八类人工/自动质检（DWPose 抽查、模糊审计、几何拒绝样本、姿态冗余、严格终审、视觉质量 v1/v2/v3） | 约 25 MB |
| `qwen21_face_feature_cache/` | 每序列一个人脸特征缓存 `.pt` + 索引 JSON | 280 个 `.pt`、60 MB |

`stats.json` 记录版本标识 `tiktok_v3_frame_inventory`、340 序列 / 92,961 帧，并给出主清单的绝对路径。

**训练读取顺序**：`manifests/pairs(_strict_v1)/<split>_pairs.jsonl` 给出 pair → 按 `sequence_id` + `source_stem`/`target_stem` 从 L1 取 RGB/mask、从 `dwpose_v1/` 取目标骨架、从 `TikTok_3d_assets_native/` 取法线/深度/网格 mask/`v6_conditions.pt`。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3/dwpose_v1"] = r"""
**DWPose 检测结果层（v1）**：`<序列号>/{dwpose, dwpose_json}/<帧名>`，共 295 个序列、7,481 帧。注意它不是全帧覆盖，而只覆盖 v3 候选池实际采样到的帧（与 `part_512` 的帧集合一致）。

- `dwpose/NNNN.png`：原生分辨率骨架连线可视化（RGB），可直接作为姿态条件图使用，也是配对清单里的 `target_dwpose` 字段指向的文件。
- `dwpose_json/NNNN.json`：结构化检测结果，字段 `sequence_id、stem、rgb、status、reason、dwpose_png、dwpose_json、body_mean_conf、body_valid_03、hand_mean_conf、face_mean_conf、detected_persons、image_width、image_height、selected_person_index、keypoints_norm、scores`。`keypoints_norm` 为 134 点归一化坐标（Body 18 + Foot 6 + Face 68 + 双手各 21），`scores` 为逐点置信度；多人场景通过 `selected_person_index` 指定目标人物。

生成脚本：`scripts/data/run_tiktok_v3_dwpose.py`（模型为 `models/DWPose` 的 ONNX 权重）；抽查面板：`qc/dwpose_sample/`。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3/manifests"] = r"""
**v3 清单中枢**：28 个文件（16 个 JSONL + 10 个 JSON + 1 个 TXT），311 MB。逐文件语义：

| 文件 | 记录数 | 作用 |
|---|---:|---|
| `frame_inventory.jsonl` | 92,961 | 全帧库存：`sequence_id/frame_name/rgb/width/height/quality.blur/assets{normal,depth,smplx_mask,pose}` |
| `candidate_pool.jsonl` | 8,644 | 采样后的候选帧池，含 mask/包围盒/四边触碰/清晰度/明暗比、`motion`、`sampling_motion_class`(SLOW/MID/FAST)、`sampling_gap`、`local_radius`、`stage` 与 `pose_checked` |
| `candidate_pool_geometry_valid.jsonl` / `candidate_pool_geometry_reject.jsonl` | — | 几何有效性筛选后的通过/拒绝子池 |
| `candidate_pool_summary.json` | — | 池子统计与阈值 |
| `condition_inventory.jsonl` | 8,644 | 条件齐备性：`paths`、`available{rgb,person_mask,densepose,smplestx,normal,depth_vis,smplx_mask…}`、`core_existing_complete`、`needs{dwpose,raw_depth}`、`smplestx_index` |
| `dwpose_results.jsonl` | 7,481 | DWPose 逐帧结果汇总（与 `dwpose_v1/*/dwpose_json` 同字段） |
| `dwpose_summary.json` | — | DWPose 通过率与置信度分布 |
| `dynamic_sampling_candidates.jsonl` | — | 按运动幅度动态选帧结果（`scripts/data/build_tiktok_v3_candidate_pool.py`） |
| `fill_missing_geometry_sequences.txt` | — | 需要补几何的序列列表，交给 `fill_tiktok_v3_missing_geometry.py` |
| `frozen_candidate_pool_v2/` | — | 冻结版候选池（`candidate_pool.jsonl` 9.9 MB + `sequence_screening.jsonl` + `dynamic_sampling_candidates.jsonl` + summary） |
| `pairs/` | — | v3 标准配对：`{train,val,test}_pairs.jsonl`、`{train,val,test}_source_pool.jsonl`、`{train,val,test}_unpaired_targets.jsonl`（train 11,042 对） |
| `pairs_strict_v1/` | — | 严格配对（同一字段结构，追加姿态冗余/模糊约束，train 8,574 对），人脸与手部联合实验默认使用这一版 |

**配对记录字段**（`pairs*/<split>_pairs.jsonl`）：`pair_id、split、sequence_id、source_video_group、source_stem、target_stem、source_rgb、target_rgb、source_person_mask、target_person_mask、target_dwpose、target_normal、target_depth_vis、target_smplx_mask、source_dwpose_json、target_dwpose_json、pose_distance、source_quality、source_choice_rank、target_pose_soft_risk`。路径全部写成仓库内绝对路径（`/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/...`），经符号链接落到本数据盘，因此换机器时需要同步改仓库链接位置而不是改清单。

`pose_distance` 为源/目标 25 点归一化坐标距离，配对时要求足够大以形成真实姿态迁移；`source_choice_rank` 记录同一目标可选源帧的优先级。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3/qc"] = r"""
**v3 质检留档区**，八类，全部是「证据 + 可复核」形式（JSONL/CSV/JSON 明细 + 联系表图片）：

| 子目录 | 内容 |
|---|---|
| `dwpose_sample/` | DWPose 骨架 24 格抽查联系表 |
| `final_blur_audit/` | 最终训练集模糊审计：`blur_scores.csv` + 最模糊 24 格 + 随机 24 格 |
| `geometry_reject_sample/` | 几何判定被拒绝的样本图 13 张 + 判定 JSON（对应 `candidate_pool_geometry_reject`） |
| `pose_redundancy_v1/` | 逐帧姿态冗余打分 `pose_redundancy_per_frame.jsonl`、`pose_redundancy_summary.json` 与非局部姿态 Top-32 展示 2 张 |
| `strict_final_audit/` | 严格配对终审 24 格抽查 |
| `visual_quality_final_v1/` | 终审决策：`visual_quality_decisions.jsonl`、`global_hard_blur.jsonl`、`source_exclude_face.jsonl`、`source_hand_risk.jsonl` + `global_hard_review/` 复核图 4 张 |
| `visual_quality_v2/` | 第二版视觉质量：`visual_quality_scores_v2.csv`、`summary_v2.json`、源帧模糊风险候选 32 与展示 2 张 |
| `visual_quality_v3/` | 第三版：区分 `face_blur` / `global_blur` / `hand_blur` 三类风险候选各 32 与展示图，附 `visual_quality_scores_v3.csv` |

对应脚本：`scripts/data/finalize_tiktok_v3_visual_quality_v1.py`、`scripts/audit/audit_tiktok_v3_{visual_quality_v2,visual_quality_v3,pose_redundancy,blur_sample,final_pairs,strict_final}.py`、`scripts/data/sample_tiktok_v3_geometry_rejects.py`。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3/qwen21_face_feature_cache"] = r"""
**人脸特征离线缓存（Qwen 2.1 Full 人脸 ROI 控制器专用）**：每序列一个 `<序列号>.pt`（280 个，共 60 MB）+ `index.json`，由 `scripts/precompute_tiktok_face_features_v65.py` / `scripts/prepare_tiktok_qwen21_face.py` 生成。

单个 `.pt` 的字典结构：

| 键 | 含义 |
|---|---|
| `schema` | 缓存格式版本（当前 1） |
| `dataset` | 数据集标识，固定 `tiktok_v3` |
| `appearance` | 外观（源人物）标识，等于序列号 |
| `original_resolution` | 原生分辨率，如 `[604, 1080]` |
| `source` | 子字典，键为源帧名（如 `0004/0080/0094`），值为该帧的人脸特征（ArcFace 身份向量 + DINO patch 特征） |
| `target_geometry` | 子字典，覆盖该序列全部可用帧名，存目标人脸几何特征 |
| `target_visibility` | 子字典，仅含采样帧，存人脸可见性/遮挡判定 |
| `thresholds` | 子字典 `{source_side, target_side, score}`，缓存生成时使用的判定阈值，用于校验缓存与配置是否匹配 |

该缓存缺失会直接导致人脸 ROI 分支退化为无身份引导，训练脚本会显式检查其存在性。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/splits"] = r"""
**序列级划分（早期口径，219 序列）**：`split_summary.json` + `train_sequences.txt`(179) + `val_sequences.txt`(20) + `test_sequences.txt`(20)。

`split_summary.json` 关键字段：`seed=2026`、`split_unit="sequence"`（同一序列的帧不会同时出现在训练与测试中）、`total_sequences=219`，各 split 帧数为 train 49,202 / val 5,357 / test 5,887；`identity_policy` 明确写着「序列互斥，但身份重叠需要在最终论文评测前另行审计」——因为 TikTok 同一位舞者可能出现在多个序列里。

**与 v3 清单的关系**：本目录是 TikTok 接入早期给出的划分基准；当前训练实际使用的 train/val/test 配对由 `TikTok_v3/manifests/pairs*` 直接给出，序列集合更大（340 序列）。两者并存时以 v3 配对清单为准，本目录用于跨版本对齐与复现早期实验。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/model"] = r"""
**官方随附的 TensorFlow 预测模型**（非本项目产物，来自 TikTok Dataset 释放包）：双层嵌套 `model/model/`，其下两个分支各保存一个 TF 检查点。

| 分支 | 检查点 | 组成文件 | 体积 |
|---|---|---|---|
| `depth_prediction/` | `model_1920000` | `checkpoint`、`model_1920000.ckpt.data-00000-of-00001`(340 MB)、`.index`、`.meta` | 355 MB |
| `normal_prediction/` | `model_1710000` | 同上四件套（`.meta` 801 KB） | 341 MB |

这是官方用来从 RGB 预测深度/法线的模型（训练步数分别 1.92M / 1.71M）。本项目**不使用**它：深度与法线一律由 SMPL-X 网格经 nvdiffrast 渲染得到（见 `TikTok_3d_assets_native/`），因此该目录只作为数据集完整性保留，合计 14 个文件、696 MB。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/showcase_single"] = r"""
**单人姿态编辑展示样本区（含配套过拟合运行产物）**：5 个 `sample_01_<序列>_<源帧>_to_<目标帧>` 目录，共 165 个文件、16.26 GB（体积主要来自检查点）。

每个样本目录的固定七图 + 面板：`01_source_rgb.png`（源图）、`02_target_rgb.png`（目标真值）、`03_person_mask.png`、`04_densepose.png`、`05_depth.png`、`06_normal.png`、`07_smplx_mask.png` 与拼合后的 `SHOWCASE_PANEL.jpg`。

`sample_01_00001_0014_to_0074/` 额外保存该样本的三档训练记录：

- `overfit_full_200/`、`overfit_full_1500/`、`overfit_v65/`：各含 `train.log`、`eval.log`、`metrics.json`；
- `overfit_full_1500/` 内另有 `checkpoints/`（`checkpoint_step_{000500,001000,001500}.pt`，单个 1.96 GB）、`condition_cache/00001_0014.pt`（该源帧的条件张量，3.6 MB）、`previews/`（32 张阶段预览）、`final_eval/`（9 张结论图：损失曲线、全阶段整体/人脸 ROI/手部 ROI 演化、真实推理全身图与人脸/手部 ROI、量化汇总，另附输入图与 `condition_cache/`）。

用途：向评审展示「同一源图 + 不同目标姿态」的编辑可控性以及人脸/手部 ROI 分支随训练步数的演化。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/logs"] = r"""
**构建作业日志留存**：`build_tiktok_3d_assets_native_full.out`（约 3.9 MB 标准输出，记录 340 个序列的原生分辨率三维资产全量重建过程，含每个序列的 PASS/FAIL 判定与帧数不一致告警）和配套的 Markdown 记录。与仓库 `logs/legacy/` 中的抽帧质检、DWPose 失败诊断日志互补：这里只覆盖 L2 资产重建。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/TikTokDataset/.complete"] = r"""
下载完成标记目录（隐藏，不生成说明）：`datasets/yasaminjafarian/tiktokdataset/<n>` 用于记录官方包的获取批次，供重跑数据管道时判断是否已完整落盘。
"""

NOTES["/data/shangguanrz/pic-edit-datasets/Hi4D-origin"] = r"""
**Hi4D 官方原始数据包**（ETH Zürich Hi4D，多视角双人交互数据集，本项目用于握手/搭肩/拥抱等接触交互训练）。

**存放形态**：整包 tar.gz，未在数据盘解压。32 个分卷共 **384.63 GB**：
`pair00_1/00_2、pair01、pair02_1/02_2、pair09、pair10、pair12、pair13_1/13_2、pair14、pair15_1/15_2、pair16、pair17_1/17_2、pair18_1/18_2、pair19_1/19_2、pair21_1/21_2、pair22、pair23_1/23_2、pair27_1/27_2、pair28、pair32_1/32_2、pair37_1/37_2`（单个 6.5–16.1 GB；同一 pair 被拆成 `_1/_2` 两卷是因为官方按体积切分）。
配套 `hi4d_urls.txt`（32 条官方下载直链，形如 `https://hi4d.ait.ethz.ch/download.php?dt=<token>&file=/pairXX.tar.gz`，token 有时效）与 `hi4d_files.txt`（清单）。

**压缩包内部结构**（以 `pair00_1.tar.gz` 为例，实测成员）：

```
<action><序号>/            # 例如 dance00/、fight00/、highfive00/，同一 pair 含多段动作序列
├── cameras/rgb_cameras.npz   # ids(8 视角)、intrinsics、dist_coeffs
├── frames/<视角名>-F<帧号>.png   # 各视角 PNG 帧，视角名如 Atlas、MatLib
├── frames_vis/                 # 官方可视化帧
├── images/<相机id>/<帧号>.jpg    # 主序列 JPG 帧，目录名为相机编号（如 88、4）
├── seg/img_seg_mask/<相机id>/…   # 2D 人像分割 mask
│   mesh_seg_mask/mesh-f<帧号>.npz # 网格级部位分割
├── smpl/<帧号>.npz               # 逐帧人体参数
└── meta.npz                      # 序列元信息
```

**逐帧 `smpl/NNNNNN.npz` 字段**（首维 2 = 两个人）：`betas (2,10)`、`global_orient (2,3)`、`body_pose (2,69)`、`transl (2,3)`、`verts (2,6890,3)`、`joints_3d (2,24,3)`、`contact (2,6890)`——`contact` 是逐顶点的接触标注，是本数据集对双人交互最关键、其他数据集没有的监督信号。`meta.npz` 提供 `betas、genders、num_persons、start、end、contact_ids`（接触区间帧号）、`mono_cam`（训练使用的单视角相机编号）。

**流向**：仓库 `datasets/Hi4D_pilot_v1/raw/` 按需只解包训练用到的 pair 与 `mono_cam` 视角，`final_assets_v4_2/` 再以符号链接组织成资产农场，`pic-edit-caches/qwen21_hi4d_controls/` 保存由此派生的控制条件张量缓存。本目录保持官方原样，任何处理都不回写这里。
"""

NOTES["/data/shangguanrz/pic-edit-models"] = r"""
大体积基座权重实体区，仅 Qwen-Image 系列两套（52 个文件、51.97 GB）：`Qwen-Image-2.1/` 是 Diffusers 组件式目录（当前训练主线），`Qwen-Image-2.1-PE-I2I/` 是扁平单模型目录（报告流水线用）。DeepGen-1.0-diffusers、DWPose、antelopev2、dinov2、SMPL 等权重仍留在仓库 `models/` 实体目录，未迁移到数据盘。
"""

NOTES["/data/shangguanrz/pic-edit-models/Qwen-Image-2.1"] = r"""
**Qwen-Image-2.1 基座（DiT 生成主干，本项目外挂控制网络的挂载对象）**，Diffusers 组件式布局，`model_index.json` 声明：

```
_class_name = QwenImage21Pipeline   (_diffusers_version 0.37.0.dev0)
processor      = transformers.Qwen3VLProcessor
text_encoder   = transformers.Qwen3VLForConditionalGeneration
transformer    = diffusers.QwenImage21Transformer2DModel
vae            = diffusers.AutoencoderKLQwenImage21
scheduler      = diffusers.FlowMatchEulerDiscreteScheduler
```

| 组件目录 | 体积 | 主要文件 |
|---|---:|---|
| `text_encoder/` | 17.53 GB | `model-{00001..00004}-of-00004.safetensors`（5.00/4.92/4.92/2.70 GB）+ `config.json`、`generation_config.json`、`model.safetensors.index.json` |
| `transformer/` | 14.23 GB | `diffusion_pytorch_model-{00001,00002}-of-00002.safetensors`（9.97/4.26 GB）+ `config.json` |
| `vae/` | 1.3 GB | `diffusion_pytorch_model.safetensors` + `config.json` |
| `processor/` | 15.9 MB | `tokenizer.json`(11.4 MB)、`vocab.json`、`merges.txt`、`added_tokens.json`、`special_tokens_map.json`、`tokenizer_config.json`、`chat_template.jinja`、`preprocessor_config.json`、`video_preprocessor_config.json` |
| `scheduler/` | 2 KB | `scheduler_config.json`（Flow Matching） |
| `assets/` | 3.3 MB | 官方示意二维码 `qr.png` |

**与训练的关系**：DiT 主干冻结，外挂姿态控制器以 32 层 Hook 残差注入，并把适配器 1536 维输出投影到 DiT 的 4096 维；`--model` 默认指向仓库 `models/Qwen-Image-2.1`（即本目录的符号链接入口）。
"""

NOTES["/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I"] = r"""
**PE-I2I 报告流水线使用的 Qwen-Image-2.1 权重副本**，与主线训练分开存放以免互相影响。扁平布局（无组件子目录）：16 个文件、18.84 GB —— `model-0000{1..4}.safetensors`（5.07/5.09/5.09/3.57 GB）、`config.json`、`generation_config.json`、`chat_template.jinja`、tokenizer 三件套与 `LICENSE`、`README.md`。
"""

NOTES["/data/shangguanrz/pic-edit-caches"] = r"""
预计算控制条件缓存实体区（1,389 个 `.pt`，逻辑体积 66.28 GB，实际只占盘约 10.2 GB）。与数据集实体的区别在于：这里的内容**可以重生成**，代价只是 GPU 时间，因此允许在空间紧张时删除重建，也允许与仓库分离存放。当前只有 Hi4D 双人控制条件缓存一项。
"""

NOTES["/data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls"] = r"""
**Qwen 2.1 在 Hi4D 上的双人控制条件张量缓存**：`control_<24位十六进制>.pt`，1,389 个，每个 47.7 MB，合计 66.28 GB（即每个缓存对应一对源/目标图像的全部几何条件）。

**文件名哈希怎么来的**（`src/pose_control/qwen21_full/qwen_runtime.py`）：
`sha256( MODEL_REVISION + prompt + kind + str(resolution) + RGB 像素字节 )`，取前 24 位；`kind` 为 `control`/`latent` 等类别，`resolution ∈ {512, 1024, native}`。换提示词、换分辨率、换模型修订号或预处理策略都会自然换键；当目录内已有 `cache_policy.json` 且策略不一致时，运行时会直接报错要求换新目录，避免读到语义不匹配的缓存。

**单个 `.pt` 的字段与形状**（实测，fp16）：

| 键 | 形状 | 含义 |
|---|---|---|
| `normal_a` / `normal_b` | (3, 512, 512) | 人物 A / B 的 SMPL-X 法线图 |
| `human_mask_a` / `human_mask_b` | (1, 512, 512) | 每个人物的人像 mask |
| `part_onehot_a` / `part_onehot_b` | (14, 512, 512) | 14 部位分割 one-hot |
| `pose_heatmap_a` / `pose_heatmap_b` | (25, 512, 512) | 25 关键点姿态热图 |
| `depth_a` / `depth_b` | (1, 512, 512) | 每人物深度图 |
| `smplx_global_a` / `smplx_global_b` | (26,) | 每人物全局 SMPL-X 条件向量 |
| `relative_geometry` | (11,) | A–B 相对几何（距离/朝向等）摘要 |
| `contact_raster` | (2, 512, 512) | 接触区域光栅图 |
| `human_mask` | (1, 512, 512) | 两人合并人像 mask |
| `task_id` | 标量 int64 | 交互任务编号（搭肩/握手/拥抱等） |
| `contact_src_person` / `contact_src_part` / `contact_dst_person` / `contact_dst_part` / `contact_contact_type` | (8,) int64 | 8 组接触三元组的主体、部位与接触类型 |
| `contact_distance` | (8, 1) | 对应接触距离 |
| `contact_valid_mask` | (8,) bool | 该槽位是否有效 |
| `person_b_valid` | 标量 bool | 是否为双人样本（单人样本 B 路为零填充） |

**消费方**：`train_hi4d_qwen21_dual.py`、`eval_hi4d_qwen21_dual.py`（`--control-cache` 默认仍写兼容路径 `/home/shangguanrz/nihaochen/qwen21_hi4d_controls`，实体已迁到本目录，仓库 `caches/qwen21_hi4d_controls` 为链接入口）。配对来源清单是 `datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_{train,val,test}.jsonl`。
"""

NOTES["/data/shangguanrz/pic-edit-archive"] = r"""
冷归档区（3,575 个文件、102.86 GB），含两项：`cleanup_20261003/`（2026-10-03 目录整理时移出仓库工作区的旧实验、旧模型与原始数据）与 `SynBody_inspection_20261003/`（SynBody 合成数据视觉质量审计打包）。归档只读：不再作为训练输入，仅供复核与再取证。
"""

NOTES["/data/shangguanrz/pic-edit-archive/cleanup_20261003"] = r"""
**2026-10-03 清理批次移出内容**（3,572 个文件、102.84 GB）。仓库内 `experiments/`、`models/`、`archive/` 的对应原路径保留符号链接，旧命令依旧可用。

| 子目录 | 文件数 | 体积 | 内容 |
|---|---:|---:|---|
| `experiments/` | 3,519 | 40.08 GB | 两个定案实验的完整产物 |
| `models/` | 47 | 57.72 GB | Qwen-Image-Edit-20B 权重（下载不完整即被归档） |
| `raw/` | 5 | 5.04 GB | 3DPW 原始压缩包 |

迁移与验收证据：仓库 `archive/cleanup_20261003/tooling/{migrate_assets.py,finalize_migration.py}`、`docs/目录清理执行记录_20261003.md`，以及本批次自带的 `assets.json` 实体切换登记（`asset_registry.json` 明确以它为准）。
"""

NOTES["/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments"] = r"""
两个已完成阶段性使命的实验产物：

1. **`iper_v6_comparative_3000/`（12.77 GB，161 文件）**：DeepGen V6 在 iPER 上 3000 步横向对照。
   - 权重：`adapter_step_{500,1000,1500,2000,2500,3000}.pt` + `checkpoint_best.pt` + `checkpoint_last.pt`，单个 1.59 GB（8 个共 12.7 GB，是本目录体积主体）；
   - 日志：`eval.log`、`launcher.log`、`train.log` 等 4 份；
   - `eval_results/eval_metrics.json`（33 KB，逐步数量化指标）；
   - `multistage_evolution_showcase/`（73 个文件）：每个样本的 `*_source.png`、`*_cond_{depth,dwpose,normal,part}.png`、`*_step_<n>.png`、`*_evolution_strip.png`；
   - `stage_evolution/`（21 个文件）：`multi_stage_evolution_grid.png` 与 `pair*_step_<n>.png` 对照。
2. **`review_20260908_v2/`（27.31 GB，3,357 文件）**：2026-09-08 正式验收评审工作区。
   - 结论文档：`EXPERIMENT_CONCLUSION.md`、`MEMORY_REPORT.md`、`QUALITY_REVIEW.md`、`RUNBOOK.md`、`ACCEPTANCE.json`；
   - `runs/`（2,935 文件、27.02 GB）：120 个运行记录的逐条产物与显存测量；
   - `logs/`（241）、`configs/`（121）、`cache/`（19）、`review_sheets/`（13、12.6 MB 评审联系表）。

这两项对应仓库入口 `experiments/iper_v6_comparative_3000`、`experiments/review_20260908_v2`（`asset_registry.json` 的 `cold_runs`）。
"""

NOTES["/data/shangguanrz/pic-edit-archive/cleanup_20261003/models"] = r"""
**Qwen-Image-Edit-20B（20B 教师/质量上界模型）归档**，47 个文件、57.72 GB。目录仍是 Diffusers 组件式布局：`text_encoder/`（16.58 GB，4 分片）、`transformer/`、`vae/`、`processor/`（15.9 MB）、`tokenizer/`（5.1 MB）、`scheduler/`，加 `model_index.json`、`README.md`。

**重要状态**：本副本下载并未完成即被归档——目录里保留着 `_download.py`、`_start.sh`、`download.log`（15 KB）、`download.pid`，可作为「当时中断」的直接证据。项目对 20B 只做了显存可行性分析（结论：单卡 84 GB 不足以稳定训练，仅作离线对照采样），因此不影响主线；如需使用请重新完整下载，仓库兼容链接为 `model-qwen-edit` / `models/Qwen-Image-Edit-20B`。
"""

NOTES["/data/shangguanrz/pic-edit-archive/cleanup_20261003/raw"] = r"""
**3DPW 原始数据归档**（`raw/3dpw/`）：5 个文件、5.04 GB，主体是 3 个 `.zip` 官方压缩包。3DPW 用于单人在动态序列上的几何/深度条件补充实验；处理后的定稿数据在仓库 `datasets/3dpw_single_final_v2/`，接入期快照在仓库 `add/data/3DPW` 与 `add/outputs`。本目录只保存官方原包，不解压、不修改。
"""

NOTES["/data/shangguanrz/pic-edit-archive/SynBody_inspection_20261003"] = r"""
**SynBody 合成人体数据视觉质量审计留档**（2 个文件、24.1 MB）。实体是单个 `inspection_archive.tgz`，解开后包含：`SynBody_report.md`（审计报告）、`annotation_audit.log`（标注核查日志）、`visual_quality_20261003/`（含 `sampling_manifest.json` 采样清单、`S100K_0*.jpg`、`HumanNeRF_0*.jpg`、`SMPLX_0*.jpg` 分来源样图与 `S100K_contact_sheet.png` 联系表）。

结论用途：SynBody 合成人体在纹理真实感与边缘光影上存在系统性缺陷，未纳入正式训练集，因此本目录以归档形式保留判定证据。注意仓库 `MANIFEST`（`dataset_samples/MANIFEST.txt`）中登记的 `SynBody/...` 样本切片与本归档同属一次调研，样本区那份已不在当前目录树中。
"""

NOTES["/data/shangguanrz/dataset_samples"] = r"""
**公开人体数据集选型样片区**（489 个文件、7.46 GB）。2026-09-29 起为第三阶段数据选型收集，只保留可肉眼判断质量的极小样本，不含完整数据集。

| 子目录 | 体积 | 对象 |
|---|---:|---|
| `abc_showcase_20261003/` | 7.45 GB | 多源数据集横向展示（DNA-Rendering、FreeMan、HumanVid、MVHumanNet、PKU-DyMVHumans） |
| `MVHumanNet++_repo/` | 8.4 MB | MVHumanNet++ 仓库内 SMPL-X 投影可视化脚本与样本 |
| `MVHumanNet_repo/` | 3.1 MB | MVHumanNet 仓库内 `visual_smpl_data` 与 `textual_description` 样例 |

`MANIFEST.txt` 是完整性核对清单：首行注明生成时间 `2026-09-29T22:48:52+08:00`、主机 `node01`、基准目录本路径，列为「sha256 前 16 位 TAB 字节数 TAB 相对路径」，逐条覆盖所有样片（含当时还未迁出的 `SynBody/`、`HumanVid`、`PKU-DyMVHumans` 等条目）。核对方式：`sha256sum` 取前 16 位与第一列比对。清单中已不存在于本目录的条目，说明其样本后来被迁到仓库 `research/` 或归档区，属于登记滞后，不影响其余条目校验。
"""

NOTES["/data/shangguanrz/dataset_samples/MVHumanNet_repo"] = r"""
**MVHumanNet 官方仓库样例节选**：
- `visual_smpl_data/`：单身份样本三件套 —— `000075.pkl`（SMPL/SMPL-X 参数，252 KB）、`000075.json`（元信息）、`000075_smplx.obj`（导出网格 740 KB），配 `0380_img.jpg`（参考图 1.0 MB）与 `camera_extrinsics.json` / `camera_intrinsics.json` / `camera_scale.pkl`（相机与尺度）。
- `textual_description/`：`sample_100_description.json`（23 KB 单条文本描述）与 `text_description_24.json`（1.0 MB，24 条描述），用于评估该数据集能否提供文本-动作对应关系。

选型判断依据：MVHumanNet 提供大规模单人序列与文本描述，但不含双人接触与逐帧 SMPL-X 拟合精度保证，因此只作为外观多样性候选，未进入主线。
"""

NOTES["/data/shangguanrz/dataset_samples/MVHumanNet++_repo"] = r"""
**MVHumanNet++ 仓库样例与 SMPL-X 投影验证**：`smplx_visualization/` 内含 `renderer.py`（15 KB 可微/光栅渲染器封装）、`vis_smplx.py`（3 KB 投影脚本）、`projected_smplx.png`（925 KB 投影结果）与 `data/` 四件小样本 —— `100842_0675_cam_00_img.jpg`（视角图）、`100842_0675_smplx.npz`（单帧 SMPL-X 参数）、`100842_cam_00_camera.npz`（相机）、`100842_smplx_params_all.npz`（全序列参数）。仓库根另有 `Teaser.jpg`（4.5 MB）与 `process_pipeline.png`（2.7 MB 处理流水线图）。

用途：验证「官方 SMPL-X 参数 + 本项目 nvdiffrast 渲染链」能否在该数据集上复现正确的投影对齐。
"""

NOTES["/data/shangguanrz/dataset_samples/abc_showcase_20261003"] = r"""
**多源数据集横向展示（2026-10-03）**：462 个文件、7.45 GB，五个来源各一个子目录，配 `README.md`（选型说明）与 `FreeMan数据集适配性与使用方案_20261004.md`（12 KB 适配方案）。

| 子目录 | 体积 | 内容 |
|---|---:|---|
| `dna_rendering/` | 18.3 MB | DNA-Rendering 官方渲染样片：`DNA_teaser.mp4`、`DNA_final_compressed.mp4` 与三段身份片段 `DNA_sample_<日期>_<姓名>_<yf>_dz<n>.mp4` 及对应抽帧 JPG |
| `freeman/` | 7.0 MB | FreeMan 论文附录场景示例与对比 PNG 4 张 |
| `humanvid/` | 34.7 MB | HumanVid 三段 pexels 源视频（`HumanVid_pexels_<id>.mp4` + 同名 `_frame.jpg`） |
| `mvhumannet/` | 7.4 MB | MVHumanNet 单帧与 SMPL-X 投影展示（与 `MVHumanNet_repo` 样片同源） |
| `pku_dymvhumans/` | 7.38 GB | PKU-DyMVHumans 完整单序列样本（详见其目录说明） |

选型结论：本项目的双人接触与手指级控制需求，只有 Hi4D（接触标注）与自建 SMPL-X 资产链能满足；上述数据集均定位为外观/动作多样性备选，未接入主线训练。
"""

NOTES["/data/shangguanrz/dataset_samples/abc_showcase_20261003/pku_dymvhumans"] = r"""
**PKU-DyMVHumans 单序列完整样本**（430 个文件、7.38 GB，是本样片区体积主体）：

- `1080_Dance_Dunhuang_Single_f14.tar`：官方导出的原始 tar（1080 分辨率，敦煌单人舞蹈，第 14 帧序列）。
- `full_sample/1080_Dance_Dunhuang_Single_f14/`（76 MB）：解包后的典型结构 —— `overview_fme_{0,50,100,150,200}.png`（多视角总览联系表）、`videos/`（60 个 `N.mp4`，逐视角短片）、`cams/00000000_cam.txt`（相机参数）、`per_view/`（逐视角目录）、`data_COLMAP/`、`data_NeuS/`（COLMAP 与 NeRF 常用两种格式，本样本仅作格式核查未拷完整纹理数据）。
- `Dance_Dunhuang_Single_f14_sample/`：轻量预览版，含 `overview_fme_200.png` 与 `per_view/`。

用途：该数据集提供多视角 + 相机标定 + COLMAP/NeuS 双格式，是「渲染条件与相机约定是否兼容」的格式试金石；因缺少双人接触信息，未进入主线。
"""

NOTES["/data/shangguanrz/add_migration"] = r"""
**`add/` 工作区从 AutoDL 迁移到本数据盘的脚本与日志**（4 个文件）：

- `add_pull_from_autodl.sh`：对远端 `root@connect.westd.seetacloud.com`（`ssh -p 12198`，`ServerAliveInterval=30`）执行 `retry_rsync()` 包装的 `rsync -aP --partial`，无限重试直到某一大目录完整落地；用于把接入期原始数据、环境与产出快照整体迁走。
- `add_pull_other.sh` + `add_pull_other.log`：其余目录的批量拉取脚本与执行记录。日志显示首个阶段 `START: iPER V1` 被 SIGINT/SIGTERM 中断（`rsync error: received SIGINT ... at rsync.c(716) [Receiver=3.2.7]`），随后改由分批重试脚本继续，因此 `add/` 的实体最终落在仓库 `add/` 而非数据盘；这里保留脚本与日志作为迁移过程的可追溯证据。
"""

# 子目录名语义表：供「子目录内容构成」自动说明使用
CHILD_DESC = {
    "images": "RGB 帧序列",
    "masks": "人像二值 mask 序列",
    "densepose": "DensePose 渲染序列",
    "smplestx": "SMPLest-X 拟合参数（params.pt + frame_map.json）",
    "normal": "SMPL-X 法线图序列",
    "depth_vis": "深度可视化图序列",
    "smplx_mask": "SMPL-X 网格投影 mask 序列",
    "part_512": "512×512 部位分割标签图（0–14）",
    "dwpose": "DWPose 骨架可视化",
    "dwpose_json": "DWPose 结构化关键点结果",
    "cameras": "相机内外参",
    "frames": "多视角原始帧",
    "frames_vis": "官方可视化帧",
    "seg": "分割标注（图像级与网格级）",
    "smpl": "逐帧 SMPL(-X) 参数",
    "checkpoints": "训练检查点",
    "condition_cache": "单样本条件张量缓存",
    "previews": "训练过程预览图",
    "final_eval": "最终评测图与指标",
    "runs": "逐运行记录产物",
    "configs": "运行配置",
    "logs": "运行日志",
    "cache": "中间缓存",
    "review_sheets": "评审用联系表",
    "per_view": "逐视角数据",
    "videos": "逐视角视频",
    "cams": "相机参数文本",
    "data_COLMAP": "COLMAP 格式数据",
    "data_NeuS": "NeuS/NeRF 格式数据",
    "manifests": "数据清单与索引",
    "qc": "质量核查记录",
    "raw": "原始未处理数据",
    "sources": "源图资产（外观/身份侧）",
    "targets": "目标图资产（几何/姿态侧）",
    "pairs": "配对清单",
    "eval_results": "评测指标结果",
    "stage_evolution": "阶段演化对比图",
    "multistage_evolution_showcase": "多阶段演化展示样本",
    "depth_prediction": "深度预测模型检查点",
    "normal_prediction": "法线预测模型检查点",
    "text_encoder": "文本编码器分片权重",
    "transformer": "DiT 主干分片权重",
    "vae": "VAE 编解码权重",
    "processor": "处理器与分词器文件",
    "tokenizer": "分词器文件",
    "scheduler": "采样器配置",
    "assets": "随附图像资源",
    "dwpose_v1": "DWPose 检测结果层",
    "qwen21_face_feature_cache": "人脸特征离线缓存",
    "experiments": "实验产物归档",
    "models": "模型权重归档",
    "data": "小样本数据",
}

# 常见清单/元数据文件的字段中文说明（自动画像段落里附在字段列表之后）
FIELD_HINTS = {
    "sequence_id": "序列号",
    "sequence": "序列号",
    "frame_name": "帧文件名",
    "stem": "帧名（不含扩展名）",
    "frame_num": "帧序号",
    "order_index": "序列内顺序索引",
    "order_idx": "序列内顺序索引",
    "rgb": "RGB 帧绝对路径",
    "width": "原生宽度",
    "height": "原生高度",
    "quality": "逐帧质量指标（如 blur 拉普拉斯方差）",
    "assets": "各条件资产存在性标记",
    "mask_area_ratio": "人像 mask 面积占画面比例",
    "bbox_height_ratio": "人体包围盒高度占比",
    "bbox_width_ratio": "人体包围盒宽度占比",
    "touch_top": "人体是否触顶",
    "touch_bottom": "人体是否触底",
    "touch_left": "人体是否触左边",
    "touch_right": "人体是否触右边",
    "person_mask": "人像 mask 路径",
    "person_sharpness": "人像区域清晰度",
    "dark_ratio": "过暗像素占比",
    "bright_ratio": "过曝像素占比",
    "basic_valid": "基础有效性判定",
    "candidate_prevalid": "候选预判定",
    "motion": "相邻帧运动幅度",
    "motion_raw": "原始运动幅度",
    "sequence_status": "序列级状态",
    "sampling_center_stem": "采样窗口中心帧",
    "sampling_motion_class": "运动强度分级（SLOW/MID/FAST）",
    "sampling_motion_score": "运动强度得分",
    "sampling_gap": "采样间隔",
    "local_radius": "局部半径",
    "stage": "该记录所处构建阶段",
    "smplx_used_for_selection": "选帧时是否参考 SMPL-X",
    "dwpose_used_for_selection": "选帧时是否参考 DWPose",
    "pose_checked": "姿态是否已核验",
    "body_mean_conf": "身体关键点平均置信度",
    "body_valid_03": "置信度>0.3 的身体点数",
    "hand_mean_conf": "手部关键点平均置信度",
    "face_mean_conf": "面部关键点平均置信度",
    "detected_persons": "检测到的人数",
    "selected_person_index": "选定的人物下标",
    "keypoints_norm": "归一化关键点坐标数组",
    "scores": "逐关键点置信度",
    "status": "状态（PASS/FAIL）",
    "reason": "判定理由",
    "error": "错误详情",
    "paths": "各条件文件路径",
    "available": "各条件可用性",
    "core_existing_complete": "核心条件是否齐备",
    "needs": "仍缺失需补的条件",
    "smplestx_index": "params.pt 对应行索引",
    "pair_id": "配对唯一编号",
    "split": "所属划分",
    "source_video_group": "源帧所属视频分组",
    "source_stem": "源帧名",
    "target_stem": "目标帧名",
    "source_rgb": "源 RGB 路径",
    "target_rgb": "目标 RGB 路径",
    "source_person_mask": "源人像 mask 路径",
    "target_person_mask": "目标人像 mask 路径",
    "target_dwpose": "目标骨架可视化路径",
    "target_normal": "目标法线路径",
    "target_depth_vis": "目标深度可视化路径",
    "target_smplx_mask": "目标网格 mask 路径",
    "source_dwpose_json": "源骨架 JSON 路径",
    "target_dwpose_json": "目标骨架 JSON 路径",
    "pose_distance": "源/目标姿态距离",
    "source_quality": "源帧质量得分",
    "source_choice_rank": "源帧优先级排名",
    "target_pose_soft_risk": "目标姿态软风险标记",
    "pair_status": "配对冻结状态",
    "target_raw_depth": "原始 float 深度路径（本项目多未保存）",
    "smplestx_params": "SMPLest-X 参数文件路径",
    "smplestx_mapping_source": "帧号映射来源文件",
    "lap_var": "拉普拉斯方差（清晰度）",
    "tenengrad": "Tenengrad 清晰度",
    "sharpness": "综合清晰度得分",
    "smplx_iou": "SMPL-X 投影与 mask 的 IoU",
    "has_normal": "法线是否可用",
    "has_depth_vis": "深度可视化是否可用",
    "has_raw_depth": "原始深度是否可用",
    "has_smplx_mask": "网格 mask 是否可用",
    "has_smplestx_pose": "SMPL-X 参数是否可用",
    "completeness": "条件完备等级",
    "sharpness_rank": "清晰度分位排名",
    "blur_pass": "模糊阈值通过",
    "mask_pass": "mask 阈值通过",
    "size_pass": "尺寸阈值通过",
    "geometry_present": "几何条件是否齐备",
    "smplx_iou_pass": "IoU 阈值通过",
    "quality_score": "综合质量得分",
    "main_candidate": "是否主候选",
    "reject_reasons": "拒绝原因列表",
    "role": "该帧在配对中的角色",
    "roles": "该帧承担的全部角色",
    "dataset_bucket": "数据桶（如 MAIN_N2C）",
    "training_status": "是否参与训练",
    "pair": "Hi4D pair 编号",
    "camera_id": "相机编号",
    "pair_type": "配对类型",
    "pair_relation": "源/目标关系",
    "source_action": "源动作名",
    "target_action": "目标动作名",
    "same_action": "是否同动作",
    "source_frame_id": "源帧号",
    "target_frame_id": "目标帧号",
    "source_index_for_target": "目标视角下的源索引",
    "source_contact": "源帧是否处于接触",
    "target_contact": "目标帧是否处于接触",
    "target_sampling_case": "目标采样情形（接触/非接触组合编号）",
    "source_rgb_note": "",
    "contact_episode_id": "接触片段编号",
    "contact_episode_start": "接触片段起始帧",
    "contact_episode_end": "接触片段结束帧",
    "contact_episode_progress": "接触片段进度",
    "target_is_episode_peak": "是否为接触峰值帧",
    "target_contact_vertex_count_A": "A 接触顶点数",
    "target_contact_vertex_count_B": "B 接触顶点数",
    "target_contact_vertex_count_total": "接触顶点总数",
    "native_width": "原生宽度",
    "native_height": "原生高度",
    "crop_applied": "是否裁剪",
    "resize_applied": "是否缩放",
    "source_role": "源图承担的角色（外观/身份）",
    "target_geometry_role": "目标图承担的角色（交互姿态）",
    "source_selection_policy": "源帧选择策略",
    "pairing_version": "配对版本",
    "v4_2_target_case": "v4_2 目标情形",
    "v4_2_policy": "v4_2 保留策略",
    "target_masks_are_control": "目标 mask 是否作为控制（false 表示仅真值）",
    "asset_version": "资产版本",
    "seed": "划分随机种子",
    "split_unit": "划分单位",
    "identity_policy": "身份重叠策略说明",
    "total_sequences": "序列总数",
    "splits": "各划分的序列数与帧数",
    "version": "清单版本名",
    "sequences": "序列数",
    "frames": "帧数",
    "manifest": "主清单路径",
    "schema": "缓存格式版本",
    "dataset": "数据集标识",
    "appearance": "外观（源人物）标识",
    "source": "源帧特征字典",
    "target_geometry": "目标帧几何特征字典",
    "target_visibility": "目标帧可见性字典",
    "thresholds": "生成缓存时使用的阈值",
    "original_resolution": "原生分辨率",
    "frame_count": "该序列帧数",
    "resolution": "分辨率约定",
    "smplestx": "拟合模型名",
    "detector_threshold": "检测器阈值配置",
    "geometry": "几何来源",
    "renderer": "渲染器",
    "camera": "相机参数取法",
    "depth": "深度定义",
    "raw_depth_saved": "是否保存原始深度",
    "normal": "法线定义",
    "normal_basis": "法线坐标基",
    "crop": "裁剪约定",
    "source_dataset": "来源数据",
    "original_masks": "官方 mask 处理方式",
    "original_densepose": "官方 DensePose 处理方式",
}
