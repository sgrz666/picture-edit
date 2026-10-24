# CHAMP HQ-Pair V1 · 高质量单人帧对预处理方案

日期 2026-10-24。来源：用户方案草案（两阶段筛选：先筛帧 → 再配对 → 后估计 → 最后几何复核）
+ 本轮对远端 `pic-edit` 仓库与 `sg` 实况的接口实测。

管线定名：`CHAMP HQ-Pair → SMPLest-X → Geometry Render → Native Adapter V3`

---

## 0. 先说与草案不符的实测结论（这些会改变做法）

| # | 草案假设 | 磁盘实况 | 影响 |
|---|---|---|---|
| 1 | 数据源是 "CHAMP 全部 832 条视频" | 我们手上是 `gulucaptain/Human-Image-Animation-Data` shard 000：整卷 1000 条，已落盘 4.3GB，已抽 10 条。全库 25 卷 / 25000 条 / 110GB。"832" 这个数字在本仓与 HF 侧都没找到出处 | 规模与成本要按 1000/卷 重算；832 需澄清指代 |
| 2 | 有 DWPose 关键点可用 | **DWPose 权重已在** `sg:models/DWPose/dw-ll_ucoco_384.onnx`，runner 在 `scripts/data/run_tiktok_v3_dwpose.py`；但数据集自带的 pose 是**渲染 mp4，不含数值关键点** | Stage 3 必须自己跑 DWPose，不能解 pose mp4 |
| 3 | `geometry_pair_dataset_v2.py` 的 Canvas 契约可直接校验 | 该文件**只在远端** `src/data/geometry_pair_dataset_v2.py`，本地快照没有；契约真实字段见 §4 | 本地读不到，改代码要在远端树或先同步 |
| 4 | `scripts/data/build_tiktok_v6_conditions.py` 是实体 | 它是**指回 `../build_tiktok_v6_conditions.py` 的软链接**（本仓 scripts/ 常见现象，md5 比对会误判重复） | 提 PR 时别当两个文件 |
| 5 | 可以直接上 GPU 批跑 | **更正（前一行是错的）**：`nvidia-smi` 报 `NVML: Driver/library version mismatch` 只是监控库不匹配，**GPU 实际可用** —— `torch.cuda.is_available()=True, count=1`。SMPLest-X 也**全在盘上**：`add/envs/smplestx_bw`（torch 2.8.0+cu128）、`add/tools/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar`（8.2GB / 687M 参数）、`yolov8x.pt`、`SMPLX_NEUTRAL.pkl`。**不需要下载任何模型** | Stage 5 无阻塞，已实测跑通 167 帧 |
| 6 | 命名 CHAMP | 本仓 `research/qwen21_depth_champ_20261023/` 里 CHAMP 指的是**深度/法线条件编码那套模型**，不是这批视频 | 建议数据集目录改名或加限定前缀，避免同名歧义（待用户定） |

另外一条硬约束：数据集卡原文**禁止商用、要求下载后 24 小时内删除**。全量 110GB 预处理与"24h 删除"直接冲突，扩规模前必须先决定许可口径。

---

## 1. 分阶段与准入判据

```
Stage 1  无损解码 + 镜头切分 + 候选抽帧(4FPS) + 邻帧差自适应加密   → frame_index.jsonl
Stage 2  人体 ROI 质量：清晰度/重影/人脸宽度/人体高度/单人判定      → frame_quality.jsonl + usable_frame_pool.jsonl
Stage 3  DWPose whole-body(onnx) 数值关键点与置信度               → assets/{seq}/dwpose/*.json
Stage 4  升级 pose_distance → 三项融合 + 最低差异门 → pairs.jsonl
Stage 5  仅对入选唯一帧跑 SMPLest-X-Huge → params.pt + frame_map.json
Stage 6  重投影 QC / Depth / Normal / Part / Mask / 3D 骨架渲染
Stage 7  写 CHAMP_v1，过 assert_pair_contract
```

本轮范围：**Stage 1 + Stage 2 的无 ROI 部分**，对已下载的 10 条视频跑通，产出帧元数据、
清晰度分布与 QC 拼图，用来定阈值 —— 不猜阈值、不先上 GPU。

---

## 2. Stage 1 设计（已实现）

- **不重编码**：源 mp4 原样保留；导出 RGB 用 PNG 无损。
- **候选密度**：起始 4 FPS（`step = round(src_fps/4)`），非等间隔傻取。
- **自适应加密**：先在 1/8 尺度灰度上算相邻候选的运动量，对运动量 ≥ 本视频 p80 的区间
  插入中点帧 —— 因为"大动作"与"运动模糊"高度共现，必须把这一段采密，
  否则后面挑不出"动作大且清晰"的帧。
- **镜头切分**：`ffprobe -f lavfi "movie=...,select=gt(scene\,0.3)"` 取切点时间戳，
  给每帧打 `shot_id`；配对只允许同 shot 内或跨 shot 但同人（Stage 4 判）。
- **每帧指标**（全部无参考，先不看人体）：
  `lap_var_full` / `lap_var_center`（中心 50% 裁剪，压背景干扰）、
  `tenengrad`（Sobel 幅值均值）、`hf_ratio`（FFT 高频能量占比）、
  `spectral_slope`（log-log 谱斜率，运动模糊会显著变陡，且对纹理/噪声比 Laplacian 稳）、
  `rel_sharp_vs_neighbors`（与 ±3 候选的中位数之比，抓"这一帧比邻帧糊"的局部拖影）。
- **阈值策略**：按草案要求**不套固定 Laplacian 常数**，改为在数据集内做分位数归一
  （`*_pctl` 字段），人工标注一批清晰/模糊后再定绝对门。

## 3. Stage 2 待办（需要 DWPose 先跑）

草案的"人脸宽度 ≥96px / 人体高度 ≥512px / 四肢重影 / 双人混入 / 严重截断"都要**人体框与关键点**
才能算，纯图像指标判不出"人糊但背景清楚"。所以 Stage 2 依赖 Stage 3 的 bbox ——
这是草案顺序里的一处循环依赖，实测后调整为：**Stage 3 的 DWPose 提前到与 Stage 2 同一趟跑**
（反正要解码一遍），先出 bbox+置信度，再算人体 ROI 指标。CPU onnx 下 10 条视频约 400 帧可接受。

## 4. 接口对齐（远端实测，Stage 7 必须满足）

`src/data/geometry_pair_dataset_v2.py`（SCHEMA `geometry_canvas_v2`）实际校验的东西：

- 字段名兼容两种拼法：`source_canvas`/`source_canvas_spec`、`target_canvas`/`target_canvas_spec`；
  `canvas_valid_mask` 映射到 `valid_image_mask`。
- `target canvas (h,w)` 必须被 `VAE_STRIDE` 整除，否则 `PairContractError`。
- `target_token_hw` 必须等于由 target canvas 推出的 `token_hw`。
- `source_rgb` 空间尺寸必须等于 **source 自己的 canvas**（参考图走独立画布，不要求与 target 同尺寸）。
- 每个条件张量必须落在 target canvas 上：`shape[-2:] == (canvas_h, canvas_w)`。
- `canvas_valid_mask` 覆盖面积要与 canvas/resized 推出的留白面积一致；
  `human_mask` 不得标到有效画布外。

→ 结论：草案"不再强行拉成方形、所有条件共享同一坐标变换"与契约一致，且**条件是 rasterise 到
target canvas 上**的，Stage 6 渲染时不能各自为政。

`build_tiktok_v6_conditions.py` 实际产出（Stage 5/6 的输入契约）：

- 读 `assets/{seq}/smplestx/params.pt` + `frame_map.json`，缺任一则跳过该序列。
- `smplx_global: [N,26] = betas[10] + rot6d[6] + transl[3] + cam[7]`
- `kps_25_coords: [N,25,3] = (x_norm, y_norm, score)`
- `part_512/{stem}.png`：512×512 uint8，0=背景、1..14=部位（55 类 SMPL-X 顶点权重经
  `PART_MAPPING_55_TO_14` 多数投票到面片）
- 汇总写 `v6_conditions.pt`

→ 草案 §4 的参数表要按这个改：v6 链路**不直接存 63 维 body_pose**，而是存 rot6d 表达的全局量；
`betas` 10 维一致；`cam` 是 7 维。hand/expression 是否保留取决于 SMPLest-X 输出定义，
落盘时必须显式记录维度语义（草案这点正确）。

## 5. pose_distance 现状（草案判断成立）

`research/tiktok_training_20261003/baseline/build_tiktok_v3_pairs_strict.py:123-139`：
18 点 → 中心归一 + 按 `norm(maxs-mins)` 尺度归一 → 共同有效点（`common.sum()<4` 直接返回 0.0）
平均欧氏距离。**确认没有任何最低差异准入阈值**，只用于"选最大"。
`source_quality()`（同文件 142-183）是 `0.5·body+0.3·face+0.2·hand` 且注释明确写着
"Ranking only, NOT a hard filtering threshold" —— 草案要的正是把它升级成硬门。

## 6. 阻塞与依赖

1. **GPU 不可用**（NVML 驱动/库版本不匹配）→ 需要人工介入，我不重启机器。
2. **SMPLest-X 缺失**：环境、代码、权重都不在 sg → Stage 5 前装到 `/data`，
   不碰 `deepgen`/`qwen21` 环境（红线：`diffusers 0.41.0.dev0` 专留给 Qwen-Image-2.1）。
3. **许可 24h 删除** vs 全量预处理 → 待用户定口径。
4. **"832 条"指代不明** → 待用户澄清。

## 7. 本轮产物

`sg:/data/shangguanrz/pic-edit-datasets/CHAMP_pilot_20261024/`
（脚本 `scripts/champ_stage1_extract.py`，产物 `manifests/videos.jsonl`、
`manifests/frame_index.jsonl`、`qc/*.png`）
