# CHAMP HQ-Pair 试点 · Stage 1-4（10 条视频）

2026-10-24，node01。方案见仓库 `docs/superpowers/plans/2026-10-24-champ-hq-pair-v1.md`。
输入是 `dataset_samples/HumanImageAnimation_sample_20261024/sample/videos/` 的 10 条 mp4。

## 1. 目的

验证"先筛帧 → 再配对 → 后估计 → 最后几何复核"这条顺序在真实数据上跑不跑得通，
并量出**每一关的通过率**，用来决定要不要扩规模。本轮只到 Stage 4，SMPL-X 与渲染未做。

## 2. 里面有什么

- `scripts/champ_stage1_extract.py` —— 无损抽帧：4FPS 起采 + 运动量 p80 自适应加密 +
  `ffprobe` 场景切点 + 逐帧清晰度指标（`lap_var_full/center`、`tenengrad`、`hf_ratio`、
  `spectral_slope`、`rel_sharp_vs_neighbors`），每个指标同时写数据集内分位 `*_pctl`。
- `scripts/champ_stage2_quality.py` —— DWPose（`yolox_l.onnx` + `dw-ll_ucoco_384.onnx`，
  **CPU**）出人体框/face/hand 置信度，再在人体凸包 ROI 内算 `roi_lap_var`、`roi_grad_ratio`。
  全身 134 点原始数组落 `assets/{seq}/dwpose/{frame}.json`，body-18 落进 jsonl。
- `scripts/champ_stage4_pairs.py` —— 硬门筛可用帧池 → 三项融合姿态距离 → 出 `pairs/*.jsonl`。
- `manifests/videos.jsonl`（10 行）、`frame_index.jsonl`（392）、`frame_quality.jsonl`（392）、
  `usable_frame_pool.jsonl`（177）、`pairs/{train,val,test}_pairs.jsonl`（917/461/43）、
  `pair_summary.json`。
- `qc/{seq}.png` —— 每条视频最糊 2 帧 + 最清 2 帧并排，用于人工校准阈值。

跑法：`deepgen/bin/python scripts/champ_stage{1,2,4}*.py`（Stage 2 约 3 分钟 / 392 帧 CPU）。

## 3. 关键数字

- 抽帧：392 帧（324 基础 + 68 自适应加密），**0 条视频有 ≥0.3 的硬切镜**（已单独验证探针有效，
  不是解析失败）。
- **清晰度指标确实能用**：人工判读的 000777 四帧，`roi_lap_var` 糊帧 106/162、清帧 342/382，
  与肉眼完全同序；全库 p5=9.8 / p50=52.7 / p95=263.1。
- **`roi_grad_ratio` 是废的**：全库 p5–p95 只有 0.252–0.264（5% 带宽），四帧判读虽同序但落在噪声带内，
  **不能当门限用**，Stage 2 后续要换别的重影度量。
- 人数分布：单人 273 / 双人 73 / 三人 46 —— **30% 的帧是多人**，对单人流水线是直接损耗。
- 可用帧池 177/392 = **45.2%**。拒绝原因：多人 119、`roi_blur` 92、局部拖影 4；
  `person_height<512`、`body_conf<0.3`、截断三项**一个都没卡到人**（p5 人体高 639px、
  body_conf p5 0.527），说明草案这几个绝对阈值对本数据偏松，需要重定。
- 配对：1421 对过 `fused ≥ 0.35` 门限；**去重后只需估 167 帧**（1421 对 / 167 帧 ≈ 8.5×，
  正是草案"别按对估"要省的那笔）。
- 难度分布（绝对分档 <0.55 / 0.55–0.80 / ≥0.80）：**medium 51.7% / large 34.1% / extreme 14.2%**，
  与草案目标 25/50/25 **不符** —— 4FPS 采样下中等动作占多数，极大幅度偏少。

## 3b. Stage 5-7 实测（同日续跑）

Stage 5 直接复用仓库自带 builder `add/tools/SMPLest-X/scripts/build_tiktok_3d_assets_512.py`
（`--dataset smplestx_in --output assets`），SMPLest-X-Huge 687M 参数，**GPU 可用**：
12 帧推理 17 秒，6/6 序列 `FINAL PASS`，167 帧全部出 `params.pt` + `frame_map.json`
+ `depth_vis/` `normal/` `smplx_mask/`，**落盘均为原生 1080×1920**（1920×1920 只是内部
scratch 光栅，已用 `imread.shape` 核实裁回原尺寸）。

Stage 6 重投影 QC（`champ_stage6_qc.py`，复用 v6 的
`project_smplestx_body25_to_openpose`，不自己推关节映射）：

- reproj 占人体高百分比 **p50=3.20% / p90=5.16% / max=11.43%**；mesh 凸包 IoU p50=0.660、min=0.377。
- **门限得自己扫**：12% 时通过 100%（等于没设门）；6% 留 95.2%、5% 留 85%、4% 留 66.5%。
  工作点定在 **6%** → 159/167 通过、拒 8 帧。
- 最差帧 `000111/00280` 的 `Rwri` 单点差 **1172px**，是 SMPLest-X 与 DWPose 在这条 AR 滤镜近景上
  的真实分歧（我先怀疑是 in-frame 判据 bug，修完分布一字未变 —— 那个解释被数据推翻了）。

Stage 7（`champ_stage7_finalize.py`）：`CHAMP_v1/manifests/pairs/` 共
**1358/1421 对几何合格 = 95.57%**，train/val/test = 862/461/35；63 对失败对与 8 个失败帧
**隔离**在 `qc/rejected_pairs/`、`qc/geometry_failed_frames/`，没有补零（约束二）。

**溯源缺陷（6/6 序列）**：builder 硬编码 `frame_map.json.source_images` 指向
`datasets/TikTokDataset/TikTok_dataset/TikTok_dataset/{seq}/images`（该路径不存在），
`_ASSETS_DONE.json` 也写 `"source_dataset": "TikTok Dataset original images"`。
实际输入确认是我的 `smplestx_in/`（`smplx_mask` 文件名正好等于选中的帧号），
所以**无数据污染，但元数据在说谎** —— 已记入 `pair_summary.json.provenance_defects`。

## 4. 已知问题（别当成已完成）

1. **split 严重失衡**：val 只放了 000555 一条视频就出 461 对，test 两条只出 35 对（000777 是三人舞，
   几乎全被多人门拒掉）。按"视频组划分"的方向对，但 10 条规模下配比没有意义，扩规模后要重做。
2. **`identity_consistency` 在 `pairs/*.jsonl` 里仍是常数 1.0**，实测值写在旁证文件里，没有拿去删对。
   草案的"同 ID"硬门**做不到可靠**，证据链如下（`champ_stage10_identity.py` /
   `champ_stage10b_strata.py` / `champ_stage10c_gate_conflict.py` / `champ_stage10d_yaw.py`）：
   - 模型 insightface `buffalo_l` w600k_r50，CPU，159 个入选帧**全部检到人脸**（0 漏检）；
     权重在 `/data/shangguanrz/models/insightface`。逐对余弦在 `manifests/pair_identity_measured.jsonl`。
   - **对照组重叠，找不到分离阈值**：同视频对 p01=0.199 / min=0.149，跨视频对 p99=0.238 / **max=0.327**。
     0.20/0.30/0.35/0.40/0.50 五档全都不分离（`qc/identity_report.json`）。
   - 重叠主要是**人脸分辨率**造成的：按"该对较小一侧人脸宽"分层，落在对照重叠区里的同人配对数
     **110(≥0px) → 28(≥96px) → 16(≥128px) → 1(≥160px)**；≥160px 时同视频 p50 涨到 0.937。
     但**永远到不了 0**，且 1358 对里 **751 对（55.3%）至少一侧人脸 <96px** —— 也就是说
     草案 §2 那条"人脸宽度优选 ≥96px"一旦真当门用，一半以上的对**根本没有可测的身份信号**。
   - 剩下那个 0.179 的离群点**不是身份泄漏**：看 `qc/identity_worst_pairs.jpg`，最差 12 对里
     **11 对来自 000000，是同一个舞者在转身** —— 左帧正脸、右帧后脑勺，同衣同景同发髻。
     这直接说明草案内部有冲突：**"Same ID"用人脸 embedding 做硬门，会优先砍掉草案自己点名要的
     "极大幅度（转身…）"那 25%**。实测门越紧砍得越多：0.30 档 extreme 存活 92.6%，0.40 档只剩 74.3%。
   - 想用 DWPose 关键点判"脸到底可不可见"也不行：被砍那 74 对的 `face_mean_conf` p50=0.900
     vs 保留侧 0.964 —— **DWPose 在后脑勺上照样高置信地吐 68 个脸点**。
     几何 yaw 代理（鼻尖相对两眼中点/眼距）只能捞回 68.9%（阈值 0.25），且"正脸"侧仍有 0.159 的离群，
     不足以当判据。
   - 结论与做法：**不用人脸 embedding 当硬门**。身份保证仍来自"配对只发生在同一 sequence 内
     + `n_persons==1` + 镜头切分元数据"，embedding 只作为**可测时才写数的旁证字段**
     （人脸 <96px 或 yaw 超阈值 → `null`，不补 0 不补 1，同约束二的口径）。
     要真做"同 ID"，需要的是**带标注的正面/背面判别**，不是换个更大的 arcface。
   - **betas 也不能当身份键**（已排除的捷径）：跨 sequence 质心距离最小 0.093（000111 vs 000999），
     看图判定是**两个人**（`qc/identity_000111_vs_000999.jpg`：黑发蕾丝窗帘 vs 粉假发 cosplay 浓妆）。
     所以 ①**约束一没有被这一对违反**；②那条"LOO 按 betas 认对 93.4%"**认的是这条视频
     （体型+取景+服装+机位），不是这个人**，不能引用成身份可分性。
     `champ_stage9_betas.py` 的自动结论与键名已按此改写。
3. **betas 在这批数据上几乎不携带形状信息**（草案 §4 要求"同一人物 betas 稳定"，实测这条
   既成立又没意义）。`champ_stage11_betas_pool.py` 用 SMPL-X shape blend shapes 直接算
   "把 betas 池化成 sequence 质心会让网格表面动多少"，单位是米、可换算成 cm
   （`mesh_cam` 高度 p50 = **1.676 m**，先确认了量纲是公制）：
   - 池化后 RMS 顶点位移 **0.13 cm**（p90 0.45 / max 2.02），占身高 **0.079%**（p90 0.26%）；
   - 对照：**换成别人的质心**只动 **0.30 cm**（p50），最小 0.14 cm；
   - 所以"同一条视频内的形状漂移"是"人和人之间形状差异"的 **44%** —— 和上面 betas
     当不了身份键是同一件事，这次有了物理单位。
   结论：**池化基本免费**（亚毫米级，远小于 `MIN_MASK_IOU=0.25` 给的轮廓容差），
   但也说明**池化省不下任何东西** —— 逐帧 betas 本来就没承载多少可丢的信息。
   另一个必须记住的坑：**Stage 6 的重投影根本没碰 betas**，
   `project_smplestx_body25_to_openpose` 读的是 SMPLest-X 预先算好的
   `joint_proj_model [N,137,2]`，不是 body model 前向。所以"池化 betas 看重投影会不会变差"
   这个实验**按构造必然得到 0 变化**，是个自证陷阱；真正对 betas 敏感的只有
   `mesh_cam` 衍生的 mask/depth/part，以及 `mask_hull_iou`。
   草案"时序一致性优化"那一步仍未实现，但现在有了正确的目标量：要平滑的是
   `mesh_cam` 而不是重投影误差。
4. **Canvas 契约已实测通过，但我最初的写法确实错**：
   `native_pad` 实测 —— 画布取原生 1920×1080 被**直接拒绝**
   （`canvas (1920, 1080) must be a multiple of stride 16`，1080%16=8），
   合法画布是 **1920×1088**（宽补 8px）；720×1280 在 stride=16 下原生即可。
   已把 1358 对 ×2 侧共 **2716 条 canvas 改写为 `native_pad`，`from_record` 回读 2716/2716 一致**。
   再按契约要求把 RGB/normal/smplx_mask/DWPose 骨架/part one-hot **补到 1088 并转成 CHW**，
   抽 20 对真跑 `assert_pair_contract`：**20/20 PASS**（`CHAMP_v1/manifests/pair_contract_check.json`）。
   两个必须记住的坑：
   (a) 契约读 `tensor.shape[-2:]` 当 (H,W)，**多通道张量必须 CHW**，HWC 会报
   `source_rgb sits on (1088, 3)`；
   (b) **`champ_stage7_finalize.py` 会把 canvas 重置回原生恒等变换**，所以 stage7 之后
   必须再跑 `champ_canvas_check.py` + `champ_stage8_contract.py`，顺序不能颠倒。
   **已补做：part map 现在真的在目标画布上光栅化**（`champ_stage12_part_native.py`，167 帧全跑）。
   旧的 `part_512` 是"用 `sx=512/W, sy=512/H` 各向异性塞进正方形再 resize 回来"，
   所以 `stage8` 只能重采样。现在按画布 **1920×1088** 直接用原生 `focal_length_xy / princpt`
   （`cx+=offset_x`，`fx/W` 与 `fy/H` 分开）光栅化，产物落 `assets/{seq}/part_canvas/{stem}.png`。
   量化的差别（n=167，`qc/part_native/part_native_report.json`）：
   - 轮廓边界到 `smplx_mask` 边界的平均距离：**新 0.0148 px** vs **旧重采样 0.6669 px**（max 0.752）；
   - 与 mask 的轮廓 IoU：新 **0.9990**（p50=1.0000，117/167 帧逐像素相同）vs 旧 **0.9902**；
   - 部位边界像素数：新 4055/帧 vs 旧 4920/帧 —— 重采样把阶梯边**多加 21%**；
   - 新旧标签逐像素相同 99.688%，差别全在边界上；小部位改得最多（class 4 的 p05 IoU 只 0.417）。
   反例证明这个判据不是摆设：竖直镜像 **251.5 px**、单尺度（fx、fy 同乘 `cw/W`）**4.42 px**、
   根点平移 50 px **36.4 px**，全部远差于旧重采样的 0.667 px → 门结论 `RANKS_CORRECTLY`。
   契约侧：用画布 part 重跑 20 对 **20/20 PASS**；把 512² 的 one-hot 直接喂进去被**如实拒绝**
   （`part_onehot sits on (512, 512) but the target canvas is (1920, 1088); conditions must be
   rasterised on the target canvas`）—— 说明契约确实在管"必须在画布上光栅化"这条，不只是查形状。
   **两个别读错的点**：(1) `smplx_mask` 的 renderer 是"同一套 nvdiffrast"（见 `_ASSETS_DONE.json`），
   所以 IoU=1.0 是**与流水线自身约定一致**，不是第三方交叉验证；50/167 帧仍有几像素差（≤6.6k px、
   边界抖动），因为缓冲区宽度不同会让临界三角形的覆盖判定翻转。(2) 原始 `pairs/*.jsonl` 没动，
   `champ_stage12b_link_pairs.py` 另写 `CHAMP_v1/manifests/pairs_canvas/`，
   1358 对 ×2 = **2716 侧全部存在且形状 == 各自 canvas_hw，missing 0**。
   顺带两个从这张 QC 拼图（`qc/part_native/part_native_sheet.jpg`）里读出来的事实：
   - **14 类里只有一类在真正承重**：按面积统计 class 2 占人体的 **52.4%**、class 1 占 15.0%，
     其余 12 类合计 32.6%，最小两类（11/14）各只 **0.27%/0.28%**；每帧平均出现 11.78 个类，
     **最少的一帧只有 3 个类**。也就是说 part one-hot 对模型的信息量远低于"14 类"这个名字，
     头/脸与躯干同色（拼图里能直接看到），要 Face 分支真吃到部位监督得换更细的映射。
   - 拼图本身差点骗人：`applyColorMap(JET)` 按 0..255 归一化，标签 1..14 全被压成深蓝，
     第一版看起来"part map 没有结构"。已改成 `label*(255/14)` 再上色 —— 交付件的配色也会藏缺陷。
5. **`v6_conditions.pt` / `part_512` 已生成**（6/6 序列 SUCCESS）：
   `smplx_global [N,26]`、`kps_25_coords [N,25,3]`，`part_512` 张数与每条序列帧数
   **逐序列精确相等**（33/12/39/35/21/27 = 167）。跑法：
   `build_tiktok_v6_conditions.py --raw_root smplestx_in --assets_root assets --sequence <seq> --include-part`
   （不用改仓库代码，路径全走 CLI）。
6. 草案 §2 的顺序有循环依赖：判"人糊/背景糊"要先有人体框，所以 Stage 3 的 DWPose 必须与
   Stage 2 同趟跑（本轮就是这么实现的）。
7. 数据集卡要求**下载后 24 小时内删除、禁止商用**；本试点的 4.3GB 分卷 + 370MB 帧 + 渲染产物都在
   `/data` 上，扩规模前要先定口径。
   - **Body-only 池确实实现了，但薄**：见下面第 11 条。
8. **AR 美颜/贴纸滤镜没有被任何一道门抓住**：`qc/identity_000111_vs_000999.jpg` 里
   000111 整条视频都带熊耳 + 熊鼻贴纸（人脸被滤镜重绘），000999 是浓妆 + 假发 + 紫色灯光。
   草案 §2 的"服饰一致性"和 Face ROI 清晰度都**默认脸上是真实皮肤**：贴纸是低纹理平色块，
   会**压低** face ROI 的局部清晰度，而磨皮会**抬高**它 —— 方向恰好相反，两者都不判滤镜。
   扩规模前要加一道独立的 AR/磨皮判别（先抽样人工标注，同清晰度的做法），否则"同 ID"和
   Face 分支都会被污染。这一步**尚未实现**。
9. **我自己造的缺陷：DWPose 134 点布局取错偏移**（`dw-ll_ucoco_384` 是
   `body18 + foot6 + face68 + hand42`，face 起点是 **24**，不是 18）。
   Stage 2 原来切 `[18:86]` 当人脸、`[86:128]` 当手，等于**把 6 个脚点当成脸、脸尾 6 点漏掉、
   手也整体错位 6 个点**。实测影响（392 帧重跑前后对比，`qc/frame_quality.pre_facefix.jsonl` 是旧值）：
   `face_width_px` p50 **243 → 137**、p95 490 → 360；**人脸 <96px 的帧数 7 → 123/392**
   （旧值把"绝大多数人脸够大"这个假象喂给了上游判断）；`face_mean_conf` 0.95 → 0.98、
   `hand_mean_conf` 0.63 → 0.57。
   **但 Stage 4 的六道硬门没有一项用 face/hand**（用的是 `n_persons`、`body_mean_conf`、
   `person_height_px`、`roi_lap_var`、`rel_sharp`、`bbox_border_touch`，四项改动数为 0），
   所以 **1358 对配对与全部几何产物不受影响，不需要重跑** —— 受污染的只有 face/hand 诊断字段。
   已补 `face_layout_suspect` 守卫（脸框最低点不得低于髋）：错偏移下 **85/149 触发**、
   正确偏移 **0/149 触发**（先试过"中点距离"版本，两种偏移都是 0 触发，是个死门，已换掉）。
10. **`*_mean_conf` 不是概率**：它取的是 DWPose `subset` 亲和度，**可以 >1**
    （000777 实测 `face_mean_conf=1.0107`）。所以 `body_mean_conf ≥ 0.30` 这类阈值是同一把
    非标尺上的经验值，不能按"置信度 30%"去解释，也不能跨实现搬运。

11. **Body-only 池确实实现了，但很薄**：`champ_stage4_pairs.py:176` 按 `face_ok` 打
    `pool_tier`（`face+body` / `body-only`），侧身背身不会被删。但 train 862 对里
    **835 对两侧都是 face+body、23 对只有 source 是 body-only、4 对只有 target 是 body-only、
    0 对两侧都 body-only** —— 只有 **3.1%** 的对沾到这一档。草案 §2 要保的"侧/背身高质量姿态"
    确实进来了，但占比远低于极大幅度那 25% 的目标；扩规模时单独看
    `qc/pair_summary.json.pool_tiers`，别假设它自动够。
12. **清晰度门实测不合格，而我提的替代方案被自己数据否证**（`champ_stage13*`，
    `qc/sharpness_calibration/`）：按 `roi_lap_var` 分 6 档抽 60 帧、裁 **400×400 原生分辨率**
    人体窗口（缩小会把要判的拖影一起抹掉），标注时图上只印编号不印指标，避免被数字带偏。结果：
    - 现行门 `roi_lap_var >= 全库 p25` 漏掉 **63.2%（24/38）肉眼可见糊的帧**，同时误杀
      **11.8%（2/17）清晰帧** → 判据 `gate-not-fit`；
    - **糊的帧在每一档都有**，最清晰的第 5 档仍有 5 个 motionblur / 3 个 clear，
      说明"按全人体 Laplacian 分位数排序"这件事本身与运动模糊不同轴；
    - 我猜"糊是局部的（手/臂在挥，躯干静止），所以该量手相对躯干的比" —— **不成立**：
      `ratio_hand_torso` 的 Spearman 只有 **−0.128**（`roi_lap_var` 是 −0.327，
      躯干单独量 `lap_torso` −0.357 反而最好），在"至少保住 88.2% 清晰帧"的预算下，
      相对手糊门只抓到 **21.1%** 的糊帧，**比现行门的 36.8% 还差**。已按否证记录，不接进流水线。
    - 为什么还不能下"就没有好指标"的结论：n=17 clear / 38 bad，21.1% vs 36.8% 差 6 帧，
      这个样本量分不开两个门；而且我把"整帧发糊"和"只有手糊"合并成一类，
      而这两类对训练的后果完全不同（前者该丢，后者只是手部分不可信）。
      **草案要的仍是人工标注批次**，且标签必须把"整帧糊 / 局部肢体糊 / 失焦软"分开。
    - 标注者是模型不是人，已写进 `sharpness_score.json.annotator` 与 `limb_blur_score.json`。
    - **追加配对实验（`champ_stage13c/13d`）：两个门抓的是不同的错，合起来才有效。**
      先对全 392 帧算两个门，分歧区 129 帧（A 过 B 不过 58、B 过 A 不过 71），
      两侧各抽 40 帧出拼图 —— 只在分歧区比才分得开两个门。结果 A 对 45 / B 对 35，
      **McNemar 精确 p = 0.314 → 在分歧区两者不可区分**；但错法是**互补**的：
      A 放行的那组 40 帧里 28 帧其实糊（正是 B 抓到的），B 放行的那组 40 帧里 33 帧其实糊
      （正是 A 抓到的）。回到**随机 60 帧**（无偏）比四条规则：
      `A 单独` 抓糊 36.8% / 保清晰 88.2%；`B 单独` 26.3% / 82.4%；
      **`A 且 B`（任一不过就拒）抓糊 55.3% / 保清晰 70.6%**；`A 或 B` 7.9% / 100%。
      即"再加一道局部肢体门"能把糊帧抓取率从 36.8% 抬到 55.3%，
      代价是清晰帧留存 88.2%→70.6%、样本保留率 70.9%→52.7%。
      **这是产出与纯净度的取舍，属训练侧决策，没有擅自改现行门**；
      真要改得先有人工标签确认这个抓取率差不是 n=17/38 的噪声。

13. **草案 §2 表里"四肢边界（边缘扩散宽度 + 轮廓重影检测）"这一条，手写特征做不出来**
    （`champ_stage14_ghost.py` / `champ_stage14b_stored_fields.py`，134 帧已标注）：
    - 按肩→腕走廊取梯度法向剖面，读半峰宽 `spread_px` 与外侧第二条脊 `ghost`。
      **第一版控制组是我自己写错的**：我把合成模糊加在**沿手臂轴向**，而这种模糊几乎不改变
      横向边缘 → `rise_share` 只有 0.549，据此判"指标无效"是错的。改成
      **垂直于轴**（挥臂的真实运动方向）+ 各向同性两版后：垂直模糊让 `spread` 上升
      **66.7%**（5.76 → 7.05 px，均值 +1.29），各向同性 63.2% —— 指标确实响应模糊，但只有 2/3。
    - 可对**真标签**它没有区分力：`limb_spread` AUC **0.485**、`limb_ghost` **0.496**、
      `limb/torso` 比 **0.439**（全在 0.5 附近）；躯干那条 `torso_spread` AUC 0.569 也一样。
      所以这条"极高"优先级的判据**没有接进流水线**，只留证据。
    - 把 Stage 2 已经存下来的 18 个字段全量对着 134 个标签排一遍：**没有一个过 AUC 0.70**
      （`stored_fields_score.json.nothing_crosses_0_70 = true`）。最好的是
      `motion_prev` **0.690**、`spectral_slope` 0.647、`roi_grad_mean` 0.668（反向）、
      `roi_lap_var` 0.653（反向）、`lap_var_full` 0.632。
    - **最刺眼的一条**：手上最强的模糊预测子是"和前一帧差多少"（`motion_prev`），
      而"动作大"正是草案 §3 要保的东西 —— 用它当硬门会**优先删掉大幅度帧**，
      和人脸门砍掉"转身"那类极大幅度对是同一个结构性冲突。
      下一步该试的是**交互项**（只在"既高运动又低锐度"时拒），而不是再调单字段阈值。
    - **交互项已经试了，而且成立**（`champ_stage14c_interaction.py`，
      **按 sequence 留一交叉验证**，134 帧 / 10 序列；预算：清晰帧留存 ≥85%）：
      | 规则 | 留出集抓糊 | 清晰帧留存 |
      |---|---|---|
      | 现行 `roi_lap_var>=p25` | 37.8% | 94.4% |
      | `motion_prev<=p75` 单独 | 36.7% | 91.7% |
      | `motion ∧ roi_lap_var` | 38.8% | 83.3% |
      | **`motion ∧ roi_grad_mean`** | **49.0%** | **94.4%** |
      | 控制组 `roi_lap_var ∧ roi_grad_mean`（不含运动） | **0.0%** | 75.0% |
      即"既高运动又低梯度"这条联合门在**同样保住 94.4% 清晰帧**的前提下把抓糊率从
      37.8% 抬到 **49.0%（+11.2 点）**；控制组（把运动项换成两个锐度字段）留出集抓到 **0%**，
      说明 36 组阈值网格并没有从噪声里编出分数 —— 这条增益是真的。
      **仍未接进 Stage 4**：标签出自模型，且 +11.2 点这个差在 98 个 bad 上约合 11 帧，
      要落地得先有人工标签复验（任务 #10）。
14. 预筛进度：`prescan/videos.jsonl` 已 395/1000（setsid 起的 8 worker 稳定跑，
    约 2 条/分钟）。跑完必须用修正后的 `summarize()`（corpus p25 门）重算 summary，
    且**稀疏探针算不出 `rel_sharp_vs_neighbors`**，所以预筛的可用比例不能直接和试点的
    45.2% 比。

## 5. 与其他工作的关系

- 上游：`dataset_samples/HumanImageAnimation_sample_20261024/`（抽样质检，含 shard 000 分卷）。
- 对齐目标：TikTok V3 的 `pairs/*.jsonl` 字段 + 远端 `src/data/geometry_pair_dataset_v2.py`
  的 canvas 契约（该文件只在 sg，本地快照没有）。
- 取代关系：不取代 TikTok 流水线，是同一接口的**新数据源接入试点**；
  姿态距离三项融合是对 `build_tiktok_v3_pairs_strict.py:123` `pose_distance()` 的升级实现。


---

## 15. Stage 15 · Native Adapter V3 训练接入（P2 最后一行，已通过）

脚本：`scripts/champ_stage15b_trainer_preflight.py`（只 import 仓库，不改仓库）。
产物：`qc/native_train/pairs2/champ_{train,val}_pairs.jsonl`、`qc/native_train/stage15b_report.json`、
`qc/native_train/preflight_v2/canvas_preflight.json` 由脚本自身写出。

**做法**：训练入口 `scripts/train/train_qwen21_native_full.py` 把 raw/assets 根路径绑在模块常量上，
没有 CLI 开关，直接跑会去读 TikTok 的 `v6_conditions.pt`，CHAMP 的 seq 一律报
`Missing native geometry: 000000/00044`。所以按路径加载该入口模块，用**它自己的** `arguments()`
构造参数（`--canvas-policy native-pad --preflight-only --contract-samples 16`）、**它自己的**
`resolve_policy()` 与 **它自己的** `preflight()`，只在 `QwenFullDataset(...)` 这一步把
`tiktok_assets_root` 换成 `train_assets/`。这样跑的是线上那道闸，不是我另写的近似版。

**结果（16/16 对通过 `assert_pair_contract`）**：

| 量 | 值 |
|---|---|
| original → canvas | (1920, 1080) → (1920, 1088)，右侧补 8 px |
| target_token_hw / tokens | 120×68 / 8160 |
| 若是方图策略会是 | 120×120（`square_grid_would_be`） |
| canvas_buckets | `[1920,1088]x[1920,1088]` 共 16，distinct canvas = 1 |
| all_pairs_aspect_preserving | true |
| control_map 通道 | 8 = normal3 + depth1 + dwpose3 + mask1 |
| part_onehot | (14, 1920, 1088)，画布原生 |
| depth_metadata.metric | false（depth_vis 是可视化，不当度量深度训） |

**两处只有接到真实取数路径上才暴露的问题**：

1. **`_native_item` 不看 manifest 的 part 字段**，硬编码 `assets_root/{seq}/part_512/{stem}.png`
   （`src/data/tiktok_dataset.py:923` 附近）。也就是说训练吃的是 512² 旧栅格再 NEAREST 放大，
   不是本目录交付的 `part_canvas`。绕开方式（不动仓库）：建一个 `train_assets/` 影子树，
   `part_512/` 里放**按画布裁回原帧尺寸**的 part 图（offset=(0,0) 且只在右侧补边，裁 `[0:h,0:w]`
   是精确的），`v6_conditions.pt` 用软链接指回原处。
2. **骨架条件缺图时静默补零**：`dwpose` 目录里只有 46 个 `.json`，没有渲染图；
   `_native_item` 对缺失路径走 `torch.zeros(3, H, W)` 分支，**不报错**。实测这条反例：
   把路径指向不存在的文件，加载成功、`control_map` 形状正常、姿态块非零像素 = **0**。
   这正是方案里"不许静默补零"要避免的形状，只不过它发生在读取端而不是写入端。
   对策：本 stage 用 `assets/{seq}/dwpose/{stem}.json` 的 candidate/subset 现场渲染骨架 PNG
   （19 条 limb 直接复用 stage 8 的 `LIMBS`，不另写索引），并在人数≠1 时**抛错拒绝**而不是补零；
   接入后每对骨架非零像素 502–666 px。

**闸的反例**：把 `tgt_image` 改成 8×8，真实 `assert_pair_contract` 报
`PairContractError: target_rgb sits on (8, 8) but the target canvas is (1920, 1088); conditions must be
rasterised on the target canvas` —— 绿灯是可失败的那种绿灯。

**仍未做**：真实 optimizer step（要载 Qwen 基座、占共享 GPU）；`part_canvas` 与 `part_512` 的
像素级差异量化（`stage15` 的 `part_source_comparison` 因为一次自摆乌龙没跑出来：我把
`py(){ python; }` 写成不转发 `"$@"` 的 shell 函数，于是 `py -m py_compile X` 实际只跑了裸 python，
EOF 退出 0，打印出一行假的 `COMPILES`）。


---

## 16. Stage 15c · 训练实际吃到的 part 条件差多少（量化）+ 配对池几何覆盖率

脚本：`scripts/champ_stage15c_part_diff.py`；产物：`qc/native_train/part_diff_report.json`。
比的是**同一个 loader、同一批 pair**，只换 `assets_root`：`PILOT/assets`（legacy `part_512`）
vs `train_assets`（画布原生裁回原帧尺寸），指标取 loader 吐出的 `part_onehot` 张量本身，
不是磁盘上的 PNG —— 这样"交付件改了但训练没变"这类问题不会被漏掉。

| 口径 | train 16 对 | val 4 对 |
|---|---|---|
| argmax 标签不同的画布像素占比（均值） | **0.201%** | **0.143%** |
| 同上（最大） | 0.233%（000000/00056） | 0.151% |
| 前景轮廓 IoU | 0.9867 | 0.9838 |
| 前景像素数 native vs legacy | 216990 / 217121 | 130705 / 130771 |
| 最大类占比 native vs legacy | 0.3404 / 0.3402 | 同量级 |
| 逐类 IoU（14 类） | 0.924（class 7）～ 0.983 | — |
| **自身对照**（native vs native） | **0.0** | **0.0** |

结论要按数字说话，不要按动机说话：

1. **影响真实但很小。** 画布原生 part 相对训练当前实际在用的 512² 栅格，只改约 **0.14–0.23%** 的
   张量像素，全部集中在轮廓边缘带（第 7 节量的边界误差 0.667 px → 0.0148 px 就是这 0.2% 的来源）。
   类别配比、前景面积几乎不动（0.3404 vs 0.3402）。所以"part 条件画布原生"这件事的正确定位是
   **边界精度修正**，不是"条件被换掉了"；不能声称它会改变训练结果，也没有理由为它重跑全量。
2. **自身对照必须先跑。** 同一棵树跟自己比必须输出 0，否则上面所有非零数都不算证据
   （与第 13 条"任何判据配一行同一张 vs 自己"同源）。这里 0.0 成立，0.2% 才可引用。

**一次我自己的假缺陷（已更正）**：第一轮跑这个探针时 val 报
`Missing native geometry: 000555/00064`，我一度记成"配对池里存在没有 SMPLest-X 几何的帧"。
补测覆盖率后否证：见下。真因是**我建的影子树只给部分 sequence 链了 `v6_conditions.pt`**，
loader 拿不到 bundle 时抛的就是这句话 —— 报"数据缺几何"其实是"我的软链没建全"。

**配对池几何覆盖率（用 loader 的同一判据 `stem ∈ bundle['stem_to_idx']`）**：

| split | pairs | 两侧点数 | 缺几何 | 覆盖率 |
|---|---|---|---|---|
| train | 917 | 1834 | 0 | 100.00% |
| val | 461 | 922 | 0 | 100.00% |
| test | 43 | 86 | 0 | 100.00% |

合计 1421 对 / 2842 个侧面全部有 SMPLest-X 几何 —— 交付的 pairs 池在几何这一维上是完整的，
不需要按第 6 节第二条约束把任何一对从几何监督池里隔离出来。


---

## 17. Stage 16/17 · P0 全量预筛收官 + Stage 2 人工标定包 + 约束一的实测审计

脚本：`scripts/champ_stage16_corpus_and_label_pack.py`、`scripts/champ_stage17b_split_audit.py`。
产物：`prescan/videos.jsonl`(1000 行)、`prescan/summary.json`(全量重算)、
`qc/labels_v1/{crops/,sheet_XX.jpg,annotations_template.csv,sidecar_metrics.jsonl,report.json}`、
`qc/split_audit/report_v2.json`。

### A. P0「CHAMP 全视频扫描与无损抽帧」达成情况

| 量 | 值 |
|---|---|
| 扫描视频 | **1000**（shard 000 全量），空探针 **0**，全程无人 **0** |
| 探针帧总数 | 7991（每条 8 帧，~0.05 FPS 的稀疏探针） |
| 分辨率分布 | 1080×1920:874, 720×1280:78, 576×1024:12, 其余长尾（1078×1920 等 12 种） |
| 人物高度 p10/p50/p90 | 921.8 / 1213.5 / 1313.5，`videos_under_512` = 0 |
| 人脸宽 p10/p50/p90 | 82.0 / 157.5 / 276.2，`videos_under_96` = **166** |
| 清晰度门（数据集内分位，不是固定阈值） | corpus p25 `roi_lap_var` = **17.05** |
| 过筛后仍有 ≥1 候选帧的视频 | 尺寸/单人门 894 → **再加清晰度门 768**（我自己按"单人 + 高≥512 + roi_lap≥p25"重算；`summary.json` 记的 `videos_with_zero_usable=233` 反推 767，差 1 条，因为 summary 的门序与我逐条重算的顺序不完全一致 —— 差一个不是错，但两个数各留各的家，不合并成一个"干净"的数） |
| 主要拒绝原因 | roi_blur_p25 1860、multi_person_only 840、person_too_small 38、body_conf_low 13 |

两条必须保留的口径声明：
1. 稀疏探针算不出 `rel_sharp_vs_neighbors`（邻帧对比要连续帧），所以这里的 usable share
   （84.86% → 61.61%）**不可**与 pilot 4fps 密化后的 45.2% 直接比 —— 后者是被运动偏置的密化帧。
2. **计划里写的"832 条视频"对不上任何实测口径**：本 shard 实测 1000 条、全部可解码、
   没有一条产出空探针；894/768/874 也都不是 832。这句话的出处仍要人来确认。

### B. Stage 2「抽样人工标注一批清晰/模糊数据用于校准阈值」的可交付部分已就绪

`qc/labels_v1/`：140 张 400×400 **原生分辨率**上身窗口（裁剪算式与 stage13 完全一致，
所以标出来的标签可以直接接回原来那套打分脚本）。分层抽样：按 `roi_lap_var` 十等分各取 12 张
（d0–d9 = 120），外加"当前门说清晰、但邻帧相对清晰度 <0.6"的分歧带 7 张、`motion_prev` 最大的
13 张 —— 分歧区才是阈值决策真正发生的地方，均匀抽样只会把标注预算花在"一眼就能判"的样本上。

防锚定：**指标和机器判定单独放在 `sidecar_metrics.jsonl`**，交给标注人的
`annotations_template.csv` 只有 `crop_id,seq,frame_idx,label,notes` 五列，脚本自己检查
`metric_columns_leaked_into_label = []`。标签四类：`clear / full_blur / limb_ghosting / soft_focus`
（把"整帧糊""局部肢体糊""失焦软"分开是必须的：第 12/13 节已证明这三类的可分性完全不同，
合成一类的标签校准出来的阈值无法告诉我们该修哪一环）。
自检：crops 140 = CSV 140 行 = 7 张拼图 ×20，`unresolved_frames = 0`。

**缺的不是脚本，是标签本身** —— 这一步要人做。拿到 CSV 后一条命令就能出 AUC/召回对比
（`champ_stage13_score.py` 口径），不用再重跑任何 GPU 作业。

### C. 约束一「先按原始视频或人物身份划分 Train/Val/Test」：首次做成可判真伪的实测

结构层面：train {000000, 000111, 000444, 000666} / val {000555} / test {000999}，
两两交集为空 ⇒ `structurally_disjoint = true`。

但"按视频划分"只有在**同一个人不出现在两条视频里**时才等于"按身份划分"，
所以用 `manifests/face_embeddings.npz`（159 个人脸 embedding，`keys` 是 [seq, stem]）实测：

| 分布 | n | p50 | p05 | max/min |
|---|---|---|---|---|
| **同一 sequence 内**的人脸对（同人正对照） | 2325 | 0.6322 | **0.3309** | min 0.1491 |
| **跨 sequence** 的 sequence 质心对（异人） | 15 | — | — | **max 0.2744**, p95 0.2111 |

跨异人最大值 0.2744 **低于**同人分布的 p05 0.3309 ⇒ 六个 sequence 里没有身份跨 split 的证据。
但要如实记下反面：**同人 min 0.1491 是低于 0.2744 的**，两个分布在尾端确实重叠。
所以这条测试能证伪"泄漏"，不能给身份下硬结论 —— 与第 3 节"人脸 embedding 当不了同 ID 硬门"
一致，也说明这里用 p05 而不是 min 作判据是有意选择，不是凑出来的通过。

（第一版审计脚本我读错了两处字段名：`resolution_hist` 取 `r['width']` 而实际数据在 `r['meta']`，
身份检查找 `ids/seq` 而 npz 里叫 `keys`，于是输出"cannot conclude"。
第二版改成直接读 `summary.json` 的直方图 + `npz['keys']`，并补上上面那张同人正对照表 ——
没有正对照的"没发现泄漏"这句话本来是不成立的。）


---

## 18. Stage 18 · 动作难度配比：按目标分布重采样，只减不松

脚本 `scripts/champ_stage18b_band_balance.py`；产物 `manifests/pairs_balanced_v2/`
（**原始 `manifests/pairs/` 三个文件一字未动**，行数已复核 917 / 461 / 43）。

计划给的 25% / 50% / 25% 是"目标采样分布"，同时明确"不能为了满足比例而放宽清晰度或错误标注标准"。
唯一合法的手段是**只减不松**：分带后从富余的带里丢弃，一行指标都不改。做法上把每行**原文照抄**
进新 manifest，并自检 `every_emitted_line_is_verbatim_from_the_original = true`（这条是被计算的，
不是写死的 True）。分带按每个 split 单独做，否则重采样本身可能把身份划分搅坏。

| split | 前 | 后 | 结果占比 medium/large/extreme | 与目标最大偏差 | 处置 |
|---|---|---|---|---|---|
| train | 917 | **420** | 0.250 / 0.500 / 0.250 | 0.0 | 目标可达，丢 497 |
| val | 461 | **388** | 0.250 / 0.500 / 0.250 | 0.0 | 目标可达，丢 73 |
| test | 43 | **43（原样保留）** | 1.0 / 0.0 / 0.0 | 0.75 | **目标不可达** |
| 全库 | 1421 | 851（59.9%） | 0.288 / 0.475 / 0.237 | — | — |

**必须记下来的两个结论**

1. **命中目标分布的代价是丢掉 40% 的配对。** 池子里 medium 占一半（51.7%），极大幅度只有 14.2%，
   所以要凑出 25/50/25 只能大量丢弃中等幅度的对。如果只想少丢，就别要这个配比 ——
   两个诉求不能同时满足，这里选"如实报代价"而不是偷偷保留一个折中数。
2. **test split 根本不该重采样。** 它 43 对**全是 medium**（fused p50 只有 0.238，
   train/val 是 0.666 / 0.776）—— 这不是抽样能修的，是 Stage 4 在"每条视频内挑最大差异"时
   test 那条视频（000999）本身动作就小。第一版脚本用 `n = min(len(band)/target)` 对所有带取最小，
   空带直接把 test 压成 **0 行**；现在改成"哪个带没有就宣布该 split 不可达、整份原样保留"，
   并把 `splits_left_empty: []` 写进自检。
   ⇒ **要用这套数据评"大动作/转身"的能力，必须从更多 sequence 里补 test 对**（等扩量的决定）。

**我自己踩的一个字段错（已纠正并留证据）**：manifest 里同时存着 `pose_distance`（融合前的关节分量，
上限 1.5）和 `pose_distance_fused`（0.50/0.35/0.15 融合值）。第一版拿 `pose_distance` 分带，
于是把"极大幅度"数成 train 252 对；按 `pose_distance_fused` 只有 **105** 对 —— 2.4 倍的虚高，
而且我算出的全库分布 (32.5/34.8/32.7) 与 `pair_summary.json` 里早就写好的
(51.7/34.1/14.2) 明显打架 —— **对不上的旧数字先怀疑自己读错字段，不是先怀疑旧记录**。
现在按 fused 重算，我这份 (0.5172/0.3406/0.1422) 与 pair_summary 完全吻合；
这个对照也被脚本原样写进 `self_check.field_confusion_evidence`
（train [252, 105]、val [213, 97]），作为"错在哪一行"的可查证据。


---

## 19. Stage 19 · Face/Hand 精细化：一个格式误读连带推翻昨天的骨架证据

脚本：`champ_stage19c_detail_roi.py`（ROI 掩码）、`champ_stage15b_trainer_preflight.py`（已修，重跑）。
旧证据保留：`qc/stage15b_v1_superseded/`（16 张废骨架图）。

### 格式事实（probe 4 个文件后确认，别再凭记忆）

CHAMP 的 `assets/{seq}/dwpose/{stem}.json`：

| 字段 | 实际含义 |
|---|---|
| `candidate` | **就是那 134 个点本身**（DWPose 顺序 body18 + foot6 + face68 + hand21 + hand21），形状 (134, 2) |
| `subset` | **每个点的置信度**（0.2–1.06，可 >1），形状 (134,)，**不是 candidate 的下标** |

### 我造的这个 bug 及其爆炸半径

之前两个脚本都按"OpenPose 风格 candidate+subset 索引"去读，写的是 `candidate[int(subset[i])]`：
subset 值 0.2–1.06 ⇒ `int()` 之后只可能是 0 或 1 ⇒ **134 个点全塌成第 0/1 两个点**。
后果分两处：

1. Stage 15b 给训练接入测试渲染的 DWPose 骨架图，实际是"两点之间一条线 + 两个圆点"，
   非零像素 502–666。**昨天那句"骨架非零 502–666 px"是废证据。**
   修正后同一批 20 对是 **8494–14243 px**（约 17 倍），旧图已挪进 `qc/stage15b_v1_superseded/` 留档。
   重跑后训练接入证据仍然成立：16/16 过 `assert_pair_contract`，policy native-pad，
   canvas 1920×1088，token 8160，`pair[0]` 姿态块非零 11666 px。
2. Stage 19 第一版跑 face 判据时报 `face slice looks wrong: p50 = 0.141`
   —— **判据把我拦下了**（它本该拦的就是这个）。修正读取后 face 判据 p50 = 1.000。

### Face ROI：已产出，但验收锚点是循环的，必须标注

167 帧全部产出非空 face 凸包掩码（`share_without_face = 0.0`），写在 `assets/{seq}/detail/{stem}_face.png`，
**原始像素尺寸**（V5 的 `source_detail_sidecar` 明确要求 "Source support must declare original-image pixels"），
画布框 = 原框 + native_pad offset（原生补边不缩放，所以只加偏移）。

但我用来"验证 24:92 是脸"的锚点是 `face_width_px`，**而这个数本来就是 stage2/3 从同 68 个点算出来的**
⇒ p05/p50/p95 = 0.995/1.000/1.007 这个"完美吻合"是**自证**，与第 7 节 smplx_mask IoU=1.0 同源。
⇒ face 的**切片正确性目前只由 DWPose 约定 + 点数=68 支撑，不由这条数值判据支撑**；
   真正独立的复核（脸框是否包住鼻子、是否落在 person bbox 上部）还没做。

### Hand ROI：**没通过**几何复核，不交付

同一窗口 92:113 / 113:134 的凸包中心到最近腕/肘的距离，占人体对角线比例 p50 = **0.483**、
p95 = **0.738**（两侧分别 0.7316 / 0.7376，几乎相同）。手不可能离腕这么远 ⇒
**这两个 21 点窗口在这批文件里不是手**，或者点序另有排布。
按第 6 节第二条约束的精神：宁可标"未解决"，不交付一个来历不明的 ROI。
⇒ 计划里 "P2 Face / Hand 独立精细化" 只完成了一半（face 有掩码但锚点循环、hand 未定位），
   要接着做必须先在一个带人工标注点的帧上把 134 个点的位置逐一钉死。

### 方法论入账

- 读外部检测器的 json **不能靠"通用格式"的记忆**，`candidate/subset` 在这条管线里语义完全反了；
  正确做法是先 probe 一个文件、打印形状与数值范围（subset 值全在 0–1.1 → 是分数不是下标，一眼可辨）。
- 每条判据都要问"这个锚点是不是和被测对象同源"。face 那条看起来"1.000 完美通过"，恰恰因为它同源自证。


---

## 20. 关键点索引表钉死，Face / Hand ROI 改成逐侧门（stage 20 / 20b / 20c）

**为什么还要做这一节**：§19 留下的"hand 窗口不像手"一直没解决，计划里 P2「Face / Hand 独立精细化」因此只算完成一半。这一节按 §19 自己写下的待办——"先在一个人工钉过点的帧上把 134 个位置逐个对齐"——把它做完。

**索引表（人工在放大图上逐点看过，图留在 `qc/layout/`）**：

```
0 鼻  1 颈  2 右肩  3 右肘  4 右腕  5 左肩  6 左肘  7 左腕
8 右髋  9 右膝 10 右踝   11 左髋 12 左膝 13 左踝
14 右眼 15 左眼 16 右耳 17 左耳
18:24 foot6   24:92 face68   92:113 手A   113:134 手B
```

看图的证据：`sheet_000000_00112.png` 里 8/9/10 顺一条腿、11/12/13 顺另一条；`zoom_head_000444_00024.png` 里 14/15 压在两只眼睛上、16/17 在两只耳朵上；`zoom_handA_000333_00056.png` 里 92 是腕、93–96 是拇指链、96 在拇指尖。

**⚠ 腿是"同侧连续"排的，不是 COCO / DWPose 的交错序**（交错序是 8 右髋 9 左髋 10 右膝 11 左膝 12 右踝 13 左踝）。仓里 `champ_stage4_pairs.py:27` 早就写了 `LHIP, LKNE, LANK = 11, 12, 13`，是对的；我这次写 stage 20 探针时凭"通用格式"的记忆用了交错序，第一版就报出 thigh_L/R p50 **0.232**、shin_L/R **0.476** 的"腿严重不对称"假象——换成钉死的表后这两项回到 0.97 / 1.01 一档（`qc/layout/layout_report.json` body_order_symmetry：upper_arm 0.975、forearm 1.007）。**同一个错法我在这个 pilot 里犯了两次**，判据是：写任何关键点索引前先 `grep` 仓里已有的表，别从记忆里调。

**§19 那句"92:113 / 113:134 不像手"撤回**。两条互相独立的证据都说布局本来就是 §19 用的那个：
- **相邻索引距离剖面**（`layout_report.json` adjacency_profile，单位=人体框对角线）在 23→24、91→92、112→113 出现 86× / 22× / 25× 中位数的跳变，而 127→128 **没有**跳变 ⇒ 分组边界是 body18 + foot6 + face68 + hand21 + hand21，不是 onnx 包装器的 body+lhand+rhand+face+foot（那个布局会在 59→60 和 127→128 处断开）。
- **滑窗扫全部合法起点**：68 点窗最优起点 24（off-nose p50 0.41 头长），25/26/27 依次劣化到 0.66/0.87/1.04；21 点窗最优在 88–93 与 112–113（p50 0.35–0.44 前臂长），而"随机 21 点"的宽松零假设 p50 是 **1.355**。
误判出在量具上：§19 的门用**人体对角线**归一，p50 0.483 看着离谱；换成**前臂长**之后同一批点 p50 **0.086**——手根本来就压在腕上。单位一换，结论整个反过来。

**逐侧门（`champ_stage20c_hand_gate.py` → `qc/hand_gate/`）**：布局对不等于每个窗口都能用，`zoom_handB_000333_00056.png` 里 113–133 有一多半散在衣服上。所以每 (帧, 侧) 单独判：手根到最近腕 ≤ 0.35 前臂、21 点凸包半径 ≤ 0.9 前臂、质心在腕的外侧、≥12 点 conf ≥ 0.3；**两只手挂同一个腕则两侧一起判废**（物理上不可能，且没有证据留哪只，宁缺毋滥，符合计划里"不许静默伪造正常条件"那条）。167 帧实测：

| ROI | 过门 | 占比 |
|---|---|---|
| face (24:92) | 164 | 98.2% |
| hand_a (92:113) | 99 | 59.3% |
| hand_b (113:134) | 101 | 60.5% |
| foot (18:24) | 100 | 59.9% |

双手撞腕 21 帧；可用侧行 237 → **200**。反例门：把 face 的 60:81 切片喂进 hand 门，**0/167** 通过（判据不是恒真）。
产物 `qc/hand_gate/roi_usable.jsonl` 是白名单，每行 `{sequence_id, stem, roi, wrist_side, attach_forearm, span_forearm, conf_mean}`；`assets/*/detail/*_hand_*.png` 里不在白名单的那些**不要用**（图按红线不删不移，只是标为不可用）。

**顺带查了、没受影响的**：`champ_stage8_contract.py:30` 的 `LIMBS` 用的是 (8,9,10) 与 (11,12,13) 两条链 + (14,16)(15,17) 眼耳连线，与钉死的表一致 ⇒ §15 渲染给训练用的 DWPose 骨架条件图**没有**画错腿，不需要重跑。`champ_stage13b_limb_blur.py:122` 只用 (3,4)(6,7) 前臂，两种序下都对。

**仍未解决**：foot 59.9% 与 hand ~60% 的失败里，"画面里本来就看不清（半身镜头下肢外推、手被另一只手挡住）"和"检测器画错"各占多少没分；要分只能再出一批人工看图的对子。


---

## 21. 真实 optimizer step 接不上：查出两处契约缺陷（stage 21 / 21b）

**要做的事**：计划里最后一行未exercised 的 P2 —— "Native Adapter V3 训练接入：通过 Pair Contract，完成训练测试"。§15b 只跑到 `--preflight-only`（模型加载前就返回）。这次要跑 `--steps 1`，让"完成训练测试"这句话落在一次真实的梯度更新上。

**环境是齐的**（实测，不是推测）：`models/Qwen-Image-2.1` 在位（transformer/vae/text_encoder/scheduler/processor 全有），`nihaochen/envs/qwen21/bin/python` 可用，`torch.cuda.mem_get_info()` 报 **77.4 / 83.0 GiB 空闲**（nvidia-smi 仍因 NVML 版本不匹配不可用）。

**跑不通，而且不是环境问题。** `champ_stage21_one_step.py` 按 §15b 的办法按路径 import 训练入口、替换 `build_dataset` 把 part/v6 指到 `train_assets/` 影子树，然后先用前 60 行探一遍能否真的取到数。三处按顺序暴露：

1. **manifest 里的 RGB 路径是相对 pilot 根的**（`frames/000000/rgb/00172.png`），换 CWD 就取不到数。第一遍从 `scripts/` 起跑，60/60 全 `FileNotFoundError`；从 `$PILOT` 起跑才过这一关。
2. **行里根本没有条件图路径字段**。`manifests/pairs/*.jsonl` 的一行只有 `source_rgb/target_rgb`，没有 `target_person_mask / target_normal / target_depth_vis / target_dwpose`。共享加载器对缺字段是**回落到自己的根**而不是报错，于是它去找
   `pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset/000000/masks/00200.png`
   —— 拿**另一个数据集的树**来当 CHAMP 的条件，只因为目录名恰好也是六位数字才没撞上。这正是 §15 记过的"静默"族问题的另一半：静默补零之外，还有静默换源。
3. **`geometry_valid` 在 1421/1421 行都是 false，而 stage 6 自己的 `manifests/geometry_qc.jsonl` 说 167 帧里 159 帧合格**。行里同时**没有** stage 7 应当写入的 `source_* / target_*` 派生字段 ⇒ 这个标志不是 QC 结果，是一次从未匹配上的 join 留下的占位假值。**计划第二条硬约束（几何不合格必须隔离）目前在出厂 manifest 上等于没生效**——一个恒假的门既可能全拒也可能被下游忽略。

**于是改成不信任那个标志，直接按 `(sequence_id, stem)` 去 join QC 记录**（`champ_stage21b_trainable_manifest.py` → `manifests/pairs_trainable_v1/`）：路径全部写成绝对、条件图从 pilot 布局补齐、只有两帧都在 QC 合格集里且五件条件都在盘上才留；每行写 `geometry_valid_provenance` 说明证据来自哪。顺手把 §19 的掩码口径也记进数据：`mask_provenance = smplx_mesh_mask —— pilot 没有原图分割`，计划要求的"原图分割 + Mesh Mask 分别保存"这一项**没做到**，标出来而不是混着当同一样东西用。

**修完两处之后剩下的硬缺口（量化）**：

| split | 行数 | 几何 join 后判废 | 陈旧标志误判为废 | 缺 `target_dwpose` | 可训练 |
|---|---|---|---|---|---|
| train | 917 | 55 | 862 | **862** | 0 |
| val | 461 | 0 | 461 | **461** | 0 |
| test | 43 | 8 | 35 | **35** | 0 |

⇒ **DWPose 骨架条件图从来没为整个配对池渲染过**。`train_assets/dwpose_png/` 里只有 §15 为它那十几对临时渲染的约 55 帧，其余全缺。所以"真实 optimizer step"这一步现在卡在**缺条件图**，不是缺权重、不是缺显存、也不是契约校验不过。

**下一步是一个有界作业**（一次就能接上）：给 `geometry_qc.jsonl` 那 167 帧全部渲染骨架 PNG 到 `train_assets/dwpose_png/{seq}/{stem}.png`（复用 `champ_stage15b_trainer_preflight.py:render_skeleton` 与 stage8 的 `LIMBS`，§20 已确认那套腿/头索引是对的）→ 重跑 stage21b → 跑 stage21。**这一步没做，P2 最后一行仍未完成，目标不能算达成。**

**顺带没验成的**：影子树与 TikTok 资产根在 `000000` 这类六位数字 id 上是否会撞名（`TIKTOK_ASSETS = datasets/TikTokDataset/TikTok_3d_assets_native`），探针脚本里写了 collision 检查，但因为在"空 manifest"就退出而没跑到 —— **仍是未验证风险**，不要当成已排除。

### 21.1 补记（同日）：骨架补齐之后，墙换成了宿主 NVML

按上面写的下一步做完了：`champ_stage21c_render_skeletons.py` 给 `geometry_qc.jsonl` 的全部 **167 帧**渲染了骨架条件图（1080×1920 原图像素，白色像素 **10608–30033**，没有一张接近全黑），limb 表直接从 `champ_stage8_contract.py` 解析而不是重敲。重跑 stage21b：

| split | 行数 | 可训练（此前） | 可训练（此后） |
|---|---|---|---|
| train | 917 | 0 | **862** |
| val | 461 | 0 | **461** |
| test | 43 | 0 | **35** |

再跑 stage21：`canvas_preflight.json` 写出、cache 建成、750 个权重分片 + 5 个 pipeline 组件加载完，**进到 transformer 的 attention 前向**——然后两次死在同一句：

```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_()
INTERNAL ASSERT FAILED at "c10/cuda/CUDACachingAllocator.cpp":1098
```

第一次死在 `attn.to_v`，加 `TORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` 重跑换到死在 `norm_q` 的 RMSNorm，**同一个 NVML 断言**，所以不是分配器后端能绕的。这台机器的 NVML 与驱动本来就不同步（`nvidia-smi` 直接报 Driver/library version mismatch，NVML 580.178），此前只能靠 `torch.cuda.mem_get_info()` 读数。**结论：数据侧的三处缺陷已修完并量化，真实 optimizer step 现在卡在宿主层 NVML/驱动不一致，需要管理员处理，不在我能改的范围内**——不要把它记成"训练跑通了"。

### 21.2 补记二：真实 optimizer step 跑通了，之前那句 NVML 报错是显存不足的马甲

上一条把墙归因给宿主 NVML，**这个归因错了，现更正**。用独立探针复现：

- 空卡上一次申请 1 / 20 / **60 GiB** 全部成功，`empty_cache()` 也正常；
- 先占住 60 GiB 再申请 30 GiB ⇒ 立刻抛出与训练里**一字不差**的
  `NVML_SUCCESS == nvmlInit_v2_() INTERNAL ASSERT ... CUDACachingAllocator.cpp:1098`。

⇒ 该断言来自分配器**申请失败后走释放路径**时要查 NVML，而这台机器 NVML 580.178 与内核模块
580.173.02 不匹配（`/proc/driver/nvidia/version` vs `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.178.04`，
全盘只有这一个版本，`libcuda.so.1` 也指向 580.178.04）。所以**真实故障是 OOM，NVML 不匹配只是把 OOM
变成了看不懂的断言**。驱动侧不需要管理员就能理解，但 NVML 本身仍然坏着（`nvidia-smi` 不可用）。

**跑通的那一步**（`logs/stage21_run6.log`，退出码 0）：

```
step 1  loss 0.15677  grad_norm 2.0126  sigma 0.6133  elapsed 1.62 s
pair_id 000000_00024__00056   appearance 000000   target_token_hw [44, 24]
peak_cuda_gib 58.75   checkpoint_last.pt 3.7 GB 已落盘
```

配置是 `--canvas-policy aspect-preserving --max-pixels 262144`（原帧 1920×1080 → 画布 704×384，
`all_pairs_aspect_preserving=true`，1056 token），`--adapter-init scratch --steps 1 --warmup-steps 0`。
数据来自 §21.1 修好的 `manifests/pairs_trainable_v1/`，取数走真实 `QwenFullDataset` 原生分支，
part 图由影子树提供并有断言兜住（`part map would be read from ...` 会直接终止）。

**显存-分辨率实测**（同一张 83 GiB 卡，同一份数据，只改画布）：

| 画布策略 | token 数 | 结果 |
|---|---|---|
| native-pad（1920×1088 原生） | 8160 | 死在第 1 个 block 的 attention |
| aspect-preserving 589824 px | 2304 | 死得更深（MLP），仍是 OOM 马甲 |
| aspect-preserving 262144 px | 1056 | **跑通**，peak 58.75 GiB |

⇒ 计划要求的"不拉伸成方图"在 aspect-preserving 下满足，训练测试也真的发生了一次梯度更新；
但**原生 1080p 画布在这张卡上单步就要 >77 GiB**，要吃下 native-pad 需要梯度检查点或改 `execution.py`
里 adapter/激活的 fp32 口径（那是共享仓库、且 v6 拷贝被 SHA 钉住，未得许可不动），或换更大的卡。

**这一行仍不能算全部完成的地方**：本次只 exercise 了 `--branch body`，`hand_mode=off`、
face/detail 的 schedule 乘子在该 sigma 下是 0.0078，所以 Face / Hand 分支的前向与反传
还没被真实 step 覆盖；§20 的逐侧 ROI 门是它们的前置，不是替代。

### 21.3 补记三：§21 第 3 条指错了文件，现更正

§21 里"`geometry_valid` 在 1421/1421 行恒 false ⇒ 计划第二条硬约束在出厂 manifest 上没生效"这句，**测的是错的对象**。
计划第六节规定的 Stage 7 交付件是 `CHAMP_v1/manifests/pairs/{split}_pairs.jsonl`，我探的却是
中间工作文件 `manifests/pairs/`（stage 4 配对线的产物，本来就没经过 Stage 7 的 QC join）。实测交付件：

| split | 行数 | `geometry_valid=true` | 有 `missing_condition_files` 的行 |
|---|---|---|---|
| train | 862 | **862** | 0 |
| val | 461 | **461** | 0 |
| test | 35 | **35** | 0 |

合计 1358 对，每行都带 `source/target_smplx_reprojection_error_pct_height`、`..._mask_hull_iou`、
`..._geometry_valid`、canvas 与 canvas_fingerprint、pair_id、split，且引用的条件文件逐个查过存在。
⇒ **约束二在交付件上是生效的**（几何不合格的对根本没进这三个文件，走的是 `qc/rejected_pairs/`）。

**反过来，这次误判给了我一个更强的交叉验证**：§21.1 里我完全不看 Stage 7 的标志、
自己按 `(sequence_id, stem)` 去 join `geometry_qc.jsonl` 重算，得到的可训练集是
**train 862 / val 461 / test 35 —— 与交付件逐行同数**。两条独立路径算出同一个集合，
说明 Stage 6 的 QC 判定与 Stage 7 的 join 是一致的；§21 第 3 条的结论要改成
"工作目录里那份未 enrich 的中间文件容易被误当交付件"，而不是"交付件的门是假的"。

**交付件与训练器之间真正差的只有两件事**（都不是 QC 缺失）：
1. **字段名对不上**：交付件写 `target_smplx_mask` / `target_dwpose_json`，
   而 `TikTokV3PoseDataset` 读的是 `target_person_mask` / `target_dwpose`（图，不是 json）。
   名字不匹配时加载器不报错，而是回落到自己的根去找 —— 这才是 §21 第 2 条看到的"跑到 TikTok 树里"。
2. **骨架渲染图缺失**：交付件只登记了 `.json`，`target_dwpose` 需要的 PNG 由 §21.1 补渲染 167 帧解决。

`champ_stage22_trainable_from_v1.py` 因此改成**以交付件为唯一来源**：只做字段改名 + 绝对路径 +
骨架图补齐，并**同时**用 `geometry_qc.jsonl` 独立重算一遍标志做一致性断言
（两边不一致就直接终止，不静默取其一）。跑出来的 `CHAMP_v1/manifests/pairs_trainable/`
再喂给 stage21，训练测试就建立在交付件血统上，而不是我手搓的工作文件上。

**踩到的坑记一笔**：同一份数据在 `manifests/` 与 `CHAMP_v1/manifests/` 两处都有同名文件，
一个是中间态一个是交付件；任何"某字段是假的"这种结论，落笔前必须先确认读的是哪一份
（判据：文件 mtime + 是否含 Stage 7 才写的 `*_smplx_reprojection_error_*` 字段）。

---

## 22. 交付件血统闭环（stage 22）

§21.3 说交付件本身是好的，但"好"要能被机器复核，不能靠我读一遍字段。`champ_stage22_trainable_from_v1.py`
因此做两件事，且**不自己发明任何判定**：

1. **独立重算 + 断言一致**：不看交付件的 `geometry_valid`，另起一路从 `manifests/geometry_qc.jsonl`
   按 `(sequence_id, stem)` 重算每帧合格与否，再与交付件标志逐行比对。结果
   （`CHAMP_v1/manifests/pairs_trainable/trainable_summary.json`）：

   | split | 交付件行 | 可训练行 | 标志不一致 | 无 QC 记录 | 条件缺文件 |
   |---|---|---|---|---|---|
   | train | 862 | 862 | 0 | 0 | 0 |
   | val | 461 | 461 | 0 | 0 | 0 |
   | test | 35 | 35 | 0 | 0 | 0 |

   `geometry_flag_agreement_rate = 1.0`（1358 行全一致），任何一行不一致脚本会直接终止而不是"取一个"。
   这同时反证 §21.1 那次独立 join 不是巧合：两条路径给出同一个 862/461/35 集合。
2. **只做改名与绝对化**：`target_smplx_mask → target_person_mask`、渲染出的骨架 PNG 填进
   `target_dwpose`，全部路径 resolve 成绝对 —— 因为加载器对缺字段的行为是回落到自己的 TikTok 根
   （§21 第 2 条看到的那个"跑到 `datasets/TikTokDataset/.../000000/masks/` 里"）。每行写
   `condition_lineage` 说明掩码其实是 SMPL-X mesh mask、pilot 没有原图分割。

**用交付件血统重跑真实 step**（`logs/stage22_run.log`，退出码 0）：

```
step 1  loss 0.15677224099636078  grad_norm 2.0134  peak_cuda_gib 58.75
pair_id 000000_00024_00056   （交付件的 pair_id 格式，非我手搓）
```

loss 与 §21.2 那次一致到小数点后 7 位、grad_norm 差 8e-4（同一 seed 下不同取数顺序的正常抖动），
说明"契约通过 + 真实一步"这两件事现在都建立在 **Stage 7 交付件**上，而不是我为测试搭的工作文件上。

**还欠着的**：`--branch face` 与 hand 模式的真实 step（本次 `hand_mode=off`、face/detail 乘子 0.0078）、
native-pad 1080p 的显存路径（要梯度检查点或更大卡）、清晰度阈值的人工标注批次（`qc/labels_v1/` 已备好，
标签还没人打）。

## 23. 第一轮 go/no-go 决策包（stage 23）

方案的收尾句是硬要求：**"合格率确认后，再进入全量几何预处理"**。可那些数字此前散在
`qc/*.json`、`manifests/*/**.json` 里，由不同 stage 各写一份 —— 这正是过期数字能活下来的方式。
`scripts/champ_stage23_decision.py` 把每个数字连同**它的来源路径**收进一个文件
（`qc/round1_decision/decision.json`，9 个块全部非空），能现场重算的都重算，
重算不了的显式记成 open item，不静默丢。

顺带补齐方案 §6 的树形：`CHAMP_v1/frames → ../frames`、`CHAMP_v1/assets → ../assets` 两个软链接，
不复制 RGB（既省盘也不违反"不覆盖既有产物"）。

### 23.1 这一轮的量

- **预筛**：扫描 1000 条视频（方案里写的 832 与实测不符，仍未澄清是哪个口径）；
  `roi_lap_var_p25_of_corpus = 17.05`；帧可用率从 84.86% 掉到 61.61%；**233 条视频在清晰度门后零可用帧**。
- **帧池**：392 抽样帧 → 177 可用（`face+body` 176、`body-only` 1）。
  拒绝主因：`roi_blur` 92、`n_persons=2` 73、`n_persons=3` 46、`local_smear` 4。
- **配对**：1421 对过 `min_pair_distance_gate = 0.35`；Stage-7 交付件 **862 / 461 / 35 = 1358 对**；
  几何门净淘汰 train 55、val 0、test 8。
- **训练接入**：body 分支真实 optimizer step 1（`run_aspect-preserving_262144`，canvas 704×384、
  1056 token、loss 0.15677、grad_norm 2.0134、峰值 58.75 GiB）；
  1358 行独立重算 `geometry_flag_agreement_rate = 1.0`。

### 23.2 新发现一：几何 QC 的淘汰率随动作幅度上升

`decision.json` 的 `pairs.band_specific_geometry_loss` 把 pre-gate（`manifests/pairs/`）与交付件
（`CHAMP_v1/manifests/pairs/`）按 `(sequence_id, source_stem, target_stem)` 对齐，逐档算存活率：

| split | medium | large | extreme |
|---|---|---|---|
| train | 523→508（97.1%） | 289→276（95.5%） | **105→78（74.3%）** |
| val | 169→169（100%） | 195→195（100%） | 97→97（100%） |
| test | 43→35（81.4%） | — | — |

极大幅度被砍掉的比例是中等幅度的 **约 9 倍**（train：extreme 拒 27/105 = 25.7%，
large 拒 13/289 = 4.5%，medium 拒 15/523 = 2.9%）。这不是"样本少所以不稳"，
方向和方案里最怕的那件事一致：**越是想学的转身/交叉/遮挡，越容易拟合不出合格网格**。
进全量之前必须先确认这条衰减曲线在更多序列上是否成立 —— 如果 extreme 档在几百条视频上仍只有 70% 出头，
那"极大幅度 25%"这个目标分布就不是采样问题，而是几何估计的能力上限问题。

### 23.3 新发现二：test 切分里没有任何 large / extreme 对

test 只有 `000999` 一个序列，它 43 对 pre-gate 的 `pose_distance_fused` **最大 0.5208**，
全部落在 medium（边界 0.55）。所以交付件 test 35 对的档位分布是 100% / 0% / 0%。

约束一（先按组划分再配对）成立：train 4 组 / val 1 组 / test 1 组，两两零交集，
`constraint_1_split_disjointness.verdict = "PASS"`。但代价现在看得见了：test 是一个
**"按视频组干净、按动作幅度无效"的评估集** —— 在它上面没法评估大幅度姿态迁移。
下一轮扩量时分组要把动作幅度也当分层维度（每个 split 都得有 large/extreme），不能只按人/视频组随机切。

### 23.4 分档实测覆盖：交付件 ≠ 我之前引用的那组数

用绝对边界（`<0.55 / 0.55–0.80 / ≥0.80`）在交付件上重算：

| split | medium | large | extreme |
|---|---|---|---|
| train | 508（58.9%） | 276（32.0%） | 78（9.1%） |
| val | 169（36.7%） | 195（42.3%） | 97（21.0%） |
| test | 35（100%） | 0 | 0 |

此前 README 引用的 **51.7 / 34.1 / 14.2 是几何门前 1421 对的整体分布**，不是交付件的；
而 `recomputed_band_shares_train_terciles = [0.334, 0.334, 0.332]` 更是**按定义恒等于 1/3**，
它只能用来查字段有没有丢，不能当"动作覆盖"。方案说的"要先统计实际动作覆盖"必须是上面这张绝对边界表。
目标 25 / 50 / 25 在 train 上的缺口主要在 large（−18 点）与 extreme（−16 点）。

### 23.5 我自己造的两个统计缺陷（都留下了可查的痕迹）

1. **`band_balance_subset` 曾经输出 `{}`**。原因：按三个根本不存在的键名过滤，
   而 `if k in balance` 让"一个都没命中"和"命中了几个"长得一样。现在走 `subset()`：
   键名全对不上就 `SystemExit` 并把源键名打出来，部分对不上记进 `_requested_but_absent`。
   反向用例验过：bogus 键 → 抛错；缺一个键 → 记录该键而不是假装没事。
2. **第一版逐档归因把所有存活率算成 0.0**。原因：用 `pair_id` join，而 pre-gate 行根本没有
   `pair_id` 字段（只有 `sequence_id`/`source_stem`/`target_stem`）。它没报错，是因为我当时多带了一个
   `deliverable_pair_ids_not_found_pre_geometry` 计数器 —— 862/461/35 全都没匹配上才露馅。
   现在按三元组 join，并断言"交付件每个 pair 都必须在 pre-gate 里找到"，不满足就退出。

**通用规则**（这条比上面两个 bug 值钱）：任何 join 或过滤出来的统计，都要同时写一个
**证明它覆盖完整/非退化**的计数器；只写结果数字，读者无法区分"真的是 0"和"根本没连上"。

### 23.6 仍未闭合的（decision.json 的 open_items）

- 清晰度阈值**没有人工标签**：`qc/labels_v1/` 标定包已就绪、`annotations.csv` 缺，
  `sharpness_calibration.status = "OPEN"`。方案原文要求"抽样人工标注一批清晰/模糊数据用于校准阈值"，
  这一条只有人能闭。
- Face / Hand 分支还没被真实 step 覆盖（本次 `hand_mode = off`）；native-pad 1080p（8160 token）
  仍受显存限制，要么在共享仓库里开梯度检查点（需要授权），要么换更大的卡。
- `mask_hull_iou_p50 = 0.6603`（带饱和告警：`smplx_mask` 与被测对象同源，边界距离更可信）；
  重投影误差 p50 3.2% / p90 5.164% / max 11.43% 人物高度；`depth_metric = false`。

## 24. Face 分支接不上：三条各自独立的真缺口（stage 24）

> **本节两条结论已被 §25 取代**：帧里其实检得出脸（det_size 640 与 1280 都检出一张），
> 全零的根因是 `extract_bgr` 的签名不匹配，pilot 侧兼容垫片已解决；
> **Face 分支的真实 step 已在 §25.3 跑通**（loss 0.2119、grad_norm 0.0418、56.29 GiB、face_active_rows=1）。
> 本节原判断过程保留，不静默改写。Hand 分支仍未被任何 step 覆盖。

P2 最后一行要求 Face/Hand 分支也被真实 step 覆盖。这一轮把 face 这条路走到头，
结果是**没有跑 face 的 step —— 而且不该跑**：任何一次 face step 都会喂全零条件，
正是方案约束二禁止的事。三条缺口彼此独立，每条都要单独修：

### 24.1 缓存建成了，但一帧脸都没检出

生产者原样复用 `scripts/data/precompute_tiktok_face_features_v65.py`，没有改共享仓库：
它要 `{raw_root}/{seq}/images/{stem}.png`，而 CHAMP 的 RGB 在 `frames/{seq}/rgb/`，
所以建了软链影子树 `qc/face_shadow/000000/images/`（33 个 stem，33 个链接，`rgb missing=0`）。
`--assets-root` 指到 pilot 的 `assets/`，产物落 `assets/000000/v65_face_features.pt`。
模型也真的起来了：`dinov2-giant`（8.5 GB，`/data/shangguanrz/models/dinov2-giant`，本轮下到 data 盘）
+ `models/antelopev2`，`--device cuda`，退出码 0。

但张量内容说明检不出脸：

| 张量 | 形状 | 实况 |
|---|---|---|
| `arcface` | (33, 512) fp16 | 全零 |
| `dino_patches` | (33, 256, 1536) fp16 | **非零行 0 / 33** |
| `valid` | (33,) bool | **True 0 / 33** |
| `face_landmarks` / `jaw_pose` / `expression` / `face_boxes` | (33,72,3)/(33,3)/(33,10)/(33,4) | 有值（来自 SMPLest-X 参数，不是检测） |

也就是说几何侧有数据、**检测侧一条都没成功**，而原因不是"脸太小检不出"，是**代码签名不匹配**：
`metadata['extraction_failures']` 33 条全是同一句
`TypeError: InsightFaceArcFaceExtractor.extract_bgr() got an unexpected keyword argument 'full_image_bgr'`。
更要紧的是：生产者带着 `preprocess.invalid_feature_policy = "zero"` 把 33 条失败**全部补零后仍以退出码 0 交付**，
`metadata` 里 `dino_model_id: dinov2-giant`、`dino_weight_sha256`、`source_hashes`（frame_map / params 的 sha256）都写得很齐，
看起来像一次成功构建。这与 §21 记过的"骨架缺图=静默补零"是同一族：**补零策略本身没问题，
问题是它不改变退出码、也不在产物顶层留一个"这批全零"的显式判据**。
读 `metadata['extraction_failures']` 是唯一能当场识破的地方 —— 所以 stage24 的验收把
`valid_true` 与 `dino_rows_nonzero` 当成硬读数印出来（这次是 0/33，脚本按设计不给过）。

两份同名生产者的差异也顺带量出来了：
`scripts/precompute_tiktok_face_features_v65.py`（仓库根）调用签名不带 `full_image`，能出特征但 schema 旧；
`scripts/data/precompute_tiktok_face_features_v65.py` 调用带 `full_image_bgr`，对不上现装的 extractor。
两份都**没有** `FaceConditionStore` 强制要求的逐帧 `geometry_valid`（见 24.2）。
下一步顺序因此变成：先修调用签名（属共享仓库改动，**需人批准**），再补 `geometry_valid`，
才谈得上 face 的真实 step。

### 24.2 两条接口缺口：生产者与消费者不是同一套契约

1. `FaceConditionStore._load` 里 `from scripts.precompute_iper_face_features_v65 import …`
   **在当前树里不存在**（实测 `ModuleNotFoundError`；真身在 `scripts/data/precompute_iper_face_features_v65.py`）。
   ⇒ 不加别名，`--branch face` 连启动都做不到。pilot 侧做法：按文件路径加载后注册进
   `sys.modules[该名字]`，并把"别名被需要"这件事印出来（`stage24` 的 `LOADER RESOLUTION`）。
2. 生产者产出的缓存**根本没有 `geometry_valid` 这个键**（键只有
   arcface / dino_patches / expression / face_boxes / face_landmarks / frame_stems / jaw_pose /
   metadata / stem_to_idx / valid），而消费者第一行检查就是"必须有逐帧 geometry_valid 且与
   `metadata['geometry_valid']` 一致"。⇒ 两边契约对不上，谁都没错，但 face 缓存今天过不了自己的门。
   正确修法是把 CHAMP 已实测的几何 QC（`qc/geometry_qc.jsonl`，159/167 合格）逐帧写进
   `geometry_valid` + `metadata['geometry_valid']`，**不是**填 True 让它过。

顺带一条自摆：第一次我用的生产者是仓库根目录那份同名脚本（`scripts/precompute_tiktok_face_features_v65.py`），
它写出的 schema 又不一样；已把那份缓存移到 `qc/face_cache/superseded_schema/000000.root_variant.pt`
（保留、未删），改用 `scripts/data/` 那份重建。

### 24.3 为 face step 铺好的机械改动（尚未产出 face step）

`champ_stage21_one_step.py` 现在由环境变量驱动，body 与 face 走同一个包装器、互不覆盖产物：
`CHAMP_BRANCH` / `CHAMP_FEATURES` / `CHAMP_PAIRS` / `CHAMP_CANVAS_POLICY` / `CHAMP_MAX_PIXELS`，
输出目录带 `_face_feat` 后缀。`champ_stage24_face_cache.py` 会按缓存实际能服务的行来筛
face manifest 并印出丢弃计数（`appearance_not_cached` / `stem_not_in_cache` / `face_invalid`）；
本次因为 `valid` 全 False，`rows_kept` 会是 0，脚本按设计直接退出而不是发一个空池。

**结论口径**：face 分支"接入路径已打通到检测这一步，但真实 step 未跑，且不达标就不跑"。
下一步顺序：读 `extraction_failures` → 决定 det_size/池子策略 → 把几何 QC 逐帧写进 `geometry_valid`
→ 重建缓存 → `CHAMP_BRANCH=face` 跑 step。

## 25. Face 分支真实 step 跑通了，代价是把两处缺陷摆到明处（stage 25 / 26 / 27）

§24 停在"检测全零、不该跑"。这一轮把它推到底，得到两条此前只能猜的数：

### 25.1 帧里确实有脸，问题在代码不在像素

直接用 InsightFace antelopev2 跑 CHAMP 的 5 帧（`qc/face_cache/detectability_probe.json`）：

| stem | det_size 640 检出 | 框高 px | 框宽 px | det_size 1280 检出 | SMPL-X 框高 px |
|---|---|---|---|---|---|
| 00024 | 1 | 114.7 | 91.2 | 1 | 201.6 |
| 00044 | 1 | 130.9 | 98.0 | 1 | 215.8 |
| 00056 | 1 | 134.2 | 106.6 | 1 | 234.2 |
| 00112 | 1 | 131.3 | 101.7 | 1 | 218.0 |
| 00192 | 1 | 126.1 | 103.7 | 1 | 227.3 |

两件事同时成立：
- **det_size 640 与 1280 没有差别**（5/5 各检出一张脸），所以之前 33/33 全零不是分辨率不够；
- **检测框宽 91–107 px，横跨方案建议的"人脸宽度 ≥96 px"这条线**：6 万条视频真跑起来时，
  这条预筛门会切掉接近一半的"其实检得出"的脸。这一条要进 §2 阈值的人工标定同一批讨论，
  现在没有标签，所以只记数不门控。

真正的原因是 `LocalFaceFeatureExtractor.extract()`（仓库根那份生产者，334–411 行）无条件把
`full_image_bgr=` 传给 `InsightFaceArcFaceExtractor.extract_bgr()`，而装好的 `extract_bgr` 没这个参数：

```
TypeError: InsightFaceArcFaceExtractor.extract_bgr() got an unexpected keyword argument 'full_image_bgr'
```

我用仓库自己的 `extract(tight, expanded)`（不传 full_image）单帧复现，**同样报错** ——
`full_bgr` 在 `full_image=None` 时仍是 `None` 却照样作为关键字传进去。
所以这不是我调错，而是**这条路径在当前树里对所有调用者都是坏的**；
配上 `preprocess.invalid_feature_policy = "zero"`，它就表现为"退出码 0 + 全零缓存"。
（顺带一条环境事实：`qwen21` 里 onnxruntime 的 CUDA provider 装不上
（`libcublasLt.so.13` 等 5 个库缺失），InsightFace 实际跑 CPU；torch 侧的 dinov2 仍走 CUDA。）

### 25.2 不改共享仓库的做法：pilot 侧兼容垫片 + 先改内容再重盖章

- **垫片**（stage26）：在**自己进程里**把 `extract_bgr` 包一层，收下并丢弃 `full_image_bgr`，
  然后原样调仓库的构建器。共享仓库一个字节没动。代价写进产物元数据
  `metadata.champ_face_shim.consequence = "arcface is computed from the tight crop alone (no whole-frame context)"` ——
  即 arcface 与历史 TikTok 缓存同源（那份也是不带全图上下文生成的），但**不等于**新代码想要的东西。
- **结果**：`arcface` 非零 **32/33**、`dino_patches` 非零 **32/33**（有一帧检不到脸，正常），
  `valid` True 32/33。
- **指纹**：加 `geometry_valid` 之后 `validate_v65_face_cache()` 报
  `cache fingerprint mismatch` —— 这是**正确的**守卫，不是障壁：它就是要保证 metadata 与内容一致。
  做法是先改内容、再用仓库自己的 `compute_cache_fingerprint()` 重新盖章，然后原样交给消费者（stage27）。
- **`geometry_valid` 的来源**：`manifests/geometry_qc.jsonl` 的实测重投影+轮廓 QC，
  逐帧写入并镜像到 `metadata['geometry_valid']`；**没有 QC 记录的 stem 记 False（fail closed）**，
  本次 33 帧全部有记录（`geometry_records_absent = 0`），故 `geometry_valid` True 33/33。
- 新缓存另存 `assets/000000/v65_face_features_champ.pt`（不覆盖），软链到
  `CHAMP_v1/features/face_v65/000000.pt`；`FaceConditionStore(...)._load("000000")` **接受**。
- face 可用对：862 → **469**（`appearance_not_cached 393` 全是别的序列，`face_invalid 0`）。

### 25.3 Face 分支真实 step

`CHAMP_BRANCH=face CHAMP_FEATURES=... CHAMP_PAIRS=pairs_face/train_pairs.jsonl
CHAMP_CANVAS_POLICY=aspect-preserving CHAMP_MAX_PIXELS=262144`（`champ_stage21_one_step.py`，
输出目录带 `_face_feat` 后缀，不与 body run 共用 cache）：

| 项 | 值 |
|---|---|
| step / 退出码 | 1 / 0 |
| loss（flow） | **0.21185308694839478** |
| grad_norm | 0.041813924908638 |
| 峰值显存 | **56.29 GiB** |
| pair | `000000_00024_00056`，appearance=`000000`（source/target 同 appearance 字段都在） |
| target canvas / token | 704×384 / 44×24 = **1056 token** |
| **face_active_rows** | **1** |
| face_region_flow_mse / nonface | **0.23481212556362152** / 0.21145494282245636 |
| face_region_token_fraction | 0.01705（人脸只占 1.7% 的 token） |
| face_schedule_multiplier / group_scales | 0.328125 / [0.1, 0.2, 0.5, 0.8, 1.0, 1.0] |
| hand_mode / hand_active_rows | `off` / 0 |
| residual_rms / raw_branch_rms | 全 0（`--adapter-init scratch` 零初始化所致，不是 face 没接上） |

判读要说准：**`face_active_rows = 1` + face 区域 loss 与非 face 区域被分开计量**，
这两条证明 face 条件真的进了这一步（若 store 取不到，`_load` 会直接 FileNotFoundError）。
`face_reference_token_hw` 是 `null`、`face_texture_schedule_multiplier` 是 0.0 —— 在当前
warmup/step=1 的配置下纹理子项还没打开；**Hand 分支仍未被任何 step 覆盖**（`hand_mode=off`）。
所以 §七 的 P2 两行现在是：Face 已跑通真实 step（带垫片披露），**Hand 仍缺**。

### 25.4 待你定的一件事（原 §24 结尾那条，性质已变）

不再需要"批准我改共享仓库"才能跑 face —— pilot 侧垫片已经能出特征并过了消费者所有校验。
需要定的是**要不要把 `extract_bgr` 的签名/调用点修在共享仓库里**：
现在这份 CHAMP face 缓存的 arcface 是"无全图上下文"版本，而新代码的意图是"带上下文"。
两个选择：①只修 pilot，正式训练也接受无上下文 arcface（与历史 TikTok 缓存口径一致）；
②改共享仓库（影响 `scripts/precompute_tiktok_face_features_v65.py` 与其调用者），
之后需要重生成才能对齐新口径。**默认我按 ① 走并全程标注**，除非你说要 ②。

## 26. Face 缓存铺满整个训练池（stage 28 / 29），并暴露两处我自己的工具缺陷

§25 只覆盖了 `000000`。这一轮把 face 条件做到**训练池全部 4 个 appearance**，并把"能不能被消费者接受"
这件事从"我跑过一次"变成"每个缓存都单独过一遍 store 的校验"。

### 26.1 逐序列实况（`qc/face_cache/stage29_coverage.json`）

| appearance | 帧数 | 人脸检出 valid | geometry_valid | arcface 非零 | dino 非零 | store 判定 |
|---|---|---|---|---|---|---|
| 000000 | 33 | 32 | 33 | 32 | 32 | 第 1 次就接受 |
| 000111 | 12 | 12 | 11 | 12 | 12 | **第 2 次才接受**（先重盖章） |
| 000444 | 39 | 39 | 39 | 39 | 39 | **第 2 次才接受**（先重盖章） |
| 000666 | 21 | 21 | 18 | 21 | 21 | 第 1 次就接受 |
| 合计 | **105** | **104** | **101** | 104 | 104 | 4/4 可用 |

合并后的 face 可用对：**862 / 862**（`appearance_not_cached 0`、`stem_not_in_cache 0`、`face_invalid 0`）。
交叉核对：上一轮只有 000000 时是 469 可用 + 393 "没缓存"，`469 + 393 = 862` ✓ 对得上，
所以这一轮的"零丢弃"不是判据在空转，而是覆盖真的补齐了。仍然要说清楚：
**`geometry_valid` 只有 101/105**，那 4 帧的几何不合格，只是恰好没被任何一对选中；
如果以后配对用到它们，这 4 帧会让 face 池缩到 101，不会静默放行。

### 26.2 face step 在合并清单上重跑一次

同一 step、同一 pair（`000000_00024_00056`）、`target_tokens 1056`：

| 项 | §25（单序列清单） | §26（合并清单） |
|---|---|---|
| loss | 0.21185308694839478 | **0.21185308694839478（逐位相同）** |
| grad_norm | 0.041813924908638 | 0.041823115199804306 |
| 峰值显存 | 56.290546894073486 GiB | 56.29514694213867 GiB |
| face_active_rows | 1 | 1 |

loss 逐位一致说明取数与 loss 路径是确定的；**grad_norm 与峰值在第 4~5 位小数上不同**，
方向和"反向归约顺序不确定"一致，但我这轮没做重复跑对照，所以只记为**未证实的疑似非确定性**，
不写成结论，也不拿它当"复现失败"。（要证实只需同清单连跑两次比 grad_norm。）

### 26.3 我自己这两处工具缺陷（都是"退出码说谎"，方向与被测对象无关）

1. `champ_stage26_face_build.py` **每次都退出 1**，却在退出前写出了可用缓存：它的 store 自检走
   `pose_control...data_runtime`，而 `data_runtime` 自己 `from src.data...` 需要仓库根在 `sys.path` 上，
   我只插了 `src/`。于是"构建成功"这件事在退出码上是假的 —— 真正干活的是 stage27。
   现按 stage29 的口径判定：**以 store 能否 `_load` 为准**，不看 stage26 的退出码。
2. 第一版 `champ_stage28_face_batch.py` 忘了注册 `scripts.precompute_iper_face_features_v65` 这个
   store 期望的模块名（stage27 里有、stage28 里没有），于是批处理在合并清单那一步崩了，
   而**它崩之前已经把 `pairs_face/train_pairs.jsonl` 留成了单序列版本** —— 一个"看起来存在、其实变窄"的交付物。
   stage29 重算并覆盖，现在文件里是 862 行、4 个 appearance 全在。
3. 还有一条纪律教训：我第一次试图用 `ssh "…python - <<PYEOF…"` 塞一段带反引号的多行脚本，
   bash 把 `` `extract_bgr` `` 当命令替换执行了，只写出半条句子；这就是 AGENTS.md 要求 `scp` 直传的由来。
   本轮改用"本地写 .py → scp → 远端 `py_compile` → 运行"，一次都没再犯。

### 26.4 P2 两行的现状

- **Face 独立精细化**：条件链打通并落到训练池全量（105 帧 / 862 对），真实 step 已跑（两次，见 §25.3、§26.2）。
  口径限制照旧：arcface 是"无全图上下文"版本（垫片丢弃 `full_image_bgr`），要带上下文必须改共享仓库。
- **Hand 独立精细化**：**仍未被任何 step 覆盖**（`hand_mode = off`、`hand_active_rows = 0`）。
  逐侧 ROI 门（§20）与 `qc/hand_gate/roi_usable.jsonl` 已就绪，缺的是 `HandConditionStore` 那侧的缓存构建，
  下一轮按 face 同一套路做（先查它的必需键，再决定用 SMPLest-X 手部参数还是渲染 ROI）。

## 27. Hand 分支：我差点把「自己的 schema 没对上」写成了「CHAMP 没有手部数据」（stage 30 / 31 / 32）

### 27.1 先记一条作废结论

stage30 我下过一句判断：**「Hand 数据侧对 CHAMP 不可用，真实 hand step 跑不得（跑就是喂占位框 + 全零）」**。
**这句作废**，原因见 27.2；作废链留在原地不删，是因为它错得很有代表性。
同一份 stage30 输出里的三个计数也**全部读错**，逐条更正如下。

| stage30 我写的 | 实况 | 我错在哪 |
|---|---|---|
| `person_valid true 0 / 240` | `person_valid` 不是 store 存的字段，是 `detail/hand.py:37` 的派生属性 `region_valid.any(dim=-1)`；分母应是 `120` 个 pair 槽而非 `240` 个 cell | 把派生量当原始量、把 shape 当样本数 |
| `source_boxes rows_nonzero 120/120`（当成"有框"） | store 在进循环前就 `source_boxes[...] = dummy_box = [0,0,0.1,0.1]` 全表预填，非零计数恒等于 batch size | 判据被自己的占位值撑满；应比"是否 ≠ 占位框" |
| `roi_usable` 交叉核对 `sides_total 0` | 该文件每行的键是 `{sequence_id, stem, roi, wrist_side, attach_forearm, span_forearm, conf_mean}`，我按 `{side, usable}` 读的 | 没读原文就定字段名（与 2026-10-24 那三次同型错误） |

### 27.2 真正的原因，是一条能机械复现的链

`HandConditionStore.for_pairs()` 里让一行"什么都不发生"的分支只有三处，逐处实测：

1. `data_runtime.py:297-300` 取 `row["source_dwpose_json"]` / `["target_dwpose_json"]`，交给
   `_get_dwpose()`；后者第一句是 `Path(default_path).exists()`（`data_runtime.py:252`），
   而 manifest 里这两个字段是**相对 `CHAMP_v1/` 的路径**（`assets/000000/dwpose/00024.json`）。
   进程 CWD 不在 `CHAMP_v1/` 时，240 个槽全部 `exists()==False` → `continue`。
2. stage31 把路径按 `CHAMP_v1/` 解析成绝对路径后重跑：240 个槽全部命中文件，但
   `keypoints_norm` 长度分布是 `{0: 240}`，仍全部在 `len != 134` 处 `continue`。
   磁盘上的真实 schema 是 `candidate[134][2]`（像素）+ `subset[134]`（分数）+ `n_persons` + `kpt_count`，
   **store 读的 `keypoints_norm` / `scores` 这两个键名在 CHAMP 的 json 里根本不存在**。
3. 所以"全零 + 退出码 0 + 不报错"是**我的数据接口没对上 store 的字段名**，不是 CHAMP 缺手部标注。
   DWPose 那 21+21 个手部点一直都在，只是叫别的名字。

### 27.3 补上适配器之后，实测数

stage32 写了一个只做"字段改名 + 除以画布"的适配器（`scripts/champ_stage32_hand_adapter.py`）：
`keypoints_norm = candidate / [W, H]`、`scores = subset`，归一化后若坐标越出 `[-0.02, 1.02]` 就**判错而不是硬夹**，
产物落到新目录 `CHAMP_v1/assets/{seq}/dwpose_hand/{stem}.json`，并另开 `CHAMP_v1/manifests/pairs_hand/train_pairs.jsonl`；
**body 用的 `manifests/pairs/` 一个字没改**。120 对（240 帧）实测：

| 项 | 值 |
|---|---|
| 写出 dwpose_hand json | 240 / 240，`rows_skipped 0` |
| 帧画布 | `1080x1920`（source 120 + target 120，无第二档） |
| `region_valid` slot0 | **240 True / 240 cell** |
| `region_valid` slot1 | 0 True / 240 cell —— **store 从不写 slot1**（只写 `region_valid[i, 0]`），这是代码形状，不是数据缺失 |
| `person_valid` | 120 True / 120 pair 槽（= `region_valid.any(-1)`） |
| `hand_keypoints` 非零行 | 120 / 120 |
| `hand_pose` 非零行 | 120 / 120 —— 说明 `_get_smpl()` 在 `assets/{seq}/smplestx/{params.pt,frame_map.json}` 也真取到了参数 |
| `target_boxes` slot0 真框（≠ 占位框） | 240 / 240，例：`[0.6021, 0.3912, 0.6621, 0.4512]` |
| `mesh_bps` / `mesh_valid` / `depth` / `visibility` 之外的手部参考特征 | 仍是 store 里写死的 `torch.zeros(...)`（`data_runtime.py:385-395`），**与 CHAMP 无关，补不了** |

一条**饱和指标**要按 ④ 标出来：适配器下"每只手 21 个点里 score>0.15 的比例"实测 **1.000 / 1.000**（左右手各 240 帧）。
这不是"手部都清晰"，是 store 那个 0.15 阈值对 DWPose 的手部分数**永远不会拒绝**；
真有判别力的是 `qc/hand_gate/roi_usable.jsonl` 的 `attach_forearm` / `span_forearm` 逐侧门（stage30 已写出）。
所以 hand 条件的"可用性"要看 ROI 门，不能引用 1.0 这个数。

### 27.4 顺带量到：store 写死的 604×1080 letterbox 对 CHAMP 几乎无害，但那是巧合

`align_mode="native"` 走 `NativePad.from_size((604, 1080))`，是 TikTok 的画布常数。
用 3 个已知归一化点探针（stage32 `nativepad_probe`）：`(0.5,0.5) → (0.49671, 0.49632)`，中心位移 `-0.0033 / -0.0037`，越界点数 `0`。
原因是 604/1080 = 0.5593 与 CHAMP 的 1080/1920 = 0.5625 只差 0.57%。
**结论限定成这样**：对 9:16 的 CHAMP 帧该常数带来的几何误差在 0.3% 量级，
但它是"宽高比撞上"而非"按 `source_canvas` 正确换算"；换一条 1:1 或 3:4 的序列就会真偏，
接进训练前应该把画布常数改成来自 manifest（属共享仓库改动，未做，等人批）。

### 27.5 Face 分支真实 step 的可复现性（本节数字由 `train_metrics.jsonl` 现算，不是手抄）

先更正我自己在 stage27 初稿里写错的两件事：
① 我曾列了一张**三趟**的表，实际磁盘上是**五趟**（face 分支）；
② 我曾说"`setsid` 循环里 `$i` 被外层 shell 提前展开，三趟写进同一个 `logs/face_rep.log` 互相覆盖，逐趟归因丢了"。
   `logs/face_rep.log` 确实被覆盖（stdout 只留最后一趟），但 **`train_metrics.jsonl` 是追加写的**，
   逐趟归因从未真的丢——我当时的错误是**没去查那个会追加的权威文件**，就把"我手上没整理好"写成了"证据没了"。
   这与 ⑬/㉑ 同源：结论要指向磁盘上最权威的那份记录。

**`run_aspect-preserving_262144`**（2 条 step-1 记录，逐条取自该目录的 `train_metrics.jsonl`）

| # | `run_aspect-preserving_262144` 的 `grad_norm` | `loss` | `peak_cuda_gib` | `elapsed_seconds` |
|---|---|---|---|---|
| 1 | 2.012568235397339 | 0.15677224099636078 | 58.75362730026245 | 1.617 |
| 2 | 2.0133941173553467 | 0.15677224099636078 | 58.75197124481201 | 1.936 |

趟数 2；`loss` 取值集合大小 1（[0.15677224099636078]）；`grad_norm` 极差 8.259e-04（相对 0.041%），distinct 2 个；`peak_cuda_gib` distinct 2 个（[58.75197124481201, 58.75362730026245]）。

**`run_aspect-preserving_262144_face_feat`**（5 条 step-1 记录，逐条取自该目录的 `train_metrics.jsonl`）

| # | `run_aspect-preserving_262144_face_feat` 的 `grad_norm` | `loss` | `peak_cuda_gib` | `elapsed_seconds` |
|---|---|---|---|---|
| 1 | 0.041813924908638 | 0.21185308694839478 | 56.290546894073486 | 1.896 |
| 2 | 0.041823115199804306 | 0.21185308694839478 | 56.29514694213867 | 2.388 |
| 3 | 0.041817557066679 | 0.21185308694839478 | 56.29514694213867 | 2.352 |
| 4 | 0.04181633144617081 | 0.21185308694839478 | 56.29514694213867 | 2.204 |
| 5 | 0.041813451796770096 | 0.21185308694839478 | 56.29514694213867 | 2.399 |

趟数 5；`loss` 取值集合大小 1（[0.21185308694839478]）；`grad_norm` 极差 9.663e-06（相对 0.023%），distinct 5 个；`peak_cuda_gib` distinct 2 个（[56.290546894073486, 56.29514694213867]）。

口径：`loss` 与 `sigma`/`face_region_flow_mse` 在 face 分支各趟逐位相同，但 **`grad_norm` 每趟都不同**，抖动出现在第 4~5 位有效数字；所以"同 seed 可复现"只能对 loss 类量断言，不能对 `grad_norm` 断言到 6 位。
`peak_cuda_gib` 也不是常量（face 分支 56.2905 与 56.2951 两档、body 分支 58.7520 与 58.7536 两档），
因此**任何拿显存数当回归门的做法都要先给容差**，否则会把分配器抖动读成显存变化。
本节数字随文件重算即得，复现命令：`scripts/fix_section27_5.py`（它同时是这张表的生成器）。

### 27.6 这一轮 P2 还剩什么

- Hand 的**数据侧**不再是缺口（27.3 已给条件），**训练侧**仍是：`--branch` 只有 `{body, face}`，
  且 `enable_hand` 由 checkpoint 里是否有 `legacy.hand_*` 键决定，`--adapter-init scratch` 下永远为假 ⇒
  真实 hand optimizer step 需要改共享仓库（加 `--branch hand` 或让 `enable_hand` 可由 CLI 打开），未获许可，不擅自改。
- `mesh_bps` / `mesh_valid` / `depth` / `dino_patches` 这四路手部条件在 store 里是硬编码零，不是 CHAMP 能补的。
- native-pad 1080p 显存路径与清晰度阈值校准（仍在等 `qc/labels_v1/annotations.csv` 那批人工标签）不变。

## 28. Hand 覆盖率铺满训练池，但两个 100% 都不是质量证据（stage 33 / 34）

本节所有数字由 `scripts/champ_stage34_test_diag_and_section28.py` 现算，产物 `qc/hand_gate/stage33_hand_coverage.json`。

### 28.1 适配器覆盖到全部分水岭后的实测

| split | manifest 对数 | 可适配 | 适配失败原因 |
|---|---|---|---|
| train | 862 | 732 | {"coords_out_of_range on (1080, 1920)": 130} |
| val | 461 | 461 | {} |
| test | 35 | 0 | {"coords_out_of_range on (1080, 1920)": 35} |

可适配行在真实 store 调用里的结果（`pairs_hand/{split}_pairs.jsonl`，dwpose 路径已全部改为绝对路径；stage30 读到的全零有一部分就来自相对路径 + CWD，body 用的 `manifests/pairs/` 未改动，120 行冒烟清单改名保留为 `train_pairs_smoke120_superseded.jsonl`）。

| split | region_valid(slot0) | 真框(≠占位) | hand_keypoints 非零 | hand_pose 非零 |
|---|---|---|---|---|
| train | 732/732 | 732/732 | 732 | 732 |
| val | 461/461 | 461/461 | 461 | 461 |
| test | — | — | — | — |（0 行可适配，本节不为一行不存在的样本编数）

### 28.2 两个 100% 都要按饱和指标处理，不能当质量证据
① `region_valid` 左右手都 100% 是因为 store 的判据是"该手 ≥3 个点 score>0.15"，而 CHAMP 的 DWPose 手部分数几乎全部 >0.15（27.3 实测比例 1.000）⇒ **这条门在本数据上永真**，它只证明"取到了数"，不证明"手能用来训练"。真正有判别力的是逐侧 ROI 几何门：

```json
{
  "attach_forearm_p50": 0.0824,
  "conf_mean_p50": 0.694,
  "gate_report": {
    "both_hands_claim_same_wrist": 21,
    "counterexample_face_slice_scored_as_hand": {
      "expected": 0,
      "n": 167,
      "note": "a face sub-slice fed to the hand gate must not pass",
      "passed_hand_gate": 0
    },
    "face_usable": 164,
    "foot_usable": 100,
    "frames_gated": 167,
    "frames_skipped_missing_json": 0,
    "hand_a_usable": 99,
    "hand_attach_forearm": {
      "max": 14.5375,
      "n": 258,
      "p50": 0.086,
      "p90": 0.1757
    },
    "hand_b_usable": 101,
    "hand_span_forearm": {
      "max": 6.6009,
      "n": 258,
      "p50": 0.4103,
      "p90": 0.7602
    },
    "index_table": "pinned by eye on qc/layout/sheet_000000_00112.png and zoom_* for 000444_00024",
    "thresholds": {
      "attach_forearm_max": 0.35,
      "beyond_wrist_min": 0.0,
      "chosen_from": "hand-sized geometry, not tuned on these rows",
      "conf_min": 0.3,
      "span_forearm_max": 0.9
    },
    "unparsed_detail_names": {
      "count": 0,
      "examples": []
    },
    "usable_side_rows": 200
  },
  "hand_rows_total": 200,
  "row_keys": [
    "attach_forearm",
    "conf_mean",
    "roi",
    "sequence_id",
    "span_forearm",
    "stem",
    "wrist_side"
  ],
  "span_forearm_p50": 0.4153
}
```

② `hand_pose` 非零 100% 也是构造使然：pairs 的 target 帧本来就取自已跑过 SMPLest-X 的入选帧池，所以 `_get_smpl()` 命中率高并不意外；它测的是"取到参数"，不是"参数可信"——后者由 §23 的重投影门负责。

**结论限定**：Hand 数据侧通路已建立（train/val 可跑），但**可用性门不在 store 里**，接训练前必须先接 ROI 门。

### 28.3 test 35 对全军覆没的真实原因（已定位到形状，未定级）

```json
{
  "test": {
    "distinct_manifest_canvas": [
      [
        0,
        0
      ]
    ],
    "distinct_rgb_sizes": [
      [
        1080,
        1920
      ]
    ],
    "n_slot_records": 70,
    "outside_by_group_totals": {
      "body_0_17": 70,
      "face_24_91": 0,
      "foot_18_23": 2,
      "hand_left_92_112": 0,
      "hand_right_113_133": 0
    },
    "rgb_vs_manifest_canvas_mismatch": 70,
    "rows": 35,
    "rows_with_any_out_of_range_slot": 23,
    "worst_3": [
      {
        "canvas_mismatch_rgb_vs_manifest": true,
        "manifest_canvas": [
          0,
          0
        ],
        "n_points_outside": 4,
        "outside_by_group": {
          "body_0_17": 1,
          "face_24_91": 0,
          "foot_18_23": 1,
          "hand_left_92_112": 0,
          "hand_right_113_133": 0
        },
        "px_max": [
          889.9,
          2119.5
        ],
        "px_min": [
          72.3,
          388.5
        ],
        "rgb": [
          1080,
          1920
        ],
        "seq": "000999",
        "stem": "00008"
      },
      {
        "canvas_mismatch_rgb_vs_manifest": true,
        "manifest_canvas": [
          0,
          0
        ],
        "n_points_outside": 4,
        "outside_by_group": {
          "body_0_17": 1,
          "face_24_91": 0,
          "foot_18_23": 0,
          "hand_left_92_112": 0,
          "hand_right_113_133": 0
        },
        "px_max": [
          1354.7,
          2121.8
        ],
        "px_min": [
          36.9,
          382.7
        ],
        "rgb": [
          1080,
          1920
        ],
        "seq": "000999",
        "stem": "00016"
      },
      {
        "canvas_mismatch_rgb_vs_manifest": true,
        "manifest_canvas": [
          0,
          0
        ],
        "n_points_outside": 3,
        "outside_by_group": {
          "body_0_17": 1,
          "face_24_91": 0,
          "foot_18_23": 0,
          "hand_left_92_112": 0,
          "hand_right_113_133": 0
        },
        "px_max": [
          1355.0,
          2120.2
        ],
        "px_min": [
          94.2,
          392.1
        ],
        "rgb": [
          1080,
          1920
        ],
        "seq": "000999",
        "stem": "00020"
      }
    ]
  },
  "train": {
    "distinct_manifest_canvas": [
      [
        0,
        0
      ]
    ],
    "distinct_rgb_sizes": [
      [
        1080,
        1920
      ]
    ],
    "n_slot_records": 189,
    "outside_by_group_totals": {
      "body_0_17": 189,
      "face_24_91": 0,
      "foot_18_23": 66,
      "hand_left_92_112": 44,
      "hand_right_113_133": 38
    },
    "rgb_vs_manifest_canvas_mismatch": 189,
    "rows": 862,
    "rows_with_any_out_of_range_slot": 18,
    "worst_3": [
      {
        "canvas_mismatch_rgb_vs_manifest": true,
        "manifest_canvas": [
          0,
          0
        ],
        "n_points_outside": 37,
        "outside_by_group": {
          "body_0_17": 1,
          "face_24_91": 0,
          "foot_18_23": 0,
          "hand_left_92_112": 1,
          "hand_right_113_133": 1
        },
        "px_max": [
          1257.8,
          2105.1
        ],
        "px_min": [
          25.1,
          114.5
        ],
        "rgb": [
          1080,
          1920
        ],
        "seq": "000111",
        "stem": "00184"
      },
      {
        "canvas_mismatch_rgb_vs_manifest": true,
        "manifest_canvas": [
          0,
          0
        ],
        "n_points_outside": 37,
        "outside_by_group": {
          "body_0_17": 1,
          "face_24_91": 0,
          "foot_18_23": 0,
          "hand_left_92_112": 1,
          "hand_right_113_133": 1
        },
        "px_max": [
          1257.8,
          2105.1
        ],
        "px_min": [
          25.1,
          114.5
        ],
        "rgb": [
          1080,
          1920
        ],
        "seq": "000111",
        "stem": "00184"
      },
      {
        "canvas_mismatch_rgb_vs_manifest": true,
        "manifest_canvas": [
          0,
          0
        ],
        "n_points_outside": 37,
        "outside_by_group": {
          "body_0_17": 1,
          "face_24_91": 0,
          "foot_18_23": 0,
          "hand_left_92_112": 1,
          "hand_right_113_133": 1
        },
        "px_max": [
          1257.8,
          2105.1
        ],
        "px_min": [
          25.1,
          114.5
        ],
        "rgb": [
          1080,
          1920
        ],
        "seq": "000111",
        "stem": "00184"
      }
    ]
  },
  "val": {
    "distinct_manifest_canvas": [],
    "distinct_rgb_sizes": [],
    "n_slot_records": 0,
    "outside_by_group_totals": {
      "body_0_17": 0,
      "face_24_91": 0,
      "foot_18_23": 0,
      "hand_left_92_112": 0,
      "hand_right_113_133": 0
    },
    "rgb_vs_manifest_canvas_mismatch": 0,
    "rows": 461,
    "rows_with_any_out_of_range_slot": 0,
    "worst_3": []
  }
}
```

判据边界要交代清楚：适配器拒绝的是"归一化后越出 [-0.02, 1.02]"的点（fail closed，宁可不产条件也不硬夹坐标，这是方案第二条约束的口径）。上表把越界点按部位分组、并把 `candidate` 像素范围与 RGB 尺寸、manifest 声明的 `*_canvas` 三方对齐：
* 若 `rgb_vs_manifest_canvas_mismatch` 或两组尺寸不一致 ⇒ 是我读错了画布，属仪器问题，下一轮改判据；
* 若一致而越界点集中在 `foot_*`/`hand_*` ⇒ 是这些帧本身肢体出画（方案里"人物严重截断直接拒绝"要处理的形态），属数据事实。
**本轮不替这两种解释选一个**——`worst_3` 与分组计数已留在上面，判级留到下一轮按表复算。

### 28.4 撤回 28.3 里我自己算坏的那一列，并记下一处真缺陷

**撤回**：`rgb_vs_manifest_canvas_mismatch`（train 189 / test 70）**读数无效**。
它比的是 RGB 实测尺寸 1080x1920 与 manifest 声明的 `*_canvas` = [(0, 0)] ——
后者是 `[(0, 0)]`，也就是说 pairs manifest 里 `source_canvas` / `target_canvas` **没有实值**。
拿一个零尺寸去比真实尺寸，每一行都必然"不一致"，这条计数什么都没测。判据不成立时不许放宽它，只能作废并写清原因。

**由此暴露的真缺陷（属我自己）**：方案第六节明确要求 manifest 带 `source_canvas` / `target_canvas`（原生宽高比与坐标变换）。
现在这些字段是空壳 ⇒ Stage 7 的 Canvas 契约在这一列上**并未真的被数据填实**，只是被覆盖了一层占位值。
修法：从 RGB 实测尺寸回填 canvas 并同步 offset/scale 定义，再重跑 stage33/34；回填前 `pairs_hand/*` 只能当作"路径与字段名已对齐"的证据，
不能当作"坐标变换已核对"的证据。

**同时成立的数据事实**（与上面的缺陷是两件事，不互相抵消）：越界点集中在身体主链，
train 越界槽 189 个全部含 body_0_17 出画、foot_18_23 有 66 个、手部 82 个；
test 越界槽 70 个含 body_0_17 出画，而 face_24_91 在 train/test 均为 0、hand 在 test 为 0。
即这些帧的人体关键点确实落在 RGB 画布之外（截断/外推），而人脸仍在框内 —— 这正是方案里"人物严重截断直接拒绝"要处理的形态。
train 因此掉 130 对、test 掉全部 35 对，**test 掉得更多不是 bug**，是它只有 35 对、基数太小（18/35 行含出画槽 vs train 18/862）。
下一轮在回填 canvas 之后重算这两档掉队率，再谈分级。

## 30. 撤回 §28.4 那条"canvas 是空壳"：又是我自己的键名错（stage 35 前置核查）

**作废**：§28.4 写的「pairs manifest 里 `source_canvas` / `target_canvas` **没有实值**、`[[0, 0]]`」不成立，
连同 §28.3 里的 `distinct_manifest_canvas: [[0, 0]]` 一并作废。
真实情况是这条字段**一直是填好的**，只是我用错了键名：我读 `c["width"]/c["height"]`，
而本 schema 用的是 `original_hw` / `resized_hw` / `canvas_hw` / `offset_xy` / `policy` / `stride` / `version`。
读不存在的键得到 `None`，我就把它写成了"空壳"—— 与 §27.1 那三条同型（按不存在的字段名读数），
而且 §28.4 里我已经作废过的 `rgb_vs_manifest_canvas_mismatch` 那一列，作废**理由**也错了：
真正的理由是键名，不是数据缺字段。两条一起留在这里，不静默改掉。

实测到的 schema 与取值分布（train 全量 862 行，862 行的 `source_canvas.original_hw` 非空）：

```json
{
  "canvas_keys": [
    "canvas_hw",
    "offset_xy",
    "original_hw",
    "policy",
    "resized_hw",
    "source_video",
    "stride",
    "token_hw",
    "valid_rect",
    "version"
  ],
  "distinct_canvas_hw_first200": [
    [
      1920,
      1088
    ]
  ],
  "distinct_original_hw": [
    [
      1920,
      1080
    ]
  ],
  "policies": [
    "native_pad"
  ],
  "row0_source_canvas": {
    "canvas_hw": [
      1920,
      1088
    ],
    "offset_xy": [
      0,
      0
    ],
    "original_hw": [
      1920,
      1080
    ],
    "policy": "native_pad",
    "resized_hw": [
      1920,
      1080
    ],
    "source_video": "frames/000000/rgb/00024.png",
    "stride": 32,
    "token_hw": [
      60,
      34
    ],
    "valid_rect": [
      0,
      0,
      1080,
      1920
    ],
    "version": "canvas_v2"
  },
  "rows_out_of_range_by_orientation_reading": {
    "hw_as_height_width": 117,
    "hw_as_width_height": 862
  }
}
```

**仍然成立的部分**（不因这次撤回而改动）：
* stage32/33 的归一化用的是 cv2 实测 `(w, h) = (1080, 1920)`，与 `original_hw = [1920, 1080]` 按 (h, w) 读一致
  —— 上面 `rows_out_of_range_by_orientation_reading` 给出两种读法各自的越界行数，用来定这个字段到底是 h,w 还是 w,h；
* 部分帧的**身体主链**关键点确实落在原始画布之外（§28.3 的分组计数），这属于"人物截断"要处理的形态，与本次键名错无关；
* 需要重新表述的只有一件事：越界要拿去跟 `canvas_hw`（带 pad 的那一档）而不是跟"没有画布"比，
  下一轮的正确判据是「同一个点分别在 `original_hw` 与 `canvas_hw` 两种基准下是否越界」，两者都越界才是真截断。

**下一轮要做的**（本轮预算已到，不留半成品）：把 stage34 的 `canvas_of()` 改成读 `original_hw`/`canvas_hw`，
重跑一遍越界分类，并把 §28.3 的 test 档结论按新基准重判。§29 里"回填 canvas"这件事本身不再需要——字段早就有；
要修的是**读它的代码**，包括我自己那份。
