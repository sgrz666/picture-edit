# Sapiens2-Pose-5B / DWPose / Multi-HMR / SMPLest-X：双人姿态估计对照（2026-10-29）

## 1. 实验目的

同一批 Hi4D 双人帧上，把 **Meta Sapiens2-Pose-5B**、**现有 DWPose-134**、**Multi-HMR**、**SMPLest-X→3D 投影**（即生产链路的"3D DWPose"骨架来源）放在同一把尺上比，回答两件事：Sapiens2 在双人条件下到底强在哪一段（找全两个人 / 给定人物框后的关节点精度 / 3D），以及能不能进生产链路。

## 2. 里面有什么

- `data/request.json`（1.1 MB）：**冻结的 127 张不同原图**。79 张来自 `mete-test` 旧集的去重（旧集 80 行实为 79 张：`pair02/backhug02/64/000016` 同时被标成 hug 与 severe，本次取 severe），48 张为新增，按场景各 12 张、确定性地按图像哈希排序选取，且与同演员同动作已选帧至少隔 15 帧。每条记录含图/哈希、真值框、`gt_xy`（Hi4D SMPL 24 关节经该帧内外参投影到像素）、`gt_xyz`、`gt_pelvis`、逐关节在框内有效性、以及 `occlusion_proxy`（按分人掩膜归属标注 0=本人独占 / 1=仅另一人掩膜 / -1=两人掩膜都覆盖或出界）。**帧池 `frames_1500.jsonl` 跨 Hi4D 的 train/val/test（本次 77/26/24），被评测的四个模型都没有用 Hi4D 训练，所以这里不影响结论；但若将来有模型在 Hi4D 上训练，不能再拿这份清单当测试集。**
- `scripts/benchmark.py`：唯一量尺。12 个**具名**身体关节；逐人用本人 GT 框对角线归一；无序误差最小匹配（不是按检测顺序，也不是身份追踪）；**没检到的人照样进分母**（端到端 PCK）；二维与三维分开，无 3D 输出的模型三维列写 `null` 而不是 0。
- `scripts/prepare.py`：从 `frames_1500.jsonl` 生成上面的请求，含去重、抽样、真值投影与掩膜归属。
- `scripts/infer_2d.py`：`dwpose_auto / dwpose_gtbox / sapiens_yolox / sapiens_detr070 / sapiens_gtbox` 五臂；一次加载、逐帧独占写、失败帧记错误但不静默跳过。
- `scripts/infer_3d.py`：`multihmr / multihmr_fov60 / smplest_gtbox / smplest_yolox` 四臂；输出相机坐标关节、pelvis、逐人虚拟 K、峰值显存与稳态耗时。
- `scripts/nms_sweep.py`：用已落盘的 DETR 原始框离线扫 NMS 阈值，回答"0.7 是不是挑出来的刀锋"。
- `scripts/run_gpu_arms.sh`：GPU 各臂串行编排（同一张卡不并发，否则耗时与显存不可比）。
- `tests/`：量尺、抽样、NMS 扫描的单元测试（本地与远端各跑一遍，全绿）。
- `summary_compact.json` / `nms_sweep.json`：本次结论的全部数字来源。逐帧明细（含每点距离、置信度）在 `sg:/data/shangguanrz/pic-edit-experiments/mete-test/dual_pose_compare_20261029_v1/report/final_v1/results.json`（18 MB，不入库）。

## 3. 关键结论

### 3.1 先撤回旧实验（`mete-test/`，2026-10-28）的两条结论

1. **关节索引错了 8/12**。旧评测把 Sapiens 308 当成 COCO-17 用（`5..16`）。官方具名表的真实索引是肩 5/6、肘 7/8、**腕 62/41**、髋 9/10、膝 11/12、踝 13/14。同一批缓存只换正确索引，GTBox 平均误差 **87.68 px → 14.28 px**（修正后管线独立重算为 14.34 px / 79 帧）。旧报告的"PCK@0.05 34.7%"不是模型能力，是量错了关节。
2. **"DETR 在拥抱/严重遮挡漏检 60–80%"主要是 NMS 阈值的产物**。旧脚本 DETR 用 `NMS=0.3`。同 79 帧上只把 NMS 改成 0.7：双人齐全 **51 → 73**，端到端 PCK@0.1 **0.816 → 0.952**（23 胜 0 负，配对差 −0.135）。在新 127 帧上扫阈值是单调曲线、不是刀锋：NMS 0.3 / 0.5 / 0.7 / 0.9 的"正好两人"帧数分别为 83 / 102 / 108 / 109，且 0.7 以上只多出 6–10 帧第三框，按分数只留两人即可回收（`capped_two` = 能匹配两人 115/127）。
   顺带一条：旧脚本的 `person_swaps` 数的是**检测列表顺序**，不是身份互换，该列已在本次作废。

### 3.2 二维：找全人 vs 框内精度要分开算（127 帧，端到端 PCK@0.1）

| 臂 | 框来源 | 双人齐全 | 关节覆盖 | 端到端 PCK@0.1 | 中位误差 px | 平均误差 px | 稳态耗时 |
|---|---|---|---|---|---|---|---|
| `multihmr` | 自带点式检测 | **123/127** | 0.984 | **0.969** | 9.0 | 13.1 | 447 ms (GPU) |
| `sapiens_gtbox` | GT 框（特权） | 127/127 | 1.000 | 0.967 | 8.6 | 15.6 | 2.8 s |
| `smplest_gtbox` | GT 框（特权） | 127/127 | 1.000 | 0.971 | 15.8 | 20.9 | 283 ms |
| `dwpose_gtbox` | GT 框（特权） | 127/127 | 1.000 | 0.939 | 11.9 | 22.6 | 119 ms (CPU) |
| `sapiens_detr070` | DETR，NMS 0.7 | 109/127（留两人 115） | 0.957 | 0.938 | 8.3 | 13.6 | 3.6 s |
| `sapiens_yolox` | 我方 YOLOX 框 | 104/127 | 0.913 | 0.904 | 8.2 | 12.3 | 3.9 s |
| `smplest_yolox` | 我方 YOLOX 框 | 104/127 | 0.913 | 0.901 | 15.8 | 19.3 | 261 ms |
| `dwpose_auto` | 我方 YOLOX 框 | 104/127 | 0.913 | 0.878 | 11.5 | 19.1 | 471 ms (CPU, 含检测) |
| `sapiens_detr030_cached` | 旧 DETR，NMS 0.3 | 51/79 | 0.823 | 0.816 | 8.1 | 11.8 | 4.9 s |

配对读数（按演员组做 cluster bootstrap，10 组，95% 区间）：

- **同样一批框，Sapiens2 比 DWPose 准**：`dwpose_auto − sapiens_yolox` = **−0.026** [−0.034, −0.018]，58 输 3 赢；`dwpose_gtbox − sapiens_gtbox` = **−0.028** [−0.044, −0.012]。
- **换更好的检测框比换姿态模型更值钱**：`dwpose_auto − sapiens_detr070` = **−0.060** [−0.088, −0.030]；`sapiens_detr070 − sapiens_gtbox` = −0.029 [−0.058, −0.008]（这是检测的残差代价）。
- **Sapiens2 与 Multi-HMR 在二维上分不出高下**：`sapiens_detr070 − multihmr` = −0.031 [−0.064, **+0.003**]，区间过零。但 Multi-HMR 的二维是自身三维的投影，与 SMPL 真值同属一族，这个二维平手对它是"结构性有利"的项，不能当成感知精度胜出。
- 拥挤场景（`severe`，32 帧）是分水岭：`sapiens_gtbox` 0.884 vs `dwpose_gtbox` 0.823 vs `smplest_gtbox` 0.910；而自动框下 `dwpose_auto` 只剩 0.629（32 帧里只有 12 帧找全两人）。

### 3.3 三维：Sapiens2 没有三维，3D 要拿 SMPLest-X 与 Multi-HMR 比（root 对齐 MPJPE，毫米）

| 臂 | 相机 | 双人齐全 | 3D 覆盖 | root MPJPE | 绝对 MPJPE | 两人髋距向量误差 |
|---|---|---|---|---|---|---|
| `multihmr` | 帧真实内参 | 123/127 | 0.984 | **59.8** | 105.3 | 93.5 |
| `multihmr_fov60` | 猜 60° FOV | 123/127 | 0.984 | 60.0 | 不可比（已置 null） | 不可比 |
| `smplest_gtbox` | 逐人裁剪虚拟相机 | 127/127 | 1.000 | 65.2 | 不可比 | 不可比 |
| `smplest_yolox` | 逐人裁剪虚拟相机 | 104/127 | 0.913 | 63.8 | 不可比 | 不可比 |

分场景 root MPJPE（分离 / 轻交互 / 拥抱 / 严重遮挡）：Multi-HMR 45.7 / 54.7 / 57.2 / **81.3**；SMPLest-X 46.2 / 58.8 / 61.1 / **94.4**。Multi-HMR 全场景略优，优势集中在严重遮挡（−13 mm）。

两条必须一起读的口径：

- **猜相机只毁绝对量，不毁相对量**。60° 猜测臂的 root 对齐误差 60.0 mm 与真内参的 59.8 mm 几乎相同，但绝对平移失去意义。所以要做"两人深度/接触"这类判断，必须给真内参，否则只能报 root 对齐值。
- **SMPLest-X 的逐人虚拟相机不共享**，两人分处不同坐标帧，跨人距离与绝对深度都不可算。表里对 SMPLest 只报 root 对齐；这不影响它的 mesh 与单人形状，但意味着它不能直接回答"两人相距多远"。

### 3.4 这个模型对项目的实际作用

1. **不能进生产链路当姿态源**：本次在手的 Pose 权重只输出二维关键点，没有 3D、深度或接触（点图 pointmap 需要另一套权重与 Open3D，本地没有）；生产链路要的是 SMPLest-X 的三维网格与逐人相机。它替代不了这一步。
2. **可以当"独立的二维标尺"**：框内精度比 DWPose 高 2.6–2.8 个百分点的 PCK@0.1，且中位误差 8.2 px（DWPose 11.5 px、SMPLest 投影 15.8 px）。旧 `workflow-multiple` 里"backhug 时 DWPose 说两人髋距 4 px、几何投影说 72 px"这类争执，用 Sapiens2 做第三方裁决比继续用 DWPose 自证更可靠。
3. **当可见点监督的候选来源要谨慎**：Sapiens2 的置信度过滤抬精度但吃覆盖（`sapiens_detr070` 阈值 0.3/0.5/0.7 时覆盖 0.954/0.944/0.901，精度 0.982/0.986/0.993；DWPose 同阈值覆盖 0.940/0.824/**0.462**），且遮挡本身没有真值，`occlusion_proxy` 只是掩膜归属。要接重投影约束，得先把"哪点真的可见"做成独立判据。
4. **成本与许可**：5B / 实测峰值 20.2–21.6 GiB、双人 2.8–3.6 s/帧（`sapiens_gtbox` 2.8 s、`sapiens_detr070` 3.6 s）；Multi-HMR 0.45 s、SMPLest 0.28 s、DWPose 0.12–0.47 s。Sapiens2 许可为 Meta 自定义条款，明确禁止监控、生物特征处理、deepfake 等用途，若要进任何面向用户的产品链路，需先过一遍许可。

### 3.5 检测比姿态更值得改

旧实验把"双人检不出来"记在 Sapiens2 头上，实际检测是独立模块。同一 127 帧上：YOLOX（我方，NMS 0.45）104/127，DETR（NMS 0.7）能匹配两人 115/127，Multi-HMR 自带点式检测 123/127。生产链路目前用的是 YOLOX，这一项有约 11 帧（9%）的现成余量，且不依赖 5B 模型。

## 4. 与其他实验的关系

- 上游：`/data/shangguanrz/pic-edit-experiments/mete-test/`（2026-10-28 的 80 帧双臂实验与 `metrics.csv`）。本目录**不覆盖**它，它的 `report/evaluation_summary.json` 保持原样，撤回写在本文第 3.1 节。
- 与本目录相邻的既有结论：`workflow-multiple/fits/fit_dual_report.json`（Multi-HMR 拟合 vs DWPose，17 对）给出 PCK@0.2 0.902、3D 绝对 MPJPE 159 mm。本次在 127 帧上用真实内参重测得到绝对 105 mm，两者定义不同（关节集合、是否用真 K、样本不同），不可直接比大小，只能都作为"Multi-HMR 三维误差在 10–16 cm 量级"的证据。
- 下游可用：`workflow-multiple` 的 `keypoints_source` 仍是 `dwpose`（拟合骨架只作度量参照）。本次没有改任何生产配置；若要动，先读第 3.4 节的四条限制。

复现（远端）：

```bash
R=/data/shangguanrz/pic-edit-experiments/mete-test/dual_pose_compare_20261029_v1
PY=/home/shangguanrz/miniconda3/envs/deepgen/bin/python
$PY $R/scripts/prepare.py --raw-manifest /home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1/frames_1500.jsonl \
    --legacy-root /data/shangguanrz/pic-edit-experiments/mete-test --run-dir $R   # 已跑过，重跑会因独占写失败
$PY $R/scripts/benchmark.py --request $R/data/request.json import-legacy \
    --legacy-root /data/shangguanrz/pic-edit-experiments/mete-test \
    --meta /data/shangguanrz/models/meta/sapiens2-code/sapiens/pose/configs/_base_/keypoints308.py
bash $R/scripts/run_gpu_arms.sh     # 串行，勿并发
$PY $R/scripts/benchmark.py --request $R/data/request.json evaluate \
    --modes dwpose_auto dwpose_gtbox sapiens_detr070 sapiens_yolox sapiens_gtbox \
            smplest_gtbox smplest_yolox multihmr multihmr_fov60 --report $R/report/final_v2
$PY $R/scripts/nms_sweep.py --request $R/data/request.json --out $R/report/nms_v3
```
