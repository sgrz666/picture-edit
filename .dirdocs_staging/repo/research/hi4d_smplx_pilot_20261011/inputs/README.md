# research/hi4d_smplx_pilot_20261011/inputs/ —— 试点输入：从 Hi4D 原始压缩包按需抽取的 16 帧图

> 工作区路径：`/home/shangguanrz/project/pic-edit/research/hi4d_smplx_pilot_20261011/inputs`　|　本级：1 个子目录 `pair14/`　|　内容：16 张 JPG（2 帧 × 8 相机），子树约 4.8 MB　|　统计时间 2026-10-12

SMPL-X 双人拟合流水线的**第一级（输入）**：没有全量解包 Hi4D，只从 `/data/shangguanrz/pic-edit-datasets/Hi4D-origin/pair14.tar.gz` 按需抽出试点两帧——`pair14/talk14/images/{4,16,28,40,52,64,76,88}/000032.jpg`（源帧）与 `pair14/hug14/images/{同 8 相机}/000062.jpg`（目标帧），共 16 张。抽取清单逐条记录在父目录 `../extraction.json`。

下游关系：本目录是唯一原始像素来源，`../initialization/`（Multi-HMR 逐视角推理）与 DWPose 观测都在这些 JPG 上运行；拟合完成后不再改动本目录。**勿增删图片**，否则与 `../fit_v1/result.json` 里的 SHA 绑定（`smplx_sha256`/`observation_sha256`）断链。各编号相机目录内部布局：`images/<相机编号>/<帧号 6 位>.jpg` 单文件。
