# research/hi4d_smplx_pilot_20261011/observations/ —— 流水线第三级：多视角 DWPose 二维观测与三角化

> 工作区路径：`/home/shangguanrz/project/pic-edit/research/hi4d_smplx_pilot_20261011/observations`　|　本级：叶子目录　|　内容：18 个 NPZ + `metadata.json`，共 101 KB　|　由 `prepare_dwpose.py` 产出

在 SMPL 投影框出的人物区域内跑现有 DWPose 模型得到逐相机二维全身点，再用相机标定做鲁棒三角化。文件两类：

- 逐视角二维：`{talk14_000032,hug14_000062}_cam{4..88}.npz`（16 个，约 3~4 KB）——`xy_native [134,2]` 加置信度。
- 三角化合集：`talk14_000032_triangulated.npz`、`hug14_000062_triangulated.npz`。实测键：`xyz_world [2,134,3]`、`valid [2,134]`、`median_reprojection_px [2,134]`、`inlier_views [2,134]`、`xy_native [8,2,134,2]`、`scores [8,2,134]`、`camera_ids [8]`（8 相机 × 2 人 × 134 点）。

`metadata.json` 记录有效三维点数：talk14/32 两人各 117 个有效点、双手 21/21 全部达到三视角条件；**hug14/62 急剧退化**——A 仅 90 点、左手 18/0，B 仅 65 点、双手 0 点（`face_contour_3d` 恒标 `invalid: view dependent`）。这就是父 README "拥抱帧手指主要靠网络先验、不能当手势真值"的原始出处。关键点排列沿用 DWPose 的 134 点定义（body18/feet6/face68/左手21/右手21）。

下游：与 `../initialization/` 一起作为 `../fit_v1/` 拟合的监督；三维可靠条件为 `observed_multiview & (inlier_views >= 3)`。本目录独立于旧 Hi4D 二维标注文件，未修改任何原始 `dwpose_A/B.npz`。
