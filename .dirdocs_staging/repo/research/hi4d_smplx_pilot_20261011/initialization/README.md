# research/hi4d_smplx_pilot_20261011/initialization/ —— 流水线第二级：Multi-HMR 逐视角 SMPL-X 初始化

> 工作区路径：`/home/shangguanrz/project/pic-edit/research/hi4d_smplx_pilot_20261011/initialization`　|　本级：叶子目录　|　内容：16 个 NPZ（2 帧 × 8 相机，约 75~153 KB/个）+ `metadata.json`，共 2.3 MB　|　由 `../initialize_multihmr.py` 产出

对 `../inputs/` 的 16 张原图各跑一次 **Multi-HMR**（模型 `multiHMR_896_L`，权重 SHA `c1b55933…202ca`，torch 2.8.0+cu128，总耗时 12.4 s），得到逐人物的 SMPL-X 初值。相机内参用 Hi4D 标定 K 按实际 resize/padding 变换到模型画布（`metadata.json` 的 `camera_source` 字段），不是旧测试里假设的 FOV 60。

命名 `talk14_000032_cam{4..88}.npz` / `hug14_000062_cam{…}.npz`。实测键布局（以 `hug14_000062_cam16.npz` 为例）：`K_canvas/canvas_affine/K_native/extrinsics` + 每人一组 `person_{0,1}_{loc,transl,transl_pelvis,rotvec,expression,shape,v3d,j3d,j2d,scores}` + `actor_assignment`、`assignment_error_px`。`metadata.json` 逐记录存 `person_count`、A/B 槽位分配与投影误差（talk 帧多数视角 4.6~8.5 px）。注意 `hug14_000062_cam4.npz`（76,743 B）明显小于其他（约 150 KB）：cam4 的旧 DWPose 绑定偏差大（父 README：加权偏差 76.68/47.18 px），该视角人物 1 槽位质量差，拟合时靠观测剔除。

下游：本目录 + `../observations/` 是 `../fit_v1/` 的两大输入；重新生成本目录需要 `add/envs/multihmr/bin/python` 环境。当前帧清单固定为两帧，扩展其他 Hi4D 帧前要先改 `--workdir` 逻辑并重新验证。
