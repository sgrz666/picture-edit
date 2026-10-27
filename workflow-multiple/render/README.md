# ABC 共相机几何渲染

`abc_scene.py` 接收重定向后的两个 SMPL-X 网格，复用仓库 `IndependentMeshRasterizer` 的 nvdiffrast 真三角形渲染；相机使用 CV 坐标，深度单位为米。A/B 分别保存完整 mask，`visible_owner` 再按共享 z-buffer 决定可见归属，因此遮挡不会把另一个人物的完整几何抹掉。

入口为 `render_scene(people, K, canvas_hw, out_dir)`。`people` 按逻辑 A/B 排列；每人提供 `vertices_cam`、`faces`、零起始 `part_face` 或 `part_vertex`、完整 `joints_cam`、`joints_137_cam`（若前者已是 SMPLest-X 137 顺序可省略），以及 `body_pose[63]`、`global_orient[3]`、`transl[3]`、`betas[10]`。画布高宽必须是 32 的倍数；输出目录中已有同名产物时拒绝覆盖。

输出 `geometry_arrays.npz` 包含两人法线 `[2,H,W,3]`、米制深度、part（0 背景、1..14 身体部位）、mask、可见归属及参数。`person_0_*`、`person_1_*`、`combined_*` 是便于检查的 PNG；三个 `*_guide.png` 均由同一相机投影实际 137 关节，再转 DWPose-134，未独立缩放。深度 PNG 共用同一显示范围，精确数值见 NPZ。

只有传入已验证的 `contact_indices[N,4]`（人物、顶点、人物、顶点）才会生成接触栅格和目标端点距离；缺失时记录 `contact_status=unavailable`，不把空白图冒充负标签。该模块不估计接触、不以最近顶点替代证据。上游是 `retarget/`；下游由 ABC geometry worker 将渲染路径写入 `scene.json`，Native 条件构建读取 NPZ。

CPU 契约测试：`python -m pytest workflow-multiple/tests/test_abc_geometry_render.py -q`。真实渲染需在已有 nvdiffrast CUDA 环境内运行 geometry worker；本模块不安装依赖。
