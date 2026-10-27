# A+B+C 几何重定向

`core.py` 是不依赖 GPU 的 full53 参数解码、A/B 左右角色、身份与动作参数合并、双人共同平移和有限接触修正。`backend.py` 使用仓库 SMPLest-X 的 neutral SMPL-X 模型恢复真实网格与 137 点。

源 A/B 保留各自 `betas`、`expression`、`jaw_pose`；C 提供 root/body/双手与双人相对布局。Multi-HMR 的 53 个旋转不包含 eye 槽，双手输入从 flat mean 转为 SMPLest-X 的 hand mean。相机使用 A 的 CanvasSpec，只对两人施加同一个相机平移以完整入画，身体尺寸始终不缩放。

A/B/C 都先按 EXIF 方向归一化，另存当前 run 的 `original.png`；`image` 指向该归一化图，`original` 记录输入文件。`interaction_K` 若有提供，必须使用归一化 C 的像素坐标。下游读取 source 的 `image` 才与其 K、mask、CanvasSpec 一致。

接触候选只能来自 C 中小于阈值的已测顶点对；修正每人的平移最多 4cm，保留 C 对应点的原有偏移，最多执行一次闭式优化。没有可靠候选时写明 uncertain，不强制吸合。距离是顶点对应证据，不冒充 FCL 三角面距离或穿透测量。

CPU 验证：`python -m pytest workflow-multiple/tests/test_abc_geometry_retarget.py workflow-multiple/tests/test_abc_estimate_contract.py -q`。入口为 `abc_geometry_worker.py --request <JSON>`；所有产物落当前 run 的 `geometry/`，上游是独立照片 A/B 和交互照片 C，下游是 Qwen 与 Native 条件构建。
