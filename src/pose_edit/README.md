# src/pose_edit/ —— 两条 workflow 线共用的推理侧共享代码层

仓库级共享层，沉淀单人线（`workflow/`）与双人线（`workflow-multiple/`）姿态编辑流水线的**建筑块**。它刻意不关心"一张图里有几个人"：缓存指纹规则、画布预处理、SMPLest-X 估计器调用封装、DWPose-134 关节表、外观相似度、提示词契约都只写一份，两边 import 同一份，避免两条线各自漂移出两份实现（历史上 133/134 关节槽错位就是两份副本各自维护造成的）。2026-10-13 把原来散在两条线里的底层提炼到这里；权威描述见 `workflow/README.md` 的「source_torso 改用实测 A 躯干」与「缓存怎么判定」两节。

## 里面有什么

模块共 7 个，逐个职责如下（文件名即职责）：

- `__init__.py` —— 包说明：声明本层是单/双人 workflow 的共享建筑块，不感知人数。
- `common.py` —— 阶段溯源、缓存指纹、画布预处理、估计器沙箱。每个阶段请求拆成两半：`identity`（设置、代码哈希、模型身份、输入内容 SHA256，不含绝对路径，可跨机器复用）与 `runtime`（本 run 的读写绝对路径，改名/搬家不失效）；阶段产物以相对 run 目录的路径+SHA256 记录。另含 `canvas_size`/`letterbox` 留边预处理与 `create_estimator_sandbox`（只可读共享估计器、可写权重走软链）。
- `estimator.py` —— 本流水线**唯一**调用 vendored SMPLest-X 估计器的入口。几何阶段与评测阶段都走它：`estimate_single` 在 run 本地沙箱里估一张图并做参数校验，`reconstruct`/`project` 做相机重建与投影，`REQUIRED_PARAMETER_KEYS` 等常量集中在此。
- `geometry.py` —— 纯 CPU 的 SMPL-X 姿态向导几何：`align_to_source_bbox(points, bbox_B, bbox_A, framing, a_torso_px=None)` 把骨架缩放进源人物身框，支持 `source_person`（B 框内切进 A 框）与 `source_torso`（B 实测躯干匹配 A 实测躯干）两种定标；`choose_subject` 按面积优先的确定性排序选主体（`auto` 拒绝猜多人）；`project_camera`、`smplx137_to_dwpose134`、`cocowholebody133_to_dwpose134`、`torso_length`、`render_poseguide`、`valid_points` 等关节表与渲染工具。
- `dwpose.py` —— DWPose ONNX 调用约定集中地：会话走 CPU、本版 `inference_detector` 只回 xyxy 且自带 0.3 阈、模型输出 133 点转本仓库 134 点（`cocowholebody133_to_dwpose134`）、1 号 neck 为派生点取其分数为两肩较弱者、按 IoU 而非检测顺序选主体（`box_iou`/`pick_person`/`detect_people`/`measure_torso`）。
- `appearance.py` —— 逐人外观相似度（DINOv2 特征，非生物识别）：`crop_person`（tight/square 两种裁法）、`appearance_matrix`/`best_assignment`/`judge_import`/`judge_appearance`，当外观最优配对与姿态最优配对不一致即身份交换信号，一致但分数低则是外观漂移。
- `prompt_contract.py` —— 与人数无关的提示词契约：`parse_pe_answer`（解析官方三段式 JSON 答案、容错 thinking/fence、强制 `<image1>` 即源画布）、`referenced_images`、`validate_generator_prompt`（生成器只允许指向它实际收到的图像）。

## 与上下层的关系

- **`workflow/` 与 `workflow-multiple/`**：两条线都 `import src.pose_edit.*`；共享层的代码属于两条线所有阶段的 `SHARED_CODE` 指纹清单，因此这里任何改动都会让两边已有 run 的阶段缓存失效、复验时报 `workflow code changed`（产物与 `stage.json` 原样保留、记录当时是谁算的，不 `--force` 不重算）。
- **`src/pose_control/`**：那边是**训练侧**适配器（DeepGen V6 / Qwen-Image-2.1 的简化与全量、dual），本层是**推理侧**流水线共享层；二者互不依赖，只在"训练产物 vs 推理产物"两种语境下分别出现。
- **`workflow/README.md`**：本层的行为（尤其 `source_torso` 实测化、`dwpose.py` 约定合成）以它的记载为权威来源。

## 注意与坑

- **`source_torso` 已改用实测 A 躯干（2026-10-14）**：`geometry.py` 的 `TORSO_TO_PERSON_BOX_HEIGHT = 0.31` 现在**仅作兜底**，原注释"5 对上还原到约 4%"不成立——单人线 8 张源帧躯干/框高从 0.259 跨到 0.524。只有拿不到 A 的 neck/hip 关键点时才退回常数，并在产物里写明走了 `measured_A` 还是 `constant_fallback`。
- **失败要响，不静默退回**：`source_torso` 下若 DWPose 在 A 中找不到与主体框重叠的人，`smplx_worker` 直接抛 `RuntimeError`，**不**悄悄退回常数——那会不留记录地改掉这条 run 的迁移尺度。
- **指纹代价**：`dwpose.py` 已加入单人线几何阶段与双人线评测的代码清单；当 `pose_framing == source_torso` 时，几何指纹额外含两个 DWPose 权重的 `model_signature`（权重变了指纹就变），其余 framing 不付哈希成本。
- 关节映射语义以安装版 SMPLest-X 的 `joints_name`/`flip_pairs` 为权威，测试里直接解析它们而非重抄常量表；133→134 的 neck 派生点规则一旦改动需同步 `dwpose.py` 与 `geometry.py` 两处。
