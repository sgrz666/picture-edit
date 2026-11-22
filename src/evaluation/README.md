# src/evaluation/ —— SDPose 姿态评分、身份评分和无 GT 指标

`evaluation/` 实现项目关键的量化评测。四臂评测的数字（如 Qwen21+控制 PSNR 25.91 > DeepGen+控制 24.54，PCK@0.02 0.979）来自这里跑出的报告；此外还承载 workflow 的**无 GT 六轴补测**（被 `workflow/evaluate2.py` 调用）与**手部结构性合理性**判断（回应合同「手部无明显结构性错误」），并封装 DWPose-134 的 ONNX 推理。

## 里面有什么

- `sdpose.py`：共享 SDPose-Wholebody + YOLO11x、固定评分随机种子、模型 SHA 和 geometry 134 槽位转换。
- `joint_pilot_metrics.py`：`PoseMetric` 将 SDPose 结果映射为 COCO17，按原生分辨率与 Body25 对照。`IdentityMetric` 用 InsightFace/glintr100 计算身份余弦。
- `no_gt_metrics.py` —— workflow 的**无 GT 评测指标集**（被 `workflow/evaluate2.py` 调用，`CODE` 指纹含本文件）。在不依赖编辑参考图的前提下回答两件事——姿态是否落到（用一个架构不同的第二估计器 SDPose 复读 guide-vs-output，含 out-of-frame 记账与 strict PCK 变体）、以及「该保持的是否保持」（身份/体型/背景/其他人物）。关键能力：
  - `estimator_agreement` / `pose_error_stats` —— 第二读者 SDPose 一致率与 strict PCK（把被丢弃的关节也算作 miss）；
  - `limb_ratios` / `ratio_drift` —— 骨长比漂移，分 `invariant`（同肢相邻骨比，投影下稳健）与 `view_sensitive`（跨身体宽度/躯干，仅作诊断、不进 headline）两族；
  - `region_metrics` —— 背景/人体区域 MAE/PSNR/SSIM，配合 `mask_from_mesh`/`mask_from_limbs`/`dilate` 生成掩码；
  - `beta_distance` —— SMPL betas 体型距离；
  - `nonsubject_regions` —— 非主体人物（框 IoU 判定）的改动检测。
  重度模型（ONNX/InsightFace）懒加载，无权重也能跑数学与单测；只用 `src.pose_edit.geometry` 的 DWPose-134 约定；不静默清零（`num()` 把 NaN/inf 转 None，`common.atomic_json` 拒绝 NaN）。
- `hand_plausibility.py` —— 仅用关键点的**手指结构性合理性**判断（回应合同「手部无明显结构性错误」）：读 `evaluate.py`/`evaluate2.py` 已写出的 DWPose-134 数组，**不额外推理**。按 COCO-WholeBody 21 槽手序（0 腕、thumb 1-4、index 5-8、middle 9-12、ring 13-16、pinky 17-20），`analyze_hand` 检查手指长度比（wrist→tip / palm，区间 0.45–3.20）与指尖融合（`COLLAPSED_TIP_DISTANCE`）；手掌过小时 `hand_too_small_to_judge` 诚实标不可判（`MIN_PALM_PX=60`，约 40px 手掌上检测器噪声已占一个指节），不硬打分。属「筛查而非解剖」：抓手套/手指拉伸等明显形变并标记 `inspect`，非门禁。
- `dwpose_onnx/` —— DWPose-134 的 ONNX 推理封装：`onnxdet.py`（`inference_detector`）、`onnxpose.py`（`inference_pose`），附带 LICENSE 与 README。
- `__init__.py` —— 空。

## 与上下层的关系

- 上游：推理产出的图像、GT、数据集元数据；第二估计器 SDPose、ArcFace（`glintr100.onnx`，可选）。
- 下游：`no_gt_metrics` 被 `workflow/evaluate2.py` 调用，产出 `evaluation/eval2_sdpose.json` 与 `stage2_sdpose.json`；`hand_plausibility` 被 `tests/test_hand_plausibility.py` 验证、设计上补测 `evaluate.py` 写出的手部关键点数组；`joint_pilot_metrics` 是四臂 CPU 评测器（`PoseMetric`/`IdentityMetric`），供四臂报告/图像辅助与 `tests/` 消费；`dwpose_onnx` 用于已有生成条件或明确标注的历史对照，包括旧 Union 评测和 Bilibili Edit LoRA 快照。

## 注意与坑

- 新评分用 SDPose 解释器，权重路径、置信度和输入契约见 [SDPose 规范](../../docs/SDPOSE_POSE_EVAL_SPEC.md)。
- `no_gt_metrics` 的 `region_metrics` 默认把背景与人体（由掩码界定）分开打分；保护区域若误把人体纳入，会虚高 scene 分数——`tests/test_no_gt_metrics.py` 专门守住「保护区域不误含人体」这条契约。
- 身份余弦对「姿态控制导致源脸崩塌」敏感（已知 0.822→0.353），评测时区分「无控制」与「有控制」两组口径；`no_gt_metrics` 的 ArcFace 随 yaw/遮挡衰减，仅同口径可比。
- `hand_plausibility` 不查径向单调与逐段长度带（卷曲手指、透视缩短均合法），只在 `palm_px >= MIN_PALM_PX` 的手上作判定，引用其 `inspect` 标记前先确认手已被放大到可测。
