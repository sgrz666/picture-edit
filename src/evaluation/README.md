# src/evaluation/ —— 四臂评测与 DWPose 骨架推理

`evaluation/` 实现项目关键的量化评测：身份保持（InsightFace 余弦）与姿态精度（DWPose-134 / COCO-to-Body25 映射），并封装 DWPose ONNX 推理。四臂评测的数字（如 Qwen21+控制 PSNR 25.91 > DeepGen+控制 24.54，PCK@0.02 0.979）来自这里跑出的报告。

## 里面有什么

- `joint_pilot_metrics.py` —— 从 `eval_iper_four_arm.py` 跟踪复制的四臂 CPU 评测器（保留算法、可见性阈值、512 坐标网格、COCO→Body25 映射）：`PoseMetric`（DWPose detect + compare）、`IdentityMetric`（InsightFace `embedding` + `cosine`）。模型 import 懒加载，路径用仓库 ROOT，便于无 ONNX/InsightFace 时也能跑报告辅助。
- `dwpose_onnx/` —— DWPose-134 的 ONNX 推理封装：`onnxdet.py`（`inference_detector`）、`onnxpose.py`（`inference_pose`），附带 LICENSE 与 README。
- `__init__.py` —— 空。

## 与上下层的关系

- 上游：推理产出的图像、GT、数据集元数据。
- 下游：被四臂评测脚本与报告生成调用；`dwpose_onnx` 被 `joint_pilot_metrics.PoseMetric` 懒加载。

## 注意与坑

- `PoseMetric` 用 CPUExecutionProvider 跑 ONNX，模型路径默认 `models/DWPose`（yolox_l.onnx、dw-ll_ucoco_384.onnx），缺权重时 detect 会失败。
- 身份余弦对"姿态控制导致源脸崩塌"敏感（已知 0.822→0.353），评测时区分"无控制"与"有控制"两组口径。
- `COCO_TO_BODY25` 是固定的 17→25 索引映射，换数据集骨架时需同步核对。
