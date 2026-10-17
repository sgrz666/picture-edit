# src/pose_control/qwen21_full/hand/ —— V6.6 手部控制分支（全量迁移副本）

本目录是从 `v6/hand/` **原样迁移（逐文件字节一致）** 到 qwen21_full 的副本，模块职责与 V6 完全一致——只传目标手部局部几何（关键点/姿势/网格 BPS/深度/可见性），不依赖目标帧 RGB。迁移到 Qwen-Image-2.1 后仅在残差输出端适配 Qwen 目标流，本目录的几何/外观编码逻辑未改。注意：在 `execution.py` 中 hand 默认 `hand_strength=0, hand_mode='off'`，即该分支默认关闭。

## 里面有什么

- `conditions.py` —— `HandFineCondition`（继承 `detail/hand.HandDetailCondition`，加 `mesh_bps`/`mesh_valid`/`depth`/`visibility`）、`HandReferenceFeatures`、`PreparedHandConditioning`，及 `from_legacy`。
- `geometry.py` —— `split_iper_134_hand_points`、`axis_angle45_to_6d96`、`local_hand_points`、`hand_heatmaps`。
- `preparer.py` —— `HandConditioningPreparer`：独立几何/外观编码（无目标帧 RGB 泄漏），含 DINO 投影、`HandResampler`、各类嵌入与 `null_appearance`。
- `adapter.py` —— `scatter_hand_roi_residuals`（逆 warp 4 个手部 ROI；深度/可见性加权重叠不翻倍）、`IndependentKVAttention`。
- `vton_layers.py` —— 改编自 VTON-HandFit 的 `HandResampler`/`HandStructureProjector`/`OneLinearLN`/`FeatureEmbedderLN`，维度适配 DeepGen 512 宽手流。
- `quality.py` —— 离线用的 `hand_quality_gate` 与 `merge_vitpose_keypoints`。
- `refiner_export.py` —— `hand_refiner_payload`：离线修复工具的 RGB/mask/depth/ROI 导出。
- `__init__.py` —— 导出 `HandControlAdapter`、`HandFineCondition`、`HandReferenceFeatures`、`HandConditioningPreparer`。

## 与上下层的关系

- 上游：data 层/ `../data_runtime.py` 提供的手部关键点、SMPL-X/MANO 姿势、参考 DINO patch。
- 下游：被 `../deepgen_adapter.py` 经 `hand_adapter` / `hand_preparer` 引用；残差经 `../control_core` 注入 Qwen 的 32 层。
- 源目录见 [../../v6/hand/README.md](../../v6/hand/README.md)；上层见 [../../qwen21_full/README.md](../../qwen21_full/README.md)。

## 注意与坑

- `HandFineCondition` 明确"推理时不需要目标 RGB"；`quality.py`/`refiner_export.py` 仅离线辅助，默认生成流程不调用。
- 手部坐标基于 bounding box 归一化（`local_hand_points`），送进 preparer 的 box 必须与关键点同坐标系。
- 本目录与 `v6/hand/` 逐文件一致；改通用手部逻辑要确认改的是哪一份。
