# src/pose_control/v6/hand/ —— V6.6 手部控制分支

`hand/` 实现 V6.6 引入的手部重定向：只传目标手部局部几何（关键点/姿势/网格 BPS/深度/可见性），不依赖目标帧 RGB，缓解 VAE 8× 下采样带来的手细节硬瓶颈。

## 里面有什么

- `conditions.py` —— `HandFineCondition`（继承 `detail/hand.HandDetailCondition`，加 `mesh_bps`/`mesh_valid`/`depth`/`visibility`）、`HandReferenceFeatures`、`PreparedHandConditioning`，及 `from_legacy`。
- `geometry.py` —— `split_iper_134_hand_points`（iPER/SMPL-X 全身布局左 92:113、右 113:134）、`axis_angle45_to_6d96`（VTON 的 16 个旋转：canonical 手腕 + 15 个 SMPL-X 手指关节 → 6D）、`local_hand_points`、`hand_heatmaps`。
- `preparer.py` —— `HandConditioningPreparer`：独立几何/外观编码（无目标帧 RGB 泄漏），含 DINO 投影、`HandResampler`、`source/person/side/region` 嵌入与 `null_appearance`。
- `adapter.py` —— `scatter_hand_roi_residuals`（逆 warp 4 个手部 ROI；深度/可见性加权重叠不翻倍）、`IndependentKVAttention`（VTON 分离 IP K/V 的动态 SDPA 等价）。
- `vton_layers.py` —— 改编自 VTON-HandFit 的 `HandResampler`/`HandStructureProjector`/`OneLinearLN`/`FeatureEmbedderLN`，维度适配 DeepGen 512 宽手流；保留投影拓扑与潜在查询注意力。
- `quality.py` —— 离线用的 `hand_quality_gate`（触发 2D 精修）与 `merge_vitpose_keypoints`（以高置信 ViTPose 替换 SMPL-X 手）。
- `refiner_export.py` —— `hand_refiner_payload`：给离线修复工具的 RGB/mask/depth/ROI 导出，默认生成路径不执行。
- `__init__.py` —— 导出 `HandControlAdapter`、`HandFineCondition`、`HandReferenceFeatures`、`HandConditioningPreparer`。

## 与上下层的关系

- 上游：data 层提供的手部关键点、SMPL-X/MANO 姿势、参考 DINO patch。
- 下游：被 `v6/deepgen_adapter.py` 经 `hand_adapter` / `hand_preparer` 引用；残差经 `control_core` 注入。
- 同构迁移版见 [../../qwen21_full/hand/README.md](../../qwen21_full/hand/README.md)。

## 注意与坑

- `HandFineCondition` 明确"推理时不需要目标 RGB"；`quality.py`/`refiner_export.py` 仅离线辅助，默认生成流程不调用。
- 手部坐标基于 bounding box 归一化（`local_hand_points`），送进 preparer 的 box 必须与关键点同坐标系。
- `vton_layers.py` 明确不引入任何 UNet 或 CUDA 专属操作，便于在 DeepGen 流中复用。
