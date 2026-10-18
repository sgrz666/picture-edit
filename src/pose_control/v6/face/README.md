# src/pose_control/v6/face/ —— V6.5 人脸增强分支

`face/` 实现 V6.5 引入的面部增强：在目标头部 ROI 内注入源人物 A 的身份与纹理（ArcFace / DINO），而不引入目标 RGB 或目标身份，从而缓解姿态控制导致的源脸崩塌（已知现象：姿态控制使源脸余弦 0.822→0.353）。

## 里面有什么

- `conditions.py` —— `FaceFineCondition`、`FaceReferenceFeatures`、`PreparedFaceConditioning` 及索引/复制/扩展工具。
- `geometry.py` —— 关键点映射 `smplx137_to_face72`（取 SMPL-X 索引 65..136）、`dwpose68_to_face72`（DWPose 68 → SMPL-X 对齐 72 槽）、`FaceGeometryTokenizer`、`normalize_face_landmarks_to_roi`。
- `content_encoder.py` —— `FaceContentEncoder`：编码多参考身份与 DINO 外观；`pool_arcface_references`（FP32 归一 + 有效均值 + 再归一）；依赖 `third_party.v65_face.stableanimator` 的 `FacePerceiver`/`FusionFaceId`。
- `spatial.py` —— `FaceSpatialEncoder`、`render_face_landmark_heatmaps`（置信加权高斯 + 软并集 mask）；依赖 `third_party.v65_face.mcld` 的 `FacePoseGuider2D`。
- `adapter.py` —— `FaceControlAdapter` 与 `scatter_face_roi_residuals`：把 ROI 残差逆 warp 回目标 token 网格并混合重叠区（不放大强度）。
- `preparer.py` —— `FaceConditioningPreparer`：组装静态脸几何、身份、纹理、mask、ROI 绑定。
- `resampler.py` —— 再导出锁定的 Visual Persona `Resampler`/`PerceiverAttention`/`FeedForward`。
- `iper_dataset.py` —— V6.5 在 iPER 上的过拟合数据集读取（含 `smplx_global` 形状校验 `[N,26]`）。
- `__init__.py` —— 导出 `FaceControlAdapter`、`FaceFineCondition`、`FaceReferenceFeatures`、`FaceConditioningPreparer` 等。

## 与上下层的关系

- 上游：data 层提供的脸关键点、ArcFace/DINO 参考、源裁剪。
- 下游：被 `v6/deepgen_adapter.py` 经 `face_adapter` / `face_preparer` 引用；残差经 `control_core` 注入，并在 `interface/strength.py` 的 `add_optional_face_residual` 处叠加。
- 同构迁移版见 [../../qwen21_full/face/README.md](../../qwen21_full/face/README.md)。

## 注意与坑

- `FaceControlAdapter` 仅把身份/纹理加到**目标头部 ROI 内**，源裁剪与 ArcFace 属于 A；务必保证不把目标 RGB 或目标身份送进该模块，否则破坏身份保持。
- ArcFace 特征维度固定 512，传入 `pool_arcface_references` 的张量须为 `[...,R,512]`。
- `smplx_global` 为 26 维且不含关节姿态，这里只用于全局朝向/形状条件，与脸部细节无关。
