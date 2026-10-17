# src/pose_control/qwen21_full/face/ —— V6.5 人脸增强分支（全量迁移副本）

本目录是从 `v6/face/` **原样迁移（逐文件字节一致）** 到 qwen21_full 的副本，模块职责与 V6 完全一致——在目标头部 ROI 内注入源人物 A 的身份与纹理（ArcFace / DINO），缓解姿态控制导致的源脸崩塌。迁移到 Qwen-Image-2.1 后仅在残差输出端适配 Qwen 目标流，本目录的几何/外观编码逻辑未改。

## 里面有什么

- `conditions.py` —— `FaceFineCondition`、`FaceReferenceFeatures`、`PreparedFaceConditioning` 及索引/复制/扩展工具。
- `geometry.py` —— 关键点映射 `smplx137_to_face72`、`dwpose68_to_face72`、`FaceGeometryTokenizer`、`normalize_face_landmarks_to_roi`。
- `content_encoder.py` —— `FaceContentEncoder`：`pool_arcface_references`（FP32 归一 + 有效均值 + 再归一）；依赖 `third_party.v65_face.stableanimator`。
- `spatial.py` —— `FaceSpatialEncoder`、`render_face_landmark_heatmaps`；依赖 `third_party.v65_face.mcld`。
- `adapter.py` —— `FaceControlAdapter` 与 `scatter_face_roi_residuals`：ROI 残差逆 warp 回目标 token 网格并混合重叠区。
- `preparer.py` —— `FaceConditioningPreparer`：组装静态脸几何、身份、纹理、mask、ROI 绑定。
- `resampler.py` —— 再导出锁定的 Visual Persona `Resampler`/`PerceiverAttention`/`FeedForward`。
- `iper_dataset.py` —— V6.5 在 iPER 上的过拟合数据集读取（含 `smplx_global` 形状校验 `[N,26]`）。
- `__init__.py` —— 导出 `FaceControlAdapter`、`FaceFineCondition`、`FaceReferenceFeatures`、`FaceConditioningPreparer` 等。

## 与上下层的关系

- 上游：data 层提供的脸关键点、ArcFace/DINO 参考、源裁剪。
- 下游：被 `../deepgen_adapter.py` 经 `face_adapter` / `face_preparer` 引用；残差经 `../control_core` 注入 Qwen 的 32 层。
- 源目录见 [../../v6/face/README.md](../../v6/face/README.md)；上层见 [../../qwen21_full/README.md](../../qwen21_full/README.md)。

## 注意与坑

- `FaceControlAdapter` 仅把身份/纹理加到**目标头部 ROI 内**，源裁剪与 ArcFace 属于 A；不要把目标 RGB 或目标身份送进该模块。
- ArcFace 特征维度固定 512，传入 `pool_arcface_references` 的张量须为 `[...,R,512]`。
- 本目录与 `v6/face/` 逐文件一致；改通用脸部逻辑要确认改的是哪一份。
