# v65_face/ —— V6.5 人脸路径的第三方模块改编

`third_party` 下为人脸（Face ID 保持）模块打包的定点上游改编，服务于 V6.5 人脸路径的身份融合与姿态引导。所有模块均为"锁定 commit + 最小改动"。

## 里面有什么

- `__init__.py`：导出 `FacePoseGuider2D`、`FusionFaceId`、`InsightFaceArcFaceExtractor`、`Resampler`、`PerceiverAttention`、`add_optional_face_residual` 等接口；指明精确来源见 `UPSTREAM_SOURCES.md`。
- `mcld.py`：MCLD 的 2D 人脸预处理与 PoseGuider 改编（上游 commit `6279ff7…`）。含 `FacePoseGuider2D`、`expand_image_region`、`safe_crop_image`、`safe_square_crop_normalized`。
- `stableanimator.py`：StableAnimator 身份融合组件改编。含 `FacePerceiver`、`FusionFaceId`、`InsightFaceArcFaceExtractor`。
- `visual_persona.py`：Visual Persona 的 face-token attention 原语改编。含 `FeedForward`、`PerceiverAttention`、`Resampler`、`IPAttnProcessor2_0`。
- `xdyna.py`：X-Dyna 控制流的少量管线工具。含 `repeat_condition_batch`、`add_optional_face_residual`。
- `UPSTREAM_SOURCES.md`：精确上游 commit、许可证状态与本项目所做的接口改动记录。
- `licenses/`：上游许可证文本（`STABLEANIMATOR-MIT.txt`、`X-DYNA-APACHE-2.0.txt`）与 `README.md`。

## 与上下层的关系

- 被 V6.5 人脸训练/推理路径 `import`。
- 上游来源与许可由 `UPSTREAM_SOURCES.md` 与 `licenses/` 锁定。
- 相关：`../README.md`、`../../THIRD_PARTY_NOTICES.md`。

## 注意

- 均为"定点 + 最小改动"，改动接口需同步更新 `UPSTREAM_SOURCES.md`，否则来源追溯断裂。
