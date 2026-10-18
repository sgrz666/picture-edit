# src/pose_control/qwen21_full/detail/ —— V6.4 细节分支（全量迁移副本）

本目录是从 `v6/detail/` **原样迁移（逐文件字节一致）** 到 qwen21_full 的副本，模块职责与 V6 完全一致——实现 V6.4 的"脸 + 手"局部细节分支：把两人、三区域（脸 / 左手 / 右手）的局部几何与参考外观编码成可被 `control_core` 注入的残差。迁移到 Qwen-Image-2.1 后，残差输出端从 DeepGen 目标流改为 Qwen 目标流，但本目录的几何/外观编码逻辑未改。

## 里面有什么

- `conditions.py` —— `DetailRegion`（FACE/LEFT_HAND/RIGHT_HAND）、`FaceHandDetailCondition`、`DetailReferenceBatch`、`PreparedFaceHandDetailConditioning`。
- `encoders.py` —— `DetailAppearanceEncoder`、`DetailSpatialEncoder`；左/右手镜像规范化 `canonicalize_left_hand_keypoints`/`restore_left_hand_keypoints`、姿态 `canonicalize_left_hand_pose`；`ReferenceFeatureProvider` 接口协议（约定 `[B,2,3,R,F]` 浮点特征）。
- `hand.py` —— `HandDetailCondition`：仅手部几何与源/目标绑定的双人结构（被 `../hand/` 复用/继承）。
- `binder.py` —— `DetailTokenBinder`：给区域 token 加 person/source、区域类型、解剖侧三类绑定（region=3、side=3 的 Embedding）。
- `preparer.py` —— `FaceHandDetailPreparer` 及 `soft_box_masks`：把 bounding box 栅格化为软 mask（无效区域严格为 0）。
- `__init__.py` —— 导出上述类与 `hand_to_legacy_detail` / `legacy_to_face_and_hand` / `references_to_hand_only` 等兼容转换函数（供 `../deepgen_adapter.py` 调用）。

## 与上下层的关系

- 上游：由 data 层或 `../data_runtime.py` 准备好的 bounding box、关键点、参考帧。
- 下游：被 `../deepgen_adapter.py` 通过 `detail` 分支名引用；残差经 `../control_core` 注入 Qwen 的 32 层（6 组）。
- 源目录见 [../../v6/detail/README.md](../../v6/detail/README.md)；上层见 [../../qwen21_full/README.md](../../qwen21_full/README.md)。

## 注意与坑

- 本目录与 `v6/detail/` 逐文件一致；修改通用细节逻辑时务必确认你改的是 v6 还是 qwen21_full 这一份。
- `hand.py` 与 `../hand/conditions.py` 的 `HandDetailCondition` 存在继承/转换关系，改手结构要两处都核对。
