# src/pose_control/v6/detail/ —— V6.4 细节控制分支

`detail/` 实现 V6.4 引入的"脸 + 手"局部细节分支（在 V6.6 把手单独成支前，脸手细节一度合并于此）。核心是把两人、三区域（脸 / 左手 / 右手）的局部几何与参考外观编码成可被 `control_core` 注入的残差。

## 里面有什么

- `conditions.py` —— `DetailRegion`（FACE/LEFT_HAND/RIGHT_HAND）、`FaceHandDetailCondition`（归一化双人脸/手几何与区域绑定）、`DetailReferenceBatch`、`PreparedFaceHandDetailConditioning`。
- `encoders.py` —— `DetailAppearanceEncoder`、`DetailSpatialEncoder`；左/右手镜像规范化 `canonicalize_left_hand_keypoints`/`restore_left_hand_keypoints`、姿态 `canonicalize_left_hand_pose`；`ReferenceFeatureProvider` 为未来外部预计算特征的接口协议（约定返回 `[B,2,3,R,F]` 浮点特征）。
- `hand.py` —— `HandDetailCondition`：仅手部几何与源/目标绑定的双人结构（被 `../hand/` 复用/继承）。
- `binder.py` —— `DetailTokenBinder`：给区域 token 加 person/source、区域类型、解剖侧三类绑定（region=3、side=3 的 Embedding）。
- `preparer.py` —— `FaceHandDetailPreparer` 及 `soft_box_masks`：把 bounding box 栅格化为软 mask（无效区域严格为 0），组装细节条件。
- `__init__.py` —— 导出上述类与 `hand_to_legacy_detail` / `legacy_to_face_and_hand` / `references_to_hand_only` 等兼容转换函数（供 `deepgen_adapter.py` 调用）。

## 与上下层的关系

- 上游：由 data 层或 `interface` 准备好的 bounding box、关键点、参考帧。
- 下游：被 `v6/deepgen_adapter.py` 通过 `detail` 分支名引用；残差经 `control_core` 注入。在 V6.6 后，纯手部细节改由 `../hand/` 承担，本目录偏脸+手混合的历史结构。
- 同构迁移版见 [../../qwen21_full/detail/README.md](../../qwen21_full/detail/README.md)。

## 注意与坑

- 本目录的 `hand.py` 与 `../hand/conditions.py` 的 `HandDetailCondition` 存在继承/转换关系，修改手结构时要两处都核对。
- `ReferenceFeatureProvider` 约定的特征形状为 `[B,2,3,R,F]`，外部预计算特征提供方需对齐该布局。
