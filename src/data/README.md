# src/data/ —— 数据集读取与几何/letterbox 预处理

`data/` 封装 iPER / TikTok / Hi4D 三个数据集的读取，以及构造控制器所需的几何条件（normal、part onehot、SMPL-X 全局、contact 等）与 letterbox 图像变换。所有加载器都遵循"只取目标几何、不读目标 RGB 作为控制"的口径。

## 里面有什么

- `iper_dataset.py` —— `IPERPoseDataset`：iPER 配对序列训练集；含水平翻转的 part/keypoint 交换表（`PART_FLIP_SWAP`、`KEYPOINT_FLIP_SWAP`）。
- `tiktok_dataset.py` —— TikTok 加载器：`TikTokPoseDataset`、`TikTokV3PoseDataset`、`TikTokV65FaceOverfitLoader`/`TikTokV65OverfitInputs`、`is_tiktok_v3_manifest`、`load_pose_dataset`、`build_tiktok_face_inputs`（把 `v65_face_features.pt` 转成 `FaceFineCondition`/`FaceReferenceFeatures`）。
- `hi4d_qwen21_dataset.py` —— Hi4D 的 N2C 控制（`hi4d_qwen21_dual_v2`）：`SMPL_TO_V6`（SMPL 24 LBS 关节 → V6 14 段约定）、letterbox 数组辅助；**仅渲染目标几何作控制，目标 RGB 与 GT mask 不读（GT mask 留给评测）**。
- `letterbox.py` —— `Letterbox` / `NativePad`（frozen dataclass）：单一记录的影像空间变换，版本 `letterbox_v1` / `native_pad_v1`，供 RGB 与几何对齐。
- `__init__.py` —— 导出上述类。

## 与上下层的关系

- 上游：仓库 `datasets/`（iPER / TikTok / Hi4D 资产）。
- 下游：产出 `ConditionBundle` 与各分支条件，喂给 `pose_control/v6/` 与 `pose_control/qwen21_full/data_runtime.py`；`build_tiktok_face_inputs` 直接对接 V6.5 face 条件。

## 注意与坑

- TikTok 模块对 `diffusers` 有特殊处理：若未预导入会构造占位模块，缺失真实 diffusers 时仅影响部分路径。
- Hi4D 数据集明确不读目标 RGB，构造控制时勿把 GT 人 mask 当作条件输入。
- letterbox 变换对 RGB 与几何必须同源记录，跨模态对齐前先校验 `version` 与元数据。
