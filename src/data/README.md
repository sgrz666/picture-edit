# src/data/ —— 数据集读取与几何/letterbox 预处理

`data/` 封装 iPER / TikTok / Hi4D 三个数据集的读取，以及构造控制器所需的几何条件（normal、part onehot、SMPL-X 全局、contact 等）与 letterbox 图像变换。所有加载器都遵循"只取目标几何、不读目标 RGB 作为控制"的口径。

## 里面有什么

- `iper_dataset.py` —— `IPERPoseDataset`：iPER 配对序列训练集；含水平翻转的 part/keypoint 交换表（`PART_FLIP_SWAP`、`KEYPOINT_FLIP_SWAP`）。
- `tiktok_dataset.py` —— TikTok 加载器：`TikTokPoseDataset`、`TikTokV3PoseDataset`、`TikTokV65FaceOverfitLoader`/`TikTokV65OverfitInputs`、`is_tiktok_v3_manifest`、`load_pose_dataset`、`build_tiktok_face_inputs`（把 `v65_face_features.pt` 转成 `FaceFineCondition`/`FaceReferenceFeatures`）。
- `hi4d_qwen21_dataset.py` —— Hi4D 的 N2C 控制（`hi4d_qwen21_dual_v2`）：`SMPL_TO_V6`（SMPL 24 LBS 关节 → V6 14 段约定）、letterbox 数组辅助；**仅渲染目标几何作控制，目标 RGB 与 GT mask 不读（GT mask 留给评测）**；缓存里带 `canvas_hw`/`native_hw` 自述画布，读取时校验（见下面「注意与坑」）。
- `canvas_contract.py` —— `CanvasSpec`（frozen，版本 `canvas_v2`）：**唯一有权决定样本如何落到模型画布的记录**。`resized_hw` 必须与单一缩放比例一致（容差 `0.5/W + 0.5/H`，即整数取整本身能带来的最大误差），否则构造时直接 `NonUniformScaleError` —— 换句话说"把 604×1080 压成 512×512"这种拉伸在本模块里根本写不出来，而不是靠调用点自觉规避。三种等比策略：`native_pad`（只补边不缩放）、`letterbox(side)`、`aspect_preserving(max_pixels=…)`（超预算时两轴同比例缩小）。附带 `token_hw=(H/16, W/16)`、`camera()`（K'=T·K，保证关键点重投影与像素同轴）、`valid_mask()`（画布有效区，≠ 人 mask）、`fingerprint()`。旧的拉伸/裁剪模式由 `LegacyStretch` 记录（含 `aspect_distortion`），`require_canonical()` 一律拒绝。
- `letterbox.py` —— `Letterbox` / `NativePad`（frozen dataclass）：单一记录的影像空间变换，版本 `letterbox_v1` / `native_pad_v1`，供 RGB 与几何对齐。与 `canvas_contract` 的等值性由 `tests/test_canvas_contract.py` 逐尺寸比对像素与 `points()`。
- `geometry_pair_dataset_v2.py` —— 版本 `geometry_canvas_v2`：**一对样本要满足什么**才允许进训练。
  `canvas_contract` 管"一张图怎么放"，这里管"一对图与它的条件是否自洽"。
  `assert_pair_contract()` 逐条硬拒：缺 canvas / 缺 `target_token_hw` / token 网格与画布不符 /
  任一条件不在目标画布上（**拒绝而不是插值救回来**）/ `human_mask` 标到 padding 区 /
  有 `depth` 却没写 `depth_metadata`（灰度可视化不能当米制深度用）。
  另给 `collate_geometry_pairs()`（混画布的 batch 直接报错，不许 stack 成方形）与
  `bucket_by_canvas()`（按 `(target_hw, source_hw)` 分桶，这是 batch>1 的唯一正道）。
  **它是校验器，不是第二个 loader**：`TikTokV3PoseDataset._native_item` 已经在用两套独立
  NativePad 逐条件栅格化，把那段逻辑抄一份过来，正是方案 §1 警告的"各 Dataset 自行 Resize"。
  所以改法是让 native item 自己吐 `source_canvas`/`target_canvas`/`target_token_hw`/
  `camera_metadata`/`depth_metadata` 并在返回前调用本模块自检。
- `__init__.py` —— 导出上述类。

## 与上下层的关系

- 上游：仓库 `datasets/`（iPER / TikTok / Hi4D 资产）。
- 下游：产出 `ConditionBundle` 与各分支条件，喂给 `pose_control/v6/` 与 `pose_control/qwen21_full/data_runtime.py`；`build_tiktok_face_inputs` 直接对接 V6.5 face 条件。
- 画布口径由两处强制执行：`data_runtime.QwenFullDataset` 的训练门（拒绝 `resize_square` / `square_crop`，除非显式 `legacy_policy_ok=True`），以及 `execution.py:_real_token_grid`（从真实矩形画布读 token 网格，不再用 `resolution//16` 猜成 `(grid, grid)`）。

## 注意与坑

- **`align_mode` 五种取值，语义不同**：`native`（只补边，走 `_native_item`）、
  `aspect_preserving`（两轴同比例缩到 `max_pixels` 预算再补边，走 `_budget_item`，**必须给 `max_pixels`**）、
  `letterbox`（缩到 `resolution` 方框内再补边）、`resize`（非等比拉伸，legacy）、
  `square_crop`（等比但丢视野，legacy）。缺省不给 `align_mode` 直接报错。
- **`_budget_item` 与 `_native_item` 是两条路，不要合并**：`alignment_meta` 被
  `face_supervision.py` 与 `execution.py` 按 `NativePad` 的字段名（`padded_width/padded_height`）读取，
  而 NativePad 的语义是"不缩放"。缩放过的画布塞进那个 schema 会让 Face 侧映射静默出错，
  所以预算路径只发 `source_canvas`/`target_canvas`（canvas_v2），不发 `alignment_meta`。
- **改 native 路径必须先做 A/B 快照**：`/tmp/native_baseline` 那套比对（40 个张量逐元素比）
  是唯一能证明"重构没改变在跑实验所看到的像素"的手段；`tests/test_canvas_contract.py` 里
  CanvasSpec 与 NativePad/Letterbox 的像素等值测试是它的前置保证。
- **`align_mode` 不再有默认值**。`TikTokV3PoseDataset` / `load_pose_dataset` / `QwenFullDataset` 缺省调用会立刻报错；复现历史 run 必须显式写 `align_mode="resize", legacy_policy_ok=True`（仓内 4 个旧 audit/run 脚本已这样标注）。
- **`load_pose_dataset` 也带门**：旧 iPER 训练入口（`train_iper_qwen21_body/face.py`）能经它走到 TikTok，
  所以 `resize` / `square_crop` 在这里同样被 `CanvasPolicyError` 拒绝，除非显式 `legacy_policy_ok=True`
  （两个脚本已加 `--legacy-policy-ok`）。iPER 清单走 `IPERPoseDataset` 分支，不看这个开关，行为不变。
- **Hi4D 不在这条隐患里，但它的缓存绑定已补**：`hi4d_qwen21_dataset.letterbox_array` 用的是
  `scale = min(SIZE/w, SIZE/h)` 单因子 + 居中补边，本来就是等比的（128×96 竖图 → 384×512 内容框，
  比例不变，`tests/test_hi4d_canvas_record.py` 钉住）。它原先的独立债务是 control 缓存键只哈希文件路径
  与 `VERSION`、不含画布变换 —— 改 `SIZE` 或改 letterbox 会让 1389 个旧缓存被静默复用。
  现在 `convert()` 写 `canvas_hw` / `native_hw`，`__getitem__` 读取时弹出并校验，校验不过直接
  `ValueError` 让人重算，记录不向下游传。**没有改缓存键、也没有把像素改走 CanvasSpec**：
  cv2 与 PIL 重采样不同，重写像素等于静默改掉已在用的控制图，换键等于强制重算 1389 个。
  旧文件不带记录，靠「任何边长 >16 的方形张量必须 = (512,512)」兜底；这条兜底已在真实词表上全量实测
  （1389/1389 打开、0 误拒、24 个 key/shape 组合、16668 个方形张量全是 512×512；
  把 `SIZE` 改 768 后同一份立刻被拒）。数字出处见
  [`../../research/canvas_native_portrait_acceptance_20261022/README.md`](../../research/canvas_native_portrait_acceptance_20261022/README.md)。
  残留：形状兜底是启发式，将来若加入方形但非空间的张量（如 24×24 关系矩阵）会被误判；新写的文件走 `canvas_hw`，不受影响。
- iPER 的 `normal_512 / dwpose_512 / part_512` 已经是方形渲染，在 Dataset 里改成补边救不回三维比例；`iper_dataset.CANVAS_POLICY = "resize_square"` 就是这条自述。真要竖图得回到原始 SMPL-X 参数与相机重渲染资产。
- `CanvasSpec.stack()` 与 `points()` 是 RGB 与各条件图唯一的共同变换；跨模态对齐前先确认两边用的是同一个 `fingerprint()`。
- TikTok 模块对 `diffusers` 有特殊处理：若未预导入会构造占位模块，缺失真实 diffusers 时仅影响部分路径。
- Hi4D 数据集明确不读目标 RGB，构造控制时勿把 GT 人 mask 当作条件输入。
- letterbox 变换对 RGB 与几何必须同源记录，跨模态对齐前先校验 `version` 与元数据。
