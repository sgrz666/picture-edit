# Global Geometry v2：21 关节条件

Native Geometry 现在默认 Body21。原 Full 的默认构造保持 legacy；Native v1 需显式 `legacy=True`。本轮未改 loss、optimizer 或训练算法，未执行训练更新。新 Pose 参数还没有训练效果结论。

## 结构与尺寸

| 环节 | 输入 → 输出 | 约定 |
|---|---|---|
| Global | 26 → 4×256 | 原 MLP 及参数形状不变，仍加 1 个 Task Token |
| Body Pose | 63 → 21×3 → 21×6 | SMPL-X 局部轴角，rotation6D 按矩阵第 0 列、第 1 列排列 |
| FK | 22×3 → 21×3 | 缓存含 pelvis，编码只取 ID 1–21；相机坐标减 pelvis、除以同 beta 标准姿态身高 |
| 关节数值特征 | 6＋3＋2＝11 → 256 → 256 | 2D 投影除以画布宽高；Joint Embedding 相加，未拼成 267 维 |
| Skeleton Attention | 21×256 → 21×256 | 两层、8 heads、FFN4、dropout0；22 节点完整树的最短路径偏置，之后选 1–21 |
| Joint Cross-Attention | Joint 256→512；Query 512 | 8 heads，独立于原 Global attention，只返回增量 |
| 几何融合 | F_base＋ZeroProjection(update) | 仅最后 512→512 投影置零；系数 1，无额外零 gate |
| Native 输出 | `[B,N_target,512]` | 双人、contact、高低尺度路径继续使用更新后的 tokens |
| Qwen | 4096↔512 Native Body | 不向主干添加关节 token，不在 32 个 block 内重算关节编码 |

新增 Pose 参数 `3,100,672`；继承 Geometry `75,966,284`；新版 Geometry 合计 `79,066,956`。Native Body 与 CHAMP Depth 维度、参数形状不变。

`BodyPoseConfig(rotation='axis_angle')` 保留 3＋3＋2＝8 维数值输入消融。`BodyPoseConfig(spatial_bias=False)` 关闭关节读取的空间偏置。默认偏置为像素距离平方的负 Gaussian，`sigma=max(16px, 0.15×同源 SMPL mask bbox 高度)`，Query 位于目标网格 cell 中心。

FK valid 和 projection valid 分开保存。后者只检查相机前方和画布边界，不删除自遮挡关节，不 clamp 越界坐标。无可投影关节的行不执行 Joint 更新；`joint_attention.last_projectable` 记录跳过状态。FK 缺失或缺 Body63 直接拒绝。

## 数据与来源

`scripts/data/build_qwen21_body_pose_cache.py` 在既有 SMPLest-X 环境运行。输入 frames JSONL 每行：

```json
{"dataset":"iper","sequence_id":"001_8","role":"target","stem":"target_f000055"}
```

TikTok 使用 `dataset=tiktok`、序列和原始 stem，role 为 target。工具通过 `smplestx/frame_map.json` 精确定位参数行；iPER 同时匹配 role，绝不从文件名猜索引。原生 SMPL69 拒绝，不截断成 Body63。

缓存保留原轴角、22 个 FK、原图像素投影、标准姿态身高、canonical Global26。模型使用 neutral NPZ、非 PCA 手部、现有 hand mean，与原 root/beta/body/hands/jaw/expression 重建顶点；和 `mesh_cam` 最大差超过 `1e-4` 停止。没有重跑估计，也不把 A 的体型拼到 B 的 RGB 监督。

Global26 保持布局：beta `[0:10]`、root6D `[10:16]`、平移 `[16:19]`、相机 `[19:23]`、重复平移 `[23:26]`。iPER 两个平移槽都使用 `transl_raw`，相机为 raw；TikTok 使用匹配模型渲染的标准字段。旧 v6 cache 不改写，legacy 继续读取旧语义。

每帧的来源记录包含参数文件、frame map、模型及 FK 实现、RGB、Normal、Part、SMPL mask、Depth 和旧 Body25 条件的 SHA256。reader 校验 schema、关节树、网格误差和全部来源；运行中资产漂移报错。记录明确旧条件图到原图的尺度，再应用一次目标 CanvasSpec。Joint UV 离线缓存为原图像素，加载时才变换到目标画布，因此同一 raw cache 可用于多个矩形预算。

iPER 的 512 条件对应 1024 原图；Depth manifest 要显式声明 `asset_to_original=[[2,0,0],[0,2,0],[0,0,1]]`。不声明仍拒绝尺寸不符；错误或重复尺度会报错。Depth 仍是 typed DepthCondition，近 1 远 0、独立 valid，PNG 不宣称 metric。

缓存只包含非学习数据。Joint Tokens 与 Geometry 每次条件编码计算一次，保留计算图。训练默认冻结旧 Geometry，但新 Pose 参数独立开放；`train_pose=False`/`--freeze-pose` 可显式冻结。不要用整体 `geometry.requires_grad_(False)` 代替分支选择。

SMPL-X ID 1–21 的顺序见 `body_pose.JOINT_NAMES`。Body25 的对应按名称定义于 `BODY25_BY_NAME`，例如 left hip 是 SMPL-X 1 / Body25 12，right shoulder 是 SMPL-X 17 / Body25 2。spine、collar、head、foot 不伪装成某个同号热图点。

## 接口与 checkpoint

`condition_injector` 的每个人独立接收 `body_pose_[a|b] [B,63]`、`joints_cam_[a|b] [B,22,3]`、`joint_uv_[a|b] [B,21,2]`、`fk_valid_[a|b] [B,22]`、`projection_valid_[a|b] [B,21]`、`body_height_[a|b] [B]` 和 `smpl_bbox_height_[a|b] [B]`。`ConditionBundle` 分别保存两个人的 Joint Tokens、UV、valid、sigma 和画布；`.to()` 保留 bool，`.index_select()` 随双人子批次选择，不串人。

```python
import torch
from src.pose_control.qwen21_full.pose_data_runtime import NativePoseDataset
from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime

dataset = NativePoseDataset(pairs, pose_manifest, max_pixels=512*512)
sample = dataset[0]
batch = {k: v[None] if torch.is_tensor(v) else [v] for k, v in sample.items()}
system = NativeDepthRuntime.from_native_checkpoint(saved_v2).to('cuda')
features = system.encode(batch, target_token_hw=sample['target_token_hw'],
                         depth=typed_depth, depth_strength=1.)
```

Depth checkpoint 做 off 对照时显式传 `depth_strength=0.`。原 Full → 新前端用 `Qwen21NativeGeometry.from_full_checkpoint(full)`；只严格读取原四个前缀，额外初始化已声明的 Pose 模块。`geometry.pose_parameters()` 单独暴露新参数。

v1 runtime 需显式 `from_native_checkpoint(saved_v1, legacy=True)` 重建，或 `upgrade_pose=True` 迁到 v2。v2 可直接 strict 重建，不再依赖 Full 文件。新 checkpoint 保存 Pose 配置、列顺序、数据 schema 与已使用来源。未知 key、旧参数缺失、shape/dtype/schema 不匹配都报错；没有宽泛 `strict=False`。

## 在 sg 上复查与使用

项目根目录 `/home/shangguanrz/project/pic-edit`；以下 `POSE_RUN` 指本轮产物，所有新输出应换成不存在的名字。

```bash
POSE_RUN=/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/global_pose_step4_20261024_v1

# 从原 SMPL-X 参数建立新 raw FK cache，不覆盖旧 cache。
add/envs/smplestx_bw/bin/python scripts/data/build_qwen21_body_pose_cache.py \
  --frames "$POSE_RUN/data/frames.jsonl" --output "$POSE_RUN/data/fk_manual_v2"

# 只迁权重/初始化新 Pose，不建 optimizer，不运行训练。
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/data/migrate_qwen21_geometry_v2.py \
  --source /data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/depth_step3_20261023_v1/training_smoke_v1/checkpoint_last.pt \
  --mode upgrade_pose --output "$POSE_RUN/checkpoints/body21_manual_v2.pt"

# 真实 checkpoint＋条件预检；不加载 Qwen、不采样。
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/run/run_qwen21_native_pose.py \
  --checkpoint "$POSE_RUN/checkpoints/native_body21_v2.pt" \
  --pairs "$POSE_RUN/data/tiktok_pairs.jsonl" --pose-manifest "$POSE_RUN/data/fk_v1/records.jsonl" \
  --output "$POSE_RUN/inference_preflight_manual_v3" --preflight-only

/home/shangguanrz/nihaochen/envs/qwen21/bin/python -m pytest \
  tests/test_qwen21_global_pose.py tests/test_qwen21_pose_cache.py \
  tests/test_qwen21_native_geometry.py tests/test_qwen21_native_depth_runtime.py -q
```

既有训练入口 `train_qwen21_native_depth.py` 默认 `--geometry-version body21`，必须传 `--pose-manifest`。用 `--preflight-only` 验证 TikTok 或 iPER；`--geometry-version legacy_v1` 是旧路径，不能混入新 Pose cache。旋转和偏置消融通过 `--pose-rotation`、`--no-pose-spatial-bias` 构建，`--upgrade-v1-native` 显式声明旧 runtime warm-start。loss 与 optimizer 原实现没有改动。本轮只执行预检，不启动训练循环。

推理入口移除 `--preflight-only` 后使用源 RGB、目标几何和可选 `--depth-mode/--depth-manifest`。KV cache 固定关闭，`--reference-resolution` 显式锁定参考图预算，目标尺寸取 canonical canvas。当前完成的是条件/checkpoint 预检及实际小型/7B DiT 前反向，未验收完整采样质量。

方法依据：[连续旋转表示](https://arxiv.org/abs/1812.07035)、[PoseFormer 空间关节建模](https://github.com/zczcwh/PoseFormer)。这里只借鉴单帧空间编码，不引入时间模块或预训练权重；不预设 6D 一定优于轴角。
