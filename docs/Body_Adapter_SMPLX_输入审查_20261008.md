# Body Adapter 的 SMPL-X 条件输入审查

审查日期：2026-10-08。以 sg 的 `/home/shangguanrz/project/pic-edit` 当前代码和 `tiktok_body_fullpass_20261004` 训练清单为准。

**结论：26 维是全局参数通道的设计维度。当前 TikTok Full Body 训练另外接收空间几何条件，但没有显式传入目标的下颌、表情和左右手指姿态。完整架构确实保留了 Face、Hand、Detail 接口；当前 Body 执行路径没有启用这些独立接口。**

本次完成数据构建、真实 dataset 样本、条件准备函数和训练控制开关的核对。仅做 CPU 审查，没有修改训练架构、旧检查点或启动新的训练。

## 1. 26 维是什么

`scripts/build_tiktok_v6_conditions.py:315–349` 构造：

| 切片，左闭右开 | 内容 | 维度 |
| --- | --- | --- |
| `[0:10]` | 体型 betas | 10 |
| `[10:16]` | 根节点整体旋转，axis-angle 转 6D | 6 |
| `[16:19]` | 整体平移 | 3 |
| `[19:26]` | 相机，`fx/W, fy/H, cx/W, cy/H, tx, ty, tz` | 7 |

合计 26。相机末尾 3 维重复记录整体平移，因此不应将它描述为 26 个相互独立的 SMPL-X 参数。根旋转也不能等同于独立的头部姿态。

`src/pose_control/qwen21_full/token_encoders.py:117–137` 的 Global encoder 明确使用 `LayerNorm(26)`，将这一通道编码为 4 个 256 维 token。面部、手部参数没有被压缩进这 26 维。

## 2. Body 实际接收到什么

`QwenFullDataset(dataset_type='tiktok_v3')` 使用的是 `TikTokV3PoseDataset`，当前训练采用 letterbox。不能用其他旧版 TikTok dataset 的字段推断这轮训练输入。

两条真实样本均包含：

| 输入 | 单个样本形状 | 当前条件路径 |
| --- | --- | --- |
| normal | `[3,512,512]` | 送入几何条件 |
| pose_heatmap | `[25,512,512]` | 送入几何条件 |
| part_onehot | `[14,512,512]` | 送入几何条件 |
| human_mask | `[1,512,512]` | 送入几何条件 |
| smplx_global | `[26]` | 送入全局参数编码 |
| depth | `[1,512,512]` | 仅在 `condition_injector.use_depth` 开启时传入 |
| control_map | `[8,512,512]` | dataset 会生成；当前 `build_condition_bundle()` 没有读取它 |

`control_map` 包含 normal/depth/DWPose/mask；它存在于 batch，并不能证明其中 DWPose 画出的面部、手部已经进入 Full Body 条件。实际读取的是独立的 `pose_heatmap`，这里为 SMPLest-X 投影映射得到的 OpenPose Body25 点，不含完整的 21 点手指和密集面部关键点。

原始 63 维 `body_pose` 也没有直接作为向量输入；身体姿态通过渲染几何和身体关键点表达。网格法线可能间接表达部分脸和手的表面几何，但这不能等同于独立的 expression、jaw、hand_pose 输入，亦不能据此证明细节精度足够。

## 3. 你记得的面部和手部能力确实存在

旧的 `FaceHandDetailCondition.smplx_detail` 定义为 103 维：

`expression(10) + jaw_pose(3) + left_hand_pose(45) + right_hand_pose(45)`。

另外有 68 点面部和每只手 21 点关键点。`detail/hand.py:128–157` 将这一旧接口拆为独立的 Face 和 Hand 条件；新版 Face 使用 72 点布局。

`Qwen21FullAdapter.prepare_conditioning()` 接受 `face_condition`、`hand_condition`、`detail_condition` 及其参考特征。下层控制器只有收到相应条件时，才构建对应的 prepared 分支。因此“完整架构有这些模块”和“本轮传入这些数据”需要分别核对。

## 4. 当前调用在哪里停止传递

sg 上 `src/pose_control/qwen21_full/execution.py:367–383`：

1. `prepare()` 总是构建 normal/Body25/part/mask/global26 的 bundle。
2. 只有 `with_face=True` 时，才通过 `FaceConditionStore` 添加 Face 条件。
3. 这里没有构建或传入 Hand、共享 Detail 条件。

`execution.py:581–593` 的实际训练调用：

- Body：`with_face=False`，`face_strength=0`。
- Face：`with_face=True`，`face_strength=1`；调度和有效性会进一步决定实际作用。
- 两条分支均为 `hand_strength=0, hand_mode='off'`。

Face 的独立训练路径读取 **B 的 jaw3、expression10、72 点面部几何**；身份参考 ArcFace/DINO 来自 **A**。所以当前单独 Face 训练有面部 SMPL-X 参数输入，Body 训练没有；不能把本次结论推广为所有历史面部实验都未传递参数。

## 5. 真实数据和调用核验

抽查旧训练清单第 96 项和验证清单第 0 项，索引从 0 开始：

| 样本 | B 帧 | jaw3 绝对值和 | expression10 绝对值和 | 左手45 | 右手45 |
| --- | --- | --- | --- | --- | --- |
| `train_0000097`，序列 `00006` | `0395` | 0.0190 | 3.4970 | 4.4703 | 4.6363 |
| `val_0000032`，序列 `00127` | `0127` | 0.1513 | 4.9519 | 4.6314 | 4.7478 |

这些原始参数均有限且非零。对应 Body 样本中不存在 `jaw_pose/expression/left_hand_pose/right_hand_pose/smplx_detail`。

进一步用 CPU capture adapter 调用**真实的 `prepare(..., with_face=False)`**，两条样本传给 `prepare_conditioning()` 的键均为：

`condition_bundle, source_latent, target_token_hw`。

其中没有 Face、Hand、Detail 参数。脚本断言全部通过，结果记录了训练清单与关键代码 SHA256。本次没有运行真实 DiT 前向/反向，也没有用两个样本估计整个数据集的细节质量；这是输入链路证据。

## 6. 对 BFS 主线的影响

先前计划的“现有 Body 对比固定 BFS Body + Body”仍可以检验同一几何输入下的底座收益，但应将其范围写为 **Body 几何分支比较**。它无法同时验证独立表情和手指控制，也不能将这些失败全部归因于底座。

若要验证完整的“保留 A 身份、服从 B 的表情与手势”，需要单独设计接通已有 Face/Hand 条件的实验，两组保持相同输入和训练范围。不要直接把 Global 的 26 改成 129：已有独立接口负责细节，改动 Global 输入还会影响继承检查点的兼容性。先验证条件来源、坐标、有效性、prepared 分支和梯度，再决定训练组合。

本次没有改动或执行 BFS 实验计划。

## 7. 怎么复查

本地证据：

- `D:/codeplus/pictureedit/research/smplx_condition_audit_20261008/body_inputs.json`
- `D:/codeplus/pictureedit/research/smplx_condition_audit_20261008/audit_body_inputs.py`

在本地 PowerShell 重跑 CPU 审查，输出新的 JSON，保留本次审查证据：

```powershell
Get-Content -Raw -LiteralPath 'D:/codeplus/pictureedit/research/smplx_condition_audit_20261008/audit_body_inputs.py' |
  ssh sg /home/shangguanrz/nihaochen/envs/qwen21/bin/python - --output /tmp/body_inputs_recheck.json
```

成功时输出 `passed: true, samples: 2`。可以用 `scp sg:/tmp/body_inputs_recheck.json .` 下载查看。重点对照 `raw_parameters`、`explicit_parameter_presence`、`prepare_argument_keys` 和 `geometry_argument_keys`；代码变动后也要检查 `code_sha256`，不能把本次结果当作之后所有版本的保证。
