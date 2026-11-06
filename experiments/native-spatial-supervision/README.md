# Native Adapter 离线空间监督

目的：对 GT 的穿衣人体、有效身体关节、可信可见手部生成离线软区域，让 Native V5 可配置地使用空间加权 Flow Matching。先验收 20 张，再处理 Bilibili 全量 1,960 帧；不启动训练、不筛掉原训练样本。

入口为仓库 `scripts/data/build_native_spatial_weights.py`，纯几何逻辑在 `src/pose_control/qwen21_full/native_v5/spatial_masks.py`。服务器产物根目录为 `/data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1`，`pilot20_v2/` 与 `full/` 分开保存。每帧有原图分辨率 Body Mask、Canvas Joint/Hand 图、分量 NPZ、权重热图、latent 权重图及 7 栏对照图；index 以原图 SHA256 和 Canvas 指纹索引。`frames/<frame_id>/ready.json` 声明每张产物的绝对路径和 SHA；重试保留独立的 `attempt_<uuid>/`，不覆盖中断产物。

已完成 20 张 CUDA 预览及全量 1,960 张缓存，全量 0 失败，生成阶段耗时 2,803.37 秒（46.72 分钟，不含模型加载），占用约 462 MB。140 张启用手部增强。独立审计逐条验证全部 GT／组件 SHA、原始记录、实际 RGB 尺寸和 Canvas；Qwen64 零系数误差与 Padding 权重均为 0，latent 权重最大 2.5。服务器相关回归 78 项通过、1 项跳过，原图和已有配对／划分清单 SHA 保持一致。默认系数 Body=.5、Joint=.5、Hand=1，真实背景基础权重 1，Padding 为 0。尚未运行 Qwen 训练或验证姿态指标改善。

已查看 20 张总览及手部近图，手部增强在 latent 网格中保留；少数花纹服饰有分割碎孔，Body Mask 是弱区域权重，不能当精确分割标注。v1 手部门控有明确局限：`BV14NHK6SE9d_000286` 画面左侧手的 DWPose 21 点平均得分 0.92599，但 MediaPipe 的 hand_a 结果为空，旧 loss_mask 和 hand_raw_pose_usable 均为 false，故第一版未给它手部额外权重；Body 和基础 Flow 监督仍存在。精细关键点可训练性不宜作为可见手部粗区域的唯一门槛。这一问题已由下节 v2 的粗区域／精细区域两级策略修复，旧缓存及标记保留，Body Mask 直接复用。

上游是 `/data/shangguanrz/datasets/bilibili/frames/frames.jsonl` 及已有 SAM2.1 权重；保留当前 Native 数据选择的 `sapiens_v1` 版本标签，实际区域来自 GT 分割和已有 2D 关键点，不重新拟合 SMPL-X。下游是 Native V5 Training 的可选 Source 空间缓存与 Loss 系数；旧 face/hand 区域项与新空间项互斥，零系数保持旧训练路径。此实验不替代数据划分、原有训练准备或姿态评测。

## 手部区域 v2 小样本修复

`visible_region_v2` 将可见手部粗区域与精细手指区域分开：DWPose 至少 8 个有效点、点分数不低于 .35、有效点平均分不低于 .45、手根与身体 wrist 一致、至少 60% 的点受 GT 人体前景支持时生成最高 .65 的粗区域；仍须拒绝明确遮挡、出画、不存在、退化或异常尺寸。可信 MediaPipe 保留最高 1 的精细区域，仍服从原 loss_mask／hand_raw_pose_usable。两者 max 融合，重叠不重复累加。原始标记没有改写；Body／Joint／有效画布及 Loss 系数保持不变。

用原 20 张作配对对照，额外加入左右手各 1 个遮挡、各 1 个出画负例，共 24 张。原 20 张增强覆盖从 6 张／7 只手变为 20 张／37 只手（40 只中的 3 只低置信度仍拒绝）；4 个负例的指定手均保持关闭。全部 24 张共 23 张／40 只手增强。这里的数字是加权覆盖数，不是手部检测准确率。`BV14NHK6SE9d_000286` 左手的独立 latent 权重增量从 0 变为 .64989，实际局部 latent 权重峰值约 2.15314。人工查看原样本放大图和 24 张总览后，确认该手区域增强在下采样后保留。

服务器产物在本实验根目录的 `hand_fix_v2/`：`input/frames24.jsonl` 原样复制 24 条记录；`selection.json` 记录选择与负例；`pilot24/` 保存新分量、索引、7 栏预览及审计；`comparisons_verified/` 保存 8 栏新旧对照、左手放大图和逐手 `comparison.json`。对照工具为本目录 `scripts/hand_region_ablation.py`。候选策略、输入 SHA、NPZ metadata 和量化后的重算地图必须一致才能报告覆盖改善，不允许拿同一份旧缓存重算新策略后冒充新缓存结果。本地 `output/hand_fix_v2/` 有报告和图的副本。

本轮复用已有 Body PNG，未加载或运行 SAM、未下载模型，24 张生成耗时 47.64 秒。Body／Joint／valid 与基线逐元素一致；24 张 policy-aware 审计通过，Padding 和零系数基线误差为 0，手部独立 latent 增量最大 .86854。服务器相关回归 90 项通过、1 项跳过。只重建这 24 张；v1 的 1,960 张完整缓存没有覆盖，v2 小样本索引不能直接替代完整训练 Source 的索引。

复测时使用新的输出目录，已有 v1 Body 缓存可直接复用（CPU，无需 SAM 环境）：

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
RUN=/data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1/hand_fix_v2
$PY scripts/data/build_native_spatial_weights.py \
  --frames "$RUN/input/frames24.jsonl" \
  --body-cache-index /data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1/full/index.json \
  --output-root "$RUN/pilot24_repeat" --hand-policy visible_region_v2 \
  --device cpu --limit 0 --canvas-long-side 512 --torch-threads 4
```

`--hand-policy legacy_strict` 可复现旧门控；CLI 默认使用 v2，Python `regional_maps` 的旧默认保留兼容，直接调用应显式选择策略。审计新索引时用上述 `frames24.jsonl`，输出另一个新 JSON 文件名。服务器完整回归命令为：

```bash
NATIVE_SPATIAL_ABLATION_SCRIPT=/data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1/scripts/hand_region_ablation.py \
  $PY -m pytest tests/test_native_spatial_masks.py tests/test_native_spatial_weights.py \
  tests/test_native_spatial_audit.py tests/test_qwen21_native_v5_training.py -q
```

## 使用与验证

在服务器使用已有 deepgen 环境做 SAM 推理，不安装任何包。以下输出目录必须尚不存在；只在源代码、输入和参数完全一致时用 `--resume` 恢复，修改配置应更换目录。

```bash
cd /home/shangguanrz/project/pic-edit
/home/shangguanrz/miniconda3/envs/deepgen/bin/python scripts/data/build_native_spatial_weights.py \
  --frames /data/shangguanrz/datasets/bilibili/frames/frames.jsonl \
  --output-root /data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v2/full \
  --limit 0 --canvas-long-side 512 --device cuda --torch-threads 4
```

`--limit 20` 为预览；`--limit 0` 保留全量。`--canvas-long-side 512` 与当前 Bilibili Native 装配一致，GT 1080×1920 对应 288×512 Canvas 和 18×32 latent；该网格不适用于采用其他 Canvas 的训练，读取器会拒绝错位。

审计本轮全量缓存（审计报告必须用新文件名）：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/audit/audit_native_spatial_weights.py \
  --index /data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1/full/index.json \
  --frames /data/shangguanrz/datasets/bilibili/frames/frames.jsonl \
  --output /data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1/full/audit_qwen64_repeat.json
```

本轮已通过的报告为 `full/audit_qwen64_v1.json`，汇总与索引分别为 `full/summary.json`、`full/index.json`。索引 SHA256 为 `d2b774b405c72769bece90a620143b2b555912a36fc2e3d07d61e2daad53fa85`。`full/training_binding.json` 提供实际索引引用及 Loss 配置片段，不能替代完整训练配置；本地 `output/full/` 保存汇总、审计及预览副本。代码提交为 `f827331c`，远端 `source/` 保存部署校验和、补丁与变更前源码。

在既有训练配置的每个所选训练／验证 Source 加 `spatial_weights: {path: 索引绝对路径, sha256: 实测索引SHA}`，对应 route 的 loss 使用 `spatial_body=.5`、`spatial_joint=.5`、`spatial_hand=1`、`spatial_max=2.5`，原 `face`、`hand` 局部 velocity 项设为 0。调三个空间系数可直接复用 NPZ，无需重跑 SAM；训练时不加载分割／关键点模型。完整配置与 preflight 命令见 `src/pose_control/qwen21_full/native_v5/training/README.md`。不得将本轮缓存数值验收当成真实 Qwen 前反向或训练通过报告。

```bash
python -m pytest tests/test_native_spatial_masks.py tests/test_native_spatial_weights.py tests/test_native_spatial_audit.py -q
```
