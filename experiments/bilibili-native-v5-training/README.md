# Bilibili 全量划分与 Native V5 训练对齐

目的：按用户确认的原视频 80/10/10 隔离 train、val、test，保留全部 1,960 帧、12,156 对及原质量标记；不新增严格双手子集，不重筛数据，不做冒烟。本轮只准备数据与正式训练配置，训练由用户以后启动。

远端数据根为 `/data/shangguanrz/datasets/bilibili`；本轮工作根为 `/data/shangguanrz/pic-edit-experiments/one/bilibili-native-v5-training-20261106-v1`，以下记为 `$RUN`。原图片、视频和原 `frames/frames.jsonl`、`pairs/pairs.jsonl` 保留。训练几何已按用户指定锁定 **Sapiens2 v1**，目录名 `sapiens_v1`。

| 划分 | 原视频 | 帧 | 配对 |
|---|---:|---:|---:|
| train | 39 | 1,560 | 9,520 |
| val | 5 | 200 | 1,490 |
| test | 5 | 200 | 1,146 |

固定 seed=42，比例按视频数定义，配对数随视频分布。[split.json](report/split.json) 记录分配表和原清单 SHA；[split_verification_groups_v2.json](report/split_verification_groups_v2.json) 已确认原视频、帧、完全相同像素互斥，group 索引与划分一致，全部行及质量字段保留。

独立像素隔离审计见 `$RUN/report/pixel_isolation_decoded_v1.json`：实际解码全部 1,960 份 RGB，按尺寸与解码后的 RGB 字节计算 SHA，得到 1,960 张唯一图像、跨集合重复 0；`passed=true`。该审计不信任原 `pixel_sha256` 字段，也不修改输入图像。

[sapiens_v1_asset_binding.json](report/sapiens_v1_asset_binding.json) 的 `passed=true`：5,880 份几何文件（每帧 `.pt`、Depth PNG、Normal PNG）及 1,960 份 RGB 已全量 SHA 核验。数据根的 `train/val/test` 各有 `frames_sapiens_v1.jsonl`、`pairs_sapiens_v1.jsonl`，绑定这套既有文件。它们是输入契约，**不是 Native manifest**，仍需生成下述 typed 侧车。

## Native 装配与当前状态

当前完整作业仍在处理，入口为 [run_full_preparation_aligned.sh](scripts/run_full_preparation_aligned.sh) → 服务器 `scripts/prepare_aligned.py`（本地 [prepare_cached.py](scripts/prepare_cached.py) 的新部署快照），仍加载同一封存的 `scripts/prepare_native_batch.py`（本地源码 [prepare_native.py](scripts/prepare_native.py)）。本轮日志为 `$RUN/report/native_preparation_20261106T225408.log`，成功报告目标为 `$RUN/report/native_preparation_20261106T225408.json`；目前不能称为装配完成或训练就绪。此作业不限制样本数，依次执行：

1. 按物理帧构建同一套 Sapiens2 v1 的 SMPL-X 资产、局部几何、pose guide 和 Source support。
2. 每 16 帧调用 [appearance.py](scripts/appearance.py) 的 `prime_dino`，每次前向最多 16 个 ROI，保持冻结 DINO FP32、224/no-center-crop；ArcFace 保持原五点对齐和 CPU 四线程。原 `hand_b` 对应左手、`hand_a` 对应右手，原 mask/raw 有效性分别保留。
3. 为全部 12,156 对装配 Native V4 body/local/reference、V5 detail/supervision 与 `single_2ref` 条件侧车，发布 `$RUN/native_sapiens_v1/manifests/{train,val,test}/pairs.jsonl` 和 `S0_prepare.json`。

[local_scene_depth.py](scripts/local_scene_depth.py) 修复了 PNG 深度 floor 量化影响局部遮挡判断的问题：全局 Body 继续使用原 PNG 数值；仅局部 `scene_z/scene_z_valid` 改为同一 Sapiens2 v1 driver 网格、同 target K/Canvas 的 FP32 渲染值，与局部子网格使用同一坐标与精度。metadata 单独记录 driver provenance，每 pair 强检 policy 与 driver SHA；真实遮挡仍保留。

当前组合为 `SingleTargetCache(LocalSceneDepth(original).assemble)`。[pair_target_cache.py](scripts/pair_target_cache.py) 只缓存最近一个 target 的物理条件；pair ID、Source 的 mask/canvas/reference，以及深度 context 等 provenance 均逐 pair 重绑定。CPU 契约覆盖与真实原 assembler 的全字段 tensor/metadata 等价性，并验证局部深度与缓存组合后的 driver 绑定。scene、cache、启动器、原 assembler 核心 4 份源码均作 SHA 快照，并纳入每个 pair 的 producer identity；原物理帧和 Source 特征 producer 保持不变，恢复时重新核验已有产物。

装配完成后接续 [configure_training.py](scripts/configure_training.py)，输出 `$RUN/configs_20261106_v1/S0.json` 至 `S3.json`、各阶段 `_prepare.json` 和 `summary.json`；再由 [preflight_native.py](scripts/preflight_native.py) 使用 **S3.json** 联合启用有效分支，通过真实 Native reader 读取 train/val 全部 11,010 对及独立 test 全部 1,146 对。目标报告为 `$RUN/report/native_data_full_20261106_v1.json`，轻量摘要为同名 `_summary.json`。摘要区分 `active_route_usable_counts_train_val` 与 `available_input_counts_train_val`，避免把 S0 关闭局部控制误解为输入不可用。这些文件须实际生成并通过后再判断结果。test 不进入训练 `sources`，val 的角色为 `validation`；数据读取检查不构造模型或 optimizer，也不启动 S3 训练。

画布沿用 HIA/Native 的 long-side 512、等比缩放和 padding。全局 Normal/Depth 读取原 `sapiens_v1` PNG，Part14、BPS 和局部子网格来自同一 v1 参数的 FK。旧参数包的 `mesh_cam` 可能未随 pose 更新，因此记录重建差异，不重新拟合或改原资产。[几何对齐报告](report/sapiens_v1_geometry_alignment.json) 已量化一个 1920×1080 样本：609,152 个前景像素只差 2 个边缘像素，IoU=0.9999967。条件采用既有 Depth、Normal 和同拟合光栅的共同 support，不改 PNG 数值、不缩减整图 RGB velocity loss 覆盖，也不删帧或配对。

环境已知 NVML 用户库为 580.178.04、内核模块为 580.173.02，真实 DINO 上 GPU 曾因此失败。进程级 `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` 已实测跑通并产出真实帧产物；当前入口保留此设置。系统驱动未改，本轮不处理系统环境修复。

## 复核与使用

本地 44 项 CPU 数据契约测试已通过，含 6 项局部深度回归及 9 项缓存契约（其中 1 项验证二者组合）；不加载模型，不做 GPU 冒烟或训练。复核命令：

```powershell
python -m unittest discover -s experiments/bilibili-native-v5-training/tests -v
```

服务器查看当前进度、复核已落盘的划分。`split_dataset_verified_v2.py` 是服务器已核验快照；报告使用新名字：

```bash
ssh sg
RUN=/data/shangguanrz/pic-edit-experiments/one/bilibili-native-v5-training-20261106-v1
QPY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
REPO=/home/shangguanrz/project/pic-edit
tail -f "$RUN/report/native_preparation_20261106T225408.log"
# Ctrl+C 只退出 tail，再运行划分复核。
python3 "$RUN/scripts/split_dataset_verified_v2.py" verify \
  --root /data/shangguanrz/datasets/bilibili \
  --report "$RUN/report/split_verify_$(date +%Y%m%dT%H%M%S).json"
```

装配成功后，接续的配置生成与全量读取命令如下。目标目录/报告若已存在，先查看结果，复核时另选新名字：

```bash
"$QPY" "$RUN/scripts/configure_training.py" \
  --native-root "$RUN/native_sapiens_v1" --output "$RUN/configs_20261106_v1"
"$QPY" -u "$RUN/scripts/preflight_native.py" \
  --config "$RUN/configs_20261106_v1/S3.json" \
  --test-manifest "$RUN/native_sapiens_v1/manifests/test/pairs.jsonl" \
  --report "$RUN/report/native_data_full_20261106_v1.json"
```

用户之后决定训练时，才手动准备冻结 Qwen VAE/文本冷缓存；必须使用新目录。以下命令不属于本轮自动执行内容：

若仍使用当前驱动环境，先沿用本轮已验证的进程级 allocator 设置。它只作用于当前 shell 启动的进程，不代表 NVML/内核驱动已修复：

```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
ENTRY="$REPO/scripts/train/train_qwen21_native_full_v5.py"
CFG="$RUN/configs_20261106_v1/S0.json"
CACHE="$RUN/qwen_frozen_$(date +%Y%m%dT%H%M%S)"
"$QPY" "$ENTRY" prepare-cache --config "$CFG" --cache-output "$CACHE"

# 用户先设置 STEPS 为正式更新数；按这一配置生成新的真实梯度报告。
: "${STEPS:?请先设置正式更新数}"
GRAD="$RUN/report/S0_grad_$(date +%Y%m%dT%H%M%S).json"
"$QPY" "$ENTRY" preflight --level grad --config "$CFG" --steps "$STEPS" \
  --cache-root "$CACHE" --cache-identity "$CACHE/index.json" --report "$GRAD"
"$QPY" "$ENTRY" train --config "$CFG" --steps "$STEPS" \
  --cache-root "$CACHE" --cache-identity "$CACHE/index.json" --gradient-report "$GRAD"
```

正式配置使用 `experiment_mode=standard`，不继承 fixed 300-step pilot 的训练预算或旧 gate。缓存通过 `--cache-identity` 绑定实际 index 及其 SHA；更新数和真实梯度报告必须与启动配置一致。S1/S2/S3 从上一阶段真实提交的 checkpoint 使用 `advance-stage --checkpoint ...` 转入，并满足对应阶段 gate；阶段模板本身不代表已训练、已验证或可直接启动。

上游为 `/data/shangguanrz/datasets/bilibili`，下游为 `src/pose_control/qwen21_full/native_v5/training`；几何与装配接口复用现有 HIA Native V5 实现。
