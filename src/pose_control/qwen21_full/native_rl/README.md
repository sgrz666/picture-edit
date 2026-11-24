# Native V6.2 / V7 在线 RL

从当前架构已提交的 Native SFT checkpoint 开始训练。入口按配置 schema 装配 V6.2 或 V7 Backend，V7 运行时仍按权重记录使用逐人物几何、Contact 和身份调制。每组候选共享同一任务的 Source、Prompt、人物绑定和 Canvas。默认每轮抽 4 个任务，每组生成 8 张图，10 步采样中第 4 步使用 Flow-SDE。Replay 以实际保存的 BF16 状态和动作重新计算 Native 特征和转移概率，首次 Replay 的 `|ratio - 1|` 上限为 `1e-4`。每轮最多回放两次，第二次起 PPO clipping 可产生约束。

训练范围由 `ParameterCatalog` 和所选 route 决定，包括启用的 Body、PoseLink 输出头，以及启用的双人 Interaction 输出头。学习率默认 `1e-5`，梯度裁剪 `1.0`，KL 系数 `0.001`。参考策略共享冻结的 Qwen 与 Native 编码器，单独保留 SFT 输出头；以相同状态、同方差的转移分布计算 KL。Native ODE 和 RL 的 velocity 走同一 V6.2 绑定入口，沿用 Qwen 原生 scheduler、VAE 和画布。

## 奖励与对照

- `--reward-profile legacy` 单人 Pose / ID / Background 为 `0.45 / 0.40 / 0.15`，双人 Pose / ID / Binding / Layout / Contact / Background 为 `0.27 / 0.23 / 0.18 / 0.13 / 0.09 / 0.10`。Pose 用 SDPose-Wholebody + YOLO11x，以 `geometry` 序映射 COCO17 计算 OKS，目标按置信度 `0.30` 固定可见点分母，候选漏点计零。双人 Pose 给较弱一人额外权重。Binding 比较正确与交叉的 ID 相似度，Contact 用已确认的 Part14 接触对和可见 COCO 关节代理。遮挡关系仍记为 `null`。
- `--reward-profile hands42` 把 SDPose 左右手各 21 点接入独立 Hands Reward。目标至少有 8 个有效手部关节时计入整图位置分。目标腕点有效、至少 7 个手指关节有效且手框对角线 ≥ 8 px 时，再按腕点对齐计算手指相对误差，以目标手框对角线归一化，Gaussian 的 σ 为 0.10。两项各占该手一半，手势不适用时用整图位置分。各只合格手等权，候选漏手、漏点或漏人计零。记录位置、手势和逐槽位 PCK。通用入口启用 SpatialReward 时默认用此 profile，其余情况默认用 `legacy`。
- ID 默认用 SCRFD + glintr100。Source 和目标都能按人体槽位绑定人脸时启用，候选漏脸计零。通用入口可指定 `--adaface-model` 与 `--adaface-rpe-extension`，仅在 rollout 使用 AdaFace-KPRPE，在固定 ODE 验证继续使用 glintr100。两个后端都按头部框一对一匹配，训练代码、模型、RPE 扩展与参考权重 SHA 进入训练签名。 [KP-RPE 论文](https://arxiv.org/abs/2403.14852) [仓库人脸识别对照](../../../../experiments/face-identity-reward-20261123/README.md)
- 背景默认用 `exp(-Masked L1 / 0.05)`，L1 归一到 `0..1`。单人、同源双人和独立双源统一以条件列表的第一张参考图 A 为背景源。双源的第二张图提供第二个人的外观，输出侧排除两个人。资格在 rollout 前固定。相同 Canvas 用像素恒等映射，同宽高比的内容框作等比重采样。排除 A 中所有已绑定的前景、目标各人前景和 16 px 边界，保留至少 10% 有效画布。A/B 背景差异不影响资格。无法对齐或缺少 mask 时记录原因。`--background-weight 0` 将该项从训练和 checkpoint 选点权重移除，评分日志继续报告背景指标。
- `--pose-only` 跑 Pose GRPO。`--aggregation sum` 先加权聚合全部适用的原始 Reward 再作组内标准化。`--aggregation gdpo` 使用 Flow-Factory v0.5.0 的逐项组归一化、权重和与采集批标准化。默认 `gdpo-guarded` 在标准化前抑制 std ≤ `0.001` 的奖励，并按当前适用的权重归一。整个批次无有效优势时跳过 optimizer。区域实验复用同一个 Flow-SDE、优势和更新内核。 [Flow-Factory v0.5.0](https://github.com/X-GenGroup/Flow-Factory/tree/ced2cd9b0465a2c9d733b9b8aed4d4351f950a22)

## SpatialReward-8B

`--spatial-model` 启用隔离的持久评分进程，使用 `sg` 上的 `/data/shangguanrz/models/rl/SpatialReward-8B`。父进程把已落盘的候选图及其 SHA 按 rollout group 交给评分进程，整组评分齐全后才计算 GDPO。每张候选分两路推理，SC 对照有序参考和独立评测指令，PQ 只看该候选。解析失败、超时或候选 ID 异常会写 `evaluation-failed.json` 并终止该批更新。所用模型文件 SHA、评分协议、参考顺序、Prompt SHA 和进程环境写入 run identity，续训时一起核对。 [SpatialReward 论文](https://proceedings.mlr.press/v306/long26a.html) [官方模型](https://huggingface.co/SpatialReward/SpatialReward-8B)

- `--spatial-references rgb` 是默认模式。`guides` 按 Native 顺序加入骨架图。`single_2ref`、`dual_4ref`、`dual_source_5ref` 各自保留原始参考顺序，独立双源通过 Native `slot → source` 绑定。候选和各 Canvas 参考都按自己的 `valid_rect` 裁剪，原始 RGB 使用完整内容。背景语义固定为 Source A，目标 GT RGB 只给现有几何及身份评分使用。
- 默认 `--spatial-mode visual-linear` 计算 `0.40·CON/25 + 0.30·NAT/25 + 0.30·ART/25`，保留 IF 作为诊断。`official` 计算 `SC^0.8 × PQ^0.2`，其中 `SC=(0.6 IF+0.4 CON)/25`、`PQ=(NAT+ART)/50`。`complementary` 将 SC 换成 `CON/25`。三种模式使用相同的 Native Prompt、四个原始子分与 Transformers 推理。输出保留模型给出的 `sc_evidence` / `pq_evidence`，其中框是裁剪后图像的模型坐标，作为诊断。 [官方评分实现](https://github.com/lorangan-ddup/SpatialReward/blob/0222484b6a30f865525e8dc3a90acf24654e096a/example/reward/server/spatial_reward_scorer.py)
- 启用后的默认权重是 `0.25`，其他奖励同比缩放。通用 `hands42` profile 的单人基础比例为 `30:20:25:15`（Body / Hands / ID / Background），双人为 `20:15:20:15:8:7:5`（Body / Hands / ID / Binding / Layout / Contact / Background）。`--spatial-weight 0` 保留评分用于观察。Guarded-GDPO 默认取 `max(0.005, --std-floor)` 作为 SpatialReward 的临时门槛，可用 `--spatial-std-floor` 固定校准值。日志记录每项的 std 和实际阈值。 [GDPO](https://arxiv.org/abs/2601.05242)
- 通用入口的 ODE 门禁检查 Pose / ID、适用的 Hands 总分及局部手势分，以及双人任务的双人检出率。启用 SpatialReward 后还检查适用的 Background / Binding。默认相对 SFT 基线可下降 `0.02`，双人检出率要求两个槽位均匹配且恰好检出两人。评分进程默认 CPU、batch 2，可用 `--spatial-python`、`--spatial-device` 和 `--spatial-batch-size` 配置。

离线校准使用已有候选图。输入 manifest 的 `schema` 为 `native_rl_spatial_candidates_v1`，包含 `groups` 列表。每组提供 `condition`（Native Qwen 条件 JSON 的 `{path, sha256}`）与 `candidates`（按顺序排列的图像 `{path, sha256}` 列表）。双源 `guides` 组另提供 `slot_sources`，取自 Native 的 `target_qwen_image_indices[0]`。可用同一份 manifest 分别跑 RGB 和骨架配置，输出逐候选子分给独立人工排序对照。

```bash
PY="$HOME/nihaochen/envs/qwen21/bin/python"
MODEL=/data/shangguanrz/models/rl/SpatialReward-8B
"$PY" scripts/eval/eval_native_spatialreward.py --model "$MODEL" --references guides \
  --manifest "$MANIFEST" --repeats 3 --output "$RUN/spatial-guides.json"
```

离线输出的 `repeat_reward_std` 是相同候选和相同配置下的重复评分标准差。方差门校准还需要用同一批图对照不同 `--batch-size`，并将候选排名与人工配对排序对照。先固定 `rgb + visual-linear`，三组只调整 `--spatial-weight 0.10/0.25/0.40`。锁定同一 SFT 父 SHA、奖励模型、阈值、rollout 配置、seed、更新预算、候选上限和留出集，分别使用新输出目录。报告有效更新数、实际生成图数、门禁和固定留出图的独立人工盲评。阶段式及动态权重放在固定权重消融之后。空间级优势需要独立区域奖励和位置级 log-prob 及 Replay 契约。

小红书区域入口 `experiments/xiaohongshu-region-sdpose-sft-rl/scripts/pipeline.py` 和共用区域 `scripts/run.py`、`scripts/rl.py` 接受相同的 `--spatial-*`、AdaFace 及 SDE 选项。默认沿用 `regional-legacy` 的权重，显式加 `--reward-profile hands42` 即使用通用单人比例。Body 继续用区域目标关键点评分，Hands 和 SpatialReward 复用共享实现。区域 AdaFace rollout 要求单人源图、目标图各有唯一可绑定人脸，候选漏检或多脸计零，留出评测照常用 glintr100。启动器可用 `--rl-python "$FACE/.venv/bin/python"` 单独指定 RL 解释器。已有区域实验按原快照复现，新对照使用新的实验根目录。

通用入口接入 AdaFace-KPRPE 时，使用已完成对照实验的 Python 环境和本地扩展。下列参数可添加到 `check-data`、`preflight`、`train` 或同配置 `resume`。

```bash
FACE=/data/shangguanrz/pic-edit-experiments/姿态估计/face
FACE_ARGS=(--adaface-model /data/shangguanrz/models/face/cvlface_adaface_vit_base_kprpe_webface4m
  --adaface-rpe-extension "$FACE/build/rpe/rpe_index_cpp.cpython-312-x86_64-linux-gnu.so")
"$FACE/.venv/bin/python" scripts/train/train_qwen21_native_v62_rl.py check-data --config "$CFG" "${FACE_ARGS[@]}"
```

## 运行

从仓库根目录执行，`CFG` 指向 Native V6.2 或 V7 SFT JSON。它要绑定完整 SFT `initialization` 及 SHA256、原视频或序列分组、Qwen model manifest 和冻结 latent cache。一个 run 使用一种条件 profile，可选 `single_2ref`、`dual_4ref` 或 `dual_source_5ref`。多 route 时用 `--route` 指定要继承的条件开关。`--models` 指定 AntelopeV2 的模型父目录。Qwen 用训练解释器，SDPose 通过 `--pose-python` 启动独立持久进程，默认使用项目的 SDPose 环境。每次进程启动加载一次模型，`--pose-device` 默认 `cpu`。Worker 异常会停止该批更新并写失败原因。

```bash
PY="$HOME/nihaochen/envs/qwen21/bin/python"
CFG=/absolute/path/to/native-v7-sft.json
RUN=/data/shangguanrz/pic-edit-experiments/native-rl-$(date -u +%Y%m%dT%H%M%SZ)

"$PY" scripts/train/train_qwen21_native_v62_rl.py check-data --config "$CFG" --groups-per-update 1
"$PY" scripts/train/train_qwen21_native_v62_rl.py preflight --config "$CFG" \
  --groups-per-update 1 --group-size 2 --steps 3 --sde-step 1 --val-limit 1 --output "$RUN/preflight"
"$PY" scripts/train/train_qwen21_native_v62_rl.py train --config "$CFG" \
  --groups-per-update 1 --group-size 2 --steps 3 --sde-step 1 --updates 1 --output "$RUN/pilot"
"$PY" scripts/train/train_qwen21_native_v62_rl.py preflight --config "$CFG" \
  --spatial-model /data/shangguanrz/models/rl/SpatialReward-8B --spatial-weight 0.25 \
  --groups-per-update 1 --group-size 4 --steps 3 --sde-step 1 --val-limit 1 --output "$RUN/spatial-preflight"
```

正式训练需要 `experiment_mode=standard` 和按视频或序列隔离的 validation。启动时打印 SFT 路径、SHA256、阶段和父链。先用固定 seed 生成 SFT 的 ODE 基线，再按 `--validate-every` 个 optimizer updates 和结尾复测。连续触发门禁达到 `--patience` 次会停止。`--hands-drop` 同时设 Hands 总分和手势分容忍度，`--two-person-drop` 设双人检出率容忍度。`--sde-step 3` 固定单步，`--sde-cycle 1 2 3` 按 rollout 批次循环。通用入口默认固定 `3`，区域入口默认循环 `1 2 3`。

### 独立视觉选点

通用入口把通过 ODE 门禁的 checkpoint 及其逐图报告写入 `summary.json` 的 `visual_review`，续训保留前段候选。即使专用指标综合分未高过 SFT，也可参与独立视觉评审。原有 `selected` 仍提供专用指标的自动选择结果。

```bash
"$PY" scripts/eval/select_native_rl_checkpoint.py prepare --run "$RUN/pilot" --output "$RUN/visual-review.json"
"$PY" scripts/eval/select_native_rl_checkpoint.py select --review "$RUN/visual-review.json" \
  --scores "$SCORES" --output "$RUN/visual-selection.json"
```

`$SCORES` 使用 `schema: native_rl_visual_scores_v1`，`review` 指向同一份评审清单的 `{path,sha256}`。`evaluator` 填 `{kind,protocol}`，`kind` 可用 `human_blind` 或 `editreward`，后者附带权重清单的 `model_manifest` 文件引用。盲评时只向评审展示固定条件和隐藏了 checkpoint 来源的图像。对 SFT 和所有入围模型，`candidates` 各填一条 `{id,scores}`，`scores` 按清单图像顺序记录 `{image_sha256,score}`，分数在 0～1。命令核对候选和 ODE 报告的 SHA，再比较独立视觉均分。最高分超过 SFT 至少要满足 `gain > --min-gain`，默认 `--min-gain 0`。独立选点写入新文件。

### 断点续训

Checkpoint 默认每 5 个有效更新保存一份，由 `--save-every` 修改。`update-*.commit.json` 绑定权重 SHA 与计数，checkpoint 内保存模型、AdamW 的 moments、采样和 RNG 状态、ODE 门禁状态及尚未回放完成的组。`resume` 使用原有 SFT 配置和相同训练参数，写入新目录。旧的仅权重快照继续用于推理或历史对照。

```bash
COMMON=(--config "$CFG" --groups-per-update 2 --group-size 2 --steps 3 --sde-step 1 --updates 10)
"$PY" scripts/train/train_qwen21_native_v62_rl.py train "${COMMON[@]}" \
  --stop-after-updates 5 --output "$RUN/first"
CKPT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["path"])' "$RUN/first/summary.json")
"$PY" scripts/train/train_qwen21_native_v62_rl.py resume "${COMMON[@]}" \
  --checkpoint "$CKPT" --output "$RUN/resumed"
```

背景默认复用 Native Source A / target 前景和 Canvas 的绑定。可选 `--background-index` 补充 Source A 前景，索引用 `schema: native_rl_background_index_v1`，`entries` 按 `source_id → pair_id` 存每条记录。记录含 `source_rgb`、`target_rgb` 和 `source_mask` 的 SHA256 文件引用，`source_mask` 使用 Source A 原图坐标，并与原 Native mask 取并集。若显式给出 `background_alignment`，`method`（`identity` 或 `content_resize`）、`canvas_id` 及 Source/target SHA 必须与 Native 绑定一致。

`--updates`（也可写 `--optimizer-updates`）按实际 `optimizer.step()` 计数，旧版命令的预算需按新语义重算。`rollout_batches` 包括零优势批，`generated_images` 统计训练候选，`replayed_transitions` 统计参与成功更新的转移。`--max-rollout-batches` 默认是更新预算的 5 倍，用于限制无有效优势时的生成成本。

`run.json` 记录模型、数据、人物、评估器 SHA 和固定资格。`rollout-*/rollout.json` 记录候选分数、PCK、ID 缺失、Swap、接触适用率、优势和样本图。`rollout-*/replay.json` 有 ratio、clip fraction、KL、梯度范数和显存。需要离线审计单步状态时加 `--keep-replay`。固定 ODE 图和逐样本指标在 `ode-*/` 与对应 JSON。

## 自动验证

```bash
NATIVE_RL_TEST_DEVICE=cuda "$PY" -m pytest -q tests/test_native_v62_rl.py tests/test_native_rl_resume.py
```

Qwen 集成测试使用小型真实 transformer 和 scheduler，验证 ODE、BF16 Replay、冻结参数和 KL。断点测试使用真实共享训练循环和小型合成策略，比对连续 10 步与第 5 步（Replay 组中间）暂停再恢复后的权重、AdamW、采样顺序和计数。真实任务的效果对照使用对应 SFT 基线及独立留出集。
