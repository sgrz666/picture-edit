# Qwen 2.1 Union 数据集适配：方案 B

## 实验目的与当前状态

检验官方预训练 Union 接入人物参考图后，控制分支 LoRA 能否改善单人、双人姿态控制；在相同数据、公共初始化、采样和更新预算下比较 joint 与 target-only 残差注入。正式组为 `native`、`union_frozen`、`union_lora_joint`、`union_lora_target`。

2026-10-16 已完成四数据集读取、分组审计、锁定抽样、条件准备和 8 张坐标检查面板。真实数据为 train **1536**、val **72**、test **128**；overfit6 是 train 内每数据集两条，共 6 条，不额外增加训练样本。本地 CPU 契约测试为 **116 passed、1 skipped**，服务器为 **117 passed**。真实 GPU smoke `smoke_deterministic_20261016_v2` 已通过：连续 4 updates 与保存恢复后的 2+2 updates，最终 LoRA 和 optimizer 的最大绝对差均为 **0**；生成了 iPER/Hi4D 两张两步图，峰值 allocated 约 **27.88 GiB**、reserved 约 **29.13 GiB**。后续安全检查完善后代码签名已变化，正式串行队列已启动并重新跑当前签名的 smoke。200 updates overfit、两组 512 updates 正式训练及正式四组测试尚未完成，当前不能判断训练收益或注入范围优劣。运行状态以具体 run 的 `gate.json`、`summary.json` 为准。

本次串行队列名为 `pilot_b_20261016T1841`，启动 PID 为 `857392`，总日志为 `/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1/logs/queue_pilot_b_20261016T1841.log`。查看真实进展：`ssh sg 'tail -n 8 /data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1/queues/pilot_b_20261016T1841/events.jsonl'`。队列在服务器独立运行，SSH 断开不影响执行；先查看状态，再决定是否需要新建另一轮，避免重复占用 GPU。

## 里面有什么与执行位置

本目录只保存可复现入口与说明：`scripts/run.py` 是 CLI；`scripts/execute.py` 串行执行门禁、训练与评测；`configs/protocol.json` 是 `role=protocol_snapshot` 的协议记录，当前 CLI 不读取它作为运行配置。运行约束由 `src/pose_control/qwen21_union/` 和数据 lock 检查，修改 JSON 不会改变训练参数。

| 路径 | 实际用途 |
|---|---|
| `/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1/code/` | 本次运行的独立源码快照、测试和实验入口 |
| `/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1/env/bin/python` | 方案 B 独立环境；torch 2.8.0+cu128、diffusers 0.41.0.dev0、transformers 5.17.0 |
| `/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1/` | 本次数据、缓存、runs、报告的产物根 |
| `/home/shangguanrz/project/pic-edit` | 现有数据、几何工具、评估配置的项目根，通过 `--project-root` 指定 |
| `/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_dataset_adaptation_v1/` | 远端主仓旧实验族结构中的对应归属；本次命令使用上面的 `/data/.../code/` 快照入口 |

本地实验目录平铺在 `experiments/`；远端主仓仍使用 `2.qwen-test` 等旧分组。不要把快照中的入口路径与远端主仓的实验族目录混用。基座复用 `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1`；官方源码与 Union 权重复用方案 A 的 `qwen21_union_migration_v1/source/VideoX-Fun` 和 `models/Qwen-Image-2.1-Fun-Controlnet-Union`，不修改其固定版本或现有 qwen21 环境。

大产物根中的文件：

- `data/{train,val,test}_{conditions,supervision}.jsonl`：条件与监督物理分离。条件中的 source、人物 Pose 参考图、combined control 均已准备为 512×512；target RGB 只在监督清单中保留原路径与 transform。
- `data/overfit6_{conditions,supervision}.jsonl`：固定六条 train 样本，只验证可学习性；其训练权重不作为正式训练初始化。
- `data/lock.json`：原清单/选择清单/原始必需资产 SHA256、组交集、实际配额、逐条排除原因、代码与模型版本。
- `data/prepared_assets_sha256.json`：绑定 lock SHA256，登记 6880 个 prepared 资产，其中 4608 张 RGB 条件图、2272 份关键点 JSON。
- `report/data_qa/`：每数据集两张面板，共 8 张；`summary.json` 记录尺寸、偏移、人物编号与有效点数，`validation.json` 记录完整数据校验结果。面板右侧目标 RGB 只用于 QA。
- `cache/`：条件缓存及仅 train 的监督 latent 缓存。缓存包含模型/代码/图像/提示词/参考顺序/变换指纹，命中不消耗 denoising RNG。
- `runs/<name>/`：工程门禁、训练与推理产物；run 名由调用者传入，重名拒绝覆盖。训练 checkpoint 保存 LoRA、optimizer、LR、RNG、采样与日志游标及协议签名。
- `queues/<name>/`：串行队列的 `events.jsonl` 与逐阶段日志。每阶段记录实际命令、退出码与耗时，门禁失败或训练未满 512 updates 时停止。
- `initialization/lora_rank16_seed3407.pt`：训练初始化时创建的公共 LoRA 初始化；正式两个 arm 必须读取同一文件。
- `environment_20261016T172034.json`：本次独立环境与官方源码位置记录。

仓库 `.gitignore` 当前忽略整个 `experiments/*`。提交时只显式 `git add -f` 这个实验的 README、`configs/protocol.json` 和小脚本；不要把 data、cache、runs、图片、权重或环境加入 Git。

## 已验证事实、协议与使用

| 数据集 | train | val | test | 防泄漏分组 |
|---|---:|---:|---:|---|
| iPER | 512 | 24 | 32 | appearance；训练读取 `dev_train_pairs.jsonl` |
| TikTok | 512 | 24 | 32 | strict_v1 的 source_video_group，再按 sequence 轮转 |
| Hi4D | 512 | 24 | 32 | 人物 pair；按动作族、same/cross-action、实际接触数分层 |
| 3DPW | 0 | 0 | 32 | sequence/pose_bin；仅 test 参与本轮图像处理与评估 |
| 合计 | 1536 | 72 | 128 | overfit6 从 train 内取子集 |

抽样 seed 为 20261015，排序使用 `SHA256("union_b_v1:" + sample_id)`。目标按 canonical path 与解码 RGB 像素去重；iPER/TikTok 同序列至少间隔 8 帧，Hi4D 同 action/camera 至少间隔 2 帧，3DPW source/target 至少相隔 8 帧。正式筛选排除了 38464 条重复目标对应的 source 行、2675 条时间近邻、247 条 3DPW 近帧配对，没有降低间隔或借用其他 split。筛选后候选数如下：

| 数据集 | train 候选 | val 候选 | test 候选 |
|---|---:|---:|---:|
| iPER | 3626 | 526 | 1067 |
| TikTok | 3380 | 443 | 437 |
| Hi4D | 594 | 161 | 240 |
| 3DPW | 不作训练抽样 | 不作调参抽样 | 346 |

`lock.json` SHA256 为 `e61eef1e5bc4b2fb24593c28639e425280aeb85263329699e6d2d829d41fe503`；prepared registry SHA256 为 `328bbb029a62946f97aa6a2386e379ef5610e6b4075bb27002a68603cff7d7b8`。当前所有 appearance/video_group/pair/sequence 的 split 交集为空，见 `lock.json` 的 `audit.group_intersections`。另一个不同层级的事实是：2026-10-15 原清单审计中，iPER 人物编号前缀在 train/val、train/test、val/test 分别交集 7、9、5 个；因此称 appearance 留出，不称完全未见身份。新的 `audit` 命令会重新记录当前前缀交集，不能把历史数字当成新审计结果。

源图与目标图分别记录原尺寸、实际四舍五入的 resize 尺寸和补边偏移。iPER 原 RGB 为 1024×1024、Pose 为 512×512；TikTok 为 604×1080；Hi4D 为 940×1280；本次 3DPW 样本为 1080×1920。RGB 补白，Pose 补黑，mask 用 nearest；关键点使用实际 resize 后的 x/y 比例。已有 134 点不再次插 neck。Hi4D 用源 mask 中心决定参考图左右顺序，所有目标人物使用同一 target transform；confidence≥0.25 且画内的点才用于重绘，失效点以 NaN 排除。严格评估仍保留高置信度出画请求：本次共有 1034 个，不能通过删点提高分数。

固定控制结构为 16 个官方控制块、129 通道；基座与原始 Union 全冻结，只训练精确 130 个 Linear 的 LoRA A/B，共 23,267,344 个 FP32 参数，rank=alpha=16、dropout=0。训练使用 AdamW lr=1e-4、32 updates warmup 后 constant、global gradient norm=1、BF16 计算。每次更新依次取 iPER/TikTok/Hi4D 各一条，micro-batch=1、accumulation=3，512 updates 正好各经过锁定的 512 条一次。两组只改变残差注入范围，共享初始化、sample/noise/timestep 轨迹。VAE 固定 posterior mode；缓存的目标 latent 只在训练监督路径读取。

首轮 `smoke_20261016T1755` 保留了失败证据：相同初态的梯度存在偏差，最终 LoRA 最大差约 1.37e-5，未达到 1e-6 门槛。第二轮在首次 CUDA 初始化前固定 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`torch.use_deterministic_algorithms(True, warn_only=False)`、`cudnn.benchmark=False`、`cudnn.deterministic=True` 后恢复检查为 0 差异，门槛没有放宽。该确定性策略计入缓存、训练与门禁签名，两组完全一致。依据为 [PyTorch 2.8 reproducibility 文档](https://docs.pytorch.org/docs/2.8/notes/randomness.html)；这项对照不能定位到某一个具体 CUDA kernel。

下面命令在服务器执行。先定义一次变量；`UNION_SESSION` 同时绑定两个训练 run 和最终测试 run，不要在中途重新赋值：

```bash
UNION_WORK=/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1
UNION_PY="$UNION_WORK/env/bin/python"
UNION_CODE="$UNION_WORK/code"
UNION_CLI="$UNION_CODE/experiments/qwen21_union_dataset_adaptation_v1/scripts/run.py"
UNION_PROJECT=/home/shangguanrz/project/pic-edit
UNION_SESSION=$(date +%Y%m%dT%H%M%S)
cd "$UNION_CODE"

# 读取状态、复核当前原清单；audit 保存带时间戳的新文件。
"$UNION_PY" "$UNION_CLI" status --workspace "$UNION_WORK"
"$UNION_PY" "$UNION_CLI" audit --workspace "$UNION_WORK" --project-root "$UNION_PROJECT"

# 运行 CPU 契约测试。它们不能代替真实 GPU smoke。
"$UNION_PY" -m pytest tests/test_qwen21_union_*.py -q -m 'not gpu'

# 推荐完整执行入口：各阶段通过后自动衔接，失败立即停止。
# 此命令与下方手工逐阶段命令二选一，不能同时重复启动。
"$UNION_PY" "$UNION_CODE/experiments/qwen21_union_dataset_adaptation_v1/scripts/execute.py" \
  --workspace "$UNION_WORK" --name "pilot_b_$UNION_SESSION"
```

队列顺序为六条缓存 → 当前签名 smoke → overfit200 → 完整缓存 → val12 的五组诊断（60 张）及评估 → joint512 → target512 → test128 四组双 seed（1024 张）及评估。每组正式训练在 step0/256/512 另外生成 val72。只有已有 smoke 与当前代码、初始化、数据和环境签名完全一致时，才可以加 `--reuse-smoke <run 名>`；旧签名会立即拒绝。

查看队列日志：`tail -f "$UNION_WORK/queues/pilot_b_$UNION_SESSION/events.jsonl"`。需要在当前阶段结束后停止后续阶段时，创建 `touch "$UNION_WORK/queues/pilot_b_$UNION_SESSION/STOP"`；它不会中断当前训练或推理。停止或失败的队列不能用同名重跑；检查对应阶段日志，使用新名字重新执行或按下方单阶段命令处理。数据与缓存复用均先核验 SHA256。

手工逐阶段执行使用同一组变量：

```bash

# 先缓存六条 train 条件及其监督，随后依次跑工程门禁与可学习性门禁。
"$UNION_PY" "$UNION_CLI" cache --workspace "$UNION_WORK" --conditions overfit6 --supervision train
"$UNION_PY" "$UNION_CLI" smoke --workspace "$UNION_WORK" --name "smoke_$UNION_SESSION"
"$UNION_PY" "$UNION_CLI" overfit --workspace "$UNION_WORK" --name "overfit6_$UNION_SESSION" --updates 200

# 两个 gate.json 均通过且签名与当前代码/数据一致后，再执行正式阶段。
"$UNION_PY" "$UNION_CLI" cache --workspace "$UNION_WORK" --conditions train,val,test --supervision train
"$UNION_PY" "$UNION_CLI" train --workspace "$UNION_WORK" --arm union_lora_joint --name "joint_$UNION_SESSION" --updates 512
"$UNION_PY" "$UNION_CLI" train --workspace "$UNION_WORK" --arm union_lora_target --name "target_$UNION_SESSION" --updates 512

# 正式 test 必须是完整 128 条、四个 arm、两个 seed、40 steps，共 1024 张。
"$UNION_PY" "$UNION_CLI" infer --workspace "$UNION_WORK" --split test --name "test128_$UNION_SESSION" \
  --arms native,union_frozen,union_lora_joint,union_lora_target --seeds 42,43 --steps 40 \
  --joint-checkpoint "$UNION_WORK/runs/joint_$UNION_SESSION/checkpoint_000512" \
  --target-checkpoint "$UNION_WORK/runs/target_$UNION_SESSION/checkpoint_000512"
"$UNION_PY" "$UNION_CLI" evaluate --workspace "$UNION_WORK" --project-root "$UNION_PROJECT" --run "test128_$UNION_SESSION"
"$UNION_PY" "$UNION_CLI" report --workspace "$UNION_WORK" --run "test128_$UNION_SESSION"
```

smoke 包括实际 CUDA/BF16 计算、10 microsteps 探针、冻结权重检查、连续 4 updates 对比保存恢复后的 2+2 updates，以及 iPER/Hi4D 各一张两步生成。overfit6 固定 200 updates、accumulation=1，在 indices 250/500/750、noise seed=43 的 18 个固定位置检查平均 loss 至少下降 20%；失败时不自动增加预算。正式训练只接受通过且签名一致的 smoke/overfit 门禁，超过显存 90% 或 OOM 会停止并记录报告。

正式推理固定 512×512、40 steps、CFG=1、Union control scale=1、BF16、关闭 KV cache、seeds 42/43。验证在 step0/256/512 使用同样 72 条、seed42；最终固定取 step512，不根据 test 选择 checkpoint。恢复训练用新的 `--name` 并显式传入 `--resume <原 checkpoint 绝对路径>`，旧 run 保留。

本次 workspace 已有 data lock，不能再次在相同目录执行 `select`。需要重建时指定全新的产物根，例如：

```bash
UNION_REBUILD="/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_rebuild_$UNION_SESSION"
"$UNION_PY" "$UNION_CLI" select --workspace "$UNION_REBUILD" --project-root "$UNION_PROJECT" --protocol pilot_b_v1
```

新建 workspace 不复制已有 run，也不自动改写旧清单中的绝对路径。源 datasets、原始媒体和既有产物均保留原位置。

## 与其他实验的关系与结论边界

本实验复用方案 A 的官方固定版本加载与参考图接入证据，是独立的数据集适配训练实验；不覆盖旧 Adapter、workflow 或方案 A 的结果。VideoX-Fun 固定提交为 `13bb3c625e8b1e457d7124d92207b49e00e4a6fe`，Union revision 为 `8a4702014d4dabb5f896fcba917e2ee0a961465f`。后续可把本次 lock、训练轨迹、checkpoint 和配对报告作为扩大数据实验的上游，当前数据 QA 与 CPU 契约不构成效果结论。

Hi4D 使用的现成 DWPose 资产在上游利用过目标 mask，报告必须标为现成数据资产协议；`target_contact=true` 不代表接触顶点数大于零，按实际数量分层。Hi4D test 只有两个 pair，保留逐 pair 结果，不作人口级显著性解释。3DPW 不参与训练或调参，当前没有可直接使用的数值关键点；评估用同一冻结 DWPose 从目标 RGB 估计，标为“GT RGB 估计姿态”。严格 PCK 同时报告 detector coverage 与仅画内版本；DINO 是外观代理，ArcFace 缺失检测不填 0 充当相似度。目标 RGB、mask、latent 不进入推理条件或提示词。
