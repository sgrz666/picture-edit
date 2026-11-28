# Qwen 2.1 手部生成先验 LoRA

用 Qwen-Image-2.1 I2I DreamBooth 训练手势编辑 LoRA。L0 使用 A RGB 和编辑文本。L1 的条件按 `[A RGB, B Pose]` 传入，两条路线都以真实 B RGB 为监督。冻结 DiT 原权重、Qwen3-VL 和 VAE，只训练 32 层 Attention 的 Q/K/V/O LoRA。Flow Matching 在 target latent 上计算速度场损失。

## L1 双图条件

- `prepare_dataset.py --with-pose` 从 InterHand 配对清单和校验过 SHA256 的目标投影读取骨架。A RGB、B Pose、B RGB 共用不裁手的 512×512 letterbox。Pose 是黑底不透明 RGB，左右手使用不同的五指配色。只绘制通过逐视角标注、正深度、画内投影及已知遮挡检查的关节和骨骼。Parquet 记录 `pose_sha256` 和画布内可信关节坐标。缺少可信腕点或骨骼的目标写入导出记录的 `pose_rejections`，源文件或标注哈希错误会终止导出。
- `train.sh` 读取导出记录里的条件列。L1 在运行目录从锁定的官方脚本和 `qwen21_multiimg.patch` 生成 `train_dreambooth_lora_qwenimage21_multiimg.py`。官方 LoRA 注入方式、优化器、Flow Matching 目标和权重格式沿用原版。L1 使用固定 512×512、每卡 micro-batch 1，支持同机多卡和显式 checkpoint 恢复。
- `prepare_cache.py` 离线运行冻结的 Qwen3-VL 和 VAE。磁盘按 pair 保存三组 latent 和 Prompt embedding，训练逐对读取，只载入 DiT。DiT 每组各 1,024 个图像 Token，顺序为 A、Pose、noisy B。前两组的视觉 Token 区段分别校验 256 个 Slot，只从预测尾部取 B 计算损失。
- `verify_smoke.py` 加载指定 step 的 LoRA，使用官方 Pipeline 做双图推理。基座和 LoRA 各在两张目标 Pose 下出图。A、指令和 seed 固定，自动选择同一 A 的另一个真实目标帧，也可传 `--swap-index`。像素响应量用于确认条件变化，手势准确性要根据目标骨架和生成图判断。

### L1 实测

- RTX 6000D 运行同一主体、同一机位的 16 对右手帧，共 192 个 micro-batch、24 次 optimizer update。第 8、16、24 步均保存 checkpoint，第 24 步的 128 个 LoRA B 矩阵全部更新。重载文件的 SHA256 为 `0c31fdc4cd385669e6ec896cf5f5c2012262491bebcdfe86fdd42be352244565`。最终导出和它的 tensor payload 相同，文件头序列化顺序使文件 SHA 不同。
- 16 个磁盘分片共 81,092,384 字节，逐个复核了 SHA256。三个 latent 网格都是 32×32，两个视觉 Slot 的半开区段为 `[8,264)` 和 `[270,526)`。缓存清单绑定基座、Parquet、实际训练脚本和辅助代码 SHA。
- 在训练样本中固定 A，只换关节 RMS 相差 57.69 px 的两个目标。重载的 LoRA 生成图在两个目标之间切换手指弯曲与伸展。基座也有换 Pose 响应，`train-swap/` 保存两种模型的同 seed 对照。
- 独立留出从 12 个双手候选中通过可信骨架检查 5 个。选定的两个目标在 41 个共同关节上的 RMS 距离为 75.97 px。生成图的下方手腕会随 Pose 调整方向，弯指目标仍接近伸指的源手势。基座和 LoRA 的换图 RGB MAE 分别是 1.51 和 3.41（0–255），用于衡量输出变化。对照、目标、Pose 和文件 SHA 见 `val-swap/`。

### 数据筛选与遮挡控制

`prepare_interhand_contact_hand_adapter.py --hand-prior-quality --max-pairs N --geometry-only` 按官方主体切分、相同机位与序列提出 A/B。筛选使用腕点相对 3D 位移、指关节夹角、掌面方向和双手相对腕点，再检查两帧的有效手部面积和 64×64 手部 ROI 的 Laplacian 方差。按左右手、普通双手和部分可见双手的 20/20/35/25 目标配额，以及小、中、大动作桶抽样。清单和 audit 保留门槛、实际分桶与配额差额。部分可见代表逐视角有效关节不足，可用来采样困难样本。

`prepare_dataset.py --with-pose --rows-per-shard 128` 生成有序 Parquet 分片。默认 `--pose-mode visible`。遮挡消融用 `--pose-mode occlusion-dashed`，仅对通过可信 3D、正深度及已知遮挡判断的骨骼画低亮度虚线。隐藏关节不会进入可见 GT 的评分分母。同一实验的训练和推理须使用同一渲染模式。

### 扩容前实机验收

- 从已解包 InterHand 筛选 1,000 个帧对，20 位主体、520 段动作，全部导出为 8 个训练 Parquet 分片。最终左手 259、右手 259、普通双手 452、部分可见双手 30。动作幅度小、中、大分别为 335、326、339。困难类别仅达到目标配额的 30/250。`data/pilot-pairs/audit.json` 记录候选及出组门槛。
- 16 对工程回放验证了缓存中断续建 3+13 和完整缓存复用 16/16。开启 `DETERMINISTIC=1` 后，单卡连续 4 步与从第 2 步恢复到第 4 步的 LoRA、优化器、调度器及 RNG 文件 SHA256 一致。常规 CUDA 模式恢复的 RNG 和调度器一致，更新后的 BF16 张量有小幅差异。两张 RTX 6000D 实测完成一次 DDP 更新，并保存两份 rank RNG。
- 独立 SDPose-Wholebody + YOLO11x 在一位留出主体的两个双手目标上得到 Hand PCK@0.10 52.1% → 59.3%，腕点相对手势分 0.583 → 0.593。两组均有 4 只合格手，手检出率均为 100%。两张图的统计仅用于检验评分链路。人工标注、可信背景/肤色区域及带真值的接触样本需要补齐。
- 追加双卡续训验证时原 Qwen 基座路径在 `sg` 失效，双卡续训逐位一致性仍待复核。已生成的缓存、权重和评分可在预检目录查看。

### 在 sg 使用 L1

```bash
ssh sg
ROOT=/data/shangguanrz/LoRA/experiments/qwen21-hand-prior-lora
DUAL="$ROOT/output/dual-20261128T053122Z"
PREFLIGHT="$ROOT/output/preflight-20261128T061115Z"
PY=/data/shangguanrz/LoRA/envs/flow-factory-v050/bin/python
NEW="$ROOT/output/dual-replay-$(date -u +%Y%m%dT%H%M%SZ)"
export CUDA_VISIBLE_DEVICES=2
export QWEN_MODEL=/data/shangguanrz/pic-edit-models/Qwen-Image-2.1
export CONDITION_CACHE="$PREFLIGHT/cache-bound"
export DETERMINISTIC=1
export INIT_LORA="$DUAL/train-l1/weights/checkpoint-24/pytorch_lora_weights.safetensors"
export INIT_LORA_SHA256=0c31fdc4cd385669e6ec896cf5f5c2012262491bebcdfe86fdd42be352244565
BASE="$DUAL/train-l1/base.sha256"
"$PY" "$ROOT/scripts/prepare_cache.py" --dataset "$PREFLIGHT/data/train" \
  --base "$QWEN_MODEL" --base-sha256 "$BASE" --out "$CONDITION_CACHE" --hf-cache "$PREFLIGHT/hf-cache"
MAX_STEPS=4 CHECKPOINT_STEPS=2 GRAD_ACCUM=2 bash "$ROOT/scripts/train.sh" \
  "$PREFLIGHT/data/train" "$BASE" "$NEW"
RESUME_CHECKPOINT="$NEW/weights/checkpoint-2" MAX_STEPS=4 CHECKPOINT_STEPS=2 GRAD_ACCUM=2 \
  bash "$ROOT/scripts/train.sh" "$PREFLIGHT/data/train" "$BASE" "$NEW-resumed"
"$PY" "$ROOT/scripts/verify_smoke.py" --run "$NEW-resumed" --base "$QWEN_MODEL" --checkpoint-step 4 \
  --dataset "$DUAL/data/val-motion" --split validation --index 3 --swap-index 0 \
  --steps 20 --out "$NEW/val-swap"
```

缓存续建时使用同一命令和同一 `--out`，已有分片逐个复核，仅计算缺失项。`--max-new-pairs` 可分批运行。完成后的 `index.json` 绑定基座、数据分片、编码代码、版本、精度、设备型号、seed 和分片 SHA。训练恢复用新输出目录、同一缓存和同一训练配置。显式从旧 LoRA 初始化时同时传 `INIT_LORA` 和 `INIT_LORA_SHA256`，继续该运行时沿用两个值。固定拓扑可设 `DETERMINISTIC=1`，从实验起点和恢复两侧均需开启。同机两卡用 `CUDA_VISIBLE_DEVICES=0,2`。

上面 `QWEN_MODEL` 写的是预检时的路径。模型迁址后将它改成实际目录，缓存、训练和推理都会按原 `base.sha256` 核对完整模型。若编码实现发生变化，为新指纹建一个缓存目录。

换 Pose 验收的数据用 `--targets-per-source 2` 保留同一个 A 的两个真实 B；一般留出行可用 `verify_smoke.py --no-swap`。在 `sg` 从实验脚本目录运行 Pose 导出或评测需将仓库快照放入 `PYTHONPATH`，例如 `PYTHONPATH="$PREFLIGHT/selection-source"`。

独立评分使用项目的 SDPose 环境运行 `evaluate_hands.py --comparison "$NEW/val-swap" --out "$NEW/scores"`。可以重复传 `--comparison` 汇总多个留出样本。输出包括手框对角线归一化的 Hand PCK@0.10、腕点相对手势分和检出率，漏检保留在分母。提供对齐且带 SHA 的前景和皮肤 mask 时可计算背景 MAE 和肤色直方图相似度。`blind/` 可交给人工标注，`blind-key.json` 用于复核时解码。用 `--marked-review review.jsonl --blind-key blind-key.json --out NEW_DIR` 汇总多指、少指和粘连率。

普通 Flow Matching 留作首批千级训练的对照。空间加权使用相同父权重、数据与训练预算另跑，待手部、腕部、背景 mask 及独立留出指标通过检查后再比较。

## L0 单图基线

### 文件和落点

- `scripts/prepare_dataset.py` 从现有 InterHand 帧对 JSONL 导出 Hugging Face Datasets 可直接读取的 Parquet。验证源和目标图像的 SHA256、尺寸、主体和图像的 split 归属。RGB 原始字节写入两个 Image 列，`caption` 可填人工核对过的目标手势描述。默认按标注中的左手、右手或双手生成通用编辑指令，只适合流程冒烟。
- `scripts/train.sh` 锁定训练脚本、Parquet 和基座 SHA256，用独立 LoRA 环境运行原版 Diffusers。
- `scripts/verify_smoke.py` 校验 LoRA tensor 名称、形状、非零 B 矩阵及 checkpoint 文件，并用同一个 Source、prompt 和 seed 出基座与重载 LoRA 的对照图。
- 本地 `tests/test_qwen21_hand_prior_dataset.py` 覆盖图像顺序、caption、SHA、尺寸和主体跨 split 泄漏。
- `sg` 的运行工作区为 `/data/shangguanrz/LoRA/experiments/qwen21-hand-prior-lora`，数据、权重、训练日志和生成图片都在每轮对应的 `output/` 内。该工作区沿用已有的 LoRA 实验族和官方源码快照。

### 实测

- `partap` 和 `partbn` 校验通过后各取一段 RGB。训练 8 对来自官方 train 主体 1，留出 2 对来自官方 val 主体 9。两组原始 RGB SHA 无交集。
- 实机使用 RTX 6000D、Diffusers 0.42.1、Torch 2.10.0+cu128，完成 8 次 micro-batch 和 1 次 optimizer update。末个 micro-batch 的 Flow Matching loss 为 0.152。Adapter SHA256 为 `0c5c2553e72a4c6ed378e052d559a24c4f3ac22251a27e243511d945bdbcea54`，对应 `train-tensorboard/weights/` 的最终导出和 `checkpoint-1/` 内的副本。
- 128 个投影共 256 个 LoRA tensor、16,777,216 个参数，128 个 B 矩阵均已更新。官方 Pipeline 成功重载 rank/alpha 16/16，在独立双手样本上用 seed 42 各生成 1 张 512×512 对照图。两张图的 RGB MAE 为 0.2853（0–255）。`comparison/` 包含 Source、Target、基座和 LoRA PNG，以及逐文件 SHA 与验收记录 `verification.json`。两张生成图都更接近 Source 的手势，本轮只验收训练和重载链路。
- 基座 24 个运行文件的清单 SHA256 为 `12075ae6ef2eed213190c27893db7816893dcc409ba2094a6bd6b8a90db51139`。第一次启动使用的 `report_to=none` 被 Accelerate 拒绝，修订后显式采用官方默认的本地 TensorBoard。

### 在 sg 复跑

本轮 `data/train` 和 `data/validation` 已从 InterHand2.6M 的官方训练和验证分卷导出。原始帧、跨帧选择记录、源与目标 RGB SHA、分卷 MD5 和基座 SHA 清单都在同一轮目录。基座来自 Hugging Face 的 `Qwen/Qwen-Image-2.1` revision `790c92633540aa0cb11d9abf19eb46d861714758`，本轮初始化直接使用官方权重，LoRA B 以零初始化。原始 I2I 训练脚本固定为 Diffusers `0905b667fe3f37f43b6ef9c6e05ad20636206938`。

```bash
ssh sg
ROOT=/data/shangguanrz/LoRA/experiments/qwen21-hand-prior-lora
SMOKE="$ROOT/output/smoke-20261128T044810Z"
PY=/data/shangguanrz/LoRA/envs/flow-factory-v050/bin/python
NEW="$ROOT/output/replay-$(date -u +%Y%m%dT%H%M%SZ)"
export CUDA_VISIBLE_DEVICES=0
bash "$ROOT/scripts/train.sh" "$SMOKE/data/train" "$SMOKE/logs/base.sha256" "$NEW"
"$PY" "$ROOT/scripts/verify_smoke.py" --run "$NEW" --checkpoint-step 1 \
  --dataset "$SMOKE/data/validation" --split validation --out "$NEW/comparison"
```

默认 BF16、512×512、rank/alpha 16/16、AdamW (1e-4)、batch size 1、梯度累积 8、seed 42，做 1 次 optimizer update。`MAX_STEPS`、`CHECKPOINT_STEPS` 和 `GRAD_ACCUM` 可设，复跑要使用新目录。基座核验会逐个读取 24 个运行文件。官方脚本会预处理全部训练图并为每对缓存 prompt embedding，同时开启 latent 缓存。扩大到万对前需先测量 CPU 内存和显存，再决定分批编码或复用缓存的方式。

本地数据契约测试可运行 `python -m pytest -q tests/test_qwen21_hand_prior_dataset.py tests/test_interhand_contact_hand_adapter.py`。

重新准备数据时，先用仓库的 `scripts/data/prepare_interhand_contact_hand_adapter.py` 按官方 split 导出真实帧对，再分别执行以下命令。LoRA 导出仅消费 RGB 和编辑文本，配对阶段的几何记录可用于筛选及诊断。

```bash
"$PY" "$ROOT/scripts/prepare_dataset.py" --manifest /path/to/pairs/manifest.jsonl \
  --split train --out /path/to/new/train
"$PY" "$ROOT/scripts/prepare_dataset.py" --manifest /path/to/pairs/manifest.jsonl \
  --split val --out /path/to/new/validation
```

## 与其他实验的关系

当前 LoRA 与 Contact Hand Adapter 分别训练、分别保存。后续局部引导和独立手部质量判别器的迁移方案见 [MGHanD 复用记录](MGHAND.md)。手势姿态评测继续采用仓库的 SDPose-Wholebody + YOLO11x 口径。手指解剖错误需要独立标注，检测置信度不能代替多指、少指、粘连的判断。
