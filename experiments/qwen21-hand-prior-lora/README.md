# Qwen 2.1 手部生成先验 LoRA

用 Qwen-Image-2.1 官方 I2I DreamBooth 脚本验证手势编辑 LoRA 的训练和重载。条件为 Source RGB 与编辑指令，监督为另一真实帧的 Target RGB。冻结 DiT 原权重、Qwen3-VL 和 VAE，只训练 32 层 Attention 的 Q/K/V/O LoRA。Flow Matching 在 target latent 上计算速度场损失。

## 文件和落点

- `scripts/prepare_dataset.py` 从现有 InterHand 帧对 JSONL 导出 Hugging Face Datasets 可直接读取的 Parquet。验证源和目标图像的 SHA256、尺寸、主体和图像的 split 归属。RGB 原始字节写入两个 Image 列，`caption` 可填人工核对过的目标手势描述。默认按标注中的左手、右手或双手生成通用编辑指令，只适合流程冒烟。
- `scripts/train.sh` 锁定训练脚本、Parquet 和基座 SHA256，用独立 LoRA 环境运行原版 Diffusers。
- `scripts/verify_smoke.py` 校验 LoRA tensor 名称、形状、非零 B 矩阵及 checkpoint 文件，并用同一个 Source、prompt 和 seed 出基座与重载 LoRA 的对照图。
- 本地 `tests/test_qwen21_hand_prior_dataset.py` 覆盖图像顺序、caption、SHA、尺寸和主体跨 split 泄漏。
- `sg` 的运行工作区为 `/data/shangguanrz/LoRA/experiments/qwen21-hand-prior-lora`，数据、权重、训练日志和生成图片都在每轮对应的 `output/` 内。该工作区沿用已有的 LoRA 实验族和官方源码快照。

## 本轮实测

- `partap` 和 `partbn` 校验通过后各取一段 RGB。训练 8 对来自官方 train 主体 1，留出 2 对来自官方 val 主体 9。两组原始 RGB SHA 无交集。
- 实机使用 RTX 6000D、Diffusers 0.42.1、Torch 2.10.0+cu128，完成 8 次 micro-batch 和 1 次 optimizer update。末个 micro-batch 的 Flow Matching loss 为 0.152。Adapter SHA256 为 `0c5c2553e72a4c6ed378e052d559a24c4f3ac22251a27e243511d945bdbcea54`，对应 `train-tensorboard/weights/` 的最终导出和 `checkpoint-1/` 内的副本。
- 128 个投影共 256 个 LoRA tensor、16,777,216 个参数，128 个 B 矩阵均已更新。官方 Pipeline 成功重载 rank/alpha 16/16，在独立双手样本上用 seed 42 各生成 1 张 512×512 对照图。两张图的 RGB MAE 为 0.2853（0–255）。`comparison/` 包含 Source、Target、基座和 LoRA PNG，以及逐文件 SHA 与验收记录 `verification.json`。两张生成图都更接近 Source 的手势，本轮只验收训练和重载链路。
- 基座 24 个运行文件的清单 SHA256 为 `12075ae6ef2eed213190c27893db7816893dcc409ba2094a6bd6b8a90db51139`。第一次启动使用的 `report_to=none` 被 Accelerate 拒绝，修订后显式采用官方默认的本地 TensorBoard。

## 在 sg 复跑

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
