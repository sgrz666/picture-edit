# experiments/qwen21_iper_body/ —— Qwen-Image-2.1 单流 7B 身体姿态适配器在 iPER 上的训练与四臂定量评测

> 工作区路径：`/home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body`　|　本级：16 个子目录 / 0 个直属文件　|　子树规模：约 820 个文件、逻辑体积 7.3 GB（含 `full_20k` 1.6 GB + 三个 `pilot_3k*` 各约 0.8–1.6 GB 检查点）
> 数据来源：本文件数字读自各 `summary.json` / `run_manifest.json` / `train_metrics.jsonl` / `eval_metrics.json`，读取时间 2026-10-12。

## 一、这一层在干什么

把项目的姿态控制从 DeepGen-1.0（SD3 MMDiT 2.47B）迁移到 **Qwen-Image-2.1（7B 单流 DiT）** 基座上，用 `src/pose_control/qwen21/body_adapter.py` 训练一个单流身体适配器，并在同一批 50 对 held-out iPER 样本上做**四臂交叉评测**（DeepGen 关/开 × Qwen21 关/开）。这是本项目的里程碑：结论是 `qwen21_on` 在保真度与姿态对齐上全面碾压 DeepGen（PSNR 25.86 dB、PCK@0.02 98.00%），同时暴露出"身体适配器形变会摧毁面部身份保持"这一核心矛盾（Face ID 从 0.8187 跌到 0.3510）。

上游：`full_20k` 的初始化继承自 `experiments/iper_v6_native_20k_v1/checkpoint_last.pt`（DeepGen 侧训好的 V6 条件），下游：`experiments/qwen21_hi4d_dual/`（双人线）与 `experiments/16-test/`（综合基准）都直接引用本目录的 `full_20k/checkpoint_best.pt` / `checkpoint_last.pt`。

## 二、层级地图

| 子目录 | 装什么 | 内容实证（真实文件名/数值） | 与本级其他子目录关系 |
| :--- | :--- | :--- | :--- |
| `smoke/` | 1 步冒烟 | `checkpoint_last.pt` 771.9 MB + `train_metrics.jsonl`（仅 1 行 step=1, loss 0.1385） | 冒烟：验证训练循环可跑 |
| `smoke_parity/` | 冒烟复现一致性对照 | 结构与 `smoke/` 逐字节同（1 行 metrics, `checkpoint_last.pt` 771.9 MB） | 冒烟：验证同种子可复现 |
| `controlled_smoke/` | 带几何控制的冒烟 | `images/` 5 文件、约 1 MB | 冒烟：验证控制注入路径 |
| `native_smoke/` | 关控制（原生基座）冒烟 | `images/` 4 文件、约 610 KB | 冒烟：对照组 |
| `pilot_3k/` | 3,000 步先导训练 | `train_metrics.jsonl` 首行 step=1；`checkpoint_*` 约 1.6 GB | 训练谱系第 1 站 |
| `pilot_3k_acc4/` | 梯度累积=4 的先导 | `run_manifest.json` `accumulation`；273 行 metrics，首行 step=1501 | 谱系第 2 站（改累积） |
| `pilot_3k_acc4_resumed/` | 续训 `pilot_3k_acc4` | 首行 step=1501 loss 0.2231；`checkpoint_last.pt` 被 `full_20k` 用作 resume | 谱系第 3 站 |
| **`full_20k/`** | **20,000 步正式全量训练（权威）** | `checkpoint_best.pt`/`checkpoint_last.pt` 各 771.9 MB；metrics 17,000 行；末行 step=20000 loss 0.2886, val_loss 0.3465, elapsed 48,617s, peak 40.68 GiB | 下游消费的权重源 |
| `pilot_dev_eval_8` | 8 样本 dev 快评 | `summary.json` num_samples=8, steps=40 | 评测：早期小样 |
| `eval_50/` | Qwen21 单臂 50 样本（steps=40） | `summary.json` checkpoint=full_20k/checkpoint_last.pt | 评测：单基座 |
| `eval_50_s30/` | 同上，steps=30 | steps=30 | 评测：步数消融 |
| `deepgen_eval_50/` | DeepGen 单臂 50 样本 on/off | `eval_metrics.json` adapter_on PSNR 24.51/SSIM 0.9755, off 见 `summary_adapter_off` | 评测：DeepGen 参照 |
| `deepgen_eval_50_s40/` | 同上，40 步采样 | adapter_on avg_psnr 24.50, avg_mae 6.63 | 评测：DeepGen 参照 |
| `four_arm_eval_50/` | 四臂交叉评测（初版，含 kv_cache 计时） | 见 §三 | 评测：跨架构对照 |
| `four_arm_eval_50_s30/` | 四臂，steps=30 | qwen21_on PSNR 25.91 | 评测：步数消融 |
| **`four_arm_eval_50_s40/`** | **四臂最终权威，steps=40** | 见 §三核心表 | 评测：里程碑裁决 |

## 三、关键文件与实证内容

**`four_arm_eval_50_s40/summary.json`（50 held-out 对，seed 42，四臂均值）**——`pairs` 指向 `datasets/iPER/iper_sampled_6src64tgt/splits/baseline_50_eval_pairs.jsonl`，训练身份 69、测试身份 19、重叠 0（严格 held-out）：

| 指标 | `deepgen_off` | `deepgen_on` | `qwen21_off` | `qwen21_on` |
| :--- | :---: | :---: | :---: | :---: |
| PSNR (dB) | 17.24 | 24.53 | 17.53 | **25.86** |
| SSIM | 0.8017 | 0.8756 | 0.8363 | **0.8969** |
| MAE (0–255) | 16.26 | 6.58 | 13.61 | **4.93** |
| PCK@0.02 | 17.44% | 95.50% | 20.31% | **98.00%** |
| PCK@0.05 | 48.48% | 98.12% | 47.05% | **99.18%** |
| 人体前景 MAE | 60.01 | 20.03 | 61.16 | **17.57** |
| 背景 MAE(vs source) | 9.97 | 6.49 | **3.03** | 5.31 |
| Face ID cos (n=47) | 0.2612 | 0.2917 | **0.8187** | 0.3510 |
| 单图耗时 (s) | **5.08** | 5.94 | 5.84 | 5.87 |
| 峰值显存 (GiB) | **14.51** | 14.53 | 32.76 | 32.76 |

`s40` 的 `paired_deltas`（bootstrap CI95，5000 次有放回重采样 seed 20260927）给出四条配对差：
- `deepgen_control_gain`（on−off）：PSNR **+7.29 dB**、PCK@0.02 **+0.781**、Face ID 仅 +0.031（DeepGen 本就没什么面部先验）。
- `qwen21_control_gain`（on−off）：PSNR **+8.33 dB**、PCK@0.02 +0.777，但 **Face ID −0.468（CI95 [−0.515,−0.418]）** —— 开启身体适配器把身份余弦硬生生砍掉一半，这是本项目当前主攻痛点的数据证据。
- `controlled_model_change`（qwen21_on − deepgen_on）：PSNR +1.33 dB、PCK@0.02 +0.025、Face ID +0.059；Qwen21 受控臂整体优于 DeepGen 受控臂。
- `backbone_native_change`（qwen21_off − deepgen_off）：Face ID +0.558 —— 未加适配器时 Qwen21 基座自带极强面部先验。

**`four_arm_eval_50_s40/` 与 `four_arm_eval_50/`、`_s30/` 的区别**：三者臂配置一致，仅推理步数/缓存策略不同（s30=30 步；base 与 s40=40 步，`qwen21.kv_cache=true`）。三者的 `qwen21_on` PSNR 都稳定在 25.86–25.91、PCK@0.02 98%，结论不随步数翻转。**s40 是本目录对外引用的权威版**。每目录 `strips/` 存 50 张四臂拼图。

**`full_20k/run_manifest.json`**：`model=Qwen-Image-2.1`，`v6_checkpoint=iper_v6_native_20k_v1/checkpoint_last.pt`，`resume=pilot_3k_acc4_resumed/checkpoint_last.pt`，`cache=/home/shangguanrz/nihaochen/qwen21_iper_cache`，`max_steps=20000, accumulation=2, micro_batch_size=2, save_every=500, seed=42`；`git_head=b1a00c2…`，torch 2.8.0+cu12.8，GPU=NVIDIA RTX 6000D。

## 四、注意事项 / 调用路径

- **当前权威权重 = `full_20k/checkpoint_best.pt`**（20k 训练中 val 最优）。下游：`qwen21_hi4d_dual/run_manifest.json` 的 `single_checkpoint` 与 `16-test` 的 qwen_adapter 引用此路径；四臂评测脚本用的是 `full_20k/checkpoint_last.pt`。二者都出自同一次训练，勿混淆"评测用 last、复用初始化用 best"。
- **`/home/shangguanrz/nihaochen/qwen21_iper_cache` 是训练/评测共享的离线潜变量缓存（不可删除）**，位于他人工作目录下，本工具链无权进入；如需重跑评测须确认该缓存完好。
- 冒烟目录（`smoke/`、`smoke_parity/`、`controlled_smoke/`、`native_smoke/`）仅 1–5 步或几张小图，用于链路连通性，**不得当作实验结论**；`pilot_3k*` 是被 `full_20k` 取代的中间谱系。
- 复现入口：`scripts/` 下的 `train_iper_qwen21_body.py`（实现 sha256 记录在 run_manifest）与四臂评测脚本；生成图像分布在各 `images/`、`strips/`。
- 本目录 `README.md` 原为 `gen_folder_docs.py` 自动生成版（已在 staging 中由人工重写替代）。
