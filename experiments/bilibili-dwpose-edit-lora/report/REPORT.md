# Bilibili 双图 Edit LoRA 首轮实测

输入为 A RGB、B 原始 DWPose 骨架和统一短指令；监督为 B RGB。基座、Qwen3-VL、VAE 冻结。

完成 300 次更新；32 对训练样本来自 8 个视频，10 对验证样本来自另外 5 个视频。测试集未使用。
画布 384×672；rank/alpha=16，BF16，AdamW，lr=5e-5，前 10 步 warmup，batch=1。

首 20 步平均 loss=0.278440，末 20 步=0.179164。随机 timestep 会影响 loss；该趋势不单独证明姿态或身份质量提高。

## 对照指标

PCK 使用 COCO17 对应关节、图像对角线归一化；未检测到的有效目标关节计为 miss。身份相似度只统计可检测人脸，并单列失败数。

| 方法 | 划分 | 图数 | PCK@0.02 | PCK@0.05 | PCK@0.10 | ID cosine | ID有效数 | 姿态失败 | 人脸失败 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | val | 10 | 0.9147 | 0.9750 | 0.9812 | 0.5106 | 10 | 0 | 0 |
| baseline | train | 8 | 0.9623 | 0.9853 | 1.0000 | 0.3579 | 8 | 0 | 0 |
| lora | val | 10 | 0.9202 | 0.9750 | 0.9750 | 0.5026 | 10 | 0 | 0 |
| lora | train | 8 | 0.9382 | 0.9779 | 1.0000 | 0.3740 | 8 | 0 | 0 |

## 换骨架诊断

固定 A、Prompt、seed。换骨架时 VLM 和 VAE 两条路径同时更换；空白骨架也经过两条路径。输出发生变化只证明模型对条件敏感，还需结合 PCK 与对照图判断方向是否正确。

| 方法 | 条件变化 | RGB MAE (0–255) | 输出关节位移/对角线 | 骨架关节位移/对角线 |
|---|---|---:|---:|---:|
| baseline | swap_pose | 7.9059 | 0.0379 | 0.0370 |
| baseline | blank_pose | 2.5869 | 0.0021 | 无有效检测 |
| lora | swap_pose | 7.2262 | 0.0377 | 0.0370 |
| lora | blank_pose | 4.3005 | 0.0064 | 无有效检测 |

## 保存、重载与恢复

- 第 100 步恢复：LoRA 权重逐张量一致=True；优化器状态数=256；RNG 已恢复=True。
- 最终 LoRA 原生加载：转换到加载 dtype 后逐张量一致=True，最大权重差=0.0009756088256835938。
- 第 300 步相对初始化的 LoRA L2 变化=5.593014。
- 本轮没有做连续 300 步与中断恢复轨迹的逐步 bit-exact 对照；恢复验证范围是权重、优化器、scheduler、RNG 状态和后续真实更新。

## 显存与内存

NVML 每 2 秒采样，全卡占用包含并行 V5；CUDA allocated/reserved 是本进程 allocator 峰值。RSS 含文件映射页。

| 阶段 | 本进程 CUDA reserved GiB | 全卡峰值 GiB | 最小空余显存 GiB | 进程 RSS 峰值 GiB | 最小可用内存 GiB |
|---|---:|---:|---:|---:|---:|
| output/pilot_20261111_v1/resources_000000_000100.json | 15.65 | 34.59 | 48.45 | 8.49 | 102.81 |
| output/pilot_20261111_v1/resources_000100_000300.json | 15.65 | 37.18 | 45.87 | 8.65 | 99.99 |
| output/comparison_20261111_v1/resources.json | 17.23 | 38.59 | 44.46 | 5.98 | 100.51 |
| report/pilot_report_20261111_v1/resources.json | 0.00 | 21.20 | 61.85 | 2.03 | 98.52 |
| cache/cache_20261111_v1/resources.json | 17.11 | 36.06 | 46.99 | 18.44 | 86.54 |
| output/preflight_20261111_v1/resources_000000_000001.json | 15.64 | 34.59 | 48.46 | 7.87 | 102.62 |
| output/probe_inference_20261111_v1/resources.json | 17.23 | 36.70 | 46.35 | 10.33 | 101.46 |

## 产物与使用

- 权重与完整恢复状态：`/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora/output/pilot_20261111_v1/checkpoint-000300/`。
- 原始对照图与推理记录：`/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora/output/comparison_20261111_v1/`。
- 本目录 `metrics.json`、`per_image.jsonl`、`pose_response.json`、`resource_summary.json` 为可复核数字；`validation_overview.png` 为验证集前 4 对概览，`training_curves.png` 为训练曲线。
- `normal_*_comparison.png` 的列顺序是 A、B 骨架、B RGB、原始 Qwen、LoRA。
- 原始视频可能含逐帧变化的字幕，本轮保留原 RGB 整图监督。小样本结果不代表全量训练或跨人泛化。
