# experiments/qwen21_iper_body/four_arm_eval_50_s40/ —— 四臂交叉评测的最终权威版（50 样本 / 40 步）

> 路径：`…/qwen21_iper_body/four_arm_eval_50_s40`　|　本级：3 文件（per_sample.csv/json、summary.json）+ `strips/`（50 张拼图）/ 53 文件 / 42.7 MB　|　数据源：`summary.json`，读取 2026-10-12

## 一、在干什么
把 DeepGen-1.0 与 Qwen-Image-2.1 两个基座、各开/关身体适配器共 **4 个臂**，在同一 50 对 held-out iPER 上跑同一 prompt、seed 42、40 步采样，做像素级+姿态级+身份级的定量裁决。**这是 `qwen21_iper_body` 对外引用结论的权威目录**（`four_arm_eval_50`/`_s30` 是其步数/缓存变体）。

## 二、核心结果（50 样本均值，摘自 summary.json.arms）
| 指标 | deepgen_off | deepgen_on | qwen21_off | qwen21_on |
| :--- | :---: | :---: | :---: | :---: |
| PSNR / SSIM / MAE | 17.24 / .8017 / 16.26 | 24.53 / .8756 / 6.58 | 17.53 / .8363 / 13.61 | **25.86 / .8969 / 4.93** |
| PCK@0.02 / @0.05 | 17.4% / 48.5% | 95.5% / 98.1% | 20.3% / 47.1% | **98.0% / 99.2%** |
| Face ID cos (n=47) | 0.2612 | 0.2917 | **0.8187** | 0.3510 |
| 人体前景 MAE | 60.01 | 20.03 | 61.16 | **17.57** |
| 耗时 / 峰值显存 | 5.08s / 14.51G | 5.94s / 14.53G | 5.84s / 32.76G | 5.87s / 32.76G |

`summary.json.paired_deltas` 记录 bootstrap CI95 配对差：`qwen21_control_gain` 中 **Face ID −0.468 [−0.515,−0.418]**（开适配器摧毁身份保持）；`deepgen_control_gain` PSNR +7.29 dB；`qwen21_control_gain` PSNR +8.33 dB。

## 三、文件与关系
- `summary.json`：臂级均值 + 9 项 metric_definitions（PCK 以 512² 对角线为阈、identity_cosine 用 AntelopeV2 人脸嵌入）。
- `per_sample.json`（4,401 条）、`per_sample.csv`（200 行×22 列）：逐样本逐臂指标，用于统计检验。
- `strips/`：50 张四臂横向拼图（deepgen_off/on、qwen21_off/on）。
- 兄弟目录：`four_arm_eval_50/`（同配置、计时含 kv_cache 差异）、`four_arm_eval_50_s30/`（30 步消融）；三者 qwen21_on PSNR 25.86–25.91、PCK 98% 一致，结论稳健。

## 四、注意事项
`generation_settings.qwen21.checkpoint=full_20k/checkpoint_last.pt`、`deepgen.checkpoint=iper_v6_native_20k_v1/checkpoint_last.pt`（step 20000）；引用本目录数字即引用项目最高水平身体控制结论。
