# experiments/qwen21_iper_body/deepgen_eval_50_s40/ —— DeepGen 单臂评测（40 步采样变体）

> 路径：`…/qwen21_iper_body/deepgen_eval_50_s40`　|　`eval_metrics.json` + `images/`（151 文件 / 51.1 MB）　|　数据源：`eval_metrics.json`，2026-10-12

## 一、在干什么
`deepgen_eval_50/` 的 40 步采样版本，DeepGen-1.0 V6 适配器（`iper_v6_native_20k_v1/checkpoint_last.pt`）在 50 对 held-out iPER 上 on/off 对照。

## 二、内容实证
`eval_metrics.json.summary`（ON）：num_samples 50，**avg_psnr 24.50 dB**、avg_ssim 0.9757（skimage 0.8747）、avg_mae 6.63、avg_mse 258.56；OFF 见 `summary_adapter_off`。与默认版差异在小数点，属采样预算消融。

## 三、关系
兄弟：`deepgen_eval_50/`（默认步数）。DeepGen 受控臂的对外权威数值以 `four_arm_eval_50_s40/` 的 `deepgen_on` 列为准。
