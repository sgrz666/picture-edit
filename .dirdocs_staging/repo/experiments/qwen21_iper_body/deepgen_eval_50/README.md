# experiments/qwen21_iper_body/deepgen_eval_50/ —— DeepGen 单臂 50 样本 on/off 对照评测

> 路径：`…/qwen21_iper_body/deepgen_eval_50`　|　`eval_metrics.json` + `images/`（151 文件 / 51.1 MB）　|　数据源：`eval_metrics.json`，2026-10-12

## 一、在干什么
在把 DeepGen 纳入四臂对照之前，先单独评 DeepGen-1.0 V6 适配器在 50 对 held-out iPER 上的开/关效果（`adapter_checkpoint=iper_v6_native_20k_v1/checkpoint_last.pt`，即 step 20000 权重）。这是 DeepGen 侧的项目参照基线，后被 `four_arm_eval_50*` 吸收为 `deepgen_off/on` 两臂。

## 二、内容实证（`eval_metrics.json`）
- `summary`（Adapter ON）：num_samples 50，**avg_psnr 24.51 dB**、avg_ssim 0.9755（skimage 0.8748）、avg_mae 6.63、avg_mse 257.67。
- `summary_adapter_off`（Adapter OFF）：见同文件，与四臂 `deepgen_off`（PSNR 17.2–17.35）一致。
- 顶层键：`per_sample` / `per_sample_adapter_off` / `metrics_adapter_on` / `metrics_adapter_off` / `generation_settings` / `pairs_jsonl`。
- `images/` 50 张生成图。

## 三、关系与注意
`deepgen_eval_50_s40/` 是其 40 步采样变体（ON avg_psnr 24.50, mae 6.63，几乎相同）。DeepGen 受控臂最终数值统一引用四臂目录。注意：这里的 SSIM 用 skimage 口径 0.875，与四臂 summary 的 SSIM 0.876 一致。
