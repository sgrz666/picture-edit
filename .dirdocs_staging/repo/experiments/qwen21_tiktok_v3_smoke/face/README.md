# experiments/qwen21_tiktok_v3_smoke/face/ —— TikTok v3 面部适配器头 2 步冒烟

> 叶子目录，6 文件　|　数据源：`train_metrics.jsonl`、`manifest.json`，2026-10-12

## 干什么
在 `body/checkpoint_last.pt` 之上加面部适配器头（仅头部参数，另存为 215.9 MB 的小检查点），2 步冒烟验证 flow/identity/perceptual/outside 四项损失的注入通道。

## 实证
`train_metrics.jsonl` 末行 step=2：flow_loss 0.1732、identity/perceptual/outside_loss 均 0.0、val_velocity_mse 0.3150、peak 20.998 GiB。检查点：`checkpoint_best_loss.pt` / `checkpoint_last.pt` / `checkpoint_step_000002.pt`。`manifest.json` 记 `body_checkpoint=body/checkpoint_last.pt`、`auxiliary_decode_resolution=512`。

## 注意
identity/perceptual 项为 0 属冒烟初期（尚未接入面部特征监督全量），不代表面部方案有效；正式面部策略结论参见项目内 `tiktok_overfit_sanity_full` 的门禁裁决。
