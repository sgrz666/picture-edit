# experiments/qwen21_tiktok_v3_smoke/body/ —— TikTok v3 身体适配器 2 步冒烟

> 叶子目录，5 文件　|　数据源：`train_metrics.jsonl`、`run_manifest.json`，2026-10-12

## 干什么
Qwen21 单流身体适配器（`src/pose_control/qwen21/body_adapter.py`）在 TikTok v3 上跑 2 步，验证从 DeepGen V6 条件（`v6_checkpoint=iper_v6_native_20k_v1/checkpoint_last.pt`）warm-start 后在 TikTok 数据上能前反向。

## 实证
`train_metrics.jsonl` 末行 step=2：loss 0.1431、val_loss 0.5235、peak_cuda_gib 30.76；`checkpoint_best.pt`/`checkpoint_last.pt`。`run_manifest.json`：TikTok 清单 `datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/`，`align_mode=square_crop`。

## 注意
冒烟，权重无质量含义；`checkpoint_last.pt` 仅被同目录 `face/` 用作初始化起点。
