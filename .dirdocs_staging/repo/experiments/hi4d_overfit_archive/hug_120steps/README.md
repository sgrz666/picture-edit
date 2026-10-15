# experiments/hi4d_overfit_archive/hug_120steps/ —— DeepGen Hi4D 双人拥抱 120 步过拟合

> 路径：`…/hi4d_overfit_archive/hug_120steps`　|　`summary.json`、`experiment_config.json`、`condition_cache/`、`comparison_grid.png`、`model_step_0000…0120.png`、`02_CONDITION_*` 条件渲染图　|　数据源：`summary.json`，2026-10-12

## 一、干什么
DeepGen-1.0 在 Hi4D pair01/cam4（hug01→hug01）单样本上的 120 步双人过拟合短程记录（inference_steps=24, seed=42）。

## 二、实证
`summary.json`：completed_steps=120、duration 186.0s；diagnostics loss step30 0.0938 / step60 0.0585 / step90 0.2089 / step120 0.0584，梯度范数 0.21–0.53。条件缓存：`condition_cache/`（`hi4d_dual_source.pt`）。

## 三、关系
被同目录 `hug_1000steps/`（更长程 1000 步）演进；二者合为本归档族。非 Qwen 双人线产物，仅作 DeepGen 双人早期留档。
