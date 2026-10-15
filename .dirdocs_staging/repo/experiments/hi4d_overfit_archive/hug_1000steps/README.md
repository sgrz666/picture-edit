# experiments/hi4d_overfit_archive/hug_1000steps/ —— DeepGen Hi4D 双人拥抱 1000 步过拟合

> 路径：`…/hi4d_overfit_archive/hug_1000steps`　|　`summary.json`、`experiment_config.json`、`condition_cache/`、`comparison_grid.png`、11 张 `model_step_0000…1000.png`、`02_CONDITION_*` 图　|　数据源：`summary.json`，2026-10-12

## 一、干什么
同一 DeepGen Hi4D 单样本（pair01/cam4 hug01→hug01，source 帧16 → target 帧48）过拟合延长到 1000 步，`save_every=100`，观察每 100 步的生成演变。

## 二、实证
`summary.json`：completed_steps=1000、duration 1045.6s；diagnostics loss step100 0.0728 → step1000 **0.0337**（step800 有 0.837 尖峰）；每步梯度范数 0.16–0.65。`row` 字段与 `hug_120steps` 同（`MAIN_N2C`, SAME_ACTION, target_contact=true）。`model_step_0100…1000.png` 逐步演变。

## 三、关系
取代 `hug_120steps/`（早期短程）；共同佐证双人接触渲染条件在 DeepGen 侧可行，推动后续迁移到 Qwen21 双人线。
