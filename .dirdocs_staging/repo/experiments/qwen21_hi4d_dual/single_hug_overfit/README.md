# experiments/qwen21_hi4d_dual/single_hug_overfit/ —— 单样本双人拥抱过拟合（300 步，产出可复用审计权重）

> 路径：`…/qwen21_hi4d_dual/single_hug_overfit`　|　叶子目录，`checkpoint_best.pt`(step 300)、`checkpoint_last.pt`、`report.md`(手写)、`train_metrics.jsonl`、`validation_full.json`、`run_manifest.json`　|　2026-10-12

## 一、干什么
在 Hi4D **单条训练样本**（`pair01/cam4/hug01`，源帧 000016 未接触 → 目标帧 000087 拥抱峰值，3047 个接触顶点、8 条有效接触关系）上，从 iPER 20k body 检查点初始化单人部分、双人新模块随机初始化，只训 300 步（前 100 冻结原单人、后 200 联合，无 iPER 回放，batch=1）。

## 二、实证
- 同噪声下 flow 验证损失：100 步 0.5679 → 200 步 0.5336 → 300 步 **0.5215**（训练样本重建测试，非泛化）。
- `train_metrics.jsonl` 末行 step 300 stage 2 loss 0.1600、val_loss 0.5215、peak 30.23 GiB。
- `report.md`（人工）明确记录：DWPose 对生成的拥抱图只检出 1 人，故 B 的姿态 PCK/接触距离/身份互换率无法可靠计算，记 `null`——"不能用缺失值证明双人指标达标"。

## 三、关系
被 `single_hug_eval/`（seed 42）、`single_hug_eval_seed43/`（seed 43）加载做 40 步采样审计。`checkpoint_best.pt` 是本族少见的"有产物"检查点，但属过拟合，非交付模型。
