# experiments/qwen21_hi4d_dual/single_pair14_action_v2_overfit/ —— pair14 + action_v2 提示的样本过拟合（300 步）

> 路径：`…/qwen21_hi4d_dual/single_pair14_action_v2_overfit`　|　`checkpoint_best.pt`(step 300)、`checkpoint_last.pt`、`prompts_{train,val}.jsonl`、`validation_full.json`、`stage1_superseded_after_step100.jsonl`、`train_metrics.jsonl`、`run_manifest.json`、子目录 `eval_seed42/`　|　2026-10-12

## 一、干什么
在 Hi4D pair14 单条样本上，用结构化 action_v2 提示词做双人过拟合（300 步，前 100 stage1 后被取代、后 200 联合），比 `single_hug_overfit` 更强调"动作语义提示"对双人接触的作用。

## 二、实证
- `train_metrics.jsonl` 末行 step 300 stage 2 loss 0.158、val_loss 0.525、peak 30.23 GiB。
- `eval_seed42/summary.json`（count=1，五模式，steps 40）：

| 模式 | 检出人数 | swap | contact_err | body_id A/B | 背景 MAE | pose_pck_a |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| off_detailed | 2.0 | 0.0 | 0.1132 | 0.821/0.778 | 4.43 | 0.333 |
| off_short | 2.0 | 0.0 | 0.0975 | 0.784/0.710 | 4.52 | 0.333 |
| full_detailed | **1.0** | null | null | **0.8689/0.7409** | 3.16 | 0.467 |
| full_short | 1.0 | null | null | 0.866/0.739 | 3.14 | 0.467 |
| geometry_short | 1.0 | null | null | 0.787/0.765 | 4.56 | 0.267 |

**读数警示**：顶层 `experiments/README.md` 引用的"Person A 0.8689 / Person B 0.7409、背景 MAE 3.16"就出自本表 `full_detailed`，但它是**单样本、且开适配器后只检出 1 人**（swap/contact/PCK_b 记 null）。body_id 是人脸嵌入余弦（单帧），双人几何达标性不可证。

## 三、关系
`eval_seed42/` 是其审计子目录（含 `images/`、`per_sample.json`、`generation_manifest.json`）。权重 `checkpoint_best.pt` 供该审计。属过拟合审计，非泛化模型。
