# experiments/qwen21_hi4d_dual/single_hug_eval/ —— single_hug_overfit 权重 seed 42 三模式审计（单样本）

> 路径：`…/qwen21_hi4d_dual/single_hug_eval`　|　`summary.json`、`per_sample.json`、`images/`、`inspect_single_hug.py`　|　数据源：`summary.json`，2026-10-12

## 一、干什么
用 `single_hug_overfit/checkpoint_best.pt`(step 300) 在**同一对样本**上以相同噪声跑 40 步、seed 42，对比三模式：`off`（原生 Qwen，无适配器）、`geometry`（仅几何注入）、`full`（完整双人适配器）。

## 二、实证（`summary.json` mean，count=1）
| 模式 | 检出人数 | identity_swap | contact_dist_err | body_id A/B | 背景 MAE | pose_pck_a |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| off | 2.0 | 0.0 | 0.1169 | 0.583 / 0.677 | 6.37 | 0.077 |
| geometry | **1.0** | null | null | 0.842 / 0.825 | 5.87 | 0.308 |
| full | **1.0** | null | null | 0.851 / 0.857 | 4.06 | 0.308 |

**读数警示**：`off`（关适配器）才检出 2 人、swap 0.0；开启双人适配器后检测器退化为 1 人，B 侧几何/接触/互换指标全部 `null`。所以"双人无互换"是原生基线现象，不代表适配器达成了双人保持。

## 三、关系
换种子的重复审计见 `single_hug_eval_seed43/`；权重来自 `single_hug_overfit/`。图板 `images/0000_panel.png`。
