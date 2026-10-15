# experiments/qwen21_hi4d_dual/single_hug_eval_seed43/ —— 同一审计的 seed 43 复现

> 路径：`…/qwen21_hi4d_dual/single_hug_eval_seed43`　|　`summary.json`、`per_sample.json`、`images/`　|　数据源：`summary.json`，2026-10-12

## 一、干什么
`single_hug_eval/` 的换种子（seed 43）重复，检验拥抱生成审计对随机性的稳健性。

## 二、实证（mean，count=1）
- `off`：检出 2.0 人、swap 0.0、contact 0.1930、body_id 0.540/0.661、背景 MAE 8.74、pose_pck_a 0.0。
- `geometry`：**检出 3.0 人**（多检出第三人，说明双人模型在拥抱态下几何注入不稳定）、body_id 0.849/0.799、contact 0.0947。
- `full`：检出 1.0 人、pose_pck_a 0.308、背景 MAE 高达 49.48（背景被明显推移，`report.md` 记"形成拥抱，但背景明显位移"）。

**结论**：两 seed 一致显示——开适配器后双人几何指标失去可靠分母，seed 43 还出现"多检一人 / 背景大位移"。审计结果不稳定，不能支撑"双人达标"。

## 三、关系
与 `single_hug_eval/`（seed 42）成对；权重同来自 `single_hug_overfit/`。
