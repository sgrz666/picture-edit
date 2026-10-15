# experiments/qwen21_iper_body/smoke/ —— 1 步训练冒烟（链路连通性验证）

> 路径：`…/qwen21_iper_body/smoke`　|　2 文件 / 771.9 MB　|　数据源：`train_metrics.jsonl`（1 行），2026-10-12

## 一、在干什么
Qwen21 身体适配器最小训练冒烟：跑 1 个 micro-step 确认加载 7B 基座、注入适配器、反向与保存检查点全链路可用。**非实验结论。**

## 二、内容实证
`train_metrics.jsonl` 仅 1 行：`step=1, micro_step=1, loss=0.1385, grad_norm=0.0664, peak_cuda_gib≈30.76`；`checkpoint_last.pt` 771.9 MB。

## 三、关系
`smoke_parity/` 是同配置的复现一致性对照（内容同构）；二者均被 `pilot_3k*`/`full_20k` 正式训练取代，权重无下游引用。
