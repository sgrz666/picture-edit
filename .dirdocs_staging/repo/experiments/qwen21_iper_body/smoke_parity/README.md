# experiments/qwen21_iper_body/smoke_parity/ —— 冒烟复现一致性对照

> 路径：`…/qwen21_iper_body/smoke_parity`　|　2 文件 / 771.9 MB　|　数据源：`train_metrics.jsonl`，2026-10-12

## 一、在干什么
与 `smoke/` 相同配置、相同 seed 再跑一次 1 步训练，验证同种子下 loss/梯度可逐位复现（parity check），排除数据/随机性管线抖动。**非实验结论。**

## 二、内容实证
结构与 `smoke/` 同构：`train_metrics.jsonl` 单行 + `checkpoint_last.pt` 771.9 MB，step 1 起始。

## 三、关系
成属于 `smoke/`；正式结论一律指向 `full_20k` 与 `four_arm_eval_50_s40`。
