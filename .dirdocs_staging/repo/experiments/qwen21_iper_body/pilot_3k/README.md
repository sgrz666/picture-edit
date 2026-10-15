# experiments/qwen21_iper_body/pilot_3k/ —— 3,000 步先导训练（谱系第 1 站）

> 路径：`…/qwen21_iper_body/pilot_3k`　|　叶子目录，约 4 文件 / 1.6 GB　|　数据源：目录实测，2026-10-12

## 一、在干什么
Qwen21 身体适配器正式全量前的 3,000 步探索性先导：在小批量 iPER 对上验证梯度、显存与收敛趋势，为后续累积策略调参提供依据。

## 二、内容实证
含 `train_metrics.jsonl`、`run_manifest.json`、`checkpoint_last.pt`（771.9 MB）等；本谱系采用默认梯度累积（`accumulation=2`）。

## 三、关系
训练谱系：`pilot_3k → pilot_3k_acc4 → pilot_3k_acc4_resumed → full_20k`（权威）。本目录权重已被后继取代，仅作过程留档，不作为下游引用点。
