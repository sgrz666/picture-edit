# experiments/qwen21_iper_body/pilot_3k_acc4_resumed/ —— acc4 先导的续训（full_20k 的直接前驱）

> 路径：`…/qwen21_iper_body/pilot_3k_acc4_resumed`　|　4 文件 / 1.6 GB　|　数据源：`full_20k/run_manifest.json` 引用，2026-10-12

## 一、在干什么
对 `pilot_3k_acc4` 断点续训，产出 `checkpoint_last.pt`——该文件被正式全量 `full_20k` 的 `run_manifest.json` 显式声明为 `resume` 起点，是整个身体适配器训练链的"接力棒"。

## 二、内容实证
含 `run_manifest.json`、`train_metrics.jsonl`、`checkpoint_best.pt` / `checkpoint_last.pt`（各 771.9 MB）。首行 metrics `step=1501`（继承自 acc4）。

## 三、关系
谱系：`pilot_3k → pilot_3k_acc4 → [本目录] → full_20k（权威）`。**本目录的 checkpoint_last.pt 是 full_20k 的训练续接起点，删除会导致 full_20k 复现断链。**
