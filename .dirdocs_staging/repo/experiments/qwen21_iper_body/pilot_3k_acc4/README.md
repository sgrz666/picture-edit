# experiments/qwen21_iper_body/pilot_3k_acc4/ —— 梯度累积=4 的先导训练（谱系第 2 站）

> 路径：`…/qwen21_iper_body/pilot_3k_acc4`　|　3 文件 / 772.0 MB　|　数据源：`train_metrics.jsonl`（273 行）、`run_manifest.json`，2026-10-12

## 一、在干什么
在 `pilot_3k` 基础上把有效批大小调大的先导尝试（micro-step 与 step 出现 3:1 关系，如首行 `step=1501, micro_step=1504`）。

## 二、内容实证
- `train_metrics.jsonl` 273 行，首行 step 1501 loss 0.2231。
- `run_manifest.json` 记录 `accumulation` 等超参；`checkpoint_best.pt` 771.9 MB。

## 三、关系
被 `pilot_3k_acc4_resumed/` 续训（该 resumed 目录的 `checkpoint_last.pt` 进而被 `full_20k` 用作 resume）。属中间谱系，非下游引用点。
