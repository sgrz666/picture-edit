# experiments/qwen21_hi4d_dual/action_prompt_v2_smoke/ —— action_v2 双人链路冒烟

> 路径：`…/qwen21_hi4d_dual/action_prompt_v2_smoke`　|　`checkpoint_best.pt`、`checkpoint_last.pt`、`train_metrics.jsonl`、`validation_full.json`、`run_manifest.json`　|　2026-10-12

## 一、干什么
`action_prompt_v2` 正式训练前的极短冒烟：跑通多样本加载 + iPER 回放混合 + 双人几何注入 + 检查点保存的完整链路。

## 二、实证
含少量步数的 `train_metrics.jsonl` 与 best/last 检查点；步数极少，不构成质量结论。

## 三、关系
对应的正式 10k run（`action_prompt_v2/`）已中止且无 ckpt。本冒烟仅证明"链路能跑"，不代表双人质量；质量现状以项目记忆/本族 `single_*` 审计的诚实结论为准。
