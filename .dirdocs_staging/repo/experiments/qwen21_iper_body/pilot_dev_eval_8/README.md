# experiments/qwen21_iper_body/pilot_dev_eval_8/ —— 8 样本 dev 冒烟评测（四臂协议上线前的快速回归）

> 路径：`…/qwen21_iper_body/pilot_dev_eval_8`　|　27 文件 / 7.2 MB（含 `images/` 与 4 拼图条）　|　数据源：`summary.json`，2026-10-12

## 一、在干什么
四臂评测正式扩展到 50 样本之前，先在 **8 个 dev 样本**上跑通全指标流水线（steps=40, seed=42, `checkpoint=full_20k/checkpoint_last.pt`, `kv_cache=true`），校验 PCK/Face ID/显存计时的采集代码可用。属小规模验证性评测，非最终结论。

## 二、内容实证
`summary.json` 顶层键 `num_samples=8 / steps=40 / seed=42 / checkpoint`；`images/` 存 8 样本生成图与对照拼图。指标口径与 50 样本版一致，仅样本量不同。

## 三、关系
是 `eval_50*`/`four_arm_eval_50*` 的"探针"版本；结论一律以 50 样本目录为准。
