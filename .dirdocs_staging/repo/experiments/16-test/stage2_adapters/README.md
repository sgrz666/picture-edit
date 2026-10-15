# experiments/16-test/stage2_adapters/ —— Stage2：挂载 DeepGen / Qwen21 身体适配器的受控编辑

> 路径：`…/16-test/stage2_adapters`　|　4 个 `results_*.json`(合计 16 例) + `images/`(34 图) + `grid_full.png`/`grid_16.png`　|　数据源：各 `results_*.json`，2026-10-12

## 干什么
在基座上加姿态 Adapter，用 `conditions_ab/` 渲染的几何条件强制贴合图二姿势，横向比较两个基座 × 两种参考输入。

## 实证
- `results_deepgen_adapter.json` / `_both.json`：`adapter=iper_v6_native_20k_v1/checkpoint_last.pt`（DeepGen），512×896、steps 50、cfg 4.0、~39–41s。
- `results_qwen_adapter_source.json` / `_both.json`：`adapter=experiments/qwen21_iper_body/full_20k/checkpoint_last.pt`（Qwen21），640×1088、steps 40、cfg 1.0、~36–42s。
- 统一条件：`condition="conditions_ab/pB (reframed to pA)"`，`refs`=source 或 both。
结论：Adapter 能严密贴合图二肢体，但"托篮球"局部交互易手掌/球融合畸变。

## 关系
消费 `../stage2_cache/16test_pA_pA.pt` 与 `../conditions_ab/`（均不可删）。上游 `stage1_baselines`，面部精修在 `stage4/5`。冒烟 `_smoke_dgad/`、`_smoke_qwad/` 分别验 DeepGen/Qwen Adapter runner。
