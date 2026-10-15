# experiments/qwen21_hi4d_dual/action_prompt_v2/ —— 多样本 action_v2 双人正式训练（已中止，无检查点）

> 路径：`…/qwen21_hi4d_dual/action_prompt_v2`　|　`completion_status.json`、`train.log`、`completion.log`、`train_metrics.jsonl`、`prompts_{train,val}.jsonl`、`run_manifest.json`、`code_snapshot/`　|　数据源：`completion_status.json`/`train_metrics.jsonl`，2026-10-12

## 一、干什么
目标为 10,000 步的 Hi4D 多人双人交互正式训练（stage1 2000 步冻结单人、iPER 回放率 0.1、seed 42、RTX 6000D）。

## 二、实证（失败留档）
- `completion_status.json`：`state="stopped_after_user_scope_correction"`，`last_recorded_step=410`，`checkpoint_saved=false`，`training_process_running=false`，note："10k multi-sample run was stopped before its first checkpoint"（`save_every=500`，410 < 500）。
- `train_metrics.jsonl` 末行 step 410 stage 1 loss 0.1024、peak 30.76 GiB、elapsed 3677s。
- `code_snapshot/` 保存当时 `dual_adapter.py` 等实现快照。

## 三、关系与注意
本目录**没有可复用权重**，只证明多人正式训练未产出。冒烟版见 `action_prompt_v2_smoke/`；样本级审计（真正有 ckpt 的）见 `single_hug_overfit/`、`single_pair14_action_v2_overfit/`。提示词审计见 `prompt_audit_pair14*`。
