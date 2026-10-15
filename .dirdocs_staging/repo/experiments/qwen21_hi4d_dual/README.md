# experiments/qwen21_hi4d_dual/ —— Qwen21 双人交互/物理接触适配器（legacy_dual_lite）在 Hi4D 上的训练与单样本审计

> 路径：`/home/shangguanrz/project/pic-edit/experiments/qwen21_hi4d_dual`　|　本级：8 个子目录 + 3 个直属文件（`run_manifest.json`、`train_metrics.jsonl`、`README.md`）　|　子树：164 文件 / 8.4 GB / 54 张 PNG
> 数据来源：`run_manifest.json`、`train_metrics.jsonl`、各 `summary.json`、`single_hug_overfit/report.md`、`action_prompt_v2/completion_status.json`，读取 2026-10-12

## 一、这一层在干什么
把 `qwen21_iper_body/full_20k/checkpoint_best.pt` 的单人身体适配器扩展到**双人交互**（拥抱/太极等），用 `src/pose_control/qwen21/dual_adapter.py`（六层零初始化残差头，即 legacy_dual_lite）注入两人 SMPL-X 网格接触信号，训练时混入 iPER 单人样本做回放排练防遗忘。本目录的**核心诚实结论**：多人正式训练**未跑成**，现有"结果"全部是**单样本过拟合审计（count=1）**，且自动检测器对拥抱图只检出 1 人——**不能用这些缺失值证明双人指标达标**（详见 §三）。

> 注意：这是与 `experiments/qwen21_dual_v8/` 并存的**另一代实现**（v8 见其目录说明）。两代不可混用。

## 二、层级地图
| 子目录 | 装什么 | 内容实证 | 关系 |
| :--- | :--- | :--- | :--- |
| `single_hug_overfit/` | 单样本(pair01/hug01)双人过拟合，300 步 | `checkpoint_best.pt`(step 300)、`report.md`（人工手写）、flow val loss 0.5679→0.5336→0.5215 | 产出 `single_hug_eval*` 用的权重 |
| `single_hug_eval/` | 上权重在 seed 42 的 off/geometry/full 三模式审计 | `summary.json` **count=1**：off 检出 2 人、swap 0.0；geometry/full **只检出 1 人**，PCK_b/接触/swap=null | 消费 `single_hug_overfit` |
| `single_hug_eval_seed43/` | 同审计换 seed 43 | `summary.json` count=1；geometry 反而检出 3 人（多检第三人） | 同上，重复种子 |
| `single_pair14_action_v2_overfit/` | pair14 + action_v2 提示过拟合 300 步 | `checkpoint_best.pt`(step 300)、`eval_seed42/summary.json` count=1 五模式对比 | 见 `eval_seed42` |
| `action_prompt_v2/` | 多样本 action_v2 正式训练（10k 目标） | `completion_status.json`: `stopped_after_user_scope_correction`, last_step=410, **checkpoint_saved=false**；train_metrics 到 step 410，无 ckpt | **失败/中止**，仅日志 |
| `action_prompt_v2_smoke/` | action_v2 冒烟 | `train_metrics.jsonl`、best/last ckpt（步数极少） | 冒烟 |
| `prompt_audit_pair14/` | pair14 提示词审计 v1（seed42/43/44 + leak_and_geometry） | 三 seed 目录 + `leak_and_geometry/` | 提示词泄漏检查 |
| `prompt_audit_pair14_v2/` | 审计 v2（seed42/43/44 + `report.md`） | 三 seed 目录 | 取代 v1 |

## 三、关键文件与实证（诚实结论）
- **正式双人训练未跑成**：根 `train_metrics.jsonl` 末行 `step=15, stage=1`，`run_manifest.json` 声明 `max_steps=10000, save_every=500`——即**第一个检查点（500 步）之前进程就终止**，整个目录根**没有任何 checkpoint**。`action_prompt_v2/completion_status.json` 明说："10k multi-sample run was stopped before its first checkpoint."，`action_prompt_v2` 停在 step 410、`checkpoint_saved=false`。
- **`single_hug_overfit/report.md`（人工手写）的关键观测限制**：DWPose 自动检测器对生成拥抱图只检出 1 人，因此 B 的 `pose_pck_b`、`contact_distance_error`、`identity_swap` 在 geometry/full 模式全部记为 `null`；仅几何 seed 43 检出 3 人。
- **`single_hug_eval/summary.json`（count=1）逐模式均值**：
  - `off`（原生、无适配器）：检出 2.0 人、identity_swap **0.0**、contact_distance_error 0.1169、body_id_a 0.583 / b 0.677、背景 MAE 6.37、pose_pck_a 0.077。
  - `geometry` / `full`（开适配器）：**只检出 1.0 人**，body_id_a 升至 0.84/0.85，背景 MAE 5.87/4.06，但双人几何指标不可算。
  - **推论**：顶层 `experiments/README.md` 所称"双人检测率 100%、无身份互换"实为**关适配器的原生基线**表现；一旦开双人适配器，检测器退化为 1 人，双人指标失去分母。seed 43 复现同象（geometry 还多检出第三人）。
- **`single_pair14_action_v2_overfit/eval_seed42/summary.json`（count=1）**：`off_detailed` 检出 2 人、swap 0.0、contact 0.1132、body_id 0.821/0.778、背景 MAE 4.43；`full_detailed`（两阶段提示）body_id_a **0.8689** / b **0.7409**、背景 MAE 3.16，但检出 1.0 人——顶层所引 0.8689/0.7409 即出于此，且同样是单样本、双人几何不可算。

## 四、注意事项 / 调用路径
- **上游依赖**：初始化权重 `experiments/qwen21_iper_body/full_20k/checkpoint_best.pt`；iPER 回放清单 `datasets/iPER/.../dev_train_pairs.jsonl`；Hi4D 双人清单 `datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_{train,val}.jsonl`。
- **不可删缓存（在他人目录，本工具无权进入）**：`control_cache=nihaochen/qwen21_hi4d_controls`、`latent_cache=nihaochen/qwen21_hi4d_latents`、`iper_cache=nihaochen/qwen21_iper_cache`。
- **可复用检查点**：仅 `single_hug_overfit/checkpoint_best.pt`、`single_pair14_action_v2_overfit/checkpoint_best.pt`（均 300 步、单样本过拟合，**非泛化模型**）。`action_prompt_v2` 无 checkpoint。
- 冒烟/审计与正式如何区分：本目录**不存在成功的正式多人训练**；`single_*` 是过拟合审计、`*_smoke`/`prompt_audit_*` 是链路/提示检查。任何"双人达标"引用必须附检出率与分母（见项目记忆规范）。
- 复现入口：`scripts/train/` 下 `train_hi4d_qwen21_dual.py`（sha256 记于 run_manifest），审计脚本 `single_hug_eval/inspect_single_hug.py`。
