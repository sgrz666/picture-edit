# research/qwen21_dual_prompt_pilot_20261013/ —— 双人提示词 pilot 的调研证据包

`research/` 下按惯例存放「带着证据材料的专题调研，可独立核验」的包。本目录对应的是实验 run `experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1`（双人 Qwen-Image-2.1 提示词 pilot）的**研究侧证据包**：把设计、数据准备、训练、交付快照与人工核验所需的物料集中一处，使该 run 的结论可以不依赖运行环境被第三方核验。落盘于 2026-10-13~10-14。

## 里面有什么

证据与脚本分层如下（均为真实文件名）：

- **设计与基线**：`setup.json`（代码镜像 `implementation/`、warm_start 权重路径与 SHA256、`old_mirror_preserved`）、`baseline_integrity.json`（4 个训练侧文件 old==new 哈希，证明训练侧未改动）、`selection_lock.json`（约 20KB 的选择锁定明细）。
- **数据准备**：`data/`（预处理产物：assets、dataset.json、samples、sources、train/val/test）、`data_prepare_process.json`/`data_prepare_result.json`/`data_prepare.log`（跑 `scripts/precompute/prepare_hi4d_dual_prompt_pilot.py`，exit 0、约 127s）。
- **训练与审计**：`training_completion_audit.json`（short/structured 两分支各完成 300 步、best_step=50、含 checkpoint SHA256 与 native 进程前后哈希）、`training_process.json`/`training.log`、`training_evidence_final_v1.json`+`.zip`（13 个清单：run_manifest、cache_manifest、shared_schedule、real_model_gates、train_metrics.jsonl 等）。
- **交付与测试证据**：`delivery_snapshot_v1.json`+`.zip` 与 `delivery_snapshot_v2.json`+`.zip`（143 文件；v2 修正背景分母措辞并 supersede v1）、`test_evidence_final_v1.json`+`.zip`（84 个测试文件，含 `evaluation_complete` 与 `run` 的 SHA256）、`evaluation_process.json`/`evaluation_result.json`/`evaluation.log`、`evaluation_detached_config.json`、`evaluation_padding_snapshot_v2.json`+`.zip`（排除留边后背景打分回归）。
- **代码镜像与恢复**：`implementation/`（docs/reports/research/scripts/src/tests 的镜像）、`recovery/`（中断审计 `interruption_audit.json`、preflight/retrain/continuation/remote_regression 各轮证据、`recover_training.py`/`retrain_structured.py`）、`figures_partial_v1/`（`figure_sources.json`、`selected_prompts.md`、`training_plot.png/svg`）。
- **核验脚本**：`eval_driver.py`（断言 `training_complete` 后驱动评估，写 `evaluation_process/result.json`）、`verify_delivery_from_manual.py`（逐项断言两分支 gates、checkpoint SHA、再核验 test 输出），以及 `regression_*.log`、`figure_final.log` 等日志。

## 与实验 run 的关系

真正的训练产物与推理结果在 `experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1`（`evaluation_process.json` 的 `--training-root` 指向它，评估输出在 `test_pair00_v1/`）。本目录是它的**影子证据包**：同一份训练闭环（两分支 0-init、backbone 不变、仅靠真实 val 分流选 best、test 全程未读）的 SHA256、manifest 与日志都留在 zip 与 json 里，让"到底训了什么、交付了什么"可被独立复核。

## 注意与坑

- **解包前先核 sha**：`delivery_snapshot_*.zip`、`test_evidence_final_v1.zip`、`training_evidence_final_v1.zip`、`evaluation_padding_snapshot_v2.zip` 都附带同名 json 记录 `sha256` 与文件数，解包或引用前务必比对，别凭文件名当真。
- **这不是可执行代码库**：`implementation/` 只是代码镜像，真正执行发生在 `experiments/4.two man /...` 那条 run 与对应 `scripts/`；本目录定位是"证据 + 可复跑的核验脚本"，不要把它当主代码树改。
- **v2 覆盖 v1**：`delivery_snapshot_v2` 显式 `supersedes` v1（仅背景分母措辞修正），引用交付物以 v2 为准。
- 训练侧 `baseline_integrity` 仅覆盖 `src/pose_control/qwen21_full/dual/` 与 `src/data/hi4d_smplx_dataset.py` 几处，未列入的文件不在此证明范围内。
