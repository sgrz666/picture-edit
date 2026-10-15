# Checkpoint 深度审计报告（第二轮：逐项引用核验）2026-10-13

方法：对 172 个候选 checkpoint，用「目录名+文件名」精确尾段在代码/配置/文档/manifest/日志/清单中全量 grep，
区分**功能或证据引用**与**纯清单登记**（experiments_audit.json 等 3 个清单文件不计为依赖），并交叉核对各 run 的 README。

## 结论摘要

| 组 | 数量 | 体量 | 判定 |
| --- | --- | --- | --- |
| G1 确认无用（零引用、非活跃run、非文档化实验） | 34 | 25.08 GB | 可删 |
| G2 有真实引用 | 73 | 229.69 GB | **必留** |
| G3 活跃run内、无外部引用 | 33 | 123.54 GB | 需你拍板 |
| G4 零引用但属文档化实验中间档 | 12 | 16.49 GB | 建议保留 |
| G5 烟测目录内的特征/条件缓存（非权重） | 20 | 0.019 GB | 可随手清 |

**与第一轮（按命名规则分档）的差异**：第一轮认为 210 GB 可删；逐项核验后，其中约 168 GB 实际被
生成配置 / 评测样本记录 / resume 清单 / 训练脚本 / 项目 README 引用或登记，**确认不可删**。

## G1 — 确认无用，建议删除

| 体量 | 类型 | 路径 |
| --- | --- | --- |
| 2.79G | A | /home/shangguanrz/nihaochen/tiktok_qwen21_full_20261003/face_smoke/checkpoint_last.pt |
| 2.79G | A | /home/shangguanrz/nihaochen/tiktok_qwen21_full_20261003/face_stable_smoke/checkpoint_step_000003.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/body/checkpoint_step_000000.pt |
| 1.56G | B | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_adapter_finetune_v1/adapter_step_3000.pt |
| 1.45G | A | /home/shangguanrz/project/pic-edit/experiments/4.two man /qwen21_hi4d_dual/action_prompt_v2_smoke/checkpoint_last.pt |
| 1.38G | A | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_v65_face_overfit_smoke/checkpoints/checkpoint_step_000002.pt |
| 1.38G | A | /home/shangguanrz/project/pic-edit/experiments/3.singel man/tiktok_v65_test_2steps/checkpoints/checkpoint_step_000002.pt |
| 1.38G | B | /home/shangguanrz/project/pic-edit/experiments/3.singel man/tiktok_v65_face_overfit_00001/checkpoints/checkpoint_step_000100.pt |
| 1.06G | A | /home/shangguanrz/project/pic-edit/experiments/4.two man /qwen21_hi4d_dual/action_prompt_v2_smoke/checkpoint_best.pt |
| 1.03G | B | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_adapter_v1/adapter_step_100.pt |
| 0.77G | A | /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/body/checkpoint_best.pt |
| 0.77G | A | /home/shangguanrz/nihaochen/iper1024/body_smoke_nogc/checkpoint_last.pt |
| 0.77G | A | /home/shangguanrz/nihaochen/qwen21_batch4_smoke/checkpoint_last.pt |
| 0.77G | A | /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_iper_body/smoke_parity/checkpoint_last.pt |
| 0.52G | B | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/champ/champ_deepgen_formal_v1/adapter_step_120.pt |
| 0.37G | A | /home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen-2.1-test/experiment/results/overfit_smoke/adapter_overfit.pt |
| 0.22G | A | /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face/checkpoint_step_000002.pt |
| 0.22G | A | /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face/checkpoint_best_loss.pt |
| 0.22G | A | /home/shangguanrz/nihaochen/iper1024/face_smoke_gc/checkpoint_step_000001.pt |
| 0.22G | A | /home/shangguanrz/nihaochen/iper1024/face_smoke_v2/checkpoint_step_000001.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/iper1024/face_2500/checkpoint_step_001500.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/iper1024/face_2500/checkpoint_step_001000.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/iper1024/face_2500/checkpoint_step_002000.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/iper1024/face_2500/checkpoint_step_000500.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/iper1024/face_2500/checkpoint_step_002500.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/qwen21_face/face_train_3k/checkpoint_step_001500.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/qwen21_face/face_train_3k/checkpoint_step_001000.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/qwen21_face/face_train_3k/checkpoint_step_002000.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/qwen21_face/face_train_3k/checkpoint_step_003000.pt |
| 0.22G | B | /home/shangguanrz/nihaochen/qwen21_face/face_train_3k/checkpoint_step_000500.pt |
| 0.22G | A | /home/shangguanrz/nihaochen/qwen21_face/face_train_smoke/checkpoint_best_loss.pt |
| 0.22G | A | /home/shangguanrz/nihaochen/iper1024/face_smoke_gc/checkpoint_last.pt |
| 0.22G | A | /home/shangguanrz/nihaochen/iper1024/face_smoke_v2/checkpoint_last.pt |
| 0.22G | A | /home/shangguanrz/nihaochen/qwen21_face/face_train_smoke/checkpoint_last.pt |

## G2 — 有真实引用，必须保留（列出来源）

- **4.66G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/BFH/checkpoint_step_000256.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFH_final_tiktok/base_provenance.json
    - <- ~/project/pic-edit/scripts/run/run_qwen21_joint_pilot.py
    - <- ~/project/pic-edit/research/qwen21_dual_upgrade_20261011/implementation/scripts/run/run_qwen21_joint_pilot.py
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/BFSHead/checkpoint_step_000256.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/logs/BFSHead_final_native_worker.log
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSHead_final_iper/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSHead_final_iper/base_provenance.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/BFSBody/checkpoint_step_000256.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSBody_final_tiktok/base_provenance.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSBody_final_tiktok/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSBody_final_iper/base_provenance.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/Qwen/checkpoint_step_000256.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_final_tiktok/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_final_iper/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_final_tiktok/base_provenance.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/J/checkpoint_step_000256.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation_512_archive/J_common_iper/base_provenance.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSBody_hotload_iper/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSBody_hotload_tiktok/base_provenance.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training_512_archive/Qwen/checkpoint_step_000256.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_final_tiktok/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_final_iper/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_final_tiktok/base_provenance.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training_512_archive/J/checkpoint_step_000256.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation_512_archive/J_common_iper/base_provenance.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSBody_hotload_iper/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSBody_hotload_tiktok/base_provenance.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/BFSHead/checkpoint_step_000128.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSHead_mid_iper/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSHead_mid_tiktok/base_provenance.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSHead_mid_iper/base_provenance.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/BFSBody/checkpoint_step_000128.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSBody_mid_tiktok/base_provenance.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/BFSBody_mid_tiktok/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/logs/BFSBody_mid_native_worker.log
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/Qwen/checkpoint_step_000128.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation_512_archive/Qwen_mid_tiktok/base_provenance.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_mid_tiktok/base_provenance.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_mid_iper/base_provenance.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/J/checkpoint_step_000128.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/J_mid_tiktok/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/J_mid_iper/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/diagnostic_failures/eval_torch_name_error_20261008/J_mid_iper/config.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training_512_archive/Qwen/checkpoint_step_000128.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation_512_archive/Qwen_mid_tiktok/base_provenance.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_mid_tiktok/base_provenance.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/Qwen_mid_iper/base_provenance.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training_512_archive/J/checkpoint_step_000128.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/J_mid_tiktok/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/evaluation/J_mid_iper/config.json
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/diagnostic_failures/eval_torch_name_error_20261008/J_mid_iper/config.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/direct/checkpoint_step_000002.pt`
    - <- ~/project/pic-edit/scripts/run/run_qwen21_joint_pilot.py
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/resume/fork_manifest.json
    - <- ~/project/pic-edit/research/qwen21_dual_upgrade_20261011/implementation/scripts/run/run_qwen21_joint_pilot.py
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/direct/checkpoint_step_000002.pt`
    - <- ~/project/pic-edit/scripts/run/run_qwen21_joint_pilot.py
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/resume/fork_manifest.json
    - <- ~/project/pic-edit/research/qwen21_dual_upgrade_20261011/implementation/scripts/run/run_qwen21_joint_pilot.py
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_before_deterministic/direct/checkpoint_step_000002.pt`
    - <- ~/project/pic-edit/scripts/run/run_qwen21_joint_pilot.py
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/resume/fork_manifest.json
    - <- ~/project/pic-edit/research/qwen21_dual_upgrade_20261011/implementation/scripts/run/run_qwen21_joint_pilot.py
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/repeat2/checkpoint_last.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/repeat2/resume_manifest.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/resume/checkpoint_last.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/resume/resume_manifest.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/repeat1/checkpoint_last.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/repeat1/resume_manifest.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/repeat2/checkpoint_last.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/repeat2/resume_manifest.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/resume/checkpoint_last.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/resume/resume_manifest.json
- **4.40G** `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/repeat1/checkpoint_last.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/repeat1/resume_manifest.json
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/training/checkpoint_step_008548.pt`
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/completion.json
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/report_revisions/before_3000_6000_20261004_112650/report_artifact.json
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/generations/step_008548/tiktok/config.json
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/training/checkpoint_step_006000.pt`
    - <- ~/nihaochen/qwen21_full/native_value_audit_20261008/generations/sampling_1024_30/per_sample.jsonl
    - <- ~/nihaochen/qwen21_full/native_value_audit_20261008/evaluation/evaluated_samples.json
    - <- ~/project/pic-edit/docs/superpowers/specs/2026-10-08-bfs-body-face-joint-pilot-design.md
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/training/checkpoint_step_003000.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/resume/run_manifest.json
    - <- ~/nihaochen/qwen21_full/native_value_audit_20261008/generations/sampling_1024_30/per_sample.jsonl
    - <- ~/nihaochen/qwen21_full/native_value_audit_20261008/evaluation/evaluated_samples.json
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/training/checkpoint_step_001000.pt`
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/generations/step_001000/tiktok/config.json
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/report_revisions/before_3000_6000_20261004_112650/report_artifact.json
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/report_manifest.json
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/body/checkpoint_step_001000.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/stage_summary/stage_evidence.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/logs/body1000_validation_worker.log
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/body_selection.json
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/body/checkpoint_step_000500.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/generations/body/step_000500/generate/config.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/commands/generate_body_000500.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/logs/body500_validation_worker.log
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/body/checkpoint_step_000250.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/logs/body250_validation_worker.log
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/generations/body/step_000250/generate/config.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/commands/generate_body_000250.json
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/body_train/checkpoint_step_000500.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/diagnostics/generations/roi_disable_group_1/config.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/diagnostics/generations/roi_disable_group_2/config.json
    - <- ~/nihaochen/qwen21_full/face_validation_20261003/manifest.json
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/body_train/checkpoint_step_000300.pt`
    - <- ~/nihaochen/qwen21_full/face_single_sample/training.log
    - <- ~/nihaochen/qwen21_full/face_single_sample/delivery_manifest.json
    - <- ~/nihaochen/qwen21_full/face_single_sample/body_stages/step_000300/config.json
- **4.01G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/body_train/checkpoint_step_000100.pt`
    - <- ~/nihaochen/qwen21_full/face_single_sample/training.log
    - <- ~/nihaochen/qwen21_full/face_single_sample/body_stages/step_000100/config.json
    - <- ~/nihaochen/qwen21_full/face_single_sample/resume_body.json
- **4.01G** `/home/shangguanrz/nihaochen/tiktok_qwen21_full_20261003/body_smoke/checkpoint_last.pt`
    - <- ~/nihaochen/tiktok_qwen21_full_20261003/verification.json
    - <- ~/project/pic-edit/research/tiktok_training_20261003/verification.json
    - <- ~/project/pic-edit/scripts/run_tiktok_qwen21_full.sh
- **4.01G** `/home/shangguanrz/nihaochen/tiktok_qwen21_full_20261003/body_smoke/checkpoint_step_000002.pt`
    - <- ~/nihaochen/tiktok_qwen21_full_20261003/generation/config.json
    - <- ~/nihaochen/tiktok_qwen21_full_20261003/verification.json
    - <- ~/project/pic-edit/research/tiktok_training_20261003/verification.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_step_003000.pt`
    - <- ~/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/resume/run_manifest.json
    - <- ~/nihaochen/qwen21_full/native_value_audit_20261008/generations/sampling_1024_30/per_sample.jsonl
    - <- ~/nihaochen/qwen21_full/native_value_audit_20261008/evaluation/evaluated_samples.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_step_002000.pt`
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/generations/step_002000/cross/config.json
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/generations/step_002000/tiktok/config.json
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/report_manifest.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_step_001000.pt`
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/generations/step_001000/tiktok/config.json
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/report_revisions/before_3000_6000_20261004_112650/report_artifact.json
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/report_manifest.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/roi_five_point/checkpoint_step_001000.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/generations/roi_five_point/step_001000/cross/config.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/face_generation_checkpoint_schema_cpu_checks.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/face_fairness_step_1000.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/roi_box/checkpoint_step_001000.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/generations/roi_box/step_001000/cross/config.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/commands/cross_roi_box_001000.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/face_generation_checkpoint_schema_cpu_checks.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/roi_five_point/checkpoint_step_000500.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/logs/roi_five_point_first_round_worker.log
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/face_fairness_step_0500.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/commands/generate_roi_five_point_000500.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/roi_box/checkpoint_step_000500.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/generations/roi_box/step_000500/generate/config.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/face_fairness_preflight.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/commands/generate_roi_box_000500.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/roi_five_point/checkpoint_step_000250.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/commands/generate_roi_five_point_000250.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/generations/roi_five_point/step_000250/generate/config.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/logs/roi_five_point_first_round_worker.log
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/roi_box/checkpoint_step_000250.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/generations/roi_box/step_000250/generate/config.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/commands/generate_roi_box_000250.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/logs/roi_box_0_250_validation_worker.log
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/face_train/checkpoint_step_002000.pt`
    - <- ~/nihaochen/qwen21_full/face_single_sample/training.log
    - <- ~/nihaochen/qwen21_full/face_single_sample/face_stages/step_002000/config.json
    - <- ~/nihaochen/qwen21_full/face_single_sample/delivery_manifest.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/face_train/checkpoint_step_001000.pt`
    - <- ~/nihaochen/qwen21_full/face_single_sample/training.log
    - <- ~/nihaochen/qwen21_full/face_single_sample/face_stages/step_001000/config.json
    - <- ~/nihaochen/qwen21_full/face_single_sample/delivery_manifest.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/face_train/checkpoint_step_000500.pt`
    - <- ~/nihaochen/qwen21_full/face_single_sample/training.log
    - <- ~/nihaochen/qwen21_full/face_single_sample/face_stages/step_000500/config.json
    - <- ~/nihaochen/qwen21_full/face_single_sample/delivery_manifest.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_repair/roi_aligned_aux/checkpoint_step_000300.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/diagnostics/plan.json
    - <- ~/nihaochen/qwen21_full/face_validation_20261003/manifest.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/diagnostics/generations/aligned_disable_group_1/config.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_repair/roi_same_aux/checkpoint_step_000300.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/diagnostics/generations/roi_disable_group_1/config.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/diagnostics/generations/roi_disable_group_2/config.json
    - <- ~/nihaochen/qwen21_full/face_repair/generations/roi_same_aux/step_000300/config.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/face_train/checkpoint_step_000300.pt`
    - <- ~/nihaochen/qwen21_full/face_single_sample/training.log
    - <- ~/nihaochen/qwen21_full/face_single_sample/delivery_manifest.json
    - <- ~/nihaochen/qwen21_full/face_single_sample/face_stages/step_000300/config.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_repair/roi_aligned_aux/checkpoint_step_000100.pt`
    - <- ~/nihaochen/qwen21_full/face_repair/roi_aligned_aux_resume.json
    - <- ~/nihaochen/qwen21_full/face_repair/generations/roi_aligned_aux/step_000100/config.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_repair/roi_same_aux/checkpoint_step_000100.pt`
    - <- ~/nihaochen/qwen21_full/face_repair/generations/roi_same_aux/step_000100/config.json
    - <- ~/nihaochen/qwen21_full/face_repair/roi_same_aux_resume.json
- **2.79G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/face_train/checkpoint_step_000100.pt`
    - <- ~/nihaochen/qwen21_full/face_single_sample/training.log
    - <- ~/nihaochen/qwen21_full/face_single_sample/resume_face.json
    - <- ~/nihaochen/qwen21_full/face_single_sample/delivery_manifest.json
- **2.79G** `/home/shangguanrz/nihaochen/tiktok_qwen21_full_20261003/face_stable_smoke/checkpoint_last.pt`
    - <- ~/nihaochen/tiktok_qwen21_full_20261003/generation/config.json
    - <- ~/nihaochen/tiktok_qwen21_full_20261003/verification.json
    - <- ~/project/pic-edit/research/tiktok_training_20261003/verification.json
- **2.40G** `/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_step_000000.pt`
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/report_revisions/before_3000_6000_20261004_112650/report_artifact.json
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/report_revisions/before_3000_6000_20261004_112650/finalize_report.py
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/report_manifest.json
- **2.40G** `/home/shangguanrz/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/training/checkpoint_step_000000.pt`
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/report_revisions/before_3000_6000_20261004_112650/report_artifact.json
    - <- ~/nihaochen/qwen21_full/tiktok_body_fullpass_20261004/report_revisions/before_3000_6000_20261004_112650/finalize_report.py
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/report_manifest.json
- **2.40G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/roi_five_point/checkpoint_step_000000.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/logs/roi_five_point_first_round_worker.log
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/face_fairness_step_0500.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/face_fairness_step_1000.json
- **2.40G** `/home/shangguanrz/nihaochen/qwen21_full/face_multisample_v1/training/roi_box/checkpoint_step_000000.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/roi_box_independent_initialization.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/face_fairness_preflight.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/commands/generate_roi_box_000000.json
- **2.40G** `/home/shangguanrz/nihaochen/qwen21_full/face_repair/roi_aligned_aux/checkpoint_step_000000.pt`
    - <- ~/nihaochen/qwen21_full/face_repair/protection_audit.json
- **2.40G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/body_train/checkpoint_step_000000.pt`
    - <- ~/nihaochen/qwen21_full/face_single_sample/delivery_manifest.json
- **2.40G** `/home/shangguanrz/nihaochen/qwen21_full/face_repair/roi_same_aux/checkpoint_step_000000.pt`
    - <- ~/nihaochen/qwen21_full/face_repair/generations_1024/roi_same_aux/step_000000/body/config.json
    - <- ~/nihaochen/qwen21_full/face_repair/protection_audit.json
- **2.40G** `/home/shangguanrz/nihaochen/qwen21_full/face_single_sample/face_train/checkpoint_step_000000.pt`
    - <- ~/nihaochen/qwen21_full/face_single_sample/training.log
    - <- ~/nihaochen/qwen21_full/face_single_sample/face_stages/step_000000/config.json
    - <- ~/nihaochen/qwen21_full/face_single_sample/delivery_manifest.json
- **1.96G** `/home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/tiktok_overfit_sanity_full/checkpoints/checkpoint_step_001500.pt`
    - <- ~/project/pic-edit/research/qwen21_dual_upgrade_20261011/implementation/src/pose_control/qwen21_full/execution.py
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/weight_loading.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/stage_summary/stage_evidence.json
- **1.59G** `/home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_v6_native_20k_v1/adapter_step_12000.pt`
    - <- ~/nihaochen/four_arm_smoke/output/summary.json
    - <- ~/project/pic-edit/experiments/1.deepgen experiment/iper/iper_v6_testdata_eval/testdata_eval.log
    - <- ~/nihaochen/four_arm_smoke/deepgen/eval_metrics.json
- **1.38G** `/home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_adapter_v1/iper_v65_face_overfit_024_6/checkpoints/checkpoint_step_001500.pt`
    - <- ~/project/pic-edit/research/qwen21_dual_upgrade_20261011/implementation/src/pose_control/qwen21_full/execution.py
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/weight_loading.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/stage_summary/stage_evidence.json
- **1.38G** `/home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/tiktok_00001_condition_repair/train_pose_part_1500/checkpoints/checkpoint_step_001500.pt`
    - <- ~/project/pic-edit/research/qwen21_dual_upgrade_20261011/implementation/src/pose_control/qwen21_full/execution.py
    - <- ~/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/weight_loading.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/stage_summary/stage_evidence.json
- **1.38G** `/home/shangguanrz/project/pic-edit/experiments/3.singel man/tiktok_v65_face_overfit_00001/checkpoints/checkpoint_step_000200.pt`
    - <- ~/project/pic-edit/logs/legacy/inference_202609.log
    - <- ~/project/pic-edit/experiments/1.deepgen experiment/tiktok_objective_eval/timestep_sweep.json
- **0.95G** `/home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_v64_detail_overfit_024_6/checkpoints/checkpoint_step_1500.pt`
    - <- ~/project/pic-edit/research/qwen21_dual_prompt_pilot_20261013/implementation/scripts/train/train_iper_v65_face.py
    - <- ~/project/pic-edit/scripts/train/train_tiktok_v65_face.py
    - <- ~/project/pic-edit/scripts/train/train_tiktok_overfit_sanity.py
- **0.77G** `/home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/body/checkpoint_last.pt`
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/acceptance/face_fairness_preflight.json
    - <- ~/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face/manifest.json
    - <- ~/nihaochen/qwen21_full/face_multisample_v1/stage_summary/stage_evidence.json
- **0.77G** `/home/shangguanrz/nihaochen/iper1024/body_smoke/checkpoint_last.pt`
    - <- ~/nihaochen/tiktok_qwen21_full_20261003/verification.json
    - <- ~/project/pic-edit/research/tiktok_training_20261003/verification.json
    - <- ~/nihaochen/tiktok_qwen21_full_20261003/body_smoke/resume_manifest.json
- **0.77G** `/home/shangguanrz/nihaochen/qwen21_batch2_smoke/checkpoint_last.pt`
    - <- ~/nihaochen/qwen21_batch2_benchmark/run_manifest.json
- **0.77G** `/home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_iper_body/smoke/checkpoint_last.pt`
    - <- ~/nihaochen/tiktok_qwen21_full_20261003/generation/config.json
    - <- ~/nihaochen/tiktok_qwen21_full_20261003/verification.json
    - <- ~/project/pic-edit/research/tiktok_training_20261003/verification.json
- **0.22G** `/home/shangguanrz/nihaochen/qwen21_face/face_train_3k/checkpoint_step_002500.pt`
    - <- ~/nihaochen/qwen21_face/face_eval_2500/config.json
    - <- ~/nihaochen/qwen21_full/weight_audit/source_fingerprints.json
    - <- ~/nihaochen/qwen21_full/weight_audit/provenance.json
- **0.22G** `/home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face/checkpoint_last.pt`
    - <- ~/project/pic-edit/scripts/run_tiktok_qwen21_full.sh

## G3 — 活跃 run 内、无外部引用（需你拍板）

说明：位于 `bfs_body_face_joint_pilot_20261008`（8-arm 研究的权重来源 run）目录内。
已核实运行中的 PID 3682236 只依赖该 run 的 `training/J/checkpoint_step_000256.pt`（属 G2），
以下文件均无任何引用；其中 preflight 下的是 2/4 步调试残骸，权重本身等同初始化。
顾虑：删除会使 preflight 的 fork_manifest.json 中已记录的 sha256 悬空。

| 体量 | 类型 | 路径 |
| --- | --- | --- |
| 4.66G | B | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/BFH/checkpoint_step_000128.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/repeat2/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/resume/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/repeat1/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/repeat2/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/resume/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/repeat1/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_before_deterministic/split/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/direct/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/repeat2/checkpoint_step_000002.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/resume/checkpoint_step_000002.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/repeat1/checkpoint_step_000002.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/direct/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/repeat2/checkpoint_step_000002.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/resume/checkpoint_step_000002.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/repeat1/checkpoint_step_000002.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_before_deterministic/split/checkpoint_step_000002.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_before_deterministic/direct/checkpoint_step_000004.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_before_deterministic/split/checkpoint_last.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/direct/checkpoint_last.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/direct/checkpoint_last.pt |
| 4.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_before_deterministic/direct/checkpoint_last.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/BFH/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/BFSHead/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/BFSBody/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check/direct/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/Qwen/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/J/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training_512_archive/Qwen/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training_512_archive/J/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_512_archive/direct/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_before_deterministic/split/checkpoint_step_000000.pt |
| 2.40G | A | /home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/preflight/resume_check_before_deterministic/direct/checkpoint_step_000000.pt |

## G4 — 零引用但属文档化实验的中间档（建议保留）

说明：`iper_v65_face_overfit_024_6/README.md` 明确写「留存 step 250~1500 共 6 个阶段检查点」；
另两个是文档化门禁/修复实验的训练目录。虽无功能引用，但属实验证据链。

| 体量 | 路径 |
| --- | --- |
| 1.96G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/tiktok_overfit_sanity_full/checkpoints/checkpoint_step_001000.pt |
| 1.96G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/tiktok_overfit_sanity_full/checkpoints/checkpoint_step_000500.pt |
| 1.38G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_adapter_v1/iper_v65_face_overfit_024_6/checkpoints/checkpoint_step_001250.pt |
| 1.38G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_adapter_v1/iper_v65_face_overfit_024_6/checkpoints/checkpoint_step_000750.pt |
| 1.38G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_adapter_v1/iper_v65_face_overfit_024_6/checkpoints/checkpoint_step_001000.pt |
| 1.38G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_adapter_v1/iper_v65_face_overfit_024_6/checkpoints/checkpoint_step_000250.pt |
| 1.38G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_adapter_v1/iper_v65_face_overfit_024_6/checkpoints/checkpoint_step_000500.pt |
| 1.38G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/tiktok_00001_condition_repair/train_pose_part_1500/checkpoints/checkpoint_step_001250.pt |
| 1.07G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/tiktok_00001_condition_repair/train_pose_part_1500/checkpoints/checkpoint_step_000750.pt |
| 1.07G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/tiktok_00001_condition_repair/train_pose_part_1500/checkpoints/checkpoint_step_001000.pt |
| 1.07G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/tiktok_00001_condition_repair/train_pose_part_1500/checkpoints/checkpoint_step_000250.pt |
| 1.07G | /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/tiktok_00001_condition_repair/train_pose_part_1500/checkpoints/checkpoint_step_000500.pt |

## G5 — 烟测目录内的特征/条件缓存（非模型权重）

- /home/shangguanrz/project/pic-edit/experiments/1.deepgen experiment/iper/iper_v65_face_overfit_smoke/condition_cache/024_6_source_motion_f000375.pt (3.5 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/qwen21_cache/source_8bb534ed99813875069805c3.pt (2.5 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/qwen21_cache/source_1df57267d4b75ca41adfb9b3.pt (2.5 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/qwen21_cache/target_0c9b14ca250802e01c6d0683.pt (0.1 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/qwen21_cache/source_67388b1f15b8d31ddf04a0d3.pt (2.5 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/qwen21_cache/source_a62ea7740808935d27e9f390.pt (2.5 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/qwen21_cache/target_7c59b1eef88366a6154db2cf.pt (0.1 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/qwen21_cache/source_47271f0e9bb0ece6b2bedd67.pt (2.5 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/qwen21_cache/target_1e7384226f44ca3ba27492a9.pt (0.1 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/qwen21_cache/target_21ba4142ec361b6b7c135d53.pt (0.1 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00032.pt (0.2 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00126.pt (0.2 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00006.pt (0.2 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00002.pt (0.2 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00127.pt (0.2 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00009.pt (0.2 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00031.pt (0.2 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00001.pt (0.2 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00008.pt (0.2 MB)
- /home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_tiktok_v3_smoke/face_features/00030.pt (0.2 MB)

## 删除执行方案（待确认）

1. 一律 `mv` 到 `/data/shangguanrz/_trash_ckpt_20261013/<原相对路径>`，不直接 `rm`，可回滚。
2. 每批 ≤10 个文件，逐批校验（删除前后文件计数与字节数对账）。
3. G2 / G4 不动；G3 需单独授权。