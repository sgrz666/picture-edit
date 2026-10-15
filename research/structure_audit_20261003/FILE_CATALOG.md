# 逐文件用途与归类索引

基于 2026-10-03 的 SSH 只读快照。分类是建议，未移动任何文件。静态引用为代码或文档文本命中，不能据零命中断言文件无用。详细引用、根目录定位代码和 SHA-256 见同目录 CSV。

| 原路径 | 类型 | 建议归属 | 行数 | 用途线索 | 静态引用文件数 |
|---|---|---|---:|---|---:|
| /home/shangguanrz/annotate_kps.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 21 |  | 0 |
| /home/shangguanrz/audit_remaining_datasets.sh | 项目外运行编排 | scripts/run/（逐项核验） | 41 |  | 0 |
| /home/shangguanrz/build_16grid.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 69 | firstof, load, lb | 0 |
| /home/shangguanrz/build_contact_sheets.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 68 | Build labelled contact sheets from candidate buckets for fast visual review. | 0 |
| /home/shangguanrz/build_full.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 37 | ld | 0 |
| /home/shangguanrz/build_interaction_set.py | 项目外数据准备工具 | scripts/data/（逐项核验） | 130 | Build a Hi4D two-person interaction test set from the native_interaction_v4_2 N2C pairs. | 0 |
| /home/shangguanrz/check_ab.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 31 | project | 0 |
| /home/shangguanrz/check_zero.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 16 | load | 0 |
| /home/shangguanrz/collect_commons.py | 项目外数据准备工具 | scripts/data/（逐项核验） | 133 | Collect HD full-body candidates from Wikimedia Commons for cross-body-type retargeting tests. | 0 |
| /home/shangguanrz/collect_commons_v2.py | 项目外数据准备工具 | scripts/data/（逐项核验） | 176 | Collect HD full-body candidates from Wikimedia Commons using API thumbnail URLs. | 0 |
| /home/shangguanrz/collect_round2.py | 项目外数据准备工具 | scripts/data/（逐项核验） | 144 | Round 2: target genuinely full-body heavy-set subjects on Wikimedia Commons. | 0 |
| /home/shangguanrz/collect_round2b.py | 项目外数据准备工具 | scripts/data/（逐项核验） | 163 | Round 2 (fixed): pick a thumbnail width strictly below the original so we never | 0 |
| /home/shangguanrz/commons_discover.py | 项目外数据准备工具 | scripts/data/（逐项核验） | 70 | api | 0 |
| /home/shangguanrz/compose_conditions.py | 项目外数据准备工具 | scripts/data/（逐项核验） | 229 | Compose every adapter input condition for Hi4D inter_01_pair00_taichi into one figure. | 0 |
| /home/shangguanrz/curate_testset.py | 项目外数据准备工具 | scripts/data/（逐项核验） | 231 | Curate the downloaded candidate pool into a cross-body-type retargeting test set. | 0 |
| /home/shangguanrz/diag_bbox.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 25 |  | 0 |
| /home/shangguanrz/diag_dwpose.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 14 |  | 0 |
| /home/shangguanrz/diag_hard12.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 61 | load, cmp | 0 |
| /home/shangguanrz/diag_joints.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 28 |  | 0 |
| /home/shangguanrz/diag_maskover.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 20 |  | 0 |
| /home/shangguanrz/diag_mesh.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 16 |  | 0 |
| /home/shangguanrz/diag_metric.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 31 |  | 0 |
| /home/shangguanrz/diag_overlay.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 25 |  | 0 |
| /home/shangguanrz/diag_reframe.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 20 |  | 0 |
| /home/shangguanrz/diag_vs_demo.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 19 | letterbox | 0 |
| /home/shangguanrz/diff_loss.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 25 | grab | 0 |
| /home/shangguanrz/draw_arch.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 175 | Draw the DeepGen-V6 vs Qwen21 adapter architecture comparison. | 0 |
| /home/shangguanrz/draw_arch2.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 165 | DeepGen-V6 vs Qwen21 adapter architecture — clean auto-sized layout. | 0 |
| /home/shangguanrz/dump_contracts.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 41 |  | 0 |
| /home/shangguanrz/dump_last.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 17 |  | 0 |
| /home/shangguanrz/dump_shapes.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 42 |  | 0 |
| /home/shangguanrz/dump_shapes2.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 26 |  | 0 |
| /home/shangguanrz/dump_shapes3.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 5 |  | 0 |
| /home/shangguanrz/eval_overfit.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 69 | m | 0 |
| /home/shangguanrz/eval_overfit2.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 68 | load_panel, mask_panel, m | 0 |
| /home/shangguanrz/hf_repo_size.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 26 |  | 0 |
| /home/shangguanrz/inspect_hi4d_params.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 49 |  | 0 |
| /home/shangguanrz/inspect_meta.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 11 |  | 0 |
| /home/shangguanrz/inter_review.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 53 | Review sheet: per interaction type, show SOURCE \| TARGET \| detailed \| minimal \| w_ref. | 0 |
| /home/shangguanrz/introspect_pipe.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 34 |  | 0 |
| /home/shangguanrz/locate_diff.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 25 |  | 0 |
| /home/shangguanrz/measure_person_region.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 75 | Re-measure w_ref fidelity inside the PERSON region only (green screen dominates full-frame SSIM). | 0 |
| /home/shangguanrz/measure_wref.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 73 | Quantify: how close is the w_ref output to the GT target? Is the model copying or regenerating? | 0 |
| /home/shangguanrz/ms_sweep.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 61 | Sweep ModelScope dataset search for anything usable for single-person pose retargeting. | 0 |
| /home/shangguanrz/nihaochen/audit_iper_face_30.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 143 | Inventory the fixed 30-step iPER face audit without changing model outputs. | 0 |
| /home/shangguanrz/nihaochen/docs/Qwen21_面部保持优化_实验记录_20260928.md | 项目外报告与证据 | reports/（建立索引后迁入） | 76 |  | 0 |
| /home/shangguanrz/nihaochen/docs/codex_briefing_image_editing.md | 项目外报告与证据 | reports/（建立索引后迁入） | 451 |  | 1 |
| /home/shangguanrz/nihaochen/docs/iPER_1024_结果.md | 项目外报告与证据 | reports/（建立索引后迁入） | 109 |  | 2 |
| /home/shangguanrz/nihaochen/docs/iPER_30步面部保持度重评_20260928.md | 项目外报告与证据 | reports/（建立索引后迁入） | 82 |  | 2 |
| /home/shangguanrz/nihaochen/docs/iPER_同等步数对比_20260928.md | 项目外报告与证据 | reports/（建立索引后迁入） | 41 |  | 1 |
| /home/shangguanrz/nihaochen/docs/pic_edit_code_guide_20260923.md | 项目外报告与证据 | reports/（建立索引后迁入） | 1309 |  | 0 |
| /home/shangguanrz/nihaochen/docs/pic_edit_code_guide_evidence_20260923/data_and_boundary_audit.md | 项目外报告与证据 | reports/（建立索引后迁入） | 260 |  | 1 |
| /home/shangguanrz/nihaochen/docs/pic_edit_code_guide_evidence_20260923/v65_face_audit.md | 项目外报告与证据 | reports/（建立索引后迁入） | 204 |  | 1 |
| /home/shangguanrz/nihaochen/docs/pic_edit_data_audit_notes.md | 项目外报告与证据 | reports/（建立索引后迁入） | 260 |  | 0 |
| /home/shangguanrz/nihaochen/docs/pic_edit_qwen21_deployment_audit_20260925.md | 项目外报告与证据 | reports/（建立索引后迁入） | 137 |  | 0 |
| /home/shangguanrz/nihaochen/docs/qwen_image21_architecture_20260925/Qwen-Image-2.1架构解析与项目迁移方案.md | 项目外报告与证据 | reports/（建立索引后迁入） | 533 |  | 0 |
| /home/shangguanrz/nihaochen/evaluate_qwen21_dev_gate.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 83 | Measure held-out Qwen21 pilot pose and face response before long training. | 1 |
| /home/shangguanrz/nihaochen/generate_iper_face_audit_docx.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 239 | Build a self-contained Word report from the reviewed 30-step face audit. | 0 |
| /home/shangguanrz/nihaochen/generate_qwen21_progress_docx.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 402 | metric, set_font, set_shading, no_split, paragraph | 0 |
| /home/shangguanrz/nihaochen/iper1024/continue_experiment.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 149 | Durable continuation. Test generation follows validation-only selection. | 0 |
| /home/shangguanrz/nihaochen/reconstruction_summary_20261001/build_summary.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 185 | Create the concise reconstruction summary from existing recorded results. | 0 |
| /home/shangguanrz/nihaochen/render_equal_step_tables.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 108 | Render the fixed 50-pair, matched-step DeepGen/Qwen21 results. | 0 |
| /home/shangguanrz/nihaochen/run_qwen21_remaining.sh | 项目外运行编排 | scripts/run/（逐项核验） | 83 |  | 0 |
| /home/shangguanrz/nihaochen/scripts/export_iper1024_word.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 77 | inline | 0 |
| /home/shangguanrz/nihaochen/scripts/revise_iper1024_word.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 84 | para, heading, table | 0 |
| /home/shangguanrz/nihaochen/summarize_iper_face_30.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 256 | Summarize the manually reviewed 30-step iPER face inventory. | 0 |
| /home/shangguanrz/patch_inference.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 18 |  | 0 |
| /home/shangguanrz/phaseD_delete_old.sh | 项目外运行编排 | scripts/run/（逐项核验） | 92 |  | 0 |
| /home/shangguanrz/preview_pairs.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 43 | Render a side-by-side preview of each test pair for visual validation. | 0 |
| /home/shangguanrz/preview_types.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 41 | Preview one sample per interaction type, source\|target, 3 rows per page. | 0 |
| /home/shangguanrz/probe_hi4d_labels.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 31 |  | 0 |
| /home/shangguanrz/proj_smpl.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 43 | project | 0 |
| /home/shangguanrz/project/pic-edit/README.md | 项目说明、依赖或环境入口（保留） | README.md | 83 |  | 2 |
| /home/shangguanrz/project/pic-edit/THIRD_PARTY_NOTICES.md | 项目说明、依赖或环境入口（保留） | THIRD_PARTY_NOTICES.md | 89 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/datasets/__init__.py | 独立数据制备工具链（保留） | add/datasets/__init__.py | 0 |  | 105 |
| /home/shangguanrz/project/pic-edit/add/datasets/dpw_single_pose_edit_dataset.py | 独立数据制备工具链（保留） | add/datasets/dpw_single_pose_edit_dataset.py | 1253 | DPWSinglePoseEditDataset | 7 |
| /home/shangguanrz/project/pic-edit/add/datasets/dpw_single_pose_edit_dataset_before_dwpose_qc_mapping_20260930_090438.py | 历史备份候选 | archive/data_pipeline/（核验后） | 1134 | DPWSinglePoseEditDataset | 0 |
| /home/shangguanrz/project/pic-edit/add/datasets/dpw_single_pose_edit_dataset_before_use_dwpose_render_20260930_090016.py | 历史备份候选 | archive/data_pipeline/（核验后） | 1134 | DPWSinglePoseEditDataset | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/01_download_humanvid_batch_v1.py | 独立数据制备工具链（保留） | add/tools/01_download_humanvid_batch_v1.py | 407 | free_gb, parse_fraction, ffprobe, build_download_order, load_batch | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/02_humanvid_sparse_person_census_v1.py | 独立数据制备工具链（保留） | add/tools/02_humanvid_sparse_person_census_v1.py | 409 | sample_video, classify_counts, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/03_humanvid_shot_detection_v1.py | 独立数据制备工具链（保留） | add/tools/03_humanvid_shot_detection_v1.py | 218 | detect_shots, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/04_humanvid_dense_person_segments_v1.py | 独立数据制备工具链（保留） | add/tools/04_humanvid_dense_person_segments_v1.py | 634 | read_frame_at, detect_batch, build_segments, quantile, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/05_humanvid_segment_sensitivity_v1.py | 独立数据制备工具链（保留） | add/tools/05_humanvid_segment_sensitivity_v1.py | 477 | load_csv, make_runs, count_isolated_blips, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/06_build_humanvid_core_segments_v1.py | 独立数据制备工具链（保留） | add/tools/06_build_humanvid_core_segments_v1.py | 300 | read_csv, write_csv, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/07_humanvid_adaptive_sampling_v1.py | 独立数据制备工具链（保留） | add/tools/07_humanvid_adaptive_sampling_v1.py | 1678 | read_csv, write_csv, quantile, read_frame, extract_pose_people | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/08_prepare_humanvid_dwpose_input_v1.py | 独立数据制备工具链（保留） | add/tools/08_prepare_humanvid_dwpose_input_v1.py | 242 | read_csv, write_csv, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/09_humanvid_dwpose_frame_pool_v1.py | 独立数据制备工具链（保留） | add/tools/09_humanvid_dwpose_frame_pool_v1.py | 467 | read_csv, write_csv, safe_float, calc_bbox_from_visible_points, analyze_person | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/10_humanvid_pose_descriptors_v1.py | 独立数据制备工具链（保留） | add/tools/10_humanvid_pose_descriptors_v1.py | 780 | read_csv, write_csv, normalize_body_pose, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/11_humanvid_frame_quality_v1.py | 独立数据制备工具链（保留） | add/tools/11_humanvid_frame_quality_v1.py | 818 | read_csv, write_csv, crop_from_points, normalized_sharpness, percentile | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/12_humanvid_pose_coverage_v1.py | 独立数据制备工具链（保留） | add/tools/12_humanvid_pose_coverage_v1.py | 1051 | read_csv, write_csv, load_descriptor, one_person_pose_distance, single_frame_distance | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/13_humanvid_video_split_v1.py | 独立数据制备工具链（保留） | add/tools/13_humanvid_video_split_v1.py | 482 | read_csv, write_csv, allocate_counts, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/14_humanvid_build_pairs_v1.py | 独立数据制备工具链（保留） | add/tools/14_humanvid_build_pairs_v1.py | 1278 | read_csv, write_csv, load_descriptor, one_person_distance, single_distance | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/15_humanvid_pair_background_audit_v1.py | 独立数据制备工具链（保留） | add/tools/15_humanvid_pair_background_audit_v1.py | 582 | read_csv, write_csv, valid_person_boxes, add_box_to_mask, prepare_image | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/16_humanvid_build_local_pairs_v2.py | 独立数据制备工具链（保留） | add/tools/16_humanvid_build_local_pairs_v2.py | 1305 | read_csv, write_csv, load_descriptor, one_person_distance, single_distance | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/17_visualize_humanvid_pairs_v2.py | 独立数据制备工具链（保留） | add/tools/17_visualize_humanvid_pairs_v2.py | 796 | read_csv, fit_image, render_one_person_pose, render_pose, add_text | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/18_humanvid_single_manual_qc_pack_v1.py | 独立数据制备工具链（保留） | add/tools/18_humanvid_single_manual_qc_pack_v1.py | 536 | read_csv, write_csv, fit_image, put_text, make_pair_card | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/19_humanvid_single_robust_pose_audit_v1.py | 独立数据制备工具链（保留） | add/tools/19_humanvid_single_robust_pose_audit_v1.py | 959 | read_csv, write_csv, load_pose, trimmed_mean, analyze_pair | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/20_humanvid_single_bone_angle_audit_v1.py | 独立数据制备工具链（保留） | add/tools/20_humanvid_single_bone_angle_audit_v1.py | 900 | read_csv, write_csv, load_dwpose, vector_angle_deg, joint_angle_deg | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/21_finalize_humanvid_single_pairs_v1.py | 独立数据制备工具链（保留） | add/tools/21_finalize_humanvid_single_pairs_v1.py | 291 | read_csv, write_csv, is_too_similar, change_strength, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/22_extract_3dpw_single_dwpose_wholebody_v1.py | 独立数据制备工具链（保留） | add/tools/22_extract_3dpw_single_dwpose_wholebody_v1.py | 2236 | sequence_images, image_path_for_row, wholebody_postprocess, detected_body_center, select_person | 1 |
| /home/shangguanrz/project/pic-edit/add/tools/22_visualize_humanvid_final_single_v1.py | 独立数据制备工具链（保留） | add/tools/22_visualize_humanvid_final_single_v1.py | 418 | read_csv, fit_image, put_text, make_card, sample_distinct_videos | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/23_extract_3dpw_single_person_masks_v1.py | 独立数据制备工具链（保留） | add/tools/23_extract_3dpw_single_person_masks_v1.py | 1546 | sequence_images, image_path, prepare_mask, mask_centroid, select_person_mask | 1 |
| /home/shangguanrz/project/pic-edit/add/tools/23_finalize_humanvid_single_pairs_v2.py | 独立数据制备工具链（保留） | add/tools/23_finalize_humanvid_single_pairs_v2.py | 434 | read_csv, write_csv, f, i, too_similar_v2 | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/24_visualize_humanvid_single_boundary_v2.py | 独立数据制备工具链（保留） | add/tools/24_visualize_humanvid_single_boundary_v2.py | 429 | read_csv, fit_image, put_text, make_card, save_sheet | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/25_humanvid_single_pair_exposure_audit_v1.py | 独立数据制备工具链（保留） | add/tools/25_humanvid_single_pair_exposure_audit_v1.py | 288 | read_csv, get_frame_id, stats, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/26_audit_humanvid_pair_usage_v1.py | 独立数据制备工具链（保留） | add/tools/26_audit_humanvid_pair_usage_v1.py | 745 | read_csv, pct, print_distribution, canonical_pair, audit_pair_rows | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/26_humanvid_single_all_candidate_bone_audit_v1.py | 独立数据制备工具链（保留） | add/tools/26_humanvid_single_all_candidate_bone_audit_v1.py | 1408 | read_csv, write_csv, find_field, build_frame_index, resolve_frame_row | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/27_audit_3dpw_single_smpl_geometry_v1.py | 独立数据制备工具链（保留） | add/tools/27_audit_3dpw_single_smpl_geometry_v1.py | 1317 | find_sequence_pkl, load_pkl, person0_item, normalize_gender, get_gender | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/27_build_3dpw_single_final_handoff_v1.py | 独立数据制备工具链（保留） | add/tools/27_build_3dpw_single_final_handoff_v1.py | 1405 | rel, image_list, find_column, build_candidate_pose_bin_lookup, normalize_pair_manifest | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/27_humanvid_single_capacity_matching_v1.py | 独立数据制备工具链（保留） | add/tools/27_humanvid_single_capacity_matching_v1.py | 883 | read_csv, write_csv, quality_rank, candidate_sort_key, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/28_humanvid_bone_metric_consistency_audit_v1.py | 独立数据制备工具链（保留） | add/tools/28_humanvid_bone_metric_consistency_audit_v1.py | 336 | read_csv, write_csv, key, as_float, close | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/28_render_3dpw_single_smpl_depth_normal_v1.py | 独立数据制备工具链（保留） | add/tools/28_render_3dpw_single_smpl_depth_normal_v1.py | 2206 | normalize_gender, build_image_cache, get_rgb_path, get_mask_path, apply_cam_pose | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/29_audit_3dpw_single_camera_reprojection_v1.py | 独立数据制备工具链（保留） | add/tools/29_audit_3dpw_single_camera_reprojection_v1.py | 1144 | 3DPW final-single camera reprojection audit v1 | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/29_humanvid_single_build_directed_candidates_v1.py | 独立数据制备工具链（保留） | add/tools/29_humanvid_single_build_directed_candidates_v1.py | 715 | read_csv, write_csv, f, i, reclassify | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/30_audit_3dpw_crosscountry_se3_correction_v1.py | 独立数据制备工具链（保留） | add/tools/30_audit_3dpw_crosscountry_se3_correction_v1.py | 997 | poses2d_array, project_to_so3, rotation_distance_deg, transform_points, reprojection_error | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/30_humanvid_single_capacity_matching_final_v1.py | 独立数据制备工具链（保留） | add/tools/30_humanvid_single_capacity_matching_final_v1.py | 909 | read_csv, write_csv, quality_rank, candidate_sort_key, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/31_finalize_humanvid_single_dataset_v1.py | 独立数据制备工具链（保留） | add/tools/31_finalize_humanvid_single_dataset_v1.py | 523 | read_csv, write_csv, directed_key, unordered_key, choose_direction | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/31_render_3dpw_single_smpl_depth_normal_official_v2.py | 独立数据制备工具链（保留） | add/tools/31_render_3dpw_single_smpl_depth_normal_official_v2.py | 2206 | normalize_gender, build_image_cache, get_rgb_path, get_mask_path, apply_cam_pose | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/32_build_humanvid_full_pipeline_manifest_v1.py | 独立数据制备工具链（保留） | add/tools/32_build_humanvid_full_pipeline_manifest_v1.py | 506 | read_csv, write_csv, find_field, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/32_review_3dpw_single_geometry_tail_v1.py | 独立数据制备工具链（保留） | add/tools/32_review_3dpw_single_geometry_tail_v1.py | 764 | find_rgb, find_mask, resize_keep, normalize_depth, geometry_overlay | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/33_build_3dpw_single_handoff_geometry_final_v2.py | 独立数据制备工具链（保留） | add/tools/33_build_3dpw_single_handoff_geometry_final_v2.py | 1954 | 3DPW single final handoff v2 | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/33_build_humanvid_asset_pilot_manifest_v2.py | 独立数据制备工具链（保留） | add/tools/33_build_humanvid_asset_pilot_manifest_v2.py | 350 | read_csv, stable_key, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/34_audit_humanvid_pilot_video_resolution_v1.py | 独立数据制备工具链（保留） | add/tools/34_audit_humanvid_pilot_video_resolution_v1.py | 304 | orientation, resolution_class, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/34_pack_3dpw_single_training_dataset_v1.py | 独立数据制备工具链（保留） | add/tools/34_pack_3dpw_single_training_dataset_v1.py | 661 | materialize_asset, process_split, verify | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/35_audit_humanvid_pair_resolution_and_crop_v1.py | 独立数据制备工具链（保留） | add/tools/35_audit_humanvid_pair_resolution_and_crop_v1.py | 1159 | read_csv, write_csv, find_first, load_dwpose_bbox, square_feasible | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/36_audit_humanvid_pair_square_crop_v1.py | 独立数据制备工具链（保留） | add/tools/36_audit_humanvid_pair_square_crop_v1.py | 534 | read_csv, write_csv, percentile, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/36_make_3dpw_showcase100_humanvid_pose_v1.py | 独立数据制备工具链（保留） | add/tools/36_make_3dpw_showcase100_humanvid_pose_v1.py | 620 | ensure_pil, find_descriptor_table, find_existing_path, build_seq_frame_to_row, RawPoseCache | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/37_run_humanvid_native_asset_pilot_v1.py | 独立数据制备工具链（保留） | add/tools/37_run_humanvid_native_asset_pilot_v1.py | 571 | read_csv, ensure_dir, safe_unlink, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/38_audit_humanvid_native_asset_pilot_v1.py | 独立数据制备工具链（保留） | add/tools/38_audit_humanvid_native_asset_pilot_v1.py | 146 | ensure_dir, read_csv, write_csv, load_image_shape, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/39_make_humanvid_native_asset_smoke_preview_v1.py | 独立数据制备工具链（保留） | add/tools/39_make_humanvid_native_asset_smoke_preview_v1.py | 135 | ensure_dir, read_csv, fit_h, pad_to_w, add_label | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/39_make_humanvid_native_asset_smoke_preview_v2.py | 独立数据制备工具链（保留） | add/tools/39_make_humanvid_native_asset_smoke_preview_v2.py | 450 | ensure_dir, read_csv, fit_h, pad_to_width, add_label | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/39_make_humanvid_native_asset_smoke_preview_v3.py | 独立数据制备工具链（保留） | add/tools/39_make_humanvid_native_asset_smoke_preview_v3.py | 468 | read_csv, fit_height, pad_width, label_tile, mask_overlay | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/3dpw_dataset_audit.py | 独立数据制备工具链（保留） | add/tools/3dpw_dataset_audit.py | 720 | get_image_size, list_images, safe_mean, safe_median | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/3dpw_frame_quality_audit.py | 独立数据制备工具链（保留） | add/tools/3dpw_frame_quality_audit.py | 1783 | get_images, extract_frame_number, make_image_lookup, resolve_image_path, clamp_bbox | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/40_build_humanvid_sapiens2_foreground_pilot_v1.py | 独立数据制备工具链（保留） | add/tools/40_build_humanvid_sapiens2_foreground_pilot_v1.py | 312 | read_csv, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/40_build_humanvid_sapiens_foreground_pilot_v2.py | 独立数据制备工具链（保留） | add/tools/40_build_humanvid_sapiens_foreground_pilot_v2.py | 321 | preprocess, get_logits, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/42_audit_humanvid_smplestx_dwpose_v1.py | 独立数据制备工具链（保留） | add/tools/42_audit_humanvid_smplestx_dwpose_v1.py | 1884 | add, read_csv, norm_path, percentile_safe, mean_safe | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/42_audit_humanvid_smplestx_dwpose_v2.py | 独立数据制备工具链（保留） | add/tools/42_audit_humanvid_smplestx_dwpose_v2.py | 1632 | read_csv, safe_mean, safe_median, safe_percentile, find_human_model_root | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/43_build_humanvid_hmr_compare16_v1.py | 独立数据制备工具链（保留） | add/tools/43_build_humanvid_hmr_compare16_v1.py | 336 | main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/45_run_humanvid_multihmr_compare16_v1.py | 独立数据制备工具链（保留） | add/tools/45_run_humanvid_multihmr_compare16_v1.py | 2248 | read_csv, ffloat, finite, safe_median, safe_mean | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/46_diagnose_multihmr_detection_failures_v1.py | 独立数据制备工具链（保留） | add/tools/46_diagnose_multihmr_detection_failures_v1.py | 179 | detection_map_stats, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/47_run_humanvid_multihmr_compare16_thr020_v1.py | 独立数据制备工具链（保留） | add/tools/47_run_humanvid_multihmr_compare16_thr020_v1.py | 2248 | read_csv, ffloat, finite, safe_median, safe_mean | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/49_build_humanvid_sam21_foreground_pilot_v1.py | 独立数据制备工具链（保留） | add/tools/49_build_humanvid_sam21_foreground_pilot_v1.py | 980 | HumanVid visible-person foreground mask pilot v1 | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/49b_build_humanvid_sam21_risk_previews_v1.py | 独立数据制备工具链（保留） | add/tools/49b_build_humanvid_sam21_risk_previews_v1.py | 345 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/50_build_humanvid_yolo11xseg_foreground_pilot_v1.py | 独立数据制备工具链（保留） | add/tools/50_build_humanvid_yolo11xseg_foreground_pilot_v1.py | 849 | HumanVid visible-person foreground mask pilot | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/51a_audit_current_asset_dependencies_v2.py | 独立数据制备工具链（保留） | add/tools/51a_audit_current_asset_dependencies_v2.py | 577 | Audit whether current training/model code actually consumes: | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/53_run_humanvid_native_assets_full_v1.py | 独立数据制备工具链（保留） | add/tools/53_run_humanvid_native_assets_full_v1.py | 1448 | HumanVid SINGLE native geometry full build v1 | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/53c_isolate_smplestx_batch001_failure_v1.py | 独立数据制备工具链（保留） | add/tools/53c_isolate_smplestx_batch001_failure_v1.py | 592 | Diagnose HumanVid batch_001 SMPLest-X row mismatch. | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/54_diagnose_all_failed_humanvid_batches_v1.py | 独立数据制备工具链（保留） | add/tools/54_diagnose_all_failed_humanvid_batches_v1.py | 648 | Diagnose every failed HumanVid geometry batch. | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/55_build_humanvid_asset_showcase40_v1.py | 独立数据制备工具链（保留） | add/tools/55_build_humanvid_asset_showcase40_v1.py | 1217 | Build 40 HumanVid per-frame asset showcase images. | 3 |
| /home/shangguanrz/project/pic-edit/add/tools/55b_build_humanvid_pair_showcase40_v1.py | 独立数据制备工具链（保留） | add/tools/55b_build_humanvid_pair_showcase40_v1.py | 1017 | HumanVid pair-level showcase. | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/55c_build_humanvid_pair_showcase40_v1.py | 独立数据制备工具链（保留） | add/tools/55c_build_humanvid_pair_showcase40_v1.py | 1066 | HumanVid pair showcase 40. | 1 |
| /home/shangguanrz/project/pic-edit/add/tools/55d_build_humanvid_pair_showcase60_best_v1.py | 独立数据制备工具链（保留） | add/tools/55d_build_humanvid_pair_showcase60_best_v1.py | 731 | HumanVid high-quality showcase candidate selection. | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_assets_v1.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_assets_v1.py | 94 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_pair_constraint_sensitivity_v1.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_pair_constraint_sensitivity_v1.py | 1965 | add_distribution_stats, concat_chunks, pose_pairwise_block, scale_ratio_block, camera_translation_block | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_pair_visibility_v3.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_pair_visibility_v3.py | 1 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_pairwise_distribution_v1.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_pairwise_distribution_v1.py | 1754 | stats, add_stats, concat, pose_pairwise_block, scale_ratio_block | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_pose_candidate_coverage_v1.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_pose_candidate_coverage_v1.py | 983 | pose_distance_block, scale_ratio_block, camera_translation_block, camera_rotation_block, add_count_stats | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_single_dynamic_sampler_v1.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_single_dynamic_sampler_v1.py | 638 | choose_target | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_single_dynamic_sampler_v2.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_single_dynamic_sampler_v2.py | 860 | choose_target | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_single_relative_motion_blur_v1.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_single_relative_motion_blur_v1.py | 1063 | pose_distance, sequence_images, load_rgb, person_crop, tenengrad | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_single_representatives_final_v1.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_single_representatives_final_v1.py | 653 | pose_distance, singleton_stats | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/audit_3dpw_single_temporal_thinning_v1.py | 独立数据制备工具链（保留） | add/tools/audit_3dpw_single_temporal_thinning_v1.py | 582 | pose_distance, greedy_select, qstats | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_all_person_assets_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_all_person_assets_v1.py | 452 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_assets_v2_1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_assets_v2_1.py | 304 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_candidate_pools_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_candidate_pools_v1.py | 2039 | pose_distance_block, scale_ratio_block, camera_translation_block, camera_rotation_block, check_descriptor_index | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_extra_person_metrics_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_extra_person_metrics_v1.py | 1154 | get_pose, pose_bbox, area_ratio, bbox_iou, joint_inside_ratio | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_frame_assets_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_frame_assets_v1.py | 231 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_frame_assets_v2.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_frame_assets_v2.py | 237 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_frame_descriptors_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_frame_descriptors_v1.py | 1059 | get_pose_array, valid_joint, midpoint, person_center, dist | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_frame_pools_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_frame_pools_v1.py | 966 | prepare_person | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_frame_quality_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_frame_quality_v1.py | 391 | laplacian_score, bbox_area_ratio, crop_safe, pose_visibility, border_risk | 1 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_frame_quality_v2.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_frame_quality_v2.py | 680 | laplacian_score, bbox_area_ratio, crop_safe, pose_visibility, visible_body_joints | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_pair_distribution_sample_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_pair_distribution_sample_v1.py | 1101 | pose_distance, symmetric_scale_ratio, camera_translation_distance, camera_rotation_distance_deg, band_candidate_positions | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_pair_quality_v3.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_pair_quality_v3.py | 345 | make_side, reject_reason | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_pairs_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_pairs_v1.py | 485 | get_images, get_sequence_pkl, load_pkl, safe_quantile, build_image_map | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_pairs_v4_dedup.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_pairs_v4_dedup.py | 434 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_person_qc_metrics_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_person_qc_metrics_v1.py | 1318 | get_pose, body_visibility, limb_coverage, pose_bbox, area_ratio | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_single_candidate_pool_v2.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_single_candidate_pool_v2.py | 1700 | pose_distance_block, scale_ratio_block, camera_translation_block, camera_rotation_block, build_csr | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_single_eval_manifests_natural_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_single_eval_manifests_natural_v1.py | 941 | get_targets, available_bins, build_split, make_summary | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_single_final_frame_pool_v2.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_single_final_frame_pool_v2.py | 810 | pose_distance, is_near_duplicate | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_single_frame_descriptors_v2.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_single_frame_descriptors_v2.py | 649 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_single_sampling_metadata_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_single_sampling_metadata_v1.py | 495 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_single_sampling_metadata_v2.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_single_sampling_metadata_v2.py | 508 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_single_temporal_representatives_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_single_temporal_representatives_v1.py | 1470 | pose_distance, build_groups, quantiles | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_3dpw_usable_frames_v1.py | 独立数据制备工具链（保留） | add/tools/build_3dpw_usable_frames_v1.py | 414 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_humanvid_url_manifest_v1.py | 独立数据制备工具链（保留） | add/tools/build_humanvid_url_manifest_v1.py | 101 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/build_usable_frames.py | 独立数据制备工具链（保留） | add/tools/build_usable_frames.py | 78 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/extract_3dpw_frame_person_stats_v3.py | 独立数据制备工具链（保留） | add/tools/extract_3dpw_frame_person_stats_v3.py | 198 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/make_3dpw_extra_person_review_v1.py | 独立数据制备工具链（保留） | add/tools/make_3dpw_extra_person_review_v1.py | 360 | sample_rows, fit_image, make_sheet | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/make_3dpw_qc_boundary_review_v1.py | 独立数据制备工具链（保留） | add/tools/make_3dpw_qc_boundary_review_v1.py | 482 | sample_df, letterbox, make_sheet | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/make_3dpw_single_v2_blur_contact_sheets.py | 独立数据制备工具链（保留） | add/tools/make_3dpw_single_v2_blur_contact_sheets.py | 798 | get_sequence_images, resolve_image, person_crop, make_contain, make_crop_thumb | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/refine_3dpw_pairs_v2.py | 独立数据制备工具链（保留） | add/tools/refine_3dpw_pairs_v2.py | 161 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/refine_3dpw_usable_frames_v1_1.py | 独立数据制备工具链（保留） | add/tools/refine_3dpw_usable_frames_v1_1.py | 325 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/run_3dpw_single_conditions_v1.sh | 独立数据制备工具链（保留） | add/tools/run_3dpw_single_conditions_v1.sh | 437 |  | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/visualize_3dpw_pairs.py | 独立数据制备工具链（保留） | add/tools/visualize_3dpw_pairs.py | 127 | resize_keep | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/visualize_3dpw_v3_review.py | 独立数据制备工具链（保留） | add/tools/visualize_3dpw_v3_review.py | 246 | resize_keep, make_preview, export_group | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/visualize_3dpw_v3_review_balanced.py | 独立数据制备工具链（保留） | add/tools/visualize_3dpw_v3_review_balanced.py | 201 | resize_keep, make_preview, diverse_select | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/visualize_dpw_single_pose_edit_dataset.py | 独立数据制备工具链（保留） | add/tools/visualize_dpw_single_pose_edit_dataset.py | 614 | tensor_to_pil, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/visualize_dpw_single_pose_edit_dataset_backup_20260930_083758.py | 历史备份候选 | archive/data_pipeline/（核验后） | 614 | tensor_to_pil, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/visualize_dpw_single_pose_edit_dataset_before_black_pose_20260930_084339.py | 历史备份候选 | archive/data_pipeline/（核验后） | 614 | tensor_to_pil, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/visualize_dpw_single_pose_edit_dataset_before_color_pose_20260930_084629.py | 历史备份候选 | archive/data_pipeline/（核验后） | 660 | tensor_to_pil, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/visualize_dpw_single_pose_edit_dataset_before_dwpose_color_v2_20260930_084922.py | 历史备份候选 | archive/data_pipeline/（核验后） | 660 | tensor_to_pil, main | 0 |
| /home/shangguanrz/project/pic-edit/add/tools/visualize_dpw_single_pose_edit_dataset_before_safe_color_pose_20260930_085112.py | 历史备份候选 | archive/data_pipeline/（核验后） | 660 | tensor_to_pil, main | 0 |
| /home/shangguanrz/project/pic-edit/audit_hi4d_qwen21_prompt_leak.py | 审计、验证或诊断入口 | scripts/audit/audit_hi4d_qwen21_prompt_leak.py | 124 | Check target-RGB isolation and target-geometry sensitivity on one Hi4D source. | 0 |
| /home/shangguanrz/project/pic-edit/audit_tiktok_v3_blur_sample.py | 审计、验证或诊断入口 | scripts/audit/audit_tiktok_v3_blur_sample.py | 472 | read_jsonl, read_gray, load_binary_mask, sharpness_scores, make_panel | 0 |
| /home/shangguanrz/project/pic-edit/audit_tiktok_v3_final_pairs.py | 审计、验证或诊断入口 | scripts/audit/audit_tiktok_v3_final_pairs.py | 289 | read_jsonl, main | 0 |
| /home/shangguanrz/project/pic-edit/audit_tiktok_v3_pose_redundancy.py | 审计、验证或诊断入口 | scripts/audit/audit_tiktok_v3_pose_redundancy.py | 1140 | read_jsonl, write_jsonl, load_pose_descriptor, pose_distance, nearest_for_radius | 0 |
| /home/shangguanrz/project/pic-edit/audit_tiktok_v3_strict_final.py | 审计、验证或诊断入口 | scripts/audit/audit_tiktok_v3_strict_final.py | 584 | read_jsonl, resize_h, fit_width, make_panel, main | 0 |
| /home/shangguanrz/project/pic-edit/audit_tiktok_v3_visual_quality_v2.py | 审计、验证或诊断入口 | scripts/audit/audit_tiktok_v3_visual_quality_v2.py | 1833 | read_jsonl, read_bgr, read_mask, edge_sharpness, load_dwpose_face | 0 |
| /home/shangguanrz/project/pic-edit/audit_tiktok_v3_visual_quality_v3.py | 审计、验证或诊断入口 | scripts/audit/audit_tiktok_v3_visual_quality_v3.py | 2612 | read_jsonl, read_bgr, read_binary_mask, edge_sharpness, load_dwpose | 0 |
| /home/shangguanrz/project/pic-edit/build_20b_memory_docx.py | 报告或可视化生成 | scripts/reports/build_20b_memory_docx.py | 299 | 生成《Qwen-Image-Edit 20B 显存占用预算与单卡可行性分析简报》Word文档 | 0 |
| /home/shangguanrz/project/pic-edit/build_architecture_docx.py | 报告或可视化生成 | scripts/reports/build_architecture_docx.py | 598 | 生成《人物图像姿态编辑系统架构设计与核心技术方案》Word文档 | 0 |
| /home/shangguanrz/project/pic-edit/build_brief_architecture_docx.py | 报告或可视化生成 | scripts/reports/build_brief_architecture_docx.py | 355 | 生成《人物图像姿态编辑项目技术架构与进展简报（科研同步版）》Word文档 | 0 |
| /home/shangguanrz/project/pic-edit/build_latest_experiment_report_docx.py | 报告或可视化生成 | scripts/reports/build_latest_experiment_report_docx.py | 377 | 生成《DeepGen 1.0 图像姿态编辑最新实验结果与显存分析简报》Word文档 | 0 |
| /home/shangguanrz/project/pic-edit/build_tiktok_v3_candidate_pool.py | 数据、条件或实验收尾入口 | scripts/data/build_tiktok_v3_candidate_pool.py | 1228 | TikTok V3 formal candidate-pool construction. | 1 |
| /home/shangguanrz/project/pic-edit/build_tiktok_v3_condition_inventory.py | 数据、条件或实验收尾入口 | scripts/data/build_tiktok_v3_condition_inventory.py | 785 | TikTok V3 - Step 6 | 0 |
| /home/shangguanrz/project/pic-edit/build_tiktok_v3_pairs.py | 数据、条件或实验收尾入口 | scripts/data/build_tiktok_v3_pairs.py | 871 | read_jsonl, load_dwpose, pose_descriptor, pose_distance, source_quality | 0 |
| /home/shangguanrz/project/pic-edit/build_tiktok_v3_pairs_strict.py | 数据、条件或实验收尾入口 | scripts/data/build_tiktok_v3_pairs_strict.py | 871 | read_jsonl, load_dwpose, pose_descriptor, pose_distance, source_quality | 0 |
| /home/shangguanrz/project/pic-edit/build_tiktok_v3_strict_visual_pool.py | 数据、条件或实验收尾入口 | scripts/data/build_tiktok_v3_strict_visual_pool.py | 345 | read_jsonl, as_bool, main | 0 |
| /home/shangguanrz/project/pic-edit/build_tiktok_v3_usable_pool.py | 数据、条件或实验收尾入口 | scripts/data/build_tiktok_v3_usable_pool.py | 321 | read_jsonl | 0 |
| /home/shangguanrz/project/pic-edit/complete_hi4d_qwen21_dual.py | 实验编排、运行或辅助工具 | scripts/run/complete_hi4d_qwen21_dual.py | 164 | Finish the long Qwen21 dual run, then evaluate the held-out Hi4D test split. | 0 |
| /home/shangguanrz/project/pic-edit/create_debug_samples.py | 数据、条件或实验收尾入口 | scripts/data/create_debug_samples.py | 170 | generate_debug_dataset | 1 |
| /home/shangguanrz/project/pic-edit/docs/adapter_v6_architecture.md | 架构、运行说明或历史计划 | docs/adapter_v6_architecture.md | 139 |  | 1 |
| /home/shangguanrz/project/pic-edit/docs/champ_deepgen_single_sample_runbook.md | 架构、运行说明或历史计划 | docs/champ_deepgen_single_sample_runbook.md | 148 |  | 2 |
| /home/shangguanrz/project/pic-edit/docs/iper_v6_training_plan.md | 架构、运行说明或历史计划 | docs/iper_v6_training_plan.md | 217 |  | 1 |
| /home/shangguanrz/project/pic-edit/docs/qwen21_hi4d_dual.md | 架构、运行说明或历史计划 | docs/qwen21_hi4d_dual.md | 58 |  | 0 |
| /home/shangguanrz/project/pic-edit/docs/superpowers/plans/2026-09-12-champ-deepgen-single-sample.md | 架构、运行说明或历史计划 | docs/superpowers/plans/2026-09-12-champ-deepgen-single-sample.md | 457 |  | 0 |
| /home/shangguanrz/project/pic-edit/docs/superpowers/plans/2026-09-12-deepgen-no-control-ablation.md | 架构、运行说明或历史计划 | docs/superpowers/plans/2026-09-12-deepgen-no-control-ablation.md | 457 |  | 0 |
| /home/shangguanrz/project/pic-edit/docs/superpowers/plans/2026-09-15-adapter-reasoning-v1.md | 架构、运行说明或历史计划 | docs/superpowers/plans/2026-09-15-adapter-reasoning-v1.md | 14 |  | 0 |
| /home/shangguanrz/project/pic-edit/docs/superpowers/plans/2026-09-15-smplx-condition-injection-v1.md | 架构、运行说明或历史计划 | docs/superpowers/plans/2026-09-15-smplx-condition-injection-v1.md | 30 |  | 0 |
| /home/shangguanrz/project/pic-edit/docs/superpowers/plans/2026-09-16-deepgen-dynamic-control-interface-v1.md | 架构、运行说明或历史计划 | docs/superpowers/plans/2026-09-16-deepgen-dynamic-control-interface-v1.md | 34 |  | 0 |
| /home/shangguanrz/project/pic-edit/docs/superpowers/specs/2026-09-12-champ-deepgen-single-sample-design.md | 架构、运行说明或历史计划 | docs/superpowers/specs/2026-09-12-champ-deepgen-single-sample-design.md | 125 |  | 0 |
| /home/shangguanrz/project/pic-edit/docs/superpowers/specs/2026-09-12-deepgen-no-control-ablation-design.md | 架构、运行说明或历史计划 | docs/superpowers/specs/2026-09-12-deepgen-no-control-ablation-design.md | 84 |  | 0 |
| /home/shangguanrz/project/pic-edit/eval_hi4d_qwen21_dual.py | 评测入口 | scripts/eval/eval_hi4d_qwen21_dual.py | 327 | Generate and score native / geometry / full Qwen21 Hi4D controls. | 2 |
| /home/shangguanrz/project/pic-edit/eval_iper_four_arm.py | 评测入口 | scripts/eval/eval_iper_four_arm.py | 320 | Evaluate saved DeepGen/Qwen21 images using the same CPU metrics and pairs. | 7 |
| /home/shangguanrz/project/pic-edit/eval_iper_qwen21_body.py | 评测入口 | scripts/eval/eval_iper_qwen21_body.py | 176 | Generate Qwen21 adapter OFF/ON images on the fixed iPER comparison pairs. | 3 |
| /home/shangguanrz/project/pic-edit/eval_iper_qwen21_face.py | 评测入口 | scripts/eval/eval_iper_qwen21_face.py | 139 | Sample the frozen Qwen21 Body checkpoint with a separately trained Face branch. | 2 |
| /home/shangguanrz/project/pic-edit/eval_iper_qwen21_full.py | 评测入口 | scripts/eval/eval_iper_qwen21_full.py | 185 | Complete 30-step sampling for the preserved full controller and Qwen21. | 5 |
| /home/shangguanrz/project/pic-edit/eval_iper_v6.py | 评测入口 | scripts/eval/eval_iper_v6.py | 414 | Evaluate trained UnifiedSMPLXAdapterV6 on iPER validation or test sets. | 5 |
| /home/shangguanrz/project/pic-edit/eval_qwen21_face_cross_person.py | 评测入口 | scripts/eval/eval_qwen21_face_cross_person.py | 116 | Qualitative A/B different-person check: B supplies geometry, never identity pixels. | 0 |
| /home/shangguanrz/project/pic-edit/eval_qwen21_face_references.py | 评测入口 | scripts/eval/eval_qwen21_face_references.py | 137 | Paired Qwen21 Body 20k probe: source image with or without its own face crop. | 2 |
| /home/shangguanrz/project/pic-edit/eval_qwen21_face_variants.py | 评测入口 | scripts/eval/eval_qwen21_face_variants.py | 188 | Measure saved Qwen21 Face variants on the same pairs and valid face subset. | 1 |
| /home/shangguanrz/project/pic-edit/experiments/16-test/derive_camera.py | 16 组横评可复现入口 | scripts/benchmarks/16_test/（路径及结果配置解耦后） | 82 | Derive the bbox-based camera that is CONSISTENT with SMPLest-X's mesh_cam. | 0 |
| /home/shangguanrz/project/pic-edit/experiments/16-test/mk_prompts.py | 16 组横评可复现入口 | scripts/benchmarks/16_test/（路径及结果配置解耦后） | 10 |  | 0 |
| /home/shangguanrz/project/pic-edit/experiments/16-test/prompts_16test.md | 16 组横评可复现入口 | scripts/benchmarks/16_test/（路径及结果配置解耦后） | 109 |  | 0 |
| /home/shangguanrz/project/pic-edit/experiments/16-test/render_ab_geometry.py | 16 组横评可复现入口 | scripts/benchmarks/16_test/（路径及结果配置解耦后） | 319 | Render V6 adapter geometry conditions from SMPLest-X fitted parameters. | 0 |
| /home/shangguanrz/project/pic-edit/experiments/16-test/run_baseline_16test.py | 16 组横评可复现入口 | scripts/benchmarks/16_test/（路径及结果配置解耦后） | 300 | 16-test stage 1: the 8 BASELINE runs (no adapter, no geometry conditions). | 3 |
| /home/shangguanrz/project/pic-edit/experiments/16-test/run_deepgen_adapter.py | 16 组横评可复现入口 | scripts/benchmarks/16_test/（路径及结果配置解耦后） | 179 | 16-test stage 2a: DeepGen + V6 adapter on the A/B photos. | 1 |
| /home/shangguanrz/project/pic-edit/experiments/16-test/run_qwen_adapter.py | 16 组横评可复现入口 | scripts/benchmarks/16_test/（路径及结果配置解耦后） | 136 | 16-test stage 2b: Qwen-Image-2.1 + Qwen21 body adapter on the A/B photos. | 1 |
| /home/shangguanrz/project/pic-edit/fill_tiktok_v3_missing_geometry.py | 数据、条件或实验收尾入口 | scripts/data/fill_tiktok_v3_missing_geometry.py | 171 | load_jsonl, write_json, main | 1 |
| /home/shangguanrz/project/pic-edit/finalize_tiktok_v3_visual_quality_v1.py | 数据、条件或实验收尾入口 | scripts/data/finalize_tiktok_v3_visual_quality_v1.py | 538 | as_float, as_bool, write_jsonl, main | 0 |
| /home/shangguanrz/project/pic-edit/model-qwen-edit/README.md | 旧 Qwen Edit 20B 资产与独立下载/启动入口 | 旧模型整组归档至 /data，保留兼容路径 | 144 |  | 2 |
| /home/shangguanrz/project/pic-edit/model-qwen-edit/_download.py | 旧 Qwen Edit 20B 资产与独立下载/启动入口 | 旧模型整组归档至 /data，保留兼容路径 | 46 | 全量下载 Qwen/Qwen-Image-Edit-2511 到本目录（带自动重试）。 | 1 |
| /home/shangguanrz/project/pic-edit/model-qwen-edit/_start.sh | 旧 Qwen Edit 20B 资产与独立下载/启动入口 | 旧模型整组归档至 /data，保留兼容路径 | 12 |  | 0 |
| /home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/deepgen_pipeline.py | 模型配套运行实现（保留，不能按权重缓存删除） | models/DeepGen-1.0-diffusers/deepgen_pipeline.py | 1426 | DeepGen Diffusers Pipeline - Standalone pipeline for DeepGen-1.0. | 10 |
| /home/shangguanrz/project/pic-edit/multi_stage_eval_v6.py | 评测入口 | scripts/eval/multi_stage_eval_v6.py | 401 | Multi-stage checkpoint inference and comparison visualization for UnifiedSMPLXAdapterV6. | 1 |
| /home/shangguanrz/project/pic-edit/plot_vram_requirement_dashboard.py | 报告或可视化生成 | scripts/reports/plot_vram_requirement_dashboard.py | 173 | 生成《DeepGen 1.0 姿态编辑实验显存需求全景看板》高清可视化图表 | 0 |
| /home/shangguanrz/project/pic-edit/pytest.ini | 项目说明、依赖或环境入口（保留） | pytest.ini | 3 |  | 0 |
| /home/shangguanrz/project/pic-edit/requirements_lock.txt | 项目说明、依赖或环境入口（保留） | requirements_lock.txt | 242 |  | 1 |
| /home/shangguanrz/project/pic-edit/requirements_qwen21.txt | 项目说明、依赖或环境入口（保留） | requirements_qwen21.txt | 16 |  | 1 |
| /home/shangguanrz/project/pic-edit/research/audit_checkpoint_headers.py | 审计、验证或诊断入口 | scripts/audit/audit_checkpoint_headers.py | 118 | Count stored tensor elements from public HF safetensors headers only. | 2 |
| /home/shangguanrz/project/pic-edit/research/deepgen_assessment/build_report.py | 报告或可视化生成 | scripts/reports/build_report.py | 249 | p, h, page, table, cite | 2 |
| /home/shangguanrz/project/pic-edit/research/deepgen_assessment/deepgen_pipeline_audited.py.txt | 研究资料或审计证据 | research/deepgen_assessment/deepgen_pipeline_audited.py.txt | 1394 |  | 0 |
| /home/shangguanrz/project/pic-edit/research/第三阶段图像编辑调研与显存预算.md | 研究资料或审计证据 | research/第三阶段图像编辑调研与显存预算.md | 258 |  | 0 |
| /home/shangguanrz/project/pic-edit/run_champ_single_overfit.py | 实验编排、运行或辅助工具 | scripts/run/run_champ_single_overfit.py | 434 | Overfit PoseConditionAdapter on one aligned CHAMP source/target pair. | 5 |
| /home/shangguanrz/project/pic-edit/run_deepgen_no_control_ablation.py | 实验编排、运行或辅助工具 | scripts/run/run_deepgen_no_control_ablation.py | 219 | Run a pure-DeepGen ablation with no Adapter or spatial-control injection. | 5 |
| /home/shangguanrz/project/pic-edit/run_demo_pose_edit.py | 实验编排、运行或辅助工具 | scripts/run/run_demo_pose_edit.py | 163 | make_comparison_grid | 2 |
| /home/shangguanrz/project/pic-edit/run_tiktok_sample_00001_smoke.py | 实验编排、运行或辅助工具 | scripts/run/run_tiktok_sample_00001_smoke.py | 231 | Smoke run runner for TikTok Sequence 00001 (Source 0014 -> Target 0074). | 1 |
| /home/shangguanrz/project/pic-edit/run_tiktok_v3_dwpose.py | 实验编排、运行或辅助工具 | scripts/run/run_tiktok_v3_dwpose.py | 479 | ensure_dir, load_rows, read_rgb, select_person, normalize_points | 0 |
| /home/shangguanrz/project/pic-edit/run_tiktok_v3_fill_missing_geometry_bg.sh | 实验编排、运行或辅助工具 | scripts/run/run_tiktok_v3_fill_missing_geometry_bg.sh | 24 |  | 0 |
| /home/shangguanrz/project/pic-edit/run_v6_comparative_3000.sh | 实验编排、运行或辅助工具 | scripts/run/run_v6_comparative_3000.sh | 80 |  | 0 |
| /home/shangguanrz/project/pic-edit/sample_tiktok_v3_dwpose_qc.py | 审计、验证或诊断入口 | scripts/audit/sample_tiktok_v3_dwpose_qc.py | 146 |  | 0 |
| /home/shangguanrz/project/pic-edit/sample_tiktok_v3_geometry_rejects.py | 审计、验证或诊断入口 | scripts/audit/sample_tiktok_v3_geometry_rejects.py | 262 |  | 0 |
| /home/shangguanrz/project/pic-edit/scripts/audit_qwen21_face_alignment.py | 审计、验证或诊断入口 | scripts/audit/audit_qwen21_face_alignment.py | 288 | CPU-only crop/alignment audit on existing single-case generated images. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/audit_qwen21_weight_compatibility.py | 审计、验证或诊断入口 | scripts/audit/audit_qwen21_weight_compatibility.py | 373 | CPU-only weight provenance, semantic mapping and in-memory module audit. | 5 |
| /home/shangguanrz/project/pic-edit/scripts/build_iper1024_comparison_panels.py | 报告或可视化生成 | scripts/reports/build_iper1024_comparison_panels.py | 89 | Make readable iPER comparison sheets with face crops from saved RGB outputs. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/build_qwen21_face_panels.py | 报告或可视化生成 | scripts/reports/build_qwen21_face_panels.py | 85 | Create paired full-image and face-detail panels for a fixed Qwen21 Face evaluation. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/build_tiktok_v6_conditions.py | 数据、条件或实验收尾入口 | scripts/data/build_tiktok_v6_conditions.py | 595 | Build DeepGen V6.3 Native Adapter conditions for the TikTok Dataset. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/check_qwen21_full_preservation.py | 审计、验证或诊断入口 | scripts/audit/check_qwen21_full_preservation.py | 126 | CPU acceptance for the unchanged V6 copy and its legacy architecture tests. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/eval_qwen21_face_repair.py | 评测入口 | scripts/eval/eval_qwen21_face_repair.py | 50 | Render and score one bounded Face repair checkpoint using the frozen inputs. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/evaluate_qwen21_weight_inheritance.py | 实验编排、运行或辅助工具 | scripts/run/evaluate_qwen21_weight_inheritance.py | 206 | CPU metrics for the fixed inheritance comparison; no image synthesis. | 2 |
| /home/shangguanrz/project/pic-edit/scripts/finalize_qwen21_face_repair.py | 数据、条件或实验收尾入口 | scripts/data/finalize_qwen21_face_repair.py | 135 | Consolidate saved Face-repair checks and the manually inspected single case. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/finalize_qwen21_face_single_sample.py | 数据、条件或实验收尾入口 | scripts/data/finalize_qwen21_face_single_sample.py | 124 | Collect bounded-run results and verify the explicit single-case criteria. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/inherit_qwen21_full_weights.py | 检查点迁移工具 | scripts/weights/inherit_qwen21_full_weights.py | 165 | Create a full-controller initialization checkpoint from an audited whitelist. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/inventory_qwen21_full_weights.py | 审计、验证或诊断入口 | scripts/audit/inventory_qwen21_full_weights.py | 151 | CPU-only, per-tensor provenance of the preserved V6/Qwen initial model. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/precompute_iper_face_features_v65.py | 数据、条件或实验收尾入口 | scripts/data/precompute_iper_face_features_v65.py | 577 | Precompute V6.5 iPER face identity, appearance, and geometry features. | 7 |
| /home/shangguanrz/project/pic-edit/scripts/precompute_iper_hand_features_v66.py | 数据、条件或实验收尾入口 | scripts/data/precompute_iper_hand_features_v66.py | 155 | Cache source-hand DINO patches from existing iPER V6 detail crops. | 2 |
| /home/shangguanrz/project/pic-edit/scripts/precompute_qwen21_face_features.py | 数据、条件或实验收尾入口 | scripts/data/precompute_qwen21_face_features.py | 217 | Cache source-only face identity/texture and target geometry for Qwen21 Face training. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/precompute_tiktok_face_features_v65.py | 数据、条件或实验收尾入口 | scripts/data/precompute_tiktok_face_features_v65.py | 608 | Precompute V6.5 TikTok face identity, appearance, and geometry features. | 4 |
| /home/shangguanrz/project/pic-edit/scripts/prepare_qwen21_face_split.py | 数据、条件或实验收尾入口 | scripts/data/prepare_qwen21_face_split.py | 149 | Fix the face-training appearance split and extract source-only face crops. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/prepare_qwen21_full_data.py | 数据、条件或实验收尾入口 | scripts/data/prepare_qwen21_full_data.py | 266 | Prepare a fixed, small real-data acceptance set for the full Qwen adapter. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/prepare_tiktok_qwen21_training.py | 数据、条件或实验收尾入口 | scripts/data/prepare_tiktok_qwen21_training.py | 158 | Audit strict TikTok splits and write separate eligible training manifests. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/probe_qwen21_face_single_sample.py | 审计、验证或诊断入口 | scripts/audit/probe_qwen21_face_single_sample.py | 50 | Post-training fixed-noise, regional-error, and one-step reconstruction diagnosis. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/run_qwen21_face_single_sample.py | 实验编排、运行或辅助工具 | scripts/run/run_qwen21_face_single_sample.py | 207 | Bounded single-pair diagnosis. Every GPU action uses the complete controller. | 5 |
| /home/shangguanrz/project/pic-edit/scripts/run_qwen21_weight_inheritance.py | 检查点迁移工具 | scripts/weights/run_qwen21_weight_inheritance.py | 261 | Verify and generate the fixed 60-image inheritance comparison, without training. | 4 |
| /home/shangguanrz/project/pic-edit/scripts/smoke_adapter_reasoner.py | 实验编排、运行或辅助工具 | scripts/run/smoke_adapter_reasoner.py | 221 | GPU/CPU structural smoke for the standalone V6.2 Adapter reasoner. | 2 |
| /home/shangguanrz/project/pic-edit/scripts/smoke_adapter_v6.py | 实验编排、运行或辅助工具 | scripts/run/smoke_adapter_v6.py | 254 | Real DeepGen structural smoke test for Unified SMPL-X Adapter V6. | 2 |
| /home/shangguanrz/project/pic-edit/scripts/smoke_deepgen_control_interface.py | 实验编排、运行或辅助工具 | scripts/run/smoke_deepgen_control_interface.py | 200 | Real DeepGen smoke for the V6.3 dynamic control interface. | 2 |
| /home/shangguanrz/project/pic-edit/scripts/split_iper_train_val.py | 数据、条件或实验收尾入口 | scripts/data/split_iper_train_val.py | 95 | Split official iPER 79 training appearances into dev-train (69) and dev-val (10). | 1 |
| /home/shangguanrz/project/pic-edit/scripts/summarize_qwen21_face_groups.py | 实验编排、运行或辅助工具 | scripts/run/summarize_qwen21_face_groups.py | 136 | Paired Face variant comparisons on common visible faces and pose groups. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/verify_qwen21_face_repair.py | 审计、验证或诊断入口 | scripts/audit/verify_qwen21_face_repair.py | 51 | Check repaired Face candidates without updating weights or generating images. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/verify_qwen21_face_single_sample.py | 审计、验证或诊断入口 | scripts/audit/verify_qwen21_face_single_sample.py | 76 | Real Face loss gradient and source/geometry boundaries; no optimizer update. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/verify_qwen21_full_resume.py | 审计、验证或诊断入口 | scripts/audit/verify_qwen21_full_resume.py | 256 | Diagnose one-step CUDA checkpoint reproducibility without extending training. | 2 |
| /home/shangguanrz/project/pic-edit/scripts/verify_qwen21_full_runtime.py | 审计、验证或诊断入口 | scripts/audit/verify_qwen21_full_runtime.py | 406 | Bounded GPU acceptance for the full Qwen controller; never updates weights. | 2 |
| /home/shangguanrz/project/pic-edit/scripts/write_iper1024_result.py | 报告或可视化生成 | scripts/reports/write_iper1024_result.py | 118 | Summarize paired 512/1024 Qwen21 iPER results without mixing pixel scales. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/write_qwen21_face_repair_report.py | 报告或可视化生成 | scripts/reports/write_qwen21_face_repair_report.py | 383 | Write the bounded Face repair report from saved evidence; no model execution. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/write_qwen21_face_report.py | 报告或可视化生成 | scripts/reports/write_qwen21_face_report.py | 171 | Write a compact, image-rich Word report from fixed Face validation/test outputs. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/write_qwen21_face_single_sample_report.py | 报告或可视化生成 | scripts/reports/write_qwen21_face_single_sample_report.py | 164 | Word report with embedded full-pixel panels and explicit training-data labels. | 1 |
| /home/shangguanrz/project/pic-edit/scripts/write_qwen21_full_report.py | 报告或可视化生成 | scripts/reports/write_qwen21_full_report.py | 559 | Build a Word acceptance report and full-resolution two-row comparison PNGs. | 3 |
| /home/shangguanrz/project/pic-edit/scripts/write_qwen21_inheritance_report.py | 报告或可视化生成 | scripts/reports/write_qwen21_inheritance_report.py | 239 | Embed measured tables and original-pixel comparison panels in a Word report. | 0 |
| /home/shangguanrz/project/pic-edit/scripts/write_qwen21_weight_audit_report.py | 报告或可视化生成 | scripts/reports/write_qwen21_weight_audit_report.py | 202 | Add provenance/semantic evidence and write the standalone Word audit report. | 0 |
| /home/shangguanrz/project/pic-edit/split_tiktok_v3_groupwise.py | 数据、条件或实验收尾入口 | scripts/data/split_tiktok_v3_groupwise.py | 526 | read_jsonl, evaluate | 0 |
| /home/shangguanrz/project/pic-edit/src/__init__.py | 核心源码（保留） | src/__init__.py | 0 |  | 105 |
| /home/shangguanrz/project/pic-edit/src/adapter/__init__.py | 核心源码（保留） | src/adapter/__init__.py | 0 |  | 105 |
| /home/shangguanrz/project/pic-edit/src/adapter/pose_adapter.py | 核心源码（保留） | src/adapter/pose_adapter.py | 165 | ResnetBlock2D, Downsample2D, PoseConditionAdapter | 4 |
| /home/shangguanrz/project/pic-edit/src/controls/__init__.py | 核心源码（保留） | src/controls/__init__.py | 0 |  | 105 |
| /home/shangguanrz/project/pic-edit/src/controls/generator.py | 核心源码（保留） | src/controls/generator.py | 344 | Camera, ControlSignalGenerator | 44 |
| /home/shangguanrz/project/pic-edit/src/data/__init__.py | 核心源码（保留） | src/data/__init__.py | 21 | Data modules for picture edit. | 105 |
| /home/shangguanrz/project/pic-edit/src/data/hi4d_qwen21_dataset.py | 核心源码（保留） | src/data/hi4d_qwen21_dataset.py | 283 | Hi4D N2C controls for the Qwen21 two-person adapter. | 7 |
| /home/shangguanrz/project/pic-edit/src/data/iper_dataset.py | 核心源码（保留） | src/data/iper_dataset.py | 297 | IPERPoseDataset | 22 |
| /home/shangguanrz/project/pic-edit/src/data/tiktok_dataset.py | 核心源码（保留） | src/data/tiktok_dataset.py | 871 | TikTok Dataset loader and single-pair overfit loader for DeepGen V6. | 11 |
| /home/shangguanrz/project/pic-edit/src/evaluation/dwpose_onnx/__init__.py | 核心源码（保留） | src/evaluation/dwpose_onnx/__init__.py | 5 | DWPose ONNX inference helpers from IDEA-Research/DWPose (Apache-2.0). | 105 |
| /home/shangguanrz/project/pic-edit/src/evaluation/dwpose_onnx/onnxdet.py | 核心源码（保留） | src/evaluation/dwpose_onnx/onnxdet.py | 125 | nms, multiclass_nms, demo_postprocess, preprocess, inference_detector | 4 |
| /home/shangguanrz/project/pic-edit/src/evaluation/dwpose_onnx/onnxpose.py | 核心源码（保留） | src/evaluation/dwpose_onnx/onnxpose.py | 360 | preprocess, inference, postprocess, bbox_xyxy2cs, _fix_aspect_ratio | 4 |
| /home/shangguanrz/project/pic-edit/src/integration/__init__.py | 核心源码（保留） | src/integration/__init__.py | 19 | Integration helpers for external pose-control datasets and DeepGen. | 105 |
| /home/shangguanrz/project/pic-edit/src/integration/champ_deepgen.py | 核心源码（保留） | src/integration/champ_deepgen.py | 206 | CHAMP sample validation and DeepGen pose-residual alignment. | 7 |
| /home/shangguanrz/project/pic-edit/src/pose_control/__init__.py | 核心源码（保留） | src/pose_control/__init__.py | 1 | Pose-conditioned controls for DeepGen. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21/__init__.py | 核心源码（保留） | src/pose_control/qwen21/__init__.py | 6 | Qwen-Image-2.1 target-token controls for iPER Body and Face editing. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21/action_prompts.py | 核心源码（保留） | src/pose_control/qwen21/action_prompts.py | 76 | Deterministic action text for editing two existing people in a source photo. | 4 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21/body_adapter.py | 核心源码（保留） | src/pose_control/qwen21/body_adapter.py | 183 | Qwen21BodyAdapter, Qwen21ControlContext | 18 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21/dual_adapter.py | 核心源码（保留） | src/pose_control/qwen21/dual_adapter.py | 186 | Two-person geometry, contact and source-appearance control for Qwen-Image-2.1. | 5 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21/dual_inference.py | 核心源码（保留） | src/pose_control/qwen21/dual_inference.py | 97 | Target-RGB-free generation boundary for Hi4D Qwen21 dual editing. | 3 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21/face_adapter.py | 核心源码（保留） | src/pose_control/qwen21/face_adapter.py | 101 | Source-identity and target-ROI face residual branch for Qwen-Image-2.1. | 23 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21/runtime.py | 核心源码（保留） | src/pose_control/qwen21/runtime.py | 152 | Pinned Qwen-Image-2.1 image/latent contract used by iPER training and evaluation. | 38 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/__init__.py | 核心源码（保留） | src/pose_control/qwen21_full/__init__.py | 81 | Unified single/dual-person SMPL-X Adapter V6. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/checkpoint.py | 核心源码（保留） | src/pose_control/qwen21_full/checkpoint.py | 399 | _has_prefix, _adapter_state, _preflight_state, _transactional_load, load_v63_checkpoint | 77 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/condition_bridge.py | 核心源码（保留） | src/pose_control/qwen21_full/condition_bridge.py | 56 | ReasonerBridgedCondition, ReasonerControlBridge | 19 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/condition_injector.py | 核心源码（保留） | src/pose_control/qwen21_full/condition_injector.py | 305 | SMPLXConditionInjector | 43 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/conditions.py | 核心源码（保留） | src/pose_control/qwen21_full/conditions.py | 294 | _to_preserving_discrete_dtype, TaskType, ContactRelationBatch, ConditionBundle, AdapterIdentityCondition | 114 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/control_core.py | 核心源码（保留） | src/pose_control/qwen21_full/control_core.py | 512 | BranchControlResiduals, CrossNorm, LowRankStageAdapter, _TinyPatchEmbed, _TinyTimeTextEmbed | 26 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/controlled_pipeline.py | 核心源码（保留） | src/pose_control/qwen21_full/controlled_pipeline.py | 231 | ControlledDeepGenPipeline | 9 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/data_runtime.py | 核心源码（保留） | src/pose_control/qwen21_full/data_runtime.py | 183 | Source-only identity and target-only geometry for the complete Qwen adapter. | 13 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/deepgen_adapter.py | 核心源码（保留） | src/pose_control/qwen21_full/deepgen_adapter.py | 612 | UnifiedSMPLXAdapterV6 | 28 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/detail/__init__.py | 核心源码（保留） | src/pose_control/qwen21_full/detail/__init__.py | 48 | Face/hand detail conditioning contracts and local trainable encoders. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/detail/binder.py | 核心源码（保留） | src/pose_control/qwen21_full/detail/binder.py | 47 | DetailTokenBinder | 6 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/detail/conditions.py | 核心源码（保留） | src/pose_control/qwen21_full/detail/conditions.py | 254 | DetailRegion, _selected, _moved, FaceHandDetailCondition, DetailReferenceBatch | 114 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/detail/encoders.py | 核心源码（保留） | src/pose_control/qwen21_full/detail/encoders.py | 203 | ReferenceFeatureProvider, canonicalize_left_hand_keypoints, restore_left_hand_keypoints, canonicalize_left_hand_pose, restore_left_hand_pose | 11 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/detail/hand.py | 核心源码（保留） | src/pose_control/qwen21_full/detail/hand.py | 216 | _map_fields, HandDetailCondition, legacy_to_face_and_hand, hand_to_legacy_detail, references_to_hand_only | 75 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/detail/preparer.py | 核心源码（保留） | src/pose_control/qwen21_full/detail/preparer.py | 211 | soft_box_masks, FaceHandDetailPreparer | 12 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/execution.py | 核心源码（保留） | src/pose_control/qwen21_full/execution.py | 398 | Training utilities for the complete V6 controller connected to Qwen21. | 22 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/experts.py | 核心源码（保留） | src/pose_control/qwen21_full/experts.py | 28 | RoutingDecision, TaskRouter | 3 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face/__init__.py | 核心源码（保留） | src/pose_control/qwen21_full/face/__init__.py | 57 | V6.5 face-conditioning contracts and encoders. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face/adapter.py | 核心源码（保留） | src/pose_control/qwen21_full/face/adapter.py | 273 | _as_batch_scalar, scatter_face_roi_residuals, _LowRankStageAdapter, FaceAttentionBlock, FaceControlAdapter | 100 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face/conditions.py | 核心源码（保留） | src/pose_control/qwen21_full/face/conditions.py | 290 | _index_selected, _moved, _expanded, _validate_same_float_dtype_device, _validate_normalized_boxes | 114 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face/content_encoder.py | 核心源码（保留） | src/pose_control/qwen21_full/face/content_encoder.py | 139 | pool_arcface_references, FaceContentEncoder | 7 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face/geometry.py | 核心源码（保留） | src/pose_control/qwen21_full/face/geometry.py | 180 | _validate_keypoint_tail, smplx137_to_face72, dwpose68_to_face72, normalize_face_landmarks_to_roi, FaceGeometryTokenizer | 125 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face/iper_dataset.py | 核心源码（保留） | src/pose_control/qwen21_full/face/iper_dataset.py | 264 | Self-contained iPER single-pair loader for V6.5 Face overfitting. | 22 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face/preparer.py | 核心源码（保留） | src/pose_control/qwen21_full/face/preparer.py | 140 | FaceConditioningPreparer | 12 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face/resampler.py | 核心源码（保留） | src/pose_control/qwen21_full/face/resampler.py | 9 | Runtime exports for the pinned Visual Persona resampler adaptation. | 15 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face/spatial.py | 核心源码（保留） | src/pose_control/qwen21_full/face/spatial.py | 122 | render_face_landmark_heatmaps, FaceSpatialEncoder | 31 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face_loss_alignment.py | 核心源码（保留） | src/pose_control/qwen21_full/face_loss_alignment.py | 149 | Differentiable ArcFace alignment for offline face losses. | 4 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/face_supervision.py | 核心源码（保留） | src/pose_control/qwen21_full/face_supervision.py | 212 | Target-RGB face eligibility exclusively for offline auxiliary training losses. | 5 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/geometry_encoder.py | 核心源码（保留） | src/pose_control/qwen21_full/geometry_encoder.py | 236 | _group_count, ConditionResBlock, NativeConditionStem, RoleFiLM, ChampConditionStemAdapter | 5 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/hand/__init__.py | 核心源码（保留） | src/pose_control/qwen21_full/hand/__init__.py | 7 | V6.6 independent, ROI-local hand control. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/hand/adapter.py | 核心源码（保留） | src/pose_control/qwen21_full/hand/adapter.py | 146 | scatter_hand_roi_residuals, IndependentKVAttention, HandFusionBlock, _StageAdapter, HandControlAdapter | 101 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/hand/conditions.py | 核心源码（保留） | src/pose_control/qwen21_full/hand/conditions.py | 134 | _transform, HandFineCondition, HandReferenceFeatures, PreparedHandConditioning | 114 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/hand/geometry.py | 核心源码（保留） | src/pose_control/qwen21_full/hand/geometry.py | 48 | split_iper_134_hand_points, axis_angle45_to_6d96, local_hand_points, hand_heatmaps | 125 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/hand/preparer.py | 核心源码（保留） | src/pose_control/qwen21_full/hand/preparer.py | 87 | HandConditioningPreparer | 12 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/hand/quality.py | 核心源码（保留） | src/pose_control/qwen21_full/hand/quality.py | 22 | Offline-only SMPL-X hand quality gate and conservative refinement merge. | 65 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/hand/refiner_export.py | 核心源码（保留） | src/pose_control/qwen21_full/hand/refiner_export.py | 24 | Optional HandRefiner interchange only; never executed in default generation. | 1 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/hand/vton_layers.py | 核心源码（保留） | src/pose_control/qwen21_full/hand/vton_layers.py | 99 | Shape-adapted VTON-HandFit Handprocessor and Perceiver Resampler primitives. | 3 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/interface/__init__.py | 核心源码（保留） | src/pose_control/qwen21_full/interface/__init__.py | 19 | Dynamic Adapter-to-DeepGen control interface. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/interface/deepgen_interface.py | 核心源码（保留） | src/pose_control/qwen21_full/interface/deepgen_interface.py | 432 | latent_token_count, align_target_residuals, DeepGenControlInterface | 6 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/interface/outputs.py | 核心源码（保留） | src/pose_control/qwen21_full/interface/outputs.py | 145 | PreparedControlConditioning, DeepGenControlOutput | 172 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/interface/strength.py | 核心源码（保留） | src/pose_control/qwen21_full/interface/strength.py | 267 | StrengthScheduleConfig, ControlStrengthController | 33 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/qwen_adapter.py | 核心源码（保留） | src/pose_control/qwen21_full/qwen_adapter.py | 518 | Qwen boundary adapters around the unchanged, complete V6 control architecture. | 14 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/qwen_runtime.py | 核心源码（保留） | src/pose_control/qwen21_full/qwen_runtime.py | 154 | Native Qwen-Image-2.1 image/latent I/O for the full V6 migration. | 5 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/reasoning/__init__.py | 核心源码（保留） | src/pose_control/qwen21_full/reasoning/__init__.py | 22 | Adapter-internal single/dual-person reasoning for DeepGen V6.2. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/reasoning/blocks.py | 核心源码（保留） | src/pose_control/qwen21_full/reasoning/blocks.py | 80 | group_count, ReasoningResBlock, CrossAttentionFFN | 23 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/reasoning/config.py | 核心源码（保留） | src/pose_control/qwen21_full/reasoning/config.py | 49 | AdapterReasoningConfig | 71 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/reasoning/contact.py | 核心源码（保留） | src/pose_control/qwen21_full/reasoning/contact.py | 144 | ContactReasoningOutput, ContactConditionReasoner | 47 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/reasoning/cross_person.py | 核心源码（保留） | src/pose_control/qwen21_full/reasoning/cross_person.py | 84 | BidirectionalCrossPersonReasoner | 14 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/reasoning/fusion.py | 核心源码（保留） | src/pose_control/qwen21_full/reasoning/fusion.py | 108 | DualFusionOutput, DualFeatureFusion | 21 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/reasoning/person_geometry.py | 核心源码（保留） | src/pose_control/qwen21_full/reasoning/person_geometry.py | 76 | PersonReasoningOutput, PersonGeometryReasoner | 5 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/reasoning/reasoner.py | 核心源码（保留） | src/pose_control/qwen21_full/reasoning/reasoner.py | 196 | SMPLXAdapterReasoner | 23 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/reasoning/state.py | 核心源码（保留） | src/pose_control/qwen21_full/reasoning/state.py | 86 | InternalControlState | 53 |
| /home/shangguanrz/project/pic-edit/src/pose_control/qwen21_full/token_encoders.py | 核心源码（保留） | src/pose_control/qwen21_full/token_encoders.py | 233 | AppearanceTokenEncoder, PersonTokenBinder, GlobalConditionTokenEncoder, RelativeGeometryTokenEncoder, ContactRasterEncoder | 10 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/__init__.py | 核心源码（保留） | src/pose_control/v6/__init__.py | 81 | Unified single/dual-person SMPL-X Adapter V6. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/checkpoint.py | 核心源码（保留） | src/pose_control/v6/checkpoint.py | 399 | _has_prefix, _adapter_state, _preflight_state, _transactional_load, load_v63_checkpoint | 77 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/condition_bridge.py | 核心源码（保留） | src/pose_control/v6/condition_bridge.py | 56 | ReasonerBridgedCondition, ReasonerControlBridge | 19 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/condition_injector.py | 核心源码（保留） | src/pose_control/v6/condition_injector.py | 305 | SMPLXConditionInjector | 43 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/conditions.py | 核心源码（保留） | src/pose_control/v6/conditions.py | 294 | _to_preserving_discrete_dtype, TaskType, ContactRelationBatch, ConditionBundle, AdapterIdentityCondition | 114 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/control_core.py | 核心源码（保留） | src/pose_control/v6/control_core.py | 512 | BranchControlResiduals, CrossNorm, LowRankStageAdapter, _TinyPatchEmbed, _TinyTimeTextEmbed | 26 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/controlled_pipeline.py | 核心源码（保留） | src/pose_control/v6/controlled_pipeline.py | 231 | ControlledDeepGenPipeline | 9 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/deepgen_adapter.py | 核心源码（保留） | src/pose_control/v6/deepgen_adapter.py | 612 | UnifiedSMPLXAdapterV6 | 28 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/detail/__init__.py | 核心源码（保留） | src/pose_control/v6/detail/__init__.py | 48 | Face/hand detail conditioning contracts and local trainable encoders. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/detail/binder.py | 核心源码（保留） | src/pose_control/v6/detail/binder.py | 47 | DetailTokenBinder | 6 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/detail/conditions.py | 核心源码（保留） | src/pose_control/v6/detail/conditions.py | 254 | DetailRegion, _selected, _moved, FaceHandDetailCondition, DetailReferenceBatch | 114 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/detail/encoders.py | 核心源码（保留） | src/pose_control/v6/detail/encoders.py | 203 | ReferenceFeatureProvider, canonicalize_left_hand_keypoints, restore_left_hand_keypoints, canonicalize_left_hand_pose, restore_left_hand_pose | 11 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/detail/hand.py | 核心源码（保留） | src/pose_control/v6/detail/hand.py | 216 | _map_fields, HandDetailCondition, legacy_to_face_and_hand, hand_to_legacy_detail, references_to_hand_only | 75 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/detail/preparer.py | 核心源码（保留） | src/pose_control/v6/detail/preparer.py | 211 | soft_box_masks, FaceHandDetailPreparer | 12 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/experts.py | 核心源码（保留） | src/pose_control/v6/experts.py | 28 | RoutingDecision, TaskRouter | 3 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/face/__init__.py | 核心源码（保留） | src/pose_control/v6/face/__init__.py | 57 | V6.5 face-conditioning contracts and encoders. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/face/adapter.py | 核心源码（保留） | src/pose_control/v6/face/adapter.py | 273 | _as_batch_scalar, scatter_face_roi_residuals, _LowRankStageAdapter, FaceAttentionBlock, FaceControlAdapter | 100 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/face/conditions.py | 核心源码（保留） | src/pose_control/v6/face/conditions.py | 290 | _index_selected, _moved, _expanded, _validate_same_float_dtype_device, _validate_normalized_boxes | 114 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/face/content_encoder.py | 核心源码（保留） | src/pose_control/v6/face/content_encoder.py | 139 | pool_arcface_references, FaceContentEncoder | 7 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/face/geometry.py | 核心源码（保留） | src/pose_control/v6/face/geometry.py | 180 | _validate_keypoint_tail, smplx137_to_face72, dwpose68_to_face72, normalize_face_landmarks_to_roi, FaceGeometryTokenizer | 125 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/face/iper_dataset.py | 核心源码（保留） | src/pose_control/v6/face/iper_dataset.py | 264 | Self-contained iPER single-pair loader for V6.5 Face overfitting. | 22 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/face/preparer.py | 核心源码（保留） | src/pose_control/v6/face/preparer.py | 140 | FaceConditioningPreparer | 12 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/face/resampler.py | 核心源码（保留） | src/pose_control/v6/face/resampler.py | 9 | Runtime exports for the pinned Visual Persona resampler adaptation. | 15 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/face/spatial.py | 核心源码（保留） | src/pose_control/v6/face/spatial.py | 122 | render_face_landmark_heatmaps, FaceSpatialEncoder | 31 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/geometry_encoder.py | 核心源码（保留） | src/pose_control/v6/geometry_encoder.py | 236 | _group_count, ConditionResBlock, NativeConditionStem, RoleFiLM, ChampConditionStemAdapter | 5 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/hand/__init__.py | 核心源码（保留） | src/pose_control/v6/hand/__init__.py | 7 | V6.6 independent, ROI-local hand control. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/hand/adapter.py | 核心源码（保留） | src/pose_control/v6/hand/adapter.py | 146 | scatter_hand_roi_residuals, IndependentKVAttention, HandFusionBlock, _StageAdapter, HandControlAdapter | 101 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/hand/conditions.py | 核心源码（保留） | src/pose_control/v6/hand/conditions.py | 134 | _transform, HandFineCondition, HandReferenceFeatures, PreparedHandConditioning | 114 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/hand/geometry.py | 核心源码（保留） | src/pose_control/v6/hand/geometry.py | 48 | split_iper_134_hand_points, axis_angle45_to_6d96, local_hand_points, hand_heatmaps | 125 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/hand/preparer.py | 核心源码（保留） | src/pose_control/v6/hand/preparer.py | 87 | HandConditioningPreparer | 12 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/hand/quality.py | 核心源码（保留） | src/pose_control/v6/hand/quality.py | 22 | Offline-only SMPL-X hand quality gate and conservative refinement merge. | 65 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/hand/refiner_export.py | 核心源码（保留） | src/pose_control/v6/hand/refiner_export.py | 24 | Optional HandRefiner interchange only; never executed in default generation. | 1 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/hand/vton_layers.py | 核心源码（保留） | src/pose_control/v6/hand/vton_layers.py | 99 | Shape-adapted VTON-HandFit Handprocessor and Perceiver Resampler primitives. | 3 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/interface/__init__.py | 核心源码（保留） | src/pose_control/v6/interface/__init__.py | 19 | Dynamic Adapter-to-DeepGen control interface. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/interface/deepgen_interface.py | 核心源码（保留） | src/pose_control/v6/interface/deepgen_interface.py | 432 | latent_token_count, align_target_residuals, DeepGenControlInterface | 6 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/interface/outputs.py | 核心源码（保留） | src/pose_control/v6/interface/outputs.py | 145 | PreparedControlConditioning, DeepGenControlOutput | 172 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/interface/strength.py | 核心源码（保留） | src/pose_control/v6/interface/strength.py | 267 | StrengthScheduleConfig, ControlStrengthController | 33 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/reasoning/__init__.py | 核心源码（保留） | src/pose_control/v6/reasoning/__init__.py | 22 | Adapter-internal single/dual-person reasoning for DeepGen V6.2. | 105 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/reasoning/blocks.py | 核心源码（保留） | src/pose_control/v6/reasoning/blocks.py | 80 | group_count, ReasoningResBlock, CrossAttentionFFN | 23 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/reasoning/config.py | 核心源码（保留） | src/pose_control/v6/reasoning/config.py | 49 | AdapterReasoningConfig | 71 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/reasoning/contact.py | 核心源码（保留） | src/pose_control/v6/reasoning/contact.py | 144 | ContactReasoningOutput, ContactConditionReasoner | 47 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/reasoning/cross_person.py | 核心源码（保留） | src/pose_control/v6/reasoning/cross_person.py | 84 | BidirectionalCrossPersonReasoner | 14 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/reasoning/fusion.py | 核心源码（保留） | src/pose_control/v6/reasoning/fusion.py | 108 | DualFusionOutput, DualFeatureFusion | 21 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/reasoning/person_geometry.py | 核心源码（保留） | src/pose_control/v6/reasoning/person_geometry.py | 76 | PersonReasoningOutput, PersonGeometryReasoner | 5 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/reasoning/reasoner.py | 核心源码（保留） | src/pose_control/v6/reasoning/reasoner.py | 196 | SMPLXAdapterReasoner | 23 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/reasoning/state.py | 核心源码（保留） | src/pose_control/v6/reasoning/state.py | 86 | InternalControlState | 53 |
| /home/shangguanrz/project/pic-edit/src/pose_control/v6/token_encoders.py | 核心源码（保留） | src/pose_control/v6/token_encoders.py | 233 | AppearanceTokenEncoder, PersonTokenBinder, GlobalConditionTokenEncoder, RelativeGeometryTokenEncoder, ContactRasterEncoder | 10 |
| /home/shangguanrz/project/pic-edit/system_environment_lock.md | 项目说明、依赖或环境入口（保留） | system_environment_lock.md | 13 |  | 0 |
| /home/shangguanrz/project/pic-edit/test_control_signals.py | 生产依赖：合成控制生成工具（保留） | test_control_signals.py（以后提取到 src/controls/） | 192 | create_synthetic_smplx_human, test_control_signal_generation | 3 |
| /home/shangguanrz/project/pic-edit/tests/test_adapter_reasoning_v1.py | 回归测试（保留） | tests/test_adapter_reasoning_v1.py | 344 | _api, _make_bundle, _build_reasoner, test_reasoning_public_contract_is_available, test_single_dual_and_mixed_outputs_have_stable_shapes | 3 |
| /home/shangguanrz/project/pic-edit/tests/test_adapter_v6_architecture.py | 回归测试（保留） | tests/test_adapter_v6_architecture.py | 385 | test_smoke_report_names_preserve_backend_results, build_tiny, make_raw_conditions, make_runtime, test_adapter_public_boundary_consumes_bundle_and_separate_identity | 2 |
| /home/shangguanrz/project/pic-edit/tests/test_analysis.py | 回归测试（保留） | tests/test_analysis.py | 39 | AcceptanceTests | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_champ_deepgen_integration.py | 回归测试（保留） | tests/test_champ_deepgen_integration.py | 216 | _write_rgb, _write_gray, _write_natural_rgb, ChampSampleTests, ResidualAlignmentTests | 2 |
| /home/shangguanrz/project/pic-edit/tests/test_condition_bridge_contract.py | 回归测试（保留） | tests/test_condition_bridge_contract.py | 117 | _make_state, test_reasoner_bridge_projects_independent_geometry_and_interaction_scenes, test_interaction_projection_has_no_bias_and_single_rows_ignore_interaction, test_bridge_consumes_both_high_resolution_control_paths, test_bridge_keeps_all_four_reasoner_paths_on_the_gradient_graph | 1 |
| /home/shangguanrz/project/pic-edit/tests/test_condition_injector_v1.py | 回归测试（保留） | tests/test_condition_injector_v1.py | 332 | _api, _person, _relations, test_condition_injector_module_is_available, test_single_person_bundle_has_stable_shapes | 3 |
| /home/shangguanrz/project/pic-edit/tests/test_deepgen_control_interface_v1.py | 回归测试（保留） | tests/test_deepgen_control_interface_v1.py | 478 | _branch_residuals, test_branch_control_residuals_validate_dynamic_token_shape, test_strength_schedule_curves, test_strength_schedule_is_zero_outside_its_window, test_strength_controller_scales_branches_and_groups_independently | 6 |
| /home/shangguanrz/project/pic-edit/tests/test_deepgen_no_control_ablation.py | 回归测试（保留） | tests/test_deepgen_no_control_ablation.py | 32 | NoControlRunnerContractTests | 1 |
| /home/shangguanrz/project/pic-edit/tests/test_deepgen_pose_patch.py | 回归测试（保留） | tests/test_deepgen_pose_patch.py | 33 | DeepGenPosePatchTests | 1 |
| /home/shangguanrz/project/pic-edit/tests/test_hi4d_geometry_registration.py | 回归测试（保留） | tests/test_hi4d_geometry_registration.py | 25 | test_vertical_orientation_uses_geometry_semantic_area, test_orientation_keeps_aligned_raster_unchanged | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_iper_v6_data_integrity.py | 回归测试（保留） | tests/test_iper_v6_data_integrity.py | 100 | dataset_paths, test_v6_dataset_loading_and_injector_pass | 1 |
| /home/shangguanrz/project/pic-edit/tests/test_iper_v6_eval_parity.py | 回归测试（保留） | tests/test_iper_v6_eval_parity.py | 67 | Tests for iPER V6.3 evaluation parity and metric comparability against 130.28M baseline. | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_iper_v6_training_pipeline.py | 回归测试（保留） | tests/test_iper_v6_training_pipeline.py | 429 | Unit tests for iPER V6.3 training pipeline under Scheme A. | 1 |
| /home/shangguanrz/project/pic-edit/tests/test_lab.py | 回归测试（保留） | tests/test_lab.py | 32 | ExperimentContractTests | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_action_prompts.py | 回归测试（保留） | tests/test_qwen21_action_prompts.py | 166 | Action text and source cache contract for Qwen21 dual-person editing. | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_control_contract.py | 回归测试（保留） | tests/test_qwen21_control_contract.py | 92 | The control hook may change target tokens but must preserve prefix tokens. | 2 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_dual_control.py | 回归测试（保留） | tests/test_qwen21_dual_control.py | 81 | Dual control must bind both source people and leave native prefix intact. | 1 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_dual_inference_boundary.py | 回归测试（保留） | tests/test_qwen21_dual_inference_boundary.py | 59 | Target pixels and evaluation-only masks must not enter generation inputs. | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_face_control.py | 回归测试（保留） | tests/test_qwen21_face_control.py | 92 | Face branch preserves original Body behavior and edits only target face tokens. | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_face_loss_alignment.py | 回归测试（保留） | tests/test_qwen21_face_loss_alignment.py | 125 | limited_threads, five_points, textured_image, test_alignment_matches_opencv_geometry_with_interpolation_tolerance, test_fixed_grid_shared_for_prediction_target_and_gradients | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_face_objective.py | 回归测试（保留） | tests/test_qwen21_face_objective.py | 76 | Behavioral checks for a Face objective that cannot hide damage in the canvas. | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_face_reference_grid.py | 回归测试（保留） | tests/test_qwen21_face_reference_grid.py | 151 | CPU checks for the reversible trained-grid Face scatter boundary. | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_full_bridge.py | 回归测试（保留） | tests/test_qwen21_full_bridge.py | 272 | limited_threads, make_model, make_prepared, runtime, test_lossless_initial_latent_bridge_and_grid | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_full_preservation.py | 回归测试（保留） | tests/test_qwen21_full_preservation.py | 225 | Audit the copied control architecture independently of the Qwen bridge. | 1 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_weight_audit.py | 回归测试（保留） | tests/test_qwen21_weight_audit.py | 33 | test_explicit_rows_preserve_unselected_state, test_invalid_mapping_fails_before_any_mutation | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_qwen21_weight_inheritance.py | 回归测试（保留） | tests/test_qwen21_weight_inheritance.py | 29 | fixture, test_saved_state_retains_unselected_rows_and_parameters, test_saved_state_detects_corruption | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_report.py | 回归测试（保留） | tests/test_report.py | 212 | ReportTests | 1 |
| /home/shangguanrz/project/pic-edit/tests/test_tiktok_data_contract.py | 回归测试（保留） | tests/test_tiktok_data_contract.py | 638 | test_tiktok_condition_cli_help, test_tiktok_face_precompute_cli_help, test_axis_angle_to_rotation_6d, test_extract_25_keypoints_from_smplx_joints, test_body25_projection_uses_named_smplestx_joints_not_mesh_vertices | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_tiktok_qwen21_face_cache.py | 回归测试（保留） | tests/test_tiktok_qwen21_face_cache.py | 21 | test_target_only_frames_keep_zero_identity_and_texture | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_tiktok_qwen21_full_chain.py | 回归测试（保留） | tests/test_tiktok_qwen21_full_chain.py | 81 | Real dataset boundaries for TikTok + the preserved full Qwen controller. | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_tiktok_qwen21_readiness.py | 回归测试（保留） | tests/test_tiktok_qwen21_readiness.py | 34 | test_preflight_rejects_source_video_leakage, test_preflight_rejects_missing_group_provenance, test_preflight_does_not_accept_zero_keypoints, test_preflight_rejects_source_equals_target | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_train_tiktok_v65_face_smoke.py | 回归测试（保留） | tests/test_train_tiktok_v65_face_smoke.py | 562 | Smoke tests for the TikTok V6.5 Face Adapter training pipeline. | 0 |
| /home/shangguanrz/project/pic-edit/tests/test_v65_face_foundation.py | 回归测试（保留） | tests/test_v65_face_foundation.py | 409 | make_condition, make_references, make_prepared, test_face_condition_contract_preserves_discrete_dtypes_and_validates_binding, test_reference_contract_enforces_one_to_three_references_and_dtype_device | 4 |
| /home/shangguanrz/project/pic-edit/tests/test_v65_face_integration.py | 回归测试（保留） | tests/test_v65_face_integration.py | 460 | _prepared_face, _legacy_detail, test_legacy_condition_splits_face_and_hand_and_hand_roundtrips_without_face, test_hand_condition_contract_preserves_discrete_types_and_cfg_order, test_face_adapter_zero_init_invalid_gradients_and_activated_signal | 3 |
| /home/shangguanrz/project/pic-edit/tests/test_v65_face_training_gradients.py | 回归测试（保留） | tests/test_v65_face_training_gradients.py | 164 | _DifferentiableVAE, _NoGradPipelineWrapper, test_training_decode_bypasses_no_grad_pipeline_wrapper, test_inference_decode_can_use_pipeline_wrapper, _FlattenEncoder | 2 |
| /home/shangguanrz/project/pic-edit/tests/test_v65_iper_face_pipeline.py | 回归测试（保留） | tests/test_v65_iper_face_pipeline.py | 451 | test_precompute_cli_can_run_directly_from_repository_root, _camera_params, test_smplx_137_face_slice_and_raw_camera_inverse_are_exact, test_retargeted_camera_without_raw_fields_fails_clearly, test_face_box_is_square_expanded_and_crop_pads_safely | 2 |
| /home/shangguanrz/project/pic-edit/tests/test_v65_upstream_migrations.py | 回归测试（保留） | tests/test_v65_upstream_migrations.py | 126 | test_runtime_modules_directly_reuse_pinned_upstream_adaptations, test_visual_persona_attention_matches_pinned_core_equations, test_stableanimator_fusion_keeps_shortcut_and_zero_delta_contract, test_xdyna_cfg_order_and_optional_residual_sum, test_vendored_sources_record_fixed_commits_and_license_status | 2 |
| /home/shangguanrz/project/pic-edit/tests/test_v66_hand_core.py | 回归测试（保留） | tests/test_v66_hand_core.py | 209 | test_hand_pose_and_confident_heatmap, _inputs, test_hand_branch_zero_init_and_invalid_gradients, test_source_index_binding_rejects_cross_person_reference, test_mode_fusion_is_exclusive_and_preserves_global | 2 |
| /home/shangguanrz/project/pic-edit/tests/test_v6_detail_branch.py | 回归测试（保留） | tests/test_v6_detail_branch.py | 587 | _core, _core_inputs, _activate_heads, test_detail_branch_is_independent_zero_initialized_and_skips_invalid_rows, test_missing_detail_skips_branch | 1 |
| /home/shangguanrz/project/pic-edit/tests/test_v6_detail_contracts.py | 回归测试（保留） | tests/test_v6_detail_contracts.py | 255 | make_condition, make_references, test_detail_condition_contract_to_and_index_select_preserve_discrete_dtypes, test_reference_contract_to_and_index_select, test_detail_condition_rejects_invalid_inputs | 2 |
| /home/shangguanrz/project/pic-edit/tests/test_v6_detail_preparer.py | 回归测试（保留） | tests/test_v6_detail_preparer.py | 202 | make_inputs, build_preparer, prepare_detail, test_preparer_emits_exact_shapes_masks_and_zero_optional_canvas, test_missing_references_use_null_appearance_but_target_invalid_tokens_stay_zero | 1 |
| /home/shangguanrz/project/pic-edit/third_party/__init__.py | 依赖或可执行 Python 包（保留） | third_party/__init__.py | 1 | Audited third-party source adaptations bundled with the project. | 105 |
| /home/shangguanrz/project/pic-edit/third_party/champ_guidance/__init__.py | 依赖或可执行 Python 包（保留） | third_party/champ_guidance/__init__.py | 5 | Single-frame adaptation of CHAMP's condition GuidanceEncoder. | 105 |
| /home/shangguanrz/project/pic-edit/third_party/champ_guidance/guidance_encoder.py | 依赖或可执行 Python 包（保留） | third_party/champ_guidance/guidance_encoder.py | 81 | A dependency-free single-frame adaptation of CHAMP's GuidanceEncoder. | 2 |
| /home/shangguanrz/project/pic-edit/third_party/v65_face/UPSTREAM_SOURCES.md | 依赖或可执行 Python 包（保留） | third_party/v65_face/UPSTREAM_SOURCES.md | 23 |  | 5 |
| /home/shangguanrz/project/pic-edit/third_party/v65_face/__init__.py | 依赖或可执行 Python 包（保留） | third_party/v65_face/__init__.py | 29 | Pinned, locally adapted building blocks used by the V6.5 face path. | 105 |
| /home/shangguanrz/project/pic-edit/third_party/v65_face/licenses/STABLEANIMATOR-MIT.txt | 依赖或可执行 Python 包（保留） | third_party/v65_face/licenses/STABLEANIMATOR-MIT.txt | 21 |  | 2 |
| /home/shangguanrz/project/pic-edit/third_party/v65_face/licenses/X-DYNA-APACHE-2.0.txt | 依赖或可执行 Python 包（保留） | third_party/v65_face/licenses/X-DYNA-APACHE-2.0.txt | 176 |  | 2 |
| /home/shangguanrz/project/pic-edit/third_party/v65_face/mcld.py | 依赖或可执行 Python 包（保留） | third_party/v65_face/mcld.py | 157 | 2D adaptations of the pinned MCLD face preprocessing and PoseGuider code. | 8 |
| /home/shangguanrz/project/pic-edit/third_party/v65_face/stableanimator.py | 依赖或可执行 Python 包（保留） | third_party/v65_face/stableanimator.py | 191 | Direct adaptations of StableAnimator identity-fusion components. | 9 |
| /home/shangguanrz/project/pic-edit/third_party/v65_face/visual_persona.py | 依赖或可执行 Python 包（保留） | third_party/v65_face/visual_persona.py | 228 | Direct adaptations of Visual Persona's face-token attention primitives. | 9 |
| /home/shangguanrz/project/pic-edit/third_party/v65_face/xdyna.py | 依赖或可执行 Python 包（保留） | third_party/v65_face/xdyna.py | 35 | Small pipeline utilities directly adapted from pinned X-Dyna control flow. | 8 |
| /home/shangguanrz/project/pic-edit/tools/render_3dpw_60_pairs_report_style_v1.py | 报告或可视化生成 | scripts/reports/render_3dpw_60_pairs_report_style_v1.py | 583 | Render already-selected 3DPW 60 showcase pairs in report style. | 0 |
| /home/shangguanrz/project/pic-edit/tools/visualize_3dpw_20_pairs_assets.py | 报告或可视化生成 | scripts/reports/visualize_3dpw_20_pairs_assets.py | 783 | Visualize 20 representative pairs from 3DPW single-person dataset. | 0 |
| /home/shangguanrz/project/pic-edit/tools/visualize_3dpw_20_pairs_assets_diverse.py | 报告或可视化生成 | scripts/reports/visualize_3dpw_20_pairs_assets_diverse.py | 880 | load_rgb, load_gray, load_depth, depth_vis, draw_pair | 0 |
| /home/shangguanrz/project/pic-edit/tools/visualize_3dpw_60_pairs_diverse_dinov2_v2.py | 报告或可视化生成 | scripts/reports/visualize_3dpw_60_pairs_diverse_dinov2_v2.py | 2594 | 3DPW 60-pair diverse visualization set v2 | 0 |
| /home/shangguanrz/project/pic-edit/tools/visualize_3dpw_60_pairs_large_pose.py | 报告或可视化生成 | scripts/reports/visualize_3dpw_60_pairs_large_pose.py | 1055 | 3DPW final dataset visualization | 0 |
| /home/shangguanrz/project/pic-edit/train_adapter.py | 训练入口 | scripts/train/train_adapter.py | 268 | PairedPoseDataset, train_adapter | 2 |
| /home/shangguanrz/project/pic-edit/train_hi4d_qwen21_dual.py | 训练入口 | scripts/train/train_hi4d_qwen21_dual.py | 345 | Train Qwen21 dual-person control from Hi4D N2C pairs. | 4 |
| /home/shangguanrz/project/pic-edit/train_iper_qwen21_body.py | 训练入口 | scripts/train/train_iper_qwen21_body.py | 355 | Train a target-token body adapter on the frozen Qwen-Image-2.1 backbone. | 5 |
| /home/shangguanrz/project/pic-edit/train_iper_qwen21_face.py | 训练入口 | scripts/train/train_iper_qwen21_face.py | 392 | Train a source-identity Qwen21 face ROI branch over the frozen Body 20k adapter. | 3 |
| /home/shangguanrz/project/pic-edit/train_iper_qwen21_full.py | 训练入口 | scripts/train/train_iper_qwen21_full.py | 354 | Bounded Body/Face training of the complete V6 architecture on Qwen21. | 3 |
| /home/shangguanrz/project/pic-edit/train_iper_v65_face.py | 训练入口 | scripts/train/train_iper_v65_face.py | 1047 | V6.5 independent Face Adapter single-sample iPER training entry point. | 8 |
| /home/shangguanrz/project/pic-edit/train_iper_v66_hand.py | 训练入口 | scripts/train/train_iper_v66_hand.py | 205 | Fixed-pair iPER V6.6 Hand overfit; legacy/off/v66 share stochastic state. | 1 |
| /home/shangguanrz/project/pic-edit/train_iper_v6_native.py | 训练入口 | scripts/train/train_iper_v6_native.py | 1299 | Native V6.3 Training Script for iPER Pose Editing (Scheme A: Single-Person Geometry). | 12 |
| /home/shangguanrz/project/pic-edit/train_tiktok_overfit_sanity.py | 训练入口 | scripts/train/train_tiktok_overfit_sanity.py | 810 | Two-phase single-sample overfit sanity test for the V6.5 TikTok pipeline. | 1 |
| /home/shangguanrz/project/pic-edit/train_tiktok_v65_face.py | 训练入口 | scripts/train/train_tiktok_v65_face.py | 686 | V6.5 independent Face Adapter single-sample TikTok training entry point. | 4 |
| /home/shangguanrz/project/pic-edit/visualize_tiktok_v3_final_global_blur.py | 报告或可视化生成 | scripts/reports/visualize_tiktok_v3_final_global_blur.py | 362 | read_jsonl, read_bgr, build_index, resize_h, fit_width | 0 |
| /home/shangguanrz/project/pic-edit/vram_lab/README.md | 依赖或可执行 Python 包（保留） | vram_lab/README.md | 84 |  | 2 |
| /home/shangguanrz/project/pic-edit/vram_lab/__init__.py | 依赖或可执行 Python 包（保留） | vram_lab/__init__.py | 1 | Reproducible DeepGen editing and adapter memory experiments. | 105 |
| /home/shangguanrz/project/pic-edit/vram_lab/acceptance.py | 依赖或可执行 Python 包（保留） | vram_lab/acceptance.py | 104 | CPU-only final audit of the completed experiment, independent of UI labels. | 21 |
| /home/shangguanrz/project/pic-edit/vram_lab/analysis.py | 依赖或可执行 Python 包（保留） | vram_lab/analysis.py | 183 | Evidence-derived selection and memory report; never invent human scores. | 3 |
| /home/shangguanrz/project/pic-edit/vram_lab/common.py | 依赖或可执行 Python 包（保留） | vram_lab/common.py | 168 | write_json, sha256, source_path, preprocess, quality_manifest | 37 |
| /home/shangguanrz/project/pic-edit/vram_lab/contact_sheet.py | 依赖或可执行 Python 包（保留） | vram_lab/contact_sheet.py | 23 | Lay out unmodified experiment images with run labels for visual review. | 0 |
| /home/shangguanrz/project/pic-edit/vram_lab/package.py | 依赖或可执行 Python 包（保留） | vram_lab/package.py | 14 | Export reviewable evidence without multi-GB adapter/cache tensors. | 4 |
| /home/shangguanrz/project/pic-edit/vram_lab/report.py | 依赖或可执行 Python 包（保留） | vram_lab/report.py | 233 | Build a dependency-free, evidence-backed CPU-only experiment report. | 43 |
| /home/shangguanrz/project/pic-edit/vram_lab/run.py | 依赖或可执行 Python 包（保留） | vram_lab/run.py | 206 | Serial, resumable experiment controller. A completed run is immutable. | 75 |
| /home/shangguanrz/project/pic-edit/vram_lab/worker.py | 依赖或可执行 Python 包（保留） | vram_lab/worker.py | 403 | A single isolated GPU job. Never run jobs concurrently on the same GPU. | 2 |
| /home/shangguanrz/quick_grid.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 42 | ld, font | 0 |
| /home/shangguanrz/qwen21_interaction_probe.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 227 | Hi4D two-person interaction probe on native Qwen-Image-2.1. | 0 |
| /home/shangguanrz/qwen21_recap.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 46 |  | 0 |
| /home/shangguanrz/qwen21_retarget_probe.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 295 | Native Qwen-Image-2.1 cross-body-type pose retargeting probe. | 0 |
| /home/shangguanrz/render_b_sheet.py | 项目外报告与可视化工具 | scripts/reports/（逐项核验） | 37 | Render B (pose reference) images large enough to describe poses precisely. | 0 |
| /home/shangguanrz/review_sheet.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 47 | Compact review sheet: A \| B \| generated, 3 samples per page. | 0 |
| /home/shangguanrz/run_smplestx.sh | 项目外运行编排 | scripts/run/（逐项核验） | 20 |  | 0 |
| /home/shangguanrz/run_stage1.sh | 项目外运行编排 | scripts/run/（逐项核验） | 13 |  | 0 |
| /home/shangguanrz/run_stage2_both.sh | 项目外运行编排 | scripts/run/（逐项核验） | 14 |  | 0 |
| /home/shangguanrz/showcam.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 11 |  | 0 |
| /home/shangguanrz/sig_qwen.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 10 |  | 0 |
| /home/shangguanrz/survey_hi4d_pairs.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 44 | Survey Hi4D native_interaction_v4_2 pairs to pick diverse two-person test samples. | 0 |
| /home/shangguanrz/vae_ceiling.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 80 | Estimate the copy ceiling: run the GT target through Qwen's VAE and measure round-trip SSIM. | 0 |
| /home/shangguanrz/verify_qwen21.py.sh | 项目外运行编排 | scripts/run/（逐项核验） | 53 |  | 0 |
| /home/shangguanrz/watch_and_run.sh | 项目外运行编排 | scripts/run/（逐项核验） | 14 |  | 0 |
| /home/shangguanrz/zoom_A.py | 项目外工具或临时诊断 | scripts/diagnostics/ 或 archive/scratch/（逐项核验） | 21 |  | 0 |
