R=/home/shangguanrz/project/pic-edit/add/outputs
echo "########## 3dpw_audit/audit_summary.txt"; cat "$R/3dpw_audit/audit_summary.txt"
echo; echo "########## 3dpw_pair/pair_build_summary.txt"; cat "$R/3dpw_pair/pair_build_summary.txt"
echo; echo "########## 3dpw_pair/pair_refine_v2_summary.txt"; cat "$R/3dpw_pair/pair_refine_v2_summary.txt"
echo; echo "########## 3dpw_pair/pair_v4_dedup_summary.txt"; cat "$R/3dpw_pair/pair_v4_dedup_summary.txt"
echo; echo "########## 3dpw_quality/quality_summary.txt"; cat "$R/3dpw_quality/quality_summary.txt"
echo; echo "########## 3dpw_assets_v2_1/usable_frames_v1_summary.json"; cat "$R/3dpw_assets_v2_1/usable_frames_v1_summary.json"
echo; echo "########## 3dpw_assets_v2_1/usable_frames_v1_1_summary.json"; cat "$R/3dpw_assets_v2_1/usable_frames_v1_1_summary.json"
echo; echo "########## wc -l of key CSVs"
for f in 3dpw_audit/person_stats.csv 3dpw_audit/sequence_stats.csv 3dpw_pair/pair_candidates_v1.csv 3dpw_pair/pair_candidates_v2.csv 3dpw_pair/pair_candidates_v3_provisional.csv 3dpw_pair/pair_candidates_v4_dedup.csv 3dpw_pair/pair_quality_v3.csv 3dpw_pair/pair_rejected_v3_provisional.csv 3dpw_quality/structural_candidates.csv 3dpw_assets_v2_1/usable_frames_v1.csv 3dpw_assets_v2_1/usable_frames_v1_1.csv 3dpw_assets_v2_1/rejected_frames_v1_1.csv 3dpw_assets_v2_1/risk_frames_v1_1.csv 3dpw_assets_v2_1/clean_frames_v1_1.csv; do printf '%s ' "$f"; wc -l < "$R/$f"; done
