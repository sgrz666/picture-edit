R=/home/shangguanrz/project/pic-edit/add/outputs
echo "###### 3dpw_v3_review tree"
find "$R/3dpw_v3_review" -maxdepth 2 | head -40
echo "COUNTS per subdir:"; for d in "$R/3dpw_v3_review"/*/; do printf '%s ' "$d"; find "$d" -type f | wc -l; done
echo "sample names:"; ls "$R/3dpw_v3_review/01_kept_occlusion_risk" | head -8
echo
echo "###### 3dpw_v3_review_balanced count"; find "$R/3dpw_v3_review_balanced" -maxdepth 1 -type f | wc -l
echo
echo "###### 3dpw_pair_preview tree"; find "$R/3dpw_pair_preview" -maxdepth 3 | head -25
echo "counts:"; for d in "$R/3dpw_pair_preview"/*/; do printf '%s ' "$d"; find "$d" -type f | wc -l; done
echo
echo "###### 3dpw_assets_v1 tree"; find "$R/3dpw_assets_v1" -maxdepth 3 | head -30
echo "counts:"; for d in "$R/3dpw_assets_v1"/*/; do printf '%s ' "$d"; find "$d" -type f | wc -l; done
echo
echo "###### 3dpw_assets_v2_1/sequence_assets"; ls "$R/3dpw_assets_v2_1/sequence_assets" | head; find "$R/3dpw_assets_v2_1/sequence_assets" -maxdepth 2 | head -20
echo
echo "###### headers of pair CSVs"
for f in pair_candidates_v1 pair_candidates_v2 pair_candidates_v3_provisional pair_candidates_v4_dedup pair_quality_v3 pair_rejected_v3_provisional sequence_pose_stats usable_frames_v1; do echo "-- $f"; head -1 "$R/3dpw_pair/$f.csv"; done
for f in frame_quality_raw frame_person_stats_v3 sequence_quality_summary structural_candidates worst_person_sharpness_100; do echo "-- quality/$f"; head -1 "$R/3dpw_quality/$f.csv"; done
for f in usable_frames_v1 clean_frames_v1 frame_quality_v2 frame_assets_manifest; do echo "-- v2_1/$f"; head -c 400 "$R/3dpw_assets_v2_1/$f.csv" 2>/dev/null; echo; done
echo "-- audit/person_stats"; head -2 "$R/3dpw_audit/person_stats.csv"
echo "-- audit/sequence_stats"; head -2 "$R/3dpw_audit/sequence_stats.csv"
