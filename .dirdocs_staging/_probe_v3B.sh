#!/bin/bash
R=/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3/manifests
echo "===== LINE COUNTS (all jsonl) ====="
find "$R" -name '*.jsonl' | sort | while read f; do printf "%s\t%s\n" "$(wc -l < "$f")" "${f#$R/}"; done
echo "===== SUMMARY JSONS ====="
for s in candidate_pool_summary.json condition_inventory_summary.json dwpose_summary.json geometry_filter_summary.json pair_summary.json pair_summary_strict_v1.json split_summary.json usable_frame_pool_summary.json summary_pre_dwpose.json fill_missing_geometry_summary.json; do
  echo "----- $s -----"; cat "$R/$s" 2>/dev/null; echo;
done
echo "===== FIRST LINE OF KEY MANIFESTS ====="
for f in frame_inventory.jsonl candidate_pool.jsonl candidate_pool_geometry_valid.jsonl candidate_pool_geometry_reject.jsonl condition_inventory.jsonl dwpose_results.jsonl usable_frame_pool.jsonl usable_frame_pool_split.jsonl frames_scored.jsonl selected_pre_dwpose.jsonl sequence_screening.jsonl dynamic_sampling_candidates.jsonl frame_fine_reject.jsonl videos.jsonl; do
  echo "----- $f -----"; head -c 1200 "$R/$f" 2>/dev/null; echo;
done
echo "===== pairs/pairs_strict first lines ====="
for f in pairs/train_pairs.jsonl pairs/val_pairs.jsonl pairs/test_pairs.jsonl pairs_strict_v1/train_pairs.jsonl pairs_strict_v1/val_pairs.jsonl pairs_strict_v1/test_pairs.jsonl; do
  echo "----- $f -----"; head -n 1 "$R/$f" 2>/dev/null | head -c 1500; echo;
done
echo "===== frozen_candidate_pool_v2 + strict_visual_v2 ====="
for f in frozen_candidate_pool_v2/candidate_pool.jsonl strict_visual_v1/train.jsonl strict_visual_v1/usable_frame_pool_strict.jsonl; do
  echo "----- $f -----"; head -n 1 "$R/$f" 2>/dev/null | head -c 1000; echo;
done
cat "$R/fill_missing_geometry_sequences.txt" 2>/dev/null
