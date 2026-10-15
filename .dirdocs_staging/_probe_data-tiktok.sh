#!/bin/bash
T=/data/shangguanrz/pic-edit-datasets/TikTokDataset
M3=$T/TikTok_v3/manifests
echo "### wc -l v3 manifests"
wc -l $M3/frame_inventory.jsonl $M3/frames_scored.jsonl $M3/frame_fine_reject.jsonl $M3/selected_pre_dwpose.jsonl $M3/sources_pre_dwpose.jsonl $M3/dwpose_results.jsonl $M3/candidate_pool.jsonl $M3/usable_frame_pool_split.jsonl $M3/condition_inventory.jsonl $M3/videos.jsonl 2>&1
echo "### wc -l pairs + frozen"
wc -l $M3/pairs/*.jsonl $M3/frozen_candidate_pool_v2/*.jsonl
echo "### geometry_valid exists?"
ls $M3 | grep -i valid
echo "### first lines"
printf 'frame_inventory: '; head -c 700 $M3/frame_inventory.jsonl; echo
printf 'videos: '; cat $M3/videos.jsonl | head -2
printf 'candidate_pool: '; head -c 700 $M3/candidate_pool.jsonl; echo
printf 'dwpose_results: '; head -c 700 $M3/dwpose_results.jsonl; echo
printf 'condition_inventory: '; head -c 700 $M3/condition_inventory.jsonl; echo
printf 'pairs/train_pairs: '; head -c 700 $M3/pairs/train_pairs.jsonl; echo
printf 'frozen/dynamic_sampling: '; head -c 700 $M3/frozen_candidate_pool_v2/dynamic_sampling_candidates.jsonl; echo
printf 'usable_frame_pool_split: '; head -c 500 $M3/usable_frame_pool_split.jsonl; echo
printf 'frame_fine_reject: '; head -c 500 $M3/frame_fine_reject.jsonl; echo
