cd /home/shangguanrz/project/pic-edit/logs/legacy || exit 1
for f in hi4d_interaction_v2.log hi4d_native_interaction_manifest.log hi4d_showcase_assets.log hi4d_showcase_sample01.log hi4d_target_assets_20260920_113531.log inference_202609.log iper_showcase_batch.log make_hi4d_v4_2_pair_qc.log make_hi4d_v4_2_pair_qc_after_dwpose.log recover_hi4d_interaction_v3_1.log regenerate_hi4d_dwpose_maskbbox_overwrite.log; do
  echo "#### $f  [$(wc -l < $f) lines]"; tail -n 7 "$f"; echo; done
