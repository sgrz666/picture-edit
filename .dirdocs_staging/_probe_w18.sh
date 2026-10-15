cd /home/shangguanrz/project/pic-edit/logs/legacy || exit 1
for f in qwen21_hi4d_dual_precompute.log qwen21_hi4d_dual_train.log qwen21_hi4d_single_hug_eval.log qwen21_hi4d_single_hug_eval_recheck.log qwen21_hi4d_single_hug_eval_seed43.log qwen21_hi4d_single_hug_overfit.log select_tiktok_v3_frames.log tiktok_showcase_00072.log tiktok_showcase_01.log audit_and_select_tiktok_v2.log; do
  echo "#### $f  [$(wc -l < $f) lines]"; tail -n 12 "$f"; echo; done
echo "#### run_tiktok_v3_dwpose_smoke.log"; tail -n 12 run_tiktok_v3_dwpose_smoke.log
echo "#### run_tiktok_v3_dwpose_full.log"; tail -n 15 run_tiktok_v3_dwpose_full.log
echo "#### fill_tiktok_v3_missing_geometry.log"; tail -n 15 fill_tiktok_v3_missing_geometry.log
echo "#### build_tiktok_v3_inventory.log"; tail -n 15 build_tiktok_v3_inventory.log
echo "#### nohup_hi4d_repair_v2.out"; tail -n 12 nohup_hi4d_repair_v2.out
echo "#### nohup_hi4d_target_assets.out"; tail -n 12 nohup_hi4d_target_assets.out
echo "#### failures json"; cat ../hi4d_dwpose_maskbbox_failures.json
