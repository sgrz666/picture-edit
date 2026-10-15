cd /home/shangguanrz/project/pic-edit/logs/legacy || exit 1
for f in audit_and_select_tiktok_v2_bg.log audit_hi4d_assets_v1.log audit_hi4d_contact_occlusion_v1.log audit_hi4d_pair_motion_v1.log build_hi4d_final_assets_v4_2.log build_hi4d_interaction_v3.log build_hi4d_interaction_v4_1.log build_hi4d_interaction_v4_2.log build_hi4d_interaction_v4.log build_tiktok_v3_candidate_pool.log; do
  echo "#### $f  [$(wc -l < $f) lines]"; tail -n 8 "$f"; echo; done
