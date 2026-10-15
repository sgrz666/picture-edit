B=/home/shangguanrz/project/pic-edit/experiments
for d in qwen21_iper_body qwen21_tiktok_v3_smoke qwen21_hi4d_dual hi4d_overfit_archive qwen-2.1-test qwen_edit_results 16-test bfs_body_face_joint_pilot_20261008 qwen21_dual_v8; do
  echo "===== DIR: $d ====="
  if [ -d "$B/$d" ]; then
    find "$B/$d" -maxdepth 2 -name README.md 2>/dev/null
    echo "-- subdirs (maxdepth1) --"
    find "$B/$d" -maxdepth 1 -mindepth 1 -printf '%y %10s %p\n' 2>/dev/null | sort -k3
  else
    echo "MISSING"
  fi
done
