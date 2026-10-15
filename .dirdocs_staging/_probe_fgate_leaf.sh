#!/bin/bash
cd /home/shangguanrz/project/pic-edit/experiments || exit 1
for d in tiktok_overfit_sanity_full tiktok_overfit_sanity_smoke iper_v65_face_overfit_024_6 iper_v65_face_overfit_smoke iper_v64_detail_overfit_024_6 tiktok_v65_face_overfit_00001 tiktok_v65_test_2steps test_smoke_detail tiktok_v65_pure_inference tiktok_00001_condition_repair; do
  for sub in checkpoints condition_cache previews gen_evals; do
    p="$d/$sub"
    if [ -d "$p" ]; then
      echo "=== $p ==="
      ls -la --time-style=+%Y-%m-%d "$p" 2>/dev/null | grep -v 'total\|目录说明\|README' | tail -n +2 | awk '{print $5, $7, $8}' | head -14
      echo "  [count] $(find "$p" -maxdepth 1 -type f | wc -l) files, du $(du -sh --si "$p" 2>/dev/null | cut -f1)"
    fi
  done
done
echo "############ LOG TAILS ############"
for f in tiktok_overfit_sanity_full/train.log iper_v65_face_overfit_024_6/train.log; do
  echo "=== tail $f ==="; tail -n 22 "$f"
done
