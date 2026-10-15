cd /home/shangguanrz/project/pic-edit/logs/legacy || exit 1
for f in frame_quality_3datasets.out frame_quality_resume.out frame_quality_v2.out hi4d_build_pairs.log hi4d_contact_strength.log hi4d_dwpose_native.log hi4d_dwpose_roi_retry.log hi4d_geometry_1500.log hi4d_geometry_1500_camera_debug.log hi4d_geometry_1500_contiguous_fix.log; do
  echo "#### $f  [$(wc -l < $f) lines]"; tail -n 8 "$f"; echo; done
