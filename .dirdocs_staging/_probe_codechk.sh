set -u
cd /data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/code || exit 1
for f in src/README.md src/pose_control/README.md scripts/README.md; do
  printf '@@@ %s | mtime=%s | bytes=%s\n' "$f" "$(date -r "$f" '+%m-%d %H:%M')" "$(stat -c%s "$f")"
  head -c 300 "$f"; printf '\n'
done
