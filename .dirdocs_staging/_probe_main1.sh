set -u
cd /home/shangguanrz/project/pic-edit || exit 1
for f in src/README.md src/pose_control/README.md configs/README.md models/README.md research/README.md results/README.md inputs/README.md patches/README.md docs/README.md logs/README.md datasets/README.md experiments/qwen21_iper_body/README.md; do
  if [ -f "$f" ]; then
    printf '@@@ %s | mtime=%s | bytes=%s\n' "$f" "$(date -r "$f" '+%Y-%m-%d')" "$(stat -c%s "$f")"
    head -c 420 "$f"
    printf '\n'
  fi
done
