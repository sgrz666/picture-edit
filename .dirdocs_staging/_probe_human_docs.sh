set -u
cd /data/shangguanrz || exit 1
for f in \
  pic-edit-datasets/TikTokDataset/README.md \
  pic-edit-datasets/Hi4D-origin/README.md \
  pic-edit-models/Qwen-Image-2.1/README.pic-edit.md \
  pic-edit-archive/cleanup_20261003/models/Qwen-Image-Edit-20B/README.pic-edit.md \
  pic-edit-archive/cleanup_20261003/experiments/iper_v6_comparative_3000/README.md \
  pic-edit-archive/cleanup_20261003/experiments/review_20260908_v2/README.md ; do
  if [ -f "$f" ]; then
    printf '@@@ FILE %s | mtime=%s | bytes=%s | lines=%s\n' "$f" "$(date -r "$f" '+%m-%d %H:%M')" "$(stat -c%s "$f")" "$(wc -l < "$f")"
  fi
done
echo '=== 顶层与各二级是否已有 README/目录说明（列出存在与否） ==='
find . -maxdepth 2 -type d -not -path './.*' | sed 's|^\./||' | while IFS= read -r d; do
  [ "$d" = "." ] && continue
  r=''; [ -f "$d/README.md" ] && r="README"
  g=''; [ -f "$d/目录说明.md" ] && g="说明"
  echo "  $d  [$r$g]"
done
