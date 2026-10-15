set -u
cd /data/shangguanrz || exit 1
for f in \
  pic-edit-datasets/TikTokDataset/README.md \
  pic-edit-datasets/Hi4D-origin/README.md \
  pic-edit-models/Qwen-Image-2.1/README.pic-edit.md \
  pic-edit-archive/cleanup_20261003/models/Qwen-Image-Edit-20B/README.pic-edit.md \
  pic-edit-archive/cleanup_20261003/experiments/iper_v6_comparative_3000/README.md \
  pic-edit-archive/cleanup_20261003/experiments/review_20260908_v2/README.md ; do
  echo "@@@@@ $f"
  cat "$f"
  echo
done
