set -u
cd /data/shangguanrz || exit 1
secs() { find "$1" -maxdepth "${2:-2}" -mindepth 1 -type d -not -path '*/.*' 2>/dev/null | sort; }
count() { find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l; find "$1" -mindepth 1 -maxdepth 1 -type f 2>/dev/null | wc -l; }
echo '##### TikTokDataset L1/L2 #####'
for d in $(secs pic-edit-datasets/TikTokDataset 2); do
  read sd fl <<< "$(count "$d")"
  sz=$(du -sh --si "$d" 2>/dev/null | cut -f1)
  echo "$d | sub=$sd files=$fl size=$sz"
done
echo '##### TikTokDataset 直属文件 #####'
ls -la pic-edit-datasets/TikTokDataset | head -30
echo '##### Hi4D-origin #####'
ls -la pic-edit-datasets/Hi4D-origin | head -8
find pic-edit-datasets/Hi4D-origin -type f | wc -l
echo '##### pic-edit-models / caches / samples / add_migration / experiments(浅) #####'
for d in pic-edit-models pic-edit-caches dataset_samples add_migration pic-edit-experiments; do
  echo "--- $d ---"
  for s in $(secs "$d" 3); do
    read sd fl <<< "$(count "$s")"
    sz=$(du -sh --si "$s" 2>/dev/null | cut -f1)
    echo "$s | sub=$sd files=$fl size=$sz"
  done
done
echo '##### pic-edit-archive #####'
for d in $(secs pic-edit-archive 2); do
  read sd fl <<< "$(count "$d")"
  sz=$(du -sh --si "$d" 2>/dev/null | cut -f1)
  echo "$d | sub=$sd files=$fl size=$sz"
done
for s in $(secs pic-edit-archive/cleanup_20261003 2); do
  read sd fl <<< "$(count "$s")"
  echo "  $s | sub=$sd files=$fl"
done
echo '##### 各顶层体积 #####'
du -sh --si pic-edit-datasets pic-edit-models pic-edit-caches dataset_samples pic-edit-archive add_migration pic-edit-experiments 2>/dev/null
