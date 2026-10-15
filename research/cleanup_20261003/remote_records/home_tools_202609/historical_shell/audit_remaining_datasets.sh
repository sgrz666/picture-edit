#!/usr/bin/env bash
# READ-ONLY audit of remaining datasets on / (no writes to any dataset)
set -u
D=/home/shangguanrz/project/pic-edit/datasets
cd "$D" || exit 1

echo "=== DATASET AUDIT START $(date -Is) ==="
echo "host=$(hostname)  df_/ :"; df -h / | tail -1
echo

for t in champ_motions_example iPER iPER_sample TikTokDataset Hi4D_pilot_v1; do
  echo "############################################################"
  echo "###### $t"
  echo "############################################################"
  [ -e "$t" ] || { echo "  MISSING"; continue; }
  echo "-- du blocks / apparent --"
  du -sh "$t" 2>/dev/null
  du -sh --apparent-size "$t" 2>/dev/null

  echo "-- object census (single find pass) --"
  find "$t" -printf '%y %n\n' 2>/dev/null | awk '
    { t[$1]++; if ($1=="f" && ($2+0)>1) h++ }
    END { printf "  files=%d  dirs=%d  symlinks=%d  hardlinked_files=%d  other=%d\n",
                 t["f"], t["d"], t["l"], h, t["p"]+t["s"]+t["c"]+t["b"] }'

  echo "-- broken symlinks --"
  printf "  count="; find "$t" -xtype l 2>/dev/null | wc -l

  echo "-- symlink target style (3 samples) --"
  find "$t" -type l 2>/dev/null | head -3 | while read -r l; do printf "  %s -> %s\n" "$l" "$(readlink "$l")"; done

  echo "-- recent writes --"
  printf "  last 24h="; find "$t" -type f -newermt '24 hours ago' 2>/dev/null | wc -l
  printf "  last 7d ="; find "$t" -type f -newermt '7 days ago' 2>/dev/null | wc -l

  echo "-- size by first-level subdir (top 12) --"
  du -sh "$t"/* 2>/dev/null | sort -h | tail -12 | sed 's/^/  /'
  echo
done

echo "=== DATASET AUDIT END $(date -Is) ==="
