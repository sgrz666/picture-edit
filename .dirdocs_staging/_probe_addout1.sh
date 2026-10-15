R=/home/shangguanrz/project/pic-edit/add/outputs
for d in "$R"/*/; do
  echo "===== $d"
  echo "N_DIRS:"; find "$d" -maxdepth 1 -type d | wc -l
  echo "N_FILES:"; find "$d" -maxdepth 1 -type f | wc -l
  echo "TOP:"; ls "$d" | head -25 | tr '\n' ' '; echo
  echo "DIRECT_FILES:"; find "$d" -maxdepth 1 -type f -printf '%s %f\n' 2>/dev/null | sort -rn | head -20
done
