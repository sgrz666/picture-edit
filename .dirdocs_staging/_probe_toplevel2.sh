#!/bin/bash
R=/home/shangguanrz/project/pic-edit
cd $R
for d in src src/pose_control configs caches inputs outputs patches logs logs/legacy docs docs/* tests tests/* third_party research results archive datasets scripts vram_lab; do
  echo "==== $d ===="
  ls -A "$d" | head -70
done
echo "==== README status of depth2 targets ===="
for d in src/* src/pose_control/* configs/* scripts/* docs/* tests/* third_party/* logs/legacy outputs/* caches/* patches/* inputs/* vram_lab/*; do
  [ -d "$d" ] || continue
  rd=none; [ -f "$d/README.md" ] && rd=$(grep -q 'gen_folder_docs' "$d/README.md" 2>/dev/null && echo GEN || echo HAND)
  echo "$d README=$rd"
done
