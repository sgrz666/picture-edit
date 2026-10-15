#!/bin/bash
R=/home/shangguanrz/project/pic-edit
cd $R
echo "=== DEPTH1 DIRS ==="
for d in src configs caches datasets docs experiments inputs logs outputs patches research results tests third_party archive vram_lab scripts scripts/run scripts/train scripts/eval scripts/data scripts/analysis scripts/misc docs/archive docs/setup docs/superpowers tests testdata third_party/* inputs logs/legacy caches outputs patches vram_lab/* models; do
  [ -d "$d" ] || continue
  nd=$(find "$d" -maxdepth 1 -type d ! -path "$d" | wc -l)
  nf=$(find "$d" -maxdepth 1 -type f | wc -l)
  sz=$(du -sh --si "$d" 2>/dev/null | cut -f1)
  rd="no"
  [ -f "$d/README.md" ] && rd=$(grep -q 'gen_folder_docs' "$d/README.md" && echo GEN || echo HAND)
  echo "$d | subdir1=$nd files1=$nf size=$sz README=$rd"
done
