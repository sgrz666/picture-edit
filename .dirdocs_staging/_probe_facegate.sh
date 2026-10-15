#!/bin/bash
cd /home/shangguanrz/project/pic-edit/experiments || exit 1
DIRS="tiktok_overfit_sanity_full tiktok_overfit_sanity_smoke iper_v65_face_overfit_024_6 iper_v65_face_overfit_smoke iper_v64_detail_overfit_024_6 tiktok_v65_face_overfit_00001 tiktok_00001_condition_repair tiktok_objective_eval tiktok_smoke_00001 tiktok_v65_pure_inference tiktok_v65_test_2steps test_smoke_detail v6_condition_inspection"

echo "######## SYMLINK CHECK (any of my dirs that are symlinks) ########"
for d in $DIRS; do
  if [ -L "$d" ]; then echo "SYMLINK: $d -> $(readlink -f "$d")"; fi
  # also check sub-level symlinks
  find "$d" -maxdepth 2 -type l -printf 'LINK %p -> %l\n' 2>/dev/null
done

echo "######## ALL README.md in my subtree + is it generated? ########"
for d in $DIRS; do
  find "$d" -maxdepth 3 -name README.md 2>/dev/null | while read r; do
    if grep -q "gen_folder_docs" "$r" 2>/dev/null; then tag=GENERATED; else tag=HANDWRITTEN_OR_OTHER; fi
    echo "$tag  $r  ($(wc -l < "$r") lines)"
  done
done

echo "######## panels / faces subdirs anywhere in my family ########"
for d in $DIRS; do
  find "$d" -maxdepth 3 -type d \( -name panels -o -name faces \) 2>/dev/null
done

echo "######## condition_repair children full tree (maxdepth 2) ########"
find tiktok_00001_condition_repair -maxdepth 2 -printf '%y %10s %p\n' 2>/dev/null | sort -k3

echo "######## per-dir subdir inventory: file counts + du ########"
for d in $DIRS; do
  echo "=== $d ==="
  find "$d" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | while read sd; do
    n=$(find "$sd" -type f 2>/dev/null | wc -l)
    sz=$(du -sh --si "$sd" 2>/dev/null | cut -f1)
    echo "  DIR $sd  files=$n  du=$sz"
  done
done
