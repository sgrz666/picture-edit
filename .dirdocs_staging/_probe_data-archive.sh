#!/bin/bash
E=/data/shangguanrz/pic-edit-experiments
P=$E/pei2i_body_face_8arm_20261009
echo "=== existing README.md anywhere in my subtree? ==="
find $E -maxdepth 4 -iname 'README*.md' -o -maxdepth 4 -iname '目录总览.md' 2>/dev/null
echo "=== pei2i top (all, incl hidden) ==="
ls -la $P
echo "=== du each subdir ==="
du -sh --si $P/*/ 2>/dev/null
echo "=== top-level JSON/markers ==="
ls -la $P/*.json $P/*.md $P/*.log 2>/dev/null
echo "=== find marker files ==="
find $P -maxdepth 2 -name '_*.json' -o -maxdepth 2 -name '_STATUS*' 2>/dev/null | head
