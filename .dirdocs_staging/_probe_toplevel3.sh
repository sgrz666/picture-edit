#!/bin/bash
R=/home/shangguanrz/project/pic-edit
cd $R
echo "== symlink checks =="
for p in caches/qwen21_hi4d_controls logs/legacy results/legacy_demo_202609 results/review_20260908 results/review_20260908_v2 datasets/* archive/*; do
  [ -e "$p" ] || continue
  if [ -L "$p" ]; then echo "$p -> $(readlink -f $p)"; else echo "$p DIR size=$(du -sh --si $p 2>/dev/null|cut -f1) d1=$(find $p -maxdepth 1 -type d|tail -n +2|wc -l) f1=$(find $p -maxdepth 1 -type f|wc -l)"; fi
done
echo "== src py linecounts =="
for d in src/adapter src/controls src/data src/evaluation src/integration src/pose_control src/pose_control/qwen21 src/pose_control/qwen21_full src/pose_control/v6; do
  echo "-- $d"; ls "$d"; echo "files=$(find $d -maxdepth 1 -type f -not -name '*.pyc'|wc -l) pylines=$(find $d -maxdepth 1 -name '*.py'|xargs wc -l 2>/dev/null|tail -1)"
done
echo "== pose_control subdirs deeper =="
find src/pose_control -mindepth 2 -maxdepth 2 -type d
echo "== scripts subdirs =="
for d in scripts/audit scripts/data scripts/eval scripts/reports scripts/run scripts/train scripts/weights; do echo "-- $d"; ls "$d" | head -40; done
echo "== scripts/run,train,eval depth2 =="
find scripts -mindepth 2 -maxdepth 2 -type d -not -name __pycache__
echo "== docs sub =="
ls docs/experiments docs/superpowers docs/superpowers/*
echo "== third_party sub =="
for d in third_party/champ_guidance third_party/v65_face; do echo "-- $d"; ls -A "$d"; done
echo "== experiments dirs =="
ls experiments
