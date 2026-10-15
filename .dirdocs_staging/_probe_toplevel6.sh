#!/bin/bash
R=/home/shangguanrz/project/pic-edit
cd $R
echo "== scripts subdir py counts and lines =="
for d in scripts/audit scripts/data scripts/eval scripts/reports scripts/run scripts/train scripts/weights; do
  echo "$d py=$(find $d -maxdepth 1 -name '*.py'|wc -l) sh=$(find $d -maxdepth 1 \( -name '*.sh' -o -name '*.ps1' \)|wc -l) lines=$(find $d -maxdepth 1 -name '*.py'|xargs wc -l 2>/dev/null|tail -1|awk '{print $1}')"
done
echo "== scripts root py count =="
find scripts -maxdepth 1 -name '*.py' | wc -l
echo "== logs/legacy mtime range =="
ls -la --time-style=+%Y-%m-%d logs/legacy | awk 'NR>1{print $6}' | sort | uniq -c
echo "== results extras =="
file results/*.png | sed 's/,.*size/ size/'
head -8 results/QWEN_EDIT_20B_MEMORY_ANALYSIS.md
ls experiments/review_20260908 experiments/review_20260908_v2 2>/dev/null | head -6
echo "== vram_lab files head =="
head -4 vram_lab/run.py
echo "== third_party v65_face =="
head -12 third_party/v65_face/UPSTREAM_SOURCES.md
ls third_party/v65_face/licenses
wc -l third_party/v65_face/*.py third_party/champ_guidance/*.py
echo "== src/controls generator.py docstring/class =="
grep -n '^class\|^def\|^\"\"\"' src/controls/generator.py src/adapter/pose_adapter.py src/integration/champ_deepgen.py src/pose_control/qwen21_full/qwen_runtime.py src/pose_control/qwen21_full/execution.py src/pose_control/qwen21_full/joint_training.py src/pose_control/v6/token_encoders.py | head -40
echo "== docs heads =="
for f in docs/PROJECT_STRUCTURE_20261003.md docs/架构审查_20261012.md docs/目录清理执行记录_20261009.md docs/adapter_v6_architecture.md docs/qwen21_hi4d_dual.md docs/iper_v6_training_plan.md docs/tiktok_qwen21_full_training.md docs/champ_deepgen_single_sample_runbook.md docs/BFS_面部保持移用评估_20261004.md docs/BFS_扩大样本对照_20261004.md; do echo "-- $f"; head -5 "$f"; done
echo "== docs/experiments + superpowers readme heads =="
head -8 docs/experiments/README.md docs/superpowers/README.md 2>/dev/null
echo "== research readme first lines =="
for d in research/*; do [ -d "$d" ] || continue; echo "-- $d"; head -4 "$d/README.md" 2>/dev/null; done
echo "== caches pt sample =="
ls -la /data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls/control_000d4c3aa7eb1504623c181b.pt
echo "== root files =="
ls -la | grep -E '^-' | awk '{print $9}'
echo "== git top status (is everything tracked?) =="
git -C . status --short | head -5; git log --oneline -3
echo "== inputs origin =="
grep -rl 'inputs/person1' src scripts --include='*.py' | head -5
grep -rl 'dual_person.jpg' src scripts --include='*.py' | head -5
echo "== who consumes logs/legacy =="
grep -rl 'logs/legacy' src scripts docs --include='*' | head
echo "== patches consumer =="
grep -rl 'deepgen_pose_control.patch' . --include='*.py' --include='*.md' --exclude-dir=.git -m1 2>/dev/null | head -6
