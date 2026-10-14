#!/usr/bin/env bash
# Run the guide-source comparison after everything else has finished.
#
# The fit twins used to poll for .workflow.lock while the batch was running, but the batch holds
# the lock for almost every second of its life, so a 60-second poll would wait for hours and still
# miss the gap. Running them after the batch is both reliable and the right order: the two arms
# and the two framings are compared first, then the guide source, one variable at a time.
set -u
cd "$(dirname "$0")" || exit 1

QWEN_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
LOG=logs_batch/fit_after.log
mkdir -p logs_batch comparisons

note() { echo "$(date -Is) $*" >> "$LOG"; }

until grep -q FINALIZE_DONE logs_batch/batch.log 2>/dev/null; do
  sleep 300
done
note "FINALIZE_DONE starting the guide-source comparison"

# Derive the list from cached fits so the guide comparison covers every pair instead of three.
# A twin only makes sense next to an existing ground-truth-guided run of the same cell.
for npz in fits/*.npz; do
  pair=$(basename "$npz" .npz)
  [ -d "runs/$pair.skeleton_only.b_canvas" ] || continue
  if [ -d "runs/$pair.skeleton_only.b_canvas.fit" ]; then
    note "SKIP_TWIN $pair (already exists)"
    continue
  fi
  ./run_fit_guide.sh "$pair" skeleton_only b_canvas
  note "FIT_TWIN $pair exit=$?"
done

"$QWEN_PY" compare_dual.py --by source --out comparisons/compare_source.json > logs_batch/fit_report.txt 2>&1 \
  || note "COMPARE_SOURCE_FAILED"
"$QWEN_PY" report_dual.py > logs_batch/report.txt 2>&1 || note "REPORT_FAILED"
note "FIT_ALL_DONE"
echo "$(date -Is) FIT_ALL_DONE" >> logs_batch/batch.log
note "FIT_AFTER_DONE"
echo "$(date -Is) FIT_AFTER_DONE" >> logs_batch/batch.log
