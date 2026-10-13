#!/usr/bin/env bash
# Finish the batch unattended: wait for the runs and the retry sweep, then re-evaluate everything
# with the current evaluator and write the final readouts.
#
# Why the re-evaluation pass: the identity metric, the target-import test and the region-based
# pose scoring were all added while runs were already completing, so the older evaluations were
# produced by older code. Comparing arms or framings across mixed evaluator versions is the kind
# of thing that looks fine and reads wrong, so every complete run gets scored once more by the
# same code before any number is quoted.
set -u
cd "$(dirname "$0")" || exit 1

QWEN_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
LOG=logs_batch/finalize.log
mkdir -p logs_batch comparisons

note() { echo "$(date -Is) $*" >> "$LOG"; }

until grep -q RETRY_DONE logs_batch/batch.log 2>/dev/null; do
  sleep 300
done
note "BATCH_AND_RETRY_DONE starting uniform re-evaluation"

completed=$("$QWEN_PY" - <<'PY'
import json
from pathlib import Path
for path in sorted(Path('runs').iterdir()):
    manifest = path / 'run.json'
    if not manifest.is_file():
        continue
    try:
        if json.loads(manifest.read_text()).get('status') == 'complete':
            print(path.name)
    except ValueError:
        continue
PY
)
count=0
for name in $completed; do
  if "$QWEN_PY" evaluate_dual.py --run "$name" --force >> "$LOG" 2>&1; then
    count=$((count + 1))
  else
    note "EVAL_FAILED $name"
  fi
done
note "REEVALUATED $count complete runs"

"$QWEN_PY" report_dual.py > logs_batch/report.txt 2>&1
for by in arm framing source; do
  "$QWEN_PY" compare_dual.py --by "$by" --out "comparisons/compare_$by.json" \
    >> logs_batch/report.txt 2>&1 || note "COMPARE_FAILED $by"
done
note "FINALIZE_DONE see logs_batch/report.txt"
echo "$(date -Is) FINALIZE_DONE" >> logs_batch/batch.log
