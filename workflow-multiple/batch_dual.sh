#!/usr/bin/env bash
# Batch the two-person test set: every pair x {with_b, skeleton_only} x {b_canvas, source_fit}.
#
# Runs are named <pair>.<arm>.<framing> so report_dual.py can group them without parsing logs.
# One run failing must not stop the rest; each step's own log is kept under logs_batch/.
set -u
cd "$(dirname "$0")"

QWEN_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
LOG_DIR=logs_batch
mkdir -p "$LOG_DIR"
INDEX="$LOG_DIR/index.tsv"
OVERALL="$LOG_DIR/batch.log"

# Guard against a second launch: run_dual.py serialises on .workflow.lock, so a duplicate batch
# would silently queue hours of work behind the one already running.
if ! flock -n .workflow.lock -c true; then
  echo "$(date -Is) REFUSING_TO_START the workflow lock is held by another run" >> "$OVERALL"
  exit 1
fi

if [ -f "$INDEX" ]; then
  echo "$(date -Is) RESUMING, $(wc -l < "$INDEX") runs already recorded in $INDEX" >> "$OVERALL"
fi

record() {  # name status seconds
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -Is)" >> "$INDEX"
}

done_run() {  # name
  awk -F'\t' -v n="$1" '$1==n && $2=="ok" {found=1} END {exit !found}' "$INDEX" 2>/dev/null
}

run_one() {  # pair_dir name arm framing
  local pair_dir="$1" name="$2" arm="$3" framing="$4"
  if done_run "$name"; then
    echo "$(date -Is) SKIP $name (already ok)" >> "$OVERALL"
    return 0
  fi
  local started=$SECONDS log="$LOG_DIR/$name.log"
  echo "$(date -Is) START $name arm=$arm framing=$framing" >> "$OVERALL"
  if "$QWEN_PY" run_dual.py --pair "$pair_dir" --name "$name" \
       --framing "$framing" --generator-references "$arm" > "$log" 2>&1; then
    "$QWEN_PY" evaluate_dual.py --run "$name" >> "$log" 2>&1 \
      && record "$name" ok $((SECONDS - started)) \
      || record "$name" evaluate_failed $((SECONDS - started))
  else
    record "$name" workflow_failed $((SECONDS - started))
  fi
  echo "$(date -Is) END $name status=$(awk -F'\t' -v n="$name" '$1==n{print $2}' "$INDEX" | tail -1) \
elapsed=$((SECONDS - started))s" >> "$OVERALL"
}

for base in pairs pairs_cross; do
  for pair_dir in "$base"/*/; do
    [ -f "$pair_dir/pair.json" ] || continue
    pair=$(basename "$pair_dir")
    # with_b is only paired with source_fit for the same-identity set; the cross-pair hard cases
    # keep the two arms at b_canvas, where the single existing failure was reproduced.
    if [ "$base" = "pairs" ]; then
      run_one "$pair_dir" "$pair.with_b.b_canvas" with_b b_canvas
    fi
    run_one "$pair_dir" "$pair.skeleton_only.b_canvas" skeleton_only b_canvas
    run_one "$pair_dir" "$pair.skeleton_only.source_fit" skeleton_only source_fit
  done
done

"$QWEN_PY" report_dual.py >> "$OVERALL" 2>&1 || echo "$(date -Is) REPORT_FAILED $?" >> "$OVERALL"
echo "$(date -Is) BATCH_DONE" >> "$OVERALL"
