#!/usr/bin/env bash
# Re-run what the batch lost, then report.
#
# This host's driver occasionally kills a CUDA context with an NVML assert inside torch, which
# shows up as a workflow_failed run unrelated to the pipeline itself. Because the geometry and
# prompt stages stay cached, a retry costs only the generation step, so sweeping the failures is
# cheaper than restarting the batch.
set -u
cd "$(dirname "$0")"

QWEN_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
LOG_DIR=logs_batch
INDEX="$LOG_DIR/index.tsv"
OVERALL="$LOG_DIR/batch.log"
MAX_SWEEPS=${MAX_SWEEPS:-3}

if ! flock -n .workflow.lock -c true; then
  echo "$(date -Is) REFUSING_TO_START the workflow lock is held by another run" >> "$OVERALL"
  exit 1
fi

record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -Is)" >> "$INDEX"; }

for sweep in $(seq 1 "$MAX_SWEEPS"); do
  mapfile -t pending < <(awk -F'\t' '{last[$1]=$2} END {for (name in last) if (last[name] != "ok") print name}' "$INDEX" | sort)
  if [ ${#pending[@]} -eq 0 ]; then
    echo "$(date -Is) RETRY_SWEEP $sweep nothing pending" >> "$OVERALL"
    break
  fi
  echo "$(date -Is) RETRY_SWEEP $sweep pending=${#pending[@]}: ${pending[*]}" >> "$OVERALL"
  for name in "${pending[@]}"; do
    framing=${name##*.}
    arm=${name%.*}; arm=${arm##*.}
    pair=${name%.*.*}
    pair_dir=pairs/$pair
    [ -d "$pair_dir" ] || pair_dir=pairs_cross/$pair
    if [ ! -f "$pair_dir/pair.json" ]; then
      echo "$(date -Is) RETRY_SKIP $name no pair directory found for $pair" >> "$OVERALL"
      continue
    fi
    started=$SECONDS
    log="$LOG_DIR/retry$sweep.$name.log"
    if "$QWEN_PY" run_dual.py --pair "$pair_dir" --name "$name" \
         --framing "$framing" --generator-references "$arm" >> "$log" 2>&1; then
      "$QWEN_PY" evaluate_dual.py --run "$name" >> "$log" 2>&1 \
        && record "$name" ok $((SECONDS - started)) \
        || record "$name" evaluate_failed $((SECONDS - started))
    else
      record "$name" workflow_failed $((SECONDS - started))
    fi
  done
done

"$QWEN_PY" report_dual.py > "$LOG_DIR/report.txt" 2>&1 || echo "$(date -Is) REPORT_FAILED $?" >> "$OVERALL"
echo "$(date -Is) RETRY_DONE sweeps=$MAX_SWEEPS report=$LOG_DIR/report.txt" >> "$OVERALL"
