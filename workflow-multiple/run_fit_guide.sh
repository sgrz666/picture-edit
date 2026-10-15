#!/usr/bin/env bash
# One fit-guided twin of a ground-truth-guided run: same pair, same arm, same framing, only the
# guide source differs. This is the unit of work behind the --by source comparison (WP7).
#
# It is a separate script rather than an inline ssh command because the inline form lost its
# variable to shell quoting and reported "run_dual.py: command not found" instead of running.
set -u
cd "$(dirname "$0")"

QWEN_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
pair=${1:?usage: run_fit_guide.sh <pair-dir-name> [arm] [framing]}
arm=${2:-skeleton_only}
framing=${3:-b_canvas}
pair_dir=pairs/$pair
[ -d "$pair_dir" ] || pair_dir=pairs_cross/$pair
if [ ! -f "$pair_dir/pair.json" ]; then
  echo "no pair directory named $pair under pairs/ or pairs_cross/" >&2
  exit 1
fi
if [ ! -f "fits/$pair.npz" ]; then
  echo "no fit for $pair; run fit_dual_report.py first" >&2
  exit 1
fi

name="$pair.$arm.$framing.fit"
mkdir -p logs_batch
log="logs_batch/fit.$name.log"

# run_dual.py does not queue: its file lock is non-blocking and it exits with "another workflow
# invocation holds the lock". Since these runs are meant to fill the gaps of a batch that is
# already running, wait for the lock instead of failing on it.
wait_for_lock() {
  local waited=0
  while ! flock -n .workflow.lock -c true; do
    if [ "$waited" -eq 0 ]; then
      echo "$(date -Is) WAIT $name (batch holds the lock)" >> logs_batch/fit.log
      waited=1
    fi
    sleep 60
  done
}

wait_for_lock
echo "$(date -Is) START $name" >> logs_batch/fit.log
if "$QWEN_PY" run_dual.py --pair "$pair_dir" --name "$name" --framing "$framing" \
     --generator-references "$arm" --keypoints-source fit > "$log" 2>&1; then
  "$QWEN_PY" evaluate_dual.py --run "$name" >> "$log" 2>&1 \
    && echo "$(date -Is) OK $name" >> logs_batch/fit.log \
    || echo "$(date -Is) EVALUATE_FAILED $name" >> logs_batch/fit.log
else
  echo "$(date -Is) WORKFLOW_FAILED $name" >> logs_batch/fit.log
fi
