#!/usr/bin/env bash
# Run the batch only while the shared GPU has room for a generation step.
#
# This card is shared with other users' long jobs. A 768/30-step generation peaked at 34 GiB and
# failed outright with a driver assert when the free pool dropped below that, so the batch is not
# allowed to keep burning ~10 minutes per run on failures it can predict. The gate pauses the
# batch loop (the in-flight run is left alone to finish or fail on its own terms) and resumes it
# when the memory returns; batch_dual.sh skips every run already recorded ok, so resuming costs
# nothing but cache-hit stage checks.
set -u
cd /home/shangguanrz/project/pic-edit/workflow-multiple || exit 1

QWEN_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
LOG=logs_batch/gate.log
NEED_GIB=40          # measured 34 GiB generation peak plus headroom for the prompt engine
POLL_SECONDS=300

free_gib() {
  "$QWEN_PY" -c 'import torch; print(int(torch.cuda.mem_get_info()[0] // 2**30))' 2>/dev/null
}

note() { echo "$(date -Is) $*" >> "$LOG"; }

while :; do
  if grep -q BATCH_DONE logs_batch/batch.log 2>/dev/null; then
    note "BATCH_ALREADY_DONE gate exiting"
    break
  fi
  free=$(free_gib)
  pid=$(pgrep -f '^bash batch_dual.sh$' | head -1)
  if [ -z "${free:-}" ]; then
    note "GPU_QUERY_FAILED torch could not report memory"
  elif [ -z "$pid" ]; then
    if [ "$free" -ge "$NEED_GIB" ]; then
      note "RESUME free=${free}GiB"
      setsid nohup bash batch_dual.sh >> logs_batch/nohup.out 2>&1 < /dev/null &
      sleep 20
    else
      note "HOLD free=${free}GiB needs ${NEED_GIB}GiB, batch not running"
    fi
  elif [ "$free" -lt "$NEED_GIB" ]; then
    note "PAUSE free=${free}GiB stopping batch loop pid=$pid (the in-flight run continues)"
    kill -TERM "$pid"
  else
    note "OK free=${free}GiB batch running pid=$pid"
  fi
  sleep "$POLL_SECONDS"
done
