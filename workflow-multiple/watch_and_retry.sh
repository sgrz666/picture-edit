#!/usr/bin/env bash
# Wait for the batch to finish, then sweep the runs it lost to transient failures.
#
# Written as a file rather than an inline ssh command: the earlier one-liner used `while ! grep`,
# and the `!` reached grep as an argument, so the watcher fell straight through to the retry
# sweep. The retry script's own lock guard is what kept that harmless.
set -u
cd /home/shangguanrz/project/pic-edit/workflow-multiple || exit 1

until grep -q BATCH_DONE logs_batch/batch.log; do
  sleep 300
done

echo "$(date -Is) WATCHER batch done, starting retry sweep" >> logs_batch/watcher.log
./retry_dual.sh
