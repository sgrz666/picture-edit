#!/bin/bash
# Launch K co-resident PE workers over one split, each taking rows[i::K].
# The GPU is shared with people; check free memory before raising K.
#   bash scripts/run_shards.sh train 3            # 3 workers, all train pairs
#   ARGS="--max-image-side 512" bash scripts/run_shards.sh train 2
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
SPLIT=${1:?usage: run_shards.sh <split> <count>}
COUNT=${2:?usage: run_shards.sh <split> <count>}
ARGS=${ARGS:---no-thinking}
mkdir -p logs
# Two launches on one card silently stack: on 2026-10-14 17:47 a wrapper that lost its
# workers advanced to the next split and started a second pair while a new wrapper was
# already launching, leaving 4 processes on a card sized for 2.
if pgrep -f "[p]e_rewrite_tiktok.py run" > /dev/null; then
    echo "REFUSING: PE workers are already running:"
    pgrep -af "[p]e_rewrite_tiktok.py run"
    echo 'stop them first: pkill -9 -f "pe_rewrite_tikto[k]"'
    exit 1
fi
for index in $(seq 0 $((COUNT - 1))); do
    LOG=logs/${SPLIT}_shard${index}_$(date +%Y%m%d_%H%M%S).log
    echo "start shard $index/$COUNT -> $LOG"
    setsid nohup $PY -B scripts/pe_rewrite_tiktok.py run --split "$SPLIT" \
        --shard-index "$index" --shard-count "$COUNT" $ARGS > "$LOG" 2>&1 < /dev/null &
done
echo "LAUNCHED $COUNT workers for split=$SPLIT; watch logs/${SPLIT}_shard*.log"
