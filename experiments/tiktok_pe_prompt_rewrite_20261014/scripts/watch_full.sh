#!/bin/bash
# Keep the multi-day prompt rewrite alive unattended.
#   nohup bash scripts/watch_full.sh > logs/watch_full.log 2>&1 &
# Stop it any time: touch STOP  (it also stops itself once FULL_DONE appears).
# The full run is resumable, so a restart never redoes a completed pair.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
LOG=logs/full_run.log
RESTARTS=0
MAX_RESTARTS=${MAX_RESTARTS:-6}
while :; do
    sleep 600
    if [ -f STOP ]; then echo "$(date -Is) STOP present, watchdog exits"; exit 0; fi
    if grep -q "FULL_DONE" "$LOG" 2>/dev/null; then echo "$(date -Is) FULL_DONE seen, watchdog exits"; exit 0; fi
    WRAP=$(pgrep -cf "run_full_s[h]ards" || true)
    WORK=$(pgrep -cf "[p]e_rewrite_tiktok.py run" || true)
    DONE=$(ls records/*/*.json 2>/dev/null | wc -l)
    echo "$(date -Is) wrapper=$WRAP workers=$WORK records=$DONE restarts=$RESTARTS"
    if [ "${WRAP:-0}" -eq 0 ]; then
        if [ "$RESTARTS" -ge "$MAX_RESTARTS" ]; then
            echo "$(date -Is) GIVING_UP after $RESTARTS restarts -- look at $LOG, then rm STOP to resume"
            touch STOP
            exit 1
        fi
        RESTARTS=$((RESTARTS + 1))
        echo "$(date -Is) RESTART #$RESTARTS (resumable, skips completed pairs)"
        K=${K:-2} setsid nohup bash scripts/run_full_shards.sh >> "$LOG" 2>&1 < /dev/null &
    fi
done
