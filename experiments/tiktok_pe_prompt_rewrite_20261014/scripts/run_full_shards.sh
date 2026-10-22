#!/bin/bash
# Full pass over all 10,850 pairs with K co-resident workers per split (default K=2).
# Resumable: a pair with records/<split>/<pair_id>.json status=complete is skipped,
# so restarting never throws away finished work.
#
# Each split is re-passed until it is actually complete. The first version advanced to
# the next split as soon as no worker was running -- if one of two workers died, the
# surviving worker's shard finished, the count hit zero, and the dead worker's remaining
# pairs were silently never redone (up to half a split lost on a 15-day run).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/home/shangguanrz/nihaochen/envs/qwen21/bin/python}
PAIRS_ROOT=${PAIRS_ROOT:-/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1}
K=${K:-2}
ARGS=${ARGS:---no-thinking}
MAX_PASSES=${MAX_PASSES:-40}

count_complete() {
    grep -l 'status.*complete' "records/$1"/*.json 2>/dev/null | wc -l
}

# Default order puts the 1,156-pair test split first: it answers "does a detailed prompt
# actually beat the fixed one?" after ~2 days instead of ~19.
for split in ${SPLITS:-test val train}; do
    want=$(wc -l < "$PAIRS_ROOT/${split}_pairs.jsonl")
    pass=0
    while :; do
        pass=$((pass + 1))
        have=$(count_complete "$split")
        echo "=== SPLIT $split pass=$pass complete=$have/$want K=$K $(date -Is) ==="
        if [ "$have" -ge "$want" ]; then
            echo "SPLIT_COMPLETE $split $have/$want"
            break
        fi
        if [ "$pass" -gt "$MAX_PASSES" ]; then
            echo "SPLIT_INCOMPLETE $split stuck at $have/$want after $pass passes -- see logs"
            break
        fi
        ARGS="$ARGS" bash scripts/run_shards.sh "$split" "$K" || { echo "LAUNCH_FAILED $split"; sleep 60; continue; }
        while :; do
            sleep 60
            ALIVE=$(pgrep -cf "[p]e_rewrite_tiktok.py run --split $split" || true)
            echo "$(date +%H:%M) split=$split alive=$ALIVE complete=$(count_complete "$split")/$want"
            [ "${ALIVE:-0}" -eq 0 ] && break
        done
        sleep 5
    done
done

$PY -B scripts/pe_rewrite_tiktok.py assemble
$PY -B scripts/pe_rewrite_tiktok.py qc --count 24
echo "=== FULL_DONE $(date -Is) ==="
