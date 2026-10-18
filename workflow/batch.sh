#!/usr/bin/env bash
# Run every test-set pair through the workflow, then score each finished run.
# A failing pair is reported and skipped so the rest of the set still completes.
set -uo pipefail
workflow_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$workflow_dir"

QWEN_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
GEOMETRY_PY="$("$QWEN_PY" -c 'import json;print(json.load(open("config.json"))["geometry_python"])')"
# experiments/ is flat on the Windows mirror and grouped by family on sg (AGENTS.md 2.1), so the
# default is resolved rather than hardcoded. An explicit argument must exist; no silent fallback.
if [ $# -ge 1 ]; then
    PAIRS="$1"
    [ -d "$PAIRS" ] || { echo "PAIRS_NOT_FOUND $PAIRS"; exit 2; }
else
    PAIRS=""
    for candidate in ../experiments/qwen-2.1-test/data/testset/pairs \
                     ../experiments/2.qwen-test/qwen-2.1-test/data/testset/pairs; do
        [ -d "$candidate" ] && { PAIRS="$candidate"; break; }
    done
    [ -n "$PAIRS" ] || { echo "PAIRS_NOT_FOUND tried both flat and grouped experiments layouts"; exit 2; }
fi
RESOLUTION="${RESOLUTION:-768}"
started_all=$(date +%s)
echo "=== pairs dir $PAIRS"
attempted=0

for directory in "$PAIRS"/*/; do
    [ -f "${directory}A_source.png" ] || continue
    attempted=$(( attempted + 1 ))
    name="$(basename "$directory")"
    started=$(date +%s)
    echo "=== PAIR $name resolution=$RESOLUTION $(date '+%F %T')"
    # A pair whose source photo holds several people cannot be guessed at, so the strict
    # default runs first and only an explicit largest-person choice is retried.
    if bash run.sh --pair "$directory" --resolution "$RESOLUTION"; then
        :
    elif bash run.sh --pair "$directory" --resolution "$RESOLUTION" --subject largest; then
        echo "SUBJECT_FALLBACK $name used --subject largest"
    else
        echo "PAIR_FAILED $name"
        continue
    fi
    "$GEOMETRY_PY" evaluate.py --run "$name" || echo "EVAL_FAILED $name"
    echo "--- $name took $(( $(date +%s) - started ))s"
done

echo "=== batch finished in $(( $(date +%s) - started_all ))s"
[ "$attempted" -gt 0 ] || { echo "BATCH_NO_PAIRS matched 0 directories holding A_source.png under $PAIRS"; exit 2; }
"$QWEN_PY" report.py
