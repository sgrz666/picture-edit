#!/usr/bin/env bash
# Run every test-set pair through the workflow, then score each finished run.
# A failing pair is reported and skipped so the rest of the set still completes.
set -uo pipefail
workflow_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$workflow_dir"

QWEN_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
GEOMETRY_PY="$("$QWEN_PY" -c 'import json;print(json.load(open("config.json"))["geometry_python"])')"
PAIRS="${1:-../experiments/qwen-2.1-test/data/testset/pairs}"
RESOLUTION="${RESOLUTION:-768}"
started_all=$(date +%s)

for directory in "$PAIRS"/*/; do
    [ -f "${directory}A_source.png" ] || continue
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
"$QWEN_PY" report.py
