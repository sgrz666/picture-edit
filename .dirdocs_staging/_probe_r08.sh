#!/bin/bash
R=/home/shangguanrz/project/pic-edit/experiments/review_20260908
for d in $R/runs/*/; do
  echo "=================== $(basename $d)"
  echo "--- summary.json ---"; cat $d/summary.json 2>/dev/null
  echo "--- events.jsonl ---"; cat $d/events.jsonl 2>/dev/null
  echo "--- environment.json ---"; cat $d/environment.json 2>/dev/null
  echo "--- gpu_samples.csv head3/tail2 ---"; head -3 $d/gpu_samples.csv 2>/dev/null; tail -2 $d/gpu_samples.csv 2>/dev/null
  echo "--- rows ---"; wc -l $d/gpu_samples.csv 2>/dev/null
  echo "--- stderr tail ---"; tail -n 12 $d/stderr.log 2>/dev/null
done
