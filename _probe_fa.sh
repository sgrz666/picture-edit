B=/home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body
echo "@@@@@ four_arm_eval_50_s40/summary.json @@@@@"
python3 -m json.tool $B/four_arm_eval_50_s40/summary.json 2>&1 | head -200
echo "@@@@@ run_manifest (s40) @@@@@"
head -c 1200 $B/four_arm_eval_50_s40/run_manifest.json 2>/dev/null
echo; echo "@@@@@ full_20k/run_manifest.json @@@@@"
python3 -m json.tool $B/full_20k/run_manifest.json 2>&1
echo "@@@@@ full_20k tail metrics @@@@@"
tail -n 2 $B/full_20k/train_metrics.jsonl
echo "@@@@@ full_20k ls @@@@@"
ls -la $B/full_20k
