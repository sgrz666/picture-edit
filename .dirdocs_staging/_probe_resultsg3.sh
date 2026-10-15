set -x
R=/home/shangguanrz/project/pic-edit
V=$R/results/review_20260908_v2
echo ==MEM-REPORT
head -60 $V/MEMORY_REPORT.md
echo ==CONCL
head -70 $V/EXPERIMENT_CONCLUSION.md
echo ==SUMMARY-STATS
python3 -c "
import csv
rows=list(csv.DictReader(open('$V/summary.csv',encoding='utf-8-sig')))
print('rows',len(rows))
from collections import Counter
print('kind',Counter(r['kind'] for r in rows))
print('group',Counter(r['group'] for r in rows))
print('status',Counter(r['status'] for r in rows))
print('res',Counter(r['resolution'] for r in rows))
print('pre',Counter(r['preprocess'] for r in rows))
print('steps',Counter(r['steps'] for r in rows))
"
echo ==RUNS-NAME-PREFIX
ls $V/runs | grep -v 目录 | sed -E 's/^([A-Z0-9]+_.*)/\1/' | awk -F_ '{print $1}' | sort | uniq -c
echo ==CACHE-DIR-CONTENT
find $V/cache/8a671c3c47a682a470a5e59550b0a20fd7ebaefb8a80d5b76ed3567c5d834dd8 -maxdepth 1
echo ==MODEL-INV
head -c 900 $V/model_inventory.json
echo
echo ==BUDGET
cat $V/budget_report.json $V/budget_carryover.json
echo ==EVIDENCE-IDX
cat $V/evidence_index.json
echo ==RUNBOOK
head -50 $V/RUNBOOK.md
echo ==RUN-SUMMARY-SAMPLE
cat $V/runs/T_S_512_cache_1_D0/summary.json 2>/dev/null | head -c 800
echo
echo ==V2-README-HUMAN?
head -30 $V/README.md
echo ==GREP-REFS
grep -rl "results/review_20260908" $R/scripts $R/src $R/docs $R/experiments/README.md $R/reports 2>/dev/null | head
grep -rn "legacy_demo_202609\|demooutput" $R/scripts $R/docs $R/README.md 2>/dev/null | head
grep -rln "archive/cleanup_20261003\|pic-edit-archive" $R/scripts $R/docs 2>/dev/null | head
