set -x
R=/home/shangguanrz/project/pic-edit
V=$R/results/review_20260908_v2
echo ==V1-README
head -30 $R/results/review_20260908/README.md
echo ==V1-PREFLIGHT
head -c 700 $R/results/review_20260908/preflight.json
echo
echo ==V1-RUN-DIR
find $R/results/review_20260908/runs/Q0_S0_42_letterbox_35_4.5 -maxdepth 2 | head -20
echo ==V1-QUALITY-HEAD
head -c 700 $R/results/review_20260908/quality_manifest.json
echo
echo ==V1-CONFIG
cat $R/results/review_20260908/configs/Q0_S0_42_letterbox_35_4.5.json
echo ==V2-CACHE
ls -la $V/cache
echo ==V2-RUN-SAMPLE
find $V/runs/T_S_512_cache_1_D0 -maxdepth 2 | head -25
find $V/runs/ADAPTER_INFER_512_S0 -maxdepth 2 | head -25
echo ==V2-CONFIG-SAMPLE
cat $V/configs/T_S_512_cache_1_D0.json
cat $V/configs/CACHE_1024_D0_8a671c3c47.json
echo ==V2-CONFIG-KINDS
ls $V/configs | sed 's/_[0-9].*//; s/\.json//' | sort | uniq -c | sort -rn | head -20
echo ==V2-REVFULL
ls -la $V/review_full | head
echo ==V2-SHEETS
ls -la $V/review_sheets | head -25
echo ==V2-SUMMARY-HEAD
head -3 $V/summary.csv
wc -l $V/summary.csv
echo ==V2-CAPACITY
cat $V/capacity.json
echo ==V2-REPRO
cat $V/reproducibility.json
echo ==V2-MEM-REPORT
head -40 $V/MEMORY_REPORT.md
echo ==V2-CONCL
head -50 $V/EXPERIMENT_CONCLUSION.md
