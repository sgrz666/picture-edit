set -x
R=/home/shangguanrz/project/pic-edit
echo ==RESULTS-ROOT-FILES
find $R/results -maxdepth 1 -type f | wc -l
echo ==LEGACY-DEMO
ls -la $R/results/legacy_demo_202609 | head -40
find $R/results/legacy_demo_202609 -maxdepth 1 -type f ! -name '目录说明.md' ! -name 'README.md' | wc -l
echo ==REVIEW1
find $R/results/review_20260908 -maxdepth 1 -printf '%y %l %s %p\n'
ls -la $R/results/review_20260908/configs | head
echo ==REVIEW1-RUNS
ls -la $R/results/review_20260908/runs
echo ==REVIEW1-SUBDIR-OF-RUN
ls $R/results/review_20260908/runs | head -2; for d in $(ls $R/results/review_20260908/runs | head -1); do find $R/results/review_20260908/runs/$d -maxdepth 2 | head -25; done
echo ==REVIEW2
find $R/results/review_20260908_v2 -maxdepth 1 -printf '%y %l %s %p\n'
echo ==REVIEW2-COUNTS
for d in cache configs review_full review_sheets runs; do echo "$d $(ls $R/results/review_20260908_v2/$d 2>/dev/null | wc -l)"; done
echo ==REVIEW2-DU
du -sh --si $R/results/review_20260908_v2/*
echo ==REVIEW2-EXP-LINK
ls -la $R/experiments/ | grep -i review
readlink $R/experiments/review_20260908 $R/experiments/review_20260908_v2 2>/dev/null
echo ==DATA-V2
ls /data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/review_20260908_v2/ 2>/dev/null | head
echo ==CONFIGS-SAMPLE
ls $R/results/review_20260908_v2/configs | head -8
ls $R/results/review_20260908_v2/configs | tail -4
head -c 900 $R/results/review_20260908_v2/configs/$(ls $R/results/review_20260908_v2/configs | head -1)
echo
echo ==RUNS-SAMPLE
ls $R/results/review_20260908_v2/runs | head -5
D1=$(ls $R/results/review_20260908_v2/runs | head -1); find $R/results/review_20260908_v2/runs/$D1 -maxdepth 2 | head -30
echo ==ACCEPTANCE
head -c 1200 $R/results/review_20260908_v2/ACCEPTANCE.json 2>/dev/null
echo ==REVIEW2-README
head -60 $R/results/review_20260908_v2/README.md 2>/dev/null
echo ==REVIEW-FULL-SHEETS
ls $R/results/review_20260908_v2/review_full | head; ls $R/results/review_20260908_v2/review_sheets | head
ls $R/results/review_20260908_v2/cache | head
