set -u
R=/data/shangguanrz
echo "===== A. dataset_samples direct files ====="
ls -la $R/dataset_samples/
echo "--- MANIFEST.txt (wc -l then full) ---"
wc -l $R/dataset_samples/MANIFEST.txt
cat $R/dataset_samples/MANIFEST.txt
echo "--- shuoming head ---"
wc -l $R/dataset_samples/目录说明.md
head -60 $R/dataset_samples/目录说明.md

echo "===== B. abc_showcase_20261003 level ====="
ls -la $R/dataset_samples/abc_showcase_20261003/
echo "--- du per source ---"
du -sh --si $R/dataset_samples/abc_showcase_20261003/*

echo "===== C. ls -la of the 4 small sources ====="
for d in dna_rendering freeman humanvid mvhumannet; do
  echo "### $d ###"
  ls -la $R/dataset_samples/abc_showcase_20261003/$d/
done

echo "===== D. file(1) on every file in small sources ====="
for d in dna_rendering freeman humanvid mvhumannet; do
  echo "### $d ###"
  find $R/dataset_samples/abc_showcase_20261003/$d -maxdepth 2 -type f -print0 | xargs -0 -r file
done

echo "===== E. pku_dymvhumans level ====="
ls -la $R/dataset_samples/abc_showcase_20261003/pku_dymvhumans/
echo "--- du ---"
du -sh --si $R/dataset_samples/abc_showcase_20261003/pku_dymvhumans/*
echo "--- biggest files ---"
find $R/dataset_samples/abc_showcase_20261003/pku_dymvhumans -type f -printf '%10s %p\n' | sort -k1 -n -r | head -60
echo "--- total file count ---"
find $R/dataset_samples/abc_showcase_20261003/pku_dymvhumans -type f | wc -l
echo "--- extension histogram ---"
find $R/dataset_samples/abc_showcase_20261003/pku_dymvhumans -type f | sed 's/.*\///' | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -30
