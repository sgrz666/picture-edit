R=/home/shangguanrz/project/pic-edit
echo "== scripts mentioning iper"
ls $R/scripts | head -60
echo "----"
grep -rl -i -e iper_assets_512 -e iper_sampled -e manifest_512 $R/scripts $R/src 2>/dev/null | head -30
echo "== logs audit tail"
tail -n 25 $R/datasets/iPER/logs/audit_512.out
echo "== logs build tail"
tail -n 25 $R/datasets/iPER/logs/build_assets_512_nvdiffrast_full.out
echo "== logs build head"
head -n 20 $R/datasets/iPER/logs/build_assets_512_nvdiffrast_full.out
echo "== logs count lines"
wc -l $R/datasets/iPER/logs/*.out
