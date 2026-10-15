D=/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast
echo "== top level"
find $D -maxdepth 1 -printf '%y %s %p\n' | sort -k3 | head -20
echo "== dir count"; find $D -maxdepth 1 -type d | wc -l; echo "== file count"; find $D -maxdepth 1 -type f | wc -l
echo "== listing"
ls $D | head -12
echo "== one asset dir tree"
find $D/024_6 -maxdepth 2 -printf '%y %s %p\n' | sort -k3 | head -70
echo "== build_summary"
find $D -maxdepth 1 -name '*.json*' -o -maxdepth 1 -name '*.jsonl' | head
cat $D/build_summary.json 2>/dev/null | head -c 3000
