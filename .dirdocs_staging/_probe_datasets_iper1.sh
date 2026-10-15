set -x
D=/home/shangguanrz/project/pic-edit/datasets/iPER
ls $D/iper_raw_1024
ls $D/iper_raw_1024/iPER_1024_video_release | head
ls $D/iper_raw_1024/iPER_1024_video_release | wc -l
find $D/iper_raw_1024/iPER_1024_video_release -maxdepth 1 -printf '%y %s %p\n' | head -20
echo ==== sampled splits
ls -la $D/iper_sampled_6src64tgt/splits
for f in $D/iper_sampled_6src64tgt/splits/*; do echo "--- $f"; head -c 400 "$f"; echo; done
echo ==== sampled one item
ls $D/iper_sampled_6src64tgt | grep -c '^0'
find $D/iper_sampled_6src64tgt/024_6 -maxdepth 2 -printf '%y %s %p\n' | sort -k3 | head -80
echo ==== summary.json
cat $D/iper_sampled_6src64tgt/summary.json
cat $D/iper_sampled_6src64tgt/smplestx_stage2_summary.json
echo ==== pairs head
head -c 2000 $D/iper_sampled_6src64tgt/pairs.jsonl; echo
wc -l $D/iper_sampled_6src64tgt/pairs.jsonl $D/iper_sampled_6src64tgt/samples.jsonl
echo ==== assets 512
ls $D/iper_assets_512_nvdiffrast | head -20
ls $D/iper_assets_512_nvdiffrast | wc -l
find $D/iper_assets_512_nvdiffrast -maxdepth 1 -type f -printf '%s %p\n'
