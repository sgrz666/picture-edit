R=/home/shangguanrz/project/pic-edit
S=$R/datasets/iPER/iper_sampled_6src64tgt
echo "== geometry dir of 001_1"
ls $S/001_1/geometry | head; find $S/001_1/geometry -type f | wc -l
echo "== _STAGE1_DONE"; cat $S/024_6/_STAGE1_DONE 2>/dev/null | head -c 300; echo
echo "== candidates_source.json head"; head -c 600 $S/024_6/candidates_source.json; echo
echo "== counts across all appearances: which have smplestx params.pt"
find $S -maxdepth 3 -name 'params.pt' | wc -l
find $S -maxdepth 3 -name '_SMPLX_DONE' | wc -l
echo "== which appearances lack geometry dir"
for a in $(ls $S | grep -E '^[0-9]'); do [ -d "$S/$a/geometry" ] || echo -n "$a "; done; echo
echo "== total png in sampled"
find $S -name '*.png' | wc -l
echo "== Hi4D top"
ls -la $R/datasets/Hi4D_pilot_v1
du -sh --si $R/datasets/Hi4D_pilot_v1/* 2>/dev/null
