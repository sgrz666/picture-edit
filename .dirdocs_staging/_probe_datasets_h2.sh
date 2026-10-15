H=/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1
echo "== summary.json"; cat $H/summary.json
echo "== split.json"; cat $H/split.json
echo "== geometry_1500_summary"; cat $H/geometry_1500_summary.json
echo "== raw listing"; ls $H/raw; echo "pair dirs: $(find $H/raw -maxdepth 1 -type d | wc -l)"
echo "== raw one sequence tree"; find $H/raw/pair01 -maxdepth 3 -printf '%y %s %p\n' | sort -k3 | head -30
echo "== raw per-camera listing"; ls $H/raw/pair01/fight01; ls $H/raw/pair01/fight01/$(ls $H/raw/pair01/fight01 | head -1)
echo "== counts of npz"; find $H/raw -name '*.npz' | wc -l; find $H/raw -type d -name images | wc -l
echo "== images subdir count of files"
C=$(find $H/raw -type d -name images | head -1); echo $C; find $C -type f | wc -l; ls $C | head -3
echo "== processed_512_1500 top"; ls $H/processed_512_1500; find $H/processed_512_1500 -maxdepth 2 -type d | wc -l
echo "== processed one seq tree depth2"
find $H/processed_512_1500/pair01 -maxdepth 2 -printf '%y %s %p\n' | sort -k3 | head -25
