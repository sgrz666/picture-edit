set -x
B=/home/shangguanrz/project/pic-edit/datasets/iPER
ls -la $B/iper_sampled_6src64tgt/ | head -20
find $B/iper_sampled_6src64tgt -maxdepth 1 -type f | head
ls $B/iper_sampled_6src64tgt/splits/ 2>/dev/null
wc -l $B/iper_sampled_6src64tgt/splits/* 2>/dev/null
head -3 $B/iper_sampled_6src64tgt/splits/*.jsonl 2>/dev/null | head -c 2000
head -3 $B/iper_sampled_6src64tgt/splits/*.txt 2>/dev/null | head -c 1000
cat $B/iper_sampled_6src64tgt/summary.json 2>/dev/null | head -c 3000
ls $B/iper_sampled_6src64tgt/*/ -d | wc -l
find $B/iper_sampled_6src64tgt/001_1 -maxdepth 2 | head -40
find $B/iper_sampled_6src64tgt/001_1 -type f | wc -l
du -sh --si $B/iper_sampled_6src64tgt/001_1
