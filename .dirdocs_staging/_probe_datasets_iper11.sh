R=/home/shangguanrz/project/pic-edit
echo "== split script docstring"
head -40 $R/scripts/data/split_iper_train_val.py
echo "== who made dev_train/baseline/showcase splits"
grep -rln "dev_train_pairs\|baseline_3000_microstep_pairs\|showcase_20k_diverse" $R/scripts $R/add/tools/SMPLest-X/scripts $R/src 2>/dev/null | grep -v pycache | head -15
echo "== iper_raw_1024 vs add/data/iper_raw_1024"
ls -la $R/datasets/iPER/iper_raw_1024
ls -la $R/add/data/ | head -20
echo "== compare one video inode"
stat -c '%h %i %s %n' $R/datasets/iPER/iper_raw_1024/iPER_1024_video_release/001_1_1.mp4 $R/add/data/iper_raw_1024/iPER_1024_video_release/001_1_1.mp4
echo "== count videos both places"
find $R/datasets/iPER/iper_raw_1024/iPER_1024_video_release -name '*.mp4' | wc -l
find $R/add/data/iper_raw_1024/iPER_1024_video_release -name '*.mp4' 2>/dev/null | wc -l
echo "== video dims probe"
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
ls $R/datasets/iPER_sample
