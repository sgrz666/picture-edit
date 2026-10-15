set -e
B=/home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body
echo "===== TOP LS ====="; ls -la $B
echo "===== SUBDIR COUNT ====="; find $B -maxdepth 1 -type d | wc -l
echo "===== SIZES ====="; du -sh --si $B/* 2>/dev/null
