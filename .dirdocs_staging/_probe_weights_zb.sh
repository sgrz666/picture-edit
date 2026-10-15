cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== TOP LEVEL ==="
ls -la
echo "=== models/ ==="
ls -la models/
echo "=== models du ==="
du -sh --si models/* 2>/dev/null
echo "=== model-qwen-edit ==="
ls -lad model-qwen-edit 2>/dev/null; readlink -f model-qwen-edit 2>/dev/null
