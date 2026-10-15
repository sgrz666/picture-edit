cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== caches symlink target ==="
ls -la caches/qwen21_hi4d_controls/ 2>&1 | head -15
du -sh --si caches/qwen21_hi4d_controls/ 2>/dev/null; find caches/qwen21_hi4d_controls/ -maxdepth 1 -type f 2>/dev/null | wc -l
echo "=== /data pic-edit-caches ==="
ls -la /data/shangguanrz/pic-edit-caches/ 2>&1 | head -20
du -sh --si /data/shangguanrz/pic-edit-caches/* 2>/dev/null | head -20
echo "=== /data top ==="
ls -la /data/shangguanrz/ 2>&1
echo "=== /data pic-edit-models ==="
ls -la /data/shangguanrz/pic-edit-models/ 2>/dev/null | head -20
echo "=== archive cleanup models dir ==="
ls -la /data/shangguanrz/pic-edit-archive/cleanup_20261003/models/Qwen-Image-Edit-20B/ 2>&1 | head -25
du -sh --si /data/shangguanrz/pic-edit-archive/cleanup_20261003/models/Qwen-Image-Edit-20B 2>/dev/null
