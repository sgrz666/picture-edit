cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== conda envs ==="
ls ~/miniconda3/envs 2>/dev/null || ls ~/anaconda3/envs 2>/dev/null; which python3; python3 -V
echo "=== data/ ==="
ls -la data/; du -sh --si data/*
echo "=== champ_overfit ==="
find data/champ_overfit -maxdepth 2 -printf '%y %s %TY-%Tm-%Td %p\n' | head -40
for d in depth normal dwpose semantic_map; do echo "  -- $d count: $(ls data/champ_overfit/$d 2>/dev/null | wc -l)"; ls data/champ_overfit/$d 2>/dev/null | head -5; done
echo "=== debug_samples ==="
ls -la data/debug_samples/ data/debug_samples/controls data/debug_samples/images
echo "=== iper_debug_001 ==="
ls -la data/iper_debug_001/ | head -30
echo "=== inputs ==="
ls -la inputs/
echo "=== caches ==="
ls -la caches/; du -sh --si caches
echo "=== outputs ==="
ls -la outputs/; du -sh --si outputs
echo "=== logs ==="
ls -la logs/; echo "-- legacy count:"; ls logs/legacy | wc -l; ls -la logs/legacy | head -60
