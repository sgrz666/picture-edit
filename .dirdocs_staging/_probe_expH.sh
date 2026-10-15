#!/bin/bash
# probe expH part 1: experiments level + repo symlinks
echo "=== REMOTE HOME ==="
ls -la /home/shangguanrz | head -40
echo "=== find repo root candidates ==="
ls -d /home/shangguanrz/*/ 2>/dev/null
