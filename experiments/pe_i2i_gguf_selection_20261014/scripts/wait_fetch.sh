#!/bin/bash
# Wait for fetch_llama_runtime.sh to finish, then print what we actually got.
set -uo pipefail
L=/data/shangguanrz/tools/llama-cpp/b11617-cuda12.8
M=/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I-GGUF
LOG=$L/.downloads/fetch.log
for _ in $(seq 1 120); do
    if grep -q FETCH_DONE "$LOG" 2>/dev/null; then break; fi
    if ! pgrep -f "[f]etch_llama_runtime.sh" >/dev/null; then echo "FETCH_PROCESS_GONE"; break; fi
    sleep 20
done
echo "=== tail fetch.log"; tail -14 "$LOG" 2>/dev/null
echo "=== weights"; ls -l "$M" 2>/dev/null
echo "=== sha"; cat "$M/SHA256SUMS" 2>/dev/null
echo "=== binaries"; find "$L" -maxdepth 3 -type f \( -name 'llama-server' -o -name 'llama-cli' -o -name 'llama-perplexity' \) 2>/dev/null | head
echo "=== version"; V=$(find "$L" -maxdepth 3 -type f -name llama-server | head -1)
if [[ -n ${V:-} ]]; then LD_LIBRARY_PATH="$L"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "$V" --version 2>&1 | tail -3; else echo "NO llama-server"; fi
