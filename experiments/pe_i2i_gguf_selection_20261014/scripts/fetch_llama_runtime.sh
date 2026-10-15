#!/bin/bash
# Fetch llama.cpp's CUDA runtime and mozophe's PE-I2I MTP GGUF weights onto the data disk.
# Resumable, size-verified, checksums recorded in a manifest, idempotent (verified files skip).
# The / partition is 95% full, so everything lands under /data.
set -uo pipefail

TAG=b11617
LLAMA_DIR=/data/shangguanrz/tools/llama-cpp/${TAG}-cuda12.8
MODEL_DIR=/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I-GGUF
DL=${LLAMA_DIR}/.downloads
LOG=${DL}/fetch.log

Q8_NAME=qwen3.5_9b_qwen_image_2.1_pe_i2i.mtp.Q8_0.gguf
Q8_SIZE=9786060352
MMP_NAME=qwen3.5_9b_qwen_image_2.1_pe_i2i.mmproj.bf16.gguf
MMP_SIZE=921704640

mkdir -p "$LLAMA_DIR" "$MODEL_DIR" "$DL"
say () { echo "$(date -Is) $*" | tee -a "$LOG"; }

size_ok () {  # size_ok <file> <expected>
    [[ -f $1 ]] && [[ $(stat -c %s "$1") -eq $2 ]]
}

pull () {  # pull <url> <dest> <expected_size>   (expected_size 0 = unknown, skip the check)
    local url=$1 dest=$2 want=$3 got round=0
    if [[ $want -gt 0 ]]; then
        if size_ok "$dest" "$want"; then say "SKIP (already complete) $(basename "$dest")"; return 0; fi
    elif [[ -s $dest ]]; then
        say "SKIP (present, size unknown) $(basename "$dest")"; return 0
    fi
    # curl's default retry set does NOT cover "curl: (18) partial file", which is exactly what
    # a long HF/LFS transfer hits here -- so resume in an outer loop until the size matches.
    while (( round < 40 )); do
        round=$(( round + 1 ))
        say "GET (round $round) $url"
        curl -sSL --retry 10 --retry-delay 5 --retry-all-errors \
             --speed-limit 65536 --speed-time 60 --max-time 3600 \
             -C - -o "$dest" "$url" >>"$LOG" 2>&1
        got=$(stat -c %s "$dest" 2>/dev/null || echo 0)
        if [[ $want -eq 0 ]]; then
            [[ $got -gt 0 ]] && { say "OK $(basename "$dest") $got bytes (size not asserted)"; return 0; }
        elif (( got == want )); then
            say "OK $(basename "$dest") $got bytes"; return 0
        else
            say "  resumed to $got/$want bytes, retrying"
        fi
        sleep 5
    done
    say "GIVE UP $(basename "$dest") after $round rounds (got $(stat -c %s "$dest" 2>/dev/null || echo 0)/${want})"
    return 1
}

pull "https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/llama-${TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"  "${DL}/llama.tar.gz"  0 || true
pull "https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/cudart-llama-${TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz" "${DL}/cudart.tar.gz" 0 || true
pull "https://huggingface.co/mozophe/Qwen-Image-2.1-PE-MTP-GGUF/resolve/main/${Q8_NAME}"  "${MODEL_DIR}/${Q8_NAME}"  "$Q8_SIZE" || exit 1
pull "https://huggingface.co/mozophe/Qwen-Image-2.1-PE-MTP-GGUF/resolve/main/${MMP_NAME}" "${MODEL_DIR}/${MMP_NAME}" "$MMP_SIZE" || exit 1

# tarballs: size unknown up front, so just require non-empty and extractable
for t in llama cudart; do
    [[ -s ${DL}/${t}.tar.gz ]] || { say "MISSING ${t}.tar.gz"; exit 1; }
    tar -tzf "${DL}/${t}.tar.gz" >/dev/null 2>&1 || { say "NOT A VALID TAR ${t}.tar.gz"; exit 1; }
    tar -xzf "${DL}/${t}.tar.gz" -C "$LLAMA_DIR"
    say "extracted ${t}.tar.gz"
done

sha256sum "${MODEL_DIR}/${Q8_NAME}" "${MODEL_DIR}/${MMP_NAME}" | tee "${MODEL_DIR}/SHA256SUMS" >>"$LOG"
say "manifest -> ${MODEL_DIR}/SHA256SUMS"

SERVER=$(find "$LLAMA_DIR" -type f -name llama-server | head -1)
CLI=$(find "$LLAMA_DIR" -type f -name llama-cli | head -1)
say "llama-server: ${SERVER:-NOT FOUND}"
say "llama-cli:    ${CLI:-NOT FOUND}"
"${SERVER:-false}" --version 2>&1 | tail -2 | tee -a "$LOG"
say "FETCH_DONE"
