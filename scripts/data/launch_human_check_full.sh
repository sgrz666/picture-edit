#!/usr/bin/env bash
# Launch full-scale sequential human-check generation into human-check-full.
set -euo pipefail

REPO_DIR="/home/shangguanrz/project/pic-edit"
PYTHON_BIN="/home/shangguanrz/miniconda3/envs/deepgen/bin/python"
SCRIPT_PATH="${REPO_DIR}/scripts/data/generate_human_check_full.py"
OUT_DIR="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_hidden_hand_v2_staging/human-check-full"
LOG_DIR="${OUT_DIR}/../logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/human_check_full_${TIMESTAMP}.log"

echo "=========================================================="
echo " Starting Sequential human-check-full Generator"
echo " Repo:      ${REPO_DIR}"
echo " Script:    ${SCRIPT_PATH}"
echo " Target:    ${OUT_DIR} (FLAT)"
echo " Log file:  ${LOG_FILE}"
echo " Order:     Sorted by clip_id (starting 000002) -> frame_id"
echo "=========================================================="

nohup "${PYTHON_BIN}" -u "${SCRIPT_PATH}" \
    --out-dir "${OUT_DIR}" \
    --watch \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${OUT_DIR}/human_check_full.pid"
echo "Daemon launched successfully with PID: ${PID}"
echo ""
echo "To monitor live logs in real-time:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To inspect live status JSON:"
echo "  cat ${OUT_DIR}/progress.json"
echo "=========================================================="
