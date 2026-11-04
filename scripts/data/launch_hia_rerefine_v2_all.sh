#!/usr/bin/env bash
# Launch full-scale HIA Sapiens2 v2 Refinement Daemon across ALL clips/frames.
set -euo pipefail

REPO_DIR="/home/shangguanrz/project/pic-edit"
PYTHON_BIN="/home/shangguanrz/miniconda3/envs/deepgen/bin/python"
SCRIPT_PATH="${REPO_DIR}/scripts/data/rerefine_hidden_hands_v2.py"
OUT_DIR="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/rerefine_v2_all_${TIMESTAMP}.log"

echo "=========================================================="
echo " Starting Full-Scale HIA Sapiens2 v2 Refinement Daemon"
echo " Repo:      ${REPO_DIR}"
echo " Script:    ${SCRIPT_PATH}"
echo " Output:    ${OUT_DIR}"
echo " Log file:  ${LOG_FILE}"
echo "=========================================================="

nohup "${PYTHON_BIN}" -u "${SCRIPT_PATH}" \
    --clips all \
    --commit \
    --device cuda:0 \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${OUT_DIR}/rerefine_v2_daemon.pid"
echo "Daemon launched successfully with PID: ${PID}"
echo ""
echo "To monitor live progress in real-time:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To inspect live status JSON:"
echo "  cat ${OUT_DIR}/progress.json"
echo "=========================================================="
