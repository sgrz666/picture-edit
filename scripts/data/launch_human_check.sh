#!/usr/bin/env bash
# Launch full comparison image generation for human-check in the background.
set -euo pipefail

REPO_DIR="/home/shangguanrz/project/pic-edit"
PYTHON_BIN="/home/shangguanrz/miniconda3/envs/deepgen/bin/python"
SCRIPT_PATH="${REPO_DIR}/scripts/data/generate_all_human_check.py"
OUT_DIR="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1/human-check"
LOG_DIR="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/human_check_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo " Starting Full Comparison Generation for Human Check"
echo " Repo:      ${REPO_DIR}"
echo " Script:    ${SCRIPT_PATH}"
echo " Output:    ${OUT_DIR}"
echo " Log file:  ${LOG_FILE}"
echo " Order:     Sorted by clip_id (starting with 000002)"
echo "=========================================================="

nohup "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --out-dir "${OUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Task launched with PID: ${PID}"
echo ""
echo "To monitor live logs:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check live progress JSON:"
echo "  cat ${OUT_DIR}/progress.json"
echo "=========================================================="
