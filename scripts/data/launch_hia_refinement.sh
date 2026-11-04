#!/usr/bin/env bash
# Launch full-scale HIA Sapiens2-guided SMPL-X refinement pipeline daemon.
set -euo pipefail

REPO_DIR="/home/shangguanrz/project/pic-edit"
PYTHON_BIN="/home/shangguanrz/miniconda3/envs/deepgen/bin/python"
SCRIPT_PATH="${REPO_DIR}/scripts/data/refine_hia_dataset_sapiens2.py"
OUT_DIR="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo " Starting Full HIA Sapiens2-guided SMPL-X Refinement Daemon"
echo " Repo:      ${REPO_DIR}"
echo " Script:    ${SCRIPT_PATH}"
echo " Output:    ${OUT_DIR}"
echo " Log file:  ${LOG_FILE}"
echo "=========================================================="

nohup "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --out-dir "${OUT_DIR}" \
    --split all \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Pipeline launched successfully with PID: ${PID}"
echo ""
echo "To monitor live progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To inspect live status JSON:"
echo "  cat ${OUT_DIR}/progress.json"
echo "=========================================================="
