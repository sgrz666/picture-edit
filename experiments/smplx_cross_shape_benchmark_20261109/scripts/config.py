"""Paths and constants for the SMPL-X cross-shape benchmark.

All heavy data lives under RUN_ROOT (default /data/shangguanrz/shape) on sg.
The repository copy only holds code, small results and the report.
"""
from __future__ import annotations

import os
from pathlib import Path

RUN_ROOT = Path(os.environ.get("SHAPE_RUN_ROOT", "/data/shangguanrz/shape"))
PIC_EDIT_ROOT = Path(os.environ.get("PIC_EDIT_ROOT", "/home/shangguanrz/project/pic-edit"))

# HBW (already downloaded; do not re-download)
HBW_ROOT = Path("/data/shangguanrz/datasets/HBW")
HBW_FULLRES = HBW_ROOT / "full_resolution_v1"
HBW_ZIP = HBW_ROOT / "HBW.zip"
HBW_LOWRES = HBW_ROOT / "low_res" / "HBW_low_resolution"

# Existing SMPLest-X run (October 2026, 705/705 images)
SMPLESTX_RUN = HBW_ROOT / "experiments" / "hbw_fullres_v1"
SMPLESTX_PRED = SMPLESTX_RUN / "predictions.jsonl"
SMPLESTX_VSHAPED = SMPLESTX_RUN / "hbw_fullres_predictions.npz"
SMPLESTX_SUMMARY = SMPLESTX_RUN / "summary_all_metrics.json"
SMPLESTX_PER_IMAGE = SMPLESTX_RUN / "per_image_all_metrics.csv"
SMPLESTX_PARAMS_GLOB = (
    PIC_EDIT_ROOT
    / "add/tools/SMPLest-X/demo/output_frames/hbwf_20261028_061623_3447408_*"
    / "hbwf_20261028_061623_3447408_*_params.pt"
)

# SMPL-X neutral model (400 shape dims; pipeline uses the first 10)
SMPLX_DIR = PIC_EDIT_ROOT / "add/tools/SMPLest-X/human_models/human_model_files/smplx"
SMPLX_NEUTRAL = SMPLX_DIR / "SMPLX_NEUTRAL.npz"

# SHAPY
SHAPY_ZIP = HBW_ROOT / "shapy_data.zip"
SHAPY_REPO = RUN_ROOT / "third_party" / "shapy"
SHAPY_CPU_REPO = RUN_ROOT / "third_party" / "shapy_cpu"
SHAPY_VENV = RUN_ROOT / "env" / "shapy"
SHAPY_HBW = RUN_ROOT / "shapy_hbw"
SHAPY_RUN = RUN_ROOT / "runs" / "shapy"
SHAPY_CKPT = SHAPY_RUN / "checkpoints" / "best_checkpoint"

# Measurement resources (SHAPY official definitions + 20K P2P regressor)
UTILITY_DIR = RUN_ROOT / "data" / "shapy_utility"

# Derived data
DATA = RUN_ROOT / "data"
KEYPOINTS = DATA / "hbw_keypoints" / "val"
GENDERS = DATA / "hbw_genders.yaml"
MANIFEST = DATA / "manifest.json"
COVERAGE = DATA / "coverage.json"
SHA256 = DATA / "sha256.json"
GT_BETAS_10 = DATA / "gt_betas_10.npz"
GT_BETAS_300 = DATA / "gt_betas_300.npz"
GT_FIT_REPORT = DATA / "gt_fit_report.csv"
POSES = DATA / "pose_templates.json"
SMPLESTX_PARAMS = DATA / "smplestx_params.npz"
PRED_SHAPY = SHAPY_RUN / "predictions.jsonl"
SHAPY_COVERAGE = SHAPY_RUN / "coverage.json"

# Outputs
OUT_P0 = RUN_ROOT / "output" / "p0"
OUT_P1 = RUN_ROOT / "output" / "p1"
FIGURES = RUN_ROOT / "output" / "figures"
LOGS = RUN_ROOT / "logs"
CACHE = RUN_ROOT / "cache"

SEED = 20261109
POSE_COUNT = 6
VIEW_AZIMUTHS = (0, 45, 90)  # front, 45 deg, side
MASK_SIZE = 512
PANEL_SIZE = 768

# Pre-registered practical thresholds
P2P_FIT_FLOOR_MM = 4.0
SHAPY_FEASIBILITY_P2P_MM = 30.0
SHAPY_FEASIBILITY_V2V_MM = 30.0
SHAPY_FEASIBILITY_HEIGHT_MM = 90.0
SHAPY_INTEGRATION_MIN_P2P_GAIN_MM = 3.0
