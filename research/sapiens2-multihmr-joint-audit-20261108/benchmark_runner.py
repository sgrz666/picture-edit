"""Hi4D Dual-Person Benchmark Runner and Evaluation Suite.

Provides standardized evaluation across the 12-frame pilot set and 127-frame full
Hi4D dataset, supporting live pipeline execution, offline evaluation of precomputed
predictions, and automated side-by-side comparison with the MultiHMR2 baseline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
# NumPy 2.x Serialization Compatibility Shim
# Fixes unpickling arrays saved with NumPy 2.x when executing in NumPy 1.x environments.
# -----------------------------------------------------------------------------
try:
    import numpy._core  # type: ignore[import-not-found]
except ImportError:
    import numpy.core as _core  # type: ignore[import-not-found]
    sys.modules["numpy._core"] = _core
    sys.modules["numpy._core.multiarray"] = _core.multiarray


# -----------------------------------------------------------------------------
# Anatomical Landmarks & Topological Evaluation Indices
# -----------------------------------------------------------------------------
BODY_JOINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# SMPL / SMPL-X evaluation indices corresponding to BODY_JOINT_NAMES
SMPLX_EVAL_IDS: List[int] = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
SMPLX_12_IDS: List[int] = SMPLX_EVAL_IDS

# MultiHMR2 Anny evaluation indices (163-bone topological layout)
ANNY_12_IDS: List[int] = [50, 76, 52, 78, 54, 80, 2, 22, 4, 24, 6, 26]

# Sapiens 308 Goliath evaluation indices
GOLIATH_12_IDS: List[int] = [5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
GOLIATH_EVAL_IDS: List[int] = GOLIATH_12_IDS

# Hi4D Interaction Categories
CATEGORIES: List[str] = ["separated", "light", "hug", "severe", "overall"]

# Canonical 12-Frame Balanced Pilot Subset (3 balanced frames per category)
CANONICAL_PILOT_IDS: List[str] = [
    "809de42848f7edd5", "e1fd100a73afa906", "d49c97889e8076ac",  # separated (3)
    "be3143ff3423db7c", "57ddb1300b306154", "48ff7ce57806b836",  # light (3)
    "af5be55a94f0b32c", "36f248e6fef3cec4", "dea75ff6c7fe2f92",  # hug (3)
    "4d008c99443fca02", "b0f30476d2d8aa44", "4ef96877728ae29e",  # severe (3)
]


# -----------------------------------------------------------------------------
# Geometric & Alignment Functions
# -----------------------------------------------------------------------------
def compute_similarity_transform(
    S1: np.ndarray,
    S2: np.ndarray,
    enforce_proper_so3: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """Computes similarity transform (scale, rotation, translation) mapping S1 to S2.

    Minimizes the sum of squared Euclidean errors:
        argmin_{s, R, t} || s * S1 @ R.T + t - S2 ||_F^2
    via Umeyama SVD analysis with reflection disambiguation.

    Args:
        S1: Predicted 3D point cloud of shape (N, 3).
        S2: Ground truth 3D point cloud of shape (N, 3).
        enforce_proper_so3: If True, strictly enforces det(R) = +1 (Umeyama 1991).
            If False, adopts benchmark-compatible column sign flip R[:, -1] *= -1.
        eps: Minimum Frobenius norm threshold to guard against point collapse.

    Returns:
        Aligned point cloud of shape (N, 3): s * S1 @ R.T + t.
        If S1 or S2 is degenerate (norm < eps), returns S1 unmodified.
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)

    mu1 = S1.mean(axis=0)
    mu2 = S2.mean(axis=0)

    S1_hat = S1 - mu1
    S2_hat = S2 - mu2

    norm1 = np.sqrt(np.sum(S1_hat**2))
    norm2 = np.sqrt(np.sum(S2_hat**2))

    if norm1 < eps or norm2 < eps:
        return S1.copy()

    S1_hat /= norm1
    S2_hat /= norm2

    # Cross-covariance matrix H = S1_hat.T @ S2_hat
    H = S1_hat.T @ S2_hat
    U, s, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    det_R = np.linalg.det(R)

    if enforce_proper_so3:
        d = np.ones(3, dtype=np.float64)
        if det_R < 0:
            d[-1] = -1.0
        R = Vt.T @ np.diag(d) @ U.T
        scale = float((s @ d) * (norm2 / norm1))
    else:
        if det_R < 0:
            R = R.copy()
            R[:, -1] *= -1.0
        scale = float(s.sum() * (norm2 / norm1))

    t = mu2 - scale * (R @ mu1)
    return scale * (S1 @ R.T) + t


def project_3d_to_2d(j3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Projects 3D camera-frame coordinates to 2D image pixel coordinates.

    Args:
        j3d: Array of 3D points of shape (N, 3).
        K: Camera intrinsics matrix of shape (3, 3).

    Returns:
        Projected 2D pixel coordinates of shape (N, 2).
    """
    j3d = np.asarray(j3d, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    homo = j3d @ K.T
    z = np.maximum(homo[:, 2:], 1e-6)
    return homo[:, :2] / z


def compute_pair_pa_mpjpe(
    p0: np.ndarray,
    p1: np.ndarray,
    gt0: np.ndarray,
    gt1: np.ndarray,
    enforce_proper_so3: bool = False,
) -> float:
    """Computes pair-wise Procrustes-aligned MPJPE in millimeters.

    Enforces a single joint similarity transform (s, R, t) simultaneously
    over the concatenated 24-joint coordinates of both interacting actors.

    Args:
        p0: Predicted joints for person 0, shape (12, 3).
        p1: Predicted joints for person 1, shape (12, 3).
        gt0: Ground truth joints for person 0, shape (12, 3).
        gt1: Ground truth joints for person 1, shape (12, 3).
        enforce_proper_so3: Whether to enforce proper SO(3) alignment.

    Returns:
        Pair-PA-MPJPE in millimeters.
    """
    pair_pred = np.vstack([p0, p1])
    pair_gt = np.vstack([gt0, gt1])
    pair_aligned = compute_similarity_transform(
        pair_pred, pair_gt, enforce_proper_so3=enforce_proper_so3
    )
    return float(1000.0 * np.mean(np.linalg.norm(pair_aligned - pair_gt, axis=-1)))


# -----------------------------------------------------------------------------
# Prediction Format Parsers
# -----------------------------------------------------------------------------
def extract_candidates(
    entry: Any,
    record: Dict[str, Any],
    model_type: str = "joint_smplx",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Extracts candidate 3D joints, 2D joints, and pelvis translations from a prediction entry.

    Supports:
        - 'joint_smplx' / 'joint_optimization': List of dicts with 'j3d', 'j2d', 'transl_pelvis'
        - 'multihmr2_anny': Dict with 'j3d' (N, 163, 3), 'transl_pelvis' (N, 3), 'num_person'
        - 'multihmr_smplx': Dict with 'predictions' containing 'j3d' (127, 3), 'j2d_orig'
        - 'sapiens_json': Dict with 'instances' containing 'xy' (308, 2) or (12, 2)

    Args:
        entry: Prediction entry for the given record.
        record: Ground truth dataset record dictionary.
        model_type: Model prediction format identifier.

    Returns:
        Tuple of (candidate_j3d, candidate_j2d, candidate_pelvis).
    """
    candidate_j3d: List[np.ndarray] = []
    candidate_j2d: List[np.ndarray] = []
    candidate_pelvis: List[np.ndarray] = []

    if entry is None:
        return candidate_j3d, candidate_j2d, candidate_pelvis

    K_mat = np.array(record.get("K", np.eye(3)), dtype=np.float64)

    if model_type in ("joint_smplx", "joint_optimization"):
        # Native pipeline prediction output: list of person dicts or dict with 'instances'/'predictions'
        p_list: List[Dict[str, Any]] = []
        if isinstance(entry, list):
            p_list = entry
        elif isinstance(entry, dict):
            if "predictions" in entry:
                p_list = entry["predictions"]
            elif "instances" in entry:
                p_list = entry["instances"]
            elif "j3d" in entry:
                # Batched format
                j3ds = entry["j3d"]
                num_p = entry.get("num_person", len(j3ds))
                pelvises = entry.get("transl_pelvis", [None] * num_p)
                j2ds = entry.get("j2d", [None] * num_p)
                for i in range(num_p):
                    p_list.append({
                        "j3d": j3ds[i],
                        "transl_pelvis": pelvises[i] if i < len(pelvises) else None,
                        "j2d": j2ds[i] if i < len(j2ds) else None,
                    })

        for p in p_list:
            if not isinstance(p, dict):
                continue
            if "j3d" in p and p["j3d"] is not None:
                j3d_raw = np.asarray(p["j3d"], dtype=np.float64)
                if j3d_raw.shape[0] > 21:
                    j3d_12 = j3d_raw[SMPLX_EVAL_IDS]
                elif j3d_raw.shape[0] == 12:
                    j3d_12 = j3d_raw
                else:
                    j3d_12 = j3d_raw[:12]
                candidate_j3d.append(j3d_12)

                if "transl_pelvis" in p and p["transl_pelvis"] is not None:
                    candidate_pelvis.append(np.asarray(p["transl_pelvis"], dtype=np.float64))
                else:
                    # Default pelvis to joint 0 if full SMPL-X or origin
                    if j3d_raw.shape[0] > 0:
                        candidate_pelvis.append(j3d_raw[0].copy())
                    else:
                        candidate_pelvis.append(np.zeros(3, dtype=np.float64))

                if "j2d" in p and p["j2d"] is not None:
                    j2d_raw = np.asarray(p["j2d"], dtype=np.float64)
                    if j2d_raw.shape[0] > 21:
                        candidate_j2d.append(j2d_raw[SMPLX_EVAL_IDS])
                    elif j2d_raw.shape[0] == 12:
                        candidate_j2d.append(j2d_raw)
                    else:
                        candidate_j2d.append(j2d_raw[:12])
                else:
                    # Project j3d to 2D
                    candidate_j2d.append(project_3d_to_2d(j3d_12, K_mat))
            elif "j2d" in p and p["j2d"] is not None:
                j2d_raw = np.asarray(p["j2d"], dtype=np.float64)
                if j2d_raw.shape[0] > 21:
                    candidate_j2d.append(j2d_raw[SMPLX_EVAL_IDS])
                elif j2d_raw.shape[0] == 12:
                    candidate_j2d.append(j2d_raw)
                else:
                    candidate_j2d.append(j2d_raw[:12])

    elif model_type == "multihmr2_anny":
        num_p = entry.get("num_person", 0)
        if num_p > 0 and "j3d" in entry and len(entry["j3d"]) > 0:
            j3ds = entry["j3d"]
            pelvises = entry.get("transl_pelvis", np.zeros((num_p, 3), dtype=np.float64))
            for i in range(num_p):
                p_j3d = np.asarray(j3ds[i][ANNY_12_IDS], dtype=np.float64)
                candidate_j3d.append(p_j3d)
                candidate_pelvis.append(np.asarray(pelvises[i], dtype=np.float64))
                candidate_j2d.append(project_3d_to_2d(p_j3d, K_mat))

    elif model_type == "multihmr_smplx":
        p_list = entry.get("predictions", [])
        for p in p_list:
            if "j3d" in p:
                j3d_raw = np.asarray(p["j3d"], dtype=np.float64)
                candidate_j3d.append(j3d_raw[SMPLX_EVAL_IDS])
            if "j2d_orig" in p:
                j2d_raw = np.asarray(p["j2d_orig"], dtype=np.float64)
                candidate_j2d.append(j2d_raw[SMPLX_EVAL_IDS])
            elif "j3d" in p:
                candidate_j2d.append(project_3d_to_2d(candidate_j3d[-1], K_mat))
            if "transl_pelvis" in p:
                candidate_pelvis.append(np.asarray(p["transl_pelvis"], dtype=np.float64))
            elif "j3d" in p:
                candidate_pelvis.append(j3d_raw[0].copy())

    elif model_type == "sapiens_json":
        instances = entry.get("instances", [])
        for inst in instances:
            if "xy" in inst:
                xy_raw = np.asarray(inst["xy"], dtype=np.float64)
                if xy_raw.shape[0] >= 308:
                    candidate_j2d.append(xy_raw[GOLIATH_12_IDS])
                elif xy_raw.shape[0] == 12:
                    candidate_j2d.append(xy_raw)
                else:
                    candidate_j2d.append(xy_raw[:12])

    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Supported formats: "
            "'joint_smplx', 'joint_optimization', 'multihmr2_anny', "
            "'multihmr_smplx', 'sapiens_json'."
        )

    return candidate_j3d, candidate_j2d, candidate_pelvis


# -----------------------------------------------------------------------------
# Metric Accumulator
# -----------------------------------------------------------------------------
class MetricAccumulator:
    """Accumulates and calculates benchmark metrics per interaction category."""

    def __init__(self) -> None:
        self.total_frames: int = 0
        self.detected_exact2_frames: int = 0
        self.detected_ge2_frames: int = 0
        self.detected_person_counts: List[int] = []

        self.pck05_list: List[bool] = []
        self.pck10_list: List[bool] = []
        self.px_err_list: List[float] = []
        self.cross_confusion_count: int = 0
        self.total_eval_joints: int = 0

        self.mpjpe_list: List[float] = []
        self.pa_mpjpe_list: List[float] = []
        self.pair_pa_mpjpe_list: List[float] = []
        self.abs_mpjpe_list: List[float] = []
        self.pelvis_vector_err_list: List[float] = []

        self.pck_3d_50mm_hits: List[bool] = []
        self.pck_3d_100mm_hits: List[bool] = []

    def update_detection(self, num_det: int) -> None:
        self.total_frames += 1
        self.detected_person_counts.append(num_det)
        if num_det == 2:
            self.detected_exact2_frames += 1
        if num_det >= 2:
            self.detected_ge2_frames += 1

    def update_2d(
        self,
        pck05: List[bool],
        pck10: List[bool],
        px_err: List[float],
        cross_confusion: int,
        total_joints: int,
    ) -> None:
        self.pck05_list.extend(pck05)
        self.pck10_list.extend(pck10)
        self.px_err_list.extend(px_err)
        self.cross_confusion_count += cross_confusion
        self.total_eval_joints += total_joints

    def update_3d(
        self,
        mpjpe: float,
        pa_mpjpe: float,
        pair_pa: float,
        abs_mpjpe: float,
        pelvis_vec: float,
        rr_joint_errors: Optional[np.ndarray] = None,
    ) -> None:
        self.mpjpe_list.append(mpjpe)
        self.pa_mpjpe_list.append(pa_mpjpe)
        self.pair_pa_mpjpe_list.append(pair_pa)
        self.abs_mpjpe_list.append(abs_mpjpe)
        self.pelvis_vector_err_list.append(pelvis_vec)

        if rr_joint_errors is not None:
            self.pck_3d_50mm_hits.extend((rr_joint_errors < 50.0).tolist())
            self.pck_3d_100mm_hits.extend((rr_joint_errors < 100.0).tolist())

    def summarize(self) -> Dict[str, Any]:
        tf = self.total_frames
        return {
            "total_frames": tf,
            "detected_exact2_rate": (self.detected_exact2_frames / tf) if tf > 0 else 0.0,
            "detected_ge2_rate": (self.detected_ge2_frames / tf) if tf > 0 else 0.0,
            "avg_detected_persons": (
                float(np.mean(self.detected_person_counts)) if self.detected_person_counts else 0.0
            ),
            "pck05": (float(np.mean(self.pck05_list)) * 100.0) if self.pck05_list else 0.0,
            "pck10": (float(np.mean(self.pck10_list)) * 100.0) if self.pck10_list else 0.0,
            "mean_px_error": (float(np.mean(self.px_err_list))) if self.px_err_list else 0.0,
            "cross_confusion_rate": (
                (self.cross_confusion_count / self.total_eval_joints * 100.0)
                if self.total_eval_joints > 0
                else 0.0
            ),
            "mpjpe_mm": float(np.mean(self.mpjpe_list)) if self.mpjpe_list else None,
            "pa_mpjpe_mm": float(np.mean(self.pa_mpjpe_list)) if self.pa_mpjpe_list else None,
            "pair_pa_mpjpe_mm": (
                float(np.mean(self.pair_pa_mpjpe_list)) if self.pair_pa_mpjpe_list else None
            ),
            "abs_mpjpe_mm": float(np.mean(self.abs_mpjpe_list)) if self.abs_mpjpe_list else None,
            "pelvis_vector_err_mm": (
                float(np.mean(self.pelvis_vector_err_list)) if self.pelvis_vector_err_list else None
            ),
            "pck_3d_50mm": (
                float(np.mean(self.pck_3d_50mm_hits)) * 100.0 if self.pck_3d_50mm_hits else None
            ),
            "pck_3d_100mm": (
                float(np.mean(self.pck_3d_100mm_hits)) * 100.0 if self.pck_3d_100mm_hits else None
            ),
            "valid_3d_pairs_evaluated": len(self.mpjpe_list),
        }


# -----------------------------------------------------------------------------
# Main Benchmark Runner Class
# -----------------------------------------------------------------------------
class Hi4DBenchmarkRunner:
    """Hi4D Dual-Person Benchmark Runner and Multi-Metric Evaluator.

    Executes full evaluation across 9 metrics on both the 12-frame pilot set
    and the full 127-frame Hi4D dataset. Supports offline evaluation of precomputed
    predictions and live pipeline inference, generating JSON reports and ASCII tables.
    """

    CANONICAL_PILOT_IDS: List[str] = CANONICAL_PILOT_IDS
    CATEGORIES: List[str] = CATEGORIES

    def __init__(
        self,
        config: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        smplx_model: Optional[Any] = None,
        device: str = "cuda:0",
    ) -> None:
        """Initializes the benchmark runner.

        Args:
            config: Optional JointOptimizationConfig instance.
            pipeline: Optional end-to-end inference pipeline instance.
            smplx_model: Optional SMPL-X model for forward kinematic evaluation.
            device: Computation device ('cuda:0', 'cpu').
        """
        self.config = config
        self.pipeline = pipeline
        self.smplx_model = smplx_model
        self.device = device

    def load_dataset(
        self,
        dataset_path: Optional[str] = None,
        use_pilot: bool = False,
        pilot_ids: Optional[List[str]] = None,
        n_pilot_per_category: int = 3,
    ) -> List[Dict[str, Any]]:
        """Loads and validates benchmark records from request.json.

        Args:
            dataset_path: Path to request.json. If None, checks config or uses default.
            use_pilot: If True, filters records to the 12-frame pilot subset.
            pilot_ids: Optional explicit list of frame IDs to override the pilot set.
            n_pilot_per_category: Number of frames per category when synthesizing pilot set.

        Returns:
            List of validated record dictionaries.

        Raises:
            FileNotFoundError: If the request file does not exist.
            ValueError: If the dataset file is malformed or records are missing required keys.
        """
        if dataset_path is None:
            if self.config is not None and hasattr(self.config, "hi4d_request_json"):
                dataset_path = self.config.hi4d_request_json
            else:
                dataset_path = (
                    "/data/shangguanrz/pic-edit-experiments/mete-test/"
                    "dual_pose_compare_20261029_v1/data/request.json"
                )

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Benchmark dataset request file not found: '{dataset_path}'")

        with open(dataset_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "records" in data:
            records = data["records"]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(f"Malformed request.json structure in '{dataset_path}'.")

        # Validate record integrity
        required_keys = {"id", "category", "gt_xy", "gt_xyz", "gt_pelvis", "gt_valid", "gt_boxes"}
        for idx, rec in enumerate(records):
            missing = required_keys - set(rec.keys())
            if missing:
                raise ValueError(
                    f"Record at index {idx} (id='{rec.get('id', 'unknown')}') is missing "
                    f"required ground truth keys: {sorted(missing)}"
                )

        if not use_pilot:
            return records

        # Filter pilot records
        if pilot_ids is not None:
            pilot_id_set = set(pilot_ids)
            selected = [r for r in records if r["id"] in pilot_id_set]
            return selected

        # Check for canonical pilot IDs
        all_ids = {r["id"] for r in records}
        has_canonical = all(pid in all_ids for pid in self.CANONICAL_PILOT_IDS)
        if has_canonical:
            id_to_record = {r["id"]: r for r in records}
            return [id_to_record[pid] for pid in self.CANONICAL_PILOT_IDS]

        # Fallback to balanced selection (first N per category in order of appearance)
        selected_by_cat: Dict[str, List[Dict[str, Any]]] = {
            "separated": [],
            "light": [],
            "hug": [],
            "severe": [],
        }
        for r in records:
            cat = r.get("category", "")
            if cat in selected_by_cat and len(selected_by_cat[cat]) < n_pilot_per_category:
                selected_by_cat[cat].append(r)

        pilot_records: List[Dict[str, Any]] = []
        for cat in ["separated", "light", "hug", "severe"]:
            pilot_records.extend(selected_by_cat[cat])
        return pilot_records

    def evaluate_predictions(
        self,
        records: List[Dict[str, Any]],
        predictions_map: Dict[str, Any],
        model_type: str = "joint_smplx",
    ) -> Dict[str, Any]:
        """Core functional evaluation routine computing all 9 metrics across categories.

        Args:
            records: List of benchmark ground truth records.
            predictions_map: Dictionary mapping record 'id' to prediction entries.
            model_type: Prediction format identifier.

        Returns:
            Comprehensive results dictionary conforming to the benchmark schema.
        """
        accumulators = {c: MetricAccumulator() for c in self.CATEGORIES}
        per_frame_list: List[Dict[str, Any]] = []
        per_sequence_map: Dict[str, Dict[str, Any]] = {}

        for r in records:
            rec_id = r["id"]
            cat = r["category"]
            seq_key = f"{r.get('pair', '')}_{r.get('action', '')}".strip("_")
            if not seq_key:
                seq_key = f"seq_{cat}"

            gt_xy = np.array(r["gt_xy"], dtype=np.float64)  # (2, 12, 2)
            gt_xyz = np.array(r["gt_xyz"], dtype=np.float64)  # (2, 12, 3)
            gt_pelvis = np.array(r["gt_pelvis"], dtype=np.float64)  # (2, 3)
            gt_valid = np.array(r["gt_valid"], dtype=bool)  # (2, 12)
            gt_boxes = np.array(r["gt_boxes"], dtype=np.float64)  # (2, 4)

            # Bounding box diagonal scale per person
            scales = np.linalg.norm(gt_boxes[:, 2:] - gt_boxes[:, :2], axis=1)
            scales = np.maximum(scales, 1.0)

            # Extract prediction candidates
            entry = predictions_map.get(rec_id, None)
            candidate_j3d, candidate_j2d, candidate_pelvis = extract_candidates(
                entry, r, model_type=model_type
            )

            num_det = len(candidate_j2d)
            cats_to_update = [cat, "overall"]

            for c in cats_to_update:
                accumulators[c].update_detection(num_det)

            frame_res: Dict[str, Any] = {
                "id": rec_id,
                "category": cat,
                "pair": r.get("pair", ""),
                "action": r.get("action", ""),
                "frame_id": r.get("frame_id", 0),
                "num_det": num_det,
                "pck05": None,
                "pck10": None,
                "mean_px_err": None,
                "mpjpe_mm": None,
                "pa_mpjpe_mm": None,
                "pair_pa_mpjpe_mm": None,
                "abs_mpjpe_mm": None,
                "pelvis_vector_err_mm": None,
            }

            if num_det == 0:
                per_frame_list.append(frame_res)
                continue

            # Permutation Matching
            matches: List[Tuple[int, int]] = []
            if num_det == 1:
                c2d = candidate_j2d[0]
                val0 = gt_valid[0]
                val1 = gt_valid[1]
                err0 = (
                    np.mean(np.linalg.norm(c2d - gt_xy[0], axis=-1)[val0] / scales[0])
                    if np.any(val0)
                    else 1e9
                )
                err1 = (
                    np.mean(np.linalg.norm(c2d - gt_xy[1], axis=-1)[val1] / scales[1])
                    if np.any(val1)
                    else 1e9
                )
                g_match = 0 if err0 <= err1 else 1
                matches = [(g_match, 0)]
            else:
                from itertools import permutations

                best_cost = 1e12
                best_matches: Optional[List[Tuple[int, int]]] = None

                for p0, p1 in permutations(range(num_det), 2):
                    if candidate_j3d:
                        # 3D root-relative matching cost
                        d0 = np.mean(
                            np.linalg.norm(
                                (candidate_j3d[p0] - candidate_pelvis[p0])
                                - (gt_xyz[0] - gt_pelvis[0]),
                                axis=-1,
                            )
                        )
                        d1 = np.mean(
                            np.linalg.norm(
                                (candidate_j3d[p1] - candidate_pelvis[p1])
                                - (gt_xyz[1] - gt_pelvis[1]),
                                axis=-1,
                            )
                        )
                    else:
                        # 2D reprojection matching cost
                        val0 = gt_valid[0]
                        val1 = gt_valid[1]
                        d0 = (
                            np.mean(
                                np.linalg.norm(candidate_j2d[p0] - gt_xy[0], axis=-1)[val0]
                                / scales[0]
                            )
                            if np.any(val0)
                            else 0.0
                        )
                        d1 = (
                            np.mean(
                                np.linalg.norm(candidate_j2d[p1] - gt_xy[1], axis=-1)[val1]
                                / scales[1]
                            )
                            if np.any(val1)
                            else 0.0
                        )
                    cost = d0 + d1
                    if cost < best_cost:
                        best_cost = cost
                        best_matches = [(0, p0), (1, p1)]

                matches = best_matches if best_matches is not None else [(0, 0), (1, 1)]

            # 2D Evaluation
            frame_pck05: List[bool] = []
            frame_pck10: List[bool] = []
            frame_px_err: List[float] = []
            frame_confused: int = 0
            frame_total_val: int = 0

            for g_idx, p_idx in matches:
                pred_2d = candidate_j2d[p_idx]
                val = gt_valid[g_idx]
                sc = scales[g_idx]
                dist_px = np.linalg.norm(pred_2d - gt_xy[g_idx], axis=-1)
                dist_norm = dist_px / sc

                # Cross-person confusion check
                other_g = 1 - g_idx
                other_dist = np.linalg.norm(pred_2d - gt_xy[other_g], axis=-1)
                sep = np.linalg.norm(gt_xy[g_idx] - gt_xy[other_g], axis=-1)
                confused = (
                    val
                    & gt_valid[other_g]
                    & (other_dist + 0.02 * sc < dist_px)
                    & (sep > 0.10 * sc)
                )

                usable = val & np.isfinite(dist_px)
                frame_pck05.extend((dist_norm[usable] < 0.05).tolist())
                frame_pck10.extend((dist_norm[usable] < 0.10).tolist())
                frame_px_err.extend(dist_px[usable].tolist())
                frame_confused += int(np.sum(confused))
                frame_total_val += int(np.sum(usable))

            for c in cats_to_update:
                accumulators[c].update_2d(
                    frame_pck05,
                    frame_pck10,
                    frame_px_err,
                    frame_confused,
                    frame_total_val,
                )

            if frame_pck05:
                frame_res["pck05"] = float(np.mean(frame_pck05)) * 100.0
            if frame_pck10:
                frame_res["pck10"] = float(np.mean(frame_pck10)) * 100.0
            if frame_px_err:
                frame_res["mean_px_err"] = float(np.mean(frame_px_err))

            # 3D Evaluation (if 3D joints available and both persons matched)
            if candidate_j3d and len(matches) == 2:
                p0, p1 = matches[0][1], matches[1][1]
                p_j3d_0 = candidate_j3d[p0]
                p_j3d_1 = candidate_j3d[p1]
                p_pel_0 = candidate_pelvis[p0]
                p_pel_1 = candidate_pelvis[p1]

                # 1. Root-relative MPJPE (mm)
                rr_0 = 1000.0 * np.linalg.norm(
                    (p_j3d_0 - p_pel_0) - (gt_xyz[0] - gt_pelvis[0]), axis=-1
                )
                rr_1 = 1000.0 * np.linalg.norm(
                    (p_j3d_1 - p_pel_1) - (gt_xyz[1] - gt_pelvis[1]), axis=-1
                )
                all_rr = np.concatenate([rr_0, rr_1])
                frame_mpjpe = float(np.mean(all_rr))

                # 2. PA-MPJPE per person (mm)
                aligned_0 = compute_similarity_transform(p_j3d_0, gt_xyz[0])
                aligned_1 = compute_similarity_transform(p_j3d_1, gt_xyz[1])
                pa_0 = 1000.0 * np.linalg.norm(aligned_0 - gt_xyz[0], axis=-1)
                pa_1 = 1000.0 * np.linalg.norm(aligned_1 - gt_xyz[1], axis=-1)
                frame_pa_mpjpe = float(np.mean(np.concatenate([pa_0, pa_1])))

                # 3. Pair-PA-MPJPE on concatenated (24, 3) joints (mm)
                frame_pair_pa = compute_pair_pa_mpjpe(p_j3d_0, p_j3d_1, gt_xyz[0], gt_xyz[1])

                # 4. Absolute MPJPE in camera coordinates (mm)
                abs_0 = 1000.0 * np.linalg.norm(p_j3d_0 - gt_xyz[0], axis=-1)
                abs_1 = 1000.0 * np.linalg.norm(p_j3d_1 - gt_xyz[1], axis=-1)
                frame_abs_mpjpe = float(np.mean(np.concatenate([abs_0, abs_1])))

                # 5. Pelvis relative displacement vector error (mm)
                pred_vec = p_pel_1 - p_pel_0
                gt_vec = gt_pelvis[1] - gt_pelvis[0]
                frame_pelvis_vec = float(1000.0 * np.linalg.norm(pred_vec - gt_vec))

                for c in cats_to_update:
                    accumulators[c].update_3d(
                        frame_mpjpe,
                        frame_pa_mpjpe,
                        frame_pair_pa,
                        frame_abs_mpjpe,
                        frame_pelvis_vec,
                        rr_joint_errors=all_rr,
                    )

                frame_res["mpjpe_mm"] = frame_mpjpe
                frame_res["pa_mpjpe_mm"] = frame_pa_mpjpe
                frame_res["pair_pa_mpjpe_mm"] = frame_pair_pa
                frame_res["abs_mpjpe_mm"] = frame_abs_mpjpe
                frame_res["pelvis_vector_err_mm"] = frame_pelvis_vec

            per_frame_list.append(frame_res)

            # Accumulate per-sequence data
            if seq_key not in per_sequence_map:
                per_sequence_map[seq_key] = {
                    "category": cat,
                    "num_frames": 0,
                    "pa_mpjpe_list": [],
                    "pair_pa_mpjpe_list": [],
                    "pck05_list": [],
                }
            seq_entry = per_sequence_map[seq_key]
            seq_entry["num_frames"] += 1
            if frame_res["pa_mpjpe_mm"] is not None:
                seq_entry["pa_mpjpe_list"].append(frame_res["pa_mpjpe_mm"])
            if frame_res["pair_pa_mpjpe_mm"] is not None:
                seq_entry["pair_pa_mpjpe_list"].append(frame_res["pair_pa_mpjpe_mm"])
            if frame_res["pck05"] is not None:
                seq_entry["pck05_list"].append(frame_res["pck05"])

        # Format per-sequence summaries
        per_sequence_summary: Dict[str, Dict[str, Any]] = {}
        for s_key, s_data in per_sequence_map.items():
            per_sequence_summary[s_key] = {
                "category": s_data["category"],
                "num_frames": s_data["num_frames"],
                "mean_pa_mpjpe_mm": (
                    float(np.mean(s_data["pa_mpjpe_list"])) if s_data["pa_mpjpe_list"] else None
                ),
                "mean_pair_pa_mpjpe_mm": (
                    float(np.mean(s_data["pair_pa_mpjpe_list"]))
                    if s_data["pair_pa_mpjpe_list"]
                    else None
                ),
                "mean_pck05": (
                    float(np.mean(s_data["pck05_list"])) if s_data["pck05_list"] else None
                ),
            }

        # Categories summary
        categories_summary = {c: accumulators[c].summarize() for c in self.CATEGORIES}

        return {
            "categories": categories_summary,
            "per_sequence": per_sequence_summary,
            "per_frame": per_frame_list,
        }

    def run_offline(
        self,
        records: List[Dict[str, Any]],
        predictions_path_or_map: Union[str, Dict[str, Any]],
        model_type: str = "joint_smplx",
    ) -> Dict[str, Any]:
        """Loads precomputed prediction files and executes benchmark evaluation.

        Args:
            records: Ground truth dataset records.
            predictions_path_or_map: Path to .pkl / .json prediction file, or prediction dictionary.
            model_type: Prediction format identifier.

        Returns:
            Benchmark evaluation summary dictionary.
        """
        if isinstance(predictions_path_or_map, (str, Path)):
            p_path = str(predictions_path_or_map)
            if not os.path.exists(p_path):
                raise FileNotFoundError(f"Prediction file not found: '{p_path}'")
            if p_path.endswith(".pkl") or p_path.endswith(".pickle"):
                with open(p_path, "rb") as f:
                    predictions_map = pickle.load(f)
            elif p_path.endswith(".json"):
                with open(p_path, "r") as f:
                    predictions_map = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported prediction file extension: '{p_path}'. Expected .pkl or .json."
                )
        elif isinstance(predictions_path_or_map, dict):
            predictions_map = predictions_path_or_map
        else:
            raise TypeError(
                f"Expected str, Path, or dict for predictions_path_or_map, got {type(predictions_path_or_map)}"
            )

        return self.evaluate_predictions(records, predictions_map, model_type=model_type)

    def run_live(
        self,
        records: List[Dict[str, Any]],
        pipeline: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Executes live end-to-end pipeline inference and evaluates results.

        Args:
            records: Ground truth dataset records.
            pipeline: Pipeline object or callable. If None, uses self.pipeline.

        Returns:
            Benchmark evaluation summary dictionary.
        """
        target_pipeline = pipeline if pipeline is not None else self.pipeline
        if target_pipeline is None:
            raise ValueError(
                "No pipeline provided for live benchmark execution. "
                "Pass pipeline to __init__ or run_live."
            )

        predictions_map: Dict[str, Any] = {}
        for r in records:
            rec_id = r["id"]
            img_path = r["image"]
            K_mat = r.get("K", None)
            category = r.get("category", None)

            # Pipeline execution: supports callable, predict(), or run()
            if callable(target_pipeline):
                pred = target_pipeline(image_path=img_path, K=K_mat, category=category)
            elif hasattr(target_pipeline, "predict"):
                pred = target_pipeline.predict(image_path=img_path, K=K_mat, category=category)
            elif hasattr(target_pipeline, "run"):
                pred = target_pipeline.run(image_path=img_path, K=K_mat, category=category)
            else:
                raise AttributeError(
                    f"Pipeline object {type(target_pipeline)} does not implement __call__, predict, or run."
                )

            predictions_map[rec_id] = pred

        return self.evaluate_predictions(records, predictions_map, model_type="joint_smplx")

    def evaluate(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        predictions_map: Optional[Dict[str, Any]] = None,
        use_pilot: bool = False,
        model_type: str = "joint_smplx",
    ) -> Dict[str, Any]:
        """Unified evaluation entry point conforming to PROJECT.md Contract 5.

        Args:
            records: Optional benchmark records list. If None, loaded from disk.
            predictions_map: Optional precomputed predictions dictionary.
            use_pilot: If True and records is None, loads the 12-frame pilot subset.
            model_type: Model prediction format.

        Returns:
            Comprehensive evaluation results dictionary.
        """
        if records is None:
            records = self.load_dataset(use_pilot=use_pilot)

        if predictions_map is None and self.pipeline is not None:
            return self.run_live(records, pipeline=self.pipeline)

        if predictions_map is None:
            raise ValueError(
                "Neither predictions_map nor an active pipeline was provided to evaluate()."
            )

        return self.evaluate_predictions(records, predictions_map, model_type=model_type)

    def compare_against_baseline(
        self,
        refined_summary: Dict[str, Any],
        baseline_summary: Optional[Dict[str, Any]] = None,
        baseline_name: str = "MultiHMR2",
    ) -> Dict[str, Any]:
        """Computes deltas and percentage improvements against baseline metrics.

        Computes:
            - diff_pa_mpjpe = baseline_pa - refined_pa (positive = improvement)
            - diff_pair_pa = baseline_pair_pa - refined_pair_pa (positive = improvement)
            - diff_pck05 = refined_pck05 - baseline_pck05 (positive = improvement)
            - diff_pelvis_vec = baseline_pelvis - refined_pelvis (positive = improvement)
            - diff_mpjpe = baseline_mpjpe - refined_mpjpe (positive = improvement)

        Args:
            refined_summary: Summary dictionary from refined pipeline evaluation.
            baseline_summary: Optional summary dictionary from baseline model.
                If None, uses canonical MultiHMR2 overall baseline defaults.
            baseline_name: Identifier name for the baseline model.

        Returns:
            Comparison dictionary with overall and per-category improvements.
        """
        if baseline_summary is None:
            # Default canonical MultiHMR2 baseline metrics
            baseline_categories = {
                "overall": {
                    "pa_mpjpe_mm": 56.07,
                    "pair_pa_mpjpe_mm": 72.45,
                    "pck05": 74.60,
                    "mpjpe_mm": 150.41,
                    "pelvis_vector_err_mm": 127.27,
                }
            }
        else:
            baseline_categories = baseline_summary.get("categories", baseline_summary)

        ref_categories = refined_summary.get("categories", refined_summary)

        def _calc_comparison_section(ref_cat: Dict[str, Any], base_cat: Dict[str, Any]) -> Dict[str, Any]:
            sec: Dict[str, Any] = {}

            # PA-MPJPE
            base_pa = base_cat.get("pa_mpjpe_mm")
            ref_pa = ref_cat.get("pa_mpjpe_mm")
            if base_pa is not None and ref_pa is not None:
                sec["pa_mpjpe_baseline_mm"] = float(base_pa)
                sec["pa_mpjpe_refined_mm"] = float(ref_pa)
                sec["pa_mpjpe_improvement_mm"] = float(base_pa - ref_pa)
                sec["pa_mpjpe_improvement_pct"] = (
                    float((base_pa - ref_pa) / base_pa * 100.0) if base_pa > 0 else 0.0
                )
                sec["pa_mpjpe_improved"] = bool(ref_pa < base_pa)

            # Pair-PA-MPJPE
            base_pair = base_cat.get("pair_pa_mpjpe_mm")
            ref_pair = ref_cat.get("pair_pa_mpjpe_mm")
            if base_pair is not None and ref_pair is not None:
                sec["pair_pa_mpjpe_baseline_mm"] = float(base_pair)
                sec["pair_pa_mpjpe_refined_mm"] = float(ref_pair)
                sec["pair_pa_mpjpe_improvement_mm"] = float(base_pair - ref_pair)
                sec["pair_pa_mpjpe_improvement_pct"] = (
                    float((base_pair - ref_pair) / base_pair * 100.0) if base_pair > 0 else 0.0
                )
                sec["pair_pa_mpjpe_improved"] = bool(ref_pair < base_pair)

            # PCK@0.05
            base_pck = base_cat.get("pck05")
            ref_pck = ref_cat.get("pck05")
            if base_pck is not None and ref_pck is not None:
                sec["pck05_baseline"] = float(base_pck)
                sec["pck05_refined"] = float(ref_pck)
                sec["pck05_improvement_pct"] = float(ref_pck - base_pck)
                sec["pck05_improved"] = bool(ref_pck > base_pck)

            # Pelvis Vector Error
            base_pel = base_cat.get("pelvis_vector_err_mm")
            ref_pel = ref_cat.get("pelvis_vector_err_mm")
            if base_pel is not None and ref_pel is not None:
                sec["pelvis_vector_baseline_mm"] = float(base_pel)
                sec["pelvis_vector_refined_mm"] = float(ref_pel)
                sec["pelvis_vector_improvement_mm"] = float(base_pel - ref_pel)
                sec["pelvis_vector_improved"] = bool(ref_pel < base_pel)

            # MPJPE
            base_mpjpe = base_cat.get("mpjpe_mm")
            ref_mpjpe = ref_cat.get("mpjpe_mm")
            if base_mpjpe is not None and ref_mpjpe is not None:
                sec["mpjpe_baseline_mm"] = float(base_mpjpe)
                sec["mpjpe_refined_mm"] = float(ref_mpjpe)
                sec["mpjpe_improvement_mm"] = float(base_mpjpe - ref_mpjpe)
                sec["mpjpe_improved"] = bool(ref_mpjpe < base_mpjpe)

            return sec

        overall_ref = ref_categories.get("overall", {})
        overall_base = baseline_categories.get("overall", {})
        overall_comp = _calc_comparison_section(overall_ref, overall_base)

        per_cat_comp: Dict[str, Any] = {}
        for c in ["separated", "light", "hug", "severe"]:
            if c in ref_categories and c in baseline_categories:
                per_cat_comp[c] = _calc_comparison_section(ref_categories[c], baseline_categories[c])

        return {
            "target_baseline": baseline_name,
            "overall": overall_comp,
            "per_category": per_cat_comp,
        }

    # API Alias
    compare_with_baseline = compare_against_baseline

    def format_comparison_table(
        self,
        results_map: Dict[str, Dict[str, Any]],
        title: str = "HI4D DUAL-PERSON 3D/2D BENCHMARK OVERALL COMPARISON",
    ) -> str:
        """Renders formatted ASCII summary table matching evaluate_hi4d_benchmark.py.

        Args:
            results_map: Dictionary mapping model names to evaluation summaries.
            title: Header title of the table.

        Returns:
            Formatted ASCII table string.
        """
        lines: List[str] = []
        width = 110
        lines.append("=" * width)
        lines.append(title.center(width))
        lines.append("=" * width)

        header = (
            f"{'Model':<28} | {'Det>=2%':<8} | {'Exact2%':<8} | "
            f"{'PCK@0.05':<9} | {'PCK@0.10':<9} | {'MPJPE':<9} | "
            f"{'PA-MPJPE':<9} | {'Pair-PA':<9} | {'PelvisVec':<9}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for mname, res in results_map.items():
            cats = res.get("categories", res)
            ov = cats.get("overall", {})

            ge2 = ov.get("detected_ge2_rate", 0.0)
            exact2 = ov.get("detected_exact2_rate", 0.0)
            pck05 = ov.get("pck05", 0.0)
            pck10 = ov.get("pck10", 0.0)
            mpjpe = ov.get("mpjpe_mm")
            pa = ov.get("pa_mpjpe_mm")
            pair_pa = ov.get("pair_pa_mpjpe_mm")
            pelvis = ov.get("pelvis_vector_err_mm")

            ge2_str = f"{ge2 * 100:.1f}%"
            exact2_str = f"{exact2 * 100:.1f}%"
            pck05_str = f"{pck05:.1f}%"
            pck10_str = f"{pck10:.1f}%"
            mpjpe_str = f"{mpjpe:.1f}" if mpjpe is not None else "N/A"
            pa_str = f"{pa:.1f}" if pa is not None else "N/A"
            pair_pa_str = f"{pair_pa:.1f}" if pair_pa is not None else "N/A"
            pelvis_str = f"{pelvis:.1f}" if pelvis is not None else "N/A"

            row = (
                f"{mname:<28} | {ge2_str:<8} | {exact2_str:<8} | "
                f"{pck05_str:<9} | {pck10_str:<9} | {mpjpe_str:<9} | "
                f"{pa_str:<9} | {pair_pa_str:<9} | {pelvis_str:<9}"
            )
            lines.append(row)

        lines.append("=" * width)
        return "\n".join(lines)

    format_summary_table = format_comparison_table

    def save_report(
        self,
        report_data: Dict[str, Any],
        output_path: str,
        comparison: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serializes benchmark results to JSON adhering to the specified schema.

        Ensures all NumPy numeric types, bools, and arrays are recursively converted
        to JSON-compliant Python native types.

        Args:
            report_data: Comprehensive results dictionary.
            output_path: Path where the output JSON will be written.
            comparison: Optional baseline comparison dictionary.
        """
        def _clean_obj(obj: Any) -> Any:
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {str(k): _clean_obj(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_clean_obj(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, float)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            return obj

        cleaned_data = _clean_obj(report_data)
        if comparison is not None:
            cleaned_data["comparison"] = _clean_obj(comparison)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(cleaned_data, f, indent=2)


# -----------------------------------------------------------------------------
# API Contract Alias
# -----------------------------------------------------------------------------
BenchmarkRunner = Hi4DBenchmarkRunner


# -----------------------------------------------------------------------------
# Standalone CLI Interface
# -----------------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Hi4D Dual-Person Benchmark Runner & Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/shangguanrz/pic-edit-experiments/mete-test/dual_pose_compare_20261029_v1/data/request.json",
        help="Path to benchmark ground truth request.json file.",
    )
    parser.add_argument(
        "--use_pilot",
        action="store_true",
        help="Restricts evaluation to the balanced 12-frame pilot subset.",
    )
    parser.add_argument(
        "--pilot_ids",
        type=str,
        default=None,
        help="Optional comma-separated list of frame IDs to override the pilot set.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "live"],
        default="offline",
        help="Execution mode: 'offline' evaluates precomputed predictions; 'live' runs inference.",
    )
    parser.add_argument(
        "--predictions_path",
        "--predictions_pkl",
        dest="predictions_path",
        type=str,
        default=None,
        help="Path to precomputed predictions pickle or JSON file (for offline mode).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="joint_smplx",
        choices=["joint_smplx", "joint_optimization", "multihmr2_anny", "multihmr_smplx", "sapiens_json"],
        help="Prediction format identifier.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON summary report destination path.",
    )
    parser.add_argument(
        "--baseline_compare",
        action="store_true",
        default=True,
        help="Compute side-by-side comparison against MultiHMR2 baseline.",
    )
    parser.add_argument(
        "--no_baseline_compare",
        dest="baseline_compare",
        action="store_false",
        help="Disable baseline comparison.",
    )
    parser.add_argument(
        "--baseline_path",
        "--baseline_pkl",
        dest="baseline_path",
        type=str,
        default="/data/shangguanrz/models/two-man/benchmark_results/M4_MultiHMR2/predictions.pkl",
        help="Path to MultiHMR2 baseline predictions pickle file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Computation device ('cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose per-frame and per-sequence breakdown.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    runner = Hi4DBenchmarkRunner(device=args.device)

    # 1. Parse pilot frame IDs
    pilot_id_list = None
    if args.pilot_ids:
        pilot_id_list = [pid.strip() for pid in args.pilot_ids.split(",") if pid.strip()]

    # 2. Load dataset records
    try:
        records = runner.load_dataset(
            dataset_path=args.dataset_path,
            use_pilot=args.use_pilot,
            pilot_ids=pilot_id_list,
        )
    except Exception as e:
        sys.stderr.write(f"Error loading dataset: {e}\n")
        return 1

    print(
        f"Loaded {len(records)} benchmark records "
        f"({'12-frame pilot' if args.use_pilot else 'full dataset'})."
    )

    # 3. Execute evaluation
    if args.mode == "offline":
        if not args.predictions_path:
            sys.stderr.write(
                "Error: Offline mode requires --predictions_path or --predictions_pkl.\n"
            )
            return 2
        try:
            results = runner.run_offline(
                records=records,
                predictions_path_or_map=args.predictions_path,
                model_type=args.model_type,
            )
        except Exception as e:
            sys.stderr.write(f"Error executing offline evaluation: {e}\n")
            return 3
    else:
        sys.stderr.write("Live pipeline mode requires pipeline integration (Phase 5).\n")
        return 4

    evaluated_model_name = (
        "Sapiens2_MultiHMR_Joint"
        if args.model_type in ("joint_smplx", "joint_optimization")
        else f"Model_{args.model_type}"
    )

    all_models_summary: Dict[str, Dict[str, Any]] = {
        evaluated_model_name: results
    }

    # 4. Optional baseline comparison
    comparison_data: Optional[Dict[str, Any]] = None
    if args.baseline_compare:
        baseline_summary = None
        if args.baseline_path and os.path.exists(args.baseline_path):
            try:
                print(f"Loading baseline from: {args.baseline_path} ...")
                baseline_res = runner.run_offline(
                    records=records,
                    predictions_path_or_map=args.baseline_path,
                    model_type="multihmr2_anny",
                )
                all_models_summary["MultiHMR2 (Baseline)"] = baseline_res
                baseline_summary = baseline_res
            except Exception as e:
                print(f"Warning: Failed to load baseline from '{args.baseline_path}': {e}")

        comparison_data = runner.compare_against_baseline(
            refined_summary=results,
            baseline_summary=baseline_summary,
            baseline_name="MultiHMR2",
        )

    # 5. Format and print ASCII summary table
    table_str = runner.format_comparison_table(all_models_summary)
    print("\n" + table_str + "\n")

    if args.verbose:
        print("Per-Category Metrics:")
        for cat, cat_data in results["categories"].items():
            print(
                f"  [{cat:<9}] Frames: {cat_data['total_frames']:<3} | "
                f"PA-MPJPE: {str(cat_data['pa_mpjpe_mm']):<6} mm | "
                f"Pair-PA: {str(cat_data['pair_pa_mpjpe_mm']):<6} mm | "
                f"PCK@0.05: {cat_data['pck05']:.1f}%"
            )

    # 6. Save structured output report
    out_path = args.output_json
    if out_path is None:
        out_dir = "/data/shangguanrz/models/two-man/benchmark_results/"
        tag = "pilot" if args.use_pilot else "full"
        out_path = os.path.join(out_dir, f"benchmark_{args.model_type}_{tag}.json")

    report_payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": args.dataset_path,
            "is_pilot": args.use_pilot,
            "num_frames": len(records),
            "evaluated_models": list(all_models_summary.keys()),
            "joint_names": BODY_JOINT_NAMES,
        },
        "models": all_models_summary,
    }
    if comparison_data is not None:
        report_payload["comparison"] = comparison_data

    try:
        runner.save_report(report_payload, out_path)
        print(f"Saved benchmark summary report to: {out_path}")
    except Exception as e:
        sys.stderr.write(f"Error saving report to '{out_path}': {e}\n")
        return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())
