"""Centralized Configuration Module for Sapiens 2 x MultiHMR Joint Optimization.

Defines JointOptimizationConfig dataclass with full hyperparameter defaults for
Stages 3A, 3B, 3C, 3D and Phase 3 interaction constraints, parameter validation,
and bidirectional JSON / YAML serialization.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Tuple, Union
import yaml


@dataclass
class JointOptimizationConfig:
  """Central configuration and hyperparameter container for the joint optimization pipeline."""

  # -----------------------------------------------------------------------
  # Optimization Step Counts
  # -----------------------------------------------------------------------
  iters_3a: int = 20  # Stage 3A: Global positioning
  iters_3b: int = 40  # Stage 3B: Torso & limbs body pose
  iters_3c: int = 50  # Stage 3C: Hand pose refinement
  iters_3d: int = 30  # Stage 3D: Facial expression refinement

  # -----------------------------------------------------------------------
  # Learning Rates
  # -----------------------------------------------------------------------
  lr_transl: float = 0.005  # Root translation
  lr_global_orient: float = 0.015  # Global orientation
  lr_body: float = 0.015  # 63-dim body pose
  lr_hand: float = 0.025  # 45-dim left/right hand poses
  lr_face: float = 0.02  # Jaw pose and expression coefficients

  # -----------------------------------------------------------------------
  # Loss Weights
  # -----------------------------------------------------------------------
  w_reproj: float = 1.0  # 2D keypoint reprojection loss weight
  w_transl_reg: float = 0.1  # Stage 3A translation prior regularization
  w_global_reg: float = 0.1  # Stage 3A global orientation prior regularization
  w_body_reg: float = 0.015  # Stage 3B initial body pose regularization
  w_hand_wilor: float = 0.01  # Stage 3C WiLoR 3D MANO target loss weight
  w_hand_flat: float = 0.005  # Stage 3C flat hand prior regularization
  w_face_emica: float = 0.1  # Stage 3D EMICA expression target loss weight
  w_face_reg: float = 0.01  # Stage 3D expression L2 regularization

  # -----------------------------------------------------------------------
  # Clamping Bounds & Ranges
  # -----------------------------------------------------------------------
  hand_delta_max: float = 0.8  # Max rotation delta for hand pose (radians)
  jaw_min_max: Tuple[float, float] = (
      -0.6,
      0.6,
  )  # Jaw angle clamping bounds (radians)
  expr_min_max: Tuple[float, float] = (
      -3.0,
      3.0,
  )  # Expression coefficient clamping bounds

  # -----------------------------------------------------------------------
  # Thresholds & Tolerances
  # -----------------------------------------------------------------------
  conf_threshold: float = 0.3  # Sapiens keypoint cutoff threshold
  sdf_tau: float = 0.005  # Penetration clearance distance (5 mm in meters)
  match_cost_threshold: float = (
      1.5  # Hungarian instance matching rejection threshold
  )

  # -----------------------------------------------------------------------
  # Dynamic Multi-Person Interaction Weights (Hi4D Categories)
  # -----------------------------------------------------------------------
  penetration_weights: Dict[str, float] = field(
      default_factory=lambda: {
          "separated": 0.1,
          "light": 0.5,
          "hug": 1.0,
          "severe": 2.0,
      }
  )
  contact_weights: Dict[str, float] = field(
      default_factory=lambda: {
          "separated": 0.0,
          "light": 0.1,
          "hug": 0.5,
          "severe": 1.0,
      }
  )

  # -----------------------------------------------------------------------
  # Device & Runtime Hardware
  # -----------------------------------------------------------------------
  device: str = "cuda:0"

  # -----------------------------------------------------------------------
  # Model Directories & Checkpoint Paths
  # -----------------------------------------------------------------------
  smplx_model_dir: str = (
      "/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files"
  )
  smplx_faces_path: str = (
      "/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"
  )
  sapiens_code_dir: str = "/data/shangguanrz/models/meta/sapiens2-code/"
  sapiens_ckpt: str = (
      "/data/shangguanrz/models/meta/sapiens2-pose-5b/sapiens2_5b_pose.safetensors"
  )
  sapiens_config: str = (
      "/data/shangguanrz/models/meta/sapiens2-code/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py"
  )
  sapiens_meta: str = (
      "/data/shangguanrz/models/meta/sapiens2-code/sapiens/pose/configs/_base_/keypoints308.py"
  )
  detr_model_dir: str = (
      "/data/shangguanrz/models/meta/detector/detr-resnet-101-dc5"
  )
  multihmr_ckpt: str = "/data/shangguanrz/models/two-man/M2_MultiHMR/"
  multihmr2_ckpt: str = "/data/shangguanrz/models/two-man/M4_MultiHMR2/"
  wilor_ckpt: str = "/data/shangguanrz/pic-edit-datasets-test/HIA-wilor/"
  emica_ckpt: str = (
      "/home/shangguanrz/project/pic-edit/experiments/8.HIA/smplx_wilor_emica_refinement_20261101/models/emica/"
  )
  hi4d_request_json: str = (
      "/data/shangguanrz/pic-edit-experiments/mete-test/dual_pose_compare_20261029_v1/data/request.json"
  )
  hi4d_benchmark_script: str = (
      "/data/shangguanrz/models/two-man/evaluate_hi4d_benchmark.py"
  )
  output_dir: str = (
      "/data/shangguanrz/models/two-man/sapiens2_multihmr_joint/benchmark_results/"
  )

  def validate(self) -> None:
    """Validates all configuration parameters, types, value bounds, and integrity.

    Raises ValueError on any invalid configuration setting.
    """
    # 1. Validate iteration counts
    for iter_param in ["iters_3a", "iters_3b", "iters_3c", "iters_3d"]:
      val = getattr(self, iter_param)
      if type(val) is not int or val <= 0:
        raise ValueError(f"{iter_param} must be a positive integer, got {val}")

    # 2. Validate learning rates
    for lr_param in [
        "lr_transl",
        "lr_global_orient",
        "lr_body",
        "lr_hand",
        "lr_face",
    ]:
      val = getattr(self, lr_param)
      if (
          not isinstance(val, (int, float))
          or isinstance(val, bool)
          or not math.isfinite(val)
          or val <= 0.0
      ):
        raise ValueError(f"{lr_param} must be positive, got {val}")

    # 3. Validate loss weights
    for weight_param in [
        "w_reproj",
        "w_transl_reg",
        "w_global_reg",
        "w_body_reg",
        "w_hand_wilor",
        "w_hand_flat",
        "w_face_emica",
        "w_face_reg",
    ]:
      val = getattr(self, weight_param)
      if (
          not isinstance(val, (int, float))
          or isinstance(val, bool)
          or not math.isfinite(val)
          or val < 0.0
      ):
        raise ValueError(f"{weight_param} must be non-negative, got {val}")

    # 4. Validate clamping bounds
    for bound_param in ["jaw_min_max", "expr_min_max"]:
      bounds = getattr(self, bound_param)
      if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
        raise ValueError(
            f"{bound_param} must be a 2-tuple of floats, got {bounds}"
        )
      lower, upper = bounds[0], bounds[1]
      if (
          not isinstance(lower, (int, float))
          or isinstance(lower, bool)
          or not math.isfinite(lower)
          or not isinstance(upper, (int, float))
          or isinstance(upper, bool)
          or not math.isfinite(upper)
      ):
        raise ValueError(
            f"{bound_param} elements must be finite numbers, got {bounds}"
        )
      if lower >= upper:
        raise ValueError(
            f"{bound_param} lower bound must be less than upper bound: {bounds}"
        )

    if (
        not isinstance(self.hand_delta_max, (int, float))
        or isinstance(self.hand_delta_max, bool)
        or not math.isfinite(self.hand_delta_max)
        or self.hand_delta_max <= 0.0
    ):
      raise ValueError(
          f"hand_delta_max must be positive, got {self.hand_delta_max}"
      )

    # 5. Validate device
    if not (
        isinstance(self.device, str)
        and re.match(r"^(cpu|cuda(:\d+)?)$", self.device)
    ):
      raise ValueError(
          f"Invalid device string: '{self.device}'. Must be 'cpu', 'cuda', or"
          " 'cuda:X'."
      )

    # 6. Validate thresholds and tolerances
    if (
        not isinstance(self.conf_threshold, (int, float))
        or isinstance(self.conf_threshold, bool)
        or not math.isfinite(self.conf_threshold)
        or not (0.0 <= self.conf_threshold <= 1.0)
    ):
      raise ValueError(
          f"conf_threshold must be in [0.0, 1.0], got {self.conf_threshold}"
      )

    if (
        not isinstance(self.sdf_tau, (int, float))
        or isinstance(self.sdf_tau, bool)
        or not math.isfinite(self.sdf_tau)
        or self.sdf_tau <= 0.0
    ):
      raise ValueError(f"sdf_tau must be positive, got {self.sdf_tau}")

    if (
        not isinstance(self.match_cost_threshold, (int, float))
        or isinstance(self.match_cost_threshold, bool)
        or not math.isfinite(self.match_cost_threshold)
        or self.match_cost_threshold <= 0.0
    ):
      raise ValueError(
          f"match_cost_threshold must be positive, got {self.match_cost_threshold}"
      )

    # 7. Validate interaction categories and weights
    required_categories = {"separated", "light", "hug", "severe"}
    if set(self.penetration_weights.keys()) != required_categories:
      missing = required_categories - set(self.penetration_weights.keys())
      raise ValueError(f"penetration_weights missing categories: {missing}")
    for cat, weight in self.penetration_weights.items():
      if (
          not isinstance(weight, (int, float))
          or isinstance(weight, bool)
          or not math.isfinite(weight)
          or weight < 0.0
      ):
        raise ValueError(
            f"penetration_weights for '{cat}' must be non-negative, got"
            f" {weight}"
        )

    if set(self.contact_weights.keys()) != required_categories:
      missing = required_categories - set(self.contact_weights.keys())
      raise ValueError(f"contact_weights missing categories: {missing}")
    for cat, weight in self.contact_weights.items():
      if (
          not isinstance(weight, (int, float))
          or isinstance(weight, bool)
          or not math.isfinite(weight)
          or weight < 0.0
      ):
        raise ValueError(
            f"contact_weights for '{cat}' must be non-negative, got {weight}"
        )

  def to_dict(self) -> Dict[str, Any]:
    """Converts configuration dataclass to dictionary, converting tuples to lists for JSON compliance."""
    data = asdict(self)
    data["jaw_min_max"] = list(self.jaw_min_max)
    data["expr_min_max"] = list(self.expr_min_max)
    return data

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "JointOptimizationConfig":
    """Instantiates JointOptimizationConfig from a dictionary."""
    d = dict(data)
    if "jaw_min_max" in d and isinstance(d["jaw_min_max"], (list, tuple)):
      d["jaw_min_max"] = tuple(d["jaw_min_max"])
    if "expr_min_max" in d and isinstance(d["expr_min_max"], (list, tuple)):
      d["expr_min_max"] = tuple(d["expr_min_max"])
    cfg = cls(**d)
    cfg.validate()
    return cfg

  def save_json(self, filepath: Union[str, Path]) -> None:
    """Saves configuration to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(self.to_dict(), f, indent=2)

  @classmethod
  def load_json(cls, filepath: Union[str, Path]) -> "JointOptimizationConfig":
    """Loads configuration from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
      data = json.load(f)
    return cls.from_dict(data)

  def save_yaml(self, filepath: Union[str, Path]) -> None:
    """Saves configuration to a YAML file."""
    with open(filepath, "w", encoding="utf-8") as f:
      yaml.safe_dump(self.to_dict(), f, sort_keys=False)

  @classmethod
  def load_yaml(cls, filepath: Union[str, Path]) -> "JointOptimizationConfig":
    """Loads configuration from a YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
      data = yaml.safe_load(f)
    return cls.from_dict(data)

  # Aliases for API flexibility
  to_json = save_json
  from_json = load_json
  to_yaml = save_yaml
  from_yaml = load_yaml
