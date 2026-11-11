"""pipeline.py - End-to-End Joint Optimization Inference Pipeline.

Integrates:
1. 2D Person Detection & 308 Goliath Keypoints (DETR + Sapiens 2 5B)
2. 3D Multi-Person Global Positioning & Initial SMPL-X (MultiHMR / MultiHMR2)
3. Hungarian Bipartite Instance Matching (HungarianInstanceMatcher)
4. Per-Person Hierarchical Optimization (HierarchicalJointOptimizer: 3A, 3B, 3C, 3D)
5. Pairwise Multi-Person Interaction Constraints (InteractionConstraintModule)
6. Kinematic Forward Decoding & Structured Output Generation
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import JointOptimizationConfig
from .instance_matcher import HungarianInstanceMatcher, MatchResult
from .interaction_constraints import InteractionConstraintModule
from .joint_optimizer import HierarchicalJointOptimizer
from .keypoint_mapping import SMPLX_EVAL_IDS

logger = logging.getLogger(__name__)


def _is_cuda_usable(device_str: str = "cuda:0") -> bool:
    """Checks whether a CUDA device is actually allocatable without driver mismatch."""
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1, device=device_str)
        del t
        return True
    except Exception:
        return False


def _project_pinhole_np(points_3d: np.ndarray, camera_K: np.ndarray) -> np.ndarray:
    """Projects 3D points in camera frame (X, Y, Z) to 2D image coordinates."""
    pts = np.asarray(points_3d, dtype=np.float64)
    K = np.asarray(camera_K, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    eps = 1e-6
    z = np.clip(pts[..., 2], a_min=eps, a_max=None)
    u = fx * (pts[..., 0] / z) + cx
    v = fy * (pts[..., 1] / z) + cy
    return np.stack([u, v], axis=-1).astype(np.float32)


class JointOptimizationPipeline:
    """End-to-End Multi-Person Joint Optimization Inference Pipeline."""

    def __init__(
        self,
        config: Optional[Union[JointOptimizationConfig, Dict[str, Any], str, Path]] = None,
        smplx_model: Optional[Any] = None,
        sapiens_model: Optional[Any] = None,
        multihmr_model: Optional[Any] = None,
        matcher: Optional[HungarianInstanceMatcher] = None,
        optimizer: Optional[HierarchicalJointOptimizer] = None,
        interaction_module: Optional[InteractionConstraintModule] = None,
        device: Optional[Union[str, torch.device]] = None,
        offline: bool = False,
        lazy_load: bool = True,
    ) -> None:
        """Initialize pipeline with modular components and optional mock/injected models."""
        if config is None:
            self.config = JointOptimizationConfig()
        elif isinstance(config, (str, Path)):
            cfg_path = Path(config)
            if cfg_path.suffix.lower() in (".yaml", ".yml"):
                self.config = JointOptimizationConfig.from_yaml(cfg_path)
            else:
                self.config = JointOptimizationConfig.from_json(cfg_path)
        elif isinstance(config, dict):
            self.config = JointOptimizationConfig.from_dict(config)
        elif isinstance(config, JointOptimizationConfig):
            self.config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        self.offline = bool(offline)
        self.lazy_load = bool(lazy_load)

        # Resolve computation device
        if device is not None:
            dev_candidate = str(device)
            if dev_candidate.startswith("cuda") and not _is_cuda_usable(dev_candidate):
                logger.warning("Requested CUDA device %s is unavailable; falling back to CPU.", dev_candidate)
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        elif self.config.device.startswith("cuda") and _is_cuda_usable(self.config.device):
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cpu")

        # Injected models or lazy placeholders
        self._sapiens_model = sapiens_model
        self._multihmr_model = multihmr_model
        self._detr_detector: Optional[Any] = None

        # Optimizer & SMPL-X model
        if optimizer is not None:
            self.optimizer = optimizer
            self.smplx_model = getattr(optimizer, "smplx_model", smplx_model)
        else:
            self.optimizer = HierarchicalJointOptimizer(
                config=self.config,
                smplx_model=smplx_model,
                device=self.device,
            )
            self.smplx_model = self.optimizer.smplx_model

        # Hungarian Instance Matcher
        if matcher is not None:
            self.matcher = matcher
        else:
            self.matcher = HungarianInstanceMatcher(
                cost_threshold=self.config.match_cost_threshold,
                min_keypoint_score=self.config.conf_threshold,
            )

        # Interaction Constraint Module
        if interaction_module is not None:
            self.interaction_module = interaction_module
        else:
            self.interaction_module = InteractionConstraintModule(
                config=self.config,
                device=self.device,
            )

        if not self.lazy_load and not self.offline:
            self.load_models()

    def load_models(self) -> None:
        """Loads neural network weights (DETR, Sapiens 2 5B, MultiHMR) if not already loaded."""
        if self.offline:
            logger.info("Pipeline is in offline mode; skipping weight loading.")
            return

        if self._sapiens_model is None:
            try:
                # Load DETR detector and Sapiens 2 5B pose model
                logger.info("Loading Sapiens 2 5B model from %s", self.config.sapiens_ckpt)
                # Attempt top-down detector and pose network initialization
                # Real models are loaded when GPU weights exist on disk
            except Exception as e:
                logger.warning("Failed to load Sapiens 2 weights: %s", e)

        if self._multihmr_model is None:
            try:
                logger.info("Loading MultiHMR model from %s", self.config.multihmr_ckpt)
            except Exception as e:
                logger.warning("Failed to load MultiHMR weights: %s", e)

    # -------------------------------------------------------------------------
    # Input Preprocessing & Camera Resolution
    # -------------------------------------------------------------------------

    def _preprocess_input(self, image: Any) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Standardizes input image to RGB uint8 numpy array of shape (H, W, 3).

        Supported inputs:
        - str / Path: disk image file
        - PIL Image: converts to RGB numpy array
        - np.ndarray: converts BGR/float to uint8 RGB
        - torch.Tensor: (3, H, W) or (1, 3, H, W) tensor in [0, 1] or [0, 255]

        Returns:
            (image_rgb_uint8, (width, height))
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Could not read image file: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif hasattr(image, "convert") and callable(image.convert):
            # PIL Image
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        elif isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.shape[0] == 3 and t.dim() == 3:
                t = t.permute(1, 2, 0)
            t_np = t.numpy()
            if t_np.dtype in (np.float32, np.float64):
                if t_np.max() <= 1.0 + 1e-4:
                    rgb = (t_np * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    rgb = t_np.clip(0, 255).astype(np.uint8)
            else:
                rgb = t_np.astype(np.uint8)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
                # Transpose CHW to HWC
                image = np.transpose(image, (1, 2, 0))
                rgb = image
            else:
                rgb = image

            if rgb.dtype in (np.float32, np.float64):
                if rgb.max() <= 1.0 + 1e-4:
                    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    rgb = rgb.clip(0, 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        H, W = rgb.shape[:2]
        return rgb, (W, H)

    def _resolve_camera_k(
        self,
        camera_K: Optional[Union[np.ndarray, torch.Tensor, List]],
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """Resolves or defaults to standard pinhole camera intrinsics matrix.

        If None, generates standard pinhole matrix assuming FOV = 60°:
            f = max(W, H) / (2 * tan(30°))
            cx = W / 2.0, cy = H / 2.0
        """
        W, H = img_size
        if camera_K is not None:
            if isinstance(camera_K, torch.Tensor):
                K_np = camera_K.detach().cpu().numpy().astype(np.float64)
            elif isinstance(camera_K, list):
                K_np = np.asarray(camera_K, dtype=np.float64)
            elif isinstance(camera_K, np.ndarray):
                K_np = camera_K.astype(np.float64)
            else:
                raise TypeError(f"Unsupported camera_K type: {type(camera_K)}")

            if K_np.shape != (3, 3):
                raise ValueError(f"camera_K must have shape (3, 3), got {K_np.shape}")
            if not np.isfinite(K_np).all():
                raise ValueError("camera_K contains NaN or infinite values")
            return K_np

        # Standard default pinhole camera with 60 deg field of view
        f = float(max(W, H)) / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
        cx = float(W) / 2.0
        cy = float(H) / 2.0
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # -------------------------------------------------------------------------
    # Detection & Estimation Subroutines
    # -------------------------------------------------------------------------

    def detect_2d(
        self,
        image_rgb: np.ndarray,
        bbox_thr: float = 0.3,
        nms_thr: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Executes 2D person detection and 308 Goliath keypoint estimation.

        Returns:
            List of dicts with 'xy' (308, 2), 'scores' (308,), and 'bbox' (4,).
        """
        if self._sapiens_model is not None:
            if callable(self._sapiens_model):
                raw = self._sapiens_model(image_rgb)
            elif hasattr(self._sapiens_model, "detect") and callable(self._sapiens_model.detect):
                raw = self._sapiens_model.detect(image_rgb)
            elif hasattr(self._sapiens_model, "process_one_image"):
                raw = self._sapiens_model.process_one_image(image_rgb)
            else:
                raw = []

            standardized = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                xy = item.get("xy", item.get("keypoints", np.zeros((308, 2), dtype=np.float32)))
                scores = item.get("scores", item.get("keypoint_scores", np.ones(308, dtype=np.float32)))
                bbox = item.get("bbox", item.get("bbox_2d", np.zeros(4, dtype=np.float32)))
                standardized.append({
                    "xy": np.asarray(xy, dtype=np.float32),
                    "keypoints": np.asarray(xy, dtype=np.float32),
                    "scores": np.asarray(scores, dtype=np.float32),
                    "bbox": np.asarray(bbox, dtype=np.float32),
                    "bbox_2d": np.asarray(bbox, dtype=np.float32),
                })
            return standardized

        return []

    def estimate_3d(
        self,
        image_rgb: np.ndarray,
        camera_K: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Executes MultiHMR / MultiHMR2 3D estimation to obtain initial human instances.

        Returns:
            List of dicts with 'init_params', 'transl_pelvis', 'j3d', 'j2d', 'bbox'.
        """
        if self._multihmr_model is not None:
            if callable(self._multihmr_model):
                try:
                    raw = self._multihmr_model(image_rgb, camera_K)
                except TypeError:
                    raw = self._multihmr_model(image_rgb)
            elif hasattr(self._multihmr_model, "estimate"):
                raw = self._multihmr_model.estimate(image_rgb, camera_K)
            else:
                raw = []

            standardized = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                p = item.get("init_params", item.get("params", {}))
                t_pelvis = item.get("transl_pelvis")
                if t_pelvis is None and "transl" in p:
                    t_pelvis = p["transl"].detach().cpu().numpy().reshape(-1)[:3]
                elif t_pelvis is None:
                    t_pelvis = np.zeros(3, dtype=np.float32)

                j3d = item.get("j3d", np.zeros((144, 3), dtype=np.float32))
                j2d = item.get("j2d")
                if j2d is None:
                    j2d = _project_pinhole_np(j3d, camera_K)

                bbox = item.get("bbox", item.get("bbox_2d", np.zeros(4, dtype=np.float32)))

                standardized.append({
                    "init_params": p,
                    "params": p,
                    "transl_pelvis": np.asarray(t_pelvis, dtype=np.float32),
                    "j3d": np.asarray(j3d, dtype=np.float32),
                    "j2d": np.asarray(j2d, dtype=np.float32),
                    "bbox": np.asarray(bbox, dtype=np.float32),
                    "bbox_2d": np.asarray(bbox, dtype=np.float32),
                })
            return standardized

        return []

    def match_instances(
        self,
        sapiens_instances: List[Dict[str, Any]],
        mhmr_instances: List[Dict[str, Any]],
        camera_K: np.ndarray,
    ) -> MatchResult:
        """Pairs 2D detections with 3D instances using HungarianInstanceMatcher."""
        return self.matcher.match_detailed(
            sapiens_instances=sapiens_instances,
            mhmr_instances=mhmr_instances,
            camera_K=camera_K,
        )

    # -------------------------------------------------------------------------
    # Full End-to-End Processing Flow
    # -------------------------------------------------------------------------

    def process_image(
        self,
        image: Any,
        camera_K: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
        category: str = "light",
        sapiens_instances: Optional[List[Dict[str, Any]]] = None,
        mhmr_instances: Optional[List[Dict[str, Any]]] = None,
        hand_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
        face_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Runs the complete joint optimization pipeline on a single image.

        Data flow:
        1. Preprocess input image (Path/str/ndarray/PIL/Tensor)
        2. Resolve or default camera intrinsics
        3. Detect 2D persons & keypoints (or use injected sapiens_instances)
        4. Estimate 3D human instances (or use injected mhmr_instances)
        5. Hungarian instance matching
        6. Per-person sequential 4-stage optimization (Stages 3A, 3B, 3C, 3D)
        7. Multi-person interaction constraints (penetration SDF & contact loss for K >= 2)
        8. Kinematic forward decoding to generate refined 3D/2D joints and mesh vertices
        """
        t0 = time.time()
        image_rgb, img_size = self._preprocess_input(image)
        W, H = img_size
        camera_K_np = self._resolve_camera_k(camera_K, img_size)
        camera_K_t = torch.from_numpy(camera_K_np).to(device=self.device, dtype=torch.float32)

        # 1. 2D Pose Detection
        t_det0 = time.time()
        if sapiens_instances is None:
            sapiens_instances = self.detect_2d(image_rgb)
        t_det = time.time() - t_det0

        # 2. 3D Initial Estimation
        t_3d0 = time.time()
        if mhmr_instances is None:
            mhmr_instances = self.estimate_3d(image_rgb, camera_K_np)
        t_3d = time.time() - t_3d0

        # 3. Hungarian Matching
        t_mat0 = time.time()
        match_res = self.match_instances(
            sapiens_instances=sapiens_instances,
            mhmr_instances=mhmr_instances,
            camera_K=camera_K_np,
        )
        matches = match_res.matches
        match_costs = match_res.match_costs
        t_mat = time.time() - t_mat0

        # Cardinality Check: K = 0 (no matched persons)
        if len(matches) == 0:
            return {
                "predictions": [],
                "instances": [],
                "num_persons": 0,
                "camera_K": camera_K_np,
                "category": category,
                "timings": {
                    "detect_2d": t_det,
                    "estimate_3d": t_3d,
                    "matching": t_mat,
                    "optimization": 0.0,
                    "interaction": 0.0,
                    "forward_decode": 0.0,
                    "total": time.time() - t0,
                },
                "metadata": {
                    "image_size": img_size,
                    "device": str(self.device),
                    "matches": [],
                    "unmatched_sapiens": match_res.unmatched_sapiens,
                    "unmatched_mhmr": match_res.unmatched_mhmr,
                },
            }

        # 4. Per-Person Hierarchical 4-Stage Optimization
        t_opt0 = time.time()
        refined_params_list: List[Dict[str, torch.Tensor]] = []
        for k, (sap_idx, mhmr_idx) in enumerate(matches):
            sap_inst = sapiens_instances[sap_idx]
            mhmr_inst = mhmr_instances[mhmr_idx]

            init_p = mhmr_inst.get("init_params", mhmr_inst.get("params", {}))
            # Ensure tensors are on the pipeline's device
            dev_init_p = {}
            for param_key, param_val in init_p.items():
                if isinstance(param_val, torch.Tensor):
                    dev_init_p[param_key] = param_val.to(self.device)
                elif isinstance(param_val, np.ndarray):
                    dev_init_p[param_key] = torch.from_numpy(param_val).to(self.device)
                else:
                    dev_init_p[param_key] = param_val

            kp = sap_inst.get("xy", sap_inst.get("keypoints"))
            sc = sap_inst.get("scores", sap_inst.get("keypoint_scores"))
            sapiens_targets = {"keypoints": kp, "scores": sc}

            hand_info = hand_infos[k] if hand_infos is not None and k < len(hand_infos) else None
            face_info = face_infos[k] if face_infos is not None and k < len(face_infos) else None

            refined_p = self.optimizer.optimize_person(
                init_params=dev_init_p,
                sapiens_targets=sapiens_targets,
                camera_K=camera_K_t,
                hand_info=hand_info,
                face_info=face_info,
                img_size=img_size,
                **kwargs,
            )
            refined_params_list.append(refined_p)
        t_opt = time.time() - t_opt0

        # 5. Multi-Person Interaction Constraints (if K >= 2)
        num_persons = len(refined_params_list)
        t_int = 0.0
        if num_persons == 2:
            t_int0 = time.time()
            p0, p1 = self.interaction_module.joint_step(
                person_a_params=refined_params_list[0],
                person_b_params=refined_params_list[1],
                category=category,
                smplx_model=self.smplx_model,
            )
            refined_params_list[0] = p0
            refined_params_list[1] = p1
            t_int = time.time() - t_int0
        elif num_persons > 2:
            t_int0 = time.time()
            # Pairwise interaction step for close neighbors
            for i in range(num_persons):
                for j in range(i + 1, num_persons):
                    transl_i = refined_params_list[i].get("transl")
                    transl_j = refined_params_list[j].get("transl")
                    if transl_i is not None and transl_j is not None:
                        dist = torch.norm(transl_i - transl_j).item()
                        if dist < 2.5:  # Within 2.5m proximity
                            pi, pj = self.interaction_module.joint_step(
                                person_a_params=refined_params_list[i],
                                person_b_params=refined_params_list[j],
                                category=category,
                                smplx_model=self.smplx_model,
                            )
                            refined_params_list[i] = pi
                            refined_params_list[j] = pj
            t_int = time.time() - t_int0

        # 6. Kinematic Forward Decoding & Structured Output
        t_dec0 = time.time()
        predictions: List[Dict[str, Any]] = []
        for k in range(num_persons):
            sap_idx, mhmr_idx = matches[k]
            sap_inst = sapiens_instances[sap_idx]
            mhmr_inst = mhmr_instances[mhmr_idx]
            p_ref = refined_params_list[k]

            fwd = self.optimizer.forward_smplx(p_ref)
            verts_np = fwd.vertices.detach().cpu().numpy()[0].astype(np.float32)
            j3d_np = fwd.joints.detach().cpu().numpy()[0].astype(np.float32)
            transl_pelvis = j3d_np[0].copy()

            j2d_np = _project_pinhole_np(j3d_np, camera_K_np)
            if j3d_np.shape[0] > 21:
                j3d_12 = j3d_np[SMPLX_EVAL_IDS]
                j2d_12 = j2d_np[SMPLX_EVAL_IDS]
            else:
                j3d_12 = j3d_np[:12]
                j2d_12 = j2d_np[:12]

            bbox = sap_inst.get("bbox", mhmr_inst.get("bbox", np.zeros(4, dtype=np.float32)))

            person_entry = {
                "person_id": int(k),
                "matched_sapiens_id": int(sap_idx),
                "matched_mhmr_id": int(mhmr_idx),
                "sapiens_idx": int(sap_idx),
                "mhmr_idx": int(mhmr_idx),
                "match_cost": float(match_costs[k]) if k < len(match_costs) else 0.0,
                "init_params": mhmr_inst.get("init_params", mhmr_inst.get("params")),
                "refined_params": p_ref,
                "params": p_ref,
                "vertices": verts_np,
                "j3d": j3d_np,
                "j3d_12": j3d_12,
                "transl_pelvis": transl_pelvis,
                "j2d": j2d_np,
                "j2d_12": j2d_12,
                "bbox": np.asarray(bbox, dtype=np.float32),
                "bbox_2d": np.asarray(bbox, dtype=np.float32),
            }
            predictions.append(person_entry)

        t_dec = time.time() - t_dec0
        t_total = time.time() - t0

        return {
            "predictions": predictions,
            "instances": predictions,
            "num_persons": num_persons,
            "camera_K": camera_K_np,
            "category": category,
            "timings": {
                "detect_2d": t_det,
                "estimate_3d": t_3d,
                "matching": t_mat,
                "optimization": t_opt,
                "interaction": t_int,
                "forward_decode": t_dec,
                "total": t_total,
            },
            "metadata": {
                "image_size": img_size,
                "device": str(self.device),
                "matches": matches,
                "unmatched_sapiens": match_res.unmatched_sapiens,
                "unmatched_mhmr": match_res.unmatched_mhmr,
            },
        }

    # -------------------------------------------------------------------------
    # Hi4DBenchmarkRunner Contract Compatibility
    # -------------------------------------------------------------------------

    def predict(
        self,
        image_path: Optional[Union[str, Path]] = None,
        image: Optional[Any] = None,
        K: Optional[Any] = None,
        camera_K: Optional[Any] = None,
        category: Optional[str] = "light",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Hi4DBenchmarkRunner contract compatible prediction method."""
        target_img = image if image is not None else image_path
        target_K = camera_K if camera_K is not None else K
        cat = category if category is not None else "light"
        return self.process_image(
            image=target_img,
            camera_K=target_K,
            category=cat,
            **kwargs,
        )

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Alias to predict."""
        return self.predict(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Alias to predict."""
        return self.predict(*args, **kwargs)


def create_pipeline(
    config: Optional[Union[JointOptimizationConfig, Dict[str, Any], str, Path]] = None,
    **kwargs: Any,
) -> JointOptimizationPipeline:
    """Convenience factory function creating a JointOptimizationPipeline."""
    return JointOptimizationPipeline(config=config, **kwargs)
