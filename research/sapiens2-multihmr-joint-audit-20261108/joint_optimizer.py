"""Hierarchical Joint Optimizer Module for Sapiens 2 x MultiHMR Joint Optimization.

Implements sequential 4-stage coarse-to-fine optimization for SMPL-X human models:
- Stage 3A: Global positioning (transl and global_orient, 20 steps Adam)
- Stage 3B: Torso & limbs body pose (body_pose, 40 steps Adam) with dynamic
  regularization decay and arm-chain kinematic masking
- Stage 3C: Hand refinement (left_hand_pose and right_hand_pose, 50 steps Adam)
  with WiLoR 3D MANO alignment, visibility gating, and delta clamping [-0.8, 0.8]
- Stage 3D: Facial expression and jaw (jaw_pose and expression, 30 steps Adam)
  with 68-point face contour reprojection, EMICA 3D target, and clamping
- optimize_person: End-to-end single person hierarchical refinement
- optimize_multi_person / optimize_batch: Multi-person independent instance optimization
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import smplx
except ImportError:
    smplx = None

try:
    from .config import JointOptimizationConfig
    from .keypoint_mapping import (
        GOLIATH_TO_SMPLX,
        SMPLX_EVAL_IDS,
        GOLIATH_EVAL_IDS,
        FACE68_GOLIATH_TO_SMPLX,
        compute_adaptive_weights,
        get_arm_chain_mask,
    )
except ImportError:
    from config import JointOptimizationConfig
    from keypoint_mapping import (
        GOLIATH_TO_SMPLX,
        SMPLX_EVAL_IDS,
        GOLIATH_EVAL_IDS,
        FACE68_GOLIATH_TO_SMPLX,
        compute_adaptive_weights,
        get_arm_chain_mask,
    )

# ---------------------------------------------------------------------------
# Keypoint Index Constants
# ---------------------------------------------------------------------------

# Stage 3A: 8 Rigid Torso Anchor Joints (Shoulders, Hips, Knees, Ankles)
STAGE_3A_TORSO_SMPLX: List[int] = [16, 17, 1, 2, 4, 5, 7, 8]
STAGE_3A_TORSO_GOLIATH: List[int] = [5, 6, 9, 10, 11, 12, 13, 14]

# Stage 3B: 12 Hi4D Benchmark Body & Limb Joints
STAGE_3B_BODY_SMPLX: List[int] = SMPLX_EVAL_IDS
STAGE_3B_BODY_GOLIATH: List[int] = GOLIATH_EVAL_IDS

# Stage 3C: Hand Finger Joints (15 per hand)
# Derived bijectively from authoritative GOLIATH_TO_SMPLX mapping
_SMPLX_TO_GOLIATH: Dict[int, int] = {v: k for k, v in GOLIATH_TO_SMPLX.items()}

STAGE_3C_LH_SMPLX: List[int] = list(range(25, 40))
STAGE_3C_LH_GOLIATH: List[int] = [49, 48, 47, 53, 52, 51, 61, 60, 59, 57, 56, 55, 45, 44, 43]

STAGE_3C_RH_SMPLX: List[int] = list(range(40, 55))
STAGE_3C_RH_GOLIATH: List[int] = [28, 27, 26, 32, 31, 30, 40, 39, 38, 36, 35, 34, 24, 23, 22]

# WiLoR 21-joint OpenPose format to SMPL-X 15-joint anatomical format:
# SMPL-X ordering: Index (MCP, PIP, DIP), Middle (MCP, PIP, DIP), Pinky (MCP, PIP, DIP), Ring (MCP, PIP, DIP), Thumb (MCP, PIP, DIP)
# WiLoR 21-joint OpenPose ordering:
#   Thumb (1..4), Index (5..8), Middle (9..12), Ring (13..16), Pinky (17..20)
# WiLoR joints mapped to SMPL-X:
#   Index: [5, 6, 7], Middle: [9, 10, 11], Pinky: [17, 18, 19], Ring: [13, 14, 15], Thumb: [1, 2, 3]
WILOR_TO_SMPLX: List[int] = [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
MANO_TO_SMPLX: List[int] = WILOR_TO_SMPLX  # Backward-compatible alias

# Stage 3D: Sapiens 68 Face Contour Landmark IDs
STAGE_3D_FACE_GOLIATH: List[int] = list(FACE68_GOLIATH_TO_SMPLX.keys())
STAGE_3D_FACE_SMPLX: List[int] = [FACE68_GOLIATH_TO_SMPLX[k] for k in STAGE_3D_FACE_GOLIATH]


class HierarchicalJointOptimizer:
    """Coarse-to-fine hierarchical optimizer refining SMPL-X parameters using Sapiens 2 2D keypoints."""

    def __init__(
        self,
        config: Optional[JointOptimizationConfig] = None,
        smplx_model: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initializes optimizer, device context, and SMPL-X kinematic decoder.

        Args:
            config: Optimization configuration instance. If None, default is used.
            smplx_model: Injected SMPL-X model or mock. If None, instantiates neutral SMPL-X.
            device: Target hardware device. If None, inferred from config/cuda availability.
        """
        if config is None:
            config = JointOptimizationConfig()
            config.validate()
        self.config = config
        self.cfg = config  # Convenience alias

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available() and self.config.device.startswith("cuda"):
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cpu")

        if smplx_model is not None:
            self.smplx_model = smplx_model.to(self.device).eval()
            self.smplx_model.requires_grad_(False)
        else:
            if smplx is None:
                raise ImportError("smplx library is required when smplx_model is not provided.")
            self.smplx_model = smplx.create(
                self.config.smplx_model_dir,
                model_type="smplx",
                gender="neutral",
                use_face_contour=True,
                num_betas=10,
                num_expression_coeffs=10,
                ext="npz",
                use_pca=False,
                flat_hand_mean=False,
                batch_size=1,
            ).to(self.device).eval()
            self.smplx_model.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Forward Pass & Geometry Utilities
    # -----------------------------------------------------------------------

    def forward_smplx(self, params: Dict[str, torch.Tensor]) -> Any:
        """Executes differentiable SMPL-X forward pass.

        Args:
            params: Dictionary containing SMPL-X parameters.

        Returns:
            Output object with .joints and .vertices.
        """
        std_params = self._standardize_params(params)
        return self.smplx_model(
            global_orient=std_params["global_orient"],
            body_pose=std_params["body_pose"],
            left_hand_pose=std_params["left_hand_pose"],
            right_hand_pose=std_params["right_hand_pose"],
            jaw_pose=std_params["jaw_pose"],
            betas=std_params["betas"],
            expression=std_params["expression"],
            transl=std_params["transl"],
        )

    def project_pinhole(
        self,
        joints_3d: torch.Tensor,
        camera_K: torch.Tensor,
        min_depth: float = 1e-4,
    ) -> torch.Tensor:
        """Differentiable pinhole projection with safe positive depth clamping.

        Args:
            joints_3d: (..., 3) 3D joints in camera coordinates.
            camera_K: (3, 3) or (..., 3, 3) camera intrinsics matrix.
            min_depth: Minimum positive depth clamp threshold in meters.

        Returns:
            (..., 2) 2D pixel coordinates.
        """
        if camera_K.dim() == 2:
            fx, fy = camera_K[0, 0], camera_K[1, 1]
            cx, cy = camera_K[0, 2], camera_K[1, 2]
        else:
            fx = camera_K[..., 0, 0]
            fy = camera_K[..., 1, 1]
            cx = camera_K[..., 0, 2]
            cy = camera_K[..., 1, 2]

        x = joints_3d[..., 0]
        y = joints_3d[..., 1]
        z = torch.clamp(joints_3d[..., 2], min=min_depth)

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        return torch.stack([u, v], dim=-1)

    project_joints = project_pinhole

    # -----------------------------------------------------------------------
    # Stage 3A: Global Positioning Optimization (20 Steps)
    # -----------------------------------------------------------------------

    def optimize_stage_3a(
        self,
        current_params: Dict[str, torch.Tensor],
        sapiens_targets: Union[Dict[str, Any], torch.Tensor],
        camera_K: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Stage 3A: Global positioning optimization (20 steps Adam).

        Optimizes transl (lr=0.005) and global_orient (lr=0.015) using 8 rigid
        torso anchor joints with L1 reprojection loss and MultiHMR quadratic priors.
        All other SMPL-X parameters strictly frozen.
        """
        orig_device = self._detect_orig_device(current_params)
        p = self._standardize_params(current_params)
        cam_K = self._standardize_camera_k(camera_K)

        init_transl = p["transl"].clone().detach()
        init_orient = p["global_orient"].clone().detach()

        transl_param = nn.Parameter(init_transl.clone())
        orient_param = nn.Parameter(init_orient.clone())

        optimizer = torch.optim.Adam(
            [
                {"params": [transl_param], "lr": self.config.lr_transl},
                {"params": [orient_param], "lr": self.config.lr_global_orient},
            ]
        )

        target_pts, target_weights = self._extract_stage_targets(
            sapiens_targets, STAGE_3A_TORSO_GOLIATH, region="body", **kwargs
        )
        has_valid_reproj = bool(target_weights.sum().item() > 1e-6)

        best_loss = float("inf")
        best_transl = init_transl.clone()
        best_orient = init_orient.clone()

        for step in range(self.config.iters_3a):
            optimizer.zero_grad()

            out = self.smplx_model(
                global_orient=orient_param,
                body_pose=p["body_pose"],
                left_hand_pose=p["left_hand_pose"],
                right_hand_pose=p["right_hand_pose"],
                jaw_pose=p["jaw_pose"],
                betas=p["betas"],
                expression=p["expression"],
                transl=transl_param,
            )

            proj_torso = self.project_pinhole(
                out.joints[0, STAGE_3A_TORSO_SMPLX], cam_K
            )

            # 1. Reprojection Loss (L1)
            if has_valid_reproj:
                l1_err = torch.norm(proj_torso - target_pts, p=1, dim=-1)
                loss_reproj = (target_weights * l1_err).sum() / (target_weights.sum() + 1e-8)
            else:
                loss_reproj = torch.tensor(0.0, device=self.device)

            # 2. MultiHMR Quadratic Priors
            loss_transl_prior = torch.norm(transl_param - init_transl, p=2).pow(2)
            loss_orient_prior = torch.norm(orient_param - init_orient, p=2).pow(2)

            total_loss = (
                self.config.w_reproj * loss_reproj
                + self.config.w_transl_reg * loss_transl_prior
                + self.config.w_global_reg * loss_orient_prior
            )

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_transl = transl_param.detach().clone()
                best_orient = orient_param.detach().clone()

            total_loss.backward()
            optimizer.step()

        p_out = {k: v.detach().clone() for k, v in p.items()}
        p_out["transl"] = best_transl
        p_out["global_orient"] = best_orient
        return self._restore_device(p_out, orig_device)

    # -----------------------------------------------------------------------
    # Stage 3B: Torso & Limbs Body Pose Optimization (40 Steps)
    # -----------------------------------------------------------------------

    def optimize_stage_3b(
        self,
        current_params: Dict[str, torch.Tensor],
        sapiens_targets: Union[Dict[str, Any], torch.Tensor],
        camera_K: torch.Tensor,
        hand_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Stage 3B: Torso and limbs body pose optimization (40 steps Adam).

        Optimizes body_pose (lr=0.015) using Smooth L1 (beta=0.01) reprojection,
        dynamic regularizer decay schedule, arm-chain masking for occluded limbs,
        and best checkpoint tracking. All other parameters strictly frozen.
        """
        orig_device = self._detect_orig_device(current_params)
        p = self._standardize_params(current_params)
        cam_K = self._standardize_camera_k(camera_K)

        init_body_pose = p["body_pose"].clone().detach()
        body_param = nn.Parameter(init_body_pose.clone())

        optimizer = torch.optim.Adam([body_param], lr=self.config.lr_body)

        target_pts, target_weights = self._extract_stage_targets(
            sapiens_targets, STAGE_3B_BODY_GOLIATH, region="body", **kwargs
        )
        has_valid_reproj = bool(target_weights.sum().item() > 1e-6)

        lock_left_arm, lock_right_arm = self._determine_arm_occlusion(sapiens_targets, hand_info)
        left_mask = get_arm_chain_mask("left").to(self.device) if lock_left_arm else None
        right_mask = get_arm_chain_mask("right").to(self.device) if lock_right_arm else None

        best_loss = float("inf")
        best_body_pose = init_body_pose.clone()

        for t in range(self.config.iters_3b):
            optimizer.zero_grad()

            out = self.smplx_model(
                global_orient=p["global_orient"],
                body_pose=body_param,
                left_hand_pose=p["left_hand_pose"],
                right_hand_pose=p["right_hand_pose"],
                jaw_pose=p["jaw_pose"],
                betas=p["betas"],
                expression=p["expression"],
                transl=p["transl"],
            )

            proj_body = self.project_pinhole(
                out.joints[0, STAGE_3B_BODY_SMPLX], cam_K
            )

            # 1. Smooth L1 Reprojection Loss (beta=0.01)
            if has_valid_reproj:
                smooth_l1_err = F.smooth_l1_loss(
                    proj_body, target_pts, beta=0.01, reduction="none"
                ).sum(dim=-1)
                loss_reproj = (target_weights * smooth_l1_err).sum() / (target_weights.sum() + 1e-8)
            else:
                loss_reproj = torch.tensor(0.0, device=self.device)

            # 2. Dynamic Regularization Decay Schedule
            lambda_reg = self.compute_dynamic_reg_decay(t, self.config.iters_3b)
            loss_reg = lambda_reg * (body_param - init_body_pose).pow(2).sum()

            total_loss = self.config.w_reproj * loss_reproj + loss_reg

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_body_pose = body_param.detach().clone()

            total_loss.backward()

            # 3. Arm-Chain Kinematic Masking: zero gradients on occluded limbs
            if lock_left_arm and body_param.grad is not None:
                body_param.grad[:, left_mask > 0] = 0.0
            if lock_right_arm and body_param.grad is not None:
                body_param.grad[:, right_mask > 0] = 0.0

            optimizer.step()

            # Ensure zero numerical drift on locked arms
            with torch.no_grad():
                if lock_left_arm:
                    body_param[:, left_mask > 0] = init_body_pose[:, left_mask > 0]
                if lock_right_arm:
                    body_param[:, right_mask > 0] = init_body_pose[:, right_mask > 0]

        p_out = {k: v.detach().clone() for k, v in p.items()}
        p_out["body_pose"] = best_body_pose
        return self._restore_device(p_out, orig_device)

    # -----------------------------------------------------------------------
    # Stage 3C: Hand Refinement Optimization (50 Steps)
    # -----------------------------------------------------------------------

    def optimize_stage_3c(
        self,
        current_params: Dict[str, torch.Tensor],
        sapiens_targets: Union[Dict[str, Any], torch.Tensor],
        camera_K: torch.Tensor,
        hand_info: Optional[Dict[str, Any]] = None,
        img_size: Tuple[int, int] = (1920, 1080),
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Stage 3C: Hand pose refinement (50 steps Adam).

        Optimizes left_hand_pose and right_hand_pose (lr=0.025) using Smooth L1
        reprojection on 15+15 finger joints, optional WiLoR 3D MANO alignment with
        median bone scale clamp [0.7, 1.4], flat hand regularization, visibility gating,
        and post-step delta clamping [-hand_delta_max, hand_delta_max] rad.
        All other parameters strictly frozen.
        """
        orig_device = self._detect_orig_device(current_params)
        p = self._standardize_params(current_params)
        cam_K = self._standardize_camera_k(camera_K)

        kp_all, sc_all = self._extract_full_sapiens_tensors(sapiens_targets, **kwargs)

        state_l, opt_l = self.classify_hand(sc_all, kp_all, "left", img_size, hand_info=hand_info)
        state_r, opt_r = self.classify_hand(sc_all, kp_all, "right", img_size, hand_info=hand_info)

        init_lh = p["left_hand_pose"].clone().detach()
        init_rh = p["right_hand_pose"].clone().detach()

        if not opt_l and not opt_r:
            p_out = {k: v.detach().clone() for k, v in p.items()}
            return self._restore_device(p_out, orig_device)

        lh_param = nn.Parameter(init_lh.clone(), requires_grad=opt_l)
        rh_param = nn.Parameter(init_rh.clone(), requires_grad=opt_r)

        active_params = [param for param in [lh_param, rh_param] if param.requires_grad]
        if not active_params:
            p_out = {k: v.detach().clone() for k, v in p.items()}
            return self._restore_device(p_out, orig_device)

        optimizer = torch.optim.Adam(active_params, lr=self.config.lr_hand)

        w_lh = compute_adaptive_weights(
            sc_all[STAGE_3C_LH_GOLIATH], region="hand", conf_threshold=self.config.conf_threshold
        ).to(self.device) if opt_l else None
        w_rh = compute_adaptive_weights(
            sc_all[STAGE_3C_RH_GOLIATH], region="hand", conf_threshold=self.config.conf_threshold
        ).to(self.device) if opt_r else None

        has_reproj_l = bool(opt_l and state_l == "visible" and w_lh is not None and w_lh.sum().item() > 1e-6)
        has_reproj_r = bool(opt_r and state_r == "visible" and w_rh is not None and w_rh.sum().item() > 1e-6)

        target_pts_lh = kp_all[STAGE_3C_LH_GOLIATH] if has_reproj_l else None
        target_pts_rh = kp_all[STAGE_3C_RH_GOLIATH] if has_reproj_r else None

        best_loss = float("inf")
        best_lh = init_lh.clone()
        best_rh = init_rh.clone()

        for step in range(self.config.iters_3c):
            optimizer.zero_grad()

            out = self.smplx_model(
                global_orient=p["global_orient"],
                body_pose=p["body_pose"],
                left_hand_pose=lh_param,
                right_hand_pose=rh_param,
                jaw_pose=p["jaw_pose"],
                betas=p["betas"],
                expression=p["expression"],
                transl=p["transl"],
            )

            proj_hands = self.project_pinhole(out.joints[0], cam_K)
            loss = torch.tensor(0.0, device=self.device)

            # 1. Sapiens 2D Finger Reprojection Loss (Smooth L1)
            if has_reproj_l:
                err_l = F.smooth_l1_loss(
                    proj_hands[STAGE_3C_LH_SMPLX], target_pts_lh, beta=0.01, reduction="none"
                ).sum(dim=-1)
                loss_reproj_l = (w_lh * err_l).sum() / (w_lh.sum() + 1e-8)
                loss = loss + self.config.w_reproj * loss_reproj_l

            if has_reproj_r:
                err_r = F.smooth_l1_loss(
                    proj_hands[STAGE_3C_RH_SMPLX], target_pts_rh, beta=0.01, reduction="none"
                ).sum(dim=-1)
                loss_reproj_r = (w_rh * err_r).sum() / (w_rh.sum() + 1e-8)
                loss = loss + self.config.w_reproj * loss_reproj_r

            # 2. WiLoR 3D MANO Target Loss
            if hand_info is not None:
                if opt_l and "left" in hand_info and hand_info["left"] is not None:
                    tgt_l = self.prepare_wilor_target(out.joints, is_right=False, wilor_dict=hand_info["left"])
                    if tgt_l is not None:
                        loss_wilor_l = F.smooth_l1_loss(out.joints[0, 25:40], tgt_l, beta=0.01)
                        loss = loss + self.config.w_hand_wilor * loss_wilor_l
                if opt_r and "right" in hand_info and hand_info["right"] is not None:
                    tgt_r = self.prepare_wilor_target(out.joints, is_right=True, wilor_dict=hand_info["right"])
                    if tgt_r is not None:
                        loss_wilor_r = F.smooth_l1_loss(out.joints[0, 40:55], tgt_r, beta=0.01)
                        loss = loss + self.config.w_hand_wilor * loss_wilor_r

            # 3. Flat Hand Prior and Regularization
            loss_flat = torch.tensor(0.0, device=self.device)
            loss_reg = torch.tensor(0.0, device=self.device)
            if opt_l:
                loss_flat = loss_flat + lh_param.pow(2).sum()
                loss_reg = loss_reg + (lh_param - init_lh).pow(2).sum()
            if opt_r:
                loss_flat = loss_flat + rh_param.pow(2).sum()
                loss_reg = loss_reg + (rh_param - init_rh).pow(2).sum()

            loss = loss + self.config.w_hand_flat * loss_flat + 0.005 * loss_reg

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_lh = lh_param.detach().clone()
                best_rh = rh_param.detach().clone()

            loss.backward()
            optimizer.step()

            # 4. Post-Step Delta Clamping [-hand_delta_max, hand_delta_max] rad
            with torch.no_grad():
                delta_bound = self.config.hand_delta_max
                if opt_l:
                    delta_l = torch.clamp(lh_param - init_lh, min=-delta_bound, max=delta_bound)
                    lh_param.copy_(init_lh + delta_l)
                if opt_r:
                    delta_r = torch.clamp(rh_param - init_rh, min=-delta_bound, max=delta_bound)
                    rh_param.copy_(init_rh + delta_r)

        p_out = {k: v.detach().clone() for k, v in p.items()}
        p_out["left_hand_pose"] = best_lh
        p_out["right_hand_pose"] = best_rh
        return self._restore_device(p_out, orig_device)

    # -----------------------------------------------------------------------
    # Stage 3D: Facial Expression & Jaw Optimization (30 Steps)
    # -----------------------------------------------------------------------

    def optimize_stage_3d(
        self,
        current_params: Dict[str, torch.Tensor],
        sapiens_targets: Union[Dict[str, Any], torch.Tensor],
        camera_K: torch.Tensor,
        face_info: Optional[Dict[str, Any]] = None,
        img_size: Tuple[int, int] = (1920, 1080),
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Stage 3D: Facial expression and jaw optimization (30 steps Adam).

        Optimizes jaw_pose and expression (lr=0.02) using Sapiens 68 face contour
        reprojection (FACE68_GOLIATH_TO_SMPLX), optional EMICA 3D target, L2 regularization,
        visibility gating, post-step jaw delta clamping [-0.3, 0.3] rad (absolute jaw_min_max),
        and expression clamping [-2.0, 2.0] std deviations (bounded by expr_min_max).
        All other parameters strictly frozen.
        """
        orig_device = self._detect_orig_device(current_params)
        p = self._standardize_params(current_params)
        cam_K = self._standardize_camera_k(camera_K)

        kp_all, sc_all = self._extract_full_sapiens_tensors(sapiens_targets, **kwargs)

        face_scores = sc_all[STAGE_3D_FACE_GOLIATH]
        c_face_mean = float(face_scores.mean().item()) if len(face_scores) > 0 else 0.0
        n_valid = int((face_scores >= 0.4).sum().item()) if len(face_scores) > 0 else 0

        # Visibility gating: skip if mean conf < 0.2 or < 10 valid points (unless EMICA target present)
        has_emica = bool(face_info is not None and ("expression" in face_info or "jaw_pose" in face_info))
        if (c_face_mean < 0.2 or n_valid < 10) and not has_emica:
            p_out = {k: v.detach().clone() for k, v in p.items()}
            return self._restore_device(p_out, orig_device)

        init_jaw = p["jaw_pose"].clone().detach()
        init_expr = p["expression"].clone().detach()

        jaw_param = nn.Parameter(init_jaw.clone())
        expr_param = nn.Parameter(init_expr.clone())

        optimizer = torch.optim.Adam([jaw_param, expr_param], lr=self.config.lr_face)

        w_face = compute_adaptive_weights(
            face_scores, region="face", conf_threshold=self.config.conf_threshold
        ).to(self.device)
        has_valid_reproj = bool(w_face.sum().item() > 1e-6 and c_face_mean >= 0.2 and n_valid >= 10)
        target_pts_face = kp_all[STAGE_3D_FACE_GOLIATH] if has_valid_reproj else None

        best_loss = float("inf")
        best_jaw = init_jaw.clone()
        best_expr = init_expr.clone()

        for step in range(self.config.iters_3d):
            optimizer.zero_grad()

            out = self.smplx_model(
                global_orient=p["global_orient"],
                body_pose=p["body_pose"],
                left_hand_pose=p["left_hand_pose"],
                right_hand_pose=p["right_hand_pose"],
                jaw_pose=jaw_param,
                betas=p["betas"],
                expression=expr_param,
                transl=p["transl"],
            )

            proj_all = self.project_pinhole(out.joints[0], cam_K)
            loss = torch.tensor(0.0, device=self.device)

            # 1. Sapiens 68 Face Contour Reprojection (Smooth L1)
            if has_valid_reproj:
                err_face = F.smooth_l1_loss(
                    proj_all[STAGE_3D_FACE_SMPLX], target_pts_face, beta=0.01, reduction="none"
                ).sum(dim=-1)
                loss_reproj = (w_face * err_face).sum() / (w_face.sum() + 1e-8)
                loss = loss + self.config.w_reproj * loss_reproj

            # 2. EMICA 3D Target Loss
            if face_info is not None:
                if "expression" in face_info and face_info["expression"] is not None:
                    tgt_exp = torch.as_tensor(face_info["expression"], dtype=torch.float32, device=self.device)
                    if tgt_exp.dim() == 1:
                        tgt_exp = tgt_exp.unsqueeze(0)
                    tgt_exp_10 = tgt_exp[:, :10]
                    loss = loss + self.config.w_face_emica * F.mse_loss(expr_param, tgt_exp_10)
                if "jaw_pose" in face_info and face_info["jaw_pose"] is not None:
                    tgt_jaw = torch.as_tensor(face_info["jaw_pose"], dtype=torch.float32, device=self.device)
                    if tgt_jaw.dim() == 1:
                        tgt_jaw = tgt_jaw.unsqueeze(0)
                    loss = loss + 0.5 * self.config.w_face_emica * F.mse_loss(jaw_param, tgt_jaw)

            # 3. L2 Regularization
            loss_reg = expr_param.pow(2).sum() + 0.1 * jaw_param.pow(2).sum()
            loss = loss + self.config.w_face_reg * loss_reg

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_jaw = jaw_param.detach().clone()
                best_expr = expr_param.detach().clone()

            loss.backward()
            optimizer.step()

            # 4. Clamping bounds
            with torch.no_grad():
                # Jaw delta clamping [-0.3, 0.3] rad relative to init, plus absolute jaw_min_max
                delta_jaw = torch.clamp(jaw_param - init_jaw, min=-0.3, max=0.3)
                clamped_jaw = torch.clamp(
                    init_jaw + delta_jaw,
                    min=self.config.jaw_min_max[0],
                    max=self.config.jaw_min_max[1],
                )
                jaw_param.copy_(clamped_jaw)

                # Expression clamping [-2.0, 2.0] standard deviations, bounded by expr_min_max
                expr_min = max(-2.0, self.config.expr_min_max[0])
                expr_max = min(2.0, self.config.expr_min_max[1])
                expr_param.clamp_(min=expr_min, max=expr_max)

        p_out = {k: v.detach().clone() for k, v in p.items()}
        expr_min = max(-2.0, self.config.expr_min_max[0])
        expr_max = min(2.0, self.config.expr_min_max[1])
        p_out["jaw_pose"] = torch.clamp(
            best_jaw, min=self.config.jaw_min_max[0], max=self.config.jaw_min_max[1]
        )
        p_out["expression"] = torch.clamp(best_expr, min=expr_min, max=expr_max)
        return self._restore_device(p_out, orig_device)

    # -----------------------------------------------------------------------
    # End-to-End Person & Multi-Person Optimization
    # -----------------------------------------------------------------------

    def optimize_person(
        self,
        init_params: Dict[str, torch.Tensor],
        sapiens_targets: Union[Dict[str, Any], torch.Tensor],
        camera_K: torch.Tensor,
        hand_info: Optional[Dict[str, Any]] = None,
        face_info: Optional[Dict[str, Any]] = None,
        img_size: Tuple[int, int] = (1920, 1080),
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Runs sequential 4-stage coarse-to-fine optimization for a single person instance.

        Executes sequentially:
            Stage 3A (transl & global_orient)
            -> Stage 3B (body_pose)
            -> Stage 3C (left_hand_pose & right_hand_pose)
            -> Stage 3D (jaw_pose & expression)

        Args:
            init_params: Initial MultiHMR SMPL-X parameter dictionary.
            sapiens_targets: Sapiens 2D detection keypoints and confidence scores.
            camera_K: (3, 3) camera intrinsics matrix.
            hand_info: Optional hand visibility and WiLoR targets dictionary.
            face_info: Optional face expression and EMICA targets dictionary.
            img_size: Image (width, height) resolution tuple.

        Returns:
            Refined SMPL-X parameter dictionary matching input device and parameter keys.
        """
        p_3a = self.optimize_stage_3a(
            current_params=init_params,
            sapiens_targets=sapiens_targets,
            camera_K=camera_K,
            **kwargs,
        )

        p_3b = self.optimize_stage_3b(
            current_params=p_3a,
            sapiens_targets=sapiens_targets,
            camera_K=camera_K,
            hand_info=hand_info,
            **kwargs,
        )

        p_3c = self.optimize_stage_3c(
            current_params=p_3b,
            sapiens_targets=sapiens_targets,
            camera_K=camera_K,
            hand_info=hand_info,
            img_size=img_size,
            **kwargs,
        )

        p_3d = self.optimize_stage_3d(
            current_params=p_3c,
            sapiens_targets=sapiens_targets,
            camera_K=camera_K,
            face_info=face_info,
            img_size=img_size,
            **kwargs,
        )

        return p_3d

    def optimize_multi_person(
        self,
        persons_params: List[Dict[str, torch.Tensor]],
        persons_targets: List[Union[Dict[str, Any], torch.Tensor]],
        camera_K: torch.Tensor,
        hand_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
        face_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
        img_size: Tuple[int, int] = (1920, 1080),
        **kwargs: Any,
    ) -> List[Dict[str, torch.Tensor]]:
        """Multi-person optimization over matched instances without parameter mixing.

        Executes independent optimize_person calls for each person instance,
        guaranteeing zero cross-talk between individual parameters.

        Args:
            persons_params: List of SMPL-X parameter dictionaries.
            persons_targets: List of Sapiens 2D target dictionaries.
            camera_K: Camera intrinsics matrix (3, 3).
            hand_infos: Optional list of hand info dictionaries.
            face_infos: Optional list of face info dictionaries.
            img_size: Image (width, height) tuple.

        Returns:
            List of refined SMPL-X parameter dictionaries.
        """
        results: List[Dict[str, torch.Tensor]] = []
        num_persons = len(persons_params)

        for i in range(num_persons):
            p_init = persons_params[i]
            p_targ = persons_targets[i] if i < len(persons_targets) else {}
            h_info = hand_infos[i] if hand_infos is not None and i < len(hand_infos) else None
            f_info = face_infos[i] if face_infos is not None and i < len(face_infos) else None

            refined = self.optimize_person(
                init_params=p_init,
                sapiens_targets=p_targ,
                camera_K=camera_K,
                hand_info=h_info,
                face_info=f_info,
                img_size=img_size,
                **kwargs,
            )
            results.append(refined)

        return results

    optimize_batch = optimize_multi_person

    # -----------------------------------------------------------------------
    # Helper & Standardization Utilities
    # -----------------------------------------------------------------------

    def compute_dynamic_reg_decay(self, step: int, total_steps: int) -> float:
        """Evaluates dynamic regularization decay schedule lambda_reg(t) = w_body_reg * (1 - t/T)^0.5."""
        if total_steps <= 0:
            return 0.0
        factor = (max(0.0, 1.0 - float(step) / float(total_steps))) ** 0.5
        return float(self.config.w_body_reg * factor)

    def classify_hand(
        self,
        scores: torch.Tensor,
        kp: torch.Tensor,
        side: str,
        img_size: Tuple[int, int] = (1920, 1080),
        hand_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """Evaluates hand visibility gating based on confidence, boundary checks, and hand_info.

        Returns:
            Tuple (state_string, should_optimize_hand)
        """
        if hand_info is not None:
            state = hand_info.get(f"{side}_state", hand_info.get(side, None))
            if state is not None:
                if isinstance(state, dict):
                    dict_state = state.get("state", state.get("status", None))
                    if dict_state is not None:
                        state_str = str(dict_state).lower()
                    elif "pred_keypoints_3d" in state:
                        state_str = "visible"
                    else:
                        state_str = ""
                else:
                    state_str = str(state).lower()
                if state_str in ("behind", "cropped", "occluded"):
                    return state_str, False
                if state_str == "wrist_only":
                    return "wrist_only", True
                if state_str == "visible":
                    return "visible", True

        W, H = img_size
        fingers = STAGE_3C_LH_GOLIATH if side == "left" else STAGE_3C_RH_GOLIATH
        wrist_idx = 62 if side == "left" else 41

        if len(scores) <= max(wrist_idx, max(fingers)):
            return "behind", False

        c_mean = float(scores[fingers].mean().item())
        n_valid = int((scores[fingers] >= self.config.conf_threshold).sum().item())
        c_wrist = float(scores[wrist_idx].item())
        p_wrist = kp[wrist_idx]

        in_frame = (
            (0.02 * W < float(p_wrist[0].item()) < 0.98 * W)
            and (0.02 * H < float(p_wrist[1].item()) < 0.98 * H)
        )

        if c_mean < 0.2:
            return "behind", False
        if not in_frame and c_mean < self.config.conf_threshold:
            return "cropped", False
        if c_wrist >= 0.4 and (n_valid < 3 or c_mean < self.config.conf_threshold):
            return "wrist_only", True
        return "visible", True

    def prepare_wilor_target(
        self,
        smplx_joints: torch.Tensor,
        is_right: bool,
        wilor_dict: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """Aligns WiLoR 3D MANO joints to SMPL-X finger joints with median bone length scale clamp [0.7, 1.4]."""
        if not wilor_dict or "pred_keypoints_3d" not in wilor_dict:
            return None

        wrist_idx = 21 if is_right else 20
        finger_slice = slice(40, 55) if is_right else slice(25, 40)

        smplx_wrist = smplx_joints[0, wrist_idx].detach()
        smplx_fingers = smplx_joints[0, finger_slice].detach()

        mano_raw = wilor_dict["pred_keypoints_3d"]
        mano_3d = torch.as_tensor(mano_raw, dtype=torch.float32, device=self.device)
        cam_t = torch.as_tensor(
            wilor_dict.get("cam_t", np.zeros(3)), dtype=torch.float32, device=self.device
        )
        mano_cam = mano_3d + cam_t

        root = mano_cam[0]
        if mano_cam.shape[0] == 21:
            rel_mano = mano_cam[WILOR_TO_SMPLX] - root
        elif mano_cam.shape[0] == 15:
            rel_mano = mano_cam - root
        else:
            rel_mano = mano_cam[:15] - root

        rel_smplx = smplx_fingers - smplx_wrist
        ratio = rel_smplx.norm(dim=-1) / rel_mano.norm(dim=-1).clamp_min(1e-6)
        scale = torch.median(ratio).clamp(0.7, 1.4)
        return smplx_wrist + rel_mano * scale

    def _determine_arm_occlusion(
        self,
        sapiens_targets: Union[Dict[str, Any], torch.Tensor],
        hand_info: Optional[Dict[str, Any]],
    ) -> Tuple[bool, bool]:
        """Determines whether left and right arm kinematic chains should be locked.

        Evaluates left and right arms independently:
        - If hand_info provides explicit state or WiLoR targets for an arm, uses it.
        - Otherwise, falls back to inspecting Sapiens keypoint confidence scores
          for that arm's wrist (62/41) and finger joints (42..61 / 21..40).
        - Returns (lock_left, lock_right) only after both arms have been evaluated.
        """
        l_state = None
        r_state = None

        if hand_info is not None:
            l_state = hand_info.get("left_state", hand_info.get("left", None))
            r_state = hand_info.get("right_state", hand_info.get("right", None))

        # Lazy extraction of Sapiens scores if at least one arm needs fallback
        sc: Optional[np.ndarray] = None

        def _get_left_sapiens_lock() -> bool:
            nonlocal sc
            if sc is None:
                _, sc_all = self._extract_full_sapiens_tensors(sapiens_targets)
                sc = sc_all.detach().cpu().numpy()
            if len(sc) > 62:
                l_conf = max(float(sc[62]), float(np.max(sc[42:62])) if len(sc) >= 62 else 0.0)
                return l_conf < self.config.conf_threshold
            return False

        def _get_right_sapiens_lock() -> bool:
            nonlocal sc
            if sc is None:
                _, sc_all = self._extract_full_sapiens_tensors(sapiens_targets)
                sc = sc_all.detach().cpu().numpy()
            if len(sc) > 41:
                r_conf = max(float(sc[41]), float(np.max(sc[21:41])) if len(sc) >= 41 else 0.0)
                return r_conf < self.config.conf_threshold
            return False

        # 1. Independent Left Arm Resolution
        if l_state is not None:
            if isinstance(l_state, dict):
                dict_state = l_state.get("state", l_state.get("status", None))
                if dict_state is not None:
                    lock_left = str(dict_state).lower() in ("behind", "cropped", "occluded")
                elif "pred_keypoints_3d" in l_state:
                    lock_left = False
                else:
                    lock_left = _get_left_sapiens_lock()
            else:
                lock_left = str(l_state).lower() in ("behind", "cropped", "occluded")
        else:
            lock_left = _get_left_sapiens_lock()

        # 2. Independent Right Arm Resolution
        if r_state is not None:
            if isinstance(r_state, dict):
                dict_state = r_state.get("state", r_state.get("status", None))
                if dict_state is not None:
                    lock_right = str(dict_state).lower() in ("behind", "cropped", "occluded")
                elif "pred_keypoints_3d" in r_state:
                    lock_right = False
                else:
                    lock_right = _get_right_sapiens_lock()
            else:
                lock_right = str(r_state).lower() in ("behind", "cropped", "occluded")
        else:
            lock_right = _get_right_sapiens_lock()

        return lock_left, lock_right

    def _standardize_params(self, params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Standardizes parameter tensor shapes to (1, D) on self.device."""
        param_dims = {
            "global_orient": 3,
            "body_pose": 63,
            "left_hand_pose": 45,
            "right_hand_pose": 45,
            "jaw_pose": 3,
            "betas": 10,
            "expression": 10,
            "transl": 3,
        }
        std: Dict[str, torch.Tensor] = {}
        for key, dim in param_dims.items():
            if key in params and params[key] is not None:
                val = params[key]
                if isinstance(val, np.ndarray):
                    t = torch.from_numpy(val).float()
                elif isinstance(val, torch.Tensor):
                    t = val.detach().float()
                else:
                    t = torch.tensor(val, dtype=torch.float32)
                t = t.to(self.device)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                std[key] = t
            else:
                std[key] = torch.zeros((1, dim), dtype=torch.float32, device=self.device)
        return std

    def _standardize_camera_k(self, camera_K: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Ensures camera intrinsics matrix is a (3, 3) float tensor on self.device."""
        if isinstance(camera_K, np.ndarray):
            t = torch.from_numpy(camera_K).float()
        elif isinstance(camera_K, torch.Tensor):
            t = camera_K.float()
        else:
            t = torch.tensor(camera_K, dtype=torch.float32)
        return t.to(self.device).reshape(3, 3)

    def _detect_orig_device(self, params: Dict[str, Any]) -> torch.device:
        """Detects incoming parameter device for output restoration."""
        for val in params.values():
            if isinstance(val, torch.Tensor):
                return val.device
        return self.device

    def _restore_device(
        self, params: Dict[str, torch.Tensor], target_device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Transfers parameter tensors to target device."""
        if target_device == self.device:
            return params
        return {k: v.to(target_device) for k, v in params.items()}

    def _extract_full_sapiens_tensors(
        self,
        sapiens_targets: Union[Dict[str, Any], torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts complete keypoints (N, 2) and scores (N,) tensors on self.device."""
        pts_raw = None
        sc_raw = None

        if isinstance(sapiens_targets, dict):
            for k in ("keypoints", "xy", "extra_keypoints"):
                if k in sapiens_targets and sapiens_targets[k] is not None:
                    pts_raw = sapiens_targets[k]
                    break
            for k in ("scores", "extra_scores"):
                if k in sapiens_targets and sapiens_targets[k] is not None:
                    sc_raw = sapiens_targets[k]
                    break
        elif isinstance(sapiens_targets, torch.Tensor):
            if sapiens_targets.shape[-1] == 2:
                pts_raw = sapiens_targets
            elif sapiens_targets.shape[-1] == 3:
                pts_raw = sapiens_targets[..., :2]
                sc_raw = sapiens_targets[..., 2]

        if "scores" in kwargs and kwargs["scores"] is not None:
            sc_raw = kwargs["scores"]
        if "keypoints" in kwargs and kwargs["keypoints"] is not None:
            pts_raw = kwargs["keypoints"]

        if pts_raw is None:
            # Missing keypoints: coordinates default to (0, 0) and confidence strictly defaults to 0.0
            pts_tensor = torch.zeros((308, 2), dtype=torch.float32, device=self.device)
            sc_tensor = torch.zeros((308,), dtype=torch.float32, device=self.device)
        else:
            if isinstance(pts_raw, np.ndarray):
                pts_tensor = torch.from_numpy(pts_raw).float().to(self.device)
            elif isinstance(pts_raw, torch.Tensor):
                pts_tensor = pts_raw.float().to(self.device)
            else:
                pts_tensor = torch.tensor(pts_raw, dtype=torch.float32, device=self.device)

            if pts_tensor.dim() == 3 and pts_tensor.shape[0] == 1:
                pts_tensor = pts_tensor.squeeze(0)
            elif pts_tensor.dim() == 1:
                if pts_tensor.shape[0] == 2:
                    pts_tensor = pts_tensor.unsqueeze(0)
                elif pts_tensor.shape[0] == 0:
                    pts_tensor = pts_tensor.reshape(0, 2)

            num_pts = pts_tensor.shape[0]
            if sc_raw is None:
                # When keypoints are explicitly provided but scores are absent, default to 1.0 confidence
                sc_tensor = torch.ones((num_pts,), dtype=torch.float32, device=self.device)
            elif isinstance(sc_raw, np.ndarray):
                sc_tensor = torch.from_numpy(sc_raw).float().to(self.device)
            elif isinstance(sc_raw, torch.Tensor):
                sc_tensor = sc_raw.float().to(self.device)
            else:
                sc_tensor = torch.tensor(sc_raw, dtype=torch.float32, device=self.device)

            if sc_tensor.dim() == 2 and sc_tensor.shape[0] == 1:
                sc_tensor = sc_tensor.squeeze(0)
            elif sc_tensor.dim() == 0:
                sc_tensor = sc_tensor.unsqueeze(0)

            # Align lengths if mismatched
            if sc_tensor.shape[0] < pts_tensor.shape[0]:
                sc_pad = torch.zeros(
                    (pts_tensor.shape[0] - sc_tensor.shape[0],),
                    dtype=torch.float32,
                    device=self.device,
                )
                sc_tensor = torch.cat([sc_tensor, sc_pad], dim=0)
            elif pts_tensor.shape[0] < sc_tensor.shape[0]:
                pts_pad = torch.zeros(
                    (sc_tensor.shape[0] - pts_tensor.shape[0], 2),
                    dtype=torch.float32,
                    device=self.device,
                )
                pts_tensor = torch.cat([pts_tensor, pts_pad], dim=0)

            # Zero-pad sub-308 keypoint tensors up to 308 keypoints to eliminate IndexError
            if pts_tensor.shape[0] < 308:
                pts_pad = torch.zeros(
                    (308 - pts_tensor.shape[0], 2),
                    dtype=torch.float32,
                    device=self.device,
                )
                sc_pad = torch.zeros(
                    (308 - sc_tensor.shape[0],),
                    dtype=torch.float32,
                    device=self.device,
                )
                pts_tensor = torch.cat([pts_tensor, pts_pad], dim=0)
                sc_tensor = torch.cat([sc_tensor, sc_pad], dim=0)

        # Identify invalid points: NaNs or Infs in coordinates OR scores
        invalid_mask = (
            torch.isnan(pts_tensor).any(dim=-1)
            | torch.isinf(pts_tensor).any(dim=-1)
            | torch.isnan(sc_tensor)
            | torch.isinf(sc_tensor)
        )
        # Zero out scores for invalid points
        sc_tensor = torch.where(invalid_mask, torch.zeros_like(sc_tensor), sc_tensor)
        # CRITICAL FIX: Zero out coordinates where invalid to prevent IEEE 754 0.0 * NaN = NaN
        pts_tensor = torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(pts_tensor), pts_tensor)

        # Unconditional final sanitization against residual NaNs/Infs and score bounds
        pts_tensor = torch.nan_to_num(pts_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        sc_tensor = torch.nan_to_num(sc_tensor, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        return pts_tensor, sc_tensor

    def _extract_stage_targets(
        self,
        sapiens_targets: Union[Dict[str, Any], torch.Tensor],
        goliath_indices: List[int],
        region: str = "body",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts 2D target points and computes adaptive confidence weights for given Goliath indices."""
        pts_tensor, sc_tensor = self._extract_full_sapiens_tensors(sapiens_targets, **kwargs)

        num_j = len(goliath_indices)
        if len(pts_tensor) == 12:
            goliath_to_12 = {gid: i for i, gid in enumerate(GOLIATH_EVAL_IDS)}
            sel_pts = []
            sel_scores = []
            for gid in goliath_indices:
                if gid in goliath_to_12:
                    idx12 = goliath_to_12[gid]
                    sel_pts.append(pts_tensor[idx12])
                    sel_scores.append(sc_tensor[idx12])
                else:
                    sel_pts.append(torch.zeros(2, dtype=torch.float32, device=self.device))
                    sel_scores.append(torch.zeros((), dtype=torch.float32, device=self.device))
            selected_pts = torch.stack(sel_pts, dim=0)
            selected_scores = torch.stack(sel_scores, dim=0)
        elif len(pts_tensor) < max(goliath_indices) + 1:
            return (
                torch.zeros((num_j, 2), dtype=torch.float32, device=self.device),
                torch.zeros((num_j,), dtype=torch.float32, device=self.device),
            )
        else:
            selected_pts = pts_tensor[goliath_indices]
            selected_scores = sc_tensor[goliath_indices]

        weights = compute_adaptive_weights(
            selected_scores,
            region=region,
            occlusion_state="visible",
            conf_threshold=self.config.conf_threshold,
        )
        return selected_pts, weights
