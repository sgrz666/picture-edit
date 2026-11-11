"""Multi-Person Interaction Constraints Module for Sapiens 2 x MultiHMR.

Provides pure PyTorch, differentiable Signed Distance Field (SDF) penetration penalty
and dynamic Hi4D interaction contact loss, operating seamlessly on CPU and CUDA.
"""

import math
import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import JointOptimizationConfig


def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Computes differentiable outward unit vertex normals from triangle faces.

    Args:
        verts: (V, 3) or (B, V, 3) vertex coordinates.
        faces: (F, 3) triangular face vertex indices (long).

    Returns:
        (V, 3) or (B, V, 3) unit normal vectors normalized with eps=1e-8.
    """
    if faces is None or faces.numel() == 0 or verts.numel() == 0:
        return torch.zeros_like(verts)

    is_batched = verts.dim() == 3
    if not is_batched:
        verts = verts.unsqueeze(0)

    B, V, _ = verts.shape
    device = verts.device

    # Ensure faces is on the same device and long dtype
    if faces.device != device:
        faces = faces.to(device)
    if faces.dtype != torch.long:
        faces = faces.long()

    # Sanitize verts
    verts_clean = torch.nan_to_num(verts, nan=0.0, posinf=1e6, neginf=-1e6)

    v0 = verts_clean[:, faces[:, 0]]
    v1 = verts_clean[:, faces[:, 1]]
    v2 = verts_clean[:, faces[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = torch.nan_to_num(face_normals, nan=0.0)

    vertex_normals = torch.zeros((B, V, 3), dtype=verts.dtype, device=device)

    for k in range(3):
        idx = faces[:, k].unsqueeze(0).unsqueeze(-1).expand(B, -1, 3)
        vertex_normals.scatter_add_(1, idx, face_normals)

    vertex_normals = F.normalize(vertex_normals, dim=-1, eps=1e-8)
    vertex_normals = torch.nan_to_num(vertex_normals, nan=0.0)
    return vertex_normals.squeeze(0) if not is_batched else vertex_normals


def compute_mesh_penetration_loss(
    verts_a: torch.Tensor,
    verts_b: torch.Tensor,
    faces: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    stride: int = 4,
    max_pen_dist: Optional[float] = None,
) -> torch.Tensor:
    """Convenience functional wrapper for penetration loss between two meshes."""
    module = InteractionConstraintModule(stride=stride, max_pen_dist=max_pen_dist)
    return module.penetration_loss(verts_a, verts_b, faces=faces, threshold=threshold)


def compute_mesh_contact_loss(
    verts_a: torch.Tensor,
    verts_b: torch.Tensor,
    r_contact: Optional[float] = None,
    threshold: Optional[float] = None,
    stride: int = 4,
) -> torch.Tensor:
    """Convenience functional wrapper for contact loss between two meshes."""
    module = InteractionConstraintModule(
        stride=stride,
        r_contact=r_contact if r_contact is not None else 0.05,
    )
    return module.contact_loss(
        verts_a, verts_b, r_contact=r_contact, threshold=threshold
    )


class InteractionConstraintModule(nn.Module):
    """Computes inter-person SDF penetration penalty and contact priors."""

    def __init__(
        self,
        config: Optional[JointOptimizationConfig] = None,
        faces: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
        stride: int = 4,
        max_pen_dist: Optional[float] = None,
        r_contact: float = 0.05,
    ) -> None:
        super().__init__()
        self.config = config or JointOptimizationConfig()
        self.stride = stride
        self.max_pen_dist = max_pen_dist
        self.r_contact = r_contact
        self.tau = getattr(self.config, "sdf_tau", 0.005)
        self.penetration_weights = getattr(
            self.config,
            "penetration_weights",
            {"separated": 0.1, "light": 0.5, "hug": 1.0, "severe": 2.0},
        )
        self.contact_weights = getattr(
            self.config,
            "contact_weights",
            {"separated": 0.0, "light": 0.1, "hug": 0.5, "severe": 1.0},
        )

        target_device = device or getattr(self.config, "device", "cpu")
        if isinstance(target_device, str) and "cuda" in target_device and not torch.cuda.is_available():
            target_device = "cpu"
        self.device = torch.device(target_device)

        if faces is not None:
            self.register_buffer("faces", faces.to(self.device).long())
        else:
            loaded_faces = self._load_smplx_faces_if_available()
            if loaded_faces is not None:
                self.register_buffer("faces", loaded_faces.to(self.device).long())
            else:
                self.register_buffer("faces", None)

    def _load_smplx_faces_if_available(self) -> Optional[torch.Tensor]:
        """Attempts to load canonical SMPL-X faces from config path."""
        faces_path = getattr(self.config, "smplx_faces_path", None)
        if faces_path and os.path.isfile(faces_path):
            try:
                npz_data = np.load(faces_path)
                if "f" in npz_data:
                    return torch.from_numpy(npz_data["f"].astype(np.int64))
            except Exception:
                pass
        return None

    def set_faces(self, faces: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        """Sets or updates the mesh face topology tensor."""
        if faces is None:
            self.register_buffer("faces", None)
            return
        if isinstance(faces, np.ndarray):
            faces_t = torch.from_numpy(faces.astype(np.int64))
        else:
            faces_t = faces.long()
        self.register_buffer("faces", faces_t.to(self.device))

    def _get_vertex_normals(
        self, verts: torch.Tensor, faces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes or estimates outward vertex normals for verts."""
        f = faces if faces is not None else self.faces
        if f is not None and f.numel() > 0 and f.max() < verts.shape[0]:
            return compute_vertex_normals(verts, f)
        # Centroid outward normal approximation for synthetic/meshless point sets
        centroid = verts.mean(dim=0, keepdim=True)
        disp = verts - centroid
        normals = F.normalize(disp, dim=-1, eps=1e-8)
        return torch.nan_to_num(normals, nan=0.0)

    def compute_sdf(
        self,
        verts_query: torch.Tensor,
        verts_target: torch.Tensor,
        normals_target: Optional[torch.Tensor] = None,
        faces_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates signed distance of query vertices to the target mesh surface.

        Args:
            verts_query: (N_q, 3) query vertices.
            verts_target: (N_t, 3) target vertices.
            normals_target: (N_t, 3) outward target surface normals (optional).
            faces_target: (F, 3) target triangular faces (optional).

        Returns:
            sdf: (N_q,) signed distances (negative indicates interior penetration).
            min_dist: (N_q,) Euclidean distances to nearest target vertices.
        """
        if verts_query.shape[0] == 0 or verts_target.shape[0] == 0:
            zero_q = (verts_query * 0.0).sum(dim=-1)
            return zero_q, zero_q

        if normals_target is None:
            normals_target = self._get_vertex_normals(verts_target, faces_target)

        d_mat = torch.cdist(verts_query, verts_target)
        min_dist, nn_idx = d_mat.min(dim=-1)

        target_nn = verts_target[nn_idx]
        target_norm = normals_target[nn_idx]

        disp = verts_query - target_nn
        sdf = (disp * target_norm).sum(dim=-1)
        return sdf, min_dist

    def penetration_loss(
        self,
        verts_a: torch.Tensor,
        verts_b: torch.Tensor,
        faces: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Computes bidirectional SDF penetration penalty between Mesh A and Mesh B."""
        tau = threshold if threshold is not None else self.tau

        if verts_a.dim() == 3 and verts_a.shape[0] == 1:
            verts_a = verts_a.squeeze(0)
        if verts_b.dim() == 3 and verts_b.shape[0] == 1:
            verts_b = verts_b.squeeze(0)

        # Sanitize NaNs/Infs before zero_loss to prevent NaN propagation on separated meshes
        va_clean = torch.nan_to_num(verts_a, nan=0.0, posinf=1e6, neginf=-1e6)
        vb_clean = torch.nan_to_num(verts_b, nan=0.0, posinf=1e6, neginf=-1e6)

        zero_loss = (va_clean * 0.0).sum() + (vb_clean * 0.0).sum()

        if va_clean.shape[0] == 0 or vb_clean.shape[0] == 0:
            return zero_loss

        # 1. Broad-phase AABB test with clearance margin
        min_a, max_a = va_clean.min(dim=0).values, va_clean.max(dim=0).values
        min_b, max_b = vb_clean.min(dim=0).values, vb_clean.max(dim=0).values
        if not ((min_a <= max_b + tau) & (max_a >= min_b - tau)).all():
            return zero_loss

        # 2. Narrow-phase evaluation with adaptive stride
        stride = self.stride if va_clean.shape[0] >= 500 else 1
        va_sub = va_clean[::stride]
        vb_sub = vb_clean[::stride]

        na = self._get_vertex_normals(va_clean, faces)[::stride]
        nb = self._get_vertex_normals(vb_clean, faces)[::stride]

        # Mesh A penetrating inside Mesh B
        sdf_a, d_a = self.compute_sdf(va_sub, vb_sub, nb)
        pen_a = F.relu(-sdf_a - tau)
        if self.max_pen_dist is not None:
            pen_a = pen_a * (d_a < self.max_pen_dist).float()
        loss_a = (pen_a ** 2).sum()

        # Mesh B penetrating inside Mesh A
        sdf_b, d_b = self.compute_sdf(vb_sub, va_sub, na)
        pen_b = F.relu(-sdf_b - tau)
        if self.max_pen_dist is not None:
            pen_b = pen_b * (d_b < self.max_pen_dist).float()
        loss_b = (pen_b ** 2).sum()

        return loss_a + loss_b

    compute_penetration_loss = penetration_loss

    def contact_loss(
        self,
        verts_a: torch.Tensor,
        verts_b: torch.Tensor,
        r_contact: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Computes proximate surface contact attraction loss."""
        tau = threshold if threshold is not None else self.tau
        r_c = r_contact if r_contact is not None else self.r_contact

        if verts_a.dim() == 3 and verts_a.shape[0] == 1:
            verts_a = verts_a.squeeze(0)
        if verts_b.dim() == 3 and verts_b.shape[0] == 1:
            verts_b = verts_b.squeeze(0)

        # Sanitize NaNs/Infs before zero_loss to prevent NaN propagation on separated meshes
        va_clean = torch.nan_to_num(verts_a, nan=0.0, posinf=1e6, neginf=-1e6)
        vb_clean = torch.nan_to_num(verts_b, nan=0.0, posinf=1e6, neginf=-1e6)

        zero_loss = (va_clean * 0.0).sum() + (vb_clean * 0.0).sum()

        if va_clean.shape[0] == 0 or vb_clean.shape[0] == 0:
            return zero_loss

        # Broad-phase proximity test
        min_a, max_a = va_clean.min(dim=0).values, va_clean.max(dim=0).values
        min_b, max_b = vb_clean.min(dim=0).values, vb_clean.max(dim=0).values
        if not ((min_a <= max_b + r_c) & (max_a >= min_b - r_c)).all():
            return zero_loss

        stride = self.stride if va_clean.shape[0] >= 500 else 1
        va_sub = va_clean[::stride]
        vb_sub = vb_clean[::stride]

        d_mat = torch.cdist(va_sub, vb_sub)

        # A to B contact
        min_d_a = d_mat.min(dim=-1).values
        mask_a = min_d_a < r_c
        if mask_a.any():
            gap_a = F.relu(min_d_a - tau) * mask_a.float()
            loss_a = (gap_a ** 2).sum() / torch.clamp(mask_a.sum(), min=1.0)
        else:
            loss_a = zero_loss

        # B to A contact
        min_d_b = d_mat.min(dim=-2).values
        mask_b = min_d_b < r_c
        if mask_b.any():
            gap_b = F.relu(min_d_b - tau) * mask_b.float()
            loss_b = (gap_b ** 2).sum() / torch.clamp(mask_b.sum(), min=1.0)
        else:
            loss_b = zero_loss

        return loss_a + loss_b

    compute_contact_loss = contact_loss

    def compute_interaction_loss(
        self,
        verts_a: torch.Tensor,
        verts_b: torch.Tensor,
        category: str = "separated",
        faces: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Calculates dynamically weighted total interaction loss."""
        if category not in self.penetration_weights or category not in self.contact_weights:
            valid_cats = sorted(list(self.penetration_weights.keys()))
            raise ValueError(
                f"Invalid interaction category '{category}'. Must be one of {valid_cats}"
            )

        pen_w = float(self.penetration_weights[category])
        cont_w = float(self.contact_weights[category])

        l_pen = self.penetration_loss(verts_a, verts_b, faces=faces)
        l_cont = self.contact_loss(verts_a, verts_b)

        pen_w_t = torch.as_tensor(pen_w, dtype=l_pen.dtype, device=l_pen.device)
        cont_w_t = torch.as_tensor(cont_w, dtype=l_cont.dtype, device=l_cont.device)

        loss_pen_weighted = pen_w_t * l_pen
        loss_cont_weighted = cont_w_t * l_cont
        total = loss_pen_weighted + loss_cont_weighted

        return {
            "total_loss": total,
            "loss_total": total,
            "penetration_loss": l_pen,
            "loss_penetration": l_pen,
            "contact_loss": l_cont,
            "loss_contact": l_cont,
            "penetration_loss_weighted": loss_pen_weighted,
            "contact_loss_weighted": loss_cont_weighted,
            "weight_penetration": pen_w,
            "weight_contact": cont_w,
        }

    compute_losses = compute_interaction_loss

    def joint_step(
        self,
        person_a_params: Dict[str, torch.Tensor],
        person_b_params: Dict[str, torch.Tensor],
        category: str,
        smplx_model: Any,
        num_steps: int = 10,
        iters: Optional[int] = None,
        lr_transl: float = 0.005,
        lr_body: float = 0.005,
        lr_orient: float = 0.002,
        anchor_weight: float = 0.01,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Applies iterative multi-person interaction refinement on interacting pairs.

        Optimizes `transl`, `body_pose`, and `global_orient` while freezing
        `betas`, `hand_pose`, `jaw_pose`, and `expression`. Preserves device
        and dictionary structures.
        """
        if iters is not None:
            num_steps = iters

        if category not in self.penetration_weights or category not in self.contact_weights:
            valid_cats = sorted(list(self.penetration_weights.keys()))
            raise ValueError(
                f"Invalid interaction category '{category}'. Must be one of {valid_cats}"
            )

        orig_dev_a = {
            k: v.device for k, v in person_a_params.items() if isinstance(v, torch.Tensor)
        }
        orig_dev_b = {
            k: v.device for k, v in person_b_params.items() if isinstance(v, torch.Tensor)
        }

        # Determine target execution device
        model_device = None
        for p in smplx_model.parameters():
            model_device = p.device
            break
        target_device = model_device or person_a_params["transl"].device

        pen_w = float(self.penetration_weights[category])
        cont_w = float(self.contact_weights[category])

        # Prepare trainable parameters on target_device (explicitly detached from upstream autograd graphs)
        transl_a = nn.Parameter(
            person_a_params["transl"].to(target_device).float().detach().clone()
        )
        transl_b = nn.Parameter(
            person_b_params["transl"].to(target_device).float().detach().clone()
        )

        body_a = (
            nn.Parameter(
                person_a_params["body_pose"].to(target_device).float().detach().clone()
            )
            if "body_pose" in person_a_params and person_a_params["body_pose"] is not None
            else None
        )
        body_b = (
            nn.Parameter(
                person_b_params["body_pose"].to(target_device).float().detach().clone()
            )
            if "body_pose" in person_b_params and person_b_params["body_pose"] is not None
            else None
        )

        orient_a = (
            nn.Parameter(
                person_a_params["global_orient"].to(target_device).float().detach().clone()
            )
            if "global_orient" in person_a_params and person_a_params["global_orient"] is not None
            else None
        )
        orient_b = (
            nn.Parameter(
                person_b_params["global_orient"].to(target_device).float().detach().clone()
            )
            if "global_orient" in person_b_params and person_b_params["global_orient"] is not None
            else None
        )

        param_groups = [{"params": [transl_a, transl_b], "lr": lr_transl}]
        body_params = [p for p in [body_a, body_b] if p is not None]
        if body_params:
            param_groups.append({"params": body_params, "lr": lr_body})
        orient_params = [p for p in [orient_a, orient_b] if p is not None]
        if orient_params:
            param_groups.append({"params": orient_params, "lr": lr_orient})

        optimizer = torch.optim.Adam(param_groups)

        f = getattr(smplx_model, "faces_tensor", self.faces)
        if f is None and hasattr(smplx_model, "faces"):
            f = torch.as_tensor(smplx_model.faces, dtype=torch.long, device=target_device)

        # Detached anchor regularization targets
        anchor_transl_a = person_a_params["transl"].to(target_device).float().detach()
        anchor_transl_b = person_b_params["transl"].to(target_device).float().detach()
        anchor_body_a = (
            person_a_params["body_pose"].to(target_device).float().detach()
            if body_a is not None
            else None
        )
        anchor_body_b = (
            person_b_params["body_pose"].to(target_device).float().detach()
            if body_b is not None
            else None
        )
        anchor_orient_a = (
            person_a_params["global_orient"].to(target_device).float().detach()
            if orient_a is not None
            else None
        )
        anchor_orient_b = (
            person_b_params["global_orient"].to(target_device).float().detach()
            if orient_b is not None
            else None
        )

        # Pre-build detached base kwargs once outside the optimization loop
        base_kwargs_a = {
            k: (v.to(target_device).detach() if isinstance(v, torch.Tensor) else v)
            for k, v in person_a_params.items()
        }
        base_kwargs_b = {
            k: (v.to(target_device).detach() if isinstance(v, torch.Tensor) else v)
            for k, v in person_b_params.items()
        }

        for _ in range(num_steps):
            optimizer.zero_grad()

            kwargs_a = dict(base_kwargs_a)
            kwargs_a["transl"] = transl_a
            if body_a is not None:
                kwargs_a["body_pose"] = body_a
            if orient_a is not None:
                kwargs_a["global_orient"] = orient_a

            kwargs_b = dict(base_kwargs_b)
            kwargs_b["transl"] = transl_b
            if body_b is not None:
                kwargs_b["body_pose"] = body_b
            if orient_b is not None:
                kwargs_b["global_orient"] = orient_b

            out_a = smplx_model(**kwargs_a)
            out_b = smplx_model(**kwargs_b)

            va = out_a.vertices if hasattr(out_a, "vertices") else out_a["vertices"]
            vb = out_b.vertices if hasattr(out_b, "vertices") else out_b["vertices"]

            if va.dim() == 3:
                va = va[0]
            if vb.dim() == 3:
                vb = vb[0]

            l_pen = self.penetration_loss(va, vb, faces=f)
            l_cont = self.contact_loss(va, vb)

            # Anchor regularization to prevent drift (decoupled per person)
            l_reg = anchor_weight * (
                ((transl_a - anchor_transl_a) ** 2).sum()
                + ((transl_b - anchor_transl_b) ** 2).sum()
            )
            if body_a is not None:
                l_reg = l_reg + (anchor_weight * 0.5) * ((body_a - anchor_body_a) ** 2).sum()
            if body_b is not None:
                l_reg = l_reg + (anchor_weight * 0.5) * ((body_b - anchor_body_b) ** 2).sum()
            if orient_a is not None:
                l_reg = l_reg + (anchor_weight * 0.5) * ((orient_a - anchor_orient_a) ** 2).sum()
            if orient_b is not None:
                l_reg = l_reg + (anchor_weight * 0.5) * ((orient_b - anchor_orient_b) ** 2).sum()

            total_loss = pen_w * l_pen + cont_w * l_cont + l_reg
            total_loss.backward()
            optimizer.step()

        # Build output dictionaries matching exact input keys and original devices
        # Detach all frozen tensors to prevent memory leaks and downstream .numpy() errors
        refined_a = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in person_a_params.items()
        }
        refined_a["transl"] = transl_a.detach().to(
            orig_dev_a.get("transl", target_device)
        )
        if body_a is not None:
            refined_a["body_pose"] = body_a.detach().to(
                orig_dev_a.get("body_pose", target_device)
            )
        if orient_a is not None:
            refined_a["global_orient"] = orient_a.detach().to(
                orig_dev_a.get("global_orient", target_device)
            )

        refined_b = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in person_b_params.items()
        }
        refined_b["transl"] = transl_b.detach().to(
            orig_dev_b.get("transl", target_device)
        )
        if body_b is not None:
            refined_b["body_pose"] = body_b.detach().to(
                orig_dev_b.get("body_pose", target_device)
            )
        if orient_b is not None:
            refined_b["global_orient"] = orient_b.detach().to(
                orig_dev_b.get("global_orient", target_device)
            )

        return refined_a, refined_b

    joint_interaction_step = joint_step
