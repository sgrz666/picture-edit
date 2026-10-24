from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from .conditions import ConditionBundle, ContactRelationBatch, PART_NAMES, TaskType
from .geometry_encoder import SpatialConditionEncoder, normalize_scene_depth
from .body_pose import BodyPoseConfig
from .token_encoders import (
    BodyPoseTokenEncoder,
    ContactRasterEncoder,
    ExplicitTaskTokenEncoder,
    GlobalConditionTokenEncoder,
    RelativeGeometryTokenEncoder,
    StructuredContactRelationEncoder,
)


class SMPLXConditionInjector(nn.Module):
    """Encode prepared SMPL-X maps and global parameters into a stable bundle.

    This module deliberately stops before cross-person reasoning and DeepGen
    residual generation. It never estimates SMPL-X parameters or renders maps.
    """

    def __init__(
        self,
        *,
        use_depth: bool = False,
        normal_backend: str = "native",
        depth_backend: str = "native",
        spatial_dim: int = 256,
        token_dim: int = 256,
        contact_channels: int = 128,
        max_contact_relations: int = 8,
        use_full_pose_tokens: bool = False,
        pose_config: BodyPoseConfig | None = None,
    ) -> None:
        super().__init__()
        if use_full_pose_tokens and pose_config is None:
            raise ValueError('full pose tokens are reserved in legacy injection; Native v2 requires explicit pose_config')
        self.pose_config = pose_config
        if pose_config is not None:
            self.body_pose_encoder = BodyPoseTokenEncoder(pose_config)
        public_contract = {
            "spatial_dim": (spatial_dim, 256),
            "token_dim": (token_dim, 256),
            "contact_channels": (contact_channels, 128),
            "max_contact_relations": (max_contact_relations, 8),
        }
        for name, (actual, expected) in public_contract.items():
            if actual != expected:
                raise ValueError(
                    f"{name}={actual} violates the fixed public contract; expected {expected}"
                )
        for name, backend in (("normal", normal_backend), ("depth", depth_backend)):
            if backend not in {"native", "champ"}:
                raise ValueError(f"unsupported {name} backend: {backend}")
        self.use_depth = use_depth
        self.normal_backend = normal_backend
        self.depth_backend = depth_backend
        self.spatial_encoder = SpatialConditionEncoder(
            use_depth=use_depth,
            output_channels=spatial_dim,
            normal_backend=normal_backend,
            depth_backend=depth_backend,
        )
        self.global_encoder = GlobalConditionTokenEncoder(token_dim=token_dim)
        self.relative_encoder = RelativeGeometryTokenEncoder(token_dim=token_dim)
        self.contact_raster_encoder = ContactRasterEncoder(contact_channels)
        self.contact_relation_encoder = StructuredContactRelationEncoder(
            token_dim=token_dim, max_relations=max_contact_relations
        )
        self.task_encoder = ExplicitTaskTokenEncoder(token_dim=token_dim)
        self.spatial_dim = spatial_dim
        self.token_dim = token_dim
        self.max_contact_relations = max_contact_relations

    @staticmethod
    def _require_floating(name: str, value: torch.Tensor) -> None:
        if not value.is_floating_point():
            raise ValueError(f"{name} must have a floating dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")

    @classmethod
    def _validate_person(
        cls,
        *,
        prefix: str,
        normal: torch.Tensor,
        pose_heatmap: torch.Tensor,
        part_onehot: torch.Tensor,
        smplx_global: torch.Tensor,
        human_mask: torch.Tensor | None,
        depth: torch.Tensor | None,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        specifications = {
            "normal": ((batch_size, 3, height, width), normal),
            "pose_heatmap": ((batch_size, 25, height, width), pose_heatmap),
            "part_onehot": ((batch_size, len(PART_NAMES), height, width), part_onehot),
            "smplx_global": ((batch_size, 26), smplx_global),
        }
        for name, (expected, value) in specifications.items():
            if tuple(value.shape) != expected:
                raise ValueError(f"{prefix}_{name} must have shape {expected}")
            cls._require_floating(f"{prefix}_{name}", value)
        if normal.amin() < -1.0001 or normal.amax() > 1.0001:
            raise ValueError(f"{prefix}_normal must be in [-1, 1]")
        if pose_heatmap.amin() < -1e-5 or pose_heatmap.amax() > 1.00001:
            raise ValueError(f"{prefix}_pose_heatmap must be in [0, 1]")
        if part_onehot.amin() < -1e-5 or part_onehot.amax() > 1.00001:
            raise ValueError(f"{prefix}_part_onehot must be binary one-hot values")
        if not torch.allclose(part_onehot, part_onehot.round(), atol=1e-5, rtol=0):
            raise ValueError(f"{prefix}_part_onehot must contain binary one-hot values")
        if torch.any(part_onehot.sum(dim=1) > 1.00001):
            raise ValueError(f"{prefix}_part_onehot must be one-hot with all-zero background")
        if human_mask is None:
            human_mask = (part_onehot.sum(dim=1, keepdim=True) > 0).to(normal.dtype)
        else:
            if tuple(human_mask.shape) != (batch_size, 1, height, width):
                raise ValueError(f"{prefix}_human_mask must have shape {(batch_size, 1, height, width)}")
            cls._require_floating(f"{prefix}_human_mask", human_mask)
            if human_mask.amin() < -1e-5 or human_mask.amax() > 1.00001:
                raise ValueError(f"{prefix}_human_mask must be in [0, 1]")
        if depth is not None:
            if tuple(depth.shape) != (batch_size, 1, height, width):
                raise ValueError(f"{prefix}_depth must have shape {(batch_size, 1, height, width)}")
            cls._require_floating(f"{prefix}_depth", depth)
        return human_mask

    @staticmethod
    def _same_device(name: str, tensors: Iterable[torch.Tensor | None], expected: torch.device) -> None:
        for value in tensors:
            if value is not None and value.device != expected:
                raise ValueError(f"{name} tensors must all be on device {expected}")

    def forward(
        self,
        *,
        normal_a: torch.Tensor,
        pose_heatmap_a: torch.Tensor,
        part_onehot_a: torch.Tensor,
        smplx_global_a: torch.Tensor,
        task_id: torch.Tensor,
        human_mask_a: torch.Tensor | None = None,
        depth_a: torch.Tensor | None = None,
        normal_b: torch.Tensor | None = None,
        pose_heatmap_b: torch.Tensor | None = None,
        part_onehot_b: torch.Tensor | None = None,
        smplx_global_b: torch.Tensor | None = None,
        human_mask_b: torch.Tensor | None = None,
        depth_b: torch.Tensor | None = None,
        person_b_valid: torch.Tensor | None = None,
        relative_geometry: torch.Tensor | None = None,
        contact_raster: torch.Tensor | None = None,
        contact_relations: ContactRelationBatch | None = None,
        body_pose_a: torch.Tensor | None = None,
        joints_cam_a: torch.Tensor | None = None,
        joint_uv_a: torch.Tensor | None = None,
        fk_valid_a: torch.Tensor | None = None,
        projection_valid_a: torch.Tensor | None = None,
        body_height_a: torch.Tensor | None = None,
        smpl_bbox_height_a: torch.Tensor | None = None,
        body_pose_b: torch.Tensor | None = None,
        joints_cam_b: torch.Tensor | None = None,
        joint_uv_b: torch.Tensor | None = None,
        fk_valid_b: torch.Tensor | None = None,
        projection_valid_b: torch.Tensor | None = None,
        body_height_b: torch.Tensor | None = None,
        smpl_bbox_height_b: torch.Tensor | None = None,
    ) -> ConditionBundle:
        if normal_a.ndim != 4:
            raise ValueError("normal_a must have shape [B,3,H,W]")
        batch_size, _, height, width = normal_a.shape
        if height % 8 or width % 8:
            raise ValueError("condition height and width must be divisible by 8")
        if task_id.shape != (batch_size,) or task_id.dtype != torch.long:
            raise ValueError("task_id must have shape [B] and dtype int64")
        if torch.any((task_id < 0) | (task_id >= len(TaskType))):
            raise ValueError("task_id contains an unknown task enum value")

        b_core = (normal_b, pose_heatmap_b, part_onehot_b, smplx_global_b)
        provided = tuple(value is not None for value in b_core)
        if any(provided) and not all(provided):
            raise ValueError("normal_b, pose_heatmap_b, part_onehot_b and smplx_global_b must all be provided")
        has_b = all(provided)
        if not has_b and any(
            value is not None
            for value in (
                human_mask_b,
                depth_b,
                person_b_valid,
                relative_geometry,
                contact_raster,
                contact_relations,
            )
        ):
            raise ValueError("person-B, relative and contact conditions require all person-B core inputs")

        device = normal_a.device
        self._same_device(
            "condition",
            (
                pose_heatmap_a, part_onehot_a, smplx_global_a, task_id,
                human_mask_a, depth_a, normal_b, pose_heatmap_b, part_onehot_b,
                smplx_global_b, human_mask_b, depth_b, person_b_valid,
                relative_geometry, contact_raster,
            ),
            device,
        )
        mask_a = self._validate_person(
            prefix="person_a",
            normal=normal_a,
            pose_heatmap=pose_heatmap_a,
            part_onehot=part_onehot_a,
            smplx_global=smplx_global_a,
            human_mask=human_mask_a,
            depth=depth_a,
            batch_size=batch_size,
            height=height,
            width=width,
        )
        valid_a = torch.ones(batch_size, dtype=torch.bool, device=device)
        valid_b = torch.zeros(batch_size, dtype=torch.bool, device=device)
        mask_b = None
        if has_b:
            valid_b = (
                torch.ones(batch_size, dtype=torch.bool, device=device)
                if person_b_valid is None
                else person_b_valid
            )
            if valid_b.shape != (batch_size,) or valid_b.dtype != torch.bool:
                raise ValueError("person_b_valid must have shape [B] and dtype bool")
            mask_b = self._validate_person(
                prefix="person_b",
                normal=normal_b,
                pose_heatmap=pose_heatmap_b,
                part_onehot=part_onehot_b,
                smplx_global=smplx_global_b,
                human_mask=human_mask_b,
                depth=depth_b,
                batch_size=batch_size,
                height=height,
                width=width,
            )

        person_count = valid_a.long() + valid_b.long()
        single_task = (task_id == int(TaskType.SINGLE)) | (task_id == int(TaskType.UNKNOWN))
        if torch.any((person_count == 1) & ~single_task):
            raise ValueError("single-person rows require SINGLE or UNKNOWN task_id")
        if torch.any((person_count == 2) & (task_id == int(TaskType.SINGLE))):
            raise ValueError("dual-person rows cannot use SINGLE task_id")

        normalized_depth_a = depth_a
        normalized_depth_b = depth_b
        if self.use_depth:
            normalized_depth_a, normalized_depth_b = normalize_scene_depth(
                depth_a, depth_b, mask_a, mask_b, valid_b
            )
        spatial_a, output_mask_a = self.spatial_encoder(
            normal=normal_a,
            pose_heatmap=pose_heatmap_a,
            part_onehot=part_onehot_a,
            human_mask=mask_a,
            depth=normalized_depth_a,
            role_id=0,
            valid=valid_a,
        )
        global_a = self.global_encoder(smplx_global_a, valid_a)

        spatial_b = output_mask_b = global_b = None
        if has_b:
            spatial_b, output_mask_b = self.spatial_encoder(
                normal=normal_b,
                pose_heatmap=pose_heatmap_b,
                part_onehot=part_onehot_b,
                human_mask=mask_b,
                depth=normalized_depth_b,
                role_id=1,
                valid=valid_b,
            )
            global_b = self.global_encoder(smplx_global_b, valid_b)

        relative_tokens = None
        if relative_geometry is not None:
            if tuple(relative_geometry.shape) != (batch_size, 11):
                raise ValueError("relative_geometry must have shape [B,11]")
            self._require_floating("relative_geometry", relative_geometry)
            relative_tokens = self.relative_encoder(relative_geometry, valid_b)

        contact_spatial = None
        if contact_raster is not None:
            if tuple(contact_raster.shape) != (batch_size, 2, height, width):
                raise ValueError("contact_raster must have shape [B,2,H,W]")
            self._require_floating("contact_raster", contact_raster)
            contact_spatial = self.contact_raster_encoder(contact_raster, valid_b)

        contact_tokens = contact_mask = None
        if contact_relations is not None:
            contact_relations = contact_relations.to(device=device, dtype=normal_a.dtype)
            contact_tokens, contact_mask = self.contact_relation_encoder(
                contact_relations, valid_b
            )

        pose_fields = {}
        raw_pose = ((body_pose_a, joints_cam_a, joint_uv_a, fk_valid_a, projection_valid_a, body_height_a, smpl_bbox_height_a),
                    (body_pose_b, joints_cam_b, joint_uv_b, fk_valid_b, projection_valid_b, body_height_b, smpl_bbox_height_b))
        if self.pose_config is None:
            if any(v is not None for group in raw_pose for v in group):
                raise ValueError('Body Pose is disabled in explicit legacy mode')
        else:
            for person, group, present, valid in zip(('a', 'b'), raw_pose, (True, has_b), (valid_a, valid_b)):
                if not present:
                    if any(v is not None for v in group):
                        raise ValueError('Body Pose B requires person-B core inputs')
                    continue
                if any(v is None for v in group):
                    raise ValueError(f'Body Pose v2 requires all canonical pose fields for person {person}')
                pose, joints, uv, fk, projected, body_height, bbox = group
                self._same_device('Body Pose', group, device)
                if bbox.shape != (batch_size,) or not torch.isfinite(bbox).all() or (bbox < 0).any():
                    raise ValueError('SMPL mask bbox height must be finite nonnegative [B]')
                if (valid & ~fk[:, 0]).any() or (valid & ~fk[:, 1:].any(1)).any():
                    raise ValueError('Valid person requires pelvis and body FK, never a zero Pose fallback')
                canvas = normal_a.new_tensor([height, width])[None].expand(batch_size, -1)
                fk = fk & valid[:, None]
                projected = projected & valid[:, None]
                tokens = self.body_pose_encoder(pose, joints, uv, fk, projected, body_height, canvas)
                sigma = (bbox * self.pose_config.bbox_sigma_ratio).clamp_min(self.pose_config.min_sigma_px)
                for key, value in (('joint_tokens', tokens), ('joint_uv', uv), ('fk_valid', fk[:, 1:]),
                                   ('projection_valid', projected), ('joint_sigma', sigma), ('pose_canvas_hw', canvas)):
                    pose_fields[f'person_{person}_{key}'] = value

        return ConditionBundle(
            person_a_spatial=spatial_a,
            person_b_spatial=spatial_b,
            person_a_mask=output_mask_a,
            person_b_mask=output_mask_b,
            person_a_global_tokens=global_a,
            person_b_global_tokens=global_b,
            relative_tokens=relative_tokens,
            contact_spatial=contact_spatial,
            contact_tokens=contact_tokens,
            contact_mask=contact_mask,
            task_token=self.task_encoder(task_id),
            person_valid=torch.stack((valid_a, valid_b), dim=1),
            person_count=person_count,
            **pose_fields,
        )
