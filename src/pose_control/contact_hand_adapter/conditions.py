"""Target-only geometry and ordered source appearance for Contact Hand Adapter."""

from dataclasses import dataclass, fields, replace

import torch

from src.data.canvas_contract import CANVAS_MULTIPLE, VAE_STRIDE, CanvasSpec


HAND_SLOTS = ("A-L", "A-R", "B-L", "B-R")
UNKNOWN, SEPARATE, TOUCHING = -1, 0, 1
JOINT_PARTS = (0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15)
JOINT_PARENTS = (0, *(0 if i % 4 == 1 else i - 1 for i in range(1, 21)))


@dataclass(frozen=True)
class HandPair:
    """One ordered edge (smaller slot first); reverse lookup transposes part relations."""

    slots: tuple[int, int]
    part_state: torch.Tensor
    part_confidence: torch.Tensor | None = None
    occlusion: torch.Tensor | None = None  # -1: second in front, +1: first in front, 0: unknown
    relative_pose: torch.Tensor | None = None  # translation and axis-angle, both in first hand frame
    surface_indices: torch.Tensor | None = None  # [E, 2] canonical 64-anchor indices
    surface_features: torch.Tensor | None = None  # [E, 4]: distance (m), normal dot, state, confidence
    surface_quality: torch.Tensor | None = None  # [E,4]: fit, normal, cross-view and penetration quality


@dataclass(frozen=True)
class HandMesh:
    """Template-indexed MANO vertices and explicit projection in target canvas coordinates."""

    vertices: torch.Tensor  # [4, 778, 3], per-hand coordinates in meters
    edges: torch.Tensor  # [2, E], undirected template adjacency added by the encoder
    anchors: torch.Tensor  # [64, K], fixed vertex groups, four for every consecutive part
    anchor_xy: torch.Tensor  # [4, 64, 2] target canvas pixels
    valid: torch.Tensor  # [4] bool
    camera_ids: tuple[str | None, ...] = (None, None, None, None)


@dataclass(frozen=True)
class HandCondition:
    canvas: CanvasSpec
    person_ids: tuple[str | None, str | None]
    joints: torch.Tensor  # [4, 21, 3], canvas x, y and confidence: wrist, then fingers root to tip
    visible: torch.Tensor  # [4, 21]
    support: torch.Tensor | None = None  # [4, H, W] target hands and wrists
    pairs: tuple[HandPair, ...] = ()
    mesh: HandMesh | None = None
    geometry_maps: torch.Tensor | None = None  # [4, 5, H, W], depth (m), normals (xyz), part/15
    map_valid: torch.Tensor | None = None  # optional [4, 5, H, W] per-channel/pixel validity
    appearance: torch.Tensor | None = None  # [4, 3, H_src, W_src], RGB in [0, 1]
    appearance_valid: torch.Tensor | None = None  # [4], matched by source actor and anatomical side
    appearance_sources: tuple[str | None, ...] = (None, None, None, None)  # SHA256 of each crop's source image
    shared_appearance_source: bool = False  # both actors appear in the first source reference
    joints_3d: torch.Tensor | None = None  # [4,21,3], camera coordinates in meters
    joint_3d_valid: torch.Tensor | None = None  # [4,21] bool
    joint_2d_valid: torch.Tensor | None = None  # [4,21] bool, calibrated projection, including out of canvas
    joint_occluded: torch.Tensor | None = None  # [4,21] bool, known physical occlusion
    hand_present: torch.Tensor | None = None  # [4] bool
    joint_rotation_6d: torch.Tensor | None = None  # [4,21,6], first two rotation columns, flattened per column
    joint_rotation_valid: torch.Tensor | None = None  # [4,21] bool, fingertips remain invalid
    mano_shape: torch.Tensor | None = None  # [4,10]
    mano_wrist_rotation_6d: torch.Tensor | None = None  # [4,6]
    mano_palm_scale: torch.Tensor | None = None  # [4], meters
    mano_valid: torch.Tensor | None = None  # [4] bool

    def to(self, device: torch.device | str) -> "HandCondition":
        return _move(self, device)

    def project_validity(self) -> torch.Tensor:
        x, y, confidence = self.joints.unbind(-1)
        declared = self.visible if self.joint_2d_valid is None else self.joint_2d_valid
        return declared & torch.isfinite(self.joints).all(-1) & (confidence > 0) & (confidence <= 1)

    def projected_joints(self) -> torch.Tensor:
        x0, y0, x1, y1 = self.canvas.valid_rect
        x, y = self.joints[..., :2].unbind(-1)
        return self.project_validity() & (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

    def active_joints(self) -> torch.Tensor:
        return self.visible & self.projected_joints()

    def anatomy_joints(self) -> torch.Tensor:
        points = self.projected_joints()
        return points if self.joint_3d_valid is None else points | self.joint_3d_valid

    def present_hands(self) -> torch.Tensor:
        if self.hand_present is not None:
            return self.hand_present
        inferred = self.project_validity()
        if self.joint_3d_valid is not None:
            inferred = inferred | self.joint_3d_valid
        return inferred.any(1)

    def spatial_joint_confidence(self) -> torch.Tensor:
        weight = torch.where(self.visible, 1., .55)
        if self.joint_occluded is not None:
            weight = torch.where(self.joint_occluded, .20, weight)
        return torch.where(self.projected_joints(), self.joints[..., 2].float() * weight, 0.)

    def validate(self) -> "HandCondition":
        h, w = self.canvas.canvas_hw
        if self.canvas.stride != VAE_STRIDE or h % CANVAS_MULTIPLE or w % CANVAS_MULTIPLE:
            raise ValueError("Hand canvas must use the native Qwen 16-pixel grid and 32-pixel alignment")
        _tensor(self.joints, (4, 21, 3), floating=True)
        _tensor(self.visible, (4, 21), dtype=torch.bool)
        self._validate_anatomy()
        if (len(self.person_ids) != 2 or any(pid is not None and (not isinstance(pid, str) or not pid)
                                             for pid in self.person_ids)
                or (self.person_ids[0] is not None and self.person_ids[0] == self.person_ids[1])):
            raise ValueError("An ordered pair of distinct person IDs is required")
        if not torch.equal(self.visible, self.active_joints()):
            raise ValueError("Visible target joints must be finite, in the valid canvas, and have confidence in (0,1]")
        hands = self.present_hands()
        for actor, person_id in enumerate(self.person_ids):
            if person_id is None and hands[actor * 2: actor * 2 + 2].any():
                raise ValueError("Present hand slots require an explicit person ID")
        if self.support is not None:
            _tensor(self.support, (4, h, w), floating=True)
            _unit(self.support)
        if (self.geometry_maps is None) != (self.map_valid is None):
            raise ValueError("Dense geometry and validity must be supplied together")
        if self.geometry_maps is not None:
            _tensor(self.geometry_maps, (4, 5, h, w), floating=True)
            _tensor(self.map_valid, (4, 5, h, w), dtype=torch.bool)
            _finite(self.geometry_maps[self.map_valid])
        if (self.appearance is None) != (self.appearance_valid is None):
            raise ValueError("Source appearance and validity must be supplied together")
        if len(self.appearance_sources) != 4 or any(
            sha is not None and (not isinstance(sha, str) or len(sha) != 64
                                 or any(c not in "0123456789abcdef" for c in sha))
            for sha in self.appearance_sources
        ):
            raise ValueError("Invalid source appearance SHA256")
        if self.appearance is not None:
            if self.appearance.ndim != 4 or self.appearance.shape[:2] != (4, 3):
                raise ValueError("Source appearance must be [4,3,H,W]")
            _tensor(self.appearance_valid, (4,), dtype=torch.bool)
            _unit(self.appearance[self.appearance_valid])
            if any(self.appearance_sources[slot] is None for slot in self.appearance_valid.nonzero().flatten().tolist()):
                raise ValueError("Valid source appearance requires per-slot source image SHA256")
        if self.mesh is not None:
            mesh = self.mesh
            _tensor(mesh.vertices, (4, 778, 3), floating=True)
            _tensor(mesh.anchor_xy, (4, 64, 2), floating=True)
            _tensor(mesh.valid, (4,), dtype=torch.bool)
            _indices(mesh.edges, 2, 778, rows=True)
            _indices(mesh.anchors, 64, 778, rows=True)
            _finite(mesh.vertices[mesh.valid])
            _finite(mesh.anchor_xy[mesh.valid])
            if self.hand_present is not None and (mesh.valid & ~hands).any():
                raise ValueError("A valid MANO mesh must belong to a present hand")
            if len(mesh.camera_ids) != 4:
                raise ValueError("MANO camera IDs must align with the four hand slots")
        seen = set()
        for pair in self.pairs:
            first, second = pair.slots
            if not 0 <= first < second < 4 or pair.slots in seen:
                raise ValueError("Hand pairs must be unique and stored in increasing slot order")
            seen.add(pair.slots)
            _tensor(pair.part_state, (16, 16))
            _states(pair.part_state)
            if pair.part_confidence is not None:
                _tensor(pair.part_confidence, (16, 16), floating=True)
                _unit(pair.part_confidence)
            if pair.occlusion is not None:
                _tensor(pair.occlusion, (16, 16))
                _states(pair.occlusion)
            if (pair.surface_indices is None) != (pair.surface_features is None):
                raise ValueError("Surface relations require indices and features")
            if pair.surface_quality is not None:
                if pair.surface_indices is None:
                    raise ValueError("Surface quality requires a surface relation")
                _tensor(pair.surface_quality, (len(pair.surface_indices), 4), floating=True)
                _unit(pair.surface_quality)
            if pair.relative_pose is not None or pair.surface_indices is not None:
                mesh = self.mesh
                same_camera = (mesh is not None and mesh.valid[first] and mesh.valid[second]
                               and mesh.camera_ids[first] and mesh.camera_ids[first] == mesh.camera_ids[second])
                if not same_camera:
                    raise ValueError("3D relations require two valid meshes in a verified common camera")
            if pair.relative_pose is not None:
                _tensor(pair.relative_pose, (6,), floating=True)
                _finite(pair.relative_pose)
            if pair.surface_indices is not None:
                _indices(pair.surface_indices, 2, 64)
                _tensor(pair.surface_features, (pair.surface_indices.shape[0], 4), floating=True)
                _finite(pair.surface_features)
                distance, normal, state, confidence = pair.surface_features.unbind(-1)
                _states(state)
                _unit(confidence)
                if (distance < 0).any() or (normal.abs() > 1).any():
                    raise ValueError("Surface distances and normal dot products are out of range")
                if pair.surface_indices.unique(dim=0).shape[0] != pair.surface_indices.shape[0]:
                    raise ValueError("Surface contact indices must be unique")
        return self

    def _validate_anatomy(self):
        for mask in (self.joint_2d_valid, self.joint_3d_valid, self.joint_occluded, self.joint_rotation_valid):
            if mask is not None:
                _tensor(mask, (4, 21), dtype=torch.bool)
        if self.joint_2d_valid is not None and not torch.equal(self.joint_2d_valid, self.project_validity()):
            raise ValueError("Valid 2D projections must have finite coordinates and confidence in (0,1]")
        if self.joint_occluded is not None and (self.joint_occluded & (self.visible | ~self.projected_joints())).any():
            raise ValueError("An occluded joint needs an in-frame projection and cannot also be visible")
        if self.hand_present is not None:
            _tensor(self.hand_present, (4,), dtype=torch.bool)
            if ((self.project_validity() | self.anatomy_joints()).any(1) & ~self.hand_present).any():
                raise ValueError("Valid joints must belong to a present hand")
        for value, valid, width in ((self.joints_3d, self.joint_3d_valid, 3),
                                    (self.joint_rotation_6d, self.joint_rotation_valid, 6)):
            if (value is None) != (valid is None):
                raise ValueError("3D coordinates and rotations each require their independent validity mask")
            if value is not None:
                _tensor(value, (4, 21, width), floating=True)
                _finite(value[valid])
                if (valid.any(1) & ~self.present_hands()).any():
                    raise ValueError("Valid 3D features must belong to a present hand")
        if self.joint_rotation_valid is not None and self.joint_rotation_valid[:, 4::4].any():
            raise ValueError("MANO fingertips have no independent joint rotation")
        fields = (self.mano_shape, self.mano_wrist_rotation_6d, self.mano_palm_scale, self.mano_valid)
        if any(value is not None for value in fields):
            if any(value is None for value in fields):
                raise ValueError("Global MANO needs shape, wrist rotation, palm size and validity")
            _tensor(self.mano_valid, (4,), dtype=torch.bool)
            if (self.mano_valid & ~self.present_hands()).any():
                raise ValueError("Valid MANO parameters must belong to a present hand")
            for value, shape in zip(fields, ((4, 10), (4, 6), (4,))):
                _tensor(value, shape, floating=True)
                _finite(value[self.mano_valid])
            if (self.mano_palm_scale[self.mano_valid] <= 0).any():
                raise ValueError("Valid MANO palm scales must be positive")


def invert_relative_pose(pose):
    """Invert translation and axis-angle expressed in the first hand's reference frame."""
    rotation = pose[3:]
    angle = rotation.norm()
    axis = rotation / angle.clamp_min(1e-8)
    xyz = pose[:3]
    inverse_xyz = -(xyz * angle.cos() - torch.linalg.cross(axis, xyz) * angle.sin()
                    + axis * axis.dot(xyz) * (1 - angle.cos()))
    return torch.cat((inverse_xyz, -rotation))


def verify_appearance_sources(condition, source_hashes):
    """Attach crop provenance to the actual ordered references, never to target RGB."""
    if condition.appearance_valid is None or not condition.appearance_valid.any():
        return
    source_hashes = tuple(source_hashes or ())
    for slot in condition.appearance_valid.nonzero().flatten().tolist():
        source = 0 if condition.shared_appearance_source else slot // 2
        if source >= len(source_hashes) or source_hashes[source] != condition.appearance_sources[slot]:
            raise ValueError(f"Source appearance for {HAND_SLOTS[slot]} does not match the ordered source reference")


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(member, device) for member in value)
    if isinstance(value, (HandCondition, HandPair, HandMesh)):
        return replace(value, **{field.name: _move(getattr(value, field.name), device) for field in fields(value)})
    return value


def _tensor(value, shape, *, dtype=None, floating=False):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
            or (dtype is not None and value.dtype != dtype) or (floating and not value.is_floating_point())):
        raise ValueError(f"Expected tensor {shape} with the declared dtype")


def _indices(value, size, end, *, rows=False):
    if (value.ndim != 2 or value.shape[0 if rows else 1] != size
            or value.shape[1] == 0 or value.dtype != torch.long
            or ((value < 0) | (value >= end)).any()):
        raise ValueError(f"Invalid template indices for [0, {end})")


def _finite(value):
    if not torch.isfinite(value).all():
        raise ValueError("A valid hand feature is non-finite")


def _unit(value):
    _finite(value)
    if ((value < 0) | (value > 1)).any():
        raise ValueError("Hand probabilities and coverage must be in [0,1]")


def _states(value):
    if not ((value == UNKNOWN) | (value == SEPARATE) | (value == TOUCHING)).all():
        raise ValueError("Expected a three-state hand relation")
