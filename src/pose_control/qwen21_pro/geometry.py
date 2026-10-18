"""Source-shape geometry for Pro controls, independent of legacy estimators.

Coordinates and lengths are explicit: body parameters are in ``camera_cv`` or
``world`` metres; CV is +X right, +Y down, +Z forward. Stable person IDs are
separate from raster slots. External body models and nvdiffrast load lazily.
No function in this module reads RGB or writes model/dataset assets.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Protocol

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


_FAMILIES = {"smpl": (69, 6890), "smplx": (63, 10475)}
HI4D_MALE_MODEL_SHA256 = "ed4d55bb3041fefc6f73b70694d6c8edc1020c0d07340be5cc651cae2c6a6ae3"
HI4D_FEMALE_MODEL_SHA256 = "6d4a1791b6b94880397e1a3a4539b703a228d2150c57de7b288389a8115f4ef0"
# Original-weight CPU audit, tensor_sha256(long [6890]); float32 LBS argmax.
HI4D_PARTS_SHA256 = {
    "male": "f6ac0bb73e15ae1b567bc0b2070f966de52fdba55023c9fa88f1f10ee66dc7d5",
    "female": "5d486c82a83e3bca518ade5da9df9fc4771dab6e1f102d80e06157bfc5ea46df",
}
_SMPL_TO_V6 = (1, 8, 11, 1, 9, 12, 1, 10, 13, 1, 10, 13,
               0, 1, 1, 0, 2, 5, 3, 6, 4, 7, 4, 7)
_SMPLX_TO_V6 = (1, 8, 11, 1, 9, 12, 1, 10, 13, 1, 10, 13,
                0, 1, 1, 0, 2, 5, 3, 6, 4, 7, 0, 0, 0) + (4,) * 15 + (7,) * 15


def _ids(person_ids):
    if not isinstance(person_ids, tuple) or not 1 <= len(person_ids) <= 2:
        raise ValueError("person_ids must be an ordered tuple of one or two stable IDs")
    if any(type(pid) is not int or pid < 0 for pid in person_ids) or len(set(person_ids)) != len(person_ids):
        raise ValueError("person_ids must be unique nonnegative integers")


def _float_tensor(name, value, shape):
    if not isinstance(value, Tensor) or tuple(value.shape) != tuple(shape) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor with shape {tuple(shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _positive(name, value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _hw(hw):
    if not isinstance(hw, tuple) or len(hw) != 2 or any(type(x) is not int or x <= 0 for x in hw):
        raise ValueError("image/canvas hw must be two positive integers")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceShape:
    family: str
    person_ids: tuple[int, ...]
    betas: Tensor

    def __post_init__(self):
        _ids(self.person_ids)
        if self.family not in _FAMILIES:
            raise ValueError("family must be smpl or smplx")
        _float_tensor("source betas", self.betas, (len(self.person_ids), 10))


@dataclass(frozen=True)
class BodyParameters:
    family: str
    person_ids: tuple[int, ...]
    betas: Tensor
    global_orient: Tensor
    body_pose: Tensor
    transl: Tensor
    frame: str
    left_hand_pose: Tensor | None = None
    right_hand_pose: Tensor | None = None
    jaw_pose: Tensor | None = None
    expression: Tensor | None = None

    def __post_init__(self):
        _ids(self.person_ids)
        if self.family not in _FAMILIES:
            raise ValueError("family must be smpl or smplx")
        if self.frame not in {"camera_cv", "world"}:
            raise ValueError("frame must be camera_cv or world")
        p = len(self.person_ids)
        for name, size in (("betas", 10), ("global_orient", 3), ("transl", 3), ("body_pose", _FAMILIES[self.family][0])):
            _float_tensor(name, getattr(self, name), (p, size))
        for name, size in (("left_hand_pose", 45), ("right_hand_pose", 45), ("jaw_pose", 3), ("expression", 10)):
            value = getattr(self, name)
            if self.family == "smplx":
                _float_tensor(name, value, (p, size))
            elif value is not None:
                raise ValueError(f"SMPL has no {name}; unavailable fingers/face must not be fabricated")
        tensors = [getattr(self, f.name) for f in fields(self) if isinstance(getattr(self, f.name), Tensor)]
        if any(t.device != self.betas.device or t.dtype != self.betas.dtype for t in tensors):
            raise ValueError("body parameter tensors must share device and dtype")

    def copied(self):
        return replace(self, **{f.name: getattr(self, f.name).detach().clone() for f in fields(self)
                                if isinstance(getattr(self, f.name), Tensor)})


@dataclass(frozen=True)
class PixelCanvasTransform:
    """Physical camera pixels -> output pixels, without changing the camera."""
    matrix: Tensor
    input_hw: tuple[int, int]
    output_hw: tuple[int, int]

    def __post_init__(self):
        _hw(self.input_hw)
        _hw(self.output_hw)
        _float_tensor("pixel canvas matrix", self.matrix, (3, 3))
        m = self.matrix.detach().cpu()
        if not torch.equal(m[2], m.new_tensor([0, 0, 1])) or torch.det(m).abs() < 1e-12:
            raise ValueError("pixel canvas matrix must be an invertible affine transform")

    @classmethod
    def letterbox(cls, input_hw, output_hw):
        _hw(input_hw)
        _hw(output_hw)
        h, w = input_hw
        oh, ow = output_hw
        scale = min(oh / h, ow / w)
        rh, rw = round(h * scale), round(w * scale)
        # Record actual rounded resize dimensions, including independent x/y scale.
        return cls(torch.tensor([[rw / w, 0., (ow - rw) // 2], [0., rh / h, (oh - rh) // 2],
                                 [0., 0., 1.]]), input_hw, output_hw)

    def apply(self, xy):
        matrix = self.matrix.to(xy)
        return xy @ matrix[:2, :2].T + matrix[:2, 2]

    def valid_fov_mask(self, *, device=None):
        """Output pixel centres inside the original physical image FoV.

        Rasterizing a complete mesh into a letterboxed canvas would otherwise
        expose off-image geometry in the padding. Apply the full inverse affine
        map, including rotation/shear/reflection, to top-down centres (x+.5,y+.5).
        The original image occupies the half-open area [0,W) x [0,H).
        """
        matrix = self.matrix.detach().to(device=device, dtype=torch.float64)
        h, w = self.output_hw
        yy, xx = torch.meshgrid(torch.arange(h, device=matrix.device, dtype=matrix.dtype) + .5,
                                torch.arange(w, device=matrix.device, dtype=matrix.dtype) + .5, indexing="ij")
        centres = torch.stack([xx, yy, torch.ones_like(xx)], -1)
        native = centres @ torch.linalg.inv(matrix).T
        input_h, input_w = self.input_hw
        return ((native[..., 0] >= 0) & (native[..., 0] < input_w)
                & (native[..., 1] >= 0) & (native[..., 1] < input_h))

    def to_dict(self):
        return {"matrix": self.matrix.detach().cpu().tolist(), "input_hw": list(self.input_hw),
                "output_hw": list(self.output_hw), "coordinates": "top_down_pixel_xy"}


@dataclass(frozen=True)
class Camera:
    """Pinhole camera. world_to_camera applies Xcv = Xworld @ R.T + t."""
    K: Tensor
    image_hw: tuple[int, int]
    world_to_camera: Tensor | None = None

    def __post_init__(self):
        _float_tensor("camera K", self.K, (3, 3))
        _hw(self.image_hw)
        k = self.K.detach().cpu()
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not torch.equal(k[2], k.new_tensor([0, 0, 1])):
            raise ValueError("camera K must be valid pinhole intrinsics")
        if self.world_to_camera is not None:
            _float_tensor("camera extrinsics", self.world_to_camera, (3, 4))
            rotation = self.world_to_camera[:, :3].detach().double().cpu()
            if not torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=torch.double), atol=1e-5, rtol=0) or torch.det(rotation) < 0:
                raise ValueError("camera extrinsics must contain a proper rotation")

    def copied(self):
        return replace(self, K=self.K.detach().clone(), world_to_camera=None if self.world_to_camera is None else self.world_to_camera.detach().clone())

    def to_camera(self, xyz, *, frame):
        if frame == "camera_cv":
            return xyz
        if frame != "world" or self.world_to_camera is None:
            raise ValueError("world coordinates require explicit camera extrinsics")
        extr = self.world_to_camera.to(xyz)
        return xyz @ extr[:, :3].T + extr[:, 3]

    def project(self, xyz, *, frame, canvas=None, z_epsilon=0.):
        if not math.isfinite(z_epsilon) or z_epsilon < 0:
            raise ValueError("z_epsilon must be finite and nonnegative")
        cv = self.to_camera(xyz, frame=frame)
        intrinsic = self.K.to(cv)
        xy = (cv[..., :2] @ intrinsic[:2, :2].T) / (cv[..., 2:3] + z_epsilon) + intrinsic[:2, 2]
        xy = torch.where((cv[..., 2:] > 0) & torch.isfinite(cv).all(-1, keepdim=True), xy, torch.full_like(xy, float("nan")))
        if canvas is not None:
            if canvas.input_hw != self.image_hw:
                raise ValueError("canvas input dimensions do not match physical camera")
            xy = canvas.apply(xy)
        return xy


@dataclass(frozen=True)
class BodyOutput:
    vertices: Tensor
    joints: Tensor

    def __post_init__(self):
        for name in ("vertices", "joints"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.ndim != 3 or value.shape[-1] != 3 or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"body {name} must be finite [P,N,3] floating tensors")
        if self.vertices.shape[0] != self.joints.shape[0]:
            raise ValueError("body vertices and joints must share person dimension")

    def copied(self):
        return BodyOutput(self.vertices.detach().clone(), self.joints.detach().clone())


class BodyBackend(Protocol):
    family: str
    vertex_count: int
    faces: Tensor
    vertex_parts: Tensor
    provenance: dict

    def forward(self, parameters: BodyParameters) -> BodyOutput: ...


class BodyModelBackend:
    """Frozen SMPL/SMPL-X weights; input pose gradients remain enabled."""
    def __init__(self, family, layer, faces, *, joint_indices=None, vertex_parts, provenance=None):
        if family not in _FAMILIES:
            raise ValueError("unknown body family")
        self.family, self.layer = family, layer.eval()
        self.vertex_count = _FAMILIES[family][1]
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        self.faces = torch.as_tensor(np.asarray(faces).copy(), dtype=torch.long)
        if self.faces.ndim != 2 or self.faces.shape[-1] != 3 or self.faces.numel() == 0 or self.faces.min() < 0 or self.faces.max() >= self.vertex_count:
            raise ValueError("invalid body triangle topology")
        self.vertex_parts = torch.as_tensor(vertex_parts, dtype=torch.long).detach().cpu().clone()
        if tuple(self.vertex_parts.shape) != (self.vertex_count,) or self.vertex_parts.min() < 0 or self.vertex_parts.max() > 13:
            raise ValueError("vertex parts must use V6 zero-based 14 part labels")
        self.joint_indices = tuple(joint_indices) if joint_indices is not None else None
        # The real mapping repeats model joints 57/56 for different semantic slots.
        if family == "smplx" and (self.joint_indices is None or len(self.joint_indices) != 137
                                  or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) or i < 0 for i in self.joint_indices)):
            raise ValueError("SMPL-X requires the actual SMPLest-X joint_idx137 mapping")
        self.provenance = dict(provenance or {})

    def forward(self, parameters):
        if parameters.family != self.family:
            raise ValueError("body model and parameter family mismatch")
        tensors = list(self.layer.parameters()) + list(self.layer.buffers())
        template = next((t for t in tensors if t.is_floating_point()), parameters.betas)
        kwargs = {f.name: getattr(parameters, f.name).to(device=template.device, dtype=template.dtype)
                  for f in fields(parameters) if isinstance(getattr(parameters, f.name), Tensor)}
        if self.family == "smplx":
            kwargs["leye_pose"] = torch.zeros_like(kwargs["global_orient"])
            kwargs["reye_pose"] = torch.zeros_like(kwargs["global_orient"])
        result = self.layer(**kwargs)
        if result.vertices.shape != (len(parameters.person_ids), self.vertex_count, 3):
            raise ValueError("body model returned an unexpected topology")
        if self.family == "smplx":
            if max(self.joint_indices) >= result.joints.shape[1]:
                raise ValueError("SMPL-X model output is missing mapped joints")
            joints = result.joints[:, self.joint_indices]
        else:
            if result.joints.shape[1] < 24:
                raise ValueError("SMPL model output is missing body joints")
            joints = result.joints[:, :24]
        return BodyOutput(result.vertices, joints)


def vertex_parts_for(body_model, person_ids):
    """Return [P,V] in state order; a shared [V] keeps the legacy ABI."""
    _ids(person_ids)
    validator = getattr(body_model, "validate_binding", None)
    if validator is not None:
        validator()
    labels = body_model.vertex_parts
    if not isinstance(labels, Tensor) or labels.dtype != torch.long or labels.numel() == 0 or labels.min() < 0 or labels.max() > 13:
        raise ValueError("vertex parts must be V6 zero-based integer labels")
    if labels.ndim == 1 and labels.shape[0] == body_model.vertex_count:
        return labels[None].expand(len(person_ids), -1)
    canonical = getattr(body_model, "person_ids", None)
    if canonical is None:
        raise ValueError("per-person vertex parts require explicit stable person_ids")
    _ids(canonical)
    if labels.shape != (len(canonical), body_model.vertex_count) or any(pid not in canonical for pid in person_ids):
        raise ValueError("per-person vertex part/person binding differs")
    return labels[[canonical.index(pid) for pid in person_ids]]


class GenderedBodyModelBackend:
    """Matched SMPL layers selected by stable raw person IDs, never batch gender."""
    family, vertex_count = "smpl", 6890

    def __init__(self, models, person_genders, *, provenance=None):
        from .cache import _payload_digest
        self.person_ids = tuple(sorted(person_genders))
        _ids(self.person_ids)
        if any(gender not in {"male", "female"} for gender in person_genders.values()):
            raise ValueError("Hi4D gender must explicitly be male or female")
        self.person_genders = dict(person_genders)
        self.models = dict(models)
        if set(models) != set(person_genders.values()) or any(model.family != "smpl" or model.vertex_count != 6890 for model in models.values()):
            raise ValueError("matched gender models must cover exactly the declared SMPL genders")
        self.faces = next(iter(models.values())).faces.detach().cpu().clone()
        if any(not torch.equal(self.faces, model.faces.cpu()) for model in models.values()):
            raise ValueError("gendered models must share the verified SMPL triangle topology")
        self.vertex_parts = torch.stack([models[person_genders[pid]].vertex_parts for pid in self.person_ids])
        self.provenance = dict(provenance or {"backend": "explicit_gendered_smpl",
            "person_genders": [{"person_id": pid, "gender": person_genders[pid]} for pid in self.person_ids]})
        self._binding = _payload_digest(self._binding_payload())
        self._layer_tensors = self._tensor_versions()

    def _binding_payload(self):
        return {"genders": self.person_genders, "person_ids": self.person_ids, "parts": self.vertex_parts,
                "faces": self.faces, "provenance": self.provenance,
                "models": {g: {"parts": m.vertex_parts, "faces": m.faces, "provenance": m.provenance}
                           for g, m in self.models.items()}}

    def _tensor_versions(self):
        return tuple((g, id(t), t._version) for g, m in self.models.items()
                     for t in (*m.layer.parameters(), *m.layer.buffers()))

    def validate_binding(self):
        from .cache import _payload_digest
        if _payload_digest(self._binding_payload()) != self._binding or self._tensor_versions() != self._layer_tensors:
            raise ValueError("gendered body model/parts/signature binding changed")

    def forward(self, parameters):
        self.validate_binding()
        if parameters.family != "smpl" or any(pid not in self.person_genders for pid in parameters.person_ids):
            raise ValueError("gendered body person/family binding differs")
        outputs = []
        for slot, pid in enumerate(parameters.person_ids):
            single = replace(parameters, person_ids=(pid,), **{f.name: getattr(parameters, f.name)[slot:slot + 1]
                             for f in fields(parameters) if isinstance(getattr(parameters, f.name), Tensor)})
            outputs.append(self.models[self.person_genders[pid]].forward(single))
        return BodyOutput(torch.cat([o.vertices for o in outputs]), torch.cat([o.joints for o in outputs]))


def dwpose134(body: BodyOutput, *, family):
    if family != "smplx" or body.joints.shape[1] != 137:
        raise ValueError("SMPL has only raw24 body joints; DWPose-134 requires mapped SMPL-X137")
    from src.pose_edit.geometry import smplx137_to_dwpose134
    order = smplx137_to_dwpose134(np.arange(137)[:, None].repeat(3, axis=1))[:, 0].astype(np.int64)
    return body.joints[:, order]


@dataclass(frozen=True)
class GeometryState:
    parameters: BodyParameters
    body: BodyOutput

    def copied(self):
        return GeometryState(self.parameters.copied(), self.body.copied())


@dataclass(frozen=True)
class GeometryRequest:
    driver: GeometryState
    requested: GeometryState
    solved: GeometryState
    camera: Camera
    source_rest_heights_m: Tensor
    provenance: dict
    solver_log: dict = field(default_factory=dict)

    @property
    def person_ids(self):
        return self.requested.parameters.person_ids

    def require_exact_gt_pairing(self):
        """Fail closed: source-beta replacement or a solved edit invalidates old RGB GT."""
        for state in (self.requested, self.solved):
            if state.parameters.family != self.driver.parameters.family or state.parameters.person_ids != self.driver.parameters.person_ids or state.parameters.frame != self.driver.parameters.frame:
                raise ValueError("original RGB is not an exact geometry pair")
            for f in fields(self.driver.parameters):
                left, right = getattr(self.driver.parameters, f.name), getattr(state.parameters, f.name)
                if isinstance(left, Tensor) and not torch.equal(left, right):
                    raise ValueError("original RGB is not an exact geometry pair: parameters changed")
            if not torch.equal(self.driver.body.vertices, state.body.vertices) or not torch.equal(self.driver.body.joints, state.body.joints):
                raise ValueError("original RGB is not an exact geometry pair: mesh/joints changed")
        return True


def retarget_request(source_shape: SourceShape, driver_pose: BodyParameters, camera: Camera, body_model: BodyBackend):
    """Re-evaluate source betas under unchanged driver pose and physical camera."""
    if source_shape.family != driver_pose.family or body_model.family != driver_pose.family:
        raise ValueError("source, driver and body model family must match")
    if source_shape.person_ids != driver_pose.person_ids:
        raise ValueError("source and driver person_ids order must match explicitly")
    if driver_pose.frame == "world" and camera.world_to_camera is None:
        raise ValueError("world parameters need camera extrinsics")
    driver = driver_pose.copied()
    requested = replace(driver.copied(), betas=source_shape.betas.detach().to(driver.betas).clone())
    # All rotations, expression and translation are zero for canonical rest stature.
    rest_fields = {f.name: torch.zeros_like(getattr(requested, f.name)) for f in fields(requested)
                   if isinstance(getattr(requested, f.name), Tensor) and f.name != "betas"}
    rest = replace(requested, **rest_fields)
    with torch.no_grad():
        driver_body = body_model.forward(driver).copied()
        requested_body = body_model.forward(requested).copied()
        rest_body = body_model.forward(rest)
        height = rest_body.vertices[..., 1].amax(1) - rest_body.vertices[..., 1].amin(1)
    if (height <= 0).any() or not torch.isfinite(height).all():
        raise ValueError("source zero-pose rest height must be positive")
    requested_state = GeometryState(requested, requested_body)
    return GeometryRequest(GeometryState(driver, driver_body), requested_state, requested_state.copied(),
                           camera.copied(), height.detach().clone(),
                           {"body_model": dict(body_model.provenance), "height": "source_beta_zero_pose_canonical_y_extent_m",
                            "pose_frame": driver.frame, "person_ids": list(driver.person_ids), "geometry_units": "metre"})


@dataclass(frozen=True)
class VertexEndpoint:
    person_id: int
    vertex_index: int

    def __post_init__(self):
        if type(self.person_id) is not int or type(self.vertex_index) is not int or self.person_id < 0 or self.vertex_index < 0:
            raise ValueError("endpoint requires a stable person ID and nonnegative vertex index")


@dataclass(frozen=True)
class VertexContact:
    a: VertexEndpoint
    b: VertexEndpoint
    target_distance_m: float = 0.
    tolerance_m: float = .01
    weight: float = 1.

    def __post_init__(self):
        if self.a == self.b:
            raise ValueError("contact must have two distinct endpoints")
        if not math.isfinite(self.target_distance_m) or self.target_distance_m < 0:
            raise ValueError("contact distance must be finite and nonnegative")
        _positive("contact tolerance", self.tolerance_m)
        _positive("contact weight", self.weight)


@dataclass(frozen=True)
class VertexAnchor:
    endpoint: VertexEndpoint
    target_xyz: Tensor
    tolerance_m: float = .005
    frame: str | None = None

    def __post_init__(self):
        _float_tensor("anchor target_xyz", self.target_xyz, (3,))
        _positive("anchor tolerance", self.tolerance_m)


@dataclass(frozen=True)
class GroundPlane:
    """Only declared feet are constrained: normal dot X + offset_m = 0."""
    normal: tuple[float, float, float]
    offset_m: float
    foot_endpoints: tuple[VertexEndpoint, ...]
    tolerance_m: float = .005
    frame: str | None = None

    def __post_init__(self):
        normal = np.asarray(self.normal, dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1., atol=1e-6):
            raise ValueError("ground plane needs an explicit unit normal")
        if not math.isfinite(self.offset_m) or not self.foot_endpoints:
            raise ValueError("ground plane needs a finite offset and explicit foot endpoints")
        _positive("ground tolerance", self.tolerance_m)


def _endpoint_index(endpoint, request):
    try:
        slot = request.person_ids.index(endpoint.person_id)
    except ValueError as error:
        raise ValueError("endpoint person_id is absent from request; IDs are not slots") from error
    if endpoint.vertex_index >= request.requested.body.vertices.shape[1]:
        raise ValueError("endpoint vertex index is outside body topology")
    return slot, endpoint.vertex_index


def _rotation_matrices(axis_angle):
    # Matrix exponential is well behaved at zero and is only used for final audit.
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*axis_angle.shape[:-1], 3, 3)
    return torch.matrix_exp(skew)


def _rotation_delta_deg(before, after):
    a = _rotation_matrices(before.reshape(-1, 3).double())
    b = _rotation_matrices(after.reshape(-1, 3).double())
    relative = b @ a.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine)).reshape(before.shape[0], -1)


@torch.inference_mode(False)
@torch.enable_grad()
def solve_contacts(request: GeometryRequest, relations, body_model: BodyBackend, *, anchors=(), ground_plane=None,
                   max_iterations=200, learning_rate=.015, max_root_fraction=.05, max_joint_degrees=5.):
    """Projected pose/root optimisation, verified constraints or requested fallback.

    Body rotations and roots are optimised; hands, face, shape, root orientation
    and camera are fixed. Anchors/plane are expressed in the parameter frame.
    This certifies the supplied constraints only, not collision-free bodies.
    With E=0 the requested tensors are returned bitwise unchanged.
    """
    relations, anchors = tuple(relations), tuple(anchors)
    if not relations:
        return replace(request, solved=request.requested.copied(), solver_log={"success": True, "reason": "no_contacts",
            "iterations": 0, "collision_guarantee": "none", "objective_history": []})
    if body_model.family != request.requested.parameters.family:
        raise ValueError("solver body model family mismatch")
    if body_model.provenance != request.provenance.get("body_model") or body_model.vertex_count != request.requested.body.vertices.shape[1]:
        raise ValueError("solver must use the same body model backend that produced requested geometry")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    _positive("learning rate", learning_rate)
    if not 0 < max_root_fraction <= .05 or not 0 < max_joint_degrees <= 5:
        raise ValueError("solver bounds cannot exceed 5% rest height or 5 degrees")
    base = request.requested.parameters.copied()
    endpoints = [( _endpoint_index(r.a, request), _endpoint_index(r.b, request)) for r in relations]
    anchor_endpoints = [_endpoint_index(a.endpoint, request) for a in anchors]
    ground_endpoints = [] if ground_plane is None else [_endpoint_index(e, request) for e in ground_plane.foot_endpoints]
    for constraint in (*anchors, *((ground_plane,) if ground_plane is not None else ())):
        if constraint.frame is not None and constraint.frame != base.frame:
            raise ValueError("anchor/ground frame differs from body parameter frame")
    root = torch.zeros_like(base.transl, requires_grad=True)
    pose = torch.zeros_like(base.body_pose, requires_grad=True)
    limit_m = request.source_rest_heights_m.to(root)[:, None] * max_root_fraction
    limit_rad = math.radians(max_joint_degrees)
    optimiser = torch.optim.Adam([root, pose], lr=learning_rate)
    history, accepted, reason, attempted = [], None, "constraints_not_converged", None

    def residuals(vertices):
        contact = torch.stack([(torch.linalg.vector_norm(vertices[a] - vertices[b]) - r.target_distance_m).abs()
                               for (a, b), r in zip(endpoints, relations)])
        anchor = torch.stack([(vertices[e] - a.target_xyz.to(vertices)).norm() for e, a in zip(anchor_endpoints, anchors)]) if anchors else vertices.new_empty(0)
        normal = None if ground_plane is None else vertices.new_tensor(ground_plane.normal)
        ground = torch.stack([(vertices[e].dot(normal) + ground_plane.offset_m).abs() for e in ground_endpoints]) if ground_endpoints else vertices.new_empty(0)
        return contact, anchor, ground

    def feasible(values):
        contact, anchor, ground = values
        return (all(float(v.detach()) <= r.tolerance_m for v, r in zip(contact, relations))
                and all(float(v.detach()) <= a.tolerance_m for v, a in zip(anchor, anchors))
                and all(float(v.detach()) <= ground_plane.tolerance_m for v in ground))

    # Caller inference/no_grad contexts must not sever input gradients in solving.
    with torch.inference_mode(False), torch.enable_grad():
        for iteration in range(max_iterations + 1):
            candidate = replace(base, body_pose=base.body_pose + pose, transl=base.transl + root)
            body = body_model.forward(candidate)
            values = residuals(body.vertices)
            attempted = tuple(v.detach().cpu().tolist() for v in values)
            terms = [v.square() * r.weight / r.tolerance_m ** 2 for v, r in zip(values[0], relations)]
            terms += [v.square() / a.tolerance_m ** 2 for v, a in zip(values[1], anchors)]
            terms += [v.square() / ground_plane.tolerance_m ** 2 for v in values[2]]
            objective = torch.stack(terms).sum() + 1e-5 * (root.square().sum() + pose.square().sum())
            if not torch.isfinite(objective):
                reason = "nonfinite_objective"
                break
            history.append(float(objective.detach()))
            if feasible(values):
                accepted, reason = GeometryState(candidate.copied(), body.copied()), "constraints_satisfied"
                break
            if iteration == max_iterations:
                break
            optimiser.zero_grad()
            if not objective.requires_grad:
                reason = "body_backend_has_no_input_gradients"
                break
            objective.backward()
            if root.grad is None or pose.grad is None or not torch.isfinite(root.grad).all() or not torch.isfinite(pose.grad).all():
                reason = "missing_or_nonfinite_input_gradients"
                break
            optimiser.step()
            with torch.no_grad():
                root.mul_(torch.clamp(limit_m / root.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1))
                rows = pose.reshape(len(request.person_ids), -1, 3)
                rows.mul_(torch.clamp(limit_rad / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1))

    candidate_state = accepted or request.requested.copied()
    root_norm = (candidate_state.parameters.transl - base.transl).norm(dim=-1)
    angle = _rotation_delta_deg(base.body_pose, candidate_state.parameters.body_pose)
    if accepted and ((root_norm > limit_m[:, 0] + 1e-7).any() or (angle > max_joint_degrees + 1e-5).any()):
        accepted, reason, candidate_state = None, "verified_bounds_exceeded", request.requested.copied()
        root_norm = (candidate_state.parameters.transl - base.transl).norm(dim=-1)
        angle = _rotation_delta_deg(base.body_pose, candidate_state.parameters.body_pose)
    final = residuals(candidate_state.body.vertices)
    log = {"success": accepted is not None, "reason": reason, "iterations": len(history) - 1,
           "objective_history": history, "objective": "sum(weight*(residual_m/tolerance_m)^2)+1e-5*(root_delta^2+body_pose_delta^2)",
           "contact_residual_m": final[0].detach().cpu().tolist(), "anchor_residual_m": final[1].detach().cpu().tolist(),
           "ground_residual_m": final[2].detach().cpu().tolist(), "attempted_residual_m": attempted,
           "root_displacement_m": root_norm.detach().cpu().tolist(), "joint_rotation_delta_deg": angle.amax(-1).cpu().tolist(),
           "contact_tolerance_m": [r.tolerance_m for r in relations], "root_limit_m": limit_m[:, 0].cpu().tolist(),
           "joint_limit_deg": max_joint_degrees, "frame": base.frame, "collision_guarantee": "none",
           "endpoints": [{"a": [r.a.person_id, r.a.vertex_index], "b": [r.b.person_id, r.b.vertex_index], "distance_m": r.target_distance_m} for r in relations]}
    return replace(request, solved=candidate_state, solver_log=log)


def contacts_from_raw_flags(vertices, raw_contact, *, person_ids, tolerance_m=.01, max_pairs=256):
    """Positive annotations indicate membership, never probability or partner index.

    Pair flagged vertices using symmetric nearest neighbours in actual geometry.
    The caller can inspect these explicit pairs before deciding to solve them.
    """
    _ids(person_ids)
    if len(person_ids) != 2 or vertices.ndim != 3 or vertices.shape[0] != 2 or vertices.shape[-1] != 3:
        raise ValueError("raw contact pairing requires two actual meshes")
    _float_tensor("raw contact", raw_contact, vertices.shape[:2])
    if not torch.isfinite(vertices).all() or (raw_contact < 0).any():
        raise ValueError("raw contact and vertices must be finite with nonnegative annotations")
    if type(max_pairs) is not int or max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    index_a, index_b = [(raw_contact[p] > 0).nonzero(as_tuple=True)[0].cpu() for p in range(2)]
    if not len(index_a) or not len(index_b):
        return ()
    from scipy.spatial import cKDTree
    a = vertices[0, index_a].detach().cpu().numpy()
    b = vertices[1, index_b].detach().cpu().numpy()
    pairs = set(zip(index_a.tolist(), index_b[torch.from_numpy(cKDTree(b).query(a)[1])].tolist()))
    pairs.update(zip(index_a[torch.from_numpy(cKDTree(a).query(b)[1])].tolist(), index_b.tolist()))
    ordered = sorted(pairs, key=lambda pair: (float((vertices[0, pair[0]] - vertices[1, pair[1]]).norm()), pair))
    return tuple(VertexContact(VertexEndpoint(person_ids[0], a), VertexEndpoint(person_ids[1], b), tolerance_m=tolerance_m)
                 for a, b in ordered[:max_pairs])


class LiteralChumpyArray:
    """Only the audited Ch.state.x ndarray; no chumpy expression evaluation."""
    def __setstate__(self, state):
        allowed = {"x", "_dirty_vars", "_itr", "_depends_on_deps"}
        if not isinstance(state, dict) or set(state) - allowed or not isinstance(state.get("x"), np.ndarray):
            raise pickle.UnpicklingError("only literal Chumpy x ndarray state is permitted; expressions are rejected")
        if state.get("_dirty_vars", set()) not in (set(), {"x"}) or state.get("_itr") is not None or state.get("_depends_on_deps", {}) != {}:
            raise pickle.UnpicklingError("Chumpy expression operands/metadata are not permitted")
        self._literal = state["x"]

    def array(self):
        value = self._literal
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise pickle.UnpicklingError("literal Chumpy array must be finite numeric data")
        return value


class _LiteralSMPLUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("chumpy.ch", "Ch"):
            return LiteralChumpyArray
        if (module, name) in {("__builtin__", "object"), ("builtins", "object")}:
            return object
        if (module, name) in {("__builtin__", "set"), ("builtins", "set")}:
            return set
        if (module, name) in {("copy_reg", "_reconstructor"), ("copyreg", "_reconstructor")}:
            return self._reconstructor
        if module in {"numpy", "numpy.core.multiarray", "numpy._core.multiarray"} and name in {"ndarray", "dtype", "_reconstruct", "scalar"}:
            return getattr(importlib.import_module(module), name)
        if module in {"scipy.sparse.csc", "scipy.sparse._csc"} and name == "csc_matrix":
            from scipy.sparse import csc_matrix
            return csc_matrix
        raise pickle.UnpicklingError(f"unapproved pickle global {module}.{name}")

    @staticmethod
    def _reconstructor(cls, base, state):
        from scipy.sparse import csc_matrix
        if cls not in (LiteralChumpyArray, csc_matrix) or base is not object or state is not None:
            raise pickle.UnpicklingError("unsupported legacy class reconstruction")
        return object.__new__(cls)


def load_literal_smpl_model(path_or_stream):
    """Read the audited male SMPL pickle without importing/installing chumpy."""
    if hasattr(path_or_stream, "read"):
        value = _LiteralSMPLUnpickler(path_or_stream, encoding="latin1").load()
    else:
        with Path(path_or_stream).open("rb") as stream:
            value = _LiteralSMPLUnpickler(stream, encoding="latin1").load()
    required = {"f": (13776, 3), "J_regressor": (24, 6890), "kintree_table": (2, 24),
                "weights": (6890, 24), "posedirs": (6890, 3, 207), "v_template": (6890, 3)}
    if not isinstance(value, dict):
        raise ValueError("SMPL literal model must be a dictionary")
    result = {}
    for key in (*required, "shapedirs"):
        if key not in value:
            raise ValueError(f"SMPL literal model missing {key}")
        array = value[key]
        if isinstance(array, LiteralChumpyArray):
            if key != "shapedirs":
                raise pickle.UnpicklingError("only top-level shapedirs may be a literal Ch array")
            array = array.array()
        if key == "J_regressor" and hasattr(array, "toarray"):
            array = array.toarray()
        if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"SMPL {key} must be finite numeric data")
        if key in required and array.shape != required[key]:
            raise ValueError(f"unexpected SMPL {key} shape {array.shape}")
        if key == "shapedirs" and (array.ndim != 3 or array.shape[:2] != (6890, 3) or array.shape[2] < 10):
            raise ValueError("SMPL shapedirs must be [6890,3,>=10]")
        result[key] = array
    if result["f"].min() < 0 or result["f"].max() >= 6890 or not np.issubdtype(result["f"].dtype, np.integer):
        raise ValueError("invalid SMPL face indices")
    if (result["weights"] < 0).any() or not np.allclose(result["weights"].sum(1), 1., atol=1e-4):
        raise ValueError("invalid SMPL skinning weights")
    return result


def load_smplestx_backend(smplest_root):
    """CPU neutral layer with the wrapper's actual joint_idx and original weights."""
    root = Path(smplest_root).resolve()
    loader = root / "human_models/human_models.py"
    if not loader.is_file():
        raise FileNotFoundError(loader)
    # Both parent search locations and child loader must bind to the requested
    # checkout. A cached parent otherwise routes imports away from sys.path.
    expected_package = loader.parent

    def validate_parent(paths, package_file=None):
        if paths is None or {Path(p).resolve() for p in paths} != {expected_package}:
            raise RuntimeError("a different human_models parent package binding is already active")
        if package_file is not None and Path(package_file).resolve().parent != expected_package:
            raise RuntimeError("a different human_models parent package file is already active")

    parent = sys.modules.get("human_models")
    if parent is not None:
        validate_parent(getattr(parent, "__path__", None), getattr(parent, "__file__", None))
    existing = sys.modules.get("human_models.human_models")
    if existing is not None and (getattr(existing, "__file__", None) is None or Path(existing.__file__).resolve() != loader):
        raise RuntimeError("a different human_models module is already imported")
    if not sys.path or sys.path[0] != str(root):
        sys.path.insert(0, str(root))
    if parent is None:
        parent_spec = importlib.util.find_spec("human_models")
        validate_parent(None if parent_spec is None else parent_spec.submodule_search_locations,
                        None if parent_spec is None else parent_spec.origin)
    module = importlib.import_module("human_models.human_models")
    if getattr(module, "__file__", None) is None or Path(module.__file__).resolve() != loader:
        raise RuntimeError("imported human_models loader path differs from the requested module binding")
    wrapper = module.SMPLX(str(root / "human_models/human_model_files"))
    layer = wrapper.layer["neutral"].cpu().eval()
    model_path = root / "human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    labels = torch.tensor(_SMPLX_TO_V6)[layer.lbs_weights.detach().cpu().argmax(1)]
    return BodyModelBackend("smplx", layer, layer.faces, joint_indices=tuple(wrapper.joint_idx), vertex_parts=labels,
                            provenance={"backend": "smplestx_neutral_cpu", "model_path": str(model_path), "model_sha256": _sha256(model_path),
                                        "loader_path": str(loader), "loader_sha256": _sha256(loader), "use_pca": False, "use_face_contour": True,
                                        "joint_mapping": list(wrapper.joint_idx), "part_labels": "SMPLX55_LBS_argmax_to_V6_14"})


def _load_hi4d_smpl_gender_backend(model_path, gender):
    from .cache import tensor_sha256
    model_path = Path(model_path).resolve()
    if gender not in {"male", "female"} or model_path.name != f"SMPL_{gender.upper()}.pkl":
        raise ValueError("Hi4D requires the explicitly matched gender model filename")
    model_hash = _sha256(model_path)
    expected = HI4D_MALE_MODEL_SHA256 if gender == "male" else HI4D_FEMALE_MODEL_SHA256
    if model_hash != expected:
        raise ValueError(f"Hi4D {gender} model fingerprint differs from the verified original weights")
    literal = load_literal_smpl_model(model_path)
    import smplx
    from smplx.utils import Struct
    layer = smplx.create(str(model_path.parent), model_type="smpl", gender=gender, batch_size=1,
                         data_struct=Struct(**literal)).cpu().eval()
    body_loader = Path(inspect.getfile(type(layer))).resolve()
    labels = torch.tensor(_SMPL_TO_V6)[layer.lbs_weights.detach().cpu().argmax(1)]
    if tensor_sha256(labels) != HI4D_PARTS_SHA256[gender]:
        raise ValueError("Hi4D matched gender model LBS/parts fingerprint differs")
    return BodyModelBackend("smpl", layer, literal["f"], vertex_parts=labels,
        provenance={"backend": f"hi4d_matched_{gender}_cpu", "model_path": str(model_path), "model_sha256": model_hash,
                    "loader": "restricted_literal_chumpy_x_v1", "loader_path": str(Path(__file__).resolve()), "loader_sha256": _sha256(__file__),
                    "smplx_loader_path": str(Path(smplx.__file__).resolve()), "smplx_loader_sha256": _sha256(smplx.__file__),
                    "smplx_body_loader_path": str(body_loader), "smplx_body_loader_sha256": _sha256(body_loader),
                    "joint_mapping": "raw_smpl24_only", "part_labels": f"matched_{gender}_SMPL24_LBS_argmax_to_V6_14"})


def load_hi4d_smpl_backend(model_path):
    """Legacy explicit male-only API; no neutral/female fallback."""
    return _load_hi4d_smpl_gender_backend(model_path, "male")


def _hi4d_genders(values):
    values = np.asarray(values)
    if values.shape != (2,) or values.dtype.kind not in "US":
        raise ValueError("Hi4D genders must be two explicit string declarations in raw A/B order")
    result = []
    for value in values:
        name = (value.decode("utf-8") if isinstance(value, bytes) else str(value)).lower()
        if name not in {"male", "female"}:
            raise ValueError("Hi4D gender must explicitly be male or female; missing/unknown is not inferred")
        result.append(name)
    return result


def load_hi4d_gender_metadata(metadata_path, parameters_path, *, person_order=(0, 1), person_ids=None, expected_pair=None):
    """Read own-action (or pair-level) meta.npz; raw A/B retain stable IDs."""
    metadata, parameters = Path(metadata_path).resolve(), Path(parameters_path).resolve()
    pair, action = parameters.parents[2], parameters.parents[1]
    if parameters.parent.name != "smpl" or not pair.name.startswith("pair") or metadata.name != "meta.npz" or metadata.parent not in {pair, action}:
        raise ValueError("Hi4D gender metadata must belong to the parameters' own pair/action")
    if expected_pair is not None and pair.name != expected_pair:
        raise ValueError("Hi4D gender metadata pair differs from the declared source group")
    person_ids = tuple(person_order) if person_ids is None else person_ids
    _ids(person_ids)
    if len(person_ids) != 2 or sorted(person_order) != [0, 1]:
        raise ValueError("Hi4D gender metadata needs the explicit raw A/B permutation")
    with np.load(metadata, allow_pickle=False) as raw:
        if "genders" not in raw:
            raise ValueError("Hi4D gender metadata is missing genders")
        genders = _hi4d_genders(raw["genders"])
    binding = {"path": str(metadata), "sha256": _sha256(metadata), "pair": pair.name,
               "parameters_path": str(parameters), "raw_genders": genders,
               "parameters_sha256": _sha256(parameters),
               "person_bindings": [{"person_id": pid, "raw_person_index": raw, "gender": genders[raw]}
                                   for pid, raw in zip(person_ids, person_order)]}
    return {pid: genders[raw] for pid, raw in zip(person_ids, person_order)}, binding


def load_hi4d_gendered_backend(body_model_root, gender_metadata, *, parameters_path,
                               person_order=(0, 1), person_ids=None,
                               driver_gender_metadata=None, driver_parameters_path=None, expected_pair=None):
    """Explicit matched male/female CPU layers, with source/driver gender agreement."""
    from .cache import tensor_sha256
    genders, source = load_hi4d_gender_metadata(gender_metadata, parameters_path,
                                               person_order=person_order, person_ids=person_ids, expected_pair=expected_pair)
    bindings = {"source": source}
    if driver_gender_metadata is not None or driver_parameters_path is not None:
        if driver_gender_metadata is None or driver_parameters_path is None:
            raise ValueError("driver gender metadata and parameters must both be explicit")
        driver_genders, driver = load_hi4d_gender_metadata(driver_gender_metadata, driver_parameters_path,
                                                         person_order=person_order, person_ids=person_ids)
        if genders != driver_genders or source["pair"] != driver["pair"]:
            raise ValueError("source/driver stable person gender or pair declarations differ")
        bindings["driver"] = driver
    root = Path(body_model_root).resolve()
    models = {gender: _load_hi4d_smpl_gender_backend(root / "smpl" / f"SMPL_{gender.upper()}.pkl", gender)
              for gender in sorted(set(genders.values()), reverse=True)}
    provenance = {"backend": "hi4d_matched_gendered_cpu_v1", "gender_metadata": bindings,
        "person_genders": sorted(source["person_bindings"], key=lambda b: b["person_id"]),
        "models": {g: {**m.provenance, "vertex_parts_sha256": tensor_sha256(m.vertex_parts)} for g, m in models.items()},
        "part_labels": "matched_per_person_SMPL24_LBS_argmax_to_V6_14", "joint_mapping": "raw_smpl24_only"}
    return GenderedBodyModelBackend(models, genders, provenance=provenance)


def _frame_record(frame_map, selector):
    records = frame_map.get("frames")
    if not isinstance(records, list):
        raise ValueError("frame_map.frames must be the actual indexed list")
    matches = []
    for position, record in enumerate(records):
        aliases = [record.get(k) for k in ("stem", "original_name", "relative_image", "rgb_1024", "staged_name")]
        if selector in aliases:
            index = record.get("smplestx_index", record.get("index", position))
            if type(index) is not int or index < 0:
                raise ValueError("frame_map tensor index is invalid")
            matches.append((index, record))
    if len(matches) != 1:
        raise ValueError(f"frame selector {selector!r} must match exactly one frame_map record")
    return matches[0]


def load_smplestx_parameters(params_path, frame_map_path, *, source_frame, driver_frame, dataset, person_id=0):
    """Read tensor indices from frame_map; retargeted iPER always uses raw camera."""
    if dataset not in {"iper", "tiktok"}:
        raise ValueError("SMPLest-X dataset must be iper or tiktok")
    params = torch.load(params_path, map_location="cpu", weights_only=True)
    frame_map = json.loads(Path(frame_map_path).read_text(encoding="utf-8"))
    si, _ = _frame_record(frame_map, source_frame)
    di, record = _frame_record(frame_map, driver_frame)
    raw = dataset == "iper" and "retargeted_cam" in params and bool(torch.as_tensor(params["retargeted_cam"]).any())
    suffix = "_raw" if raw else ""
    translation, focal, principal = "transl" + suffix, "focal_length_xy" + suffix, "princpt" + suffix

    def tensor(key, index):
        if key not in params or not isinstance(params[key], Tensor) or params[key].ndim < 1 or index >= len(params[key]):
            raise ValueError(f"missing or invalid indexed SMPLest-X parameter {key}")
        return params[key][index:index + 1].detach().float().clone()

    pose = {name: tensor(name, di) for name in ("betas", "body_pose", "global_orient", "left_hand_pose", "right_hand_pose", "jaw_pose", "expression")}
    driver = BodyParameters("smplx", (person_id,), transl=tensor(translation, di), frame="camera_cv", **pose)
    shape = SourceShape("smplx", (person_id,), tensor("betas", si))
    fx, fy = tensor(focal, di).flatten().tolist()
    cx, cy = tensor(principal, di).flatten().tolist()
    width = int(record["width"] if "width" in record else tensor("width", di).item())
    height = int(record["height"] if "height" in record else tensor("height", di).item())
    cam = Camera(torch.tensor([[fx, 0, cx], [0, fy, cy], [0., 0., 1.]]), (height, width))
    return shape, driver, cam, {"params_path": str(params_path), "params_sha256": _sha256(params_path),
        "frame_map_path": str(frame_map_path), "frame_map_sha256": _sha256(frame_map_path), "source_tensor_index": si,
        "driver_tensor_index": di, "camera_keys": [translation, focal, principal], "dataset": dataset}


def load_hi4d_parameters(source_path, driver_path, *, person_ids=None, person_order=(0, 1)):
    """person_order selects raw A/B; custom person_ids, if supplied, are in output slot order.

    Without custom IDs the stable IDs are the raw A/B indices themselves, so
    ``person_order=(1,0)`` produces ``person_ids=(1,0)`` and keeps B identified as B.
    """
    person_ids = tuple(person_order) if person_ids is None else person_ids
    _ids(person_ids)
    if len(person_ids) != 2 or sorted(person_order) != [0, 1]:
        raise ValueError("Hi4D requires two people and an explicit permutation of raw A/B")
    with np.load(source_path, allow_pickle=False) as source, np.load(driver_path, allow_pickle=False) as driver:
        def tensor(payload, key):
            if key not in payload:
                raise ValueError(f"missing Hi4D {key}")
            return torch.from_numpy(np.asarray(payload[key])[list(person_order)].copy()).float()
        shape = SourceShape("smpl", person_ids, tensor(source, "betas"))
        params = BodyParameters("smpl", person_ids, frame="world", **{k: tensor(driver, k) for k in ("betas", "body_pose", "global_orient", "transl")})
        flags = tensor(driver, "contact")
        _float_tensor("Hi4D raw contact", flags, (2, 6890))
        if (flags < 0).any():
            raise ValueError("Hi4D raw contact must contain nonnegative annotation values")
    return shape, params, flags, {"source_params_path": str(source_path), "source_params_sha256": _sha256(source_path),
        "driver_params_path": str(driver_path), "driver_params_sha256": _sha256(driver_path), "person_order": list(person_order),
        "joint_availability": "raw24_body_only; fingers/face_unavailable", "contact_annotation": "positive_membership_not_probability_or_partner_index"}


def camera_to_clip(vertices_cv, camera, canvas, *, near, far):
    _positive("near plane", near)
    _positive("far plane", far)
    if far <= near or canvas.input_hw != camera.image_hw:
        raise ValueError("invalid clip planes or canvas camera dimensions")
    h, w = canvas.output_hw
    intrinsic = canvas.matrix.to(vertices_cv) @ camera.K.to(vertices_cv)
    image_homogeneous = vertices_cv @ intrinsic.T
    z = vertices_cv[..., 2]
    return torch.stack([2 * image_homogeneous[..., 0] / w - z,
                        z - 2 * image_homogeneous[..., 1] / h,
                        (far + near) / (far - near) * z - 2 * far * near / (far - near), z], -1)


@dataclass(frozen=True)
class DenseGeometry:
    normal_cv: Tensor             # [P,H,W,3], unit camera-space vectors, zero background
    depth_m: Tensor               # [P,H,W], metric camera Z
    mask: Tensor                  # [P,H,W], per-person isolated triangle coverage
    visibility: Tensor            # [P,H,W], visible person in composed scene
    part: Tensor                  # [P,H,W], 0 background, 1..14 V6 body parts
    contact: Tensor               # [P,H,W], projected explicit endpoint support
    visible_owner: Tensor         # [H,W], 0 background, slot + 1
    person_ids: tuple[int, ...]
    metadata: dict

    def typed_spatial(self, *, target_hw=None):
        """Pro codec: scene-shared near0/far1; part -1 background/0..13.

        Optional target_hw is the latent grid. Geometry already occupies the
        output pixel canvas, so this only reduces the grid, without letterboxing
        again. Area/nearest and foreground weighting match Pro preparation.
        Raw metres, CV normals and PNG-style part IDs remain on this object.
        """
        hw = self.depth_m.shape[-2:] if target_hw is None else target_hw
        _hw(tuple(hw))
        def reduce(value, *, labels=False):
            if value.ndim == 3:
                value = value[:, None]
            return F.interpolate(value.float(), size=hw, mode="nearest" if labels else "area")
        support = reduce(self.mask).clamp(0, 1)
        values = self.depth_m[self.mask]
        bounds = (values.min(), values.max()) if values.numel() else (self.depth_m.new_tensor(0), self.depth_m.new_tensor(0))
        normalized_depth = torch.where(self.mask, (self.depth_m - bounds[0]) / (bounds[1] - bounds[0]).clamp_min(1e-6), 0)
        labels = reduce(self.part - 1, labels=True).long()
        labels[support <= 0] = -1
        return {"normal": (F.normalize(reduce(self.normal_cv.permute(0, 3, 1, 2)), dim=1, eps=1e-8) * support)[None],
                "depth": (reduce(normalized_depth) * support)[None], "mask": support[None],
                "visibility": torch.minimum(reduce(self.visibility).clamp(0, 1), support)[None],
                "part": labels[None], "contact": (reduce(self.contact) * support)[None]}


def compose_dense_geometry(normal_cv, depth_m, mask, part, contact, *, person_ids, camera, canvas):
    _ids(person_ids)
    if camera.image_hw != canvas.input_hw:
        raise ValueError("dense canvas input dimensions must match the physical camera")
    h, w = canvas.output_hw
    shape = (len(person_ids), h, w)
    _float_tensor("dense CV normals", normal_cv, (*shape, 3))
    _float_tensor("dense metric depth", depth_m, shape)
    _float_tensor("dense contact", contact, shape)
    if tuple(mask.shape) != shape or mask.dtype != torch.bool or tuple(part.shape) != shape or part.dtype != torch.long:
        raise ValueError("dense mask/part must be bool/long [P,H,W]")
    if (mask & ~canvas.valid_fov_mask(device=mask.device)[None]).any():
        raise ValueError("dense foreground extends outside the original physical camera FoV")
    if (depth_m[mask] <= 0).any() or (depth_m[~mask] != 0).any() or (part[~mask] != 0).any() or (part[mask] < 1).any() or (part > 14).any():
        raise ValueError("dense depth/part values are inconsistent with triangle coverage")
    if (contact < 0).any() or (contact > 1).any() or (contact[~mask] != 0).any():
        raise ValueError("dense endpoint support must be in [0,1] and inside triangle coverage")
    if (normal_cv[~mask] != 0).any() or not torch.allclose(normal_cv[mask].norm(dim=-1), torch.ones_like(depth_m[mask]), atol=1e-4, rtol=0):
        raise ValueError("dense normals must be unit CV vectors on foreground and zero background")
    nearest, slot = depth_m.masked_fill(~mask, float("inf")).min(0)
    owner = torch.where(torch.isfinite(nearest), slot + 1, torch.zeros_like(slot))
    visibility = owner[None] == torch.arange(1, len(person_ids) + 1, device=owner.device)[:, None, None]
    return DenseGeometry(normal_cv, depth_m, mask, visibility, part, contact, owner, person_ids,
                         {"normal_basis": "camera_cv_xyz", "depth_units": "metre", "depth_axis": "camera_positive_z",
                          "part_encoding": "0=background; 1..14=V6_body_parts", "owner_encoding": "0=background; slot+1=visible_person",
                          "person_ids": list(person_ids), "pixel_canvas_transform": canvas.to_dict(),
                          "physical_camera_K": camera.K.cpu().tolist(),
                          "physical_world_to_camera": None if camera.world_to_camera is None else camera.world_to_camera.cpu().tolist(),
                          "row_order": "top_down", "physical_fov_clipping": "inverse_canvas_pixel_centres_inside_[0,W)x[0,H)",
                          "contact_encoding": "interpolated_explicit_vertex_endpoint_support",
                          "typed_codec": {"depth": "shared_scene_minmax_near0_far1;masked_background0", "part": "-1=background;0..13=V6_body_parts"},
                          "typed_depth_range_m": None if not mask.any() else [float(depth_m[mask].min()), float(depth_m[mask].max())]})


class NVDiffDenseRenderer:
    """Real triangle rasterization, one bottom-up -> top-down row flip.

    CV normals are computed directly from the CV mesh. There is no RGB normal
    encoding/sign guess and no point splatting. Output contacts are endpoint
    support on triangles, not a claim of all-body contact/collision detection.
    """
    def __init__(self, *, device="cuda:0"):
        import nvdiffrast.torch as dr
        self.dr, self.device = dr, torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("nvdiffrast dense renderer requires CUDA")
        self.context = dr.RasterizeCudaContext(device=self.device)

    @torch.no_grad()
    def render(self, state: GeometryState, camera: Camera, body_model: BodyBackend, *, canvas=None, contacts=()):
        if body_model.family != state.parameters.family:
            raise ValueError("renderer body family mismatch")
        canvas = canvas or PixelCanvasTransform.letterbox(camera.image_hw, camera.image_hw)
        valid_fov = canvas.valid_fov_mask(device=self.device)
        vertices = camera.to_camera(state.body.vertices, frame=state.parameters.frame).to(self.device, torch.float32)
        if (vertices[..., 2] <= 0).any():
            raise ValueError("renderer requires meshes in front of the physical camera")
        faces = body_model.faces.to(self.device, torch.int32).contiguous()
        labels = vertex_parts_for(body_model, state.parameters.person_ids).to(self.device)
        triangle_labels = F.one_hot(labels[:, faces.long()], 14).sum(2).argmax(-1) + 1
        support = torch.zeros(vertices.shape[:2], device=self.device)
        ids = state.parameters.person_ids
        for contact in contacts:
            for endpoint in (contact.a, contact.b):
                if endpoint.person_id not in ids or endpoint.vertex_index >= vertices.shape[1]:
                    raise ValueError("render contact endpoint is outside body/person topology")
                support[ids.index(endpoint.person_id), endpoint.vertex_index] = 1
        normals, depths, masks, parts, projected_contacts = [], [], [], [], []
        for slot, verts in enumerate(vertices):
            triangle = verts[faces.long()]
            face_normal = torch.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0], dim=-1)
            vertex_normal = torch.zeros_like(verts)
            for corner in range(3):
                vertex_normal.index_add_(0, faces[:, corner].long(), face_normal)
            vertex_normal = F.normalize(vertex_normal, dim=-1)
            near = max(1e-4, float(verts[:, 2].min()) * .5)
            far = max(near + 1, float(verts[:, 2].max()) * 1.5)
            clip = camera_to_clip(verts, camera, canvas, near=near, far=far)[None].contiguous()
            raster, _ = self.dr.rasterize(self.context, clip, faces, resolution=canvas.output_hw, grad_db=False)
            attributes = torch.cat([vertex_normal, verts[:, 2:3], support[slot, :, None]], -1)[None].contiguous()
            interpolated, _ = self.dr.interpolate(attributes, raster, faces)
            # nvdiffrast rows start at the bottom. Flip raster and attributes once.
            raster = raster[0].flip(0)
            interpolated = interpolated[0].flip(0)
            mask = (raster[..., 3] > 0) & valid_fov
            face_index = (raster[..., 3].long() - 1).clamp_min(0)
            normals.append(F.normalize(interpolated[..., :3], dim=-1) * mask[..., None])
            depths.append(interpolated[..., 3] * mask)
            masks.append(mask)
            parts.append(torch.where(mask, triangle_labels[slot, face_index], torch.zeros_like(face_index)))
            projected_contacts.append(interpolated[..., 4].clamp(0, 1) * mask)
        result = compose_dense_geometry(torch.stack(normals), torch.stack(depths), torch.stack(masks), torch.stack(parts),
                                        torch.stack(projected_contacts), person_ids=ids, camera=camera, canvas=canvas)
        result.metadata.update({"renderer": "nvdiffrast_dense_triangle_v1", "row_flip_count": 1,
                                "normal_estimation": "area_weighted_CV_mesh_vertex_normals",
                                "part_labels": body_model.provenance.get("part_labels", "explicit_backend_vertex_parts")})
        return result


def projected_part_rois(state, camera, body_model, *, canvas=None, margin_px=0.):
    """Source mesh face/hand bounds only; SMPL hand bounds are not 21 finger joints."""
    if margin_px < 0 or not math.isfinite(margin_px):
        raise ValueError("ROI margin must be finite and nonnegative")
    xy = camera.project(state.body.vertices, frame=state.parameters.frame, canvas=canvas)
    labels = vertex_parts_for(body_model, state.parameters.person_ids).to(xy.device)
    output = []
    for slot, person in enumerate(xy):
        bounds = {}
        for name, part in (("face", 0), ("left_hand", 4), ("right_hand", 7)):
            points = person[(labels[slot] == part) & torch.isfinite(person).all(-1)]
            bounds[name] = None if not len(points) else torch.cat([points.amin(0) - margin_px, points.amax(0) + margin_px]).detach().cpu().tolist()
        output.append({"mesh_part_bounds_xyxy": bounds, "landmark_availability": "SMPLX137" if state.parameters.family == "smplx" else "SMPL24_body_only_no_fingers",
                       "bounds_provenance": "projected_body_mesh_parts; head_bounds_are_not_face_landmarks"})
    return output
