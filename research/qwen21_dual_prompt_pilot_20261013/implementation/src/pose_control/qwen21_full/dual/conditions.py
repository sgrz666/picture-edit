"""Immutable, versioned source/target geometry boundary for dual_v8.

The generation input deliberately cannot carry target photographs or dataset rows.
Small images are accepted only through the explicit construction-only fixture flag.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass, fields
import math
from typing import ClassVar

import torch

from ..conditions import _to_preserving_discrete_dtype

UIDs = tuple[tuple[str | None, str | None], ...]


def _tensor(name, value, shape, *, dtype=None):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if dtype is not None:
        if value.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
    elif not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _range(name, value, low=0., high=1.):
    if value.numel() and ((value < low).any() or (value > high).any()):
        raise ValueError(f"{name} must be in [{low}, {high}]")


def _resolution(height, width, allow_test_resolution):
    if (height, width) == (512, 512):
        return
    if not allow_test_resolution or height != width or height < 16 or height % 16:
        raise ValueError("production resolution must be 512x512; explicit fixtures must be square and divisible by 16")


def _uids(uids, batch):
    if not isinstance(uids, tuple) or len(uids) != batch:
        raise ValueError("person_uids must contain one tuple per batch row")
    for row in uids:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError("person_uids rows must be (person_A, person_B) tuples")
        values = [uid for uid in row if uid is not None]
        if any(not isinstance(uid, str) or not uid.strip() for uid in values):
            raise ValueError("person UIDs must be nonempty strings or None")
        if len(values) != len(set(values)):
            raise ValueError("valid person UIDs must be unique within each row")


class _Transfer:
    def to(self, *args, **kwargs):
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                if isinstance(self,PreparedDualState) and field.name.startswith('weights_'):
                    # These are probabilities with a conservation contract. BF16
                    # rounding can turn a valid sum of one into 1.0078125.
                    destination = torch.empty(0,device=value.device).to(*args,**kwargs)
                    value = value.to(device=destination.device,dtype=torch.float32)
                else:
                    value = _to_preserving_discrete_dtype(value, *args, **kwargs)
            elif isinstance(value, _Transfer):
                value = value.to(*args, **kwargs)
            values[field.name] = value
        if isinstance(self, (SourceCondition, GeometryCondition)):
            values["allow_test_resolution"] = self._allow_test_resolution
        return type(self)(**values)

    def _same_device(self):
        devices = {getattr(self, item.name).device for item in fields(self)
                   if isinstance(getattr(self, item.name), torch.Tensor)}
        if len(devices) > 1:
            raise ValueError("all condition tensors must share a device")


@dataclass(frozen=True)
class SourceCondition(_Transfer):
    rgb: torch.Tensor
    masks: torch.Tensor
    person_uids: UIDs
    person_latents: torch.Tensor
    latent_masks: torch.Tensor
    dino_patches: torch.Tensor
    dino_valid: torch.Tensor
    version: str = "source_v1"
    allow_test_resolution: InitVar[bool] = False
    _allow_test_resolution: ClassVar[bool] = False

    def __post_init__(self, allow_test_resolution):
        object.__setattr__(self, "_allow_test_resolution", bool(allow_test_resolution))
        self.validate()

    @property
    def person_valid(self):
        return torch.tensor([[uid is not None for uid in row] for row in self.person_uids],
                            dtype=torch.bool, device=self.rgb.device)

    def validate(self):
        if self.rgb.ndim != 4:
            raise ValueError("rgb must have shape [B,3,H,W]")
        batch, _, height, width = self.rgb.shape
        if batch == 0:
            raise ValueError("batch cannot be empty")
        _resolution(height, width, self._allow_test_resolution)
        _uids(self.person_uids, batch)
        _tensor("rgb", self.rgb, (batch, 3, height, width))
        _tensor("masks", self.masks, (batch, 2, 1, height, width))
        if self.person_latents.ndim != 5:
            raise ValueError("person_latents must have shape [B,2,64,h,w]")
        latent_hw = self.person_latents.shape[-2:]
        if min(latent_hw) < 1:
            raise ValueError("latent spatial dimensions must be positive")
        _tensor("person_latents", self.person_latents, (batch, 2, 64, *latent_hw))
        _tensor("latent_masks", self.latent_masks, (batch, 2, 1, *latent_hw))
        if self.dino_patches.ndim != 4:
            raise ValueError("dino_patches must have shape [B,2,M,1536]")
        count = self.dino_patches.shape[2]
        _tensor("dino_patches", self.dino_patches, (batch, 2, count, 1536))
        _tensor("dino_valid", self.dino_valid, (batch, 2, count), dtype=torch.bool)
        for name in ("rgb", "masks", "latent_masks"):
            _range(name, getattr(self, name))
        valid = self.person_valid
        if (~self.masks.flatten(2).any(-1) & valid).any():
            raise ValueError("valid source person has an empty mask")
        if (~self.latent_masks.flatten(2).any(-1) & valid).any():
            raise ValueError("valid source person has an empty latent mask")
        for name in ("masks", "person_latents", "latent_masks", "dino_patches", "dino_valid"):
            if getattr(self, name)[~valid].count_nonzero():
                raise ValueError(f"invalid source person must have zero {name}")
        if self.dino_patches[~self.dino_valid].count_nonzero():
            raise ValueError("invalid DINO patches must be zero")
        if self.version != "source_v1":
            raise ValueError("unsupported source version")
        self._same_device()
        return self


@dataclass(frozen=True)
class GeometryCondition(_Transfer):
    normal: torch.Tensor
    depth: torch.Tensor
    part: torch.Tensor
    pose: torch.Tensor
    amodal_mask: torch.Tensor
    visible_owner: torch.Tensor
    global32: torch.Tensor
    relative14: torch.Tensor
    part_positions: torch.Tensor
    part_orientations: torch.Tensor
    part_joints: torch.Tensor
    part_valid: torch.Tensor
    part_confidence: torch.Tensor
    contact_endpoints: torch.Tensor
    contact_features: torch.Tensor
    contact_valid: torch.Tensor
    contact_map: torch.Tensor
    person_valid: torch.Tensor
    person_uids: UIDs
    version: str = "smplx_dual_v1"
    geometry_hash: str = ""
    allow_test_resolution: InitVar[bool] = False
    _allow_test_resolution: ClassVar[bool] = False

    def __post_init__(self, allow_test_resolution):
        object.__setattr__(self, "_allow_test_resolution", bool(allow_test_resolution))
        self.validate()

    def validate(self):
        if self.normal.ndim != 5:
            raise ValueError("normal must have shape [B,2,3,H,W]")
        batch, _, _, height, width = self.normal.shape
        if batch == 0:
            raise ValueError("batch cannot be empty")
        _resolution(height, width, self._allow_test_resolution)
        specs = {"normal":(batch,2,3,height,width), "depth":(batch,2,1,height,width),
                 "part":(batch,2,14,height,width), "pose":(batch,2,25,height,width),
                 "amodal_mask":(batch,2,1,height,width), "visible_owner":(batch,2,1,height,width),
                 "global32":(batch,2,32), "relative14":(batch,2,14),
                 "part_positions":(batch,2,14,3), "part_orientations":(batch,2,14,3),
                 "part_joints":(batch,2,14,3), "part_confidence":(batch,2,14),
                 "contact_map":(batch,1,height,width)}
        for name, shape in specs.items():
            _tensor(name, getattr(self, name), shape)
        _tensor("person_valid", self.person_valid, (batch,2), dtype=torch.bool)
        _tensor("part_valid", self.part_valid, (batch,2,14), dtype=torch.bool)
        if self.contact_endpoints.ndim != 3:
            raise ValueError("contact_endpoints must have shape [B,E,2]")
        count = self.contact_endpoints.shape[1]
        if count > 8:
            raise ValueError("at most 8 cross-person contact edges are supported")
        _tensor("contact_endpoints", self.contact_endpoints, (batch,count,2), dtype=torch.long)
        _tensor("contact_features", self.contact_features, (batch,count,15))
        _tensor("contact_valid", self.contact_valid, (batch,count), dtype=torch.bool)
        _uids(self.person_uids, batch)
        uid_valid = torch.tensor([[uid is not None for uid in row] for row in self.person_uids],
                                 dtype=torch.bool, device=self.person_valid.device)
        if not torch.equal(uid_valid, self.person_valid) or not self.person_valid.any(-1).all():
            raise ValueError("geometry UIDs must match valid persons; each scene needs a valid person")
        _range("normal", self.normal, -1, 1)
        for name in ("amodal_mask", "visible_owner", "part", "pose", "part_confidence", "contact_map"):
            _range(name, getattr(self, name))
        if (self.visible_owner > self.amodal_mask+1e-6).any() or (self.visible_owner.sum(1) > 1+1e-6).any():
            raise ValueError("visible_owner must be disjoint ownership inside amodal masks")
        if (self.part.sum(2, keepdim=True) > self.amodal_mask+1e-6).any():
            raise ValueError("part membership must be one-hot inside amodal masks")
        if (~self.amodal_mask.flatten(2).any(-1) & self.person_valid).any():
            raise ValueError("valid geometry person has an empty amodal mask")
        if (self.depth[self.amodal_mask > 0] <= 0).any():
            raise ValueError("masked depth must be positive metric camera-z")
        person_fields = ("normal", "depth", "part", "pose", "amodal_mask", "visible_owner", "global32", "relative14", "part_positions", "part_orientations", "part_joints", "part_valid", "part_confidence")
        for name in person_fields:
            if getattr(self, name)[~self.person_valid].count_nonzero():
                raise ValueError(f"invalid geometry person must have zero {name}")
        if self.relative14[~self.person_valid.all(-1)].count_nonzero():
            raise ValueError("relative geometry must be zero for single-person rows")
        if (self.part_valid & ~self.person_valid[..., None]).any():
            raise ValueError("invalid persons cannot have valid parts")
        for name in ('part_positions','part_orientations','part_joints','part_confidence'):
            if getattr(self,name)[~self.part_valid].count_nonzero():
                raise ValueError(f"invalid parts must have zero {name}")
        lengths = self.part_orientations.float().norm(dim=-1)[self.part_valid]
        if ((lengths > .01) & ((lengths-1).abs() > .01)).any():
            raise ValueError("part orientations must be world unit directions or zero when unavailable")
        endpoints = self.contact_endpoints[self.contact_valid]
        if endpoints.numel() and ((endpoints < 0).any() or (endpoints >= 28).any()):
            raise ValueError("contact endpoints must be flattened part indices in [0,27]")
        if endpoints.numel() and (endpoints[:, 0]//14 == endpoints[:, 1]//14).any():
            raise ValueError("contact endpoints must be cross-person")
        if count:
            endpoint_valid = self.part_valid.flatten(1).gather(1, self.contact_endpoints.clamp(0,27).flatten(1)).reshape(batch,count,2)
            if (self.contact_valid & ~endpoint_valid.all(-1)).any():
                raise ValueError("contact endpoints must refer to valid parts")
        if self.contact_features[~self.contact_valid].count_nonzero():
            raise ValueError("invalid contacts must have zero features")
        if self.contact_map[~self.contact_valid.any(-1)].count_nonzero():
            raise ValueError("empty contact rows must have a zero contact map")
        if self.version != "smplx_dual_v1" or not isinstance(self.geometry_hash, str) or not self.geometry_hash:
            raise ValueError("geometry requires smplx_dual_v1 and a nonempty geometry_hash")
        self._same_device()
        return self


@dataclass(frozen=True)
class DualGenerationInput(_Transfer):
    source: SourceCondition
    geometry: GeometryCondition
    prompts: tuple[str, ...]
    seeds: tuple[int, ...]

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.source, SourceCondition) or not isinstance(self.geometry, GeometryCondition):
            raise ValueError("generation requires validated source and geometry conditions")
        self.source.validate()
        self.geometry.validate()
        batch = self.source.rgb.shape[0]
        if self.geometry.normal.shape[0] != batch or self.source.person_uids != self.geometry.person_uids:
            raise ValueError("source and geometry person UIDs must match slot by slot")
        if self.source.rgb.device != self.geometry.normal.device:
            raise ValueError("source and geometry must share a device")
        if not isinstance(self.prompts, tuple) or len(self.prompts) != batch or any(not isinstance(s,str) for s in self.prompts):
            raise ValueError("prompts must be one string per batch row")
        if not isinstance(self.seeds, tuple) or len(self.seeds) != batch or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integer tuples, one per row")
        return self


@dataclass(frozen=True)
class ControlStrengths:
    geometry: float = 1.
    appearance: float = 1.
    contact: float = 1.
    max_ratio: float | None = None

    def __post_init__(self):
        for name in ("geometry", "appearance", "contact"):
            value = getattr(self,name)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} strength must be finite and nonnegative")
        if self.max_ratio is not None and (not math.isfinite(self.max_ratio) or self.max_ratio <= 0):
            raise ValueError("max_ratio must be positive and finite")

    @property
    def disabled(self):
        return self.geometry == self.appearance == self.contact == 0

    def to(self, *args, **kwargs):
        return self


@dataclass(frozen=True)
class PreparedDualState(_Transfer):
    dense: torch.Tensor
    nodes: torch.Tensor
    node_valid: torch.Tensor
    appearance: torch.Tensor
    appearance_valid: torch.Tensor
    contacts: torch.Tensor
    contact_valid: torch.Tensor
    weights_geometry: torch.Tensor
    weights_appearance: torch.Tensor
    weights_contact: torch.Tensor
    person_valid: torch.Tensor
    target_hw: tuple[int, int]

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.dense.ndim != 4:
            raise ValueError("dense must have shape [B,2,N,d]")
        batch, _, count, width = self.dense.shape
        if not isinstance(self.target_hw, tuple) or len(self.target_hw) != 2 or min(self.target_hw) < 1 or math.prod(self.target_hw) != count:
            raise ValueError("target_hw must match the dense token count")
        specs = {"dense":(batch,2,count,width), "nodes":(batch,2,14,width),
                 "appearance":(batch,2,16,width), "contacts":(batch,self.contacts.shape[1],width),
                 "weights_geometry":(batch,2,count,1), "weights_appearance":(batch,2,count,1),
                 "weights_contact":(batch,count,1)}
        for name, shape in specs.items():
            _tensor(name,getattr(self,name),shape)
        for name, shape in {"node_valid":(batch,2,14), "appearance_valid":(batch,2,16),
                            "contact_valid":(batch,self.contacts.shape[1]), "person_valid":(batch,2)}.items():
            _tensor(name,getattr(self,name),shape,dtype=torch.bool)
        for name in ("weights_geometry", "weights_appearance", "weights_contact"):
            _range(name, getattr(self,name))
        if (self.weights_geometry.sum(1)+self.weights_contact > 1+1e-5).any():
            raise ValueError("geometry/contact routing exceeds one")
        if (self.weights_appearance.sum(1) > 1+1e-5).any():
            raise ValueError("appearance routing exceeds one")
        for name in ("dense","nodes","appearance","weights_geometry","weights_appearance"):
            if getattr(self,name)[~self.person_valid].count_nonzero():
                raise ValueError(f"invalid prepared person must have zero {name}")
        if self.nodes[~self.node_valid].count_nonzero() or self.appearance[~self.appearance_valid].count_nonzero() or self.contacts[~self.contact_valid].count_nonzero():
            raise ValueError("invalid prepared tokens must be zero")
        self._same_device()
        return self
