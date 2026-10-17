"""显式人物绑定的任务条件；此 payload 不接收目标外观监督。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace

import torch
from torch import Tensor

SPATIAL_CHANNELS = {"normal": 3, "depth": 1, "mask": 1, "visibility": 1, "contact": 1, "part": 1}


def _signature(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ProAdapterConfig:
    hidden_dim: int = 4096
    width: int = 512
    heads: int = 8
    stage_indices: tuple[int, ...] = (0, 4, 8, 12)
    source_dim: int = 64
    identity_dim: int = 512
    max_parts: int = 128

    def __post_init__(self):
        for name in ("hidden_dim", "width", "heads", "source_dim", "identity_dim", "max_parts"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not isinstance(self.stage_indices, tuple) or not self.stage_indices:
            raise ValueError("stage_indices must be a nonempty tuple")
        if any(type(i) is not int or not 0 <= i < 16 for i in self.stage_indices):
            raise ValueError("stage_indices must be within the 16 official control stages")
        if len(set(self.stage_indices)) != len(self.stage_indices):
            raise ValueError("stage_indices must be unique")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["stage_indices"] = list(self.stage_indices)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> ProAdapterConfig:
        values = dict(payload)
        if "stage_indices" in values:
            values["stage_indices"] = tuple(values["stage_indices"])
        return cls(**values)

    def signature(self) -> str:
        return _signature(self.to_dict())


def _tensor(name, value, shape, *, kind="float", unit=False):
    if not isinstance(value, Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    correct = {"float": value.is_floating_point(), "bool": value.dtype == torch.bool,
               "long": value.dtype == torch.long}[kind]
    if not correct:
        raise ValueError(f"{name} must have {kind} dtype")
    if kind == "float":
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
        if unit and ((value < 0).any() or (value > 1).any()):
            raise ValueError(f"{name} must be in [0, 1]")


def _convert(value: Tensor, device, dtype) -> Tensor:
    return value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)


@dataclass(frozen=True)
class ContactRelations:
    src_person: Tensor
    src_part: Tensor
    dst_person: Tensor
    dst_part: Tensor
    relative: Tensor
    confidence: Tensor
    valid: Tensor

    def validate(self, *, batch: int, people: int, person_valid: Tensor, max_parts: int = 128):
        if not isinstance(self.valid, Tensor) or self.valid.ndim != 2:
            raise ValueError("relations.valid must have shape [B,E]")
        shape = (batch, self.valid.shape[1])
        for name in ("src_person", "src_part", "dst_person", "dst_part"):
            _tensor(f"relations.{name}", getattr(self, name), shape, kind="long")
        _tensor("relations.relative", self.relative, (*shape, 6))
        _tensor("relations.confidence", self.confidence, shape, unit=True)
        _tensor("relations.valid", self.valid, shape, kind="bool")
        for person_name, part_name in (("src_person", "src_part"), ("dst_person", "dst_part")):
            person, part = getattr(self, person_name), getattr(self, part_name)
            valid_person = person[self.valid]
            valid_part = part[self.valid]
            if ((valid_person < 0) | (valid_person >= people)).any():
                raise ValueError("valid relation endpoint has invalid person index")
            if ((valid_part < 0) | (valid_part >= max_parts)).any():
                raise ValueError("valid relation endpoint has invalid part index")
            # Clamp only masked placeholders; valid endpoints were checked above.
            enabled = person_valid.gather(1, person.clamp(0, people - 1))
            if (self.valid & ~enabled).any():
                raise ValueError("valid relation endpoint must name a valid person")
        return self

    def to(self, device=None, dtype=None) -> ContactRelations:
        return replace(self, **{f.name: _convert(getattr(self, f.name), device, dtype) for f in fields(self)})

    def repeat_batch(self, n: int) -> ContactRelations:
        if type(n) is not int or n <= 0:
            raise ValueError("repeat count must be a positive integer")
        return replace(self, **{f.name: getattr(self, f.name).repeat((n,) + (1,) * (getattr(self, f.name).ndim - 1)) for f in fields(self)})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, payload: Mapping) -> ContactRelations:
        return cls(**dict(payload))


_GEOMETRY_TARGET_FIELDS = {
    prefix + suffix
    for prefix in ("target", "groundtruth", "gt")
    for suffix in ("geometry", "hw", "grid", "shape", "pose", "camera", "mesh", "keypoints", "depth",
                   "normal", "mask", "visibility", "part", "contact", "parameters", "transform", "geometrycodec", "normalbasis")
}
_APPEARANCE_FIELDS = ("rgb", "image", "latent", "identity", "arcface", "appearance", "embedding", "faceid")


def _availability_record(value):
    if type(value) is bool:
        return True
    if not isinstance(value, Mapping) or not set(value) <= {"valid", "available", "reason"}:
        return False
    flags = [value[k] for k in ("valid", "available") if k in value]
    return bool(flags) and all(type(flag) is bool for flag in flags) and ("reason" not in value or isinstance(value["reason"], str))


def _metadata(value, *, source=False, availability=False):
    """结构化 JSON provenance：外观只允许 source 命名域，目标仅允许几何字段。

    普通 codec/hash/path/transform 元数据可自由嵌套。source/source_* 显式声明源
    外观来源；target/gt 标记始终优先于继承的 source 命名域。availability 可记录
    bool 或 {valid/available: bool, reason: str}，不能以特征列表代替有效性记录。
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("provenance metadata keys must be strings")
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized == "availability":
                if not isinstance(child, Mapping) or any(not _availability_record(record) for record in child.values()):
                    raise ValueError("provenance availability must map each field to bool or a valid/available/reason record")
            targeted = ("target" in normalized and not normalized.startswith("retarget")) or normalized.startswith(("groundtruth", "gt"))
            appearance = any(term in normalized for term in _APPEARANCE_FIELDS) or normalized in {"feature", "features"}
            if targeted and (appearance or normalized not in _GEOMETRY_TARGET_FIELDS):
                raise ValueError(f"provenance metadata cannot contain supervision key: {key}")
            child_source = (source or normalized.startswith("source")) and not targeted
            if appearance and not child_source and not (availability and _availability_record(child)):
                raise ValueError(f"provenance appearance metadata must be inside source fields: {key}")
            _metadata(child, source=child_source, availability=normalized == "availability")
    elif isinstance(value, (tuple, list)):
        for child in value:
            _metadata(child, source=source)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("provenance metadata must contain JSON values, not tensor payloads")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("provenance metadata cannot contain nonfinite numbers")


@dataclass(frozen=True)
class TaskCondition:
    target_hw: tuple[int, int]
    person_ids: tuple[int, ...]
    spatial: Mapping[str, Tensor]
    spatial_valid: Mapping[str, Tensor]
    person_mask: Tensor
    owner: Tensor
    face_mask: Tensor
    hand_mask: Tensor
    person_valid: Tensor
    source_tokens: Tensor
    source_token_valid: Tensor
    face_identity: Tensor
    face_identity_valid: Tensor
    relations: ContactRelations
    source_fingerprint: str
    provenance: dict

    def validate(self, config: ProAdapterConfig | None = None):
        config = config or ProAdapterConfig()
        if not isinstance(self.target_hw, tuple) or len(self.target_hw) != 2 or any(type(v) is not int or v <= 0 for v in self.target_hw):
            raise ValueError("target_hw must be a positive (H,W) tuple")
        if not isinstance(self.person_valid, Tensor) or self.person_valid.ndim != 2:
            raise ValueError("person_valid must have shape [B,P]")
        b, p = self.person_valid.shape
        if b <= 0 or not 1 <= p <= 2:
            raise ValueError("conditions require B>=1 and one or two people")
        if not isinstance(self.person_ids, tuple) or len(self.person_ids) != p or any(type(v) is not int for v in self.person_ids) or len(set(self.person_ids)) != p:
            raise ValueError("person_ids must be unique integer identifiers for each person slot")
        h, w = self.target_hw
        n = h * w
        _tensor("person_valid", self.person_valid, (b, p), kind="bool")
        for name in ("person_mask", "owner", "face_mask"):
            _tensor(name, getattr(self, name), (b, p, n), unit=True)
        _tensor("hand_mask", self.hand_mask, (b, p, 2, n), unit=True)
        if (self.owner.sum(dim=1) > 1 + 1e-6).any():
            raise ValueError("owner sum over people must be <=1")
        if not isinstance(self.source_tokens, Tensor) or self.source_tokens.ndim != 4:
            raise ValueError("source_tokens must have shape [B,P,M,source_dim]")
        m = self.source_tokens.shape[2]
        _tensor("source_tokens", self.source_tokens, (b, p, m, config.source_dim))
        _tensor("source_token_valid", self.source_token_valid, (b, p, m), kind="bool")
        _tensor("face_identity", self.face_identity, (b, p, config.identity_dim))
        _tensor("face_identity_valid", self.face_identity_valid, (b, p), kind="bool")
        if (not isinstance(self.spatial, Mapping) or not isinstance(self.spatial_valid, Mapping)
                or set(self.spatial) != set(SPATIAL_CHANNELS) or set(self.spatial_valid) != set(SPATIAL_CHANNELS)):
            raise ValueError("spatial and spatial_valid must contain exactly six modalities: normal/depth/mask/visibility/contact/part")
        for name, value in self.spatial.items():
            if name not in SPATIAL_CHANNELS:
                raise ValueError(f"unknown spatial modality: {name}")
            _tensor(f"spatial.{name}", value, (b, p, SPATIAL_CHANNELS[name], h, w), kind="long" if name == "part" else "float", unit=name in ("mask", "visibility", "contact"))
            _tensor(f"spatial_valid.{name}", self.spatial_valid[name], (b, p), kind="bool")
            if name == "part" and ((value < -1).any() or (value >= config.max_parts).any()):
                raise ValueError("part labels must be -1 or within max_parts")
        if not isinstance(self.relations, ContactRelations):
            raise ValueError("relations must be ContactRelations")
        tensors = [getattr(self, name) for name in ("person_mask", "owner", "face_mask", "hand_mask", "person_valid", "source_tokens", "source_token_valid", "face_identity", "face_identity_valid")]
        tensors += list(self.spatial.values()) + list(self.spatial_valid.values()) + list(self.relations.to_dict().values())
        if any(value.device != self.person_valid.device for value in tensors):
            raise ValueError("all condition tensors must be on the same device")
        invalid_person = ~self.person_valid
        if (self.owner.gt(0) & invalid_person[..., None]).any():
            raise ValueError("invalid person must have zero owner")
        if (self.face_identity_valid & invalid_person).any() or (self.source_token_valid & invalid_person[..., None]).any():
            raise ValueError("invalid person cannot enable source tokens or face identity")
        if any((valid & invalid_person).any() for valid in self.spatial_valid.values()):
            raise ValueError("invalid person cannot enable spatial modalities")
        if (self.owner > self.person_mask).any():
            raise ValueError("owner must not exceed person_mask")
        visibility = self.spatial["visibility"].flatten(2) * self.spatial_valid["visibility"][..., None]
        if (self.owner > visibility).any():
            raise ValueError("owner must not exceed valid visibility; unavailable visibility requires owner=0")
        self.relations.validate(batch=b, people=p, person_valid=self.person_valid, max_parts=config.max_parts)
        if not isinstance(self.source_fingerprint, str) or not self.source_fingerprint:
            raise ValueError("source_fingerprint must be nonempty")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a dictionary")
        _metadata(self.provenance)
        return self

    def visible_owner(self) -> Tensor:
        """验证后保持原 owner 的 soft 权重，不再重复乘 visibility 使其衰减。"""
        visibility = self.spatial["visibility"].flatten(2) * self.spatial_valid["visibility"][..., None]
        support = torch.minimum(self.person_mask, visibility) * self.person_valid[..., None]
        return torch.minimum(self.owner, support)

    def to(self, device=None, dtype=None) -> TaskCondition:
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                values[field.name] = _convert(value, device, dtype)
        return replace(self, **values,
                       spatial={k: _convert(v, device, dtype) for k, v in self.spatial.items()},
                       spatial_valid={k: _convert(v, device, dtype) for k, v in self.spatial_valid.items()},
                       relations=self.relations.to(device, dtype), provenance=copy.deepcopy(self.provenance))

    def repeat_batch(self, n: int) -> TaskCondition:
        if type(n) is not int or n <= 0:
            raise ValueError("repeat count must be a positive integer")
        def repeat(value):
            return value.repeat((n,) + (1,) * (value.ndim - 1))
        values = {f.name: repeat(getattr(self, f.name)) for f in fields(self) if isinstance(getattr(self, f.name), Tensor)}
        return replace(self, **values, spatial={k: repeat(v) for k, v in self.spatial.items()},
                       spatial_valid={k: repeat(v) for k, v in self.spatial_valid.items()},
                       relations=self.relations.repeat_batch(n), provenance=copy.deepcopy(self.provenance))

    def layout_signature(self) -> str:
        # Include binding order and actual tensor layout, but never tensor values.
        tensor_layout = {f.name: {"shape": list(getattr(self, f.name).shape), "dtype": str(getattr(self, f.name).dtype)}
                         for f in fields(self) if isinstance(getattr(self, f.name), Tensor)}
        return _signature({"version": 1, "target_hw": self.target_hw, "person_ids": self.person_ids,
                           "tensors": tensor_layout, "relations": {k: list(v.shape) for k, v in self.relations.to_dict().items()},
                           "spatial": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in self.spatial.items()}})

    def to_dict(self) -> dict:
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload["relations"] = self.relations.to_dict()
        payload["spatial"] = dict(self.spatial)
        payload["spatial_valid"] = dict(self.spatial_valid)
        payload["provenance"] = copy.deepcopy(self.provenance)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> TaskCondition:
        values = dict(payload)
        values["target_hw"] = tuple(values["target_hw"])
        values["person_ids"] = tuple(values["person_ids"])
        if not isinstance(values["relations"], ContactRelations):
            values["relations"] = ContactRelations.from_dict(values["relations"])
        return cls(**values)
