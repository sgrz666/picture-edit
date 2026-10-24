"""Pair-bound depth I/O, component checkpoints and Native training primitives."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn

from .depth_condition import canvas_identity, load_depth_record
from .native_body import NativeBodyConfig, Qwen21NativeBody
from .native_depth import NativeDepthConfig, Qwen21NativeDepth
from .native_depth_v2 import NativeDepthV2Config, Qwen21NativeDepthV2, DEPTH_SCHEMA
from .depth_condition_v2 import DepthConditionV2
from .native_geometry import Qwen21NativeGeometry, POSE_PREFIXES, LEGACY_SCHEMA, SCHEMA
from .body_pose import BodyPoseConfig, pose_kwargs_from_batch
from .reasoning import AdapterReasoningConfig

LEGACY_ARCHITECTURE = "qwen21_native_depth_runtime_v1"
BODY21_ARCHITECTURE = "qwen21_native_depth_runtime_v2"
ARCHITECTURE = "qwen21_native_depth_runtime_v3"


def _validate_header(saved):
    if not isinstance(saved, Mapping) or not isinstance(saved.get('geometry_config'), Mapping):
        raise ValueError('Missing Native checkpoint architecture/geometry config')
    state = saved.get('state_dict')
    if not isinstance(state, Mapping):
        raise ValueError('Native checkpoint requires a state_dict')
    extra = state.get('depth._extra_state')
    depth_schema = None if saved.get('depth_config') is None else (extra.get('schema') if isinstance(extra, Mapping) else None)
    signature = (saved.get('architecture'), saved['geometry_config'].get('schema'), depth_schema)
    supported = {(LEGACY_ARCHITECTURE, LEGACY_SCHEMA, None),
                 (LEGACY_ARCHITECTURE, LEGACY_SCHEMA, 'qwen21_native_depth_v1'),
                 (BODY21_ARCHITECTURE, SCHEMA, None),
                 (BODY21_ARCHITECTURE, SCHEMA, 'qwen21_native_depth_v1'),
                 (ARCHITECTURE, SCHEMA, DEPTH_SCHEMA)}
    if signature not in supported or (saved.get('depth_config') is not None and depth_schema is None):
        raise ValueError('Native checkpoint architecture/schema differ')
    if depth_schema is None and any(isinstance(k, str) and k.startswith('depth.') for k in state):
        raise ValueError('Depth-off checkpoint contains undeclared Depth state')
    if depth_schema is not None and extra.get('config') != saved['depth_config']:
        raise ValueError('Depth config/header differs from component state')


@dataclass(frozen=True)
class NativeTrainabilityConfig:
    geometry: bool = False
    pose: bool = True
    body: bool = True
    champ_cnn: bool = True
    champ_attention: bool = True
    depth_detail: bool = True
    depth_fusion: bool = True

    def __post_init__(self):
        if any(type(v) is not bool for v in asdict(self).values()):
            raise ValueError('Native trainability flags must be bool')

    @classmethod
    def all_native(cls):
        return cls(**{name: True for name in cls.__dataclass_fields__})


def _check_state(selected, expected):
    if selected.keys() != expected.keys():
        raise ValueError('Native checkpoint component keys differ')
    for key, value in selected.items():
        template = expected[key]
        if torch.is_tensor(template):
            if (not torch.is_tensor(value) or value.shape != template.shape or value.dtype != template.dtype
                    or value.is_meta or (value.is_floating_point() and not torch.isfinite(value).all())):
                raise ValueError(f'Invalid Native checkpoint tensor: {key}')
        elif value != template:
            raise ValueError(f'Native component schema/config differs: {key}')


class NativeDepthRecords:
    def __init__(self, path, *, mode):
        if mode not in ("camera_z", "relative_png"):
            raise ValueError("Choose an explicit depth mode")
        self.mode, self.path = mode, Path(path)
        self.records = {}
        self.fingerprints = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("pair_id")
            if not isinstance(key, str) or not key or key in self.records:
                raise ValueError("Depth manifest requires unique nonempty pair_id")
            if row.get("kind") != mode or not isinstance(row.get("target_stem"), str):
                raise ValueError("Depth manifest kind/target_stem does not match this run")
            for name in ("depth_path", "valid_path"):
                value = Path(row[name])
                row[name] = str(value if value.is_absolute() else self.path.parent / value)
            row["source_id"] = key
            self.records[key] = row
        if not self.records:
            raise ValueError("Depth manifest is empty")

    def condition(self, sample):
        provenance = sample.get("pair_metadata", {})
        key = provenance.get("pair_id")
        if key not in self.records:
            raise ValueError(f"Depth manifest is missing pair {key!r}")
        row = self.records[key]
        if row["target_stem"] != provenance.get("target_stem"):
            raise ValueError("Depth target frame differs from the geometry pair")
        condition = load_depth_record(row, sample["target_canvas"])
        condition.validate(target_token_hw=sample["target_token_hw"],
                           expected_canvas_ids=(canvas_identity(sample["target_canvas"]),))
        canvas = sample["valid_image_mask"]
        if canvas.shape != condition.canvas_valid.shape[1:] or not torch.equal(canvas.bool(), condition.canvas_valid[0]):
            raise ValueError("Depth and RGB canvas validity differ")
        if not condition.available.all():
            raise ValueError("Enabled depth training requires actual valid depth; no zero-filled fallback")
        meta = condition.metadata[0]
        signature = tuple(meta.get(k) for k in ("source_sha256", "valid_sha256", "canvas_id", "camera_id",
                                               "units", "scale_status", "kind", "input_polarity"))
        if key in self.fingerprints and self.fingerprints[key] != signature:
            raise ValueError(f"Depth asset/provenance changed during this run: {key}")
        self.fingerprints[key] = signature
        return condition


class NativeDepthRuntime(nn.Module):
    def __init__(self, geometry, body, depth=None, *, train_geometry=False, train_pose=True,
                 trainability: NativeTrainabilityConfig | None = None, train_native_all=False):
        super().__init__()
        if (not isinstance(geometry, Qwen21NativeGeometry) or not isinstance(body, Qwen21NativeBody)
                or (depth is not None and not isinstance(depth, Qwen21NativeDepth))):
            raise ValueError("Expected Native geometry/body and optional Native Depth")
        width = geometry.reasoner.config.hidden_dim
        if body.config.geometry_dim != width or (depth is not None and depth.config.geometry_dim != width):
            raise ValueError("Geometry, Depth and Native Body dimensions differ")
        self.geometry, self.body, self.depth = geometry, body, depth
        if isinstance(depth, Qwen21NativeDepthV2) and geometry.legacy:
            raise ValueError('Depth V2 uses the current Body21 frontend; legacy reconstruction remains separate')
        if train_native_all and trainability is not None:
            raise ValueError('Select all-Native or an explicit trainability config')
        config = (NativeTrainabilityConfig.all_native() if train_native_all else
                  trainability or NativeTrainabilityConfig(geometry=bool(train_geometry), pose=bool(train_pose)))
        self.set_trainability(config)
        self.pose_provenance = {}
        self.depth_provenance = {}

    def named_parameter_groups(self):
        groups = {key: [] for key in NativeTrainabilityConfig.__dataclass_fields__}
        for name, parameter in self.named_parameters():
            if name.startswith('geometry.'):
                key = 'pose' if name[len('geometry.'):].startswith(POSE_PREFIXES) else 'geometry'
            elif name.startswith('body.'):
                key = 'body'
            elif name.startswith('depth.encoder.attention.'):
                key = 'champ_attention'
            elif name.startswith('depth.encoder.'):
                key = 'champ_cnn'
            elif name.startswith('depth.detail_encoder.'):
                key = 'depth_detail'
            elif name.startswith(('depth.mixer.', 'depth.gate.', 'depth.projection.')):
                key = 'depth_fusion'
            else:
                raise ValueError(f'Undeclared Native parameter group: {name}')
            groups[key].append((name, parameter))
        return groups

    def set_trainability(self, config):
        if not isinstance(config, NativeTrainabilityConfig):
            raise ValueError('Expected NativeTrainabilityConfig')
        self.trainability = config
        self.train_geometry = config.geometry
        self.train_pose = config.pose and not self.geometry.legacy
        for key, group in self.named_parameter_groups().items():
            for _, parameter in group:
                parameter.requires_grad_(getattr(config, key))
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.geometry.train(mode and self.train_geometry)
        if not self.geometry.legacy:
            self.geometry.condition_injector.body_pose_encoder.train(mode and self.train_pose)
            self.geometry.reasoner.person_reasoner.joint_attention.train(mode and self.train_pose)
        self.body.train(mode and self.trainability.body)
        if self.depth is not None:
            self.depth.encoder.train(mode and self.trainability.champ_cnn)
            self.depth.encoder.attention.train(mode and self.trainability.champ_attention)
            if isinstance(self.depth, Qwen21NativeDepthV2):
                self.depth.detail_encoder.train(mode and self.trainability.depth_detail)
                self.depth.mixer.train(mode and self.trainability.depth_fusion)
                self.depth.gate.train(mode and self.trainability.depth_fusion)
        return self

    def encode(self, batch, *, target_token_hw, depth=None, depth_strength=1.):
        device = next(self.body.parameters()).device
        structured = batch.get('depth_condition_v2')
        if structured is not None:
            if depth is not None:
                raise ValueError('Supply Depth via explicit argument or structured batch, not both')
            depth = DepthConditionV2.stack(structured) if isinstance(structured, list) else structured
        if not self.geometry.legacy:
            for person, name in (('a', 'pose_provenance'), ('b', 'pose_provenance_b')):
                if person == 'b' and 'normal_b' not in batch:
                    continue
                provenance = batch.get(name)
                if provenance is None:
                    raise ValueError(f'Body Pose v2 requires separate cache provenance for person {person}')
                rows = provenance if isinstance(provenance, list) else [provenance]
                if len(rows) != batch['normal' if person == 'a' else 'normal_b'].shape[0]:
                    raise ValueError('Pose provenance batch size differs')
                for row in rows:
                    if not isinstance(row, Mapping) or not row.get('record_id') or not row.get('record_sha256'):
                        raise ValueError('Invalid pose provenance')
                    # A different canvas is allowed for the same immutable raw record.
                    source = {k: v for k, v in row.items() if k != 'canvas_id'}
                    for previous in self.pose_provenance.values():
                        if previous['record_id'] == row['record_id'] and source != {k: v for k, v in previous.items() if k != 'canvas_id'}:
                            raise ValueError('Body Pose source changed during runtime')
                    key = f"{person}:{row['record_id']}@{row.get('canvas_id', '')}"
                    self.pose_provenance[key] = dict(row)
        from .data_runtime import build_condition_bundle
        bundle = build_condition_bundle(self.geometry, batch, device=device, dtype=torch.float32)
        features = self.geometry(bundle, target_token_hw=target_token_hw)
        if self.depth is None:
            if depth is not None:
                raise ValueError("Depth was supplied to an off runtime")
            return features
        if isinstance(self.depth, Qwen21NativeDepthV2) and depth_strength != 0:
            if not isinstance(depth, DepthConditionV2):
                raise ValueError('Depth V2 runtime requires its typed condition')
            provenance = batch.get('pose_provenance')
            rows = provenance if isinstance(provenance, list) else [provenance]
            if len(rows) != len(depth.metadata):
                raise ValueError('Depth/Pose provenance batch differs')
            for source, meta in zip(rows, depth.metadata):
                if (source['record_id'] != meta['pose_record_id'] or source['record_sha256'] != meta['pose_record_sha256']
                        or source.get('canvas_id') != meta['pose_canvas_id']):
                    raise ValueError('Depth V2 cannot read geometry from another pose frame')
                key = meta['source_id'] + '@' + meta['canvas_id']
                if key in self.depth_provenance and self.depth_provenance[key] != meta:
                    raise ValueError('Depth V2 provenance changed during runtime')
                self.depth_provenance[key] = deepcopy(meta)
        return self.depth(features, None if depth is None else depth.to(device), strength=depth_strength)

    def checkpoint_payload(self, *, metadata):
        architecture = ARCHITECTURE if isinstance(self.depth, Qwen21NativeDepthV2) else (LEGACY_ARCHITECTURE if self.geometry.legacy else BODY21_ARCHITECTURE)
        return {"architecture": architecture,
                "geometry_config": self.geometry.get_extra_state(),
                "body_config": self.body.config.to_dict(),
                "depth_config": None if self.depth is None else asdict(self.depth.config),
                'depth_version': 'off' if self.depth is None else ('v2' if isinstance(self.depth, Qwen21NativeDepthV2) else 'v1'),
                'trainability': asdict(self.trainability),
                "state_dict": self.state_dict(), "metadata": {**dict(metadata), 'pose_provenance': self.pose_provenance,
                                                               'depth_provenance': self.depth_provenance}}

    @classmethod
    def from_native_checkpoint(cls, saved, *, train_geometry=False, train_pose=True, legacy=False, upgrade_pose=False,
                               upgrade_depth=False, trainability=None, train_native_all=False):
        _validate_header(saved)
        old = saved['geometry_config'].get('schema') == LEGACY_SCHEMA
        if old and not (legacy or upgrade_pose):
            raise ValueError('v1 checkpoint requires explicit legacy=True or upgrade_pose=True')
        if (legacy and upgrade_pose) or (not old and (legacy or upgrade_pose)):
            raise ValueError('legacy/upgrade_pose only select explicit v1 reconstruction or migration')
        pose_config = None if old else BodyPoseConfig(**saved['geometry_config']['pose_config'])
        config = saved["geometry_config"]["reasoning_config"]
        body_config = dict(saved["body_config"])
        body_config["injection_layers"] = tuple(body_config["injection_layers"])
        with torch.device("cpu"):
            geometry = Qwen21NativeGeometry(reasoning_config=AdapterReasoningConfig(**config),
                                            legacy=old and not upgrade_pose, pose_config=pose_config).float()
            body = Qwen21NativeBody(NativeBodyConfig(**body_config)).float()
            if saved['depth_config'] is None:
                if upgrade_depth:
                    raise ValueError('Depth-off source has no CHAMP Encoder; explicitly initialize from CHAMP weights')
                depth = None
            elif saved['architecture'] == ARCHITECTURE:
                if upgrade_depth:
                    raise ValueError('Depth is already V2; use exact reconstruction')
                depth = Qwen21NativeDepthV2(NativeDepthV2Config(**saved['depth_config'])).float()
            elif upgrade_depth:
                depth = Qwen21NativeDepthV2(NativeDepthV2Config(**saved['depth_config'])).float()
            else:
                depth = Qwen21NativeDepth(NativeDepthConfig(**saved['depth_config'])).float()
        result = cls(geometry, body, depth, train_geometry=train_geometry, train_pose=train_pose,
                     trainability=trainability, train_native_all=train_native_all)
        result.inheritance_report = result.load_native_weights(saved, include_depth=True,
                                                               upgrade_pose=upgrade_pose, upgrade_depth=upgrade_depth)
        result.pose_provenance = dict(saved.get('metadata', {}).get('pose_provenance', {}))
        if not upgrade_depth:
            result.depth_provenance = deepcopy(saved.get('metadata', {}).get('depth_provenance', {}))
        return result

    def load_native_weights(self, saved, *, include_depth, upgrade_pose=False, upgrade_depth=False):
        """Explicit warm start, never optimizer/RNG resume or silent architecture conversion."""
        _validate_header(saved)
        geometry_config = self.geometry.get_extra_state()
        if upgrade_pose:
            if self.geometry.legacy:
                raise ValueError('Pose migration requires a v2 destination')
            geometry_config = {k: v for k, v in geometry_config.items() if k in
                               ('schema', 'reasoning_config', 'normal_backend', 'use_depth', 'grid_policy')}
            geometry_config['schema'] = LEGACY_SCHEMA
        if (saved.get("geometry_config") != geometry_config
                or saved.get("body_config") != self.body.config.to_dict()):
            raise ValueError("Native checkpoint architecture/geometry/body config differs")
        if upgrade_depth and (not include_depth or not isinstance(self.depth, Qwen21NativeDepthV2)
                              or saved.get('depth_config') is None or saved['architecture'] == ARCHITECTURE):
            raise ValueError('Explicit Depth upgrade requires a V1 Depth source and V2 destination')
        if include_depth and not upgrade_depth and saved.get("depth_config") != (None if self.depth is None else asdict(self.depth.config)):
            raise ValueError("Native checkpoint depth config differs")
        source = saved.get("state_dict")
        if not isinstance(source, Mapping):
            raise ValueError("Missing Native state_dict")
        if any(not isinstance(k, str) or not k.startswith(('geometry.', 'body.', 'depth.')) for k in source):
            raise ValueError('Native checkpoint contains undeclared state keys')
        prefixes = ("geometry.", "body.", "depth.") if include_depth else ("geometry.", "body.")
        expected = {k: v for k, v in self.state_dict().items() if k.startswith(prefixes)}
        selected = {k: v for k, v in source.items() if k.startswith(prefixes)}
        if upgrade_pose:
            expected = {k: v for k, v in expected.items() if not k.startswith(tuple('geometry.' + p for p in POSE_PREFIXES))}
            expected['geometry._extra_state'] = geometry_config
        initialized = []
        if upgrade_depth:
            # Validate ALL V1 component keys/tensors, including the deliberately
            # discarded old output head, before any state is copied.
            with torch.device('cpu'):
                old_depth = Qwen21NativeDepth(NativeDepthConfig(**saved['depth_config'])).float()
            old_expected = {'depth.' + k: v for k, v in old_depth.state_dict().items()}
            old_selected = {k: v for k, v in source.items() if k.startswith('depth.')}
            _check_state(old_selected, old_expected)
            if (old_depth.config.geometry_dim != self.depth.config.geometry_dim
                    or old_depth.config.attention_scale != self.depth.config.attention_scale):
                raise ValueError('Depth migration preserves geometry width and CHAMP attention placement')
            initialized = [k for k in expected if k.startswith('depth.') and not k.startswith('depth.encoder.')]
            expected = {k: v for k, v in expected.items() if not k.startswith('depth.') or k.startswith('depth.encoder.')}
            selected = {k: v for k, v in selected.items() if not k.startswith('depth.') or k.startswith('depth.encoder.')}
        _check_state(selected, expected)
        if upgrade_pose:
            selected['geometry._extra_state'] = self.geometry.get_extra_state()
        if upgrade_depth:
            # Do not retain a previously active V2 head if this method is called on
            # an existing object: an explicit migration always starts at zero.
            self.depth.reset_new_fusion()
        self.load_state_dict({**self.state_dict(), **selected}, strict=True)
        return {"included_depth": include_depth, "loaded_keys": len(selected), "optimizer_resumed": False,
                'depth_upgraded': upgrade_depth, 'initialized_depth_keys': initialized,
                'discarded_depth_keys': ['depth.projection.weight', 'depth.projection.bias'] if upgrade_depth else []}


def native_velocity_loss(prediction, target, coverage):
    if (prediction.shape != target.shape or prediction.ndim != 3
            or coverage.shape != prediction.shape[:2] or not torch.isfinite(coverage).all()
            or bool(((coverage < 0) | (coverage > 1)).any()) or not bool(coverage.sum() > 0)):
        raise ValueError("Flow loss requires matching [B,N,C] and nonempty 0..1 canvas coverage")
    squared = (prediction.float() - target.float()).square()
    return (squared * coverage[..., None]).sum() / (coverage.sum() * prediction.shape[-1])
