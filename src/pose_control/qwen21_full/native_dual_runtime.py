"""Independent V3 runtime. Legacy Full/single runtimes and Native Body are untouched."""
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from .body_pose import BodyPoseConfig
from .dual_depth_condition import DualDepthConditionV3, CONDITION_SCHEMA, IDENTITY_SCHEMA, RELATION_CHANNELS, INTERACTION_CHANNELS, FAMILIES
from .native_dual_depth import NativeDualConfig, DualControlOptions, NativeDualDepth, RelationRasterEncoder, PartRelationEncoder, GatedNativeResidual, interaction_context
from .native_dual_identity import NativeDualIdentityBinder
from .native_depth import _ResidualAdd
from .native_depth_runtime import NativeDepthRuntime, ARCHITECTURE as V2_ARCHITECTURE, _check_state
from .native_geometry import Qwen21NativeGeometry, POSE_PREFIXES
from .native_body import Qwen21NativeBody, NativeBodyConfig
from .reasoning import AdapterReasoningConfig
from .data_runtime import build_condition_bundle

ARCHITECTURE = 'native_dual_geometry_v3'
DUAL_GEOMETRY_PREFIXES = ('condition_injector.relative_encoder.', 'condition_injector.contact_raster_encoder.',
    'condition_injector.contact_relation_encoder.', 'reasoner.role_embedding.', 'reasoner.cross_person_reasoner.',
    'reasoner.contact_reasoner.', 'reasoner.dual_fusion.', 'interaction_high_downsample.')


@dataclass(frozen=True)
class DualTrainability:
    geometry_shared: bool = False
    body21: bool = True
    geometry_dual: bool = True
    native_body: bool = True
    champ_cnn: bool = True
    champ_attention: bool = True
    detail_cnn: bool = True
    depth_fusion: bool = True
    relation_raster: bool = True
    relation_tokens: bool = True
    interaction_fusion: bool = True
    identity_binding: bool = True

    def __post_init__(self):
        if any(type(x) is not bool for x in asdict(self).values()):
            raise ValueError('Trainability flags must be bool')

    @classmethod
    def all_native(cls):
        return cls(**{k: True for k in cls.__dataclass_fields__})


class NativeDualRuntime(nn.Module):
    def __init__(self, geometry=None, body=None, config=None, *, trainability=None):
        super().__init__()
        self.config = config or NativeDualConfig()
        self.geometry = geometry or Qwen21NativeGeometry()
        self.body = body or Qwen21NativeBody()
        if self.geometry.legacy or self.geometry.reasoner.config.hidden_dim != 512 or self.body.config.geometry_dim != 512:
            raise ValueError('V3 requires current Body21 and production 512-dimensional Native Body')
        self.depth = NativeDualDepth(self.config.depth_config())
        self.relation_raster = RelationRasterEncoder()
        self.relation_tokens = PartRelationEncoder()
        self.interaction_fusion = GatedNativeResidual(len(INTERACTION_CHANNELS))
        self.identity_binder = NativeDualIdentityBinder()
        self.identity_fusion = GatedNativeResidual(len(INTERACTION_CHANNELS)+2)
        self.family_embedding = nn.Embedding(len(FAMILIES), 256)
        nn.init.normal_(self.family_embedding.weight, std=.02)
        with torch.no_grad():
            self.family_embedding.weight[0].zero_()
        self.provenance = {}
        self.identity_provenance = {}
        self.inheritance_report = {}
        self.set_trainability(trainability or DualTrainability())

    def get_extra_state(self):
        return {'architecture': ARCHITECTURE, 'condition_schema': CONDITION_SCHEMA, 'identity_schema': IDENTITY_SCHEMA,
                'config': asdict(self.config), 'relation_channels': list(RELATION_CHANNELS),
                'interaction_channels': list(INTERACTION_CHANNELS), 'family_order': list(FAMILIES),
                'part_relations': '196x256_to_28x256_row_column', 'pair_numeric_dim': 9,
                'contact': 'official_one_based_zero_no_record; directed; inferred_separate',
                'geometry_source': 'paired_target_frame; camera_cv; same_scene_scale',
                'geometry_config': self.geometry.get_extra_state(), 'body_config': self.body.config.to_dict()}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Dual runtime schema/config differs')

    def named_parameter_groups(self):
        groups = {k: [] for k in DualTrainability.__dataclass_fields__}
        for name, p in self.named_parameters():
            if name.startswith('geometry.'):
                relative = name[len('geometry.'):]
                key = 'body21' if relative.startswith(POSE_PREFIXES) else ('geometry_dual' if relative.startswith(DUAL_GEOMETRY_PREFIXES) else 'geometry_shared')
            elif name.startswith('body.'):
                key = 'native_body'
            elif name.startswith('depth.encoder.attention.'):
                key = 'champ_attention'
            elif name.startswith('depth.encoder.'):
                key = 'champ_cnn'
            elif name.startswith('depth.detail_encoder.'):
                key = 'detail_cnn'
            elif name.startswith('depth.'):
                key = 'depth_fusion'
            elif name.startswith('family_embedding.'):
                key = 'geometry_dual'
            elif name.startswith(('identity_binder.', 'identity_fusion.')):
                key = 'identity_binding'
            else:
                key = name.split('.')[0]
            if key not in groups:
                raise ValueError('Undeclared V3 parameter: '+name)
            groups[key].append((name, p))
        return groups

    def set_trainability(self, config):
        if not isinstance(config, DualTrainability):
            raise ValueError('Expected DualTrainability')
        self.trainability = config
        for key, values in self.named_parameter_groups().items():
            for _, p in values:
                p.requires_grad_(getattr(config, key))
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'trainability'):
            self.geometry.train(mode and self.trainability.geometry_shared)
            for name, module in self.geometry.named_modules():
                key = 'body21' if (name+'.').startswith(POSE_PREFIXES) else 'geometry_dual'
                if (name+'.').startswith(POSE_PREFIXES + DUAL_GEOMETRY_PREFIXES):
                    module.train(mode and getattr(self.trainability, key))
            self.body.train(mode and self.trainability.native_body)
            self.depth.encoder.train(mode and self.trainability.champ_cnn)
            self.depth.encoder.attention.train(mode and self.trainability.champ_attention)
            self.depth.detail_encoder.train(mode and self.trainability.detail_cnn)
            for module in (self.depth.mixer, self.depth.gate, self.depth.projection):
                module.train(mode and self.trainability.depth_fusion)
            self.relation_raster.train(mode and self.trainability.relation_raster)
            self.relation_tokens.train(mode and self.trainability.relation_tokens)
            self.interaction_fusion.train(mode and self.trainability.interaction_fusion)
            self.identity_binder.train(mode and self.trainability.identity_binding)
            self.identity_fusion.train(mode and self.trainability.identity_binding)
        return self

    def encode(self, batch, *, target_token_hw, depth=None, identity=None, options=None):
        options = options or DualControlOptions()
        if not isinstance(options, DualControlOptions):
            raise ValueError('Expected explicit DualControlOptions')
        depth = depth if depth is not None else batch.get('dual_depth')
        if not isinstance(depth, DualDepthConditionV3):
            raise ValueError('V3 requires typed dual geometry even for Depth-off ablations')
        device = next(self.parameters()).device
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('Native V3 parameters must stay FP32')
        depth = depth.to(device).validate(target_token_hw=target_token_hw)
        for i, meta in enumerate(depth.metadata):
            if options.depth_mode == 'full_geometry' and meta['mode'] != 'full_geometry':
                raise ValueError('No physical fallback from a PNG-only record')
            if meta.get('geometry_config') != asdict(self.config):
                raise ValueError('Prepared geometry thresholds/config differ from checkpoint')
            for suffix, person in (('', depth.person_a), ('_b', depth.person_b)):
                rows = batch.get('pose_provenance'+suffix)
                rows = rows if isinstance(rows, list) else [rows]
                row = rows[i]
                if (not isinstance(row, dict) or row.get('record_sha256') != meta['record_sha256']
                        or row.get('canvas_id') != meta['canvas_id'] or row.get('person_id') != person.metadata[i]['person_id']):
                    raise ValueError('Dual Pose/Depth provenance or person slot differs')
            key = meta['pair_id']+'@'+meta['canvas_id']
            if key in self.provenance and self.provenance[key] != meta:
                raise ValueError('Dual source changed during runtime')
            self.provenance[key] = deepcopy(meta)
        with torch.autocast(device_type=device.type, enabled=False):
            bundle = build_condition_bundle(self.geometry, batch, device=device, dtype=torch.float32)
            if not bundle.person_valid.all():
                raise ValueError('V3 is dual-only; single rows use the original V1/V2 runtime')
            for person, p in enumerate(('a', 'b')):
                binding = self.family_embedding(depth.model_family[:, person])[:, None]
                setattr(bundle, f'person_{p}_global_tokens', getattr(bundle, f'person_{p}_global_tokens')+binding)
                setattr(bundle, f'person_{p}_joint_tokens', getattr(bundle, f'person_{p}_joint_tokens')+binding)
            state = self.geometry.reasoner(bundle)
            features = self.geometry.project_state(state, target_token_hw=target_token_hw)
            value = features.geometry
            if options.depth_mode != 'off' and options.depth_strength:
                update = self.depth(value, depth, use_detail=options.detail)
                value = _ResidualAdd.apply(value, update*options.depth_strength)
            context = interaction_context(depth).flatten(2).transpose(1, 2)
            support = F.max_pool2d(depth.projected.any(1).float(), 16).flatten(1)
            interaction = features.interaction
            if options.interaction and options.interaction_strength:
                if options.raster and options.relation_strength:
                    interaction = interaction + self.relation_raster(depth)*options.relation_strength
                if options.part_contact and options.relation_strength:
                    interaction = interaction + self.relation_tokens(value, features.interaction, depth)*options.relation_strength
                value = _ResidualAdd.apply(value, self.interaction_fusion(value, interaction, context, support)*options.interaction_strength)
            if options.identity and options.identity_strength:
                if identity is None:
                    raise ValueError('Identity is enabled; source-only Qwen latent binding is required')
                identity = identity.to(device).validate()
                if identity.source_latent.shape[0] != value.shape[0]:
                    raise ValueError('Identity/Geometry batch differs')
                declared = batch.get('identity_metadata')
                if not isinstance(declared, list) or len(declared) != len(identity.metadata):
                    raise ValueError('Identity must be bound to the declared source batch')
                for source_meta, declared_meta, target_meta in zip(identity.metadata, declared, depth.metadata):
                    if source_meta != declared_meta:
                        raise ValueError('Identity source differs from the reference image/canvas provenance')
                    key = target_meta['pair_id']+'@'+source_meta['source_canvas_id']
                    if key in self.identity_provenance and self.identity_provenance[key] != source_meta:
                        raise ValueError('Identity reference changed during runtime')
                    self.identity_provenance[key] = deepcopy(source_meta)
                if any(list(m['target_person_ids']) != list(d['person_ids']) for m, d in zip(identity.metadata, depth.metadata)):
                    raise ValueError('Identity target slots differ from the geometry identities')
                weight = F.avg_pool2d(depth.owner_weights.flatten(0, 1), 16).reshape(value.shape[0], 2, 1, *target_token_hw)
                update = self.identity_binder(identity, state.person_tokens, state.person_token_mask, weight)
                flags = identity.source_valid.gather(1, identity.source_indices).float()[:, None].expand(-1, value.shape[1], -1)
                identity_support = weight.sum(1).flatten(1)
                value = _ResidualAdd.apply(value, self.identity_fusion(value, update, torch.cat((context, flags), -1), identity_support)*options.identity_strength)
        return replace(features, geometry=value, interaction=interaction)

    def checkpoint_payload(self, *, metadata):
        return {'architecture': ARCHITECTURE, 'runtime_config': self.get_extra_state(),
                'trainability': asdict(self.trainability), 'state_dict': self.state_dict(),
                'metadata': {**deepcopy(metadata), 'provenance': deepcopy(self.provenance),
                             'identity_provenance': deepcopy(self.identity_provenance),
                             'inheritance': deepcopy(self.inheritance_report)}}

    @classmethod
    def from_checkpoint(cls, saved, *, trainability=None):
        if saved.get('architecture') != ARCHITECTURE:
            raise ValueError('V3 reconstruction only; migrate older checkpoints explicitly')
        cfg = saved['runtime_config']
        geometry = Qwen21NativeGeometry(reasoning_config=AdapterReasoningConfig(**cfg['geometry_config']['reasoning_config']),
            pose_config=BodyPoseConfig(**cfg['geometry_config']['pose_config']))
        body_config = {**cfg['body_config'], 'injection_layers': tuple(cfg['body_config']['injection_layers'])}
        result = cls(geometry, Qwen21NativeBody(NativeBodyConfig(**body_config)), NativeDualConfig(**cfg['config']),
                     trainability=trainability or DualTrainability(**saved['trainability']))
        if result.get_extra_state() != cfg:
            raise ValueError('V3 checkpoint header differs')
        _check_state(saved['state_dict'], result.state_dict())
        result.load_state_dict(saved['state_dict'], strict=True)
        result.provenance = deepcopy(saved.get('metadata', {}).get('provenance', {}))
        result.identity_provenance = deepcopy(saved.get('metadata', {}).get('identity_provenance', {}))
        result.inheritance_report = deepcopy(saved.get('metadata', {}).get('inheritance', {}))
        return result

    @classmethod
    def from_depth_v2(cls, saved):
        if saved.get('architecture') != V2_ARCHITECTURE:
            raise ValueError('Explicit migration source must be Body21 + single Depth V2')
        base = NativeDepthRuntime.from_native_checkpoint(saved)
        if base.depth is None:
            raise ValueError('Missing V2 CHAMP/Detail source')
        d = base.depth.config
        result = cls(base.geometry, base.body, NativeDualConfig(attention_scale=d.attention_scale,
            body_jump_threshold=d.body_jump_threshold, relative_jump_threshold=d.relative_jump_threshold, epsilon=d.epsilon))
        inherited = {k: v for k, v in base.depth.state_dict().items() if k not in ('projection.weight', 'projection.bias', '_extra_state')}
        expected = {k: v for k, v in result.depth.state_dict().items() if k not in ('projection.weight', 'projection.bias', '_extra_state')}
        _check_state(inherited, expected)
        result.depth.load_state_dict({**result.depth.state_dict(), **inherited}, strict=True)
        source_names = {'geometry.'+k for k in base.geometry.state_dict()} | {'body.'+k for k in base.body.state_dict()} | {'depth.'+k for k in inherited}
        result.inheritance_report = {'source_architecture': V2_ARCHITECTURE, 'inherited_keys': sorted(source_names),
            'discarded_keys': ['depth.projection.weight', 'depth.projection.bias', 'depth._extra_state'],
            'initialized_keys': sorted(set(result.state_dict())-source_names), 'optimizer_resumed': False,
            'zero_outputs': ['depth.projection', 'interaction_fusion.output', 'identity_fusion.output']}
        return result
