"""V6.1 runtime reuses V6 encoders and keeps an independent strict checkpoint."""
from copy import deepcopy
from dataclasses import replace

from ..native_v6.runtime import NativeFullRuntimeV6
from ..native_v5.contracts import NativeConfigV5
from ..native_dual_runtime import NativeDualRuntime, ARCHITECTURE as BASE_ARCH
from ..native_depth_runtime import _check_state
from .contracts import PoseLinkConfigV61, SPATIAL_SCHEMA
from .poselink import PoseLinkAttentionV61
from .prepared import PreparedNativeFullControlV61
from .spatial import build_spatial_condition

ARCHITECTURE = 'native_full_v61'


class NativeFullRuntimeV61(NativeFullRuntimeV6):
    def __init__(self, base=None, config=None, *, profile='poselink-full', bps_convention=None,
                 face_parameter_basis=None, pose_config=None):
        config = config or NativeConfigV5()
        pose_config = pose_config or PoseLinkConfigV61(hidden_dim=config.hidden_dim,
                                                     injection_layers=config.injection_layers)
        if not isinstance(pose_config, PoseLinkConfigV61):
            raise ValueError('V6.1 requires explicit versioned PoseLink config')
        super().__init__(base, config, profile=profile, bps_convention=bps_convention,
                         face_parameter_basis=face_parameter_basis, pose_config=pose_config)
        self.poselink = PoseLinkAttentionV61(pose_config)
        self.set_trainability(profile)

    def get_extra_state(self):
        return {**super().get_extra_state(), 'architecture': ARCHITECTURE,
            'spatial_schema': SPATIAL_SCHEMA, 'repair_schema': 'native_pose_repair_v61',
            'token_coordinates': 'target_canvas_normalized_xy', 'qwen_tokens_added': 0}

    def encode(self, batch, **kwargs):
        previous = super().encode(batch, **kwargs)
        spatial = build_spatial_condition(batch, previous.pose, self.pose_config)
        metadata = tuple({**m, 'architecture': ARCHITECTURE} for m in previous.metadata)
        core = replace(previous.core, metadata=metadata)
        return PreparedNativeFullControlV61(core, previous.relation, previous.relation_valid,
            previous.local, metadata, previous.pose, spatial).validate()

    def checkpoint_payload(self, *, metadata):
        return {**super().checkpoint_payload(metadata=metadata), 'architecture': ARCHITECTURE}

    @classmethod
    def from_v5_checkpoint(cls, saved, **kwargs):
        raise ValueError('V6.1 requires explicit V5 to V6 to V6.1 initialization')

    @classmethod
    def from_checkpoint(cls, saved):
        if saved.get('architecture') != ARCHITECTURE:
            raise ValueError('V6.1 exact rebuild only; explicitly initialize from V6')
        cfg, state = saved['runtime_config'], saved['state_dict']
        base = NativeDualRuntime.from_checkpoint({'architecture': BASE_ARCH, 'runtime_config': cfg['base'],
            'state_dict': {k[5:]: v for k, v in state.items() if k.startswith('base.')},
            'trainability': saved['base_trainability'],
            'metadata': deepcopy(saved.get('metadata', {}).get('base_provenance', {}))})
        result = cls(base, NativeConfigV5.from_dict(cfg['config']), profile=saved['profile'],
            bps_convention=cfg['bps_convention'], face_parameter_basis=cfg['face_parameter_basis'],
            pose_config=PoseLinkConfigV61.from_dict(cfg['pose_config']))
        result.set_trainability(saved['profile'], overrides=saved['trainability'])
        if result.get_extra_state() != cfg: raise ValueError('V6.1 strict architecture header differs')
        _check_state(state, result.state_dict())
        result.load_state_dict(state, strict=True)
        meta = saved.get('metadata', {})
        result.inheritance_report = deepcopy(meta.get('inheritance', {}))
        for dest, key in (('source_provenance', 'appearance_sources'), ('local_provenance', 'local_sources'),
                          ('detail_provenance', 'detail_sources')):
            setattr(result, dest, deepcopy(meta.get(key, {})))
        return result

    @classmethod
    def from_v6_checkpoint(cls, saved, *, pose_config=None):
        if pose_config is None and saved.get('runtime_config', {}).get('pose_config', {}).get('max_pose_tokens_per_reference', 256) > 256:
            raise ValueError('V6 token cap exceeds 256; provide an explicit V6.1 reduced-budget pose_config')
        previous = NativeFullRuntimeV6.from_checkpoint(saved)
        if pose_config is None:
            values = previous.pose_config.to_dict()
            cap = values['max_pose_tokens_per_reference']
            # Small verification models keep their actual original token cap.
            budget = min(cap, 256)
            values.update(global_token_budget=(budget+1)//2, joint_token_budget=budget//4,
                          local_token_budget=budget-(budget+1)//2-budget//4)
            pose_config = PoseLinkConfigV61.from_dict(values)
        result = cls(previous.base, previous.config, pose_config=pose_config,
                     bps_convention=previous.bps_convention, face_parameter_basis=previous.face_parameter_basis)
        state = result.state_dict()
        inherited = previous.state_dict()
        for key, value in inherited.items():
            if key != '_extra_state':
                if key not in state: raise ValueError('V6 parameter has no V6.1 destination: '+key)
                state[key] = value
        _check_state(state, result.state_dict())
        result.load_state_dict(state, strict=True)
        result.inheritance_report = {'source_architecture': 'native_full_v6',
            'inherited': [n for n, _ in result.named_parameters() if not n.startswith('poselink.')],
            'adapted': [n for n, _ in previous.named_parameters() if n.startswith('poselink.')],
            'new_initialized': [n for n, _ in result.named_parameters() if n not in dict(previous.named_parameters())],
            'v6_inheritance': deepcopy(previous.inheritance_report), 'optimizer_inherited': False,
            'behavior_change': 'spatial support and multi-scale default on; all three features off reproduces V6'}
        for name in ('source_provenance', 'local_provenance', 'detail_provenance'):
            setattr(result, name, deepcopy(getattr(previous, name)))
        return result
