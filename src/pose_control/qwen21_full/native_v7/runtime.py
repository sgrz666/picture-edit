"""V7 runtime: shared frontend and strict, independent checkpoint reconstruction."""
from copy import deepcopy
from dataclasses import replace

from ..native_v62.runtime import NativeFullRuntimeV62
from ..native_v5.contracts import NativeConfigV5
from ..native_dual_runtime import NativeDualRuntime,ARCHITECTURE as BASE_ARCH
from ..native_depth_runtime import _check_state
from .contracts import PoseLinkConfigV7
from .local import NativeLocalControllerV7
from .poselink import PoseLinkAttentionV7

ARCHITECTURE='native_full_v7'
_V7_HEADER={
    'architecture':ARCHITECTURE,
    'pose_mask_fusion':'person_scoped_sampled_key_residual_v1',
    'side_rope':'target_canvas_normalized_xy_fp32_v1',
    'contact_bridge':'pre_identity_geometry_person_part_only_v1',
    'identity_modulation':'source_bound_own_face_tokens_v1',
}


class NativeFullRuntimeV7(NativeFullRuntimeV62):
    def __init__(self,base=None,config=None,*,profile='poselink-full',bps_convention=None,
                 face_parameter_basis=None,pose_config=None):
        config=config or NativeConfigV5()
        pose_config=pose_config or PoseLinkConfigV7(hidden_dim=config.hidden_dim,injection_layers=config.injection_layers)
        if not isinstance(pose_config,PoseLinkConfigV7):raise ValueError('V7 PoseLink configuration required')
        super().__init__(base,config,profile=profile,bps_convention=bps_convention,
            face_parameter_basis=face_parameter_basis,pose_config=pose_config)
        self.poselink=PoseLinkAttentionV7(pose_config)
        for name,hand in (('face_controller',False),('hand_controller',True)):
            previous=getattr(self,name)
            controller=NativeLocalControllerV7(config,hand=hand,identity_modulation=pose_config.identity_modulation)
            controller.load_state_dict(previous.state_dict(),strict=False)
            setattr(self,name,controller)
        self.set_trainability(profile)

    def get_extra_state(self):
        return {**super().get_extra_state(),**_V7_HEADER}

    def set_ablation(self,**switches):
        """Apply inference switches consistently, preserving every weight object."""
        allowed={'mask_pose_fusion','spatial_rope','geometry_contact','identity_modulation'}
        unknown=set(switches)-allowed
        if unknown:raise ValueError('Unknown V7 ablation switches: '+', '.join(sorted(unknown)))
        # Validate all requested values before changing any effective setting.
        config=replace(self.pose_config,**switches)
        self.pose_config=config;self.poselink.config=config
        self.face_controller.identity_modulation_enabled=config.identity_modulation
        self.hand_controller.identity_modulation_enabled=config.identity_modulation
        return self

    def encode(self,batch,**kwargs):
        prepared=super().encode(batch,**kwargs)
        metadata=tuple({**m,'architecture':ARCHITECTURE} for m in prepared.metadata)
        return replace(prepared,core=replace(prepared.core,metadata=metadata),metadata=metadata).validate()

    def checkpoint_payload(self,*,metadata):
        return {**super().checkpoint_payload(metadata=metadata),'architecture':ARCHITECTURE}

    @classmethod
    def from_checkpoint(cls,saved):
        if saved.get('architecture')!=ARCHITECTURE:
            raise ValueError('V7 exact rebuild only; explicitly initialize weights from V6.2')
        cfg,state=saved['runtime_config'],saved['state_dict']
        if any(cfg.get(key)!=value for key,value in _V7_HEADER.items()):
            raise ValueError('V7 strict architecture/schema header differs')
        base=NativeDualRuntime.from_checkpoint({'architecture':BASE_ARCH,'runtime_config':cfg['base'],
            'state_dict':{k[5:]:v for k,v in state.items() if k.startswith('base.')},
            'trainability':saved['base_trainability'],
            'metadata':deepcopy(saved.get('metadata',{}).get('base_provenance',{}))})
        result=cls(base,NativeConfigV5.from_dict(cfg['config']),profile=saved['profile'],
            bps_convention=cfg['bps_convention'],face_parameter_basis=cfg['face_parameter_basis'],
            pose_config=PoseLinkConfigV7.from_dict(cfg['pose_config']))
        result.set_trainability(saved['profile'],overrides=saved['trainability'])
        if result.get_extra_state()!=cfg:raise ValueError('V7 strict architecture/schema header differs')
        _check_state(state,result.state_dict());result.load_state_dict(state,strict=True)
        meta=saved.get('metadata',{});result.inheritance_report=deepcopy(meta.get('inheritance',{}))
        for dest,key in (('source_provenance','appearance_sources'),('local_provenance','local_sources'),('detail_provenance','detail_sources')):
            setattr(result,dest,deepcopy(meta.get(key,{})))
        return result

    @classmethod
    def from_v62_checkpoint(cls,saved,*,pose_config=None):
        previous=NativeFullRuntimeV62.from_checkpoint(saved)
        config=pose_config or PoseLinkConfigV7.from_dict(previous.pose_config.to_dict())
        result=cls(previous.base,previous.config,pose_config=config,bps_convention=previous.bps_convention,
            face_parameter_basis=previous.face_parameter_basis)
        state=result.state_dict();old_state=previous.state_dict()
        for key,value in old_state.items():
            if key=='_extra_state':continue
            if key not in state:raise ValueError('V6.2 tensor has no V7 destination: '+key)
            state[key]=value
        _check_state(state,result.state_dict());result.load_state_dict(state,strict=True)
        old_names=set(dict(previous.named_parameters()))
        adapted_prefixes=('poselink.','face_controller.','hand_controller.')
        names=tuple(dict(result.named_parameters()))
        result.inheritance_report={'source_architecture':'native_full_v62',
            'inherited':[n for n in names if n in old_names and not n.startswith(adapted_prefixes)],
            'adapted':[n for n in names if n in old_names and n.startswith(adapted_prefixes)],
            'new_initialized':[n for n in names if n not in old_names],
            'v62_inheritance':deepcopy(previous.inheritance_report),'optimizer_inherited':False,'rng_inherited':False,
            'behavior_change':'V7 switches default on; disable all four for V6.2 exact behavior'}
        for name in ('source_provenance','local_provenance','detail_provenance'):
            setattr(result,name,deepcopy(getattr(previous,name)))
        return result

    @classmethod
    def from_v61_checkpoint(cls,saved,**kwargs):
        raise ValueError('V7 requires explicit V6.2 weight initialization')
