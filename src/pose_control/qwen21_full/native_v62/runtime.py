"""Independent V6.2 checkpoint header; all inherited encoders remain unchanged."""
from copy import deepcopy
from dataclasses import replace

from ..native_v61.runtime import NativeFullRuntimeV61
from ..native_v5.contracts import NativeConfigV5
from ..native_dual_runtime import NativeDualRuntime,ARCHITECTURE as BASE_ARCH
from ..native_depth_runtime import _check_state
from .contracts import PoseLinkConfigV62,ROUTING_SCHEMA,GATE_SCHEMA
from .conditions import build_routing_conditions,build_branch_gates
from .poselink import PoseLinkAttentionV62
from .prepared import PreparedNativeFullControlV62

ARCHITECTURE='native_full_v62'


class NativeFullRuntimeV62(NativeFullRuntimeV61):
    def __init__(self,base=None,config=None,*,profile='poselink-full',bps_convention=None,
                 face_parameter_basis=None,pose_config=None):
        config=config or NativeConfigV5()
        pose_config=pose_config or PoseLinkConfigV62(hidden_dim=config.hidden_dim,injection_layers=config.injection_layers)
        if not isinstance(pose_config,PoseLinkConfigV62):raise ValueError('V6.2 PoseLink configuration required')
        super().__init__(base,config,profile=profile,bps_convention=bps_convention,
                         face_parameter_basis=face_parameter_basis,pose_config=pose_config)
        self.poselink=PoseLinkAttentionV62(pose_config);self.set_trainability(profile)

    def get_extra_state(self):
        return {**super().get_extra_state(),'architecture':ARCHITECTURE,'routing_schema':ROUTING_SCHEMA,
            'branch_gate_schema':GATE_SCHEMA,'contact_resolution':'part_level; unknown_not_positive',
            'local_gate_policy':'existing_owned_roi; no_second_soft_multiplier'}

    def encode(self,batch,**kwargs):
        old=super().encode(batch,**kwargs)
        routing=build_routing_conditions(batch,old.pose,old.spatial,self.pose_config)
        gates=build_branch_gates(batch,old.pose,old.spatial,self.pose_config)
        metadata=tuple({**m,'architecture':ARCHITECTURE} for m in old.metadata)
        return PreparedNativeFullControlV62(replace(old.core,metadata=metadata),old.relation,old.relation_valid,
            old.local,metadata,old.pose,old.spatial,routing,gates).validate()

    def checkpoint_payload(self,*,metadata):
        return {**super().checkpoint_payload(metadata=metadata),'architecture':ARCHITECTURE}

    @classmethod
    def from_checkpoint(cls,saved):
        if saved.get('architecture')!=ARCHITECTURE:raise ValueError('V6.2 exact rebuild only; explicitly initialize from V6.1')
        cfg,state=saved['runtime_config'],saved['state_dict']
        base=NativeDualRuntime.from_checkpoint({'architecture':BASE_ARCH,'runtime_config':cfg['base'],
            'state_dict':{k[5:]:v for k,v in state.items() if k.startswith('base.')},
            'trainability':saved['base_trainability'],
            'metadata':deepcopy(saved.get('metadata',{}).get('base_provenance',{}))})
        result=cls(base,NativeConfigV5.from_dict(cfg['config']),profile=saved['profile'],
            bps_convention=cfg['bps_convention'],face_parameter_basis=cfg['face_parameter_basis'],
            pose_config=PoseLinkConfigV62.from_dict(cfg['pose_config']))
        result.set_trainability(saved['profile'],overrides=saved['trainability'])
        if result.get_extra_state()!=cfg:raise ValueError('V6.2 strict header differs')
        _check_state(state,result.state_dict());result.load_state_dict(state,strict=True)
        meta=saved.get('metadata',{});result.inheritance_report=deepcopy(meta.get('inheritance',{}))
        for dest,key in (('source_provenance','appearance_sources'),('local_provenance','local_sources'),('detail_provenance','detail_sources')):
            setattr(result,dest,deepcopy(meta.get(key,{})))
        return result

    @classmethod
    def from_v61_checkpoint(cls,saved,*,pose_config=None):
        old=NativeFullRuntimeV61.from_checkpoint(saved)
        config=pose_config or PoseLinkConfigV62.from_dict(old.pose_config.to_dict())
        result=cls(old.base,old.config,pose_config=config,bps_convention=old.bps_convention,
                   face_parameter_basis=old.face_parameter_basis)
        state=result.state_dict()
        for key,value in old.state_dict().items():
            if key!='_extra_state':
                if key not in state:raise ValueError('V6.1 key has no V6.2 destination: '+key)
                state[key]=value
        _check_state(state,result.state_dict());result.load_state_dict(state,strict=True)
        result.inheritance_report={'source_architecture':'native_full_v61',
            'inherited':[n for n,_ in result.named_parameters() if not n.startswith('poselink.')],
            'adapted':[n for n,_ in result.named_parameters() if n.startswith('poselink.')],
            'new_initialized':[],'v61_inheritance':deepcopy(old.inheritance_report),'optimizer_inherited':False,
            'behavior_change':'person routing and branch gates default on; disable all new features for V6.1 equivalence'}
        for name in ('source_provenance','local_provenance','detail_provenance'):
            setattr(result,name,deepcopy(getattr(old,name)))
        return result

    @classmethod
    def from_v6_checkpoint(cls,saved,**kwargs):
        raise ValueError('V6.2 requires explicit V6 to V6.1 to V6.2 initialization')

    @classmethod
    def from_v5_checkpoint(cls,saved,**kwargs):
        raise ValueError('V6.2 requires the explicit version initialization chain')
