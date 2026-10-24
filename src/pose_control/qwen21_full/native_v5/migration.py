"""Explicit V4 -> V5 strict inheritance. This module never runs migration on import."""
from copy import deepcopy
import torch
from ..native_full_runtime import NativeFullRuntimeV4
from ..native_depth_runtime import _check_state
from .runtime import NativeFullRuntimeV5
from .contracts import NativeConfigV5

LOCAL_SHARED=('query.','spatial_projection.','geometry_attention.','appearance_attention.','self_norm.',
              'shared_attention.','shared_ffn.','layer_embedding.','time.','side.')
OPTIONAL_WARM=('face_controller.shared_attention.','face_controller.shared_ffn.','face_controller.self_norm.',
    'hand_controller.shared_attention.','hand_controller.shared_ffn.','hand_controller.self_norm.',
    'face_controller.stage_films.','hand_controller.stage_films.','face_controller.stage_lowrank.',
    'hand_controller.stage_lowrank.','strength.log_group_gain.')


def evidence_valid(evidence):
    return (isinstance(evidence,dict) and type(evidence.get('step'))==int and evidence['step']>0
        and all(isinstance(evidence.get(k),str) and len(evidence[k])==64 for k in ('checkpoint_sha256','training_log_sha256')))


def migrate_v4_to_v5(saved,*,config=None,allow_fresh_hand=False,allow_fresh_detail=False,
                     warm_start_native_local=True,warm_start=None):
    if not allow_fresh_detail: raise ValueError('New Source64 Detail Bridge requires explicit allow_fresh_detail; CHAMP Detail is a different module')
    evidence=saved.get('metadata',{}).get('module_training_evidence',{})
    if not evidence_valid(evidence.get('hand')) and not allow_fresh_hand:
        raise ValueError('Hand training source unconfirmed; explicitly allow_fresh_hand')
    old=NativeFullRuntimeV4.from_checkpoint(saved)
    schedules_adapted=[]
    if config is None:
        raw=old.config.to_dict()
        for family,branches in (('face',('face_identity','face_texture','face_geometry')),('hand',('hand_geometry','hand_texture'))):
            points=tuple(map(tuple,raw[family+'_schedule']))
            if points!=((0.,1.),(1.,1.)):
                schedules_adapted.extend((b,{'curve':'piecewise','knots':points}) for b in branches)
            raw[family+'_schedule']=((0.,1.),(1.,1.))
        config=NativeConfigV5(**raw,schedules=tuple(schedules_adapted))
    result=NativeFullRuntimeV5(old.base,config,bps_convention=old.bps_convention,face_parameter_basis=old.face_parameter_basis)
    template=result.state_dict();source=saved['state_dict'];inherited={};adapted={};resets={};used=set()
    for key,value in source.items():
        compatible=key.startswith(('base.','face_preparer.','hand_preparer.'))
        if warm_start_native_local and key.startswith(('face_controller.','hand_controller.')):
            compatible=key.split('.',1)[1].startswith(LOCAL_SHARED)
        if not compatible or key not in template: continue
        if key.startswith('face_preparer.geometry_tokenizer.parameter_encoder.') and old.face_parameter_basis is None:
            resets[key]='No verified Face parameter basis';continue
        if key.startswith('hand_preparer.structure.mano_bps.') and old.bps_convention is None:
            resets[key]='No verified BPS basis/frame/scale';continue
        _check_state({key:value},{key:template[key]})
        template[key]=value;used.add(key)
        if key.startswith(('face_controller.','hand_controller.')): adapted[key]=key+'; shared Native operation, separated context semantics'
        else: inherited[key]=key
    warm_sources=[]
    for job in warm_start or ():
        # Optional legacy warm-start is never accepted on a caller-provided
        # "compatible=True" alone. Its source must carry the actual conventions.
        raw=job['checkpoint'];state=raw[job['state_key']];meta=raw.get('metadata',{})
        required={'time_convention':'normalized_sigma','roi_coordinate_convention':'native_target_roi','control_width':512}
        if any(meta.get(k)!=v for k,v in required.items()): raise ValueError('Legacy warm-start lacks verified time/coordinate/width conventions')
        selected=set()
        for src,dst in job['mapping'].items():
            if not dst.startswith(OPTIONAL_WARM) or src not in state or dst not in template: raise ValueError('Undeclared warm-start tensor: '+dst)
            _check_state({dst:state[src]},{dst:template[dst]});template[dst]=state[src]
            adapted[dst]=job['source_sha256']+':'+src;selected.add(src)
        warm_sources.append({'source_sha256':job['source_sha256'],'selected':sorted(selected),
            'excluded':sorted(set(state)-selected),'conventions':required})
    _check_state(template,result.state_dict());result.load_state_dict(template,strict=True)
    fresh=sorted(set(template)-set(inherited)-set(adapted))
    result.inheritance_report={'source_architecture':'native_full_v4','target_architecture':'native_full_v5',
        'inherited':inherited,'adapted':adapted,'semantic_resets':resets,
        'discarded':sorted(set(source)-used),'new_initialized':fresh,'partial_tensors':[],
        'inactive_retained':['base.interaction_fusion.*'],'inactive_reason':'V5 Interaction is a separate dynamic residual',
        'allow_fresh_hand':allow_fresh_hand,'allow_fresh_detail':allow_fresh_detail,
        'hand_status':'training_evidence_supplied' if evidence_valid(evidence.get('hand')) else 'fresh_unverified',
        'source_detail_status':'fresh_unverified; new Native64 input, not legacy16 Detail',
        'warm_start_sources':warm_sources,'optimizer_resumed':False,'runtime_acceptance':'not_run',
        'legacy_native_schedules_adapted':schedules_adapted,
        'zero_outputs':['interaction_head.output_heads','face_controller.output_heads','hand_controller.output_heads',
            'face_controller.static_output','hand_controller.static_output'],
        'initialization_not_v41_equivalence':True,
        'tensor_elements':{'inherited':sum(template[k].numel() for k in inherited if torch.is_tensor(template[k])),
            'adapted':sum(template[k].numel() for k in adapted if torch.is_tensor(template[k])),
            'fresh':sum(template[k].numel() for k in fresh if torch.is_tensor(template[k]))}}
    return result
