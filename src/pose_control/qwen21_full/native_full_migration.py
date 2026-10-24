"""Explicit encoder-only V3 -> V4 migration. Never called during reconstruction."""
from copy import deepcopy
import torch
from .native_dual_runtime import NativeDualRuntime
from .native_full_runtime import NativeFullRuntimeV4
from .native_depth_runtime import _check_state

FACE_NEW=('geometry_tokenizer.jaw_partial.','geometry_tokenizer.expression_partial.')
HAND_NEW=('null_pose','keypoint_availability.','modality_embedding.')
LEGACY_DISCARD=('condition_injector.','reasoner.','body_adapter.','face_adapter.','hand_adapter.','hand_detail.',
    'face_detail.','geometry.','body.','depth.','interface.','output_bridges.','qwen_bridges.','_extra_state')


def migrate_v3_to_v4(v3,face,*,face_prefix='face_preparer.',hand=None,hand_prefix='hand_preparer.',
        allow_fresh_hand=False,semantics,config=None,warm_start_attention=False,extra_discard_prefixes=()):
    if hand is None and not allow_fresh_hand:
        raise ValueError('Hand training source unconfirmed: provide verified Hand state or explicitly allow_fresh_hand')
    if semantics.get('policy')!='encoder-only': raise ValueError('Explicit encoder-only policy required')
    result=NativeFullRuntimeV4(NativeDualRuntime.from_checkpoint(v3),config,
        bps_convention=semantics.get('target_bps_convention'),face_parameter_basis=semantics.get('target_face_parameter_basis'))
    target=result.state_dict();inherited={};partial=[];discarded=[];reasons={}
    for k in v3['state_dict']: inherited['base.'+k]='V3 exact'
    used_sources={}
    for branch,saved,prefix,new in (('face',face,face_prefix,FACE_NEW),('hand',hand,hand_prefix,HAND_NEW)):
        if saved is None: continue
        source=saved.get('state_dict')
        if not isinstance(source,dict): raise ValueError('Explicit state_dict required for '+branch)
        expected={k[len(branch+'_preparer.'):]:v for k,v in target.items() if k.startswith(branch+'_preparer.') and not k[len(branch+'_preparer.'):].startswith(new)}
        selected={k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)}
        if selected.keys()!=expected.keys():
            raise ValueError(branch+' encoder keys differ; missing='+str(sorted(expected.keys()-selected.keys()))+' unknown='+str(sorted(selected.keys()-expected.keys())))
        _check_state(selected,expected)
        source_meta=saved.get('metadata',{})
        bps_keys=('basis_sha256','seed','coordinate','normalization','encoding','sides','mirror')
        old_bps=source_meta.get('bps_convention',{})
        new_bps=semantics.get('target_bps_convention',{})
        bps_compatible=all(k in old_bps and k in new_bps and old_bps[k]==new_bps[k] for k in bps_keys)
        old_face=source_meta.get('face_parameter_basis')
        new_face=semantics.get('target_face_parameter_basis')
        face_compatible=old_face is not None and new_face is not None and old_face==new_face
        used=set()
        for name,value in selected.items():
            dst=branch+'_preparer.'+name;src=prefix+name;used.add(src)
            reset=False
            if branch=='face' and name.startswith('geometry_tokenizer.parameter_encoder.'):
                reset=not face_compatible
            if branch=='hand' and name.startswith('structure.mano_bps.'):
                reset=not bps_compatible
            if reset:
                reasons[dst]='semantic mismatch or missing basis convention';discarded.append(branch+':'+src);continue
            if branch=='hand' and name=='spatial.0.weight':
                value=value.clone();value[:,22:23]=target[dst][:,22:23]
                partial.append({'source':branch+':'+src,'destination':dst,'inherited_channels':[0,22],
                    'reset_channels':[22,23],'reason':'old relative/unknown depth -> camera_z root-relative'})
            target[dst]=value;inherited[dst]=branch+':'+src
        if warm_start_attention:
            maps=({'self_norm.':'shared_block.self_norm.','shared_attention.':'shared_block.self_attention.','shared_ffn.':'shared_block.ffn.'}
                  if branch=='face' else {'self_norm.':'self_norm.','shared_attention.':'shared_attention.','shared_ffn.':'shared_ffn.'})
            adapter_prefix=prefix.replace('preparer.','adapter.')
            for dest_prefix,old_prefix in maps.items():
                expected_attention={k:v for k,v in target.items() if k.startswith(branch+'_controller.'+dest_prefix)}
                for dst,template in expected_attention.items():
                    src=adapter_prefix+old_prefix+dst[len(branch+'_controller.'+dest_prefix):]
                    if src not in source: raise ValueError('Missing optional shared attention key '+src)
                    _check_state({dst:source[src]},{dst:template})
                    target[dst]=source[src];inherited[dst]=branch+':'+src;used.add(src)
        for key in source:
            if key in used: continue
            if not key.startswith((*LEGACY_DISCARD,*extra_discard_prefixes,'face_preparer.','hand_preparer.')):
                raise ValueError('Undeclared legacy key; audit and declare a discard prefix: '+key)
            discarded.append(branch+':'+key)
        used_sources[branch]=sorted(used)
    # Strict load of a complete destination, with every fresh key explicitly listed.
    _check_state(target,result.state_dict());result.load_state_dict(target,strict=True)
    result.inheritance_report={'policy':'encoder-only','warm_start_attention':warm_start_attention,
        'inherited':inherited,'partial_tensors':partial,'discarded':sorted(set(discarded)),
        'new_initialized':sorted(set(target)-set(inherited)), 'semantic_resets':reasons,
        'semantics':deepcopy(semantics),'hand_status':'fresh_unverified' if hand is None else 'source_supplied_unvalidated',
        'allow_fresh_hand':allow_fresh_hand,'face_training_evidence':semantics.get('face_training_evidence'),
        'source_keys_used':used_sources,'optimizer_resumed':False,
        'zero_heads':['face_controller.output_heads','hand_controller.output_heads','face_controller.static_output','hand_controller.static_output'],
        'acceptance':'not_run'}
    return result
