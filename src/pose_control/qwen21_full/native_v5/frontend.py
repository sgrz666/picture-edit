"""Read shared modules directly. Never encode fused V4 then subtract Interaction."""
from dataclasses import replace,asdict
from copy import deepcopy
import torch
from torch.nn import functional as F
from ..data_runtime import build_condition_bundle
from ..dual_depth_condition import DualDepthConditionV3
from ..depth_condition_v2 import DepthConditionV2
from ..native_depth_v2 import Qwen21NativeDepthV2
from ..native_dual_depth import interaction_context
from ..native_dual_identity import _roi
from ..native_depth import _ResidualAdd
from .source_bank import NativeSourceBankV5


def identity_update(binder,identity,state,weights,pooling):
    b=state.person_tokens.shape[0];bank=isinstance(identity,NativeSourceBankV5)
    latent=None if bank else identity.source_latent.float();updates=[]
    sizes={'roi48':((8,4),(4,4)),'body8':((2,4),None),'bodyhead8':((2,3),(1,2))}[pooling]
    for p in range(2):
        qvalid=state.person_token_mask[:,p]&state.person_valid[:,p,None]
        source=identity.source_indices[:,p].clamp_min(0);device=state.person_tokens.device
        rows=torch.arange(b,device=device)
        context=[];valid=[]
        for region,size in enumerate(sizes):
            if size is None: continue
            if bank:
                pooled=[];availability=[]
                for row in range(b):
                    entry=identity.sources[int(identity.target_qwen_image_indices[row,p])]
                    index=int(identity.target_person_indices[row,p])
                    mask=(entry.person_mask if region==0 else entry.head_mask)[row:row+1,index]
                    values,v=_roi(entry.latent_map(row),mask,size);pooled.append(values);availability.append(v)
                values=torch.cat(pooled);v=torch.cat(availability)
            else:
                mask=(identity.person_mask if region==0 else identity.head_mask)[rows,source]
                values,v=_roi(latent,mask,size)
            yy,xx=torch.meshgrid(torch.linspace(-1,1,size[0],device=device),torch.linspace(-1,1,size[1],device=device),indexing='ij')
            position=torch.stack((xx,yy),-1).reshape(-1,2)
            context.append(binder.encoder(values)+binder.region.weight[region]+binder.slot.weight[p]+binder.position(position)[None])
            valid.append(v)
        query=state.person_tokens[:,p]+binder.query_binding(binder.slot.weight[p])[None,None]
        updates.append(binder.cross_attention(query,binder.context_projection(torch.cat(context,1)),torch.cat(valid,1),qvalid))
    return (torch.stack(updates,1)*weights[...,None]).sum(1)


def encode_frontend(runtime,batch,grid,base_options,options,identity):
    base=runtime.base;device=next(base.parameters()).device;kind=batch['native_scene_kind']
    if kind not in ('single','dual'): raise ValueError('Explicit V5 single/dual scene kind required')
    depth=batch.get('dual_depth' if kind=='dual' else 'depth_condition_v2')
    if isinstance(depth,list): depth=DepthConditionV2.stack(depth)
    if kind=='dual':
        if not isinstance(depth,DualDepthConditionV3): raise ValueError('V5 dual requires the original typed V3 geometry, including Depth-off')
        depth=depth.to(device).validate(target_token_hw=grid)
        for i,meta in enumerate(depth.metadata):
            if meta.get('geometry_config')!=asdict(base.config): raise ValueError('Dual geometry thresholds differ')
            if base_options.depth_mode=='full_geometry' and meta['mode']!='full_geometry': raise ValueError('No physical fallback from PNG')
            for suffix,person in (('',depth.person_a),('_b',depth.person_b)):
                row=batch['pose_provenance'+suffix][i]
                if any(row.get(k)!=v for k,v in {'record_sha256':meta['record_sha256'],'canvas_id':meta['canvas_id'],'person_id':person.metadata[i]['person_id']}.items()):
                    raise ValueError('Original dual Body/Depth source differs')
            key=meta['pair_id']+'@'+meta['canvas_id']
            if key in base.provenance and base.provenance[key]!=meta: raise ValueError('V5 dual provenance changed')
            base.provenance[key]=deepcopy(meta)
    elif depth is not None:
        if not isinstance(depth,DepthConditionV2): raise ValueError('Single depth must be explicit V2')
        depth=depth.to(device).validate(target_token_hw=grid)
        for row,meta in zip(batch['pose_provenance'],depth.metadata):
            if row['record_sha256']!=meta['pose_record_sha256'] or row['canvas_id']!=meta['pose_canvas_id']: raise ValueError('Single Pose/Depth differs')
    bundle=build_condition_bundle(base.geometry,batch,device=device,dtype=torch.float32)
    if kind=='dual' and not bundle.person_valid.all(): raise ValueError('Dual scene lost a person')
    if kind=='single' and bundle.person_valid[:,1].any(): raise ValueError('Single scene cannot replicate Person A into B')
    families=depth.model_family if kind=='dual' else batch['model_family_a'].to(device)[:,None]
    for p in range(2 if kind=='dual' else 1):
        family=families[:,p]
        if family.dtype!=torch.long or ((family<0)|(family>1)).any(): raise ValueError('Explicit model family required')
        binding=base.family_embedding(family)[:,None];prefix='person_'+('a' if p==0 else 'b')
        for suffix in ('global_tokens','joint_tokens'):
            setattr(bundle,prefix+'_'+suffix,getattr(bundle,prefix+'_'+suffix)+binding)
    state=base.geometry.reasoner(bundle);features=base.geometry.project_state(state,target_token_hw=grid)
    value=features.geometry;relation=torch.zeros_like(value);rv=torch.zeros(value.shape[:2],dtype=torch.bool,device=device)
    if base_options.depth_mode!='off' and base_options.depth_strength:
        if depth is None: raise ValueError('Enabled Depth has no condition')
        if kind=='dual': value=_ResidualAdd.apply(value,base.depth(value,depth,use_detail=base_options.detail)*base_options.depth_strength)
        else: value=Qwen21NativeDepthV2.forward(base.depth,features,depth,strength=base_options.depth_strength,use_detail=base_options.detail).geometry
    if kind=='dual':
        context=interaction_context(depth).flatten(2).transpose(1,2)
        projected=F.max_pool2d(depth.projected.any(1).float(),16).flatten(1)
        weights=F.avg_pool2d(depth.owner_weights.flatten(0,1),16).reshape(value.shape[0],2,-1)
        if base_options.interaction and runtime.strength.enabled('interaction',options):
            relation=features.interaction
            available=state.interaction_valid.clone()
            if base_options.raster and base_options.relation_strength:
                relation=relation+base.relation_raster(depth)*base_options.relation_strength
                available |= depth.camera_available
            if base_options.part_contact and base_options.relation_strength:
                relation=relation+base.relation_tokens(value,features.interaction,depth)*base_options.relation_strength
                available |= depth.part_distance_valid.flatten(1).any(1)|depth.contacts.valid.any(1)|(depth.part_contact[...,1]>0).flatten(1).any(1)|(depth.part_contact[...,3]>0).flatten(1).any(1)
            rv=(projected>0)&state.person_valid.all(1)[:,None]&available[:,None]
            # IMPORTANT: base.interaction_fusion is deliberately NOT called.
    else:
        owner=batch.get('single_owner_weights')
        if owner is None: raise ValueError('Single V5 requires declared visible ownership')
        owner=F.avg_pool2d(owner.to(device).float(),16).flatten(1)
        weights=torch.stack((owner,torch.zeros_like(owner)),1)
        context=value.new_zeros(*value.shape[:2],12);context[...,0]=owner;context[...,2]=owner;context[...,7]=owner
        projected=owner
        if isinstance(depth,DepthConditionV2):
            context[...,4]=F.avg_pool2d(depth.camera_z_valid.float(),16).flatten(1)
            context[...,10]=depth.context.camera_available.float()[:,None]
    if base_options.identity and base_options.identity_strength:
        if identity is None: raise ValueError('Reuse actual Source64 prefix for body identity')
        identity=identity.to(device).validate()
        bank=isinstance(identity,NativeSourceBankV5)
        if bank: identity.validate_target(batch)
        elif tuple(batch['identity_metadata'])!=identity.metadata: raise ValueError('V5 identity provenance differs from declared Source')
        if not torch.equal(identity.source_indices,batch['source_indices'].to(device)): raise ValueError('V5 body Source mapping differs')
        for i,im in enumerate(identity.metadata):
            target_ids=depth.metadata[i]['person_ids'] if kind=='dual' else [batch['pose_provenance'][i]['person_id']]
            if im['target_person_ids'][:len(target_ids)]!=target_ids: raise ValueError('Identity target slot differs')
            import hashlib,json
            key=hashlib.sha256(json.dumps(im,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            base.identity_provenance[key]=deepcopy(im)
        update=identity_update(base.identity_binder,identity,state,weights,runtime.config.identity_pooling)
        flags=identity.source_valid.gather(1,identity.source_indices.clamp_min(0)).float()*bundle.person_valid
        flags=flags[:,None].expand(-1,value.shape[1],-1)
        value=_ResidualAdd.apply(value,base.identity_fusion(value,update,torch.cat((context,flags),-1),weights.sum(1))*base_options.identity_strength)
    return replace(features,geometry=value,interaction=relation),relation,rv
