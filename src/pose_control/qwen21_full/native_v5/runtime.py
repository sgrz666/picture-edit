"""V5 explicit frontend/Interaction/local streams; old runtimes stay independent."""
from dataclasses import asdict
from copy import deepcopy
import math,hashlib,json
import torch
from torch import nn
from ..native_full_runtime import NativeFullRuntimeV4
from ..native_dual_runtime import NativeDualRuntime,DualTrainability,ARCHITECTURE as BASE_ARCH
from ..native_dual_depth import DualControlOptions
from ..native_local_condition import PreparedNativeFullControlV4
from ..native_depth_runtime import _check_state
from ..depth_condition_v2 import DepthConditionV2
from .contracts import NativeConfigV5,NativeControlOptionsV5,PreparedNativeFullControlV5,ARCHITECTURE,BRANCHES
from .strength import NativeStrengthController
from .detail import NativeDetailAppearanceBridge
from .local import NativeFacePreparerV5,NativeHandPreparerV5,NativeLocalControllerV5,prepare_local_v5
from .frontend import encode_frontend


class NativeInteractionResidualHead(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.projection=nn.Linear(512,512);self.modulation=nn.Linear(512,512)
        self.layer=nn.Embedding(6,512);self.time=nn.Sequential(nn.Linear(1,512),nn.SiLU(),nn.Linear(512,512))
        self.fusion=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,1024),nn.SiLU(),nn.Linear(1024,512))
        self.output_heads=nn.ModuleList(nn.Linear(512,config.hidden_dim,bias=False) for _ in range(6))
        for head in self.output_heads: nn.init.zeros_(head.weight)

    def forward(self,query,prepared,*,layer,sigma):
        if layer not in self.config.injection_layers or not prepared.relation_valid.any(): return query.new_zeros(*query.shape[:2],self.config.hidden_dim)
        slot=self.config.injection_layers.index(layer);r=self.projection(prepared.relation)
        value=query+r+query*self.modulation(r).tanh()+self.layer.weight[slot]+self.time(sigma.float()[:,None])[:,None]
        return self.output_heads[slot](self.fusion(value))*prepared.relation_valid[...,None]


class NativeFullRuntimeV5(NativeFullRuntimeV4):
    def __init__(self,base=None,config=None,*,profile='joint',bps_convention=None,face_parameter_basis=None):
        # Do not construct V4 dynamic controllers or any DeepGen model.
        nn.Module.__init__(self);self.base=base or NativeDualRuntime();self.config=config or NativeConfigV5()
        if not isinstance(self.config,NativeConfigV5): raise ValueError('Explicit V5 config required')
        if (self.body.config.hidden_dim,self.body.config.num_layers,self.body.config.width)!=(self.config.hidden_dim,self.config.num_layers,512):
            raise ValueError('V5/Qwen Body dimensions differ')
        if self.body.config.num_groups!=6: raise ValueError('V5 requires the declared six Native Body groups')
        keys=('basis_sha256','seed','coordinate','normalization','encoding','sides','mirror')
        if bps_convention is not None and (not isinstance(bps_convention,dict) or any(k not in bps_convention for k in keys)): raise ValueError('Complete BPS convention required')
        if face_parameter_basis is not None and not isinstance(face_parameter_basis,dict): raise ValueError('Face basis must be a named record')
        self.bps_convention=deepcopy(bps_convention);self.face_parameter_basis=deepcopy(face_parameter_basis)
        self.face_preparer=NativeFacePreparerV5();self.hand_preparer=NativeHandPreparerV5(canonical=self.config.canonical_hand)
        self.face_controller=NativeLocalControllerV5(self.config);self.hand_controller=NativeLocalControllerV5(self.config,hand=True)
        self.detail_bridge=NativeDetailAppearanceBridge();self.interaction_head=NativeInteractionResidualHead(self.config)
        self.strength=NativeStrengthController(self.config)
        self.inheritance_report={};self.source_provenance={};self.local_provenance={};self.detail_provenance={}
        self.set_trainability(profile)

    def get_extra_state(self):
        return {'architecture':ARCHITECTURE,'config':self.config.to_dict(),'base':self.base.get_extra_state(),
            'local_schema':'native_local_condition_v4','reference_schema':'native_local_reference_v4',
            'detail_schema':'native_detail_references_v5','modalities':['identity','texture','geometry','detail'],
            'face_tokens':{'identity':4,'texture':8,'texture_delta':4,'geometry':4},'hand_tokens':[4,8],
            'interaction':'raw_reasoner+relations; separate_dynamic; legacy_fusion_inactive',
            'bps_convention':deepcopy(self.bps_convention),'face_parameter_basis':deepcopy(self.face_parameter_basis),
            'source_images_in_qwen':1,'qwen_tokens_added':0,'dtype':'fp32','sigma':'normalized; progress=1-sigma'}

    def set_extra_state(self,value):
        if value!=self.get_extra_state(): raise ValueError('V5 checkpoint architecture/schema differs')

    def _validate_sources(self,c,r,batch,identity):
        if len(c.metadata)!=len(r.metadata): raise ValueError('V5 reference batch differs')
        for i,(meta,source) in enumerate(zip(c.metadata,r.metadata)):
            for key in ('source_canvas_id','source_rgb_sha256','source_person_ids'):
                if meta.get(key)!=source.get(key): raise ValueError('V5 Source binding differs: '+key)
            if source.get('reference_count')!=r.arcface.shape[2] or len(source.get('reference_sources',()))!=r.arcface.shape[2]:
                raise ValueError('Declare all V5 local references')
            for p in range(2):
                if not c.person_valid[i,p]: continue
                src=int(c.source_indices[i,p]);provenance=batch['pose_provenance' if p==0 else 'pose_provenance_b'][i]
                if meta['source_person_ids'][src]!=meta['target_person_ids'][p] or provenance['person_id']!=meta['target_person_ids'][p]:
                    raise ValueError('V5 stable person mapping differs')
                if provenance['record_sha256']!=meta['driver_record_sha256'] or provenance['canvas_id']!=meta['target_canvas_id']:
                    raise ValueError('V5 local geometry differs from original Body')
                if c.expression_valid[i,p] and self.face_parameter_basis:
                    expected=self.face_parameter_basis.get('expression_basis_sha256')
                    if expected and meta.get('expression_basis_sha256',[None,None])[p]!=expected: raise ValueError('Face basis differs')
                for side in range(2):
                    if not c.mesh_valid[i,p,side]: continue
                    keys=('basis_sha256','seed','coordinate','normalization','encoding','sides','mirror')
                    raw=meta.get('bps')
                    if raw is None or any(k not in raw[p][side] for k in keys): raise ValueError('Valid BPS needs explicit convention')
                    convention={k:deepcopy(raw[p][side][k]) for k in keys}
                    if self.bps_convention is None: self.bps_convention=convention
                    if self.bps_convention!=convention: raise ValueError('BPS basis/frame/scale differs')
            typed=batch.get('dual_depth') if batch['native_scene_kind']=='dual' else batch.get('depth_condition_v2')
            if isinstance(typed,list): typed=DepthConditionV2.stack(typed)
            if typed is not None:
                dm=typed.metadata[i]
                if any(dm.get(k)!=meta.get(k) for k in ('camera_id','scale_group_id')): raise ValueError('V5 local/depth camera or scale differs')
            if identity is not None and (identity.metadata[i]['source_rgb_sha256']!=source['source_rgb_sha256'] or not torch.equal(identity.source_indices[i],c.source_indices[i])):
                raise ValueError('Local and body identities differ')
            if batch.get('supervised_target_rgb',False) and meta.get('counterfactual_geometry') and not meta.get('paired_rgb_compatible',False):
                raise ValueError('Counterfactual geometry cannot use original target RGB')
            # Multiple explicitly declared reference sets for one primary image
            # are legitimate. Bind each complete set instead of rejecting R1/R3
            # mixtures as a mutation of the primary image.
            fingerprint=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            key=source['source_rgb_sha256']+'@'+source['source_canvas_id']+'@'+fingerprint
            self.source_provenance[key]=deepcopy(source)
            target_key=meta['driver_record_sha256']+'@'+meta['target_canvas_id']+'@'+fingerprint+'@'+str(meta.get('body_record_sha256',''))
            compact=deepcopy(meta)
            compact['bps']=[[{k:v for k,v in side.items() if k!='basis'} for side in person] for person in meta.get('bps',())]
            if target_key in self.local_provenance and self.local_provenance[target_key]!=compact: raise ValueError('V5 local source changed')
            self.local_provenance[target_key]=compact

    def named_parameter_groups(self):
        groups={k:[] for k in ('inactive','face_encoder','hand_encoder','face_controller','hand_controller',
            'face_heads','hand_heads','detail_bridge','interaction_controller','interaction_heads')}
        for key,rows in self.base.named_parameter_groups().items():
            dest='inactive' if key=='interaction_fusion' else key
            groups.setdefault(dest,[]).extend(('base.'+n,p) for n,p in rows)
        for name,p in self.named_parameters():
            if name.startswith('base.'): continue
            first=name.split('.')[0]
            if first=='strength': key='strength_'+name.split('.')[2]
            elif first in ('face_preparer','hand_preparer'): key=first.split('_')[0]+'_encoder'
            elif first in ('face_controller','hand_controller'):
                family=first.split('_')[0];relative=name[len(first)+1:]
                head=relative.startswith(('output_heads.','static_output.'))
                active_head=relative.startswith('output_heads.') if self.config.local_mode=='dynamic' else relative.startswith('static_output.')
                key='inactive' if head and not active_head else family+('_heads' if head else '_controller')
            elif first=='interaction_head': key='interaction_heads' if name.startswith('interaction_head.output_heads.') else 'interaction_controller'
            elif first=='detail_bridge': key='detail_bridge'
            else: raise ValueError('Undeclared V5 parameter '+name)
            groups.setdefault(key,[]).append((name,p))
        return groups

    def set_trainability(self,profile='joint',*,overrides=None):
        profiles=('frozen','joint','all-native','face-heads','face-full','hand-heads','hand-full','detail-heads','detail-full','interaction-heads','interaction-full')
        if profile not in profiles: raise ValueError('Unknown V5 training profile')
        groups=self.named_parameter_groups();flags={k:False for k in groups}
        if profile in ('joint','all-native'):
            flags={k:k!='inactive' and (k!='geometry_shared' or profile=='all-native') for k in groups}
        elif profile!='frozen':
            family,extent=profile.split('-')
            if family=='detail':
                for k in ('face_heads','hand_heads','strength_detail'): flags[k]=True
                if extent=='full':
                    for k in ('detail_bridge','face_controller','hand_controller'): flags[k]=True
            else:
                flags[family+'_heads']=True
                for k in flags:
                    if k.startswith('strength_'+family): flags[k]=True
                if extent=='full':
                    flags[family+'_controller']=True
                    if family in ('face','hand'): flags[family+'_encoder']=True
                    else:
                        for k in ('geometry_dual','relation_raster','relation_tokens'): flags[k]=True
        for key,value in (overrides or {}).items():
            if key not in flags or type(value)!=bool or (key=='inactive' and value): raise ValueError('Invalid V5 group override '+key)
            flags[key]=value
        self.base.set_trainability(DualTrainability(**{k:(False if k=='interaction_fusion' else flags[k]) for k in DualTrainability.__dataclass_fields__}))
        for key,rows in groups.items():
            for _,p in rows: p.requires_grad_(flags[key])
        self.profile=profile;self.trainability=flags;self.train(self.training);return self

    def encode(self,batch,*,target_token_hw,options=None,base_options=None,identity=None,source_latent=None):
        options=options or NativeControlOptionsV5();base_options=base_options or DualControlOptions()
        if any(p.dtype!=torch.float32 for p in self.parameters()): raise ValueError('V5 controls must remain FP32')
        device=next(self.body.parameters()).device
        with torch.autocast(device_type=device.type,enabled=False):
            features,relation,rv=encode_frontend(self,batch,target_token_hw,base_options,options,identity)
            local=None
            if (options.face and options.face_strength>0) or (options.hand and options.hand_strength>0):
                c=batch['local_condition_v4'].to(device).validate();r=batch['local_references_v4'].to(device).validate()
                if not torch.equal(c.person_valid,features.person_valid): raise ValueError('V5 local person validity differs from Body')
                self._validate_sources(c,r,batch,identity)
                refs=batch.get('detail_references_v5');refs=None if refs is None else refs.to(device).validate()
                if refs is not None:
                    for m,cm in zip(refs.metadata,c.metadata):
                        for key in ('source_rgb_sha256','source_canvas_id','source_person_ids'):
                            if m.get(key)!=cm.get(key): raise ValueError('Detail Source provenance differs: '+key)
                        fingerprint=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                        self.detail_provenance[fingerprint]=deepcopy(m)
                if source_latent is not None:
                    sh,sw=batch['source_token_hw']
                    if source_latent.shape!=(features.geometry.shape[0],sh*sw,64): raise ValueError('Use the actual flattened primary64 Source latent')
                    source_latent=source_latent.to(device).float().transpose(1,2).reshape(-1,64,sh,sw)
                local=prepare_local_v5(self.face_preparer,self.hand_preparer,self.detail_bridge,c,r,refs,source_latent,options,self.strength,self.config)
        metadata=tuple({'scene_kind':batch['native_scene_kind'],'architecture':ARCHITECTURE,'options':asdict(options),
            'body_control_source':'geometry_depth_identity_without_interaction'} for _ in range(features.geometry.shape[0]))
        core=PreparedNativeFullControlV4(features,None,metadata)
        return PreparedNativeFullControlV5(core,relation,rv,local,metadata).validate()

    def checkpoint_payload(self,*,metadata):
        return {'architecture':ARCHITECTURE,'runtime_config':self.get_extra_state(),'profile':self.profile,
            'trainability':deepcopy(self.trainability),'base_trainability':asdict(self.base.trainability),
            'state_dict':self.state_dict(),'metadata':{**deepcopy(metadata),'inheritance':deepcopy(self.inheritance_report),
                'appearance_sources':deepcopy(self.source_provenance),'local_sources':deepcopy(self.local_provenance),
                'detail_sources':deepcopy(self.detail_provenance),'base_provenance':{
                    'provenance':deepcopy(self.base.provenance),'identity_provenance':deepcopy(self.base.identity_provenance)}}}

    @classmethod
    def from_checkpoint(cls,saved):
        if saved.get('architecture')!=ARCHITECTURE: raise ValueError('V5 exact rebuild only; explicitly migrate V4')
        cfg=saved['runtime_config'];state=saved['state_dict']
        base=NativeDualRuntime.from_checkpoint({'architecture':BASE_ARCH,'runtime_config':cfg['base'],
            'state_dict':{k[5:]:v for k,v in state.items() if k.startswith('base.')},'trainability':saved['base_trainability'],
            'metadata':deepcopy(saved.get('metadata',{}).get('base_provenance',{}))})
        result=cls(base,NativeConfigV5.from_dict(cfg['config']),profile=saved['profile'],
            bps_convention=cfg['bps_convention'],face_parameter_basis=cfg['face_parameter_basis'])
        result.set_trainability(saved['profile'],overrides=saved['trainability'])
        if result.get_extra_state()!=cfg: raise ValueError('V5 header differs')
        _check_state(state,result.state_dict());result.load_state_dict(state,strict=True)
        result.inheritance_report=deepcopy(saved.get('metadata',{}).get('inheritance',{}))
        result.source_provenance=deepcopy(saved.get('metadata',{}).get('appearance_sources',{}))
        result.local_provenance=deepcopy(saved.get('metadata',{}).get('local_sources',{}))
        result.detail_provenance=deepcopy(saved.get('metadata',{}).get('detail_sources',{}))
        return result
