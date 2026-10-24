"""Explicit V4 runtime, sharing one V3 Body/Depth/Identity parameter set.

Architecture implementation only; real-asset and Qwen acceptance is deferred.
"""
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F

from .native_dual_runtime import NativeDualRuntime, DualTrainability, ARCHITECTURE as V3_ARCHITECTURE
from .native_dual_depth import DualControlOptions
from .native_dual_identity import _roi
from .native_depth_runtime import _check_state
from .native_depth_v2 import Qwen21NativeDepthV2
from .depth_condition_v2 import DepthConditionV2, _TensorRecord, _tensor
from .data_runtime import build_condition_bundle
from .native_depth import _ResidualAdd
from .native_face import NativeFacePreparer
from .native_hand import NativeHandPreparer
from .native_local_condition import NativeLocalConditionV4, NativeLocalReferencesV4, PreparedNativeFullControlV4
from .native_local_control import NativeLocalConfigV4, NativeFaceController, NativeHandController, prepare_local

ARCHITECTURE='native_full_v4'


@dataclass(frozen=True)
class FullControlOptionsV4:
    face: bool = True
    hand: bool = True
    face_strength: float = 1.
    hand_strength: float = 1.

    def __post_init__(self):
        if type(self.face) is not bool or type(self.hand) is not bool:
            raise ValueError('Local switches must be bool')
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in (self.face_strength,self.hand_strength)):
            raise ValueError('Local strengths must be finite and nonnegative')


@dataclass(frozen=True)
class NativeFullIdentityV4(_TensorRecord):
    source_latent: torch.Tensor
    person_mask: torch.Tensor
    head_mask: torch.Tensor
    source_indices: torch.Tensor
    source_valid: torch.Tensor
    person_valid: torch.Tensor
    metadata: tuple[dict,...]

    def validate(self):
        b,c,h,w=self.source_latent.shape;d=self.source_latent.device
        _tensor(self.source_latent,(b,64,h,w),d,'source_latent')
        for name,shape in (('person_mask',(b,2,1,h*16,w*16)),('head_mask',(b,2,1,h*16,w*16)),
                           ('source_valid',(b,2)),('person_valid',(b,2))):
            _tensor(getattr(self,name),shape,d,name,boolean=True)
        if self.source_indices.shape!=(b,2) or self.source_indices.dtype!=torch.long or self.source_indices.device!=d:
            raise ValueError('Identity mapping requires int64 [B,2]')
        if not self.person_valid[:,0].all() or self.person_valid[:,1].any():
            raise ValueError('This identity contract is the explicit single path; dual uses V3 identity unchanged')
        if ((self.source_indices[:,0]<0)|(self.source_indices[:,0]>1)).any():
            raise ValueError('Active identity source index is missing')
        if (self.head_mask & ~self.person_mask).any() or (self.person_mask[:,0]&self.person_mask[:,1]).any():
            raise ValueError('Identity masks conflict')
        rows=torch.arange(b,device=d);src=self.source_indices[:,0]
        if not self.source_valid[rows,src].all() or not self.person_mask[rows,src].flatten(1).any(1).all():
            raise ValueError('Active person requires a real source ROI')
        if len(self.metadata)!=b: raise ValueError('Identity provenance batch differs')
        for i,meta in enumerate(self.metadata):
            if meta.get('role')!='source_only' or not meta.get('source_rgb_sha256') or not meta.get('source_canvas_id'):
                raise ValueError('Single identity source provenance required')
            if meta['source_person_ids'][int(src[i])]!=meta['target_person_ids'][0]:
                raise ValueError('Single source identity mapping differs')
        return self


def _single_identity(binder,identity,state,owner):
    """Reuse every V3 binder parameter; B has no ROI, token or residual."""
    identity.validate()
    b=state.person_tokens.shape[0];rows=torch.arange(b,device=owner.device)
    src=identity.source_indices[:,0]
    context=[];valid=[]
    for region,size,mask in ((0,(8,4),identity.person_mask),(1,(4,4),identity.head_mask)):
        tokens,mask_valid=_roi(identity.source_latent,mask[rows,src],size)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,size[0],device=owner.device),torch.linspace(-1,1,size[1],device=owner.device),indexing='ij')
        position=torch.stack((xx,yy),-1).reshape(-1,2)
        context.append(binder.encoder(tokens)+binder.region.weight[region]+binder.slot.weight[0]+binder.position(position)[None])
        valid.append(mask_valid)
    query=state.person_tokens[:,0]+binder.query_binding(binder.slot.weight[0])[None,None]
    return binder.cross_attention(query,binder.context_projection(torch.cat(context,1)),torch.cat(valid,1),state.person_token_mask[:,0])*owner[...,None]


class NativeFullRuntimeV4(nn.Module):
    def __init__(self,base=None,config=None,*,profile='joint',bps_convention=None,face_parameter_basis=None):
        super().__init__()
        self.base=base or NativeDualRuntime()
        self.config=config or NativeLocalConfigV4()
        if (self.body.config.hidden_dim!=self.config.hidden_dim or self.body.config.num_layers!=self.config.num_layers
                or self.body.config.width!=512):
            raise ValueError('V4 local/body Hidden and layer contracts differ')
        keys=('basis_sha256','seed','coordinate','normalization','encoding','sides','mirror')
        if bps_convention is not None and (not isinstance(bps_convention,dict) or any(k not in bps_convention for k in keys)):
            raise ValueError('Declare the complete BPS convention')
        if face_parameter_basis is not None and not isinstance(face_parameter_basis,dict): raise ValueError('Face parameter basis must be a named convention record')
        self.bps_convention=None if bps_convention is None else {k:deepcopy(bps_convention[k]) for k in keys}
        self.face_parameter_basis=deepcopy(face_parameter_basis)
        self.face_preparer=NativeFacePreparer()
        self.hand_preparer=NativeHandPreparer()
        self.face_controller=NativeFaceController(self.config)
        self.hand_controller=NativeHandController(self.config)
        self.inheritance_report={}
        self.source_provenance={}
        self.local_provenance={}
        self.set_trainability(profile)

    @property
    def body(self):
        return self.base.body

    def get_extra_state(self):
        return {'architecture':ARCHITECTURE,'local_condition':'native_local_condition_v4',
            'local_reference':'native_local_reference_v4','base':self.base.get_extra_state(),
            'config':self.config.to_dict(),'face_tokens':[4,8,4],'hand_tokens':[4,8],
            'bps_convention':deepcopy(self.bps_convention),'face_parameter_basis':deepcopy(self.face_parameter_basis),
            'bps':'1024; wrist_local; standard_same_beta_hand_length; nearest_vertex; explicit_basis_sha',
            'hand_depth':'asinh((camera_z-wrist_z)/standard_height)',
            'rot6d':'first_two_columns','source_only':True,'qwen_tokens_added':0}

    def set_extra_state(self,value):
        if value!=self.get_extra_state(): raise ValueError('V4 checkpoint architecture/schema differs')

    def named_parameter_groups(self):
        groups={k:[('base.'+n,p) for n,p in rows] for k,rows in self.base.named_parameter_groups().items()}
        for branch in ('face','hand'):
            groups[branch+'_encoder']=list((branch+'_preparer.'+n,p) for n,p in getattr(self,branch+'_preparer').named_parameters())
            controller=getattr(self,branch+'_controller')
            groups[branch+'_controller']=[];groups[branch+'_heads']=[]
            for n,p in controller.named_parameters():
                key=branch+('_heads' if n.startswith(('output_heads.','static_output.')) else '_controller')
                groups[key].append((branch+'_controller.'+n,p))
        return groups

    def set_trainability(self,profile='joint',*,overrides=None):
        profiles=('face-heads','face-full','hand-heads','hand-full','joint','all-native','frozen')
        if profile not in profiles: raise ValueError('Unknown V4 trainability profile')
        groups=self.named_parameter_groups()
        flags={key:False for key in groups}
        if profile in ('joint','all-native'):
            flags.update({key:True for key in flags})
            flags['geometry_shared']=profile=='all-native'
        elif profile!='frozen':
            branch,extent=profile.split('-')
            for suffix in ('heads',) if extent=='heads' else ('encoder','controller','heads'):
                flags[branch+'_'+suffix]=True
        for key,value in (overrides or {}).items():
            if key not in flags or type(value) is not bool: raise ValueError('Unknown/invalid freeze group '+key)
            flags[key]=value
        self.base.set_trainability(DualTrainability(**{k:flags[k] for k in DualTrainability.__dataclass_fields__}))
        for key,rows in groups.items():
            for _,p in rows: p.requires_grad_(flags[key])
        self.profile=profile;self.trainability=flags
        self.train(self.training)
        return self

    def optimizer_groups(self,*,default_lr,learning_rates=None):
        """Return groups only; caller explicitly constructs the existing AdamW."""
        rates=learning_rates or {}
        if set(rates)-set(self.named_parameter_groups()): raise ValueError('Unknown learning-rate group')
        output=[]
        for name,rows in self.named_parameter_groups().items():
            params=[p for _,p in rows if p.requires_grad]
            lr=rates.get(name,default_lr)
            if not math.isfinite(lr) or lr<=0: raise ValueError('Learning rates must be positive')
            if params: output.append({'name':name,'params':params,'lr':lr})
        return output

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'trainability'):
            for branch in ('face','hand'):
                getattr(self,branch+'_preparer').train(mode and self.trainability[branch+'_encoder'])
                getattr(self,branch+'_controller').train(mode and (self.trainability[branch+'_controller'] or self.trainability[branch+'_heads']))
        return self

    def encode(self,batch,*,target_token_hw,options=None,base_options=None,identity=None):
        options=options or FullControlOptionsV4();base_options=base_options or DualControlOptions()
        if any(p.dtype!=torch.float32 for p in self.parameters()): raise ValueError('V4 geometry/control parameters must remain FP32')
        device=next(self.body.parameters()).device
        kind=batch.get('native_scene_kind')
        if kind not in ('single','dual'): raise ValueError('V4 requires explicit single/dual scene kind')
        if kind=='dual':
            features=self.base.encode(batch,target_token_hw=target_token_hw,options=base_options,identity=identity)
        else:
            features=self._encode_single(batch,target_token_hw,base_options,identity,device)
        face=options.face and options.face_strength>0;hand=options.hand and options.hand_strength>0
        local=None
        if face or hand:
            condition=batch.get('local_condition_v4');references=batch.get('local_references_v4')
            if not isinstance(condition,NativeLocalConditionV4) or not isinstance(references,NativeLocalReferencesV4):
                raise ValueError('Enabled local branches require explicit V4 geometry and source references')
            condition=condition.to(device).validate();references=references.to(device).validate()
            if not torch.equal(condition.person_valid,features.person_valid): raise ValueError('Local person validity differs from Body')
            self._validate_sources(condition,references,batch,identity)
            with torch.autocast(device_type=device.type,enabled=False):
                local=prepare_local(self.face_preparer,self.hand_preparer,condition,references,
                    face_enabled=face,hand_enabled=hand,config=self.config)
        sf=sh=None
        if local is not None and self.config.local_mode=='static':
            with torch.autocast(device_type=device.type,enabled=False):
                sf=self.face_controller.static_delta(local,target_hw=target_token_hw,regions=(0,1)) if face else None
                sh=self.hand_controller.static_delta(local,target_hw=target_token_hw,regions=(2,3,4,5)) if hand else None
        return PreparedNativeFullControlV4(features,local,tuple({'scene_kind':kind,'options':asdict(options)} for _ in range(features.geometry.shape[0])),sf,sh).validate()

    def _validate_sources(self,condition,references,batch,identity):
        if len(references.metadata)!=len(condition.metadata): raise ValueError('Reference batch differs')
        rows=batch.get('pose_provenance');rows=rows if isinstance(rows,list) else [rows]
        for i,(meta,source) in enumerate(zip(condition.metadata,references.metadata)):
            for key in ('source_canvas_id','source_rgb_sha256','source_person_ids'):
                if meta.get(key)!=source.get(key): raise ValueError('Local source binding differs: '+key)
            if source.get('reference_count')!=references.arcface.shape[2]: raise ValueError('Undeclared reference count')
            if len(source.get('reference_sources',()))!=references.arcface.shape[2]: raise ValueError('Declare every reference source')
            if rows[i] is None or meta['driver_record_sha256']!=rows[i]['record_sha256'] or meta['target_canvas_id']!=rows[i]['canvas_id']:
                raise ValueError('Local geometry and Body pose/canvas provenance differ')
            for person in range(2):
                if not condition.person_valid[i,person]: continue
                source_index=int(condition.source_indices[i,person])
                if meta['source_person_ids'][source_index]!=meta['target_person_ids'][person]: raise ValueError('Local stable source mapping differs')
                provenance=batch.get('pose_provenance' if person==0 else 'pose_provenance_b')
                provenance=provenance if isinstance(provenance,list) else [provenance]
                if provenance[i].get('person_id')!=meta['target_person_ids'][person]: raise ValueError('Local target person slot differs from Body')
                if provenance[i].get('record_sha256')!=meta['driver_record_sha256']: raise ValueError('Both local persons must share the same Driver scene')
                if self.face_parameter_basis and condition.expression_valid[i,person]:
                    expected=self.face_parameter_basis.get('expression_basis_sha256')
                    if expected and meta.get('expression_basis_sha256',[None,None])[person]!=expected:
                        raise ValueError('Face parameter basis differs from inherited/configured encoder')
                for side in range(2):
                    if not condition.mesh_valid[i,person,side]: continue
                    raw=meta.get('bps')
                    if raw is None: raise ValueError('Valid BPS needs explicit basis/frame/scale provenance')
                    keys=('basis_sha256','seed','coordinate','normalization','encoding','sides','mirror')
                    spec=raw[person][side]
                    if any(k not in spec for k in keys): raise ValueError('Incomplete BPS convention')
                    convention={k:deepcopy(spec[k]) for k in keys}
                    if self.bps_convention is None: self.bps_convention=convention
                    if convention!=self.bps_convention: raise ValueError('BPS basis/frame/normalization differs from checkpoint')
            typed=batch.get('dual_depth') if batch.get('native_scene_kind')=='dual' else batch.get('depth_condition_v2')
            if isinstance(typed,list): typed=typed[i]
            if typed is not None:
                dm=typed.metadata[i if len(typed.metadata)>i else 0]
                if any(dm.get(k)!=meta.get(k) for k in ('camera_id','scale_group_id')): raise ValueError('Local/Depth camera or scale differs')
            if identity is not None:
                if source.get('source_rgb_sha256')!=identity.metadata[i].get('source_rgb_sha256') or not torch.equal(condition.source_indices[i],identity.source_indices[i]):
                    raise ValueError('Local/full-body source identity differs')
            if batch.get('supervised_target_rgb',False) and meta.get('counterfactual_geometry') and not meta.get('paired_rgb_compatible',False):
                raise ValueError('Counterfactual source expression cannot use original B RGB supervision')
            key=source['source_rgb_sha256']+'@'+source['source_canvas_id']
            if key in self.source_provenance and source!=self.source_provenance[key]: raise ValueError('Source appearance provenance changed')
            self.source_provenance[key]=deepcopy(source)
            target_key=meta['driver_record_sha256']+'@'+meta['target_canvas_id']
            compact={k:deepcopy(v) for k,v in meta.items() if k!='bps'}
            compact['bps']=[[{k:v for k,v in side.items() if k!='basis'} for side in person] for person in meta.get('bps',())]
            if target_key in self.local_provenance and compact!=self.local_provenance[target_key]: raise ValueError('Local geometry provenance changed')
            self.local_provenance[target_key]=compact

    def _encode_single(self,batch,grid,options,identity,device):
        with torch.autocast(device_type=device.type,enabled=False):
            bundle=build_condition_bundle(self.base.geometry,batch,device=device,dtype=torch.float32)
            if bundle.person_valid[:,1].any(): raise ValueError('Single path must not contain Person B')
            family=batch.get('model_family_a')
            if family is None: raise ValueError('Single V4 requires explicit model_family_a (SMPL-X=0, SMPL=1)')
            family=family.to(device=device,dtype=torch.long)
            if family.shape!=(bundle.person_valid.shape[0],) or ((family<0)|(family>1)).any(): raise ValueError('Invalid model family')
            rows=batch.get('pose_provenance');rows=rows if isinstance(rows,list) else [rows]
            if len(rows)!=bundle.person_valid.shape[0] or any(not isinstance(row,dict) or not row.get('record_sha256') or not row.get('canvas_id') for row in rows):
                raise ValueError('Single Body21 provenance required even with Depth/local off')
            binding=self.base.family_embedding(family)[:,None]
            bundle.person_a_global_tokens=bundle.person_a_global_tokens+binding
            bundle.person_a_joint_tokens=bundle.person_a_joint_tokens+binding
            state=self.base.geometry.reasoner(bundle)
            features=self.base.geometry.project_state(state,target_token_hw=grid)
            depth=batch.get('depth_condition_v2')
            if isinstance(depth,list): depth=DepthConditionV2.stack(depth)
            if options.depth_mode!='off' and options.depth_strength:
                if not isinstance(depth,DepthConditionV2): raise ValueError('Single V4 depth requires DepthConditionV2')
                depth=depth.to(device)
                rows=batch.get('pose_provenance');rows=rows if isinstance(rows,list) else [rows]
                for row,meta in zip(rows,depth.metadata):
                    if row['record_sha256']!=meta['pose_record_sha256'] or row['canvas_id']!=meta['pose_canvas_id']:
                        raise ValueError('Single Depth/Pose provenance differs')
                features=Qwen21NativeDepthV2.forward(self.base.depth,features,depth,strength=options.depth_strength,use_detail=options.detail)
            if options.identity and options.identity_strength:
                if not isinstance(identity,NativeFullIdentityV4): raise ValueError('Single V4 identity input required')
                identity=identity.to(device).validate()
                declared=batch.get('identity_metadata')
                if not isinstance(declared,list) or tuple(declared)!=identity.metadata: raise ValueError('Single identity must bind the declared Source batch')
                for row,im in zip(rows,identity.metadata):
                    if im['target_person_ids'][0]!=row.get('person_id'): raise ValueError('Single identity target person differs')
                    key='single:'+im['source_rgb_sha256']+'@'+im['source_canvas_id']
                    if key in self.base.identity_provenance and self.base.identity_provenance[key]!=im: raise ValueError('Single identity source changed')
                    self.base.identity_provenance[key]=deepcopy(im)
                owner=batch.get('single_owner_weights')
                if owner is None or owner.shape!=(features.geometry.shape[0],1,grid[0]*16,grid[1]*16): raise ValueError('Single identity needs exact visible owner mask')
                owner=owner.to(device).float()
                if not torch.isfinite(owner).all() or ((owner<0)|(owner>1)).any(): raise ValueError('Owner coverage must be finite0..1')
                owner=F.avg_pool2d(owner,16).flatten(1)
                update=_single_identity(self.base.identity_binder,identity,state,owner)
                context=features.geometry.new_zeros(*features.geometry.shape[:2],14)
                context[...,0]=owner;context[...,2]=owner;context[...,7]=owner;context[...,12]=1
                if isinstance(depth,DepthConditionV2):
                    context[...,4]=F.avg_pool2d(depth.camera_z_valid.to(device).float(),16).flatten(1)
                    context[...,10]=depth.context.camera_available.to(device).float()[:,None]
                features=replace(features,geometry=_ResidualAdd.apply(features.geometry,
                    self.base.identity_fusion(features.geometry,update,context,owner)*options.identity_strength))
            return features

    def checkpoint_payload(self,*,metadata):
        return {'architecture':ARCHITECTURE,'runtime_config':self.get_extra_state(),
            'profile':self.profile,'trainability':deepcopy(self.trainability),'state_dict':self.state_dict(),
            'metadata':{**deepcopy(metadata),'inheritance':deepcopy(self.inheritance_report),'appearance_sources':deepcopy(self.source_provenance),
                        'local_sources':deepcopy(self.local_provenance),
                        'base_provenance':{'provenance':deepcopy(self.base.provenance),'identity_provenance':deepcopy(self.base.identity_provenance),
                                           'inheritance':deepcopy(self.base.inheritance_report)}}}

    @classmethod
    def from_checkpoint(cls,saved):
        if saved.get('architecture')!=ARCHITECTURE: raise ValueError('V4 exact rebuild only; explicitly migrate V3')
        cfg=saved['runtime_config'];prefix='base.'
        base=NativeDualRuntime.from_checkpoint({'architecture':V3_ARCHITECTURE,'runtime_config':cfg['base'],
            'state_dict':{k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)},
            'metadata':deepcopy(saved.get('metadata',{}).get('base_provenance',{})),
            'trainability':{k:saved['trainability'][k] for k in DualTrainability.__dataclass_fields__}})
        result=cls(base,NativeLocalConfigV4.from_dict(cfg['config']),profile=saved['profile'],
            bps_convention=cfg.get('bps_convention'),face_parameter_basis=cfg.get('face_parameter_basis'))
        result.set_trainability(saved['profile'],overrides=saved['trainability'])
        if result.get_extra_state()!=cfg: raise ValueError('V4 runtime header differs')
        _check_state(saved['state_dict'],result.state_dict());result.load_state_dict(saved['state_dict'],strict=True)
        result.inheritance_report=deepcopy(saved.get('metadata',{}).get('inheritance',{}))
        result.source_provenance=deepcopy(saved.get('metadata',{}).get('appearance_sources',{}))
        result.local_provenance=deepcopy(saved.get('metadata',{}).get('local_sources',{}))
        return result
