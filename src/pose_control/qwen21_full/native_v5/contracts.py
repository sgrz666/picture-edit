"""V5 raw Source/Training sidecars and gradient-preserving prepared controls."""
from dataclasses import dataclass,asdict,replace
import math
import torch
from ..depth_condition_v2 import _TensorRecord,_tensor
from ..native_local_condition import PreparedNativeFullControlV4
from ..native_local_control import NativeLocalConfigV4

ARCHITECTURE='native_full_v5'
DETAIL_SCHEMA='native_detail_references_v5'
SUPERVISION_SCHEMA='native_training_supervision_v5'
MODALITIES=('identity','texture','geometry','detail')
BRANCHES=('body','interaction','detail','face_identity','face_texture','face_geometry','hand_geometry','hand_texture')


def valid_five_points(points):
    if points.shape!=(5,2) or not torch.isfinite(points).all(): return False
    distances=torch.cdist(points.float(),points.float())+torch.eye(5,device=points.device)*1e8
    singular=torch.linalg.svdvals(points.float()-points.float().mean(0))
    return bool(distances.amin()>1e-4 and singular[0]>1e-4 and singular[1]>singular[0]*1e-5)


@dataclass(frozen=True)
class NativeConfigV5(NativeLocalConfigV4):
    strength_preset: str='constant'
    schedules: tuple=()
    stage_modulation: str='none'
    canonical_hand: bool=False
    hand_fusion: str='parallel'
    identity_pooling: str='roi48'
    feather_px: int=8

    def __post_init__(self):
        super().__post_init__()
        if self.strength_preset not in ('constant','legacy_full') or self.stage_modulation not in ('none','film_rank64'):
            raise ValueError('Unknown V5 strength/stage profile')
        if type(self.canonical_hand)!=bool or self.hand_fusion not in ('parallel','sequential2'):
            raise ValueError('Explicit canonical-hand/fusion mode required')
        if self.identity_pooling not in ('roi48','body8','bodyhead8') or type(self.feather_px)!=int or not 0<=self.feather_px<=32:
            raise ValueError('Unknown identity budget or feather radius')
        if len({x[0] for x in self.schedules})!=len(self.schedules) or any(x[0] not in BRANCHES for x in self.schedules):
            raise ValueError('Duplicate/unknown branch schedule')
        constant=((0.,1.),(1.,1.))
        if tuple(map(tuple,self.face_schedule))!=constant or tuple(map(tuple,self.hand_schedule))!=constant:
            raise ValueError('V5 uses named schedules; legacy face_schedule/hand_schedule fields must stay constant')

    @classmethod
    def from_dict(cls,value):
        return cls(**{**value,'injection_layers':tuple(value['injection_layers']),
            'face_schedule':tuple(map(tuple,value['face_schedule'])),'hand_schedule':tuple(map(tuple,value['hand_schedule'])),
            'schedules':tuple((x[0],x[1]) for x in value.get('schedules',()))})


@dataclass(frozen=True)
class NativeControlOptionsV5:
    face: bool=True
    hand: bool=True
    detail: bool=True
    interaction: bool=True
    body_strength: float=1.
    interaction_strength: float=1.
    detail_strength: float=1.
    face_strength: float=1.
    hand_strength: float=1.
    face_identity_gain: float=1.
    face_texture_gain: float=1.
    face_geometry_gain: float=1.
    hand_geometry_gain: float=1.
    hand_texture_gain: float=1.

    def validate(self):
        for name,value in asdict(self).items():
            if name in ('face','hand','detail','interaction'):
                if type(value)!=bool: raise ValueError('Control switches require bool')
            elif type(value) not in (float,int) or not math.isfinite(value) or value<0:
                raise ValueError('Finite nonnegative V5 strength required: '+name)
        return self

    def __post_init__(self): self.validate()
    def to(self,*args,**kwargs): return replace(self)
    def index_select(self,indices):
        if indices.dtype!=torch.long: raise ValueError('Control indices require int64')
        return replace(self)
    @classmethod
    def stack(cls,records):
        if not records or any(r!=records[0] for r in records): raise ValueError('Bucket batches by uniform runtime controls')
        return replace(records[0])

    def strength(self,branch):
        if branch=='body': return self.body_strength
        if branch=='interaction': return self.interaction_strength if self.interaction else 0.
        if branch=='detail': return self.detail_strength if self.detail else 0.
        family=branch.split('_')[0]
        return getattr(self,branch+'_gain')*getattr(self,family+'_strength') if getattr(self,family) else 0.


@dataclass(frozen=True)
class NativeDetailReferencesV5(_TensorRecord):
    boxes: torch.Tensor                 # [B,6,4], primary Source canvas, unclipped
    support: torch.Tensor               # [B,6,Hs,Ws], bool, visible ownership
    valid: torch.Tensor                 # [B,6]
    source_indices: torch.Tensor        # [B,2], target->physical Source
    metadata: tuple[dict,...]

    def validate(self):
        b=self.boxes.shape[0];d=self.boxes.device
        _tensor(self.boxes,(b,6,4),d,'Source detail boxes')
        _tensor(self.valid,(b,6),d,'Source detail validity',boolean=True)
        _tensor(self.support,(b,6,*self.support.shape[-2:]),d,'Source detail support',boolean=True)
        if self.source_indices.shape!=(b,2) or self.source_indices.dtype!=torch.long or self.source_indices.device!=d:
            raise ValueError('Detail source mapping requires int64 [B,2]')
        if ((self.source_indices<-1)|(self.source_indices>1)).any() or (self.source_indices[:,0]<0).any():
            raise ValueError('Source indices outside physical slots; only inactive B may use -1')
        if ((self.boxes[...,2:]<=self.boxes[...,:2]).any(-1)&self.valid).any(): raise ValueError('Valid Source ROI needs positive area')
        if (self.valid & ~self.support.flatten(2).any(-1)).any(): raise ValueError('Valid Source detail has no visible support')
        if len(self.metadata)!=b: raise ValueError('Detail metadata batch differs')
        for m in self.metadata:
            if m.get('schema')!=DETAIL_SCHEMA or m.get('role')!='primary_source_only': raise ValueError('Source-only Detail provenance required')
            for key in ('source_rgb_sha256','source_canvas_id'):
                if not m.get(key): raise ValueError('Missing detail '+key)
            if len(m.get('source_person_ids',[]))!=2: raise ValueError('Stable Source person IDs required')
        return self


@dataclass(frozen=True)
class NativeTrainingSupervisionV5(_TensorRecord):
    weights: torch.Tensor              # [B,6,Ht,Wt], final visible regions
    auxiliary_valid: torch.Tensor      # [B,2]
    landmarks5: torch.Tensor           # [B,2,5,2], target canvas pixels
    paired: torch.Tensor               # [B]
    metadata: tuple[dict,...]

    def validate(self):
        b=self.weights.shape[0];d=self.weights.device
        _tensor(self.weights,(b,6,*self.weights.shape[-2:]),d,'Training region weights')
        _tensor(self.landmarks5,(b,2,5,2),d,'Training alignment')
        _tensor(self.auxiliary_valid,(b,2),d,'Auxiliary eligibility',boolean=True)
        _tensor(self.paired,(b,),d,'Paired supervision',boolean=True)
        if ((self.weights<0)|(self.weights>1)).any() or (self.weights.sum(1)>1.00001).any(): raise ValueError('Training masks duplicate ownership')
        if (self.auxiliary_valid & ~self.paired[:,None]).any(): raise ValueError('Unpaired target cannot provide auxiliary supervision')
        if (self.auxiliary_valid & ~(self.weights[:,:2].flatten(2).sum(-1)>0)).any(): raise ValueError('Auxiliary face must have visible owned support')
        for row,person in self.auxiliary_valid.nonzero().tolist():
            if not valid_five_points(self.landmarks5[row,person]): raise ValueError('Auxiliary face alignment is degenerate')
        if len(self.metadata)!=b or any(m.get('schema')!=SUPERVISION_SCHEMA or m.get('role')!='training_only' for m in self.metadata):
            raise ValueError('Training-only supervision provenance required')
        for m in self.metadata:
            if any(not m.get(k) for k in ('driver_record_sha256','driver_rgb_sha256','target_canvas_id')) or len(m.get('target_person_ids',[]))!=2:
                raise ValueError('Training supervision needs original Driver/Canvas/stable-person fingerprints')
        return self


@dataclass(frozen=True)
class PreparedLocalV5(_TensorRecord):
    spatial: torch.Tensor              # [B,6,512,16,16]
    spatial_valid: torch.Tensor        # [B,6]
    tokens: torch.Tensor               # [B,6,4,16,512], modality-separated
    token_valid: torch.Tensor          # [B,6,4,16]
    detail: torch.Tensor               # [B,6,512,16,16]
    detail_valid: torch.Tensor
    boxes: torch.Tensor
    pixel_weights: torch.Tensor
    region_valid: torch.Tensor
    metadata: tuple[dict,...]

    def validate(self):
        b=self.tokens.shape[0];d=self.tokens.device;h,w=self.pixel_weights.shape[-2:]
        for name,shape in (('tokens',(b,6,4,16,512)),('spatial',(b,6,512,16,16)),('detail',(b,6,512,16,16)),
                           ('boxes',(b,6,4)),('pixel_weights',(b,6,h,w))): _tensor(getattr(self,name),shape,d,name)
        for name,shape in (('token_valid',(b,6,4,16)),('spatial_valid',(b,6)),('detail_valid',(b,6)),('region_valid',(b,6))):
            _tensor(getattr(self,name),shape,d,name,boolean=True)
        if (self.pixel_weights<0).any() or (self.pixel_weights.sum(1)>1.00001).any(): raise ValueError('Invalid V5 owner weights')
        if len(self.metadata)!=b: raise ValueError('Local metadata differs')
        return self


@dataclass(frozen=True)
class PreparedNativeFullControlV5(_TensorRecord):
    core: PreparedNativeFullControlV4
    relation: torch.Tensor
    relation_valid: torch.Tensor
    local: PreparedLocalV5 | None
    metadata: tuple[dict,...]

    @property
    def body(self): return self.core.body
    def validate(self):
        self.core.validate();b,n,_=self.body.geometry.shape;d=self.body.geometry.device
        _tensor(self.relation,(b,n,512),d,'Separate Interaction')
        _tensor(self.relation_valid,(b,n),d,'Interaction validity',boolean=True)
        if self.local is not None:
            self.local.validate()
            if self.local.pixel_weights.shape!=(b,6,self.body.target_token_hw[0]*16,self.body.target_token_hw[1]*16):
                raise ValueError('Local and Body grids differ')
        if len(self.metadata)!=b: raise ValueError('Prepared metadata differs')
        return self
