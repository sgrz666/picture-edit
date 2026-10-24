"""V4 pixel ownership and sparse, current-hidden local residual controllers.

No learned feature is cached offline. Region order is declared in the schema.
Only output projections are zero initialized; input paths retain normal weights.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .native_dual_identity import ContextDelta
from .native_local_condition import PreparedNativeLocalV4


@dataclass(frozen=True)
class NativeLocalConfigV4:
    injection_layers: tuple = (5, 10, 15, 21, 26, 31)
    hidden_dim: int = 4096
    num_layers: int = 32
    verification_only: bool = False
    width: int = 512
    min_spatial_cells: int = 4
    tie_tolerance: float = 1e-4
    local_mode: str = 'dynamic'
    face_schedule: tuple = ((0., 1.), (1., 1.))
    hand_schedule: tuple = ((0., 1.), (1., 1.))

    def __post_init__(self):
        if type(self.verification_only) is not bool or self.width!=512:
            raise ValueError('Control width remains512; verification switch must be bool')
        if (not self.verification_only and (self.hidden_dim != 4096 or self.num_layers!=32 or tuple(self.injection_layers) != (5,10,15,21,26,31))):
            raise ValueError('V4 preserves Qwen4096/control512 and the six declared layers')
        if (type(self.hidden_dim) is not int or self.hidden_dim<=0 or type(self.num_layers) is not int or self.num_layers<6
                or len(self.injection_layers)!=6 or len(set(self.injection_layers))!=6
                or any(type(i) is not int or not 0<=i<self.num_layers for i in self.injection_layers)):
            raise ValueError('Six unique local layers must fit the explicitly declared model')
        if type(self.min_spatial_cells) is not int or self.min_spatial_cells < 1 or self.local_mode not in ('dynamic','static'):
            raise ValueError('Invalid local mode/cell threshold')
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance <= 0:
            raise ValueError('Tie tolerance must be positive')
        for points in (self.face_schedule, self.hand_schedule):
            if len(points)<2 or points[0][0]!=0 or points[-1][0]!=1:
                raise ValueError('Schedule must cover progress 0..1')
            if any(not math.isfinite(v) or v<0 for p in points for v in p) or any(a[0]>=b[0] for a,b in zip(points,points[1:])):
                raise ValueError('Schedule knots must be finite and strictly increasing')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value,'injection_layers':tuple(value['injection_layers']),
                      'face_schedule':tuple(tuple(p) for p in value['face_schedule']),
                      'hand_schedule':tuple(tuple(p) for p in value['hand_schedule'])})


def schedule_value(sigma, knots):
    progress=1-sigma.float()
    result=torch.full_like(progress,knots[-1][1])
    for (x0,y0),(x1,y1) in reversed(tuple(zip(knots,knots[1:]))):
        value=y0+(y1-y0)*(progress-x0)/(x1-x0)
        result=torch.where(progress<=x1,value,result)
    return result


def resolve_local_ownership(condition, *, face=True, hand=True, tie_tolerance=1e-4):
    """Resolve six overlapping surfaces BEFORE reducing pixels to latent cells.

    Trusted Z is compared with the scene Z. Unknown Z uses only the declared
    visible part/person intersection. A tie shares a pixel instead of doubling it.
    """
    h,w=condition.region_masks.shape[-2:]
    device=condition.region_masks.device
    owner_indices=torch.tensor((0,1,0,0,1,1),device=device)
    active=condition.region_valid & condition.person_valid[:,owner_indices]
    enabled=torch.tensor((face,face,hand,hand,hand,hand),device=device)
    active=active & enabled[None]
    yy,xx=torch.meshgrid((torch.arange(h,device=device,dtype=torch.float32)+.5)/h,(torch.arange(w,device=device,dtype=torch.float32)+.5)/w,indexing='ij')
    boxes=condition.boxes
    inside=(xx>=boxes[...,0,None,None])&(xx<boxes[...,2,None,None])&(yy>=boxes[...,1,None,None])&(yy<boxes[...,3,None,None])
    support=condition.region_masks*condition.visibility*inside*active[...,None,None]
    owner=condition.person_owner[:,owner_indices]
    tolerance=condition.body_height[:,owner_indices,None,None]*tie_tolerance
    known=condition.local_z_valid & condition.scene_z_valid
    visible_z=condition.local_z <= condition.scene_z+tolerance
    conservative=condition.part_visible & (owner>0)
    support=support*torch.where(known,visible_z,conservative)*owner
    candidate=condition.local_z_valid & (support>0)
    nearest=torch.where(candidate,condition.local_z,torch.inf).amin(1,keepdim=True)
    # When two local regions have trusted depths, only front/tied regions survive.
    support=support*(~candidate | (condition.local_z<=nearest+tolerance))
    return support/support.sum(1,keepdim=True).clamp_min(1.)


def prepare_local(face, hand, condition, references, *, face_enabled=True, hand_enabled=True, config=None):
    config=config or NativeLocalConfigV4()
    pieces=[]
    b=condition.person_valid.shape[0]
    for enabled,module,count in ((face_enabled,face,2),(hand_enabled,hand,4)):
        if enabled:
            pieces.append(module(condition,references))
        else:
            x=condition.face_landmarks
            pieces.append((x.new_zeros(b,count,512,16,16),x.new_zeros(b,count,16,512),
                torch.zeros(b,count,16,dtype=torch.bool,device=x.device),
                torch.zeros(b,count,16,dtype=torch.bool,device=x.device),
                torch.zeros(b,count,dtype=torch.bool,device=x.device)))
    spatial,tokens,valid,geometry,sv=(torch.cat([p[i] for p in pieces],1) for i in range(5))
    weights=resolve_local_ownership(condition,face=face_enabled,hand=hand_enabled,tie_tolerance=config.tie_tolerance)
    region_valid=valid.any(-1)&(weights.flatten(2).sum(-1)>0)
    return PreparedNativeLocalV4(spatial,tokens,valid,geometry,sv,condition.boxes,
        weights*region_valid[...,None,None],region_valid,condition.metadata).validate()


def target_region_grid(prepared, target_hw):
    """Actual target cell centers in each unclipped ROI, no synthetic hidden grid."""
    h,w=target_hw
    if prepared.pixel_weights.shape[-2:]!=(h*16,w*16):
        raise ValueError('Local pixel canvas differs from actual Qwen target grid')
    boxes=prepared.boxes
    yy,xx=torch.meshgrid((torch.arange(h,device=boxes.device,dtype=torch.float32)+.5)/h,
                         (torch.arange(w,device=boxes.device,dtype=torch.float32)+.5)/w,indexing='ij')
    points=torch.stack((xx,yy),-1)[None,None]
    span=(boxes[...,2:]-boxes[...,:2]).clamp_min(1e-8)
    grid=(points-boxes[...,:2][...,None,None,:])/span[...,None,None,:]
    weights=F.avg_pool2d(prepared.pixel_weights,16).flatten(2)
    return grid*2-1,weights


class NativeLocalController(nn.Module):
    def __init__(self, config=None, *, anatomical_sides=False):
        super().__init__()
        self.config=config or NativeLocalConfigV4()
        self.query=nn.Sequential(nn.LayerNorm(self.config.hidden_dim),nn.Linear(self.config.hidden_dim,512))
        self.spatial_projection=nn.Linear(512,512)
        self.geometry_attention=ContextDelta()
        self.appearance_attention=ContextDelta()
        self.self_norm=nn.LayerNorm(512)
        self.shared_attention=nn.MultiheadAttention(512,8,batch_first=True,dropout=0.)
        self.shared_ffn=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,2048),nn.GELU(),nn.Linear(2048,512))
        self.layer_embedding=nn.Embedding(6,512)
        self.time=nn.Sequential(nn.Linear(1,512),nn.SiLU(),nn.Linear(512,512))
        self.side=nn.Embedding(2,512) if anatomical_sides else None
        self.output_heads=nn.ModuleList(nn.Linear(512,self.config.hidden_dim) for _ in range(6))
        self.static_output=nn.Linear(512,512)
        for layer in (*self.output_heads,self.static_output):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self, hidden, prepared, *, layer, sigma, target_hw, regions):
        if layer not in self.config.injection_layers:
            return hidden.new_zeros(hidden.shape,dtype=torch.float32)
        grid,weights=target_region_grid(prepared,target_hw)
        result=hidden.new_zeros(hidden.shape,dtype=torch.float32)
        slot=self.config.injection_layers.index(layer)
        for row in range(hidden.shape[0]):
            for region in regions:
                cells=(weights[row,region]>0).nonzero(as_tuple=True)[0]
                if not cells.numel() or not prepared.region_valid[row,region]:
                    continue
                query=self.query(hidden[row:row+1,cells].float())+self.layer_embedding.weight[slot]
                query=query+self.time(sigma[row:row+1,None].float())[:,None]
                if self.side is not None:
                    query=query+self.side.weight[(region-2)%2]
                valid=prepared.token_valid[row:row+1,region]
                gm=prepared.geometry_token[row:row+1,region]
                qv=torch.ones(1,cells.numel(),dtype=torch.bool,device=hidden.device)
                context=prepared.tokens[row:row+1,region]
                update=self.appearance_attention(query,context,valid&~gm,qv)
                if cells.numel()>=self.config.min_spatial_cells:
                    update=update+self.geometry_attention(query,context,valid&gm,qv)
                    if prepared.spatial_valid[row,region]:
                        sampled=F.grid_sample(prepared.spatial[row:row+1,region],
                            grid[row:row+1,region].reshape(1,-1,1,2)[:,cells],align_corners=False)
                        update=update+self.spatial_projection(sampled.squeeze(-1).transpose(1,2))
                fused=query+update
                normalized=self.self_norm(fused)
                fused=fused+self.shared_attention(normalized,normalized,normalized,need_weights=False)[0]
                fused=fused+self.shared_ffn(fused)
                delta=self.output_heads[slot](fused)*weights[row:row+1,region,cells,None]
                # Out-of-place accumulation preserves the graph when several
                # anatomical regions cover the same cell during recomputation.
                flat=result.reshape(-1,self.config.hidden_dim).index_add(0,cells+row*hidden.shape[1],delta[0])
                result=flat.reshape_as(result)
        return result

    def static_delta(self, prepared, *, target_hw, regions):
        grid,weights=target_region_grid(prepared,target_hw)
        b=prepared.spatial.shape[0];n=target_hw[0]*target_hw[1]
        result=prepared.tokens.new_zeros(b,n,512)
        for region in regions:
            spatial_ok=(weights[:,region]>0).sum(1)>=self.config.min_spatial_cells
            valid=prepared.token_valid[:,region]&(~prepared.geometry_token[:,region]|spatial_ok[:,None])
            context=(prepared.tokens[:,region]*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
            sampled=F.grid_sample(prepared.spatial[:,region],grid[:,region],align_corners=False).flatten(2).transpose(1,2)
            sampled=sampled*(prepared.spatial_valid[:,region]&spatial_ok)[:,None,None]
            result=result+self.static_output(sampled+context[:,None])*weights[:,region,:,None]
        return result


class NativeFaceController(NativeLocalController):
    pass


class NativeHandController(NativeLocalController):
    def __init__(self, config=None):
        super().__init__(config,anatomical_sides=True)
