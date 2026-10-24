"""Native block/sigma controls. No sampler counters or hidden cross-branch gains."""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from .contracts import BRANCHES


@dataclass(frozen=True)
class Schedule:
    curve: str='constant'
    start: float=0.
    end: float=1.
    first: float=1.
    last: float=1.
    knots: tuple=()

    def __post_init__(self):
        if self.curve not in ('constant','linear','cosine','piecewise') or not 0<=self.start<self.end<=1:
            raise ValueError('Invalid denoise schedule/window')
        if any(not math.isfinite(x) or x<0 for x in (self.first,self.last)): raise ValueError('Invalid schedule multiplier')
        if self.curve=='piecewise':
            if len(self.knots)<2 or self.knots[0][0]!=0 or self.knots[-1][0]!=1:
                raise ValueError('Piecewise schedule must cover [0,1]')
            if any(not math.isfinite(v) or v<0 for p in self.knots for v in p) or any(a[0]>=b[0] for a,b in zip(self.knots,self.knots[1:])):
                raise ValueError('Invalid piecewise knots')

    def value(self,sigma):
        if sigma.ndim!=1 or not torch.isfinite(sigma).all() or ((sigma<0)|(sigma>1)).any(): raise ValueError('Real normalized sigma [B] required')
        p=1-sigma.float()
        if self.curve=='piecewise':
            out=torch.full_like(p,self.knots[-1][1])
            for (x0,y0),(x1,y1) in reversed(tuple(zip(self.knots,self.knots[1:]))):
                out=torch.where(p<=x1,y0+(y1-y0)*(p-x0)/(x1-x0),out)
            return out
        q=((p-self.start)/(self.end-self.start)).clamp(0,1)
        if self.curve=='cosine': q=.5*(1-torch.cos(torch.pi*q))
        out=torch.full_like(p,self.first) if self.curve=='constant' else self.first+(self.last-self.first)*q
        return out*((p>=self.start)&(p<=self.end))

    def potentially_active(self):
        if self.curve=='constant': return self.first>0
        if self.curve=='piecewise': return any(p[1]>0 for p in self.knots)
        return max(self.first,self.last)>0


class NativeStrengthController(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.log_group_gain=nn.ParameterDict({b:nn.Parameter(torch.zeros(6)) for b in BRANCHES})
        self.register_buffer('prior',torch.ones(len(BRANCHES),6))
        schedules={b:Schedule() for b in BRANCHES}
        if config.strength_preset=='legacy_full':
            for b in ('detail','face_identity','face_geometry'):
                schedules[b]=Schedule('cosine',.35,1.,0.,1.)
            for b in ('face_texture','hand_texture'): schedules[b]=Schedule('cosine',.65,1.,0.,1.)
            schedules['hand_geometry']=Schedule('cosine',0.,1.,.15,1.)
            for b in ('face_identity','face_geometry','face_texture'):
                self.prior[BRANCHES.index(b)]=torch.tensor([.1,.2,.5,.8,1.,1.])
            for b in ('hand_geometry','hand_texture'): self.prior[BRANCHES.index(b)]=torch.tensor([.1,.2,.4,.7,1.,1.])
        for b,raw in config.schedules: schedules[b]=Schedule(**{**raw,'knots':tuple(map(tuple,raw.get('knots',())))})
        self.schedules=schedules

    def get_extra_state(self): return {'schedules':{k:asdict(v) for k,v in self.schedules.items()},'branches':BRANCHES,'progress':'1-sigma'}
    def set_extra_state(self,value):
        if value!=self.get_extra_state(): raise ValueError('V5 schedule/header differs')

    def enabled(self,branch,options):
        return options.strength(branch)>0 and self.schedules[branch].potentially_active()

    def forward(self,branch,layer,sigma,options):
        k=6*layer//self.config.num_layers;i=BRANCHES.index(branch)
        return options.strength(branch)*self.log_group_gain[branch][k].exp()*self.prior[i,k]*self.schedules[branch].value(sigma)
