"""Identity-adaptive local Query modulation using already source-bound tokens."""
import torch
from torch import nn
from torch.nn import functional as F

from ..native_v5.local import NativeLocalControllerV5


class IdentityQueryModulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity_norm=nn.LayerNorm(512)
        self.output=nn.Linear(512,1024)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,query,identity):
        scale,shift=self.output(self.identity_norm(identity)).chunk(2,-1)
        return query+F.layer_norm(query,(512,))*scale[:,None]+shift[:,None]


class NativeLocalControllerV7(NativeLocalControllerV5):
    def __init__(self,config,*,hand=False,identity_modulation=True):
        if type(identity_modulation) is not bool:raise ValueError('identity_modulation requires bool')
        super().__init__(config,hand=hand)
        self.identity_modulation_enabled=identity_modulation
        self.identity_modulation=IdentityQueryModulation()

    def _modality_query(self,query,prepared,row,region,modality):
        if not self.identity_modulation_enabled or modality not in (0,1):return query
        person=region if region<2 else (region-2)//2
        valid=prepared.token_valid[row:row+1,person,0]
        if not valid.any():return query
        tokens=prepared.tokens[row:row+1,person,0]
        identity=(tokens*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        return self.identity_modulation(query,identity)
