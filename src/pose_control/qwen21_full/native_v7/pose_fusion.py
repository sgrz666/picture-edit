"""Residual Pose/Mask fusion on the existing person-scoped sampled feature set."""
import torch
from torch import nn


class PersonPoseMaskFusion(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.mask_encoder=nn.Sequential(nn.Linear(16,width),nn.SiLU())
        self.output=nn.Linear(2*width,width)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,features,samples):
        # No pooling or cross-person attention here: every residual reads exactly
        # one existing Key and that Key's own support/part evidence. Shared
        # weights avoid assigning a distinct encoder to an arbitrary person slot.
        person=samples.person_ids.clamp_min(0)[...,None]
        evidence=torch.cat((samples.owner.gather(-1,person),
            samples.person_weights.gather(-1,person),samples.parts),-1).to(features)
        mask=self.mask_encoder(evidence)
        residual=self.output(torch.cat((features,mask),-1))
        return features+residual*(samples.valid & (samples.person_ids>=0))[...,None]
