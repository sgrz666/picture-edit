"""V7 extends V6.2 sampling/attention; cross-person values are geometry only."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..native_v62.poselink import PoseLinkAttentionV62
from ..native_v62.conditions import routing_permissions
from .contracts import PoseLinkConfigV7
from .pose_fusion import PersonPoseMaskFusion


def apply_xy_rope(value,xy):
    """Rotate [B,H,N,D] with global normalized Canvas XY, in FP32.

    X and Y receive equal even subspaces; this is independent of native Qwen
    Frame/role axes. Different Q/K sequence lengths require no special case.
    """
    if value.ndim!=4 or value.shape[-1]%4:
        raise ValueError('XY RoPE requires [B,H,N,D] with D divisible by four')
    if tuple(xy.shape)!=(value.shape[0],value.shape[2],2):
        raise ValueError('XY RoPE coordinate shape differs')
    axis=value.shape[-1]//2
    with torch.autocast(device_type=value.device.type,enabled=False):
        frequency=10000.**(-torch.arange(0,axis,2,device=value.device,dtype=torch.float32)/axis)
        original=value.float();pieces=[]
        for index in range(2):
            phase=xy[...,index].float()[:,None,:,None]*frequency*(2*math.pi)
            pairs=original[...,index*axis:(index+1)*axis].reshape(*value.shape[:-1],axis//2,2)
            even,odd=pairs.unbind(-1)
            pieces.append(torch.stack((even*phase.cos()-odd*phase.sin(),
                even*phase.sin()+odd*phase.cos()),-1).flatten(-2))
        return torch.cat(pieces,-1).to(value.dtype)


class PoseLinkAttentionV7(PoseLinkAttentionV62):
    def __init__(self,config):
        if not isinstance(config,PoseLinkConfigV7):raise ValueError('V7 PoseLink configuration required')
        super().__init__(config)
        self.mask_pose_fusion=PersonPoseMaskFusion(config.width)
        self.contact_geometry_projection=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,config.width))
        self.contact_person_projection=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,config.width))
        self.contact_part_projection=nn.Linear(14,config.width)
        self.contact_key_projection=nn.Linear(config.width,config.width)
        self.contact_value_projection=nn.Linear(config.width,config.width)
        # Existing head prefixes make every new contact projection a true S1
        # PoseLink head; bridge encoders remain S2/S3 controller parameters.
        for layer in config.injection_layers:
            head=nn.Linear(config.width,config.hidden_dim)
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            self.output_heads['contact_'+str(layer)]=head

    def _pose_key_features(self,features,samples):
        return self.mask_pose_fusion(features,samples) if self.config.mask_pose_fusion else features

    def _pose_contact_positive(self,routing):
        if self.config.geometry_contact:return torch.zeros_like(routing.contact_positive)
        return super()._pose_contact_positive(routing)

    def _routed_chunk(self,query,key,value,query_xy,key_xy,*args):
        if self.config.spatial_rope:
            query,key=apply_xy_rope(query,query_xy),apply_xy_rope(key,key_xy)
        return super()._routed_chunk(query,key,value,query_xy,key_xy,*args)

    def _contact_chunk(self,query,key,value,query_xy,key_xy,allowed):
        if self.config.spatial_rope:
            query,key=apply_xy_rope(query,query_xy),apply_xy_rope(key,key_xy)
        attended=F.scaled_dot_product_attention(query,key,value,attn_mask=allowed[:,None],dropout_p=0.)
        return attended.transpose(1,2).flatten(2)

    def _merge_side_attention(self,delta,has_keys,query,samples,geometry,spatial,routing,xy,layer):
        cfg=self.config
        if not cfg.geometry_contact or not cfg.use_contact:return delta,has_keys
        b,_,n,dim=query.shape
        # Select only the sampled Key's own raw person token. The bridge never
        # accesses samples.hidden, Source RGB, identity or interaction features.
        person=samples.person_ids.clamp_min(0)
        own_person=samples.person_tokens.gather(2,person[:,:,None,None].expand(-1,-1,1,512)).squeeze(2)
        key_features=(self.contact_geometry_projection(samples.geometry)
            +self.contact_person_projection(own_person)+self.contact_part_projection(samples.parts))
        key=self.contact_key_projection(key_features).reshape(b,-1,cfg.heads,dim).transpose(1,2)
        value=self.contact_value_projection(key_features).reshape(b,-1,cfg.heads,dim).transpose(1,2)
        key_supported=(samples.valid & (samples.person_ids>=0)
            & (samples.person_weights.gather(-1,person[...,None])[...,0]>cfg.routing_epsilon))
        query_person=routing.query_person_ids.clamp_min(0)
        query_support=torch.maximum(geometry.owner,geometry.projected).gather(-1,query_person[...,None])[...,0]
        query_supported=(routing.query_person_ids>=0)&(query_support>cfg.routing_epsilon)&(geometry.coverage>0)
        chunks=[];bridge_valid=[]
        for start in range(0,n,cfg.query_chunk_size):
            stop=start+cfg.query_chunk_size;qids=routing.query_person_ids[:,start:stop]
            allowed=routing_permissions(qids,routing.query_parts[:,start:stop],samples.person_ids,samples.parts,
                routing.contact_positive,routing.contact_forbidden,key_supported,epsilon=cfg.routing_epsilon)
            allowed=(allowed & (qids[:,:,None]!=samples.person_ids[:,None])
                & query_supported[:,start:stop,None] & geometry.person_valid.all(1)[:,None,None])
            bridge_valid.append(allowed.any(-1))
            args=(query[:,:,start:stop],key,value,xy[:,start:stop],samples.xy,allowed)
            chunks.append(checkpoint(self._contact_chunk,*args,use_reentrant=False)
                if self.training and torch.is_grad_enabled() else self._contact_chunk(*args))
        valid=torch.cat(bridge_valid,1)
        bridge=self.output_heads['contact_'+str(layer)](torch.cat(chunks,1))*valid[...,None]
        # Mask each branch before combining: a bridge edge cannot expose the
        # ordinary Pose head's trained bias when it has no own-person keys.
        return delta*has_keys[...,None]+bridge,has_keys|valid
