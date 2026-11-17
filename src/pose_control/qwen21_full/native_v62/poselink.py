"""V6.2 direct person routing within the unchanged side Attention budget."""
import math

import torch
from torch.utils.checkpoint import checkpoint

from ..native_v6.poselink import _grid
from ..native_v61.poselink import PoseLinkAttentionV61
from .contracts import PoseLinkConfigV62
from .conditions import routing_permissions
from .tokens import sample_routed_keys
from .observation import observe_permissions


class PoseLinkAttentionV62(PoseLinkAttentionV61):
    def __init__(self,config):
        if not isinstance(config,PoseLinkConfigV62):raise ValueError('V6.2 configuration required')
        super().__init__(config)

    def build_pose_keys(self,joint,layout,geometry,spatial,routing=None):
        if not self.config.person_routing:
            return super().build_pose_keys(joint,layout,geometry,spatial)
        if routing is None:raise ValueError('V6.2 routing conditions required for person-scoped keys')
        layout.validate();geometry.validate();spatial.validate(geometry);routing.validate(geometry)
        return sample_routed_keys(joint,layout,geometry,spatial,routing,self.config)

    def _routed_chunk(self,query,key,value,query_xy,key_xy,query_owner,key_owner,query_contact,key_contact,
                      weight,allowed):
        logits=torch.matmul(query,key.transpose(-1,-2))/math.sqrt(query.shape[-1])
        logits=logits+self._attention_bias(query_xy,key_xy,query_owner,key_owner,query_contact,key_contact)
        logits=logits+weight.clamp_min(torch.finfo(logits.dtype).tiny).log()[:,None,None]
        logits=logits.masked_fill(~allowed[:,None],torch.finfo(logits.dtype).min)
        probability=torch.softmax(logits.float(),-1).to(value.dtype)*allowed[:,None]
        probability=probability/probability.sum(-1,keepdim=True).clamp_min(1.)
        return torch.matmul(probability,value).transpose(1,2).flatten(2)

    def _pose_key_features(self,features,samples):
        """Version extension point; V6.2 preserves its exact sampled features."""
        return features

    def _pose_contact_positive(self,routing):
        return routing.contact_positive if self.config.use_contact else torch.zeros_like(routing.contact_positive)

    def _merge_side_attention(self,delta,has_keys,query,samples,geometry,spatial,routing,xy,layer):
        """No parameters or arithmetic changes for the V6.2 implementation."""
        return delta,has_keys

    def forward(self,target_hidden,joint_hidden,layout,geometry,*,layer,sigma,strength=1.,spatial=None,routing=None):
        cfg=self.config
        if not cfg.person_routing:
            return super().forward(target_hidden,joint_hidden,layout,geometry,layer=layer,sigma=sigma,
                                   strength=strength,spatial=spatial)
        if spatial is None or routing is None:raise ValueError('V6.2 routing requires prepared spatial and routing records')
        geometry.validate();spatial.validate(geometry);routing.validate(geometry);layout.validate()
        b,n=geometry.geometry.shape[:2]
        if target_hidden.shape!=(b,n,cfg.hidden_dim) or joint_hidden.shape!=(b,layout.joint_length,cfg.hidden_dim):
            raise ValueError('V6.2 Hidden layout differs')
        if layout.target_token_hw!=geometry.target_token_hw or layout.target_indices.numel()!=n:
            raise ValueError('V6.2 target grid differs')
        if target_hidden.device!=joint_hidden.device or geometry.geometry.device!=target_hidden.device:
            raise ValueError('V6.2 Hidden/geometry device differs')
        if (not torch.is_tensor(sigma) or sigma.shape!=(b,) or not sigma.is_floating_point()
                or not torch.isfinite(sigma).all() or ((sigma<0)|(sigma>1)).any()):
            raise ValueError('V6.2 sigma requires finite floating [B] in [0,1]')
        if not torch.is_tensor(strength) and type(strength) not in (float,int):raise ValueError('Invalid V6.2 strength')
        gain=torch.as_tensor(strength,device=target_hidden.device)
        if gain.ndim>1 or (gain.ndim==1 and gain.shape!=(b,)) or not torch.isfinite(gain).all() or (gain<0).any():
            raise ValueError('V6.2 strength requires nonnegative finite scalar or [B]')
        if type(layer) is not int:raise ValueError('Layer requires int')
        if layer not in cfg.injection_layers:return target_hidden.new_zeros(target_hidden.shape)
        dtype=self.query_projection.weight.dtype
        target,joint=target_hidden.to(dtype),joint_hidden.to(dtype)
        samples=self.build_pose_keys(joint,layout,geometry,spatial,routing)
        keys=self._pose_key_features(self.pose_projection(self.pose_norm(samples.hidden)),samples)
        if cfg.use_geometry:
            persons=self.person_projection(samples.person_tokens)
            keys=keys+self.geometry_projection(samples.geometry)+(persons*samples.person_weights[...,None]).sum(2)
        if cfg.use_interaction and samples.interaction is not None:keys=keys+self.interaction_projection(samples.interaction)
        if cfg.use_role:keys=keys+self.role_embedding((samples.person_ids+1).clamp_min(0))
        if cfg.multiscale_tokens:keys=keys+self.token_type_embedding(samples.type_ids)
        condition=self.layer_embedding.weight[layer]+self.time_projection(sigma.to(target)[:,None])
        query=self.query_projection(self.query_norm(target))+condition[:,None];keys=keys+condition[:,None]
        heads,dim=cfg.heads,cfg.width//cfg.heads
        query=query.reshape(b,n,heads,dim).transpose(1,2)
        key=self.key_projection(keys).reshape(b,-1,heads,dim).transpose(1,2)
        value=self.value_projection(keys).reshape(b,-1,heads,dim).transpose(1,2)
        xy=_grid(geometry.target_token_hw,b,target);active=geometry.person_valid[:,None].to(dtype)
        owner,contact=geometry.owner.to(dtype)*active,geometry.contact.to(dtype)*active
        positive=self._pose_contact_positive(routing)
        chunks=[];has_keys=[]
        for start in range(0,n,cfg.query_chunk_size):
            stop=start+cfg.query_chunk_size
            allowed=routing_permissions(routing.query_person_ids[:,start:stop],routing.query_parts[:,start:stop],
                samples.person_ids,samples.parts,positive,routing.contact_forbidden,samples.valid,
                epsilon=cfg.routing_epsilon)
            observe_permissions(allowed,routing.query_person_ids[:,start:stop],samples.person_ids,layer=layer)
            has_keys.append(allowed.any(-1))
            args=(query[:,:,start:stop],key,value,xy[:,start:stop],samples.xy,owner[:,start:stop],samples.owner,
                  contact[:,start:stop],samples.contact,samples.weight,allowed)
            chunks.append(checkpoint(self._routed_chunk,*args,use_reentrant=False)
                if self.training and torch.is_grad_enabled() else self._routed_chunk(*args))
        delta=self.output_heads[str(layer)](torch.cat(chunks,1))
        delta,has_keys=self._merge_side_attention(delta,torch.cat(has_keys,1),query,samples,
            geometry,spatial,routing,xy,layer)
        gain=gain.to(dtype)*self.layer_log_gains[str(layer)].exp()
        if gain.ndim:gain=gain[:,None,None]
        result=delta*gain*geometry.coverage.to(dtype)[...,None]*has_keys[...,None]
        return result*spatial.edit_gate[...,None].to(dtype) if cfg.edit_region_gate else result
