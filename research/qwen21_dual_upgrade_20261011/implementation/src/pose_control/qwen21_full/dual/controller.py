"""Target-hidden residual controller without changing native attention or RoPE."""
from __future__ import annotations

import math
import torch
from torch import nn

from .appearance import DualAppearanceEncoder
from .conditions import ControlStrengths, DualGenerationInput, PreparedDualState
from .geometry import DualGeometryEncoder
from .reasoner import DualGeometryReasoner
from .routing import compute_routing


def sinusoidal(value: torch.Tensor, width: int):
    frequency = torch.exp(-math.log(10000)*torch.arange(width//2,device=value.device,dtype=torch.float32)/max(width//2-1,1))
    angle = value.float()[...,None]*frequency
    return torch.cat((angle.sin(),angle.cos()),dim=-1).to(value.dtype)


def position_encoding(target_hw,width,device,dtype):
    y,x = torch.meshgrid(torch.arange(target_hw[0],device=device,dtype=torch.float32),
                         torch.arange(target_hw[1],device=device,dtype=torch.float32),indexing="ij")
    return torch.cat((sinusoidal(x.flatten(),width//2),sinusoidal(y.flatten(),width//2)),dim=-1).to(dtype)


class ContextAttention(nn.Module):
    """An attention update that is exactly zero when no context token is valid."""
    def __init__(self,width):
        super().__init__()
        self.query_norm,self.context_norm = nn.LayerNorm(width),nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width,8,batch_first=True)

    def forward(self,query,context,valid):
        output = torch.zeros_like(query)
        rows = valid.any(-1).nonzero(as_tuple=False).flatten()
        if rows.numel():
            q = self.query_norm(query.index_select(0,rows))
            kv = self.context_norm(context.index_select(0,rows))
            update = self.attention(q,kv,kv,key_padding_mask=~valid.index_select(0,rows),need_weights=False)[0]
            output = output.index_copy(0,rows,update)
        return output


def _ffn(width):
    return nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))


class DualV8Adapter(nn.Module):
    architecture_version = "dual_v8"

    def __init__(self,width=512,hidden_size=4096,rank=64,layer_indices=(3,8,13,18,23,28)):
        super().__init__()
        if width <= 0 or width % 8 or hidden_size <= 0 or rank <= 0:
            raise ValueError("width must be positive and divisible by 8; hidden_size and rank must be positive")
        if not layer_indices or len(set(layer_indices)) != len(layer_indices) or any(type(index) is not int or index < 0 for index in layer_indices):
            raise ValueError("layer_indices must be distinct nonnegative integers")
        self.width,self.hidden_size,self.rank = width,hidden_size,rank
        self.layer_indices = tuple(layer_indices)
        self.geometry_encoder = DualGeometryEncoder()
        self.reasoner = DualGeometryReasoner(width)
        self.appearance_encoder = DualAppearanceEncoder(width)
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size,width)
        self.stage_embedding = nn.Embedding(len(layer_indices),width)
        self.time_embedding = nn.Sequential(nn.Linear(width*2,width*2),nn.SiLU(),nn.Linear(width*2,width))
        self.local_query_norm,self.local_geometry_norm = nn.LayerNorm(width),nn.LayerNorm(width)
        self.local_geometry = nn.Sequential(nn.Linear(width*2,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.geometry_attention = ContextAttention(width)
        self.appearance_attention = ContextAttention(width)
        self.contact_attention = ContextAttention(width)
        self.branch_ffn = nn.ModuleDict({branch:_ffn(width) for branch in ("geometry","appearance","contact")})
        self.output_heads = nn.ModuleDict()
        for layer in layer_indices:
            heads = nn.ModuleDict()
            for branch in ("geometry","appearance","contact"):
                head = nn.Sequential(nn.Linear(width,rank),nn.SiLU(),nn.Linear(rank,hidden_size))
                # Only the last projection is zero; gradients reach it on step one.
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                heads[branch] = head
            self.output_heads[str(layer)] = heads
        self.last_diagnostics = {}
        self._budget_terms = {}
        self._layer_residuals = {}
        self._reference_hidden = None
        self._cumulative = None

    @property
    def config(self):
        return {"width":self.width,"hidden_size":self.hidden_size,"rank":self.rank,
                "layer_indices":list(self.layer_indices)}

    def freeze_warm_geometry(self):
        for module in (self.geometry_encoder.spatial_encoder,self.reasoner.person_reasoner):
            module.requires_grad_(False)
        self.reasoner.shared_role_offset.requires_grad_(False)
        return self

    def prepare(self,generation_input: DualGenerationInput):
        generation_input.validate()
        spatial,mask = self.geometry_encoder(generation_input.geometry)
        dense,nodes,node_valid,contacts,contact_valid,target_hw = self.reasoner(spatial,mask,generation_input.geometry)
        appearance,appearance_valid = self.appearance_encoder(generation_input.source)
        geometry_weights,appearance_weights,contact_weights = compute_routing(generation_input.geometry,target_hw)
        return PreparedDualState(dense,nodes,node_valid,appearance,appearance_valid,
                                 contacts,contact_valid,geometry_weights,appearance_weights,
                                 contact_weights,generation_input.geometry.person_valid,target_hw)

    @staticmethod
    def _batch_scalar(value,batch,device,dtype,name):
        scalar = torch.as_tensor(value,device=device,dtype=dtype)
        if scalar.numel() == 1:
            scalar = scalar.reshape(1).expand(batch)
        elif tuple(scalar.shape) != (batch,):
            raise ValueError(f"{name} must be scalar or [B]")
        if not torch.isfinite(scalar).all():
            raise ValueError(f"{name} must be finite")
        return scalar

    @staticmethod
    def _stats(value,support,cumulative,hidden,reference=None):
        count = int(support.sum())
        local = value.detach().float()[support]
        rms = float(local.square().mean().sqrt()) if count else 0.
        base = float(hidden.detach().float()[support].square().mean().sqrt()) if count else 0.
        cumulative_rms = float(cumulative.detach().float()[support].square().mean().sqrt()) if count else 0.
        reference = hidden if reference is None else reference
        reference_rms = float(reference.detach().float()[support].square().mean().sqrt()) if count else 0.
        return {"rms":rms,"hidden_rms":base,"ratio":rms/max(base,1e-8),
                "max":float(local.abs().max()) if count else 0., "count":count,
                "nonzero_tokens":int((local.abs().amax(-1)>0).sum()) if count else 0,
                "cumulative":cumulative_rms,"cumulative_ratio":cumulative_rms/max(reference_rms,1e-8)}

    @staticmethod
    def _cumulative_scale(previous,delta,limit):
        """Largest step along delta inside the first-hidden RMS budget ball.

        The optional inference limiter is a scalar applied to all three branches,
        so branch attribution is retained. The training penalty remains smooth
        and independent of this detached safety scale.
        """
        with torch.no_grad():
            previous,delta = previous.detach().float(),delta.detach().float()
            a = delta.square().sum(-1,keepdim=True)
            b = 2*(previous*delta).sum(-1,keepdim=True)
            c = previous.square().sum(-1,keepdim=True)-limit.square()
            root = (-b+(b.square()-4*a*c).clamp_min(0).sqrt())/(2*a.clamp_min(1e-20))
            exceeds = (previous+delta).square().sum(-1,keepdim=True)>limit.square()
            return torch.where(exceeds,root.clamp(0,1),torch.ones_like(root))

    def residual(self,layer_index,target_hidden,prepared,timestep,sigma,strengths=ControlStrengths()):
        if layer_index not in self.layer_indices:
            raise ValueError(f"layer {layer_index} is not a controlled layer")
        if target_hidden.ndim != 3 or tuple(target_hidden.shape) != (prepared.dense.shape[0],math.prod(prepared.target_hw),self.hidden_size):
            raise ValueError("target_hidden must have shape [B,N,hidden_size] matching prepared target_hw")
        if target_hidden.device != prepared.dense.device or target_hidden.dtype != prepared.dense.dtype:
            raise ValueError("target_hidden and prepared state must share device and dtype")
        if not isinstance(strengths,ControlStrengths):
            raise ValueError("strengths must be ControlStrengths")
        if (layer_index == self.layer_indices[0] or self._reference_hidden is None
                or self._reference_hidden.shape != target_hidden.shape):
            self.last_diagnostics = {}
            self._budget_terms = {}
            self._layer_residuals = {}
            self._reference_hidden = target_hidden.detach().clone()
            self._cumulative = torch.zeros_like(target_hidden.detach())
        # Block checkpointing revisits layers in reverse. Replacing keyed values
        # and summing only preceding stages prevents double accumulation.
        preceding = self.layer_indices[:self.layer_indices.index(layer_index)]
        previous_branches = {branch:sum((self._layer_residuals[i][branch] for i in preceding
            if i in self._layer_residuals),torch.zeros_like(target_hidden))
            for branch in ("geometry","appearance","contact")}
        previous = sum(previous_branches.values())
        if strengths.disabled:
            delta = torch.zeros_like(target_hidden)
            support = torch.zeros(target_hidden.shape[:2],device=target_hidden.device,dtype=torch.bool)
            self.last_diagnostics[layer_index] = {branch:self._stats(delta,support,self._cumulative,target_hidden)
                for branch in ("geometry","appearance","contact","total")}
            return delta
        batch,count,_ = target_hidden.shape
        time = self._batch_scalar(timestep,batch,target_hidden.device,target_hidden.dtype,"timestep")
        flow = self._batch_scalar(sigma,batch,target_hidden.device,target_hidden.dtype,"sigma")
        stage = self.stage_embedding.weight[self.layer_indices.index(layer_index)].to(target_hidden.dtype)
        temporal = self.time_embedding(torch.cat((sinusoidal(time,self.width),sinusoidal(flow*1000,self.width)),dim=-1))
        query = self.query_projection(self.hidden_norm(target_hidden))
        query = query+position_encoding(prepared.target_hw,self.width,query.device,query.dtype)[None]+stage[None,None]+temporal[:,None]
        person_query = query[:,None].expand(-1,2,-1,-1).reshape(batch*2,count,self.width)
        branches = {}
        if strengths.geometry:
            local = self.local_geometry(torch.cat((self.local_query_norm(person_query),
                    self.local_geometry_norm(prepared.dense.flatten(0,1))),dim=-1))
            geometry = local+self.geometry_attention(person_query,prepared.nodes.flatten(0,1),prepared.node_valid.flatten(0,1))
            geometry = geometry+self.branch_ffn["geometry"](geometry)
            projected = self.output_heads[str(layer_index)]["geometry"](geometry).reshape(batch,2,count,self.hidden_size)
            active = prepared.node_valid.any(-1)[...,None,None]
            branches["geometry"] = (projected*prepared.weights_geometry*active).sum(1)*strengths.geometry
        else:
            branches["geometry"] = torch.zeros_like(target_hidden)
        if strengths.appearance:
            appearance = self.appearance_attention(person_query,prepared.appearance.flatten(0,1),prepared.appearance_valid.flatten(0,1))
            appearance = appearance+self.branch_ffn["appearance"](appearance)
            projected = self.output_heads[str(layer_index)]["appearance"](appearance).reshape(batch,2,count,self.hidden_size)
            active = prepared.appearance_valid.any(-1)[...,None,None]
            branches["appearance"] = (projected*prepared.weights_appearance*active).sum(1)*strengths.appearance
        else:
            branches["appearance"] = torch.zeros_like(target_hidden)
        if strengths.contact and prepared.contact_valid.any():
            contact = self.contact_attention(query,prepared.contacts,prepared.contact_valid)
            contact = contact+self.branch_ffn["contact"](contact)
            active = prepared.contact_valid.any(-1)[:,None,None]
            branches["contact"] = self.output_heads[str(layer_index)]["contact"](contact)*prepared.weights_contact*active*strengths.contact
        else:
            branches["contact"] = torch.zeros_like(target_hidden)
        delta = sum(branches.values())
        hidden_rms = torch.linalg.vector_norm(target_hidden.detach().float(),dim=-1,keepdim=True)/math.sqrt(self.hidden_size)
        support = {"geometry":((prepared.weights_geometry*prepared.node_valid.any(-1)[...,None,None]).sum(1)>0).squeeze(-1)&bool(strengths.geometry),
                   "appearance":((prepared.weights_appearance*prepared.appearance_valid.any(-1)[...,None,None]).sum(1)>0).squeeze(-1)&bool(strengths.appearance),
                   "contact":(prepared.weights_contact>0).squeeze(-1)&prepared.contact_valid.any(-1)[:,None]&bool(strengths.contact)}
        union = support['geometry']|support['appearance']|support['contact']
        # Only this block's differentiable delta is stored here. Construct the
        # cumulative loss after forward, outside per-block checkpoint recompute.
        self._budget_terms[layer_index] = (delta,hidden_rms.detach(),union)
        reference_rms = torch.linalg.vector_norm(self._reference_hidden.float(),dim=-1,keepdim=True)/math.sqrt(self.hidden_size)
        if strengths.max_ratio is not None:
            rms = torch.linalg.vector_norm(delta.detach().float(),dim=-1,keepdim=True)/math.sqrt(self.hidden_size)
            scale = (strengths.max_ratio*hidden_rms.detach()/rms.clamp_min(1e-8)).clamp_max(1).to(delta.dtype)
            cumulative_scale = self._cumulative_scale(previous,delta*scale,
                strengths.max_ratio*reference_rms*math.sqrt(self.hidden_size)).to(delta.dtype)
            scale = scale*cumulative_scale
            branches = {branch:value*scale for branch,value in branches.items()}
            delta = delta*scale
        self._layer_residuals[layer_index] = {branch:value.detach() for branch,value in branches.items()}
        cumulative = previous+delta
        self.last_diagnostics[layer_index] = {branch:self._stats(value,support[branch],previous_branches[branch]+value,target_hidden,self._reference_hidden)
                                             for branch,value in branches.items()}
        self.last_diagnostics[layer_index]["total"] = self._stats(delta,union,cumulative,target_hidden,self._reference_hidden)
        self._cumulative = cumulative.detach()
        return delta

    def budget_loss(self,max_ratio=.05):
        if not math.isfinite(max_ratio) or max_ratio <= 0:
            raise ValueError("budget max_ratio must be positive and finite")
        terms = []
        cumulative = None
        cumulative_support = None
        if self._reference_hidden is not None:
            reference_rms = torch.linalg.vector_norm(self._reference_hidden.float(),dim=-1,keepdim=True)/math.sqrt(self.hidden_size)
        for layer in self.layer_indices:
            if layer not in self._budget_terms:
                continue
            delta,hidden_rms,support = self._budget_terms[layer]
            cumulative = delta if cumulative is None else cumulative+delta
            cumulative_support = support if cumulative_support is None else cumulative_support|support
            for value,baseline,roi in ((delta,hidden_rms,support),
                                       (cumulative,reference_rms,cumulative_support)):
                if bool(roi.any()):
                    # vector_norm defines the zero gradient, unlike taking sqrt
                    # of mean(square) at the deliberately zero initial output.
                    rms = torch.linalg.vector_norm(value.float(),dim=-1,keepdim=True)/math.sqrt(self.hidden_size)
                    terms.append((rms-baseline*max_ratio).clamp_min(0)[roi].square().mean())
        if not terms:
            return self.query_projection.weight.sum()*0
        return torch.stack(terms).mean()
