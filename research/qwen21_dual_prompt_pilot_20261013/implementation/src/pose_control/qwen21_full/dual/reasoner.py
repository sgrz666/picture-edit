"""Per-person geometry, symmetric mutual attention, and sparse contact graph."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from ..reasoning.config import AdapterReasoningConfig
from ..reasoning.person_geometry import PersonGeometryReasoner
from .conditions import GeometryCondition

# head, torso, upper/lower arms, hands, upper/lower legs, feet (shared schema14).
ANATOMICAL_EDGES = ((0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
                    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13))


class MutualGeometryAttention(nn.Module):
    """One shared direction with directed relative geometry and world distances."""
    def __init__(self,width,heads=8):
        super().__init__()
        self.width, self.heads = width,heads
        self.query_norm,self.context_norm = nn.LayerNorm(width),nn.LayerNorm(width)
        self.query,self.key,self.value = (nn.Linear(width,width) for _ in range(3))
        self.output = nn.Linear(width,width)
        self.relative = nn.Linear(14,width)
        self.relative_bias = nn.Linear(14,heads*4,bias=False)
        self.distance_bias = nn.Sequential(nn.Linear(4,width),nn.SiLU(),nn.Linear(width,heads))
        self.ffn = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))

    def forward(self,query,context,query_valid,context_valid,query_pos,context_pos,relative):
        rows = (query_valid.any(-1) & context_valid.any(-1)).nonzero(as_tuple=False).flatten()
        output = query
        if rows.numel():
            qbase,kvbase = query.index_select(0,rows),context.index_select(0,rows)
            relation = relative.index_select(0,rows)
            q = self.query(self.query_norm(qbase)+self.relative(relation)[:,None])
            k = self.key(self.context_norm(kvbase))
            v = self.value(self.context_norm(kvbase)+self.relative(relation)[:,None])
            shape = (rows.numel(),14,self.heads,self.width//self.heads)
            q,k,v = (tensor.reshape(shape).transpose(1,2) for tensor in (q,k,v))
            delta = context_pos.index_select(0,rows)[:,None]-query_pos.index_select(0,rows)[:,:,None]
            distance = delta.norm(dim=-1,keepdim=True)
            pair = torch.cat((delta,distance),dim=-1)
            bias = self.distance_bias(pair).permute(0,3,1,2)
            coeff = self.relative_bias(relation).reshape(-1,self.heads,4)
            bias = bias+torch.einsum("bhc,bqkc->bhqk",coeff,pair)
            score = q@k.transpose(-1,-2)/math.sqrt(self.width//self.heads)+bias
            score = score.masked_fill(~context_valid.index_select(0,rows)[:,None,None],float("-inf"))
            attn = score.float().softmax(-1).to(v.dtype)
            update = self.output((attn@v).transpose(1,2).reshape(-1,14,self.width))
            result = qbase+update
            result = (result+self.ffn(result))*query_valid.index_select(0,rows)[...,None]
            output = output.index_copy(0,rows,result)
        return output*query_valid[...,None]


class ContactGraphLayer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.heads,self.width = 8,width
        self.node_norm,self.edge_norm = nn.LayerNorm(width),nn.LayerNorm(width)
        self.query,self.key,self.value = (nn.Linear(width,width) for _ in range(3))
        self.edge_bias = nn.Linear(width,8)
        self.edge_value = nn.Linear(width,width)
        self.output = nn.Linear(width,width)
        self.edge_update = nn.Sequential(nn.Linear(width*3,width*2),nn.SiLU(),nn.Linear(width*2,width))
        self.ffn = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
        adjacency = torch.eye(28,dtype=torch.bool)
        for offset in (0,14):
            for left,right in ANATOMICAL_EDGES:
                adjacency[left+offset,right+offset] = adjacency[right+offset,left+offset] = True
        self.register_buffer("adjacency",adjacency,persistent=False)

    def forward(self,nodes,node_valid,endpoints,edges,edge_valid,confidence):
        norm = self.node_norm(nodes)
        batch,count,width = norm.shape
        shape = (batch,count,self.heads,width//self.heads)
        q,k,v = (linear(norm).reshape(shape).transpose(1,2) for linear in (self.query,self.key,self.value))
        score = q@k.transpose(-1,-2)/math.sqrt(width//self.heads)
        adjacency = self.adjacency[None].expand(batch,-1,-1).clone()
        left,right = endpoints.unbind(-1)
        gather = lambda index: norm.gather(1,index[...,None].expand(-1,-1,norm.shape[-1]))
        left_node,right_node = gather(left),gather(right)
        edge_norm = self.edge_norm(edges)
        confidence = confidence.clamp(0,1)*edge_valid
        edge_active = edge_valid & (confidence>0)
        bias = self.edge_bias(edge_norm)+confidence.clamp_min(1e-8).log()[...,None]
        values = self.edge_value(edge_norm)*confidence[...,None]
        pair_bias = norm.new_zeros(batch,count*count,self.heads)
        pair_values = norm.new_zeros(batch,count*count,width)
        # Store both directions at the explicitly named physical endpoints.
        for src,dst in ((left,right),(right,left)):
            indices = src*count+dst
            pair_bias = pair_bias.scatter_add(1,indices[...,None].expand_as(bias),bias*edge_active[...,None])
            pair_values = pair_values.scatter_add(1,indices[...,None].expand_as(values),values*edge_active[...,None])
            added = torch.zeros_like(adjacency.flatten(1),dtype=torch.long).scatter_add(1,indices,edge_active.long())
            adjacency = adjacency | added.reshape_as(adjacency).bool()
        score = score+pair_bias.reshape(batch,count,count,self.heads).permute(0,3,1,2)
        allowed = adjacency & node_valid[:,:,None] & node_valid[:,None,:]
        # Invalid queries use a harmless self row, then are exactly zeroed.
        allowed = allowed | (torch.eye(count,device=norm.device,dtype=torch.bool)[None] & ~node_valid[:,:,None])
        attention = score.masked_fill(~allowed[:,None],float('-inf')).float().softmax(-1).to(v.dtype)
        edge_values = pair_values.reshape(batch,count,count,self.heads,width//self.heads).permute(0,3,1,2,4)
        update = attention@v+(attention[...,None]*edge_values).sum(-2)
        update = self.output(update.transpose(1,2).reshape(batch,count,width))
        nodes = nodes+update
        nodes = (nodes+self.ffn(nodes))*node_valid[...,None]
        edges = (edges+self.edge_update(torch.cat((left_node,right_node,edge_norm),dim=-1)))*edge_active[...,None]
        return nodes,edges


class DualGeometryReasoner(nn.Module):
    def __init__(self,width=512):
        super().__init__()
        self.width = width
        self.person_reasoner = PersonGeometryReasoner(AdapterReasoningConfig(hidden_dim=width))
        self.global32 = nn.Sequential(nn.Linear(32,512),nn.SiLU(),nn.Linear(512,4*256))
        self.relative14 = nn.Sequential(nn.Linear(14,256),nn.SiLU(),nn.Linear(256,256))
        # A single shared offset can inherit the old A role; no A/B ID embeddings.
        self.shared_role_offset = nn.Parameter(torch.zeros(width))
        self.part_descriptor = nn.Sequential(nn.Linear(26,width),nn.SiLU(),nn.Linear(width,width))
        self.part_embedding = nn.Embedding(14,width)
        self.mutual = nn.ModuleList(MutualGeometryAttention(width) for _ in range(2))
        self.contact_features = nn.Sequential(nn.Linear(15,width),nn.SiLU(),nn.Linear(width,width))
        self.contact_endpoint = nn.Linear(width*2,width)
        self.contact_graph = nn.ModuleList(ContactGraphLayer(width) for _ in range(2))
        self.node_to_dense = nn.Linear(width,width,bias=False)

    @staticmethod
    def _part_shape(part):
        height,width = part.shape[-2:]
        y,x = torch.meshgrid(torch.linspace(-1,1,height,device=part.device,dtype=part.dtype),
                             torch.linspace(-1,1,width,device=part.device,dtype=part.dtype),indexing="ij")
        mass = part.sum((-2,-1)).clamp_min(1e-8)
        mean_x = (part*x).sum((-2,-1))/mass
        mean_y = (part*y).sum((-2,-1))/mass
        var_x = (part*x.square()).sum((-2,-1))/mass-mean_x.square()
        var_y = (part*y.square()).sum((-2,-1))/mass-mean_y.square()
        covariance = (part*x*y).sum((-2,-1))/mass-mean_x*mean_y
        area = part.mean((-2,-1))
        return torch.stack((area,mean_x,mean_y,var_x.clamp_min(0),var_y.clamp_min(0),covariance),dim=-1)

    def forward(self,spatial,mask,geometry: GeometryCondition):
        batch = spatial.shape[0]
        target_hw = (spatial.shape[-2]//2,spatial.shape[-1]//2)
        rows = geometry.person_valid.flatten().nonzero(as_tuple=False).flatten()
        select = lambda tensor: tensor.flatten(0,1).index_select(0,rows)
        globals = self.global32(select(geometry.global32)).reshape(-1,4,256)
        task = self.relative14(select(geometry.relative14))[:,None]
        output = self.person_reasoner(select(spatial),select(mask),globals,task)
        tokens = (output.tokens+self.shared_role_offset[None,None])*output.token_mask[...,None]
        dense = tokens.new_zeros(batch*2,math.prod(target_hw),self.width).index_copy(0,rows,tokens)
        dense = dense.reshape(batch,2,math.prod(target_hw),self.width)
        membership = F.adaptive_avg_pool2d(geometry.part.flatten(0,1),target_hw).reshape(batch,2,14,-1)
        pooled = torch.einsum("bpk n,bpn d->bpk d",membership,dense)/membership.sum(-1,keepdim=True).clamp_min(1e-8)
        # global32[:10] is the source person's shape beta, not the pose donor's.
        source_shape = geometry.global32[...,:10,None].transpose(-1,-2).expand(-1,-1,14,-1)
        descriptor = torch.cat((geometry.part_positions,self._part_shape(geometry.part),
            geometry.part_confidence[...,None],geometry.part_orientations,geometry.part_joints,source_shape),dim=-1)
        node_valid = geometry.part_valid & geometry.person_valid[...,None]
        nodes = (pooled+self.part_descriptor(descriptor)+self.part_embedding.weight[None,None])*node_valid[...,None]
        initial_nodes = nodes
        # Directions read the same previous state at each layer.
        for layer in self.mutual:
            previous = nodes
            nodes = torch.stack(tuple(layer(previous[:,slot],previous[:,1-slot],
                    node_valid[:,slot],node_valid[:,1-slot],geometry.part_positions[:,slot],
                    geometry.part_positions[:,1-slot],geometry.relative14[:,slot]) for slot in (0,1)),dim=1)
        count = geometry.contact_endpoints.shape[1]
        contacts = nodes.new_zeros(batch,count,self.width)
        contact_rows = geometry.contact_valid.any(-1).nonzero(as_tuple=False).flatten()
        if contact_rows.numel():
            flat = nodes.index_select(0,contact_rows).flatten(1,2)
            valid = node_valid.index_select(0,contact_rows).flatten(1,2)
            endpoints = geometry.contact_endpoints.index_select(0,contact_rows).clamp(0,27)
            edge_valid = geometry.contact_valid.index_select(0,contact_rows)
            confidence = geometry.contact_features.index_select(0,contact_rows)[...,12].clamp(0,1)
            left,right = endpoints.unbind(-1)
            gather = lambda index: flat.gather(1,index[...,None].expand(-1,-1,self.width))
            edge = self.contact_features(geometry.contact_features.index_select(0,contact_rows))
            edge = (edge+self.contact_endpoint(torch.cat((gather(left),gather(right)),dim=-1)))*edge_valid[...,None]
            for layer in self.contact_graph:
                flat,edge = layer(flat,valid,endpoints,edge,edge_valid,confidence)
            nodes = nodes.index_copy(0,contact_rows,flat.reshape(-1,2,14,self.width))
            contacts = contacts.index_copy(0,contact_rows,edge.to(contacts.dtype))
        # Amodal membership returns each node update only to that person's own grid.
        update = self.node_to_dense(nodes-initial_nodes)
        dense = dense+torch.einsum("bpkn,bpkd->bpnd",membership,update)
        dense = dense*geometry.person_valid[...,None,None]
        return dense,nodes,node_valid,contacts,geometry.contact_valid,target_hw
