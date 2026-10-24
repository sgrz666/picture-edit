"""Mask-aware V4 wrappers around compatible V6.5 face feature modules."""
import torch
from torch import nn
from torch.nn import functional as F

from .face.geometry import FaceGeometryTokenizer
from .face.content_encoder import FaceContentEncoder, pool_arcface_references
from .face.spatial import FaceSpatialEncoder, render_face_landmark_heatmaps
from .native_local_condition import RIGID_FACE_SLOTS


def gather_source(value, condition):
    rows=torch.arange(value.shape[0],device=value.device)[:,None]
    return value[rows,condition.source_indices.clamp_min(0)]


class NativeFaceGeometryTokenizer(FaceGeometryTokenizer):
    def __init__(self):
        super().__init__()
        self.jaw_partial=nn.Sequential(nn.Linear(3,512),nn.SiLU(),nn.Linear(512,512))
        self.expression_partial=nn.Sequential(nn.Linear(10,512),nn.SiLU(),nn.Linear(512,512))

    def forward(self, points, point_valid, jaw, jaw_valid, expression, expression_valid):
        leading=points.shape[:-2]
        index=self.landmark_index(torch.arange(72,device=points.device)).expand(*leading,-1,-1)
        encoded=self.point_encoder(torch.cat((points,index),-1))*points[...,2,None]*point_valid[...,None]
        complete=jaw_valid & expression_valid
        context=torch.cat((encoded,self.parameter_encoder(torch.cat((jaw,expression),-1))[...,None,:],
            self.jaw_partial(jaw)[...,None,:],self.expression_partial(expression)[...,None,:]),-2).reshape(-1,75,512)
        valid=torch.cat((point_valid,complete[...,None],(jaw_valid & ~expression_valid)[...,None],
                         (expression_valid & ~jaw_valid)[...,None]),-1).reshape(-1,75)
        active=valid.any(1).nonzero(as_tuple=True)[0]
        result=points.new_zeros(context.shape[0],4,512)
        if active.numel():
            q=self.queries[None].expand(active.numel(),-1,-1)
            attended=self.attention(q,context[active],context[active],key_padding_mask=~valid[active],need_weights=False)[0]
            tokens=q+attended
            tokens=self.output_norm(tokens+self.feed_forward(tokens))
            result=result.index_copy(0,active,tokens)
        return result.reshape(*leading,4,512),valid.any(1).reshape(*leading)


class NativeFacePreparer(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry_tokenizer=NativeFaceGeometryTokenizer()
        self.content_encoder=FaceContentEncoder()
        self.spatial_encoder=FaceSpatialEncoder()
        self.source_index_embedding=nn.Embedding(16,512)
        self.person_slot_embedding=nn.Embedding(2,512)

    def forward(self, condition, references):
        b=condition.person_valid.shape[0]
        active=condition.person_valid & condition.face_roi_valid
        boxes=condition.face_boxes
        span=(boxes[...,2:]-boxes[...,:2]).clamp_min(1e-6)
        points=condition.face_landmarks.clone()
        points[...,:2]=(points[...,:2]-boxes[...,:2][...,None,:])/span[...,None,:]
        valid=condition.face_point_valid & active[...,None] & ((points[...,:2]>=0)&(points[...,:2]<=1)).all(-1)
        jv,ev=condition.jaw_valid & active,condition.expression_valid & active
        for i,meta in enumerate(condition.metadata):
            mode=meta['expression_mode']
            retarget=meta.get('expression_retargeted',[False,False])
            for p in range(2):
                if mode=='off' or (mode=='source' and not retarget[p]):
                    rigid=torch.zeros(72,dtype=torch.bool,device=valid.device)
                    rigid[list(RIGID_FACE_SLOTS)]=True
                    valid[i,p] &= rigid
                    jv[i,p]=False;ev[i,p]=False
        points[...,2] *= valid
        geometry,gv=self.geometry_tokenizer(points,valid,condition.jaw_pose,jv,condition.expression,ev)
        arc=gather_source(references.arcface,condition)
        av=gather_source(references.arcface_valid,condition) & active[...,None]
        dv=gather_source(references.face_dino_valid,condition) & active[...,None]
        dino=gather_source(references.face_dino,condition)
        identity=self.content_encoder.fusion_face_id.project_identity(pool_arcface_references(arc,av))
        id_valid=av.any(-1)
        identity=identity*id_valid[...,None,None]
        r=dino.shape[2]
        flat=dino.reshape(b*2*r,256,1536)
        ids=dv.flatten().nonzero(as_tuple=True)[0]
        encoded=flat.new_zeros(flat.shape[0],8,512)
        if ids.numel(): encoded=encoded.index_copy(0,ids,self.content_encoder.resampler(flat[ids]))
        encoded=encoded.reshape(b,2,r,8,512)
        texture=(encoded*dv[...,None,None]).sum(2)/dv.sum(2).clamp_min(1)[...,None,None]
        texture_valid=dv.any(-1)
        delta=torch.zeros_like(identity)
        both=(id_valid & texture_valid).flatten().nonzero(as_tuple=True)[0]
        if both.numel():
            update=self.content_encoder.face_perceiver(identity.reshape(-1,4,512)[both],texture.reshape(-1,8,512)[both])
            delta=delta.reshape(-1,4,512).index_copy(0,both,update).reshape(b,2,4,512)
        identity=identity+delta
        slots=gather_source(condition.source_slot_ids,condition)
        binding=self.source_index_embedding(slots)+self.person_slot_embedding.weight[None]
        tokens=torch.cat((identity+binding[...,None,:],texture+binding[...,None,:],geometry+binding[...,None,:]),-2)
        mask=torch.cat((id_valid[...,None].expand(-1,-1,4),texture_valid[...,None].expand(-1,-1,8),gv[...,None].expand(-1,-1,4)),-1)
        tokens=tokens*mask[...,None]
        heat,masks=render_face_landmark_heatmaps(points,gv)
        spatial=points.new_zeros(b*2,512,16,16)
        ids=gv.flatten().nonzero(as_tuple=True)[0]
        if ids.numel():
            value=self.spatial_encoder(heat.reshape(-1,72,128,128)[ids],masks.reshape(-1,128,128)[ids])
            spatial=spatial.index_copy(0,ids,value)
        geometry_mask=torch.zeros_like(mask);geometry_mask[...,12:]=mask[...,12:]
        return spatial.reshape(b,2,512,16,16),tokens,mask,geometry_mask,gv
