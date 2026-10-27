"""Separate modality increments with shared sparse Query/backbone/output heads."""
import torch
from torch import nn
from torch.nn import functional as F
from ..native_face import NativeFacePreparer,gather_source,encode_available
from ..native_hand import NativeHandPreparer
from ..face.content_encoder import pool_arcface_references
from ..face.spatial import render_face_landmark_heatmaps
from ..native_local_condition import RIGID_FACE_SLOTS
from ..native_local_control import NativeLocalController,resolve_local_ownership,target_region_grid
from ..native_dual_identity import ContextDelta
from .contracts import PreparedLocalV5
from .detail import inner_feather


class NativeFacePreparerV5(NativeFacePreparer):
    def forward(self,c,r,*,identity=True,texture=True,geometry=True):
        b=c.person_valid.shape[0];active=c.person_valid&c.face_roi_valid
        boxes=c.face_boxes;span=(boxes[...,2:]-boxes[...,:2]).clamp_min(1e-6)
        points=c.face_landmarks.clone();points[...,:2]=(points[...,:2]-boxes[...,:2][...,None,:])/span[...,None,:]
        pv=c.face_point_valid&active[...,None]&((points[...,:2]>=0)&(points[...,:2]<=1)).all(-1)
        jv=c.jaw_valid&active;ev=c.expression_valid&active
        for row,meta in enumerate(c.metadata):
            for p in range(2):
                if meta['expression_mode']=='off' or (meta['expression_mode']=='source' and not meta.get('expression_retargeted',[False,False])[p]):
                    rigid=torch.zeros(72,dtype=torch.bool,device=points.device);rigid[list(RIGID_FACE_SLOTS)]=True
                    pv[row,p]&=rigid;jv[row,p]=False;ev[row,p]=False
        points[...,2]*=pv
        geo=points.new_zeros(b,2,4,512);gv=torch.zeros_like(active)
        spatial=points.new_zeros(b,2,512,16,16);sv=torch.zeros_like(active)
        if geometry:
            geo,gv=self.geometry_tokenizer(points,pv,c.jaw_pose,jv,c.expression,ev)
            sv=gv&pv.any(-1)
            heat,mask=render_face_landmark_heatmaps(points,gv)
            ids=sv.flatten().nonzero(as_tuple=True)[0]
            if ids.numel(): spatial=spatial.flatten(0,1).index_copy(0,ids,self.spatial_encoder(heat.reshape(-1,72,128,128)[ids],mask.reshape(-1,128,128)[ids])).reshape(b,2,512,16,16)
        av=gather_source(r.arcface_valid,c)&active[...,None]
        dv=gather_source(r.face_dino_valid,c)&active[...,None]
        iv=av.any(-1)&identity;tv=dv.any(-1)&texture
        raw=points.new_zeros(b,2,4,512);tex=points.new_zeros(b,2,8,512)
        # Perceiver needs the raw identity as a query even when its output branch
        # is off. It is still classified exclusively as a TEXTURE increment.
        need_id=iv|(tv&av.any(-1))
        if need_id.any():
            raw=encode_available(self.content_encoder.fusion_face_id.project_identity,
                pool_arcface_references(gather_source(r.arcface,c),av),need_id,(4,512))
        if texture:
            patches=gather_source(r.face_dino,c);count=patches.shape[2]
            encoded=encode_available(self.content_encoder.resampler,patches,dv,(8,512))
            tex=(encoded*dv[...,None,None]).sum(2)/dv.sum(2).clamp_min(1)[...,None,None]
        delta=torch.zeros_like(raw);both=(tv&av.any(-1)).flatten().nonzero(as_tuple=True)[0]
        if both.numel(): delta=delta.flatten(0,1).index_copy(0,both,self.content_encoder.face_perceiver(raw.flatten(0,1)[both],tex.flatten(0,1)[both])).reshape_as(raw)
        tokens=points.new_zeros(b,2,4,16,512);valid=torch.zeros(b,2,4,16,dtype=torch.bool,device=points.device)
        tokens[:,:,0,:4]=raw;valid[:,:,0,:4]=iv[...,None]
        tokens[:,:,1,:8]=tex;valid[:,:,1,:8]=tv[...,None]
        tokens[:,:,1,8:12]=delta;valid[:,:,1,8:12]=(tv&av.any(-1))[...,None]
        tokens[:,:,2,:4]=geo;valid[:,:,2,:4]=gv[...,None]
        slots=gather_source(c.source_slot_ids,c)
        binding=self.source_index_embedding(slots)+self.person_slot_embedding.weight[None]
        tokens=(tokens+binding[:,:,None,None])*valid[...,None]
        return spatial,tokens,valid,sv


class NativeHandPreparerV5(NativeHandPreparer):
    def __init__(self,*,canonical=False):
        super().__init__();self.canonical=canonical
        self.canonical_2d=nn.Sequential(nn.Linear(63,512),nn.SiLU(),nn.Linear(512,512)) if canonical else None

    def forward(self,c,r,*,texture=True,geometry=True):
        from dataclasses import replace
        if not geometry:
            c=replace(c,hand_point_valid=torch.zeros_like(c.hand_point_valid),hand_pose_valid=torch.zeros_like(c.hand_pose_valid),mesh_valid=torch.zeros_like(c.mesh_valid))
        if not texture: r=replace(r,hand_dino_valid=torch.zeros_like(r.hand_dino_valid))
        spatial,old,valid,gm,sv=super().forward(c,r);b=old.shape[0]
        tokens=old.new_zeros(b,4,4,16,512);mask=torch.zeros(b,4,4,16,dtype=torch.bool,device=old.device)
        tokens[:,:,2,:4]=old[:,:,:4];mask[:,:,2,:4]=valid[:,:,:4]
        if texture: tokens[:,:,1,:8]=old[:,:,4:12];mask[:,:,1,:8]=valid[:,:,4:12]
        if self.canonical and geometry:
            # A fresh coordinate-only branch; inherited Hand45/BPS and spatial
            # weights continue to receive their original anatomical convention.
            points=c.hand_keypoints.clone();box=c.hand_boxes;span=(box[...,2:]-box[...,:2]).clamp_min(1e-6)
            points[...,:2]=(points[...,:2]-box[...,:2][...,None,:])/span[...,None,:]
            points[:,:,0,:,0]=1-points[:,:,0,:,0]
            pv=c.hand_point_valid&c.hand_roi_valid[...,None]&c.person_valid[...,None,None]
            usable=pv.any(-1);usable[:,:,1]=False
            value=encode_available(self.canonical_2d,(points*pv[...,None]).flatten(-2),usable,(512,)).flatten(1,2)
            slots=gather_source(c.source_slot_ids,c)
            binding=(self.source_embedding(slots)[:,:,None]+self.person_embedding.weight[None,:,None]
                     +self.side_embedding.weight[None,None]+self.region_embedding).flatten(1,2)
            value=value+binding
            for region in (0,2):
                ok=usable.flatten(1,2)[:,region]
                tokens[:,region,2,1]=torch.where(ok[:,None],value[:,region],tokens[:,region,2,1])
        return spatial,tokens,mask,sv


def prepare_local_v5(face,hand,bridge,c,r,detail_refs,source_latent,options,strength,config,*,source_bank=None):
    b=c.person_valid.shape[0];x=c.face_landmarks;parts=[]
    for family,module,count in (('face',face,2),('hand',hand,4)):
        branches=('face_identity','face_texture','face_geometry') if family=='face' else ('hand_texture','hand_geometry')
        active={s:strength.enabled(s,options) for s in branches}
        if any(active.values()):
            kwargs={'texture':active[family+'_texture'],'geometry':active[family+'_geometry']}
            if family=='face': kwargs['identity']=active['face_identity']
            parts.append(module(c,r,**kwargs))
        else:
            parts.append((x.new_zeros(b,count,512,16,16),x.new_zeros(b,count,4,16,512),
                torch.zeros(b,count,4,16,dtype=torch.bool,device=x.device),torch.zeros(b,count,dtype=torch.bool,device=x.device)))
    spatial,tokens,mask,sv=(torch.cat([p[i] for p in parts],1) for i in range(4))
    detail=x.new_zeros(b,6,512,16,16);dv=torch.zeros(b,6,dtype=torch.bool,device=x.device)
    if strength.enabled('detail',options) and ((options.face and options.face_strength>0) or (options.hand and options.hand_strength>0)):
        if source_bank is None and detail_refs is None: raise ValueError('V5 Detail needs explicit Source sidecar, including unavailable declarations')
        if source_bank is None and source_latent is None: raise ValueError('Reuse the actual primary Qwen64 prefix for Detail')
        from dataclasses import replace
        detail_condition=replace(c,face_roi_valid=c.face_roi_valid if options.face and options.face_strength>0 else torch.zeros_like(c.face_roi_valid),
            hand_roi_valid=c.hand_roi_valid if options.hand and options.hand_strength>0 else torch.zeros_like(c.hand_roi_valid))
        detail,dv=bridge(source_latent,detail_refs,detail_condition) if source_bank is None else bridge.forward_bank(source_bank,detail_condition)
    enabled=torch.tensor([options.face]*2+[options.hand]*4,device=x.device)[None]
    dv&=enabled
    weights=resolve_local_ownership(c,face=options.face,hand=options.hand,tie_tolerance=config.tie_tolerance)
    weights=inner_feather(weights,c.boxes,config.feather_px)
    rv=(mask.flatten(2).any(-1)|dv)&enabled&(weights.flatten(2).sum(-1)>0)
    return PreparedLocalV5(spatial,sv,tokens,mask,detail,dv,c.boxes,weights*rv[...,None,None],rv,c.metadata).validate()


class NativeLocalControllerV5(NativeLocalController):
    def __init__(self,config,*,hand=False):
        super().__init__(config,anatomical_sides=hand);self.hand=hand
        self.identity_attention=ContextDelta()
        self.detail_projection=nn.Linear(512,512)
        self.output_heads=nn.ModuleList(nn.Linear(512,config.hidden_dim,bias=False) for _ in range(6))
        self.static_output=nn.Linear(512,512,bias=False)
        self.stage_films=nn.ModuleList(nn.Linear(1,1024) for _ in range(6)) if config.stage_modulation=='film_rank64' else None
        self.stage_lowrank=nn.ModuleList(nn.Sequential(nn.Linear(512,64,bias=False),nn.SiLU(),nn.Linear(64,512,bias=False)) for _ in range(6)) if self.stage_films is not None else None
        self.hand_fusions=nn.ModuleList(ContextDelta() for _ in range(2)) if hand and config.hand_fusion=='sequential2' else None
        for head in (*self.output_heads,self.static_output): nn.init.zeros_(head.weight)

    def _refine(self,value,slot,sigma):
        if self.stage_films is not None:
            scale,shift=self.stage_films[slot](sigma.float()[:,None]).chunk(2,-1)
            value=value+value*scale[:,None]+shift[:,None]+self.stage_lowrank[slot](value)
        n=self.self_norm(value)
        value=value+self.shared_attention(n,n,n,need_weights=False)[0]
        return value+self.shared_ffn(value)

    def _updates(self,q,p,row,region,cells,grid,slot,sigma,strength,options,layer):
        result=torch.zeros_like(q);family='hand' if self.hand else 'face'
        for mi,name in ((0,'identity'),(1,'texture'),(2,'geometry'),(3,'detail')):
            if self.hand and mi==0: continue
            branch='detail' if mi==3 else family+'_'+name
            gain=strength(branch,layer,sigma,options)
            if mi==3: gain=gain*(options.hand_strength if self.hand else options.face_strength)
            if not gain.any(): continue
            if mi in (2,3) and cells.numel()<self.config.min_spatial_cells: continue
            valid=p.token_valid[row:row+1,region,mi];context=p.tokens[row:row+1,region,mi]
            if mi==3:
                if not p.detail_valid[row,region]: continue
                sampled=F.grid_sample(p.detail[row:row+1,region],grid[row:row+1,region].reshape(1,-1,1,2)[:,cells],align_corners=False)
                update=self.detail_projection(sampled.squeeze(-1).transpose(1,2))
            else:
                if not valid.any() and not (mi==2 and p.spatial_valid[row,region]): continue
                query=q
                if self.hand_fusions is not None and mi==1 and p.token_valid[row,region,2].any():
                    geometry=p.tokens[row:row+1,region,2];gv=p.token_valid[row:row+1,region,2]
                    for fusion in self.hand_fusions: geometry=geometry+fusion(geometry,context,valid,gv)
                    context=geometry;valid=gv
                attention=self.identity_attention if mi==0 else (self.appearance_attention if mi==1 else self.geometry_attention)
                update=attention(query,context,valid,torch.ones(q.shape[:2],dtype=torch.bool,device=q.device))
                if mi==2 and p.spatial_valid[row,region]:
                    sampled=F.grid_sample(p.spatial[row:row+1,region],grid[row:row+1,region].reshape(1,-1,1,2)[:,cells],align_corners=False)
                    update=update+self.spatial_projection(sampled.squeeze(-1).transpose(1,2))
            result=result+self._refine(update,slot,sigma)*gain[:,None,None]
        return result

    def forward(self,hidden,p,*,layer,sigma,target_hw,regions,strength,options):
        if layer not in self.config.injection_layers: return torch.zeros_like(hidden,dtype=torch.float32)
        grid,weights=target_region_grid(p,target_hw);slot=self.config.injection_layers.index(layer)
        result=torch.zeros_like(hidden,dtype=torch.float32)
        for row in range(hidden.shape[0]):
            for region in regions:
                cells=(weights[row,region]>0).nonzero(as_tuple=True)[0]
                if not cells.numel() or not p.region_valid[row,region]: continue
                q=self.query(hidden[row:row+1,cells].float())+self.layer_embedding.weight[slot]
                q=q+self.time(sigma[row:row+1,None].float())[:,None]
                if self.side is not None: q=q+self.side.weight[(region-2)%2]
                update=self._updates(q,p,row,region,cells,grid,slot,sigma[row:row+1],strength,options,layer)
                delta=self.output_heads[slot](update)*weights[row:row+1,region,cells,None]
                result=result.reshape(-1,self.config.hidden_dim).index_add(0,cells+row*hidden.shape[1],delta[0]).reshape_as(result)
        return result

    def static_delta_v5(self,p,*,layer,sigma,target_hw,regions,strength,options):
        """Static ablation: no current Hidden, same independent branch controls."""
        grid,weights=target_region_grid(p,target_hw);b=p.tokens.shape[0];n=target_hw[0]*target_hw[1]
        result=p.tokens.new_zeros(b,n,512);slot=self.config.injection_layers.index(layer)
        for row in range(b):
            for region in regions:
                cells=(weights[row,region]>0).nonzero(as_tuple=True)[0]
                if not cells.numel() or not p.region_valid[row,region]: continue
                q=p.tokens.new_zeros(1,cells.numel(),512)
                update=self._updates(q,p,row,region,cells,grid,slot,sigma[row:row+1],strength,options,layer)
                delta=self.static_output(update)*weights[row:row+1,region,cells,None]
                result=result.reshape(-1,512).index_add(0,cells+row*n,delta[0]).reshape_as(result)
        return result
