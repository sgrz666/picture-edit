"""V4 Hand: independent pose/BPS/2D/appearance validity and native sparse output."""
import torch
from torch import nn
from torch.nn import functional as F

from .hand.vton_layers import HandStructureProjector,HandResampler,OneLinearLN
from .hand.geometry import axis_angle45_to_6d96,hand_heatmaps
from .native_face import gather_source,encode_available


def roi_sample(value,boxes,size=16,mode='bilinear'):
    # value [B,R,C,H,W]; boxes remain unclipped; support is checked separately.
    b,r,c,h,w=value.shape
    yy,xx=torch.meshgrid((torch.arange(size,device=value.device,dtype=value.dtype)+.5)/size,
                         (torch.arange(size,device=value.device,dtype=value.dtype)+.5)/size,indexing='ij')
    base=torch.stack((xx,yy),-1)
    grid=boxes[...,:2][...,None,None,:]+base*(boxes[...,2:]-boxes[...,:2])[...,None,None,:]
    return F.grid_sample(value.flatten(0,1),grid.reshape(b*r,size,size,2)*2-1,mode=mode,
                         align_corners=False).reshape(b,r,c,size,size)


class NativeHandPreparer(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure=HandStructureProjector(512)
        self.dino_projection=OneLinearLN(1536,512)
        self.resampler=HandResampler(dim=512,depth=2,heads=8)
        self.spatial=nn.Sequential(nn.Conv2d(23,256,3,padding=1),nn.SiLU(),nn.Conv2d(256,512,3,padding=1),nn.SiLU())
        self.source_embedding=nn.Embedding(16,512)
        self.person_embedding=nn.Embedding(2,512)
        self.side_embedding=nn.Embedding(2,512)
        self.region_embedding=nn.Parameter(torch.randn(512)*.02)
        self.null_appearance=nn.Parameter(torch.zeros(8,512))
        self.null_pose=nn.Parameter(torch.zeros(512))
        self.keypoint_availability=nn.Linear(21,512,bias=False)
        self.modality_embedding=nn.Linear(4,512,bias=False)

    def forward(self,condition,references):
        b=condition.person_valid.shape[0]
        active=condition.hand_roi_valid & condition.person_valid[...,None]
        points=condition.hand_keypoints.clone()
        boxes=condition.hand_boxes
        span=(boxes[...,2:]-boxes[...,:2]).clamp_min(1e-6)
        points[...,:2]=(points[...,:2]-boxes[...,:2][...,None,:])/span[...,None,:]
        pv=condition.hand_point_valid & active[...,None] & ((points[...,:2]>=0)&(points[...,:2]<=1)).all(-1)
        points[...,2]*=pv
        has_2d=pv.any(-1)
        has_pose=condition.hand_pose_valid & active
        has_mesh=condition.mesh_valid & active
        pose_features=self.null_pose.expand(b,2,2,-1).clone()
        ids=has_pose.flatten().nonzero(as_tuple=True)[0]
        if ids.numel():
            encoded=axis_angle45_to_6d96(condition.hand_pose.reshape(-1,45)[ids])
            pose_features=pose_features.reshape(-1,512).index_copy(0,ids,self.structure.mano_6d(encoded)).reshape(b,2,2,512)
        bps=self.structure.null_bps.expand(b,2,2,-1).clone()
        ids=has_mesh.flatten().nonzero(as_tuple=True)[0]
        if ids.numel(): bps=bps.reshape(-1,512).index_copy(0,ids,self.structure.mano_bps(condition.mesh_bps.reshape(-1,1024)[ids])).reshape(b,2,2,512)
        side=torch.arange(2,device=points.device,dtype=points.dtype).reshape(1,1,2,1).expand(b,2,-1,-1)
        has_3d=has_pose|has_mesh
        three=points.new_zeros(b,2,2,512)
        if has_3d.any():
            handtype=encode_available(self.structure.handtype,side,has_3d,(512,))
            three=encode_available(self.structure.combined,torch.cat((bps,pose_features,handtype),-1),has_3d,(512,))
        two=encode_available(self.structure.mano_2d,(points[...,:2]*pv[...,None]).flatten(-2),has_2d,(512,))
        two=two+encode_available(self.keypoint_availability,points[...,2],has_2d,(512,))
        slots=gather_source(condition.source_slot_ids,condition)
        binding=self.source_embedding(slots)[:,:,None]+self.person_embedding.weight[None,:,None]+self.side_embedding.weight[None,None]+self.region_embedding
        geometry=torch.stack((three,two,(three+two)*.5,three-two),-2)+binding[...,None,:]
        gm=torch.stack((has_3d,has_2d,has_3d&has_2d,has_3d&has_2d),-1)
        geometry*=gm[...,None]
        heat=hand_heatmaps(points)
        masks=heat.amax(-3)
        z=roi_sample(condition.local_z[:,2:,None],boxes.flatten(1,2),mode='nearest').reshape(b,2,2,1,16,16)
        zv=roi_sample(condition.local_z_valid[:,2:,None].float(),boxes.flatten(1,2),mode='nearest').reshape(b,2,2,1,16,16)>.5
        depth=torch.where(zv,torch.asinh((z-condition.wrist_z[...,None,None,None])/condition.body_height[...,None,None,None,None].clamp_min(1e-6)),0.)
        spatial_input=torch.cat((heat,masks[...,None,:,:],depth),-3)
        modality=torch.stack((has_2d,has_pose,has_mesh,zv.flatten(-3).any(-1)),-1).float()
        sv=has_2d|has_3d
        spatial=encode_available(self.spatial,spatial_input,sv,(512,16,16))
        modality_features=encode_available(self.modality_embedding,modality,sv,(512,))
        spatial=(spatial+(three+two+binding+modality_features)[...,None,None])*sv[...,None,None,None]
        patches=gather_source(references.hand_dino,condition)
        valid=gather_source(references.hand_dino_valid,condition)&active[...,None]
        r=patches.shape[3]
        flat=patches.reshape(-1,256,1536)
        ids=valid.flatten().nonzero(as_tuple=True)[0]
        appearance=flat.new_zeros(flat.shape[0],8,512)
        if ids.numel(): appearance=appearance.index_copy(0,ids,self.resampler(self.dino_projection(flat[ids])))
        appearance=appearance.reshape(b,2,2,r,8,512)
        appearance=(appearance*valid[...,None,None]).sum(3)/valid.sum(3).clamp_min(1)[...,None,None]
        appearance=torch.where(valid.any(-1)[...,None,None],appearance,self.null_appearance)+binding[...,None,:]
        # A Null appearance is explicit context only when another trustworthy hand
        # condition exists; an empty region never becomes an active fake hand.
        hand_active=active & (sv|valid.any(-1))
        am=hand_active[...,None].expand(-1,-1,-1,8)
        tokens=torch.cat((geometry,appearance,geometry.new_zeros(b,2,2,4,512)),-2)
        mask=torch.cat((gm,am,torch.zeros(b,2,2,4,dtype=torch.bool,device=points.device)),-1)
        geometry_mask=torch.zeros_like(mask);geometry_mask[...,:4]=gm
        return spatial.flatten(1,2),tokens.flatten(1,2),mask.flatten(1,2),geometry_mask.flatten(1,2),sv.flatten(1,2)
