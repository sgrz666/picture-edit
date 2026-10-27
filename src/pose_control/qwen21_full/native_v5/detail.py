"""Primary Qwen64 latent -> six target-local detail maps, never a second backbone."""
import torch
from torch import nn
from torch.nn import functional as F


class NativeDetailAppearanceBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(64,256,1),nn.GroupNorm(16,256),nn.SiLU(),
            nn.Conv2d(256,512,3,padding=1),nn.GroupNorm(32,512),nn.SiLU())
        self.source_embedding=nn.Embedding(16,512)
        self.region_embedding=nn.Embedding(3,512)

    def forward_bank(self,bank,condition):
        """Reuse this same bridge per region on its actual independent source."""
        bank.validate(require_latent=True);b=condition.person_valid.shape[0]
        first=bank.sources[0].latent
        out=first.new_zeros(b,6,512,16,16,dtype=torch.float32)
        valid=torch.zeros(b,6,dtype=torch.bool,device=first.device)
        yy,xx=torch.meshgrid((torch.arange(16,device=first.device)+.5)/16,
            (torch.arange(16,device=first.device)+.5)/16,indexing='ij')
        unit=torch.stack((xx,yy),-1)
        for row in range(b):
            for region,p in enumerate((0,1,0,0,1,1)):
                if not condition.person_valid[row,p] or not condition.region_valid[row,region]: continue
                entry=bank.sources[int(bank.target_qwen_image_indices[row,p])]
                if entry.detail is None: raise ValueError('Each source requires its own explicit Detail availability sidecar')
                kind=0 if region<2 else 1+(region-2)%2
                person=int(bank.target_person_indices[row,p]);src_region=person if kind==0 else 2+2*person+kind-1
                refs=entry.detail
                if not refs.valid[row,src_region]: continue
                latent=entry.latent_map(row);box=refs.boxes[row,src_region]
                grid=(box[:2]+unit*(box[2:]-box[:2]))[None]*2-1
                support=F.interpolate(refs.support[row:row+1,src_region:src_region+1].float(),size=entry.token_hw,mode='area')
                denominator=F.grid_sample(support,grid,align_corners=False)
                if not (denominator>1e-6).any(): continue
                numerator=F.grid_sample(latent*support,grid,align_corners=False)
                value=self.encoder(numerator/denominator.clamp_min(1e-6))
                binding=self.source_embedding.weight[entry.source_slot_id]+self.region_embedding.weight[kind]
                out[row,region]=(value[0]+binding[:,None,None])*(denominator[0]>1e-6)
                valid[row,region]=True
        return out,valid

    def forward(self,latent,references,condition):
        references.validate();b=condition.person_valid.shape[0]
        if latent.ndim!=4 or latent.shape[:2]!=(b,64) or not torch.isfinite(latent).all():
            raise ValueError('Detail requires the actual primary Source [B,64,Hs/16,Ws/16] latent')
        if references.support.shape[-2:]!=(latent.shape[-2]*16,latent.shape[-1]*16): raise ValueError('Source latent/Canvas mismatch')
        if not torch.equal(references.source_indices,condition.source_indices): raise ValueError('Detail source mapping differs')
        out=latent.new_zeros(b,6,512,16,16,dtype=torch.float32);valid=torch.zeros(b,6,dtype=torch.bool,device=latent.device)
        yy,xx=torch.meshgrid((torch.arange(16,device=latent.device)+.5)/16,(torch.arange(16,device=latent.device)+.5)/16,indexing='ij')
        unit=torch.stack((xx,yy),-1)
        for row in range(b):
            for region,p in enumerate((0,1,0,0,1,1)):
                if not condition.person_valid[row,p] or not condition.region_valid[row,region]: continue
                source=int(condition.source_indices[row,p]);kind=0 if region<2 else 1+(region-2)%2
                src_region=source if kind==0 else 2+2*source+kind-1
                if not references.valid[row,src_region]: continue
                box=references.boxes[row,src_region];grid=(box[:2]+unit*(box[2:]-box[:2]))[None]*2-1
                support=F.interpolate(references.support[row:row+1,src_region:src_region+1].float(),size=latent.shape[-2:],mode='area')
                denominator=F.grid_sample(support,grid,align_corners=False)
                if not (denominator>1e-6).any(): continue
                numerator=F.grid_sample(latent[row:row+1].float()*support,grid,align_corners=False)
                sampled=numerator/denominator.clamp_min(1e-6)
                value=self.encoder(sampled)
                binding=self.source_embedding(condition.source_slot_ids[row,source])+self.region_embedding.weight[kind]
                out[row,region]=(value[0]+binding[:,None,None])*(denominator[0]>1e-6)
                valid[row,region]=True
        return out,valid


def inner_feather(weights,boxes,max_radius):
    """Pure Torch interior erosion ramp. It never expands visibility/ownership."""
    if max_radius==0: return weights
    b,r,h,w=weights.shape;out=torch.zeros_like(weights)
    for i in range(b):
        for j in range(r):
            side=float(torch.min((boxes[i,j,2:]-boxes[i,j,:2])*weights.new_tensor([w,h])))
            radius=max(1,min(max_radius,int(max(0.,side)*.1)))
            mask=(weights[i:i+1,j:j+1]>0).float();ramp=mask.clone();eroded=mask
            for _ in range(radius):
                eroded=-F.max_pool2d(-F.pad(eroded,(1,1,1,1),value=0.),3,1)
                ramp=ramp+eroded
            out[i,j]=weights[i,j]*ramp[0,0]/(radius+1)
    return out
