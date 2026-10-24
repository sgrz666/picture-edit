"""Explicit local velocity/auxiliary objectives; default is unchanged velocity-only."""
from dataclasses import dataclass,asdict
import torch
from torch.nn import functional as F
from ..native_depth_runtime import native_velocity_loss
from ..face_loss_alignment import build_arcface_alignment_grid,align_face_rgb


@dataclass(frozen=True)
class LocalLossConfig:
    profile: str='velocity_only'
    auxiliary_every: int=4
    auxiliary_sigma_max: float=.35

    def __post_init__(self):
        if self.profile not in ('velocity_only','local_velocity','local_aux') or type(self.auxiliary_every)!=int or self.auxiliary_every<1 or not 0<self.auxiliary_sigma_max<=1:
            raise ValueError('Unknown local loss profile/auxiliary gate')
    @property
    def coefficients(self): return (0.,0.,0.,0.) if self.profile=='velocity_only' else ((.5,.5,0.,0.) if self.profile=='local_velocity' else (.5,.5,.1,.05))


def region_velocity(squared,weight):
    denominator=weight.sum()*squared.shape[-1]
    if not bool(denominator>0): return squared.sum()*0
    return (squared*weight[...,None]).sum()/denominator


def local_velocity_loss(prediction,target,coverage,supervision,config):
    base=native_velocity_loss(prediction,target,coverage);squared=(prediction.float()-target.float()).square()
    supervision.validate()
    if not supervision.paired.all(): raise ValueError('Unpaired/retargeted RGB cannot supervise V5')
    weights=F.avg_pool2d(supervision.weights,16).flatten(2)
    if weights.shape[0]!=prediction.shape[0] or weights.shape[-1]!=prediction.shape[1]: raise ValueError('Loss grid differs from actual Qwen target')
    # Pixel supports already carry canvas/owner coverage: intersect, do not
    # multiply partial boundary coverage twice.
    face_weight=torch.minimum(weights[:,:2].sum(1),coverage.float())
    hand_weight=torch.minimum(weights[:,2:].sum(1),coverage.float())
    face=region_velocity(squared,face_weight);hand=region_velocity(squared,hand_weight)
    f,h,_,_=config.coefficients
    total=base+f*face+h*hand
    return total,{'velocity':base,'face_velocity':face,'hand_velocity':hand,
        'face_area':face_weight.sum(),'hand_area':hand_weight.sum()}


def decode_qwen64(pipe,tokens,hw):
    if tokens.shape[-1]!=64 or tokens.shape[1]!=(hw[0]//16)*(hw[1]//16): raise ValueError('Decode actual Qwen64 target grid')
    latents=pipe._unpack_latents(tokens,*hw,pipe.vae_scale_factor).to(pipe.vae.dtype)
    mean=torch.tensor(pipe.vae.config.latents_mean,device=latents.device,dtype=latents.dtype).view(1,-1,1,1,1)
    std=torch.tensor(pipe.vae.config.latents_std,device=latents.device,dtype=latents.dtype).view(1,-1,1,1,1)
    pixels=(pipe.vae.decode(latents*std+mean,return_dict=False)[0][:,:,0].float()*.5+.5).clamp(0,1)
    return pixels[:,:3]*pixels[:,3:4]+(1-pixels[:,3:4]) if pixels.shape[1]==4 else pixels[:,:3]


def frozen_evaluators(identity_path,perceptual_path,*,device,sha):
    from ..native_full_assets import sha256
    paths=(identity_path,perceptual_path)
    if len(sha)!=2 or any(sha256(p)!=v for p,v in zip(paths,sha)): raise ValueError('Pinned evaluator SHA differs')
    models=[]
    for path in paths:
        model=torch.jit.load(str(path),map_location=device).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        models.append(model)
    return tuple(models)


def _feature(model,image):
    value=model(image.float());return (value[-1] if isinstance(value,(tuple,list)) else value).flatten(1)


def auxiliary_loss(pipe,noisy,prediction,sigma,target_pixels,supervision,config,*,step,evaluators):
    zero=prediction.sum()*0;_,_,identity_weight,perceptual_weight=config.coefficients
    eligible=supervision.auxiliary_valid & (sigma[:,None]<=config.auxiliary_sigma_max)
    if not identity_weight or step%config.auxiliary_every or not eligible.any(): return zero,{'identity':zero,'perceptual':zero,'eligible':0}
    if evaluators is None or any(p.requires_grad for model in evaluators for p in model.parameters()): raise ValueError('Auxiliary losses require explicitly frozen evaluators')
    x0=noisy.float()-sigma[:,None,None]*prediction.float()
    decoded=decode_qwen64(pipe,x0,target_pixels.shape[-2:])
    ii=[];pp=[]
    for row,person in eligible.nonzero().tolist():
        grid=build_arcface_alignment_grid(supervision.landmarks5[row:row+1,person],decoded.shape[-2:],device=decoded.device)
        generated=align_face_rgb(decoded[row:row+1],grid=grid)*2-1
        reference=align_face_rgb(target_pixels[row:row+1].float(),grid=grid)*2-1
        with torch.no_grad(): target_id=_feature(evaluators[0],reference);target_perc=_feature(evaluators[1],reference)
        ii.append((1-F.cosine_similarity(_feature(evaluators[0],generated),target_id)).mean())
        pp.append((_feature(evaluators[1],generated)-target_perc).square().mean())
    identity=torch.stack(ii).mean();perceptual=torch.stack(pp).mean()
    return identity_weight*identity+perceptual_weight*perceptual,{'identity':identity,'perceptual':perceptual,'eligible':len(ii)}
