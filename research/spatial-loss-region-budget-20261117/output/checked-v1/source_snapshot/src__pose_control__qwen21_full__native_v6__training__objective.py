"""Window normalization reuses V5 loss kernels, not accumulated autograd graphs."""
from ...native_v5.losses import LocalLossConfig,local_velocity_loss,auxiliary_loss
import torch
from torch.nn import functional as F
from ..spatial_weights import spatial_token_weights,spatial_velocity_loss

def counts_for_window(entries,routes,update):
    counts={'velocity':len(entries),'face':0,'hand':0,'identity':0,'perceptual':0}
    for e in entries:
        loss=routes[e['route']].loss;status=e['status']
        counts['face']+=int(loss.face>0 and status['face_area']>0)
        counts['hand']+=int(loss.hand>0 and status['hand_area']>0)
        aux=update%loss.auxiliary_every==0 and e['sigma']<=loss.auxiliary_sigma_max
        if aux:
            counts['identity']+=status['auxiliary_faces'] if loss.identity>0 else 0
            counts['perceptual']+=status['auxiliary_faces'] if loss.perceptual>0 else 0
    return counts

def compute_micro_loss(pipe,prediction,target_velocity,coverage,supervision,noisy,sigma,target_rgb,loss_spec,counts,update,evaluators,spatial=None,identity_regularizer=None):
    cfg=LocalLossConfig(auxiliary_every=loss_spec.auxiliary_every,auxiliary_sigma_max=loss_spec.auxiliary_sigma_max,
        weights=(loss_spec.face,loss_spec.hand,loss_spec.identity,loss_spec.perceptual))
    _,parts=local_velocity_loss(prediction,target_velocity,coverage,supervision,cfg)
    if any(loss_spec.spatial_coefficients):
        if not isinstance(spatial,dict) or not {'components','valid','target_token_hw'}<=set(spatial):
            raise ValueError('Spatial weighting requires offline components and actual target grid')
        valid=spatial['valid'].to(device=prediction.device)
        components={k:v.to(device=prediction.device) for k,v in spatial['components'].items()}
        weights=spatial_token_weights(components,valid,spatial['target_token_hw'],loss_spec.spatial_coefficients,loss_spec.spatial_max)
        pixels=valid[:,None] if valid.ndim==3 else valid
        if pixels.ndim==2:pixels=pixels[None,None]
        pooled_valid=F.interpolate(pixels.float(),size=tuple(spatial['target_token_hw']),mode='area').flatten(1)
        if pooled_valid.shape!=coverage.shape or not torch.allclose(pooled_valid,coverage.float(),atol=1e-6,rtol=0):
            raise ValueError('Spatial valid pooling differs from actual target coverage')
        # weights already contain valid coverage; multiplying coverage again would
        # suppress partially padded tokens twice.
        parts['velocity']=spatial_velocity_loss(prediction,target_velocity,weights)
        parts['velocity_numerator']=((prediction.float()-target_velocity.float()).square()*weights[...,None]).sum()
        parts['velocity_denominator']=weights.sum()*prediction.shape[-1]
        parts['spatial_weight_sum']=weights.sum()
        parts['spatial_weight_max']=weights.max()
    result=loss_spec.velocity*parts['velocity']/counts['velocity']
    for name,key,area in (('face','face_velocity','face_area'),('hand','hand_velocity','hand_area')):
        if counts[name] and bool(parts[area]>0):result=result+getattr(loss_spec,name)*parts[key]/counts[name]
    _,aux=auxiliary_loss(pipe,noisy,prediction,sigma,target_rgb,supervision,cfg,step=update,evaluators=evaluators,identity_regularizer=identity_regularizer)
    for name in ('identity','perceptual'):
        if counts[name] and aux['eligible']:result=result+getattr(loss_spec,name)*aux[name]*aux['eligible']/counts[name]
    return result,{**parts,**aux,'weighted_micro_objective':result,'window_denominators':counts}
