"""Window normalization reuses V5 loss kernels, not accumulated autograd graphs."""
from ..losses import LocalLossConfig,local_velocity_loss,auxiliary_loss

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

def compute_micro_loss(pipe,prediction,target_velocity,coverage,supervision,noisy,sigma,target_rgb,loss_spec,counts,update,evaluators):
    cfg=LocalLossConfig(auxiliary_every=loss_spec.auxiliary_every,auxiliary_sigma_max=loss_spec.auxiliary_sigma_max,
        weights=(loss_spec.face,loss_spec.hand,loss_spec.identity,loss_spec.perceptual))
    _,parts=local_velocity_loss(prediction,target_velocity,coverage,supervision,cfg)
    result=loss_spec.velocity*parts['velocity']/counts['velocity']
    for name,key,area in (('face','face_velocity','face_area'),('hand','hand_velocity','hand_area')):
        if counts[name] and bool(parts[area]>0):result=result+getattr(loss_spec,name)*parts[key]/counts[name]
    _,aux=auxiliary_loss(pipe,noisy,prediction,sigma,target_rgb,supervision,cfg,step=update,evaluators=evaluators)
    for name in ('identity','perceptual'):
        if counts[name] and aux['eligible']:result=result+getattr(loss_spec,name)*aux[name]*aux['eligible']/counts[name]
    return result,{**parts,**aux,'weighted_micro_objective':result,'window_denominators':counts}
