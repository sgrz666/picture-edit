"""Pure NumPy contracts for full53 conversion and bounded pair placement."""
from __future__ import annotations
import numpy as np

PARAMETER_SIZES = {'global_orient':3,'body_pose':63,'left_hand_pose':45,
    'right_hand_pose':45,'jaw_pose':3,'expression':10,'betas':10,'transl':3}


def decode_full53(rotvec, shape, expression):
    rotvec=np.asarray(rotvec,dtype=np.float32)
    if rotvec.shape!=(53,3) or not np.isfinite(rotvec).all():
        raise ValueError('Multi-HMR pose must be finite full53 [53,3]')
    out={name:rotvec[s].reshape(-1).copy() for name,s in {
        'global_orient':slice(0,1),'body_pose':slice(1,22),'left_hand_pose':slice(22,37),
        'right_hand_pose':slice(37,52),'jaw_pose':slice(52,53)}.items()}
    for name,value in [('betas',shape),('expression',expression)]:
        value=np.asarray(value,dtype=np.float32).reshape(-1)
        if value.shape!=(10,) or not np.isfinite(value).all():
            raise ValueError(f'Multi-HMR {name} must contain 10 finite values')
        out[name]=value.copy()
    return out


def pelvis_translation(pelvis_cam, rest_pelvis):
    pelvis_cam,rest_pelvis=np.asarray(pelvis_cam),np.asarray(rest_pelvis)
    if pelvis_cam.shape!=(3,) or rest_pelvis.shape!=(3,) or not np.isfinite([pelvis_cam,rest_pelvis]).all():
        raise ValueError('Camera/rest pelvis must be finite [3]')
    return (pelvis_cam-rest_pelvis).astype(np.float32)


def map_roles(pelvis_cam, role_mapping):
    pelvis_cam=np.asarray(pelvis_cam)
    if pelvis_cam.shape!=(2,3) or not np.isfinite(pelvis_cam).all() or (pelvis_cam[:,2]<=0).any():
        raise ValueError('Exactly two finite camera-space pelvises required')
    if set(role_mapping)!={'A','B'} or set(role_mapping.values())!={'left','right'}:
        raise ValueError('A/B must map to distinct left/right roles')
    order=np.argsort(pelvis_cam[:,0]/pelvis_cam[:,2],kind='stable')
    return [int(order[0 if role_mapping[p]=='left' else 1]) for p in ('A','B')]


def mix_parameters(source, driver, left_hand_mean, right_hand_mean):
    out={}
    for name,size in PARAMETER_SIZES.items():
        fallback_hand=name in ('left_hand_pose','right_hand_pose') and not bool(driver.get('hand_pose_available',True))
        origin=source if name in ('betas','expression','jaw_pose') or fallback_hand else driver
        value=np.asarray(origin[name],dtype=np.float32).reshape(-1)
        if value.shape!=(size,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid {name} for retarget')
        out[name]=value.copy()
    # The driver is flat_hand_mean=True, the source model False. SMPL-X adds its
    # pose_mean after receiving axis angles, so subtract it exactly once.
    if bool(driver.get('hand_pose_available',True)):
        out['left_hand_pose']-=np.asarray(left_hand_mean,dtype=np.float32).reshape(45)
        out['right_hand_pose']-=np.asarray(right_hand_mean,dtype=np.float32).reshape(45)
    return out


def place_pair(vertices,k,canvas_hw,margin=.06):
    """Fit both real meshes with one camera translation; never resize bodies."""
    vertices=np.asarray(vertices,dtype=np.float64);k=np.asarray(k,dtype=np.float64)
    if vertices.ndim!=3 or vertices.shape[0]!=2 or vertices.shape[-1]!=3 or not np.isfinite(vertices).all():
        raise ValueError('Two finite meshes required')
    if k.shape!=(3,3) or k[0,0]<=0 or k[1,1]<=0 or not 0<=margin<.45:
        raise ValueError('Invalid shared camera or margin')
    h,w=canvas_hw;flat=vertices.reshape(-1,3)
    centre=(flat.min(0)+flat.max(0))/2
    # Centre the pair on the canvas optical ray, including off-centre source K.
    ray=np.array([(w/2-k[0,2])/k[0,0],(h/2-k[1,2])/k[1,1],1.])
    def candidate(z):
        delta=ray*z-centre
        moved=vertices+delta
        uv=moved@k.T;uv=uv[...,:2]/uv[...,2:]
        fits=(moved[...,2].min()>1e-3 and uv[...,0].min()>=w*margin and
            uv[...,0].max()<=w*(1-margin) and uv[...,1].min()>=h*margin and
            uv[...,1].max()<=h*(1-margin))
        return fits,moved,delta
    lower=max(.01,float(centre[2]-flat[:,2].min())+.01)
    upper=max(lower*2,2.)
    for _ in range(30):
        if candidate(upper)[0]:break
        upper*=2
    else:raise RuntimeError('Pair cannot fit shared camera within finite placement budget')
    for _ in range(48):
        middle=(lower+upper)/2
        if candidate(middle)[0]:upper=middle
        else:lower=middle
    _,moved,delta=candidate(upper)
    return moved.astype(np.float32),delta.astype(np.float32),{
        'placement_scale':1.0,'shared_translation_m':delta.tolist(),
        'margin_fraction':margin,'camera_depth_m':upper,'iterations':48}


def bounded_contact_shift(vertices,indices,reference_deltas,max_shift=.04,regularization=.1):
    """Preserve reliable C endpoint offsets under shape swap, with bounded translation."""
    indices=np.asarray(indices,dtype=np.int64).reshape(-1,4)
    reference_deltas=np.asarray(reference_deltas,dtype=np.float64).reshape(-1,3)
    if len(indices)!=len(reference_deltas) or max_shift<0 or regularization<0:
        raise ValueError('Invalid contact correction contract')
    if not len(indices):
        return np.zeros((2,3),np.float32),{'status':'uncertain_no_reliable_candidates','candidate_count':0,'iterations':0}
    if (indices[:,0]!=0).any() or (indices[:,2]!=1).any():
        raise ValueError('Contact candidates must be ordered logical source A -> B')
    a=np.asarray(vertices[0])[indices[:,1]];b=np.asarray(vertices[1])[indices[:,3]]
    residual=(b-a)-reference_deltas
    relative=-residual.mean(0)/(1+regularization)
    norm=np.linalg.norm(relative)
    if norm>2*max_shift:relative*=2*max_shift/norm
    shifts=np.stack([-.5*relative,.5*relative]).astype(np.float32)
    before=np.linalg.norm(b-a,axis=-1);after=np.linalg.norm(b+shifts[1]-a-shifts[0],axis=-1)
    error_before=float(np.mean(np.sum(residual**2,axis=1)))
    error_after=float(np.mean(np.sum((residual+relative)**2,axis=1)))
    if not np.isfinite(shifts).all() or error_after>error_before+1e-8:
        raise RuntimeError('Bounded correction failed its finite/non-increasing objective check')
    return shifts,{'status':'bounded_translation_correction','candidate_count':len(indices),'iterations':1,
        'max_person_shift_m':max_shift,'before_mean_m':float(before.mean()),'after_mean_m':float(after.mean()),
        'reference_residual_mse_before':error_before,'reference_residual_mse_after':error_after,
        'shifts_m':shifts.tolist(),'penetration_status':'not_measured_no_FCL'}
