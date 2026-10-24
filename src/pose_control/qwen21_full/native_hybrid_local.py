"""2D-only Hi4D local conditions on the exact existing V3 canvas.

No SMPL hand surface is presented as finger depth, no missing pose is presented
as a neutral rotation, and no geometry/appearance encoder runs in this module.
"""
import numpy as np
import torch
from torch.nn import functional as F

from src.data.canvas_contract import CanvasSpec
from .native_local_condition import NativeLocalConditionV4, NativeLocalReferencesV4, LOCAL_SCHEMA, REFERENCE_SCHEMA
from .native_hybrid_assets import HybridConfig


def expanded_box(points, valid, expand):
    if not np.any(valid): return np.zeros(4,np.float32),False
    xy=points[valid];lo=xy.min(0);hi=xy.max(0);span=hi-lo
    if (span<=0).any(): return np.zeros(4,np.float32),False
    return np.concatenate((lo-span*expand,hi+span*expand)).astype(np.float32),True


def mask_box(mask):
    y,x=mask.nonzero(as_tuple=True)
    if not x.numel(): return np.zeros(4,np.float32),False
    h,w=mask.shape
    return np.array([float(x.min())/w,float(y.min())/h,float(x.max()+1)/w,float(y.max()+1)/h],np.float32),True


def roi_mask(box, hw):
    h,w=hw
    yy,xx=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
    return (xx>=float(box[0]))&(xx<float(box[2]))&(yy>=float(box[1]))&(yy<float(box[3]))


def local_observations(asset, spec, official_masks, head_masks, wrist_uv, config):
    """Source and driver each invoke this with their own CanvasSpec exactly once."""
    if list(spec.original_hw)!=list(asset['metadata']['original_hw']): raise ValueError('Observation native grid differs from Canvas original')
    xy=spec.points(asset['xy_native']/np.array([spec.original_hw[1],spec.original_hw[0]],np.float32))
    scores=asset['scores'];valid=asset['point_valid'].copy() & asset['person_assigned'][:,None]
    valid &= (scores>=config.point_score)
    face=np.zeros((2,72,3),np.float32);fv=np.zeros((2,72),bool)
    hands=np.zeros((2,2,21,3),np.float32);hv=np.zeros((2,2,21),bool)
    fb=np.zeros((2,4),np.float32);fr=np.zeros(2,bool)
    hb=np.zeros((2,2,4),np.float32);hr=np.zeros((2,2),bool)
    reasons=[]
    # Region confidence does not establish ownership outside the official mask.
    for p in range(2):
        face[p,4:,:2]=xy[p,24:92];face[p,4:,2]=scores[p,24:92];fv[p,4:]=valid[p,24:92]
        source='dwpose_2d'
        if fv[p].sum()>=config.min_face_points:
            fb[p],fr[p]=expanded_box(face[p,:,:2],fv[p],config.roi_expand)
        if not fr[p]:
            fb[p],fr[p]=mask_box(head_masks[p]);source='trusted_smpl_head_roi' if fr[p] else 'unavailable'
        report={'face_roi_source':source,'face_points':int(fv[p].sum()),'hands':[]}
        y,x=official_masks[p].nonzero(as_tuple=True)
        bbox_height=float(y.max()-y.min()+1) if y.numel() else 0.
        tolerance=max(config.wrist_min_px,config.wrist_bbox_ratio*bbox_height)
        for side,start in enumerate((92,113)):
            hands[p,side,:,:2]=xy[p,start:start+21];hands[p,side,:,2]=scores[p,start:start+21]
            hv[p,side]=valid[p,start:start+21]
            wrist=xy[p,start]*np.array([spec.canvas_hw[1],spec.canvas_hw[0]],np.float32)
            anchor=wrist_uv[p,side]
            wrist_ok=bool(hv[p,side,0] and np.isfinite(anchor).all() and np.linalg.norm(wrist-anchor)<=tolerance)
            if wrist_ok and hv[p,side].sum()>=config.min_hand_points:
                hb[p,side],hr[p,side]=expanded_box(hands[p,side,:,:2],hv[p,side],config.roi_expand)
            report['hands'].append({'reliable_points':int(hv[p,side].sum()),'wrist_matches':wrist_ok,
                'roi_available':bool(hr[p,side]),'reason':None if hr[p,side] else 'untrusted_2d_hand_location'})
        reasons.append(report)
    return dict(face=face,face_valid=fv,face_boxes=fb,face_roi=fr,hands=hands,hand_valid=hv,
                hand_boxes=hb,hand_roi=hr,reasons=reasons)


def build_local(sample, asset, official_masks, *, source_meta, target_meta, body_record_sha256, config=None):
    config=config or HybridConfig();dual=sample['dual_depth'].validate()
    spec=CanvasSpec.from_record(sample['target_canvas']);source_spec=CanvasSpec.from_record(sample['source_canvas'])
    if asset['metadata']['geometry_sha256']!=target_meta['record_sha256'] or asset['metadata']['person_ids']!=target_meta['person_ids']:
        raise ValueError('Hybrid observation belongs to another driver or person order')
    if dual.model_family.tolist()!=[[1,1]]: raise ValueError('Hybrid uses native SMPL, never relabelled SMPL-X')
    masks=torch.stack([spec.stack(torch.as_tensor(m,dtype=torch.float32),nearest=True)>.5 for m in official_masks])
    owner=dual.owner_weights[0,:,0].float()
    # Unknown scene order (including V3 tie) cannot license either person's local residual.
    exclusive=(owner>.99999)
    people=(dual.person_a,dual.person_b)
    heads=torch.stack([p.context.part_onehot[0,0]>.5 for p in people]) & masks & exclusive
    wrist=np.stack([sample['joint_uv'][[19,20]].numpy(),sample['joint_uv_b'][[19,20]].numpy()])
    # joint_uv excludes pelvis: local wrist indices 19/20 are left/right by name.
    for p,key in enumerate(('projection_valid','projection_valid_b')):
        wrist[p,~sample[key][[19,20]].numpy()]=np.nan
    obs=local_observations(asset,spec,masks,heads,wrist,config)
    h,w=spec.canvas_hw
    def z(*shape,boolean=False): return torch.zeros(1,*shape,dtype=torch.bool if boolean else torch.float32)
    scene_valid=dual.person_depth_valid.any(1)
    scene_z=torch.where(scene_valid,torch.where(dual.person_depth_valid,dual.person_camera_z,torch.inf).amin(1),0.)
    logical={identity:i for i,identity in enumerate(sorted(source_meta['person_ids']))}
    values=dict(person_valid=torch.ones(1,2,dtype=torch.bool),source_indices=sample['source_indices'][None].clone(),
        source_slot_ids=torch.tensor([[logical[k] for k in source_meta['person_ids']]],dtype=torch.long),face_landmarks=torch.from_numpy(obs['face'])[None],
        face_point_valid=torch.from_numpy(obs['face_valid'])[None],jaw_pose=z(2,3),jaw_valid=z(2,boolean=True),
        expression=z(2,10),expression_valid=z(2,boolean=True),face_boxes=torch.from_numpy(obs['face_boxes'])[None],
        face_roi_valid=torch.from_numpy(obs['face_roi'])[None],hand_pose=z(2,2,45),hand_pose_valid=z(2,2,boolean=True),
        hand_keypoints=torch.from_numpy(obs['hands'])[None],hand_point_valid=torch.from_numpy(obs['hand_valid'])[None],
        mesh_bps=z(2,2,1024),mesh_valid=z(2,2,boolean=True),hand_boxes=torch.from_numpy(obs['hand_boxes'])[None],
        hand_roi_valid=torch.from_numpy(obs['hand_roi'])[None],wrist_z=z(2,2),region_masks=z(6,h,w),visibility=z(6,h,w),
        local_z=z(6,h,w),local_z_valid=z(6,h,w,boolean=True),part_visible=z(6,h,w,boolean=True),
        person_owner=owner[None].clone(),scene_z=scene_z.float(),scene_z_valid=scene_valid,
        body_height=dual.body_height.float().clone())
    boxes=np.concatenate((obs['face_boxes'],obs['hand_boxes'].reshape(4,4)))
    enabled=np.concatenate((obs['face_roi'],obs['hand_roi'].reshape(4)))
    for r,p in enumerate((0,1,0,0,1,1)):
        support=roi_mask(boxes[r],(h,w)) & masks[p] & exclusive[p] & (owner[p]>0)
        if r<2: support &= heads[p]
        support &= bool(enabled[r])
        values['region_masks'][0,r]=support.float()
    # Without local Camera-Z there is no defensible ordering of hand/face or
    # hand/hand overlap, even on the same person. Disable all contested pixels.
    ambiguous=(values['region_masks'][0]>0).sum(0)>1
    values['visibility'][0]=(~ambiguous).float()[None].expand(6,-1,-1)
    values['part_visible'][0]=(values['region_masks'][0]>0)&~ambiguous
    coverage=(values['region_masks']*values['visibility'])[0]
    report={'people':obs['reasons'],'assignment':asset['metadata']['assignment'],
        'candidate_failures':asset['metadata'].get('provenance',{}).get('candidate_failures',[]),
        'support_pixels':coverage.sum((1,2)).tolist(),
        'canvas_coverage':coverage.mean((1,2)).tolist(),
        'effective_target_cells':F.avg_pool2d(coverage[:,None],16,16).sum((1,2,3)).tolist(),
        'ambiguous_overlap_pixels':int(ambiguous.sum())}
    values['metadata']=({'schema':LOCAL_SCHEMA,'data_profile':'hybrid_hi4d','body_model_family':'smpl',
        'geometry_source':'hi4d_official_smpl','face_geometry_source':'dwpose_2d_or_trusted_head_roi',
        'hand_geometry_source':'dwpose_2d','local_depth_source':'unavailable','hand_pose_convention':'unavailable',
        'normal_space':'camera_cv','expression_mode':config.expression_mode,'expression_retargeted':[False,False],
        'counterfactual_geometry':False,'paired_rgb_compatible':True,
        'source_canvas_id':source_spec.fingerprint(),'target_canvas_id':spec.fingerprint(),
        'source_person_ids':source_meta['person_ids'],'target_person_ids':target_meta['person_ids'],
        'source_rgb_sha256':source_meta['rgb']['sha256'],'driver_rgb_sha256':target_meta['rgb']['sha256'],
        'source_asset_sha256':source_meta['record_sha256'],'driver_record_sha256':target_meta['record_sha256'],
        'body_record_sha256':body_record_sha256,'camera_id':target_meta['camera_id'],
        'scale_group_id':target_meta['scale_group_id'],'target_observation':asset['metadata'],
        'visibility_policy':'roi_intersect_official_mask_intersect_exclusive_v3_owner;unknown_local_overlap_off',
        'config':config.record(),'coverage_report':report},)
    return NativeLocalConditionV4(**values).validate(),report


def empty_references(source_meta, source_spec, *, declaration):
    if declaration!='explicit_empty': raise ValueError('Empty appearance must be explicitly requested')
    meta={'schema':REFERENCE_SCHEMA,'role':'source_only','reference_count':1,
        'source_rgb_sha256':source_meta['rgb']['sha256'],'source_canvas_id':source_spec.fingerprint(),
        'source_person_ids':source_meta['person_ids'],'availability_declaration':declaration,
        'reference_sources':[{'rgb':source_meta['rgb'],'canvas_id':source_spec.fingerprint(),
            'canvas':source_spec.metadata(),'person_ids':source_meta['person_ids'],'geometry_sha256':source_meta['record_sha256']}]}
    return NativeLocalReferencesV4(torch.zeros(1,2,1,512),torch.zeros(1,2,1,dtype=torch.bool),
        torch.zeros(1,2,1,256,1536),torch.zeros(1,2,1,dtype=torch.bool),torch.zeros(1,2,2,1,256,1536),
        torch.zeros(1,2,2,1,dtype=torch.bool),torch.zeros(1,2,4),torch.zeros(1,2,2,4),(meta,)).validate()
