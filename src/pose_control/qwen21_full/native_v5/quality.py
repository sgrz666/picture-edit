"""Offline hand audit/repair interchange. Never infer 3D from repaired 2D."""
from dataclasses import replace
import torch
from ..hand.quality import hand_quality_gate,merge_vitpose_keypoints


def audit_hands(condition):
    condition.validate();h,w=condition.region_masks.shape[-2:]
    box=condition.hand_boxes;pixels=(box[...,2:]-box[...,:2])*box.new_tensor([w,h])
    side=pixels.amin(-1);confidence=condition.hand_keypoints[...,2].mean(-1)
    outside=((condition.hand_keypoints[...,:2]<0)|(condition.hand_keypoints[...,:2]>1)).any(-1).float().mean(-1)
    return {'mode':'audit_only','legacy_request_refinement':hand_quality_gate(condition.hand_keypoints,box).tolist(),
        'native_request_review':((side<.035*min(h,w))|(confidence<.35)|(outside>.2)).tolist(),
        'short_side_canvas_px':side.tolist(),'mean_detection_score':confidence.tolist(),
        'outside_fraction':outside.tolist(),'validity_changed':False,'threshold_status':'unaccepted_engineering_defaults'}


def export_repair(condition,rgb_reference):
    condition.validate()
    if any(m.get('driver_rgb_sha256')!=rgb_reference['sha256'] for m in condition.metadata): raise ValueError('Repair RGB must be the same driver frame')
    return {'schema':'native_hand_repair_job_v5','role':'offline_geometry_repair','rgb':rgb_reference,
        'points':condition.hand_keypoints,'point_valid':condition.hand_point_valid,'boxes':condition.hand_boxes,
        'support':condition.region_masks[:,2:],'camera_z':condition.local_z[:,2:],
        'camera_z_valid':condition.local_z_valid[:,2:],'metadata':condition.metadata,
        'requires_camera_z':False,'relative_depth_is_not_camera_z':True}


def import_repair(condition,payload,*,body_wrist_uv,trusted_person_support,repair_sha256,trusted_support_sha256,min_confidence=.25):
    if payload.get('schema')!='native_hand_repair_result_v5' or payload.get('role')!='offline_geometry_repair': raise ValueError('Explicit offline repair schema required')
    if any(k in payload for k in ('hand_pose','mesh_bps','camera_z','mesh_cam')): raise ValueError('2D repair cannot replace 3D geometry/depth')
    if len(payload.get('metadata',()))!=len(condition.metadata): raise ValueError('Repair metadata batch differs')
    for old,new in zip(condition.metadata,payload['metadata']):
        for k in ('driver_rgb_sha256','driver_record_sha256','target_canvas_id','target_person_ids','camera_id'):
            if old.get(k)!=new.get(k): raise ValueError('Repair source differs: '+k)
    points=torch.as_tensor(payload['points'],dtype=torch.float32,device=condition.hand_keypoints.device)
    requested=torch.as_tensor(payload['replace'],dtype=torch.bool,device=points.device)
    if points.shape!=condition.hand_keypoints.shape or requested.shape!=condition.hand_roi_valid.shape or not torch.isfinite(points).all(): raise ValueError('Malformed repaired points')
    h,w=condition.region_masks.shape[-2:];wrist=points[...,0,:2]*points.new_tensor([w,h])
    support=torch.as_tensor(trusted_person_support,dtype=torch.bool,device=points.device)
    if support.shape!=(points.shape[0],2,h,w): raise ValueError('Explicit same-frame official/verified person support required')
    heights=points.new_zeros(points.shape[0],2)
    for row in range(points.shape[0]):
        for p in range(2):
            ys=support[row,p].nonzero()[:,0]
            if ys.numel(): heights[row,p]=ys.max()-ys.min()+1
    tolerance=torch.maximum(points.new_tensor(16.),heights*.1)[...,None]
    accepted=requested & ((points[...,2]>=min_confidence).sum(-1)>=6) & (points[...,0,2]>=min_confidence)
    accepted &= points[...,2].mean(-1)>=min_confidence
    accepted &= (wrist-body_wrist_uv).norm(dim=-1)<=tolerance
    accepted &= condition.person_valid[...,None]
    for row,p,side in accepted.nonzero().tolist():
        xy=points[row,p,side,points[row,p,side,:,2]>=min_confidence,:2]
        if not (xy.amax(0)>xy.amin(0)).all(): accepted[row,p,side]=False
    merged=merge_vitpose_keypoints(condition.hand_keypoints,points,accepted,min_confidence=min_confidence)
    pv=torch.where(accepted[...,None],merged[...,2]>=min_confidence,condition.hand_point_valid)
    boxes=condition.hand_boxes.clone()
    for row,p,side in accepted.nonzero().tolist():
        xy=merged[row,p,side,pv[row,p,side],:2];lo=xy.amin(0);hi=xy.amax(0);pad=(hi-lo)*.1
        boxes[row,p,side]=torch.cat((lo-pad,hi+pad))
    regions=condition.region_masks.clone();visibility=condition.visibility.clone();part_visible=condition.part_visible.clone()
    yy,xx=torch.meshgrid((torch.arange(h,device=points.device)+.5)/h,(torch.arange(w,device=points.device)+.5)/w,indexing='ij')
    for row,p,side in accepted.nonzero().tolist():
        r=2+2*p+side;box=boxes[row,p,side]
        inside=(xx>=box[0])&(xx<box[2])&(yy>=box[1])&(yy<box[3])
        visible=inside&support[row,p]&(condition.person_owner[row,p]>.99999)
        regions[row,r]=visible.float();visibility[row,r]=1;part_visible[row,r]=visible
    # A repaired observation still has no fine local Z. Suppress unknown overlaps
    # with every existing local region, never infer contact from box intersections.
    unknown=(regions>0)&~condition.local_z_valid
    contested=unknown.sum(1,keepdim=True)>1
    visibility=visibility*~contested
    metadata=tuple({**m,'hand_2d_repair_sha256':repair_sha256,'hand_repair_support_sha256':trusted_support_sha256,
        'hand_2d_repair_accepted':accepted[i].tolist()} for i,m in enumerate(condition.metadata))
    return replace(condition,hand_keypoints=merged,hand_point_valid=pv,hand_boxes=boxes,
        hand_roi_valid=condition.hand_roi_valid|accepted,region_masks=regions,visibility=visibility,part_visible=part_visible,metadata=metadata).validate()
