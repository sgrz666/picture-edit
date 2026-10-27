"""Official Hi4D frame binding and explicit SMPL-to-SMPL-X contact transfer."""
from pathlib import Path
import json
import numpy as np

def resolve_record(path):
    path=Path(path).resolve()
    if path.is_dir():path=path/'pair.json'
    record=json.loads(path.read_text(encoding='utf8'))
    if record.get('schema')=='dual_pose_pair_v1':
        item=record['B'];meta=record['manifest'];smpl=Path(item['smpl_original'])
        record=dict(schema='hi4d_official_frame_v1',official_smpl=str(smpl),meta=str(smpl.parent.parent/'meta.npz'),
            camera_file=item['camera_original'],rgb=str(path.parent/item['image']),camera_id=meta['camera_id'],
            pair=meta['pair'],action=meta['target_action'],frame_id=meta['target_frame_id'])
    required=('official_smpl','meta','camera_file','rgb','camera_id','pair','action','frame_id')
    if any(k not in record for k in required):raise ValueError('Hi4D record requires official_smpl/meta/camera_file/rgb/camera_id/pair/action/frame_id')
    for key in ('official_smpl','meta','camera_file','rgb'):
        asset=Path(record[key]);asset=asset if asset.is_absolute() else path.parent/asset
        if not asset.is_file():raise FileNotFoundError(asset)
        record[key]=str(asset.resolve())
    return record

def ordered_contacts(contact,vertices,ordering):
    from src.pose_control.qwen21_full.dual_hi4d_geometry import decode_contact
    links,_=decode_contact(np.asarray(contact),np.asarray(vertices),1.)
    reverse={int(original):slot for slot,original in enumerate(ordering)};result=set()
    for p,a,q,b in links:
        p,q=reverse[int(p)],reverse[int(q)]
        result.add((p,int(a),q,int(b)) if p==0 else (q,int(b),p,int(a)))
    result=np.asarray(sorted(result),np.int64).reshape(-1,4)
    delta=np.asarray([vertices[ordering[1],b]-vertices[ordering[0],a] for _,a,_,b in result],np.float32).reshape(-1,3)
    return result,delta

def attach_contacts(drivers,targets,links,max_error=.75):
    """Map only within the same body part in a normalized anatomical bone frame.

    Original official vertex pairs remain in the source record. Returned target
    pairs are estimated surface attachments, never labelled official SMPL-X IDs.
    """
    from src.pose_control.qwen21_full.body_pose import PARENTS_22
    output=[];deltas=[];errors=[];rejected=[];low_confidence=[]
    for number,(p,a,q,b) in enumerate(links):
        attached=[];residual=[]
        for person,vertex in ((int(p),int(a)),(int(q),int(b))):
            source,target=drivers[person],targets[person]
            part=int(source['vertex_part'][vertex]);joint=int(source['dominant_joint'][vertex])
            if joint>=22:joint=20 if joint==22 else 21
            parent=PARENTS_22[joint]
            def scale(joints,height):
                length=np.linalg.norm(joints[joint]-joints[parent]) if parent>=0 else float(height)
                return max(float(length),1e-5)
            source_scale=scale(source['joints_cam'],source['body_height'])
            target_scale=scale(target['joints_cam'],target.get('body_height',source['body_height']))
            frame=source['joint_frames'][joint]
            reference=(source['vertices_cam'][vertex]-source['joints_cam'][joint])@frame/source_scale
            candidates=np.flatnonzero(target['part_vertex']==part)
            if not len(candidates):attached.append(-1);residual.append(float('inf'));continue
            local=(target['vertices_cam'][candidates]-target['joints_cam'][joint])@frame/target_scale
            distance=np.linalg.norm(local-reference,axis=-1);best=int(distance.argmin())
            attached.append(int(candidates[best]));residual.append(float(distance[best]))
        if min(attached)<0:
            rejected.append({'source_link':number,'reason':'anatomical_surface_attachment_uncertain',
                'normalized_error':[r if np.isfinite(r) else None for r in residual]});continue
        if max(residual)>max_error:
            low_confidence.append({'source_link':number,'normalized_error':residual,
                'meaning':'official positive part relation retained; target surface localization approximate'})
        output.append((0,attached[0],1,attached[1]));errors.append(residual)
        deltas.append(drivers[1]['vertices_cam'][b]-drivers[0]['vertices_cam'][a])
    return np.asarray(output,np.int64).reshape(-1,4),np.asarray(deltas,np.float32).reshape(-1,3),dict(
        source_topology='SMPL',target_topology='SMPL-X',source='official_hi4d_one_based_contact',
        correspondence='estimated_bone_local_same_part_surface_attachment',official_link_count=len(links),
        candidate_count=len(output),normalized_attachment_errors=errors,rejected=rejected,low_confidence_links=low_confidence,
        status='official_contacts_transferred' if output else 'no_transferable_official_positive_contact')
