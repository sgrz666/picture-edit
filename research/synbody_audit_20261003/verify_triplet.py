import json
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parent
base=root/'extracted/HumanNeRF-subset'
source='028';reference='042';source_frame=200;target_frame=150;view='00'
a=np.load(base/source/'smplx.npz',allow_pickle=True)['smplx'].item()
b=np.load(base/reference/'smplx.npz',allow_pickle=True)['smplx'].item()
checks={
    'different_shape':not np.array_equal(a['betas'],b['betas']),
    'same_body_pose_at_target':np.array_equal(a['body_pose'][target_frame],b['body_pose'][target_frame]),
    'same_left_hand_pose_at_target':np.array_equal(a['left_hand_pose'][target_frame],b['left_hand_pose'][target_frame]),
    'same_right_hand_pose_at_target':np.array_equal(a['right_hand_pose'][target_frame],b['right_hand_pose'][target_frame]),
    'same_global_orient_at_target':np.array_equal(a['global_orient'][target_frame],b['global_orient'][target_frame]),
    'source_target_poses_differ':not np.array_equal(a['body_pose'][source_frame],a['body_pose'][target_frame]),
}
def image_path(sid,frame):return str(base/sid/'img'/view/f'{frame:04d}.jpg')
def mask_path(sid,frame):return str(base/sid/'mask'/view/f'{frame:04d}.png')
sample={
    'task':'single_person_cross_identity_pose_reference',
    'kind':'dataset_example_only_not_model_output',
    'source':{'id':source,'frame':source_frame,'view':view,'rgb':image_path(source,source_frame),'mask':mask_path(source,source_frame),'smplx':str(base/source/'smplx.npz'),'camera':str(base/source/'camera.json')},
    'pose_reference':{'id':reference,'frame':target_frame,'view':view,'rgb':image_path(reference,target_frame),'mask':mask_path(reference,target_frame),'smplx':str(base/reference/'smplx.npz'),'camera':str(base/reference/'camera.json')},
    'supervision_only':{'id':source,'frame':target_frame,'rgb':image_path(source,target_frame),'mask':mask_path(source,target_frame)},
    'checks':checks,
    'quality_gate':{
        'review_type':'manual_visual_review_20261003',
        'geometry_pairing_valid':True,
        'eligible_for_photoreal_rgb_training':False,
        'status':'rejected_for_photoreal_rgb_supervision',
        'reasons':[
            'Weak or implausible foot-to-scene contact in source and target RGB.',
            'Contact shadows and person-scene integration are visually insufficient.',
            'Rendered person appearance does not meet the everyday-photo source requirement.'
        ],
        'allowed_usage':['geometry_and_identity_slot_debug_only'],
        'scope':'This A/B/C example; broader subset findings are a limited 28-image visual sample.'
    },
    'beta_l2':float(np.linalg.norm(a['betas']-b['betas'])),
    'caveat':'Copy B local joint rotations; retain A shape. Translation, camera and body projection require A-space alignment. Target RGB is supervision only.',
}
for group in ['source','pose_reference','supervision_only']:
    for field in ['rgb','mask']:
        p=Path(sample[group][field]);assert p.is_file(),str(p)
        with Image.open(p) as im:im.verify()
assert all(checks.values()),checks
(root/'example_ABC.json').write_text(json.dumps(sample,indent=2))
print(json.dumps({'geometry_checks':checks,'quality_gate':sample['quality_gate'],'beta_l2':sample['beta_l2'],'manifest':str(root/'example_ABC.json')},indent=2))
