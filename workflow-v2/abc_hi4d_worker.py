#!/usr/bin/env python3
"""Export calibrated official Hi4D body/contact geometry without a pose estimator."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def official_person(params,index,backend,extr,K,hw):
    """Reconstruct the exact audited gender model and verify both coordinate frames."""
    import torch
    from scipy.spatial.transform import Rotation
    from src.pose_control.qwen21_full.body_pose import PARENTS_22,matrix_to_rotation_6d
    layer=backend.layer
    if tuple(layer.parents[:22].tolist())!=PARENTS_22:raise ValueError('Official body22 joint tree differs')
    values={name:torch.as_tensor(params[name][index:index+1],dtype=torch.float32) for name in ('betas','body_pose','global_orient','transl')}
    with torch.inference_mode():
        body=layer(**values)
        rest=layer(betas=values['betas'],body_pose=torch.zeros(1,69),global_orient=torch.zeros(1,3),transl=torch.zeros(1,3))
    mesh=body.vertices[0].numpy();joints=body.joints[0,:24].numpy()
    error=float(np.abs(mesh-params['verts'][index]).max());joint_error=float(np.abs(joints-params['joints_3d'][index]).max())
    if max(error,joint_error)>1e-4:raise ValueError(f'Official Hi4D mesh/joints cannot be reconstructed: {error}, {joint_error}')
    R=np.asarray(extr[:,:3],np.float32);t=np.asarray(extr[:,3],np.float32)
    root=R@Rotation.from_rotvec(params['global_orient'][index]).as_matrix()
    translated=R@params['transl'][index]+t+(R-np.eye(3))@rest.joints[0,0].numpy()
    camera_values={**values,'global_orient':torch.from_numpy(Rotation.from_matrix(root).as_rotvec().astype(np.float32))[None],
        'transl':torch.from_numpy(translated.astype(np.float32))[None]}
    with torch.inference_mode():camera_body=layer(**camera_values)
    camera_mesh=mesh@R.T+t;camera_joints=joints@R.T+t
    camera_error=float(np.abs(camera_body.vertices[0].numpy()-camera_mesh).max())
    if camera_error>1e-4:raise ValueError('Official world-to-camera pelvis convention differs')
    h,w=hw
    global26=np.concatenate((params['betas'][index],matrix_to_rotation_6d(torch.from_numpy(root.astype(np.float32))).numpy(),
        translated,[K[0,0]/w,K[1,1]/h,K[0,2]/w,K[1,2]/h],translated)).astype(np.float32)
    return dict(mesh_cam=camera_mesh.astype(np.float32),joints_cam=camera_joints.astype(np.float32),
        faces=backend.faces.numpy().astype(np.int32),vertex_part=backend.vertex_parts.numpy(),
        body_pose=params['body_pose'][index,:63].copy(),root_rotation=root.astype(np.float32),global26=global26,
        body_height=float(np.ptp(rest.vertices[0].numpy()[:,1])),mesh_error=error,joint_error=joint_error,camera_mesh_error=camera_error)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text(encoding='utf8'));frame=request['official_record'];settings=request['settings']
    out=Path(request['out_dir']);out.mkdir(parents=True,exist_ok=True)
    with Image.open(request['image']) as im:given=ImageOps.exif_transpose(im).convert('RGB')
    with Image.open(frame['rgb']) as im:bound=ImageOps.exif_transpose(im).convert('RGB')
    if given.size!=bound.size or not np.array_equal(np.asarray(given),np.asarray(bound)):
        raise ValueError('Interaction C pixels differ from the official Hi4D frame; do not bind another photo to its annotations')
    if (out/'original.png').exists():raise FileExistsError('Official C output exists; use a fresh run')
    given.save(out/'original.png');h,w=given.height,given.width
    with np.load(frame['official_smpl'],allow_pickle=False) as f:params=dict(f)
    with np.load(frame['meta'],allow_pickle=False) as f:meta=dict(f)
    with np.load(frame['camera_file'],allow_pickle=False) as f:cameras=dict(f)
    index=list(cameras['ids']).index(int(frame['camera_id']));K=cameras['intrinsics'][index].astype(np.float32);extr=cameras['extrinsics'][index].astype(np.float32)
    if np.abs(cameras['dist_coeffs'][index]).max()>1e-12:raise ValueError('Use an explicitly rectified Hi4D frame with pinhole calibration')
    from src.pose_control.qwen21_full.body_pose import PARENTS_22
    from src.pose_control.qwen21_pro.geometry import _load_hi4d_smpl_gender_backend
    from scipy.spatial.transform import Rotation
    import torch,smplx
    torch.set_num_threads(4)
    model_root=Path(settings.get('hi4d_model_root',str(ROOT/'add/tools/SMPLest-X/human_models/human_model_files/smpl')))
    people=[];rows=[]
    for person in range(2):
        gender=str(meta['genders'][person]);model_file=model_root/f'SMPL_{gender.upper()}.pkl'
        backend=_load_hi4d_smpl_gender_backend(model_file,gender);model=backend.layer
        body=official_person(params,person,backend,extr,K,(h,w))
        local=Rotation.from_rotvec(np.concatenate((np.zeros((1,3)),body['body_pose'].reshape(21,3)))).as_matrix()
        local[0]=body['root_rotation'];frames=[]
        for j,parent in enumerate(PARENTS_22):frames.append(local[j] if parent<0 else frames[parent]@local[j])
        filename=out/f'person_{person}.npz'
        with filename.open('xb') as stream:
            np.savez_compressed(stream,vertices_cam=body['mesh_cam'],mesh_cam=body['mesh_cam'],joints_cam=body['joints_cam'],
                faces=body['faces'],vertex_part=body['vertex_part'],dominant_joint=model.lbs_weights.argmax(1).numpy(),
                joint_frames=np.stack(frames).astype(np.float32),body_height=np.array(body['body_height'],np.float32),
                body_pose=body['body_pose'],global_orient=Rotation.from_matrix(body['root_rotation']).as_rotvec().astype(np.float32),
                transl=body['global26'][16:19],betas=params['betas'][person],K=K,hand_pose_available=np.array(False))
        rows.append(dict(index=person,params_npz=str(filename),pelvis_cam=body['joints_cam'][0].tolist(),
            mesh_error_m=body['mesh_error'],joint_error_m=body['joint_error'],camera_mesh_error_m=body['camera_mesh_error']))
        people.append(body)
    vertices=np.stack([p['mesh_cam'] for p in people])
    contact_path=out/'official_contact.npz'
    with contact_path.open('xb') as stream:np.savez_compressed(stream,contact=params['contact'],vertices_cam=vertices)
    from hi4d import ordered_contacts
    from render.abc_scene import project_points,stick_contact_image
    links,_=ordered_contacts(params['contact'],vertices,[0,1]);endpoints=[]
    for person in range(2):
        selected=np.unique(links[:,1] if person==0 else links[:,3]);endpoints.append(project_points(vertices[person,selected],K))
    guide=stick_contact_image([project_points(p['joints_cam'][:22],K) for p in people],(w,h),endpoints=endpoints,
        contact_pairs=[project_points(np.stack((vertices[0,a],vertices[1,b])),K) for _,a,_,b in links])
    guide.save(out/'official_stick_contact.png')
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result=dict(schema='abc_hi4d_official_fit_v1',source='hi4d_official_smpl_and_contact',
        image=str(out/'original.png'),people=rows,official_record=frame,
        official_contact_npz=str(contact_path),official_stick_contact=str(out/'official_stick_contact.png'),
        official_asset_sha256={key:sha(frame[key]) for key in ('official_smpl','meta','camera_file','rgb')},
        camera={'K':K.tolist(),'original_hw':[h,w],'intrinsics_source':'official_hi4d_calibration'},
        hand_pose_available=False,topology='SMPL_6890_body22_no_finger_or_face_pose',
        contact_encoding='one_based_zero_no_record',official_contact_count=len(links))
    (out/'fit.json').write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
