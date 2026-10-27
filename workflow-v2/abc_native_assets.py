"""Adapt measured workflow SMPL-X meshes to existing Native V5 raw contracts.

Geometry preparation runs in the SMPLest-X environment. No RGB supervision or
learned Adapter outputs are stored in these records.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.canvas_contract import CanvasSpec,native_pad
from src.pose_control.qwen21_full.native_full_assets import sha256,write_record,read_record,local_geometry

def target_canvas(scene):
    # Render outputs already use K_canvas. Giving them the original A transform
    # would resize/reproject a second time.
    return native_pad(CanvasSpec.from_record(scene['camera']['canvas']).canvas_hw)

def source_reference_job(source,observation,boxes,models,*,hash_file=sha256):
    spec=CanvasSpec.from_record(source['canvas']);h,w=spec.canvas_hw
    points=np.asarray(observation['points'],np.float32);scores=np.asarray(observation['scores'])
    points=spec.pixel_points(points)
    face_points=points[24:92];face_valid=scores[24:92]>=.5
    groups=(list(range(36,42)),list(range(42,48)),[30],[48],[54])
    five_valid=all(face_valid[g].all() for g in groups)
    box_valid=(boxes[:,2:]>boxes[:,:2]).all(-1)
    person=dict(face_box=boxes[0].tolist(),hand_boxes=boxes[1:].tolist(),
        identity_valid=bool(five_valid and box_valid[0]),
        face_texture_valid=bool(face_valid.sum()>=8 and box_valid[0]),
        hand_texture_valid=[bool((scores[start:start+21]>=.5).sum()>=6 and box_valid[i+1])
            for i,start in enumerate((92,113))])
    if person['identity_valid']:
        person['arcface_five_points_canvas_pixels']=np.stack([face_points[g].mean(0) for g in groups]).tolist()
    ids=[source['person_id'],'__inactive__']
    return dict(**models,role='source_only',source_person_ids=ids,
        modalities={'arcface':True,'face_dino':True,'hand_dino':True},
        references=[dict(rgb={'path':source['image'],'sha256':hash_file(source['image'])},
            canvas=source['canvas'],person_ids=ids,geometry_sha256=hash_file(source['params_npz']),people=[person])])

def native_person(path,model,K,hw,person_id):
    from retarget.core import PARAMETER_SIZES
    from src.pose_control.qwen21_full.body_pose import axis_angle_to_matrix,matrix_to_rotation_6d
    from smplx.joint_names import JOINT_NAMES
    with np.load(path,allow_pickle=False) as archive:params={k:archive[k].copy() for k in archive.files}
    values={k:torch.from_numpy(params[k].reshape(1,n)).float() for k,n in PARAMETER_SIZES.items()}
    values.update(leye_pose=torch.zeros(1,3),reye_pose=torch.zeros(1,3))
    with torch.inference_mode():
        output=model.layer(**values,return_full_pose=True)
        rest={name:torch.zeros_like(value) for name,value in values.items()};rest['betas']=values['betas']
        reference=model.layer(**rest)
    mesh=output.vertices[0].numpy();joints=output.joints[0].numpy();full=output.full_pose[0].numpy()
    expected=params.get('vertices_cam',params.get('mesh_cam'))
    error=float(np.abs(mesh-expected).max())
    if error>1e-4:raise ValueError(f'Native imported mesh differs from workflow fit: {error}')
    rotations=axis_angle_to_matrix(torch.from_numpy(full.reshape(55,3))).numpy();world=[]
    for j,parent in enumerate(model.layer.parents.tolist()):world.append(rotations[j] if parent<0 else world[parent]@rotations[j])
    hand_joints=[];vertices=[];lengths=[]
    for side,name in enumerate(('left','right')):
        names=[name+'_wrist']
        for finger in ('thumb','index','middle','ring','pinky'):
            names.extend([name+'_'+finger+str(i) for i in (1,2,3)]+[name+'_'+finger])
        hand_joints.append(joints[[JOINT_NAMES.index(n) for n in names]])
        selected=np.flatnonzero(model.vertex_part==(4 if side==0 else 7));vertices.append(selected)
        lengths.append(float(np.linalg.norm(reference.vertices[0].numpy()[selected]-reference.joints[0,20+side].numpy(),axis=-1).max()))
    root=rotations[0];transl=values['transl'][0].numpy();h,w=hw
    global26=np.concatenate((values['betas'][0].numpy(),matrix_to_rotation_6d(torch.from_numpy(root)).numpy(),transl,
        np.array([K[0,0]/w,K[1,1]/h,K[0,2]/w,K[1,2]/h]),transl)).astype(np.float32)
    expression_sha=hashlib.sha256(model.layer.expr_dirs.numpy().astype('<f4').tobytes()).hexdigest()
    return dict(mesh_cam=mesh,joints_cam=joints[:22],faces=model.faces,face_part=model.face_part,
        body_pose=full[3:66],jaw_pose=full[66:69],expression_full=values['expression'][0].numpy(),
        hand_pose=np.stack((full[75:120],full[120:165])),root_rotation=root,global26=global26,
        body_height=float(np.ptp(reference.vertices[0].numpy()[:,1])),
        face72_cam=joints[list(model.joint_indices)][65:137],face72_valid=np.ones(72,bool),
        hand21_cam=np.stack(hand_joints),hand21_valid=np.ones((2,21),bool),wrist_cam=joints[[20,21]],
        wrist_rotation=np.stack([world[20],world[21]]),hand_vertices=vertices,standard_hand_length=lengths,
        metadata=dict(person_id=person_id,expression_basis_sha256=expression_sha,expression_retargeted=True,
            model_family='smplx',model=model.provenance,parameter_sha256=sha256(path),fk_mesh_max_error=error))

def target_sample(scene,driver,arrays,local,dual_config,record_sha):
    from src.pose_control.qwen21_full.native_dual_depth import NativeDualConfig
    from src.pose_control.qwen21_full.dual_depth_geometry import build_dual_condition,relative11
    from src.pose_control.qwen21_full.body_pose import BODY25_BY_NAME,JOINT_NAMES
    from src.pose_control.qwen21_full.conditions import TaskType,ContactRelationBatch
    from src.pose_control.qwen21_full.native_local_condition import NativeLocalConditionV4,LOCAL_SCHEMA
    spec=target_canvas(scene);h,w=spec.canvas_hw;people=driver['people'];ids=[p['metadata']['person_id'] for p in people]
    camera_id=hashlib.sha256(np.asarray(driver['K'],np.float32).tobytes()).hexdigest()
    meta=dict(record_id='abc:'+record_sha,record_sha256=record_sha,pair_id=record_sha,
        person_ids=ids,model_family_indices=[0,0],units='model_unit',scale_status='model_scale',
        camera_id=camera_id,scale_group_id=record_sha,contact_source='official_hi4d_smpl_transferred_to_target' if scene['interaction'].get('source')=='hi4d_official_smpl_and_contact' else 'estimated_from_C_not_official_labels')
    cache=dict(z=arrays['depth'],valid=arrays['mask'],normal=arrays['normal_cam'],part=arrays['part'],K=driver['K'],
        joints_cam=np.stack([p['joints_cam'] for p in people]),body_height=np.array([p['body_height'] for p in people],np.float32),
        part_distance=np.zeros((14,14),np.float32),part_distance_valid=np.zeros((14,14),bool),
        closest_points=np.zeros((14,14,2,3),np.float32),closest_points_valid=np.zeros((14,14),bool),
        part_contact=np.zeros((14,14,4),np.float32),contact_indices=arrays['contact_indices'],
        contact_distance=arrays['contact_distance_m']/np.mean([p['body_height'] for p in people]))
    from retarget.collision import ensure_fcl
    from src.pose_control.qwen21_full.dual_hi4d_geometry import part_distances
    ensure_fcl()
    distance,witness,available,witness_valid,intersects=part_distances(people)
    cache.update(part_distance=distance,closest_points=witness,part_distance_valid=available,closest_points_valid=witness_valid)
    meta['surface_intersection']=intersects.tolist()
    if scene['interaction'].get('source')=='hi4d_official_smpl_and_contact':
        for a,b in arrays.get('contact_parts',np.empty((0,2),int)):
            if min(a,b)>=0:cache['part_contact'][a,b,:2]=1
    depth=build_dual_condition(cache,meta,spec,NativeDualConfig(**dual_config))
    depth=replace(depth,contacts=replace(depth.contacts,kind=torch.ones_like(depth.contacts.kind),metadata=({'source':meta['contact_source'],'index_base':0,
        'confidence':'official_positive_with_estimated_SMPLX_attachment' if scene['interaction'].get('source')=='hi4d_official_smpl_and_contact' else 'geometric_candidate'},)))
    owner=depth.owner_weights[0,:,0];scene_valid=depth.person_a.camera_z_valid|depth.person_b.camera_z_valid
    scene_z=torch.minimum(torch.where(depth.person_a.camera_z_valid,depth.person_a.camera_z,torch.inf),
        torch.where(depth.person_b.camera_z_valid,depth.person_b.camera_z,torch.inf))
    scene_z=torch.where(scene_valid,scene_z,0.)
    sample=dict(source_canvas=scene['sources'][0]['canvas'],target_canvas=spec.metadata(),
        source_token_hw=CanvasSpec.from_record(scene['sources'][0]['canvas']).token_hw,target_token_hw=spec.token_hw,
        valid_image_mask=CanvasSpec.from_record(scene['sources'][0]['canvas']).valid_mask(),
        native_scene_kind='dual',supervised_target_rgb=False,task_id=torch.tensor(int(TaskType.DUAL)),
        person_b_valid=torch.tensor(True),dual_depth=depth,source_indices=torch.tensor([0,1]),
        pair_metadata=dict(pair_id=record_sha,cross_person=True,dataset_type='abc_unpaired_smplx_v1'))
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');boxes=[]
    for p,condition in enumerate((depth.person_a,depth.person_b)):
        person=people[p];joints=torch.from_numpy(person['joints_cam']).float();k=torch.from_numpy(driver['K']).float()
        uv=joints@k.T;uv=uv[:,:2]/joints[:,2,None].clamp_min(1e-8)
        valid=(joints[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        heat=torch.zeros(25,h,w)
        for name,index in BODY25_BY_NAME.items():
            joint=JOINT_NAMES.index(name)+1
            if valid[joint]:heat[index]=torch.exp(-((xx+.5-uv[joint,0])**2+(yy+.5-uv[joint,1])**2)/50)
        if valid[0]:heat[8]=torch.exp(-((xx+.5-uv[0,0])**2+(yy+.5-uv[0,1])**2)/50)
        ys,xs=condition.valid[0,0].nonzero(as_tuple=True)
        box=torch.tensor([float(xs.min())/w,float(ys.min())/h,float(xs.max()+1)/w,float(ys.max()+1)/h]);boxes.append(box)
        row=dict(normal=condition.context.normal_cam[0]*torch.tensor([1.,-1.,-1.])[:,None,None],
            pose_heatmap=heat,part_onehot=condition.context.part_onehot[0],human_mask=condition.context.smpl_mask[0].float(),
            smplx_global=torch.from_numpy(person['global26']),body_pose=torch.from_numpy(person['body_pose']),
            joints_cam=joints,joint_uv=uv[1:],fk_valid=torch.ones(22,dtype=torch.bool),projection_valid=valid[1:],
            body_height=torch.tensor(person['body_height'],dtype=torch.float32),smpl_bbox_height=(box[3]-box[1])*h,
            pose_provenance=dict(record_id=meta['record_id']+f':{p}',record_sha256=record_sha,canvas_id=spec.fingerprint(),person_id=ids[p]))
        sample.update({key+('' if p==0 else '_b'):value for key,value in row.items()})
    rel,rv=relative11(torch.from_numpy(cache['joints_cam']),torch.from_numpy(np.stack([p['root_rotation'] for p in people])),
        torch.from_numpy(cache['body_height']),torch.stack(boxes))
    idx=torch.zeros(1,8,dtype=torch.long)
    sample.update(relative_geometry=rel,relative_orientation_valid=torch.tensor(rv),contact_raster=torch.from_numpy(arrays['contact_raster']),
        contact_relations=ContactRelationBatch(idx.clone(),idx.clone(),idx.clone(),idx.clone(),idx.clone(),torch.zeros(1,8,1),torch.zeros(1,8,dtype=torch.bool)))
    def zeros(*shape,boolean=False):return torch.zeros(1,*shape,dtype=torch.bool if boolean else torch.float32)
    values=dict(person_valid=torch.ones(1,2,dtype=torch.bool),source_indices=torch.tensor([[0,1]]),source_slot_ids=torch.tensor([[0,1]]),
        face_landmarks=zeros(2,72,3),face_point_valid=zeros(2,72,boolean=True),jaw_pose=zeros(2,3),jaw_valid=torch.ones(1,2,dtype=torch.bool),
        expression=zeros(2,10),expression_valid=torch.ones(1,2,dtype=torch.bool),face_boxes=zeros(2,4),face_roi_valid=zeros(2,boolean=True),
        hand_pose=zeros(2,2,45),hand_pose_valid=torch.ones(1,2,2,dtype=torch.bool),hand_keypoints=zeros(2,2,21,3),hand_point_valid=zeros(2,2,21,boolean=True),
        mesh_bps=zeros(2,2,1024),mesh_valid=torch.ones(1,2,2,dtype=torch.bool),hand_boxes=zeros(2,2,4),hand_roi_valid=zeros(2,2,boolean=True),wrist_z=zeros(2,2),
        region_masks=zeros(6,h,w),visibility=zeros(6,h,w),local_z=zeros(6,h,w),local_z_valid=zeros(6,h,w,boolean=True),part_visible=zeros(6,h,w,boolean=True),
        person_owner=owner[None],scene_z=scene_z,scene_z_valid=scene_valid,body_height=torch.from_numpy(cache['body_height'][None]))
    for p,person in enumerate(people):
        for field,key in (('face_landmarks','face_points'),):
            values[field][0,p]=torch.from_numpy(local[key][p][0]);values['face_point_valid'][0,p]=torch.from_numpy(local[key][p][1])
        for field,value in (('jaw_pose',person['jaw_pose']),('expression',person['expression_full']),('hand_pose',person['hand_pose']),
                ('mesh_bps',local['bps'][p]),('face_boxes',local['boxes'][p,0]),('hand_boxes',local['boxes'][p,1:]),('wrist_z',person['wrist_cam'][:,2])):
            values[field][0,p]=torch.from_numpy(value)
        for side in range(2):
            values['hand_keypoints'][0,p,side]=torch.from_numpy(local['hand_points'][p][side][0]);values['hand_point_valid'][0,p,side]=torch.from_numpy(local['hand_points'][p][side][1])
        for region in range(3):
            slot=p if region==0 else 2+p*2+region-1
            surface=local['regions'][p][region];support=torch.from_numpy(surface['valid'])
            values['local_z'][0,slot]=torch.from_numpy(surface['z']);values['local_z_valid'][0,slot]=support
            values['region_masks'][0,slot]=support.float();values['visibility'][0,slot]=1;values['part_visible'][0,slot]=support&(owner[p]>0)
            if region==0:values['face_roi_valid'][0,p]=support.any()
            else:values['hand_roi_valid'][0,p,region-1]=support.any()
        if not scene['targets'][p].get('hand_geometry_available',True):
            values['hand_pose_valid'][0,p]=False;values['hand_point_valid'][0,p]=False;values['mesh_valid'][0,p]=False
            values['hand_pose'][0,p]=0;values['mesh_bps'][0,p]=0;values['hand_keypoints'][0,p]=0
            values['local_z_valid'][0,2+2*p:4+2*p]=False;values['local_z'][0,2+2*p:4+2*p]=0
    source=scene['sources'][0]
    values['metadata']=(dict(schema=LOCAL_SCHEMA,expression_mode='source',expression_retargeted=[True,True],
        counterfactual_geometry=True,paired_rgb_compatible=False,normal_space='camera_cv',hand_pose_convention='full_axis_angle_after_pose_mean',
        target_canvas_id=spec.fingerprint(),source_canvas_id=CanvasSpec.from_record(source['canvas']).fingerprint(),source_rgb_sha256=sha256(source['image']),
        source_person_ids=['source_A','source_B'],target_person_ids=ids,driver_record_sha256=record_sha,
        driver_rgb_sha256=sha256(scene['interaction']['image']),camera_id=camera_id,scale_group_id=record_sha,
        source_asset_sha256=hashlib.sha256(''.join(sha256(s['params_npz']) for s in scene['sources']).encode()).hexdigest(),
        expression_basis_sha256=[p['metadata']['expression_basis_sha256'] for p in people],bps=local['bps_metadata']),)
    values['metadata'][0]['fine_hand_geometry_source']='unavailable_in_official_SMPL' if scene['interaction'].get('source')=='hi4d_official_smpl_and_contact' else 'estimated_SMPLX'
    return sample,NativeLocalConditionV4(**values).validate()

def prepare(request):
    from retarget.backend import SourceBodyModel
    from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
    from src.pose_control.qwen21_full.native_dual_depth import NativeDualConfig
    from dataclasses import asdict
    scene=json.loads(Path(request['scene']).read_text(encoding='utf8'));out=Path(request['native_geometry_dir']);out.mkdir(parents=True,exist_ok=True)
    model=SourceBodyModel(request['settings']['smplest_root']);renderer=IndependentMeshRasterizer()
    records=[]
    for source in scene['sources']:
        with np.load(source['params_npz'],allow_pickle=False) as archive:K=archive['K'].copy()
        spec=CanvasSpec.from_record(source['canvas'])
        person=native_person(source['params_npz'],model,K,spec.original_hw,source['person_id'])
        asset={'people':[person],'K':K,'metadata':{'original_hw':spec.original_hw}}
        local=local_geometry(asset,spec,renderer=renderer)
        raw=renderer.render(person,spec.camera(K),spec.canvas_hw)
        records.append(write_record(out/(source['person_id']+'.npz'),{'asset':asset,'local':local,'raster':raw},schema='abc_native_source_geometry_v1'))
    spec=target_canvas(scene);K=np.asarray(scene['camera']['K'],np.float32)
    driver={'people':[native_person(t['params_npz'],model,K,spec.original_hw,t['person_id']) for t in scene['targets']],
        'K':K,'metadata':{'original_hw':spec.original_hw}}
    local=local_geometry(driver,spec,renderer=renderer)
    with np.load(scene['geometry_arrays'],allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
    config=asdict(NativeDualConfig(**request.get('dual_config',{})))
    sample,condition=target_sample(scene,driver,arrays,local,config,sha256(request['scene']))
    target=write_record(out/'target.npz',{'sample':sample,'local':condition},schema='abc_native_target_geometry_v1')
    (out/'index.json').write_text(json.dumps({'sources':records,'target':target,'dual_config':config,
        'scene_sha256':sha256(request['scene']),'target_rgb_loaded':False},indent=2),encoding='utf8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
    prepare(json.loads(args.request.read_text(encoding='utf8')))
