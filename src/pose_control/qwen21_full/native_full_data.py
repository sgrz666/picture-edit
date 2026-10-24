"""Manifest-only V4 readers and offline same-fit cache assembly.

Dataset access never renders, estimates a body, or extracts appearance. Inference
does not open target RGB. Cache files contain raw/physical conditions only.
"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from src.data.canvas_contract import CanvasSpec
from .native_full_assets import read_record,write_record,sha256,local_geometry
from .native_local_condition import NativeLocalConditionV4,NativeLocalReferencesV4,LOCAL_SCHEMA
from .native_full_runtime import NativeFullIdentityV4
from .dual_data_runtime import identity_from_reference as dual_identity

PAIR_SCHEMA='native_full_pair_v4'
BODY_SCHEMA='native_full_body_record_v4'


def _verified(record):
    if sha256(record['path'])!=record['sha256']: raise ValueError('Input asset SHA differs: '+record['path'])
    return Path(record['path'])


def _rgb(record,spec):
    with Image.open(_verified(record)) as image:
        value=np.asarray(spec.image(image.convert('RGB'),fill=(128,128,128)),np.float32).copy()
    return torch.from_numpy(value).permute(2,0,1)/127.5-1


class NativeFullDatasetV4(torch.utils.data.Dataset):
    def __init__(self,manifest,*,load_target=False):
        self.rows=[json.loads(line) for line in Path(manifest).read_text(encoding='utf8').splitlines() if line.strip()]
        self.load_target=load_target
        ids=[r.get('pair_id') for r in self.rows]
        if not ids or any(not isinstance(key,str) or not key for key in ids) or len(set(ids))!=len(ids): raise ValueError('Unique nonempty V4 pairs required')
        for row in self.rows:
            if row.get('schema')!=PAIR_SCHEMA or row.get('scene_kind') not in ('single','dual'):
                raise ValueError('Explicit V4 manifest and scene kind required')

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        sample=read_record(row['body'],schema=BODY_SCHEMA)
        if 'src_image' in sample or 'tgt_image' in sample: raise ValueError('Body cache must not embed RGB')
        local=read_record(row['local'],schema=LOCAL_SCHEMA).validate()
        refs=read_record(row['references'],schema='native_local_reference_v4').validate()
        source=CanvasSpec.from_record(sample['source_canvas']);target=CanvasSpec.from_record(sample['target_canvas'])
        if source.fingerprint()!=local.metadata[0]['source_canvas_id'] or target.fingerprint()!=local.metadata[0]['target_canvas_id']:
            raise ValueError('Source/driver Canvas differs from local cache')
        if refs.metadata[0]['source_rgb_sha256']!=row['source_rgb']['sha256']: raise ValueError('Reference cache is from another source image')
        if sample['pair_metadata']['pair_id']!=row['pair_id']: raise ValueError('Body pair differs from manifest')
        for key in ('source','target'):
            if sample['pair_metadata'].get(key+'_rgb_sha256')!=row[key+'_rgb']['sha256']:
                raise ValueError('Paired RGB declaration differs from cached geometry: '+key)
        sample.update(native_scene_kind=row['scene_kind'],local_condition_v4=local,local_references_v4=refs,
            src_image=_rgb(row['source_rgb'],source),supervised_target_rgb=self.load_target)
        if self.load_target:
            if local.metadata[0].get('counterfactual_geometry') and not local.metadata[0].get('paired_rgb_compatible',False):
                raise ValueError('Counterfactual expression/geometry cannot use original target RGB')
            sample['tgt_image']=_rgb(row['target_rgb'],target)
        return sample


def collate_native_full_v4(samples):
    from .depth_condition_v2 import _TensorRecord
    from .conditions import ContactRelationBatch
    if not samples or any(set(s)!=set(samples[0]) for s in samples): raise ValueError('Bucket single/dual and train/inference separately')
    output={}
    for key in samples[0]:
        values=[s[key] for s in samples];first=values[0]
        if key in ('source_token_hw','target_token_hw','native_scene_kind','supervised_target_rgb'):
            if any(v!=first for v in values): raise ValueError('Bucket exact grids/scene kind')
            output[key]=first
        elif isinstance(first,_TensorRecord): output[key]=type(first).stack(values)
        elif isinstance(first,ContactRelationBatch):
            output[key]=ContactRelationBatch(**{n:torch.cat([getattr(v,n) for v in values]) for n in first.__dataclass_fields__})
        elif torch.is_tensor(first): output[key]=torch.stack(values)
        else: output[key]=deepcopy(values)
    return output


def identity_from_reference_v4(batch,source_latent,source_token_hw):
    if batch['native_scene_kind']=='dual': return dual_identity(batch,source_latent,source_token_hw)
    b,n,c=source_latent.shape
    if c!=64 or n!=source_token_hw[0]*source_token_hw[1] or tuple(source_token_hw)!=tuple(batch['source_token_hw']):
        raise ValueError('Reuse the actual64-channel reference prefix, without re-encoding')
    d=source_latent.device;person=batch['source_person_mask'].to(d)
    active=torch.tensor([[True,False]],device=d).expand(b,-1)
    return NativeFullIdentityV4(source_latent.float().transpose(1,2).reshape(b,64,*source_token_hw),
        person,batch['source_head_mask'].to(d),batch['source_indices'].to(d),person.flatten(2).any(2),active,
        tuple(batch['identity_metadata'])).validate()


def assemble_full_cache(job,*,renderer):
    """Future executable builder: all target conditions use one imported SMPL-X fit.

    FCL is optional only when the manifest explicitly declares distances absent.
    Source appearance is a separately built source-only record.
    """
    from .native_dual_depth import NativeDualConfig
    from .dual_depth_geometry import build_dual_condition,person_condition,relative11,ownership
    from .dual_hi4d_geometry import part_distances
    from .body_pose import BODY25_BY_NAME,JOINT_NAMES
    from .conditions import PART_NAMES,ContactRelationBatch,TaskType
    from .depth_geometry import onto_canvas
    source=read_record(job['source_asset'],schema='native_smplx_asset_v4')
    driver=read_record(job['driver_asset'],schema='native_smplx_asset_v4')
    if job['units']!=driver['metadata'].get('units') or job['scale_status']!=driver['metadata'].get('scale_status'):
        raise ValueError('Depth scale must match imported geometry; metre needs an explicit bound calibration asset')
    source_spec=CanvasSpec.from_record(job['source_canvas']);target_spec=CanvasSpec.from_record(job['target_canvas'])
    if source_spec.original_hw!=tuple(source['metadata']['original_hw']) or target_spec.original_hw!=tuple(driver['metadata']['original_hw']):
        raise ValueError('Each fit must use its own original image and CanvasSpec')
    count=len(driver['people']);ids=[p['metadata']['person_id'] for p in driver['people']]
    if count not in (1,2) or len(source['people']) not in (1,2): raise ValueError('One or two real persons required')
    source_ids=[p['metadata']['person_id'] for p in source['people']]
    if len(job['source_masks'])!=len(source_ids): raise ValueError('Declare one visible mask per real Source person')
    for asset,rgb in ((source,job['source_rgb']),(driver,job['target_rgb'])):
        with Image.open(_verified(rgb)) as image:
            if (image.height,image.width)!=tuple(asset['metadata']['original_hw']): raise ValueError('Fit/RGB original dimensions differ')
        if asset['metadata'].get('rgb_sha256')!=rgb['sha256']: raise ValueError('Fit provenance belongs to another RGB frame')
    mapping=job['source_indices']
    if len(mapping)!=2 or any(mapping[p]<0 or mapping[p]>=len(source_ids) or source_ids[mapping[p]]!=ids[p] for p in range(count)):
        raise ValueError('Stable source identity mapping differs')
    record_sha=job['driver_asset']['sha256'];record_id=job['pair_id']+':smplx'
    meta={**driver['metadata'],'record_id':record_id,'record_sha256':record_sha,'pair_id':job['pair_id'],
        'person_ids':ids,'model_family_indices':[0]*count,'units':job['units'],'scale_status':job['scale_status']}
    config=NativeDualConfig(**job.get('dual_config',{}))
    raw=[renderer.render(p,driver['K'],target_spec.original_hw) for p in driver['people']]
    cache={name:np.stack([r[name] for r in raw]) for name in ('z','valid','normal','part')}
    cache.update(K=driver['K'],joints_cam=np.stack([p['joints_cam'] for p in driver['people']]),
        body_height=np.array([p['body_height'] for p in driver['people']],np.float32))
    if count==2:
        if job.get('surface_distance')=='fcl':
            distance,points,valid,wv,intersect=part_distances(driver['people'])
        elif job.get('surface_distance')=='unavailable':
            distance=np.zeros((14,14),np.float32);points=np.zeros((14,14,2,3),np.float32);valid=np.zeros((14,14),bool);wv=valid.copy();intersect=valid.copy()
        else: raise ValueError('Declare FCL or unavailable, never infer from missing packages')
        cache.update(part_distance=distance,closest_points=points,part_distance_valid=valid,closest_points_valid=wv,
            part_contact=np.zeros((14,14,4),np.float32),contact_indices=np.empty((0,4),np.int64),contact_distance=np.empty(0,np.float32))
        meta['contact_source']='unavailable; not_negative_labels';meta['surface_intersection']=intersect.tolist()
        depth=build_dual_condition(cache,meta,target_spec,config)
        people=(depth.person_a,depth.person_b)
        scene_z=torch.minimum(torch.where(people[0].camera_z_valid,people[0].camera_z,torch.inf),torch.where(people[1].camera_z_valid,people[1].camera_z,torch.inf))
        scene_valid=people[0].camera_z_valid|people[1].camera_z_valid;scene_z=torch.where(scene_valid,scene_z,0.)
        owner=depth.owner_weights[0,:,0]
    else:
        depth=person_condition(cache,meta,target_spec,0,config.depth_config().geometry_config(),physical=True)
        people=(depth,);scene_z=depth.camera_z;scene_valid=depth.camera_z_valid
        owner=torch.cat((depth.valid[0].float(),torch.zeros_like(depth.valid[0]).float()),0)
    basis=None
    if job.get('bps_basis'):
        basis=np.load(_verified(job['bps_basis']),allow_pickle=False)
    local=local_geometry(driver,target_spec,renderer=renderer,seed=job.get('bps_seed',20261024),basis=basis)
    h,w=target_spec.canvas_hw;source_h,source_w=source_spec.canvas_hw
    source_mask=torch.zeros(2,1,source_h,source_w,dtype=torch.bool);source_head=torch.zeros_like(source_mask)
    for p,ref in enumerate(job['source_masks']):
        with Image.open(_verified(ref)) as im:
            original=torch.from_numpy(np.asarray(im.convert('L')).copy()>0)[None]
        mask=source_spec.stack(original.float(),nearest=True)>.5
        source_mask[p]=mask
        rendered=renderer.render(source['people'][p],source['K'],source_spec.original_hw)
        head=source_spec.stack(torch.from_numpy(rendered['part']==1)[None].float(),nearest=True)>.5
        source_head[p]=head&mask
    sample={'source_canvas':source_spec.metadata(),'target_canvas':target_spec.metadata(),'source_token_hw':source_spec.token_hw,
        'target_token_hw':target_spec.token_hw,'valid_image_mask':target_spec.valid_mask(),'source_person_mask':source_mask,'source_head_mask':source_head,
        'source_indices':torch.tensor(mapping,dtype=torch.long),'task_id':torch.tensor(int(TaskType.SINGLE if count==1 else TaskType.DUAL)),
        'pair_metadata':{'pair_id':job['pair_id'],'cross_person':False,'dataset_type':'native_full_smplx_v4',
            'source_rgb_sha256':job['source_rgb']['sha256'],'target_rgb_sha256':job['target_rgb']['sha256'],
            'source_asset_sha256':job['source_asset']['sha256'],'driver_asset_sha256':job['driver_asset']['sha256']},
        'identity_metadata':{'schema':'native_dual_identity_v3' if count==2 else 'native_single_identity_v4','role':'source_only',
            'source_canvas_id':source_spec.fingerprint(),'source_rgb_sha256':job['source_rgb']['sha256'],
            'source_person_ids':source_ids+(['__inactive__'] if len(source_ids)==1 else []),
            'target_person_ids':ids+(['__inactive__'] if count==1 else []),'source_indices':mapping,
            'source_geometry_sha256':job['source_asset']['sha256'],'source_mask_sha256':[r['sha256'] for r in job['source_masks']]}}
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');body_boxes=[]
    for p,condition in enumerate(people):
        person=driver['people'][p];joints=torch.from_numpy(person['joints_cam']).float();k=torch.from_numpy(target_spec.camera(driver['K'])).float()
        uv=joints@k.T;uv=uv[:,:2]/joints[:,2,None].clamp_min(1e-8)
        x0,y0,x1,y1=target_spec.valid_rect
        valid=(joints[:,2]>0)&(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)
        heat=torch.zeros(25,h,w)
        for name,index in BODY25_BY_NAME.items():
            j=JOINT_NAMES.index(name)+1
            if valid[j]: heat[index]=torch.exp(-((xx+.5-uv[j,0])**2+(yy+.5-uv[j,1])**2)/50)
        if valid[0]: heat[8]=torch.exp(-((xx+.5-uv[0,0])**2+(yy+.5-uv[0,1])**2)/50)
        ys,xs=condition.valid[0,0].nonzero(as_tuple=True)
        box=torch.tensor([float(xs.min())/w,float(ys.min())/h,float(xs.max()+1)/w,float(ys.max()+1)/h]) if xs.numel() else torch.zeros(4)
        body_boxes.append(box)
        row={'normal':condition.context.normal_cam[0]*torch.tensor([1.,-1.,-1.])[:,None,None],
            'pose_heatmap':heat,'part_onehot':condition.context.part_onehot[0],'human_mask':condition.context.smpl_mask[0].float(),
            'smplx_global':target_spec.global_camera(torch.from_numpy(person['global26'])), 'body_pose':torch.from_numpy(person['body_pose']),
            'joints_cam':joints,'joint_uv':uv[1:],'fk_valid':torch.ones(22,dtype=torch.bool),'projection_valid':valid[1:],
            'body_height':torch.tensor(person['body_height'],dtype=torch.float32),'smpl_bbox_height':(box[3]-box[1])*h,
            'pose_provenance':{'record_id':record_id+f':{p}','record_sha256':record_sha,'canvas_id':target_spec.fingerprint(),'person_id':ids[p]}}
        sample.update({key+('' if p==0 else '_b'):value for key,value in row.items()})
    if count==2:
        sample['dual_depth']=depth;sample['person_b_valid']=torch.tensor(True)
        rel,rv=relative11(torch.from_numpy(cache['joints_cam']),torch.from_numpy(np.stack([p['root_rotation'] for p in driver['people']])),torch.from_numpy(cache['body_height']),torch.stack(body_boxes))
        sample['relative_geometry']=rel;sample['relative_orientation_valid']=torch.tensor(rv);sample['contact_raster']=torch.zeros(2,h,w)
        idx=torch.zeros(1,8,dtype=torch.long)
        sample['contact_relations']=ContactRelationBatch(idx.clone(),idx.clone(),idx.clone(),idx.clone(),idx.clone(),torch.zeros(1,8,1),torch.zeros(1,8,dtype=torch.bool))
    else:
        sample['depth_condition_v2']=depth;sample['model_family_a']=torch.tensor(0,dtype=torch.long);sample['single_owner_weights']=owner[:1]
    def zeros(*shape,boolean=False): return torch.zeros(1,*shape,dtype=torch.bool if boolean else torch.float32)
    pv=torch.tensor([[True,count==2]])
    values=dict(person_valid=pv,source_indices=torch.tensor([mapping],dtype=torch.long),source_slot_ids=torch.tensor([job['source_slot_ids']],dtype=torch.long),
        face_landmarks=zeros(2,72,3),face_point_valid=zeros(2,72,boolean=True),jaw_pose=zeros(2,3),jaw_valid=zeros(2,boolean=True),
        expression=zeros(2,10),expression_valid=zeros(2,boolean=True),face_boxes=zeros(2,4),face_roi_valid=zeros(2,boolean=True),
        hand_pose=zeros(2,2,45),hand_pose_valid=zeros(2,2,boolean=True),hand_keypoints=zeros(2,2,21,3),hand_point_valid=zeros(2,2,21,boolean=True),
        mesh_bps=zeros(2,2,1024),mesh_valid=zeros(2,2,boolean=True),hand_boxes=zeros(2,2,4),hand_roi_valid=zeros(2,2,boolean=True),wrist_z=zeros(2,2),
        region_masks=zeros(6,h,w),visibility=zeros(6,h,w),local_z=zeros(6,h,w),local_z_valid=zeros(6,h,w,boolean=True),
        part_visible=zeros(6,h,w,boolean=True),person_owner=owner[None],scene_z=scene_z,scene_z_valid=scene_valid,body_height=zeros(2))
    for p in range(count):
        person=driver['people'][p];values['face_landmarks'][0,p]=torch.from_numpy(local['face_points'][p][0]);values['face_point_valid'][0,p]=torch.from_numpy(local['face_points'][p][1])
        values['jaw_pose'][0,p]=torch.from_numpy(person['jaw_pose']);values['jaw_valid'][0,p]=True
        expression=person['expression_full']
        if expression.shape==(10,): values['expression'][0,p]=torch.from_numpy(expression);values['expression_valid'][0,p]=True
        values['face_boxes'][0,p]=torch.from_numpy(local['boxes'][p,0]);values['body_height'][0,p]=person['body_height']
        values['hand_pose'][0,p]=torch.from_numpy(person['hand_pose']);values['hand_pose_valid'][0,p]=True
        values['mesh_bps'][0,p]=torch.from_numpy(local['bps'][p]);values['mesh_valid'][0,p]=True
        values['hand_boxes'][0,p]=torch.from_numpy(local['boxes'][p,1:]);values['wrist_z'][0,p]=torch.from_numpy(person['wrist_cam'][:,2])
        for side in range(2):
            values['hand_keypoints'][0,p,side]=torch.from_numpy(local['hand_points'][p][side][0]);values['hand_point_valid'][0,p,side]=torch.from_numpy(local['hand_points'][p][side][1])
        for r in range(3):
            region=p if r==0 else 2+p*2+r-1
            surface=local['regions'][p][r];support=torch.from_numpy(surface['valid'])
            values['local_z'][0,region]=torch.from_numpy(surface['z']);values['local_z_valid'][0,region]=support
            values['region_masks'][0,region]=support.float();values['visibility'][0,region]=1
            values['part_visible'][0,region]=support&(owner[p]>0)
            if r==0: values['face_roi_valid'][0,p]=support.any()
            else: values['hand_roi_valid'][0,p,r-1]=support.any()
    reference_meta=read_record(job['references'],schema='native_local_reference_v4').metadata[0]
    if reference_meta['reference_sources'][0]['geometry_sha256']!=job['source_asset']['sha256']:
        raise ValueError('Local Source ROI geometry differs from full Source asset')
    mode=job.get('expression_mode','source')
    retargeted=[p['metadata'].get('expression_retargeted',False) for p in driver['people']]+([False] if count==1 else [])
    if mode=='source' and any(not retargeted[p] for p in range(count)):
        raise ValueError('Source SMPL-X parameters are available: import driver with expression_source, or explicitly choose driver/off')
    if mode=='source' and driver['metadata'].get('expression_source_sha256')!=job['source_asset']['sha256']:
        raise ValueError('Retargeted Source expression fingerprint differs')
    counterfactual=bool(driver['metadata'].get('counterfactual_geometry',False))
    metadata={'schema':LOCAL_SCHEMA,'expression_mode':mode,'expression_retargeted':retargeted,
        'counterfactual_geometry':counterfactual,'paired_rgb_compatible':not counterfactual,'normal_space':'camera_cv','hand_pose_convention':'full_axis_angle_after_pose_mean',
        'target_canvas_id':target_spec.fingerprint(),'source_canvas_id':source_spec.fingerprint(),'source_rgb_sha256':job['source_rgb']['sha256'],
        'source_person_ids':sample['identity_metadata']['source_person_ids'],'target_person_ids':sample['identity_metadata']['target_person_ids'],
        'driver_record_sha256':record_sha,'camera_id':meta['camera_id'],'scale_group_id':meta['scale_group_id'],
        'driver_rgb_sha256':job['target_rgb']['sha256'],
        'source_asset_sha256':job['source_asset']['sha256'],'bps':local['bps_metadata'],'reference_sources':reference_meta['reference_sources']}
    metadata['expression_basis_sha256']=[p['metadata']['expression_basis_sha256'] for p in driver['people']]+([None] if count==1 else [])
    values['metadata']=(metadata,)
    return sample,NativeLocalConditionV4(**values).validate()
