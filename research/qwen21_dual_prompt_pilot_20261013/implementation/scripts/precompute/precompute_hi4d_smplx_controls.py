"""Build the approved two-frame SMPL-X pilot, with separate generation/truth files."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.data.hi4d_smplx_dataset import (SMPLXRenderer,body25_heatmaps,file_sha256,
    geometry_vectors,hi4d_contact_pairs,letterbox_affine,read_mask,read_rgb,
    smplx_part_labels,source_person_crops,visible_owner,anatomical_part_geometry,
    source_cache_metadata,validate_source_feature_cache,validate_generation_manifest,
    validate_geometry_schema,bind_supervision_manifest,hi4d_asset_identity)
from src.pose_control.qwen21_full.dual.schema import (ANATOMICAL_PART_JOINT_INDICES,
    SMPLX55_JOINT_NAMES,SMPLX14_PARTITION_SPEC,generation_versions,geometry_schema_metadata)


def project(points,K,E):
    camera = points@E[:,:3].T+E[:,3]
    value = camera@K.T
    return value[:,:2]/value[:,2:]


def pose_observations(work,action,frame,camera,K,E,fit_indices):
    path = work/'observations'/f'{action}_{frame:06d}_cam{camera}.npz'
    obs = np.load(path)
    tri = np.load(work/'observations'/f'{action}_{frame:06d}_triangulated.npz')
    fitted = np.load(work/'fit_v1/wholebody134_by_camera.npz')
    view = int(np.flatnonzero(fitted['camera_ids']==camera)[0])
    affine = letterbox_affine(int(obs['width']),int(obs['height']))
    xy = obs['xy_native'][:,:24].copy()
    xy = xy*affine[0,0]+affine[:2,2]
    score = obs['scores'][:,:24].copy()
    provenance = []
    for i,index in enumerate(fit_indices):
        kinds = np.full(24,'camera_dwpose',dtype='<U24')
        missing = score[i]<.25
        reliable = tri['valid'][i,:24]&missing
        xy[i,reliable] = project(tri['xyz_world'][i,:24][reliable],K,E)
        score[i,reliable] = .7
        kinds[reliable] = 'multiview_triangulated'
        missing = score[i]<.25
        if missing.any():
            xyz = fitted['xyz_world'][index,view,:24]
            xy[i,missing] = project(xyz[missing],K,E)
            score[i,missing] = .25
            kinds[missing] = 'fitted_smplx_prior'
        provenance.append(kinds.tolist())
    return np.stack([body25_heatmaps(xy[i],score[i]) for i in range(2)]),provenance


def contact_controls(raw,vertices,labels,normals_world,owner,K,E,confidence_parts):
    pairs = hi4d_contact_pairs(raw['contact'])
    if not len(pairs):
        return np.empty((0,2),np.int64),np.empty((0,15),np.float32),np.zeros((1,512,512),np.float32),[]
    ea,va = cKDTree(vertices[0]).query(raw['verts'][0,pairs[:,0]])
    eb,vb = cKDTree(vertices[1]).query(raw['verts'][1,pairs[:,1]])
    parts = np.stack((labels[va],labels[vb]),-1)
    records = []
    yy,xx = np.mgrid[:512,:512]
    for pa,pb in np.unique(parts,axis=0):
        select = (parts[:,0]==pa)&(parts[:,1]==pb)
        # Pick a real corresponding vertex pair near this patch's centre, not a
        # mean point inside the bodies. SMPL->SMPL-X transfer remains a proxy.
        candidates = np.flatnonzero(select)
        centres = (vertices[0,va[select]]+vertices[1,vb[select]])/2
        pick = candidates[np.argmin(np.linalg.norm(centres-centres.mean(0),axis=-1))]
        anchors = np.stack((vertices[0,va[pick]],vertices[1,vb[pick]]))
        mapping_error = float((ea[select]+eb[select]).mean()/2)
        confidence = float(np.exp(-mapping_error/.03)*min(confidence_parts[0,pa],confidence_parts[1,pb]))
        if confidence<.05: continue
        delta = anchors[1]-anchors[0]
        nA,nB = normals_world[0][va[pick]],normals_world[1][vb[pick]]
        uv = project(anchors,K,E)
        visibility = []
        for i,point in enumerate(uv):
            x,y = np.round(point).astype(int)
            visibility.append(float(owner[i,y,x]) if 0<=x<512 and 0<=y<512 else 0.)
        features = np.r_[delta,np.linalg.norm(delta),nA,nB,np.dot(nA,nB),0.,confidence,visibility]
        records.append(dict(parts=[int(pa),int(pb)],features=features.astype(np.float32),
            anchors=anchors.tolist(),mapping_error_m=mapping_error,confidence=confidence,
            candidates=int(select.sum()),signed_gap_valid=False,
            original_smpl_pairs=pairs[select].tolist()))
    records.sort(key=lambda r:r['confidence']*np.sqrt(r['candidates']),reverse=True)
    records=records[:8]
    contact_map = np.zeros((512,512),np.float32)
    for r in records:
        for x,y in project(np.asarray(r['anchors']),K,E):
            contact_map = np.maximum(contact_map,np.exp(-((xx-x)**2+(yy-y)**2)/(2*12**2))*r['confidence'])
    contact_map *= np.minimum(owner.sum(0),1)
    endpoints = np.asarray([[r['parts'][0],14+r['parts'][1]]for r in records],np.int64).reshape(-1,2)
    features = np.asarray([r.pop('features')for r in records],np.float32).reshape(-1,15)
    return endpoints,features,contact_map[None],records


def controls(root,work,out,fit,indices,action,frame,camera,labels,heights,renderer):
    obs_path = work/'observations'/f'{action}_{frame:06d}_cam{camera}.npz'
    obs = np.load(obs_path)
    K = letterbox_affine(int(obs['width']),int(obs['height']))@obs['K']
    E = obs['extrinsics']
    vertices = fit['vertices_world'][indices]
    rendered = [renderer.render(v,E[:,:3],E[:,3],K) for v in vertices]
    amodal = np.stack([r['mask'] for r in rendered])
    depth = np.stack([r['depth'] for r in rendered])
    owner = visible_owner(amodal,depth)
    pose,pose_provenance = pose_observations(work,action,frame,camera,K,E,indices)
    part_confidence = np.full((2,14),.9,np.float32)
    part_confidence[:,[4,7]]=.25  # occluded finger pose remains a prior in this pilot
    positions = np.stack([[v[labels==part].mean(0) for part in range(14)]for v in vertices])
    orientations,part_joints,part_valid = anatomical_part_geometry(fit['joints_world'][indices])
    part_valid &= np.isfinite(positions).all(-1)
    for value in (positions, orientations, part_joints, part_confidence):
        value[~part_valid] = 0
    source_indices = []
    for actor in fit['actor_ids'][indices]:
        match = np.flatnonzero((fit['actions']=='talk14') & (fit['frame_ids']==32) & (fit['actor_ids']==actor))
        if len(match) != 1:
            raise ValueError('pilot requires one source-frame shape per actor')
        source_indices.append(int(match[0]))
    source_betas = fit['betas'][source_indices]
    if not np.allclose(source_betas,fit['betas'][indices],rtol=0,atol=1e-6):
        raise ValueError('target fitted mesh must preserve the source actor beta10')
    global32,relative14 = geometry_vectors(source_betas,fit['global_orient'][indices],
        fit['joints_world'][indices,0],E[:,:3],E[:,3],K,amodal,heights)
    raw_path = root/'datasets/Hi4D_pilot_v1/raw/pair14'/action/'smpl'/f'{frame:06d}.npz'
    raw = np.load(raw_path)
    normals = [r['vertex_normals_camera']@E[:,:3]for r in rendered]
    endpoints,features,contact_map,contacts = contact_controls(raw,vertices,labels,normals,owner,K,E,part_confidence)
    values = dict(normal=np.stack([r['normal']for r in rendered]),depth=depth[:,None],
        part=np.stack([r['part']for r in rendered]),pose=pose,amodal_mask=amodal[:,None].astype(np.float32),
        visible_owner=owner[:,None],global32=global32,relative14=relative14,
        part_positions=positions.astype(np.float32),part_orientations=orientations,part_joints=part_joints,
        part_valid=part_valid,part_confidence=part_confidence,
        contact_endpoints=endpoints,contact_features=features,contact_valid=np.ones(len(endpoints),bool),
        contact_map=contact_map.astype(np.float32),person_valid=np.ones(2,bool),
        person_uids=np.array(['pair14_actor0','pair14_actor1']),
        **geometry_schema_metadata(), source_shape_shared=np.array(True),
        source_shape_fit_indices=np.asarray(source_indices,np.int64),
        anatomical_joint_indices=np.asarray(ANATOMICAL_PART_JOINT_INDICES,np.int64),
        anatomical_joint_names=np.asarray(SMPLX55_JOINT_NAMES),
        smplx_sha256=file_sha256(work/'fit_v1/smplx_world.npz'),
        observation_sha256=file_sha256(obs_path),contact_source_sha256=file_sha256(raw_path),
        vertex_partition_sha256=hashlib.sha256(np.asarray(labels,dtype='<i8').tobytes()).hexdigest())
    name=f'{action}_{frame:06d}_cam{camera}'
    np.savez_compressed(out/f'{name}_geometry.npz',**values)
    info=dict(version=values['schema_version'],**geometry_schema_metadata(),
        pair='pair14',action=action,target_frame=int(frame),source_frame=32,camera=int(camera),
        person_uids=values['person_uids'].tolist(),fit_indices=np.asarray(indices).tolist(),
        geometry=(out/f'{name}_geometry.npz').name,geometry_sha256=file_sha256(out/f'{name}_geometry.npz'),
        fitted_pseudo_annotation=True,source_shape_shared=True,
        source_shape_fit_indices=source_indices,source_shape_verification='target/source beta10 allclose atol=1e-6 rtol=0',
        anatomical_joint_indices=list(ANATOMICAL_PART_JOINT_INDICES),
        anatomical_joint_names=list(SMPLX55_JOINT_NAMES),
        anatomical_orientation_rule='unit world bone direction: neck->head; pelvis->spine3; upper/lower limb proximal->distal; wrist->mean 15 finger joints; ankle->foot',
        partition_spec=SMPLX14_PARTITION_SPEC,vertex_partition_sha256=values['vertex_partition_sha256'],
        smplx_sha256=file_sha256(work/'fit_v1/smplx_world.npz'),observation_sha256=file_sha256(obs_path),
        contact_source_sha256=file_sha256(raw_path),contacts=contacts,pose_provenance=pose_provenance,
        K512=K.tolist(),extrinsics=E.tolist(),source_neutral_heights=heights.tolist(),
        signed_gap_valid=False,contact_mapping='Hi4D one-based SMPL correspondences -> nearest fitted SMPL-X surfaces')
    (out/f'{name}_geometry.json').write_text(json.dumps(info,indent=2)+'\n')
    panels = []
    for i in range(2):
        panels.extend([np.moveaxis((values['normal'][i]+1)*.5,-3,-1)*amodal[i,...,None],
            np.repeat(owner[i,...,None],3,-1),np.repeat(pose[i].max(0)[...,None],3,-1)])
    panels.append(np.repeat(contact_map[0,...,None],3,-1))
    sheet = Image.new('RGB',(512*len(panels),544),'white')
    draw=ImageDraw.Draw(sheet)
    for n,(panel,title) in enumerate(zip(panels,['A normals','A visible owner','A body25','B normals','B visible owner','B body25','contact confidence'])):
        sheet.paste(Image.fromarray((panel.clip(0,1)*255).astype(np.uint8)),(n*512,32));draw.text((n*512+8,8),title,fill='black')
    sheet.save(out/f'{name}_controls.png')
    return out/f'{name}_geometry.npz'


@torch.inference_mode()
def source_features(source_rgb,source_masks,dino_path,vae_path,out,device,*,person_uids):
    # This signature contains no target image, target mask, or target feature.
    from transformers import AutoModel,AutoImageProcessor
    from diffusers import AutoencoderKLQwenImage21
    rgb=(read_rgb(source_rgb).permute(1,2,0).numpy()*255).round().astype(np.uint8)
    masks=np.stack([read_mask(p)[0].numpy()for p in source_masks])
    crops,coverage,boxes=source_person_crops(rgb,masks)
    processor=AutoImageProcessor.from_pretrained(dino_path,local_files_only=True)
    dino=AutoModel.from_pretrained(dino_path,local_files_only=True).eval().to(device)
    patches,valid=[],[]
    for i in range(2):
        image=Image.fromarray(crops[i]).resize((224,224),Image.Resampling.BICUBIC)
        # Keep the whole square person crop; the original 256->224 centre crop
        # would discard outer source pixels and misalign our patch coverage.
        inputs=processor(images=image,return_tensors='pt',do_resize=False,do_center_crop=False)
        tokens=dino(**{k:v.to(device)for k,v in inputs.items()}).last_hidden_state[0,1:]
        if tokens.shape!=(256,1536):raise ValueError(f'expected DINOv2-G patches, got {tokens.shape}')
        mask=F.adaptive_avg_pool2d(torch.from_numpy(coverage[i])[None,None],(16,16)).flatten()>.5
        if not mask.any():raise ValueError('source DINO crop has no valid person patches')
        patches.append(tokens.float().cpu()*mask[:,None]);valid.append(mask)
        Image.fromarray(crops[i]).save(out/f'source_person_{i}.png')
    del dino,processor
    gc.collect();torch.cuda.empty_cache()
    vae=AutoencoderKLQwenImage21.from_pretrained(vae_path,torch_dtype=torch.bfloat16,local_files_only=True).eval().to(device)
    means=torch.tensor(vae.config.latents_mean,device=device,dtype=vae.dtype)[None,:,None,None,None]
    stds=torch.tensor(vae.config.latents_std,device=device,dtype=vae.dtype)[None,:,None,None,None]
    latents=[]
    for crop in crops:
        tensor=torch.from_numpy(crop.copy()).permute(2,0,1).float()/127.5-1
        rgba=torch.cat((tensor,torch.ones(1,512,512)),0)[None,:,None].to(device=device,dtype=vae.dtype)
        latent=(vae.encode(rgba).latent_dist.mode()-means)/stds
        latents.append(latent[0,:,0].float().cpu())
    del vae
    gc.collect();torch.cuda.empty_cache()
    latent_masks=F.adaptive_avg_pool2d(torch.from_numpy(coverage)[:,None],(32,32))
    metadata=source_cache_metadata(source_rgb,source_masks,person_uids)
    encoder_provenance=dict(vae_path=str(vae_path),dino_path=str(dino_path),
        dino_sha256=file_sha256(Path(dino_path)/'model.safetensors'),
        vae_config_sha256=file_sha256(Path(vae_path)/'config.json'),
        vae_weights_sha256=file_sha256(Path(vae_path)/'diffusion_pytorch_model.safetensors'),
        dino_whole_crop_size=224,dino_min_patch_coverage=.5)
    metadata['encoder_provenance']=encoder_provenance
    payload=dict(person_latents=torch.stack(latents),latent_masks=latent_masks,
        dino_patches=torch.stack(patches),dino_valid=torch.stack(valid),crop_boxes=torch.from_numpy(boxes),
        metadata=metadata)
    torch.save(payload,out/'source_features.pt')
    provenance=dict(source_rgb=str(source_rgb),source_masks=list(map(str,source_masks)),
        metadata=metadata,source_sha256=metadata['source_rgb_sha256'],
        source_mask_sha256=metadata['source_mask_sha256'],source_features_sha256=file_sha256(out/'source_features.pt'),
        **encoder_provenance)
    (out/'source_features.json').write_text(json.dumps(provenance,indent=2)+'\n')


def upgrade_verified_pilot_controls(legacy,out,fit_path,*,smplx_model_path=None):
    """Augment verified legacy caches into a new directory without encoding/rasterizing.

    This is a narrowly scoped converter for the two pilot frames. It verifies
    input hashes, saved crops, encoder provenance and typed fit/camera vectors
    before attaching the new contract. Original files and encoded tensors stay
    byte-for-byte/tensor-for-tensor intact. Missing original VAE weight hashes
    are stated explicitly instead of claiming retrospective verification.
    """
    legacy,out,fit_path=map(Path,(legacy,out,fit_path))
    if out.resolve()==legacy.resolve() or out.exists():
        raise ValueError('upgrade output must be a new directory distinct from legacy controls')
    row=json.loads((legacy/'generation.json').read_text())
    source_info=json.loads((legacy/'source_features.json').read_text())
    old_feature_path=Path(row['source_features'])
    old_feature_sha=file_sha256(old_feature_path)
    if source_info.get('source_features_sha256')!=old_feature_sha:
        raise ValueError('legacy source feature SHA256 does not match its original provenance')
    metadata=source_cache_metadata(row['source_rgb'],row['source_masks'],row['person_uids'])
    if source_info.get('source_sha256')!=metadata['source_rgb_sha256'] or source_info.get('source_mask_sha256')!=metadata['source_mask_sha256']:
        raise ValueError('legacy source RGB/mask SHA256 does not match its original provenance')
    if source_info.get('dino_whole_crop_size')!=224 or source_info.get('dino_min_patch_coverage')!=.5:
        raise ValueError('legacy source crop/encoder coverage convention is not the verified pilot convention')
    rgb=(read_rgb(row['source_rgb']).permute(1,2,0).numpy()*255).round().astype(np.uint8)
    masks=np.stack([read_mask(p)[0].numpy()for p in row['source_masks']])
    crops,_,_=source_person_crops(rgb,masks)
    for i,crop in enumerate(crops):
        if not np.array_equal(np.asarray(Image.open(legacy/f'source_person_{i}.png').convert('RGB')),crop):
            raise ValueError(f'legacy source person {i} crop does not match the current crop rule')
    for path,key in ((Path(source_info['dino_path'])/'model.safetensors','dino_sha256'),
                     (Path(source_info['vae_path'])/'config.json','vae_config_sha256')):
        if file_sha256(path)!=source_info.get(key):
            raise ValueError(f'legacy encoder provenance {key} no longer matches the encoder file')
    if source_info.get('vae_weights_sha256') is not None:
        if file_sha256(Path(source_info['vae_path'])/'diffusion_pytorch_model.safetensors')!=source_info['vae_weights_sha256']:
            raise ValueError('legacy VAE weight SHA256 no longer matches the encoder file')
    encoder_keys=('dino_path','vae_path','dino_sha256','vae_config_sha256',
                  'dino_whole_crop_size','dino_min_patch_coverage')
    metadata['encoder_provenance']={key:source_info[key]for key in encoder_keys}
    metadata['encoder_provenance'].update(vae_weights_sha256=source_info.get('vae_weights_sha256'),
        vae_weight_hash_provenance=('verified_original_sha256' if source_info.get('vae_weights_sha256')
            else 'not_recorded_by_legacy_cache; original VAE config SHA256 verified'))
    metadata['upgraded_from_features_sha256']=old_feature_sha
    features=torch.load(old_feature_path,map_location='cpu',weights_only=True)
    if 'metadata' in features:
        validate_source_feature_cache(features,dict(row,versions=generation_versions()))
    features=dict(features,metadata=metadata)
    fit=np.load(fit_path,allow_pickle=False)
    fit_sha=file_sha256(fit_path)
    if smplx_model_path is None:
        smplx_model_path=fit_path.parents[3]/'add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz'
    with np.load(smplx_model_path,allow_pickle=False) as body_model:
        # Raw official SMPL-X archives call this `weights`; smplx.create
        # registers that array as the runtime object's `lbs_weights` buffer.
        labels=smplx_part_labels(body_model['weights'])
    vertex_partition_sha=hashlib.sha256(np.asarray(labels,dtype='<i8').tobytes()).hexdigest()
    manifests={}
    geometries={}
    for name in ('generation.json','generation_talk_control.json'):
        current=json.loads((legacy/name).read_text())
        if current['source_rgb']!=row['source_rgb'] or current['source_masks']!=row['source_masks'] or current['person_uids']!=row['person_uids'] or current['source_features']!=row['source_features']:
            raise ValueError('pilot manifests must bind the same source frame, masks, UIDs and cache')
        if current['versions']!={'source':'source_v1','geometry':'smplx_dual_v1'}:
            raise ValueError('upgrade supports only the verified legacy pilot versions')
        geometry_path=Path(current['target_geometry'])
        if geometry_path.name not in geometries:
            match=re.fullmatch(r'(talk14|hug14)_(\d{6})_cam4_geometry\.npz',geometry_path.name)
            if match is None:
                raise ValueError('upgrade supports only the fitted talk32/hug62 camera4 pilot')
            action,frame=match[1],int(match[2])
            if (action,frame) not in (('talk14',32),('hug14',62)):
                raise ValueError('upgrade supports only the fitted talk32/hug62 camera4 pilot')
            info=json.loads(geometry_path.with_suffix('.json').read_text())
            if info.get('version')!='smplx_dual_v1' or info.get('source_shape_shared') is not True or info.get('smplx_sha256')!=fit_sha:
                raise ValueError('legacy geometry fit/schema/source shape provenance is not verified')
            if 'geometry_sha256' in info and info['geometry_sha256']!=file_sha256(geometry_path):
                raise ValueError('legacy geometry SHA256 does not match its original provenance')
            with np.load(geometry_path,allow_pickle=False) as payload:
                values={key:payload[key].copy()for key in payload.files}
            if list(map(str,values['person_uids']))!=current['person_uids']:
                raise ValueError('legacy geometry UIDs do not match the source actor slots')
            indices=np.flatnonzero((fit['actions']==action)&(fit['frame_ids']==frame))
            source_indices=np.flatnonzero((fit['actions']=='talk14')&(fit['frame_ids']==32))
            if not np.array_equal(fit['actor_ids'][indices],[0,1]) or not np.array_equal(fit['actor_ids'][source_indices],[0,1]):
                raise ValueError('legacy fit actor ordering is not the verified pilot order')
            if not np.allclose(fit['betas'][indices],fit['betas'][source_indices],rtol=0,atol=1e-6):
                raise ValueError('legacy target mesh does not preserve source actor beta10')
            K,E=np.asarray(info['K512']),np.asarray(info['extrinsics'])
            global32,relative14=geometry_vectors(fit['betas'][source_indices],fit['global_orient'][indices],
                fit['joints_world'][indices,0],E[:,:3],E[:,3],K,values['amodal_mask'][:,0],np.asarray(info['source_neutral_heights']))
            if not np.allclose(values['global32'],global32,rtol=0,atol=1e-5) or not np.allclose(values['relative14'],relative14,rtol=0,atol=1e-5):
                raise ValueError('legacy global/relative geometry differs from the typed source fit and camera convention')
            orientations,joints,valid=anatomical_part_geometry(fit['joints_world'][indices])
            valid &= values['part_valid']
            for value in (orientations,joints,values['part_positions'],values['part_confidence']):
                value[~valid]=0
            values.update(geometry_schema_metadata(),part_orientations=orientations,part_joints=joints,
                part_valid=valid,source_shape_shared=np.array(True),source_shape_fit_indices=source_indices,
                anatomical_joint_indices=np.asarray(ANATOMICAL_PART_JOINT_INDICES,np.int64),
                anatomical_joint_names=np.asarray(SMPLX55_JOINT_NAMES),smplx_sha256=fit_sha,
                vertex_partition_sha256=vertex_partition_sha)
            info.update(geometry_schema_metadata(),source_shape_fit_indices=source_indices.tolist(),
                pair='pair14',action=action,target_frame=frame,source_frame=32,camera=4,
                person_uids=current['person_uids'],fit_indices=indices.tolist(),
                source_shape_verification='target/source beta10 allclose atol=1e-6 rtol=0',
                anatomical_joint_indices=list(ANATOMICAL_PART_JOINT_INDICES),
                anatomical_joint_names=list(SMPLX55_JOINT_NAMES),partition_spec=SMPLX14_PARTITION_SPEC,
                vertex_partition_sha256=vertex_partition_sha,upgraded_from_geometry_sha256=file_sha256(geometry_path))
            geometries[geometry_path.name]=(values,info)
        manifests[name]=dict(current,source_features=str(out/'source_features.pt'),
            target_geometry=str(out/geometry_path.name),versions=generation_versions())
    out.mkdir(parents=True)
    torch.save(features,out/'source_features.pt')
    feature_sha=file_sha256(out/'source_features.pt')
    source_info.update(metadata=metadata,source_masks=row['source_masks'],source_features_sha256=feature_sha,
        upgraded_from_features_sha256=old_feature_sha)
    (out/'source_features.json').write_text(json.dumps(source_info,indent=2)+'\n')
    for name,(values,info) in geometries.items():
        path=out/name
        np.savez_compressed(path,**values)
        info.update(geometry=name,geometry_sha256=file_sha256(path))
        path.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    for name,current in manifests.items():
        current['source_features_sha256']=feature_sha
        (out/name).write_text(json.dumps(current,indent=2)+'\n')
    for path in legacy.glob('*.png'):
        shutil.copy2(path,out/path.name)
    if (legacy/'supervision.json').exists():
        truth=bind_supervision_manifest(json.loads((legacy/'supervision.json').read_text()),manifests['generation.json'])
        (out/'supervision.json').write_text(json.dumps(truth,indent=2)+'\n')
    fit.close()
    return out


def bind_verified_pilot_supervision(controls,out,root):
    """Write v3 truth bindings using verified v2 files and real Hi4D provenance."""
    controls,out,root=map(Path,(controls,out,root))
    if out.exists() or out.resolve()==controls.resolve():
        raise ValueError('supervision output must be a new directory distinct from existing controls')
    manifests={name:json.loads((controls/name).read_text())for name in ('generation.json','generation_talk_control.json')}
    main=manifests['generation.json']
    for row in manifests.values():
        validate_generation_manifest(row)
        if row['source_rgb']!=main['source_rgb'] or row['source_masks']!=main['source_masks'] or row['source_features']!=main['source_features'] or row['person_uids']!=main['person_uids']:
            raise ValueError('supervision binding requires the same verified source in both generation manifests')
        if file_sha256(row['source_features'])!=row['source_features_sha256']:
            raise ValueError('supervision source feature content SHA256 differs from its generation manifest')
    validate_source_feature_cache(torch.load(main['source_features'],map_location='cpu',weights_only=True),main)
    source=hi4d_asset_identity(main['source_rgb'],target=False)
    fit_path=root/'research/hi4d_smplx_pilot_20261011/fit_v1/smplx_world.npz'
    fit_sha=file_sha256(fit_path)
    geometry_records={}
    with np.load(fit_path,allow_pickle=False) as fit:
        source_indices=np.flatnonzero((fit['actions']==source['action'])&(fit['frame_ids']==source['frame']))
        if not np.array_equal(fit['actor_ids'][source_indices],[0,1]):
            raise ValueError('supervision source frame/actor order differs from the fitted pilot')
        expected_uids=[f"{source['pair']}_actor{int(actor)}"for actor in fit['actor_ids'][source_indices]]
        if main['person_uids']!=expected_uids:
            raise ValueError('supervision generation UIDs do not match the actual fitted source actor IDs')
        for row in manifests.values():
            path=Path(row['target_geometry'])
            if path.name in geometry_records:
                continue
            match=re.fullmatch(r'(talk14|hug14)_(\d{6})_cam(\d+)_geometry\.npz',path.name)
            if match is None:
                raise ValueError('supervision geometry has no verifiable fitted action/frame/camera identity')
            action,frame,camera=match[1],int(match[2]),int(match[3])
            indices=np.flatnonzero((fit['actions']==action)&(fit['frame_ids']==frame))
            if not np.array_equal(fit['actor_ids'][indices],[0,1]) or camera!=source['camera']:
                raise ValueError('supervision geometry identity differs from the fitted pilot or source camera')
            info=json.loads(path.with_suffix('.json').read_text())
            actual_sha=file_sha256(path)
            if info.get('geometry_sha256')!=actual_sha or info.get('smplx_sha256')!=fit_sha:
                raise ValueError('supervision geometry/fit SHA256 differs from verified v2 provenance')
            with np.load(path,allow_pickle=False) as payload:
                validate_geometry_schema(payload,row['versions'])
                if list(map(str,payload['person_uids']))!=expected_uids:
                    raise ValueError('supervision geometry UIDs differ from actual fitted actors')
            observation=root/f'research/hi4d_smplx_pilot_20261011/observations/{action}_{frame:06d}_cam{camera}.npz'
            raw_contact=root/f"datasets/Hi4D_pilot_v1/raw/{source['pair']}/{action}/smpl/{frame:06d}.npz"
            if file_sha256(observation)!=info.get('observation_sha256') or file_sha256(raw_contact)!=info.get('contact_source_sha256'):
                raise ValueError('supervision action/frame/camera identity differs from actual observation/contact provenance')
            with np.load(observation,allow_pickle=False) as obs:
                expected_K=letterbox_affine(int(obs['width']),int(obs['height']))@obs['K']
                if not np.allclose(info['K512'],expected_K,rtol=0,atol=1e-5) or not np.allclose(info['extrinsics'],obs['extrinsics'],rtol=0,atol=1e-6):
                    raise ValueError('supervision camera calibration differs from its actual observation file')
            info.update(pair=source['pair'],action=action,target_frame=frame,source_frame=source['frame'],
                camera=camera,person_uids=expected_uids,fit_indices=indices.tolist(),
                identity_provenance=dict(fit=str(fit_path),fit_sha256=fit_sha,
                    fit_indices=indices.tolist(),observation=str(observation),raw_contact=str(raw_contact),
                    source_asset=str(main['source_rgb'])),
                copied_geometry_from=str(path),copied_geometry_sha256=actual_sha)
            geometry_records[path.name]=(path,info)
    old_truth=json.loads((controls/'supervision.json').read_text())
    target=hi4d_asset_identity(old_truth['target_rgb'],target=True)
    for key,value in (('pair',target['pair']),('action',target['action']),('target_frame',target['frame']),
                      ('camera',target['camera']),('source_frame',source['frame'])):
        if old_truth.get(key)!=value:
            raise ValueError(f'supervision original {key} differs from actual source/target asset identity')
    selected_info=geometry_records[Path(main['target_geometry']).name][1]
    if (selected_info['pair'],selected_info['action'],selected_info['target_frame'],selected_info['camera'])!=(target['pair'],target['action'],target['frame'],target['camera']):
        raise ValueError('supervision target asset identity does not match selected generation geometry')
    raw=root/f"datasets/Hi4D_pilot_v1/raw/{target['pair']}/{target['action']}"
    target_files=[old_truth['target_rgb'],*old_truth['target_masks']]
    actual_raw_files=[raw/f"images/{target['camera']}/{target['frame']:06d}.jpg",
        *[raw/f"seg/img_seg_mask/{target['camera']}/{actor}/{target['frame']:06d}.png"for actor in (0,1)]]
    if len(old_truth['target_masks'])!=2 or any(file_sha256(path)!=file_sha256(actual)for path,actual in zip(target_files,actual_raw_files)):
        raise ValueError('supervision target RGB/mask files do not match actual Hi4D frame/actor content')
    out.mkdir(parents=True)
    for name,(path,info) in geometry_records.items():
        shutil.copy2(path,out/name)
        (out/name).with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    for name,row in manifests.items():
        row=dict(row,target_geometry=str(out/Path(row['target_geometry']).name))
        manifests[name]=row
        (out/name).write_text(json.dumps(row,indent=2)+'\n')
    bound=bind_supervision_manifest(old_truth,manifests['generation.json'])
    (out/'supervision.json').write_text(json.dumps(bound,indent=2)+'\n')
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/shangguanrz/project/pic-edit'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dino',type=Path,default=Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/code/models/dinov2-giant'))
    p.add_argument('--skip-source-features',action='store_true')
    p.add_argument('--upgrade-from',type=Path,help='augment verified legacy controls into a new output without encoding or rasterizing')
    p.add_argument('--bind-supervision-from',type=Path,help='bind verified controls to actual GT content in a new output directory')
    a=p.parse_args()
    if a.bind_supervision_from is not None:
        result=bind_verified_pilot_supervision(a.bind_supervision_from,a.output,a.root)
        print(json.dumps({'supervision_bound_controls':str(result),'reencoded':False}),flush=True)
        return
    if a.upgrade_from is not None:
        result=upgrade_verified_pilot_controls(a.upgrade_from,a.output,
            a.root/'research/hi4d_smplx_pilot_20261011/fit_v1/smplx_world.npz',
            smplx_model_path=a.root/'add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz')
        print(json.dumps({'upgraded_controls':str(result),'reencoded':False}),flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    import smplx
    from smplx.joint_names import JOINT_NAMES
    if tuple(JOINT_NAMES[:55]) != SMPLX55_JOINT_NAMES:
        raise ValueError('installed SMPL-X does not use the canonical first 55 anatomical joint names')
    work=a.root/'research/hi4d_smplx_pilot_20261011'
    fit=np.load(work/'fit_v1/smplx_world.npz')
    model=smplx.create(str(a.root/'add/tools/SMPLest-X/human_models/human_model_files'),'smplx',
        gender='neutral',batch_size=2,use_pca=False,flat_hand_mean=True).eval()
    labels=smplx_part_labels(model.lbs_weights.numpy())
    with torch.no_grad(): neutral=model(betas=torch.from_numpy(fit['betas'][:2]).float()).vertices.numpy()
    heights=np.ptp(neutral[:,:,1],axis=1)
    if not np.allclose(fit['betas'][:2],fit['betas'][2:]):raise ValueError('pilot target mesh does not preserve source actor shape')
    renderer=SMPLXRenderer(fit['faces'],labels)
    paths=[]
    for action,frame in [('talk14',32),('hug14',62)]:
        indices=np.flatnonzero((fit['actions']==action)&(fit['frame_ids']==frame))
        if not np.array_equal(fit['actor_ids'][indices],[0,1]):raise ValueError('fitted actor ordering is inconsistent')
        paths.append(controls(a.root,work,a.output,fit,indices,action,frame,4,labels,heights,renderer))
    assets=a.root/'datasets/Hi4D_pilot_v1/final_assets_v4_2'
    source=assets/'sources/pair14/cam4/talk14/000032'
    target=assets/'targets/pair14/hug14/cam4/000062'
    masks=[source/'mask_A.png',source/'mask_B.png']
    person_uids=['pair14_actor0','pair14_actor1']
    if not a.skip_source_features:
        source_features(source/'rgb.jpg',masks,a.dino,a.root/'models/Qwen-Image-2.1/vae',a.output,'cuda',person_uids=person_uids)
    generation=dict(source_rgb=str(source/'rgb.jpg'),source_masks=list(map(str,masks)),
        source_features=str(a.output/'source_features.pt'),target_geometry=str(paths[1]),
        source_features_sha256=file_sha256(a.output/'source_features.pt'),
        person_uids=person_uids,prompt='Make the two existing people hug. Preserve their appearance and background.',
        seed=42,resolution=512,versions=generation_versions(),target_geometry_known=True)
    validate_source_feature_cache(torch.load(a.output/'source_features.pt',map_location='cpu',weights_only=True),generation)
    (a.output/'generation.json').write_text(json.dumps(generation,indent=2)+'\n')
    switched=dict(generation,target_geometry=str(paths[0]))
    (a.output/'generation_talk_control.json').write_text(json.dumps(switched,indent=2)+'\n')
    truth=dict(target_rgb=str(target/'rgb.jpg'),target_masks=[str(target/'gt_mask_A.png'),str(target/'gt_mask_B.png')],
        pair='pair14',source_frame=32,target_frame=62,action='hug14',camera=4)
    truth=bind_supervision_manifest(truth,generation)
    (a.output/'supervision.json').write_text(json.dumps(truth,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'generation':generation,'control_switch':'actual fitted talk32 geometry; not hug34'},indent=2),flush=True)


if __name__=='__main__':main()
