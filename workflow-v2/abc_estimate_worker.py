#!/usr/bin/env python3
"""Export source SMPLest-X or full Multi-HMR C geometry in its isolated environment."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from retarget.core import decode_full53,pelvis_translation


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def interaction_forward(model,tensor,camera,threshold):
    # demo.forward_model unconditionally enables fp16 autocast, whose millimetre
    # quantisation cannot satisfy the full-parameter reconstruction contract.
    return model(tensor.float(),K=camera.float(),is_training=False,
        det_thresh=float(threshold),nms_kernel_size=3)


def normalize_interaction_image(image_path,out_dir):
    target=Path(out_dir)/'original.png'
    if target.exists():raise FileExistsError(f'Interaction original already exists: {target}')
    with Image.open(image_path) as image:ImageOps.exif_transpose(image).convert('RGB').save(target)
    return target


def multi_hmr(request):
    import torch
    from retarget.backend import joint_mapping
    root=Path(request['project_root']);settings=request['settings'];out=Path(request['out_dir'])
    image_path=normalize_interaction_image(request['image'],out)
    repo=root/'add/tools/multi-hmr'
    os.environ['MULTIHMR_DINOV2_REPO']=str(root/'add/tools/dinov2_e1277af')
    os.environ['XFORMERS_DISABLED']='1';os.environ['PYOPENGL_PLATFORM']='egl'
    sys.path.insert(0,str(repo));os.chdir(repo)
    from demo_smplx_only import load_model,open_image,get_camera_parameters
    name=settings.get('multihmr_model','multiHMR_896_L')
    if not (repo/'models/multiHMR'/f'{name}.pt').is_file():raise FileNotFoundError(f'Multi-HMR checkpoint {name}')
    torch.set_num_threads(4)
    model=load_model(name,device=torch.device('cuda')).eval();size=int(model.img_size)
    with Image.open(image_path) as image:width,height=image.size
    resized=ImageOps.contain(Image.new('RGB',(width,height)),(size,size)).size
    sx,sy=resized[0]/width,resized[1]/height
    pad=((size-resized[0])//2,(size-resized[1])//2)
    fov=float(settings.get('interaction_fov',60.))
    K_model=get_camera_parameters(size,fov=fov,device=torch.device('cuda'))
    affine=np.array([[sx,0,pad[0]],[0,sy,pad[1]],[0,0,1]],np.float64)
    if settings.get('interaction_K') is not None:
        K_native=np.asarray(settings['interaction_K'],np.float64)
        if K_native.shape!=(3,3) or not np.isfinite(K_native).all():raise ValueError('interaction_K must be finite 3x3')
        K_model=torch.tensor(affine@K_native,dtype=torch.float32,device='cuda')[None]
    else:K_native=np.linalg.inv(affine)@K_model[0].cpu().numpy()
    tensor,_=open_image(str(image_path),size,device=torch.device('cuda'))
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        humans=interaction_forward(model,tensor,K_model,settings.get('interaction_detection_threshold',.2))
    if len(humans)!=2:raise ValueError(f'C must contain exactly two people, Multi-HMR detected {len(humans)}')
    # Contour adds the estimator's final17 guide landmarks. It must be enabled
    # explicitly: the upstream Multi-HMR layer normally exports only127 joints.
    import smplx
    layer=smplx.SMPLX(str(repo/'models/smplx/SMPLX_NEUTRAL.npz'),use_pca=False,
        flat_hand_mean=True,use_face_contour=True,num_betas=10,num_expression_coeffs=10).cpu().eval()
    if model.num_betas!=10:raise ValueError('Only existing adult10-beta checkpoint supported')
    mapping=joint_mapping(settings['smplest_root']);records=[]
    for i,human in enumerate(humans):
        get=lambda k:human[k].detach().float().cpu().numpy()
        params=decode_full53(get('rotvec'),get('shape'),get('expression'))
        no_trans={k:torch.from_numpy(v.reshape(1,-1)) for k,v in params.items()}
        no_trans.update(transl=torch.zeros(1,3),leye_pose=torch.zeros(1,3),reye_pose=torch.zeros(1,3))
        with torch.inference_mode():untranslated=layer(**no_trans)
        rest_pelvis=untranslated.joints[0,0].numpy()
        params['transl']=pelvis_translation(get('transl_pelvis').reshape(3),rest_pelvis)
        kwargs={k:torch.from_numpy(v.reshape(1,-1)) for k,v in params.items()}
        kwargs.update(leye_pose=torch.zeros(1,3),reye_pose=torch.zeros(1,3))
        with torch.inference_mode():reconstructed=layer(**kwargs)
        vertices=reconstructed.vertices[0].numpy();joints=reconstructed.joints[0].numpy()
        mesh_error=float(np.max(np.abs(vertices-get('v3d'))))
        exported=get('j3d');joint_error=float(np.max(np.abs(joints[:len(exported)]-exported)))
        uv=joints@K_model[0].cpu().numpy().T;uv=uv[:,:2]/uv[:,2:]
        projection_error=float(np.max(np.abs(uv[:len(exported)]-get('j2d'))))
        if max(mesh_error,joint_error)>1e-4 or projection_error>.25:
            raise ValueError(f'Multi-HMR reconstruction mismatch mesh={mesh_error} joints={joint_error} projection={projection_error}')
        path=out/f'person_{i}.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,**params,vertices_cam=vertices,
            mesh_cam=vertices,joints_cam=joints,joints_137_cam=joints[list(mapping)],
            faces=layer.faces.astype(np.int32),rotvec=get('rotvec'),K=K_native.astype(np.float32),
            exported_primary_transl=get('transl'),exported_pelvis_cam=get('transl_pelvis').reshape(3),
            leye_pose=np.zeros(3,np.float32),reye_pose=np.zeros(3,np.float32))
        records.append({'index':i,'params_npz':str(path),'pelvis_cam':joints[0].tolist(),
            'mesh_error_m':mesh_error,'joint_error_m':joint_error,'projection_max_error_px':projection_error})
    result={'schema':'abc_interaction_fit_v1','image':str(image_path),
        'original':str(Path(request['image']).resolve()),'people':records,'camera':{'K':K_native.tolist(),
        'K_model':K_model[0].cpu().tolist(),'original_hw':[height,width],'fov_degrees':fov,
        'intrinsics_source':'override_in_exif_normalized_pixels' if settings.get('interaction_K') is not None else f'Multi-HMR FOV{fov:g}',
        'model_resize_hw':[resized[1],resized[0]],'pad_xy':list(pad)},
        'full_pose_layout':'global0/body1:22/left22:37/right37:52/jaw52',
        'hand_mean_convention':'flat_hand_mean=True','model':name,'person_center':'head','inference_dtype':'float32',
        'translation_conversion':'exported_camera_pelvis - standard_SMPLX_untranslated_pelvis'}
    write_json(out/'fit.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text(encoding='utf-8'));out=Path(request['out_dir']);out.mkdir(parents=True,exist_ok=True)
    if request['kind']=='source':
        from retarget.backend import fit_source
        result=fit_source(request['image'],out,request['settings']);write_json(out/'fit.json',result)
    elif request['kind']=='interaction':multi_hmr(request)
    else:raise ValueError('kind must be source or interaction')


if __name__=='__main__':main()
