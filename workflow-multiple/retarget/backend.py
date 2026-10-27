"""Validated SMPLest-X model, source fitting, and shape-aware retarget meshes."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np

from .core import PARAMETER_SIZES


def joint_mapping(smplest_root):
    path=Path(smplest_root)/'human_models/human_models.py'
    result=[]
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='joint_idx' for t in node.targets):
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError):continue
            if isinstance(value,(list,tuple)) and len(value)==137:result.append(tuple(value))
    if len(result)!=1:raise ValueError('Expected exactly one authoritative SMPLest-X137 joint map')
    return result[0]


class SourceBodyModel:
    def __init__(self,smplest_root):
        import torch
        from src.pose_control.qwen21_pro.geometry import load_smplestx_backend
        self.torch=torch
        backend=load_smplestx_backend(smplest_root)
        self.layer=backend.layer
        self.joint_indices=backend.joint_indices
        self.faces=backend.faces.numpy().astype(np.int32)
        self.vertex_part=backend.vertex_parts.numpy().astype(np.int64)
        # Majority triangle labels use actual SMPL-X topology and LBS partitions.
        labels=self.vertex_part[self.faces]
        self.face_part=np.stack([(labels==i).sum(1) for i in range(14)],1).argmax(1)
        self.provenance=backend.provenance
        self.left_hand_mean=self.layer.left_hand_mean.detach().cpu().numpy().copy()
        self.right_hand_mean=self.layer.right_hand_mean.detach().cpu().numpy().copy()

    def forward(self,params):
        torch=self.torch
        kwargs={k:torch.as_tensor(np.asarray(params[k]).reshape(1,n),dtype=torch.float32)
            for k,n in PARAMETER_SIZES.items()}
        kwargs.update(leye_pose=torch.zeros(1,3),reye_pose=torch.zeros(1,3))
        with torch.inference_mode():out=self.layer(**kwargs)
        raw=out.joints[0].numpy().copy()
        vertices=out.vertices[0].numpy().copy()
        if not np.isfinite(vertices).all() or not np.isfinite(raw).all():
            raise ValueError('SMPL-X produced non-finite geometry')
        return {'vertices_cam':vertices,'mesh_cam':vertices,'joints_cam':raw,
            'joints_137_cam':raw[list(self.joint_indices)],'faces':self.faces,
            'part_vertex':self.vertex_part,'part_face':self.face_part,
            **{k:np.asarray(params[k],np.float32).reshape(n) for k,n in PARAMETER_SIZES.items()}}

    def rest_pelvis(self,betas):
        params={k:np.zeros(n,np.float32) for k,n in PARAMETER_SIZES.items()}
        params['betas']=np.asarray(betas,np.float32)
        return self.forward(params)['joints_cam'][0]


def fit_source(image,out_dir,settings):
    import torch
    from src.pose_edit.estimator import estimate_single,reconstruct,project
    root=Path(settings['smplest_root']);out_dir=Path(out_dir)
    params,staging=estimate_single(str(image),out_dir,settings,root)
    mapped,error=reconstruct(params,root)
    _,camera=project(params,mapped)
    model=SourceBodyModel(root)
    values={k:params[k][0].detach().cpu().numpy().astype(np.float32) for k in PARAMETER_SIZES}
    person=model.forward(values)
    recheck=float(np.max(np.abs(person['vertices_cam']-params['mesh_cam'][0].numpy())))
    if recheck>1e-4:raise ValueError(f'Source common-model mesh mismatch {recheck}m')
    K=np.eye(3,dtype=np.float32);K[0,0],K[1,1]=camera['focal_xy'];K[:2,2]=camera['principal_xy']
    person.update(K=K,joint_proj_model=params['joint_proj_model'][0].numpy(),
        focal_length_xy=np.asarray(camera['focal_xy']),princpt=np.asarray(camera['principal_xy']),
        leye_pose=np.zeros(3,np.float32),reye_pose=np.zeros(3,np.float32))
    target=out_dir/'params.npz'
    with target.open('xb') as f:np.savez_compressed(f,**person)
    return {'params_npz':str(target),'camera':K.tolist(),'checks':{
        'mesh_max_error_m':error,'common_model_mesh_error_m':recheck,**camera},'estimator':staging,
        'model':model.provenance,'hand_mean_convention':'flat_hand_mean=False'}
