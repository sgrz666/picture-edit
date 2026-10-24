"""Offline V4 asset tools. Importing this module never loads SMPL-X/ArcFace/DINO.

Source/driver canvases and complete model conventions are explicit job inputs.
The executable tools are intentionally separate from runtime dataset loading.
"""
from dataclasses import fields,is_dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def write_record(path,value,*,schema):
    """NPZ + JSON tree, no pickle and no learned features in geometry caches."""
    arrays={}
    def pack(v):
        if isinstance(v,np.generic): return pack(v.item())
        if torch.is_tensor(v) or isinstance(v,np.ndarray):
            if torch.is_tensor(v) and v.requires_grad: raise ValueError('Never cache/detach learnable conditions offline')
            a=v.detach().cpu().numpy() if torch.is_tensor(v) else v
            if a.dtype.hasobject: raise ValueError('Object arrays are forbidden')
            key='array_'+str(len(arrays));arrays[key]=a
            return {'array':key,'tensor':torch.is_tensor(v)}
        if is_dataclass(v):
            if type(v).__name__ not in ('NativeLocalConditionV4','NativeLocalReferencesV4','DepthConditionV2','DepthFusionContext',
                    'DualDepthConditionV3','DualContactRelations','ContactRelationBatch'):
                raise ValueError('Only declared raw condition records may be cached')
            return {'record':type(v).__name__,'fields':{f.name:pack(getattr(v,f.name)) for f in fields(v)}}
        if isinstance(v,dict): return {'dict':{k:pack(x) for k,x in v.items()}}
        if isinstance(v,(list,tuple)): return {'tuple' if isinstance(v,tuple) else 'list':[pack(x) for x in v]}
        if v is None or type(v) in (bool,int,float,str): return v
        raise ValueError('Undeclared cache value '+str(type(v)))
    tree=pack(value)
    arrays['record_json']=np.asarray(json.dumps({'schema':schema,'tree':tree},sort_keys=True))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream: np.savez_compressed(stream,**arrays)
    return {'path':str(path.resolve()),'sha256':sha256(path),'schema':schema}


def read_record(record,*,schema):
    from .native_local_condition import NativeLocalConditionV4,NativeLocalReferencesV4
    from .native_full_runtime import NativeFullIdentityV4
    from .depth_condition_v2 import DepthConditionV2,DepthFusionContext
    from .dual_depth_condition import DualDepthConditionV3,DualContactRelations
    from .conditions import ContactRelationBatch
    registry={c.__name__:c for c in (NativeLocalConditionV4,NativeLocalReferencesV4,NativeFullIdentityV4,
        DepthConditionV2,DepthFusionContext,DualDepthConditionV3,DualContactRelations,ContactRelationBatch)}
    if record.get('schema')!=schema or sha256(record['path'])!=record['sha256']: raise ValueError('Cache schema/SHA differs')
    with np.load(record['path'],allow_pickle=False) as archive:
        header=json.loads(str(archive['record_json']))
        if header['schema']!=schema: raise ValueError('Cache header differs')
        def unpack(v):
            if not isinstance(v,dict): return v
            if 'array' in v:
                a=archive[v['array']].copy()
                return torch.from_numpy(a) if v['tensor'] else a
            if 'record' in v:
                if v['record'] not in registry: raise ValueError('Unknown typed cache object')
                return registry[v['record']](**{k:unpack(x) for k,x in v['fields'].items()})
            if 'dict' in v: return {k:unpack(x) for k,x in v['dict'].items()}
            if 'tuple' in v: return tuple(unpack(x) for x in v['tuple'])
            if 'list' in v: return [unpack(x) for x in v['list']]
            raise ValueError('Unknown record tree')
        return unpack(header['tree'])


def import_smplx(job):
    """Reconstruct with ORIGINAL model settings; accept only matching whole meshes.

    No dimensional guess/PCA padding is allowed. Optional face attachments and
    Hand21 joint indices are declared by the asset producer, never guessed from
    the 137-point detector layout.
    """
    import smplx
    source=Path(job['parameters']);source_sha=sha256(source)
    with np.load(source,allow_pickle=False) as data: params={k:data[k].copy() for k in data.files}
    if any(not np.isfinite(v).all() for v in params.values() if np.issubdtype(v.dtype,np.number)):
        raise ValueError('SMPL-X input contains nonfinite parameters/geometry')
    required=('betas','global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose',
              'leye_pose','reye_pose','expression','transl','mesh_cam','K','camera_R','camera_t')
    if set(required)-params.keys(): raise ValueError('Missing complete SMPL-X parameters: '+str(set(required)-params.keys()))
    count=len(job['people'])
    if count not in (1,2) or params['body_pose'].shape!=(count,63): raise ValueError('SMPL-X body pose must be63; SMPL69 is not a substitute')
    if params['betas'].shape!=(count,10): raise ValueError('Current Global26 requires exactly10 declared beta coefficients')
    people=[];k=params['K'].astype(np.float32)
    if k.shape!=(3,3) or not np.allclose(k[2],[0,0,1]) or k[0,0]<=0 or k[1,1]<=0:
        raise ValueError('Original pinhole K required')
    expression_source=None
    if job.get('expression_source'):
        expression_source=read_record(job['expression_source'],schema='native_smplx_asset_v4')
    camera_r=params['camera_R'].astype(np.float32);camera_t=params['camera_t'].astype(np.float32).reshape(3)
    if camera_r.shape!=(3,3) or not np.allclose(camera_r.T@camera_r,np.eye(3),atol=1e-5) or np.linalg.det(camera_r)<.999:
        raise ValueError('Camera_R must be a declared proper CV rotation')
    for index,spec in enumerate(job['people']):
        options=spec['model_options']
        for key in ('use_pca','num_pca_comps','flat_hand_mean','num_expression_coeffs','gender'):
            if key not in options: raise ValueError('Missing original SMPL-X model option '+key)
        model_file=Path(spec['model_file'])
        if sha256(model_file)!=spec['model_sha256']: raise ValueError('SMPL-X model fingerprint differs')
        model=smplx.SMPLX(str(model_file),batch_size=1,**options).float().eval()
        from .body_pose import PARENTS_22
        if tuple(model.parents[:22].tolist())!=PARENTS_22: raise ValueError('SMPL-X Body21 named tree differs')
        basis_sha=hashlib.sha256(model.expr_dirs.cpu().numpy().astype('<f4').tobytes()).hexdigest()
        if basis_sha!=spec['expression_basis_sha256']: raise ValueError('Expression basis tensor SHA differs from declared original model')
        values={name:torch.from_numpy(params[name][index:index+1]).float() for name in required[:10]}
        if values['expression'].shape[-1]!=options['num_expression_coeffs']: raise ValueError('Expression basis differs; no truncation')
        handdim=options['num_pca_comps'] if options['use_pca'] else 45
        if any(values[name].shape!=(1,handdim) for name in ('left_hand_pose','right_hand_pose')): raise ValueError('Original PCA/axis-angle hand shape differs')
        with torch.no_grad():
            out=model(**values,return_full_pose=True)
            mesh=out.vertices[0].cpu().numpy()@camera_r.T+camera_t
            joints=out.joints[0].cpu().numpy()@camera_r.T+camera_t
            full=out.full_pose[0].cpu().numpy()
            if full.shape!=(165,): raise ValueError('Expected complete55-joint SMPL-X pose')
            error=float(np.abs(mesh-params['mesh_cam'][index]).max())
            if error>1e-4: raise ValueError('Whole-mesh FK mismatch: '+str(error))
            redirected=False;expression_changed=False
            if expression_source is not None:
                match=[p for p in expression_source['people'] if p['metadata']['person_id']==spec['person_id']]
                if len(match)!=1 or match[0]['metadata']['expression_basis_sha256']!=spec['expression_basis_sha256']:
                    raise ValueError('Source expression person/basis mismatch; no coefficient truncation')
                source_person=match[0]
                source_expression=torch.from_numpy(source_person['expression_full']).float()[None]
                source_jaw=torch.from_numpy(source_person['jaw_pose']).float()[None]
                if source_expression.shape!=values['expression'].shape: raise ValueError('Source expression basis dimension differs')
                expression_changed=not (torch.allclose(source_expression,values['expression'],atol=1e-7,rtol=0) and torch.allclose(source_jaw,values['jaw_pose'],atol=1e-7,rtol=0))
                values['expression']=source_expression;values['jaw_pose']=source_jaw
                out=model(**values,return_full_pose=True)
                mesh=out.vertices[0].cpu().numpy()@camera_r.T+camera_t
                joints=out.joints[0].cpu().numpy()@camera_r.T+camera_t
                full=out.full_pose[0].cpu().numpy();redirected=True
            rest={name:torch.zeros_like(value) for name,value in values.items()};rest['betas']=values['betas']
            reference=model(**rest,return_full_pose=True)
            rest_mesh=reference.vertices[0].cpu().numpy();rest_joints=reference.joints[0].cpu().numpy()
            height=float(np.ptp(rest_mesh[:,1]))
            # The rotation pivot belongs to CURRENT beta + expression. Standard
            # height uses neutral expression, but must not replace this pivot.
            pelvis_rest=out.joints[0,0].cpu().numpy()-values['transl'][0].numpy()
        with np.load(spec['segmentation'],allow_pickle=False) as data: segmentation={key:data[key].copy() for key in data.files}
        faces=np.asarray(model.faces,dtype=np.int64)
        face_part=segmentation['face_part14']
        if face_part.shape!=(len(faces),) or not np.isin(face_part,np.arange(14)).all(): raise ValueError('Same-mesh Part14 triangle partition required')
        person={'mesh_cam':mesh.astype(np.float32),'joints_cam':joints[:22].astype(np.float32),'faces':faces,
            'body_pose':full[3:66].copy(),'jaw_pose':full[66:69].copy(),'expression_full':values['expression'][0].numpy().copy(),
            'hand_pose':np.stack((full[75:120],full[120:165])).copy(),'body_height':height,'face_part':face_part,
            'face72_cam':np.zeros((72,3),np.float32),'face72_valid':np.zeros(72,bool),
            'hand21_cam':np.zeros((2,21,3),np.float32),'hand21_valid':np.zeros((2,21),bool),
            'wrist_cam':joints[[20,21]].astype(np.float32),'wrist_rotation':[], 'hand_vertices':[], 'standard_hand_length':[]}
        # Actual wrist frames from the complete FK tree, including camera root.
        from .body_pose import axis_angle_to_matrix
        rotations=axis_angle_to_matrix(torch.from_numpy(full.reshape(55,3))).numpy()
        from .body_pose import matrix_to_rotation_6d
        root=camera_r@rotations[0]
        transl=camera_r@values['transl'][0].numpy()+camera_t+(camera_r-np.eye(3))@pelvis_rest
        oh,ow=job['original_hw']
        person['root_rotation']=root.astype(np.float32)
        person['global26']=np.concatenate((params['betas'][index,:10],matrix_to_rotation_6d(torch.from_numpy(root)).numpy(),
            transl,np.array([k[0,0]/ow,k[1,1]/oh,k[0,2]/ow,k[1,2]/oh]),transl)).astype(np.float32)
        world=[]
        for j,parent in enumerate(model.parents.tolist()): world.append(rotations[j] if parent<0 else world[parent]@rotations[j])
        for side,wrist in enumerate((20,21)):
            ids=segmentation['left_hand_vertices' if side==0 else 'right_hand_vertices'].astype(np.int64)
            if ids.ndim!=1 or not len(ids) or ids.min()<0 or ids.max()>=len(mesh): raise ValueError('Invalid same-model hand segmentation')
            person['hand_vertices'].append(ids)
            person['wrist_rotation'].append(camera_r@world[wrist])
            length=float(np.linalg.norm(rest_mesh[ids]-rest_joints[wrist],axis=-1).max())
            if length<=0: raise ValueError('Invalid standard same-beta hand length')
            person['standard_hand_length'].append(length)
        attachments=spec.get('face_attachments')
        if attachments:
            if spec.get('face72_layout')!='smplx137_65_136_first4_extras_then_dw68': raise ValueError('Explicit canonical Face72 attachment order required')
            with np.load(attachments,allow_pickle=False) as data:
                triangle=data['triangle'].astype(np.int64);bary=data['barycentric'].astype(np.float32)
            if triangle.shape!=(72,) or bary.shape!=(72,3) or not np.isfinite(bary).all() or (bary<0).any() or not np.allclose(bary.sum(-1),1,atol=1e-5) or (triangle<0).any() or (triangle>=len(faces)).any():
                raise ValueError('Explicit72 face barycentric attachments required')
            person['face72_cam']=(mesh[faces[triangle]]*bary[...,None]).sum(1).astype(np.float32);person['face72_valid'][:]=True
        if 'hand_joint_indices' in spec:
            from smplx.joint_names import JOINT_NAMES as MODEL_JOINT_NAMES
            ids=np.asarray(spec['hand_joint_indices'],dtype=np.int64)
            if ids.shape!=(2,21) or ids.min()<0 or ids.max()>=len(joints): raise ValueError('Declared Hand21 joint map differs')
            for side,side_name in enumerate(('left','right')):
                expected=[side_name+'_wrist']
                for finger in ('thumb','index','middle','ring','pinky'):
                    expected.extend([side_name+'_'+finger+str(i) for i in (1,2,3)]+[side_name+'_'+finger])
                names=[MODEL_JOINT_NAMES[int(j)] for j in ids[side]]
                if names!=expected: raise ValueError('Hand21 indices must be anatomical wrist/thumb/index/middle/ring/pinky order')
            person['hand21_cam']=joints[ids].astype(np.float32);person['hand21_valid'][:]=True
        from .conditions import PART_NAMES
        if tuple(spec['part_names'])!=PART_NAMES: raise ValueError('Explicit canonical Part14 ordering required')
        person['metadata']={'person_id':spec['person_id'],'model_family':'smplx','gender':options['gender'],
            'model_sha256':spec['model_sha256'],'model_options':options,'parameter_sha256':source_sha,'fk_mesh_max_error':error,
            'segmentation_sha256':sha256(spec['segmentation']),'hand_pose_convention':'full_axis_angle_after_pose_mean',
            'expression_basis_sha256':spec['expression_basis_sha256'],'expression_count':options['num_expression_coeffs'],
            'face_attachments_sha256':sha256(attachments) if attachments else None,
            'face72_layout':spec.get('face72_layout'),
            'expression_retargeted':redirected,'counterfactual_geometry':expression_changed,
            'hand_joint_indices':spec.get('hand_joint_indices'),'part_names':list(PART_NAMES),
            'raw_parameters':{n:params[n][index].tolist() for n in required[:10]}}
        people.append(person)
    calibration=None
    if job.get('calibration'):
        record=job['calibration']
        if sha256(record['path'])!=record['sha256']: raise ValueError('Calibration asset SHA differs')
        calibration=json.loads(Path(record['path']).read_text(encoding='utf8'))
        factor=calibration.get('metres_per_model_unit',0.)
        if calibration.get('parameter_sha256')!=source_sha or not np.isfinite(factor) or factor<=0:
            raise ValueError('Metre calibration must bind the original parameter asset and a positive scale')
        for person in people:
            for name in ('mesh_cam','joints_cam','wrist_cam','face72_cam','hand21_cam'): person[name]=person[name]*factor
            person['body_height']*=factor;person['standard_hand_length']=[v*factor for v in person['standard_hand_length']]
            person['global26'][16:19]*=factor;person['global26'][23:26]*=factor
        calibration={**calibration,'asset_sha256':record['sha256']}
    return {'people':people,'K':k,'metadata':{**job['provenance'],'schema':'native_smplx_asset_v4',
        'units':'model_unit' if calibration is None else 'metre','scale_status':'model_scale' if calibration is None else 'calibrated',
        'calibration':calibration,
        'expression_source_sha256':None if expression_source is None else job['expression_source']['sha256'],
        'counterfactual_geometry':any(p['metadata']['counterfactual_geometry'] for p in people),
        'original_hw':job['original_hw'],'normal_space':'camera_cv','camera_id':job['camera_id'],
        'scale_group_id':job['scale_group_id'],'parameter_sha256':source_sha,'camera_R':camera_r.tolist(),
        'camera_t':(camera_t if calibration is None else camera_t*calibration['metres_per_model_unit']).tolist(),
        'camera_t_raw_model_unit':camera_t.tolist()}}


def bps1024(vertices,wrist,rotation,hand_length,*,seed=20261024,basis=None):
    if basis is None:
        rng=np.random.default_rng(seed);direction=rng.normal(size=(1024,3))
        direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        basis=(direction*rng.random((1024,1))**(1/3)).astype(np.float32)
    basis=np.asarray(basis,dtype=np.float32)
    if basis.shape!=(1024,3) or hand_length<=0 or not np.isfinite(basis).all(): raise ValueError('Invalid BPS basis/hand scale')
    local=(vertices-wrist)@rotation/hand_length
    distances=np.full(1024,np.inf,np.float32)
    for start in range(0,len(local),2048):
        distances=np.minimum(distances,np.linalg.norm(basis[:,None]-local[None,start:start+2048],axis=-1).min(1))
    digest=hashlib.sha256(basis.astype('<f4').tobytes()).hexdigest()
    return distances,{'basis':basis,'seed':seed,'basis_sha256':digest,'coordinate':'wrist_local_cv_fk',
        'normalization':'same_beta_standard_wrist_to_farthest_hand_vertex','encoding':'nearest_vertex_euclidean',
        'sides':['anatomical_left','anatomical_right'],'mirror':'none; separate_anatomical_wrist_frames'}


def local_geometry(asset,canvas,*,renderer,seed=20261024,basis=None):
    """Render independent local meshes, retain occluded surfaces and unclipped xy."""
    people=asset['people'];h,w=canvas.canvas_hw
    maps=[];surfaces=[];bps=[];boxes=[];face=[];hand=[];bps_meta=[]
    k=canvas.camera(asset['K']).astype(np.float32)
    def project(points,valid):
        uv=points@k.T;pv=valid&(points[:,2]>0)
        xy=uv[:,:2]/np.maximum(uv[:,2:],1e-8)
        return np.concatenate((xy/np.array([w,h]),pv[:,None].astype(np.float32)),1).astype(np.float32),pv
    for person in people:
        surfaces.append(renderer.render(person,k,(h,w)))
        f,pv=project(person['face72_cam'],person['face72_valid']);face.append((f,pv))
        hp=[project(points,valid) for points,valid in zip(person['hand21_cam'],person['hand21_valid'])];hand.append(hp)
        localmaps=[];localboxes=[];values=[];conventions=[]
        for region in range(3):
            from .conditions import PART_NAMES
            selected=person['face_part']==PART_NAMES.index(('head','left_hand','right_hand')[region])
            if not selected.any():
                localmaps.append({'z':np.zeros((h,w),np.float32),'valid':np.zeros((h,w),bool)})
                localboxes.append(np.zeros(4,np.float32));continue
            mesh={**person,'faces':person['faces'][selected],'face_part':person['face_part'][selected]}
            localmaps.append(renderer.render(mesh,k,(h,w)))
            vertices=person['mesh_cam'][np.unique(mesh['faces'])]
            projected,_=project(vertices,np.ones(len(vertices),bool))
            low=projected[:,:2].min(0);high=projected[:,:2].max(0);pad=(high-low)*.1
            localboxes.append(np.concatenate((low-pad,high+pad)).astype(np.float32))
        for side in range(2):
            value,convention=bps1024(person['mesh_cam'][person['hand_vertices'][side]],person['wrist_cam'][side],
                person['wrist_rotation'][side],person['standard_hand_length'][side],seed=seed,basis=basis)
            values.append(value);conventions.append(convention)
        maps.append(localmaps);boxes.append(localboxes);bps.append(values);bps_meta.append(conventions)
    return {'surfaces':surfaces,'regions':maps,'boxes':np.asarray(boxes),'face_points':face,'hand_points':hand,
            'bps':np.asarray(bps,dtype=np.float32),'bps_metadata':bps_meta,'canvas':canvas.metadata(),'canvas_id':canvas.fingerprint()}
