"""Build a shared-camera SMPL-X pair from separately fitted A/B and full C."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .core import PARAMETER_SIZES,map_roles,mix_parameters,pelvis_translation,place_pair,bounded_contact_shift
from .backend import SourceBodyModel


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def contact_candidates(driver,vertex_parts,threshold=.025,max_candidates=256):
    """C-only proximity with opposing surface normals; no forced nearest contact."""
    from scipy.spatial import cKDTree
    def normals(person):
        vertices=person['vertices_cam'];faces=person['faces']
        tri=vertices[faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        vertex=np.zeros_like(vertices)
        for corner in range(3):np.add.at(vertex,faces[:,corner],n)
        return vertex/np.maximum(np.linalg.norm(vertex,axis=1,keepdims=True),1e-12)
    a,b=[d['vertices_cam'] for d in driver];na,nb=[normals(d) for d in driver]
    distance,index=cKDTree(b).query(a,k=1)
    # Limb/torso evidence excludes feet sharing the ground and coincident heads.
    allowed=np.isin(vertex_parts,[1,2,3,4,5,6,7])
    facing=np.sum(na*nb[index],axis=1)<-.15
    candidate=np.flatnonzero((distance<=threshold)&facing&allowed&allowed[index])
    if len(candidate)>max_candidates:candidate=candidate[np.linspace(0,len(candidate)-1,max_candidates).astype(int)]
    pairs=np.stack([np.zeros(len(candidate),int),candidate,np.ones(len(candidate),int),index[candidate]],1)
    deltas=b[index[candidate]]-a[candidate]
    return pairs.astype(np.int64),deltas.astype(np.float32),{
        'method':'C_vertex_proximity_and_opposing_triangle_normals','threshold_m':threshold,
        'candidate_count':len(candidate),'surface_distance_method':'vertex_correspondence_not_FCL',
        'status':'measured_candidates' if len(candidate) else 'uncertain_no_reliable_candidates'}


def build_scene(request):
    from src.data.canvas_contract import CanvasSpec
    from render.abc_scene import render_scene
    root=Path(request['project_root']);run=Path(request['run_dir']);out=run/'geometry'
    settings=request['settings'];model=SourceBodyModel(settings['smplest_root'])
    sources=json.loads((out/'sources.json').read_text(encoding='utf-8'))['sources']
    for i,source in enumerate(sources):
        fitted=json.loads((out/f'source_{"AB"[i]}'/'fit.json').read_text(encoding='utf-8'))
        source.update(params_npz=fitted['params_npz'],fit_checks=fitted['checks'],model=fitted['model'],
            hand_mean_convention=fitted['hand_mean_convention'])
    interaction=json.loads((out/'interaction/fit.json').read_text(encoding='utf-8'))
    original_drivers=[load_npz(p['params_npz']) for p in interaction['people']]
    ordering=map_roles(np.asarray([p['pelvis_cam'] for p in interaction['people']]),request.get('role_mapping',{'A':'left','B':'right'}))
    drivers=[original_drivers[i] for i in ordering];people=[];mixed_params=[]
    for i in range(2):
        source=load_npz(sources[i]['params_npz']);driver=drivers[i]
        params=mix_parameters(source,driver,model.left_hand_mean,model.right_hand_mean)
        params['transl']=np.zeros(3,np.float32)
        untranslated=model.forward(params)
        params['transl']=pelvis_translation(driver['joints_cam'][0],untranslated['joints_cam'][0])
        person=model.forward(params);people.append(person);mixed_params.append(params)
    pairs,deltas,candidate_diag=contact_candidates(drivers,model.vertex_part,
        threshold=float(settings.get('contact_candidate_threshold_m',.025)),max_candidates=256)
    shifts,contact_diag=bounded_contact_shift([p['vertices_cam'] for p in people],pairs,deltas,
        max_shift=float(settings.get('contact_max_shift_m',.04)))
    for i in range(2):
        mixed_params[i]['transl']+=shifts[i];people[i]=model.forward(mixed_params[i])
    spec=CanvasSpec.from_record(sources[0]['canvas']);source_k=load_npz(sources[0]['params_npz'])['K']
    k=np.asarray(spec.camera(source_k),np.float32)
    _,placement,placement_diag=place_pair(np.stack([p['vertices_cam'] for p in people]),k,spec.canvas_hw,
        margin=float(settings.get('target_margin',.06)))
    targets=[]
    for i in range(2):
        mixed_params[i]['transl']+=placement;people[i]=model.forward(mixed_params[i])
        folder=out/f'target_{"AB"[i]}';folder.mkdir(parents=True,exist_ok=True)
        param_path=folder/'params.npz';mesh_path=folder/'mesh.npz'
        with param_path.open('xb') as f:np.savez_compressed(f,**people[i],K=k,
            leye_pose=np.zeros(3,np.float32),reye_pose=np.zeros(3,np.float32))
        with mesh_path.open('xb') as f:np.savez_compressed(f,vertices_cam=people[i]['vertices_cam'],
            joints_cam=people[i]['joints_cam'],joints_137_cam=people[i]['joints_137_cam'],
            faces=people[i]['faces'],part_vertex=people[i]['part_vertex'],part_face=people[i]['part_face'])
        targets.append({'person_id':'source_'+"AB"[i],'source_index':i,'interaction_index':ordering[i],
            'interaction_role':request.get('role_mapping',{'A':'left','B':'right'})['AB'[i]],
            'params_npz':str(param_path),'mesh_npz':str(mesh_path),
            'parameter_sources':{'betas':'source_'+"AB"[i],'expression':'source_'+"AB"[i],
                'jaw_pose':'source_'+"AB"[i],'global_orient':'C','body_pose':'C','hands':'C'}})
    rendered=render_scene(people,k,spec.canvas_hw,out/'render',contact_indices=pairs)
    scene={'schema':'abc_geometry_v1','sources':sources,'targets':targets,
        'camera':{'K':k.tolist(),'canvas':spec.metadata(),'coordinate':'cv','unit':'m',
            'source':'source_A_estimated_camera_then_CanvasSpec_once'},
        'guides':rendered['guides'],'geometry_arrays':rendered['geometry_arrays'],
        'renders':rendered['renders'],'interaction':interaction,'role_mapping':request.get('role_mapping',{'A':'left','B':'right'}),
        'diagnostics':{'contact_candidates':candidate_diag,'contact_correction':contact_diag,
            'pair_placement':placement_diag,'render':rendered['diagnostics'],
            'preservation':'A/B betas+expression+jaw; C body+global+fullhands',
            'hand_conversion':'C flat_hand_mean=True to source False by subtracting mean',
            'source_fit_checks':[s['fit_checks'] for s in sources],
            'interaction_fit_checks':interaction['people']}}
    (out/'scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return scene
