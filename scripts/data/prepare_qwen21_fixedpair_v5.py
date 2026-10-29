#!/usr/bin/env python3
"""Explicit preparation of the four user-selected V5 pairs; never starts training."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
TROOT=Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1')
HROOT=Path('/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-fixedpair-v5_20261025_v1')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
    return ref(p)
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def producer_snapshot():
    """Bind produced data to immutable producer bytes, not a mutable checkout."""
    p=TROOT/'source/asset_producers'/(sha(__file__)+'.py')
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with p.open('xb') as f:f.write(Path(__file__).read_bytes())
    return ref(p)


def lock():
    cases=[(TROOT,ROOT/'experiments/3.singel man/tiktok_pe_prompt_rewrite_20261014/prompts/train.jsonl',
        ('train_0000247','train_0000212'),'tiktok_pei2i_q4_plan_a_v1','single_2ref'),
        (HROOT,ROOT/'experiments/4.two man /hi4d_pe_prompt_rewrite_20261020/prompts/hi4d_train_prompts_v2.jsonl',
        ('train_pair12_highfive12_cam4_000011_000082','train_pair12_highfive12_cam4_000011_000112'),
        'hi4d_pei2i_q4_plan_a_v2','dual_4ref')]
    for output,bank,ids,schema,profile in cases:
        found={r['pair_id']:r for r in rows(bank)};selected=[]
        for role,key in zip(('train','observation'),ids):
            r=found[key];text=r['compiled_prompt_inference']
            if r['schema']!=schema or hashlib.sha256(text.encode()).hexdigest()!=r['compiled_prompt_inference_sha256']:
                raise ValueError('Existing final Plan A prompt fingerprint differs')
            selected.append({'role':role,'pair_id':key,'bank_record':r,'prompt':text,'prompt_sha256':r['compiled_prompt_inference_sha256']})
        write(output/'data/selection.json',{'schema':'native_v5_fixedpair_selection_v1','profile':profile,
            'bank':ref(bank),'pairs':selected,'updates':300,'save_and_generate_every':30,'seed':42,
            'loss':'velocity_only','canvas_policy':'native_pad','primary_source_index':0,
            'stage_budgets':[60,60,120,60],'continuous_stage_authorization':
            {'source':'direct_user_message','text':'不要停下直接训练300步就行'}})
        (output/'README.md').write_text('# V5 固定配对训练\n\n'
            '目的：1对训练＋1对同场景未训练姿态观察；总预算300个成功update。\n\n'
            'data/selection.json锁定Plan A原文、图片角色与阶段预算；source/保存实施前源码。\n'
            '当前状态：准备中，尚未开始optimizer更新。后续状态写入report与追加日志。\n\n'
            '沿用Native V5，原图native_pad，多图Qwen只以image1提供Adapter外观。\n',encoding='utf8')


def human_model_map():
    path=ROOT/'add/tools/SMPLest-X/human_models/human_models.py'
    tree=ast.parse(path.read_text())
    values=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='joint_idx' for t in node.targets):
            value=ast.literal_eval(node.value)
            if len(value)==137:values.append(tuple(value))
    if len(values)!=1:raise ValueError('Require the actual original SMPLest-X137 joint map')
    return values[0],ref(path)


def appearance_models():
    dino=ROOT/'models/antelopev2/dinov2-giant'
    files={str(p.relative_to(dino)):sha(p) for p in sorted(dino.rglob('*')) if p.is_file() and
        (p.suffix in ('.safetensors','.bin','.json'))}
    if not any(n.endswith(('.safetensors','.bin')) for n in files):raise ValueError('DINO weights absent')
    return {'arcface_model':str(ROOT/'models/antelopev2/glintr100.onnx'),
        'arcface_sha256':sha(ROOT/'models/antelopev2/glintr100.onnx'),
        'dino_model':str(dino),'dino_files':files,'device':'cuda'}


def tiktok_import():
    import numpy as np
    import torch
    import smplx
    from PIL import Image
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.native_full_assets import import_smplx,write_record,local_geometry
    from src.pose_control.qwen21_full.dual_hi4d_geometry import SMPL_TO_PART,IndependentMeshRasterizer
    from src.pose_control.qwen21_full.conditions import PART_NAMES
    from src.pose_edit.geometry import smplx137_to_dwpose134,render_poseguide
    from smplx.joint_names import JOINT_NAMES
    torch.set_num_threads(4)
    out=TROOT/'data/smplx_import_v3';out.mkdir(parents=True,exist_ok=False)
    original=ROOT/'datasets/TikTokDataset/TikTok_3d_assets_native/00031/smplestx/params.pt'
    params=torch.load(original,map_location='cpu',weights_only=False)
    mapping=json.loads(original.with_name('frame_map.json').read_text())
    indices={x['stem']:x['index'] for x in mapping['frames']}
    model_file=ROOT/'add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz'
    options=dict(use_pca=False,num_pca_comps=6,flat_hand_mean=False,num_expression_coeffs=10,gender='neutral',use_face_contour=True)
    model=smplx.SMPLX(str(model_file),batch_size=1,**options).float().eval().requires_grad_(False)
    faces=model.faces.astype(np.int64);lbs=model.lbs_weights.numpy()
    part_map=np.array([*SMPL_TO_PART[:22],0,0,0,*([4]*15),*([7]*15)],np.int64)
    if len(part_map)!=lbs.shape[1]:raise ValueError('Original SMPL-X55 skinning map differs')
    part_weights=np.stack([lbs[:,part_map==i].sum(1) for i in range(14)],1)
    vp=part_weights.argmax(1);fp=part_weights[faces].sum(1).argmax(1)
    segment=out/'segmentation.npz'
    with segment.open('xb') as f:np.savez_compressed(f,face_part14=fp,
        left_hand_vertices=np.flatnonzero(vp==4),right_hand_vertices=np.flatnonzero(vp==7))
    joint_map,map_ref=human_model_map();joint_map=np.asarray(joint_map)
    hand_ids=[]
    for side in ('left','right'):
        names=[side+'_wrist']
        for finger in ('thumb','index','middle','ring','pinky'):
            names += [side+'_'+finger+str(i) for i in (1,2,3)]+[side+'_'+finger]
        hand_ids.append([JOINT_NAMES.index(n) for n in names])
    basis_sha=hashlib.sha256(model.expr_dirs.numpy().astype('<f4').tobytes()).hexdigest()
    raster=IndependentMeshRasterizer('cuda');records={};source=None
    for stem in ('0265','0374','0126'):
        i=indices[stem];rgb=ROOT/f'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset/00031/images/{stem}.png'
        with Image.open(rgb) as im:hw=(im.height,im.width)
        k=np.eye(3,dtype=np.float32);k[0,0],k[1,1]=params['focal_length_xy'][i].tolist();k[:2,2]=params['princpt'][i].numpy()
        names=('betas','global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','expression','transl')
        raw={n:params[n][i:i+1].numpy().astype(np.float32) for n in names}
        raw.update(leye_pose=np.zeros((1,3),np.float32),reye_pose=np.zeros((1,3),np.float32),
            mesh_cam=params['mesh_cam'][i:i+1].numpy(),K=k,camera_R=np.eye(3,dtype=np.float32),camera_t=np.zeros(3,np.float32))
        parameter_file=out/(stem+'_parameters.npz')
        with parameter_file.open('xb') as f:np.savez_compressed(f,**raw)
        job={'parameters':str(parameter_file),'people':[{'model_options':options,'model_file':str(model_file),
            'model_sha256':sha(model_file),'person_id':'tiktok:00031:0','segmentation':str(segment),
            'expression_basis_sha256':basis_sha,'part_names':list(PART_NAMES),'hand_joint_indices':hand_ids}],
            'original_hw':list(hw),'camera_id':'tiktok:00031:camera','scale_group_id':'tiktok:00031:model',
            'provenance':{'rgb_sha256':sha(rgb),'producer':producer_snapshot(),'original_parameters':ref(original),
                'frame_map':ref(original.with_name('frame_map.json')),'original_param_index':i,
                'eye_pose_source':'SMPLest-X original model default zero; mesh reconstruction verifies this',
                'joint_map_source':map_ref}}
        asset=import_smplx(job)
        with torch.no_grad():posed=model(**{n:torch.from_numpy(raw[n]) for n in (*names,'leye_pose','reye_pose')})
        j137=posed.joints[0,joint_map].numpy();uv=j137@k.T;uv=uv[:,:2]/uv[:,2:]
        # joint_proj_model is the estimator's 12x16 crop heatmap, not original
        # camera pixels (see src/pose_edit/estimator.py). Validate camera-space
        # joints against the independently verified Body21 record instead.
        verified_fk=torch.load(TROOT/f'data/body21_fk_v1/tiktok__00031__target__{stem}.pt',weights_only=True)
        if not np.allclose(posed.joints[0,:22].numpy(),verified_fk['joints_cam'].numpy(),atol=1e-4,rtol=0):
            raise ValueError('SMPL-X joint/camera reconstruction differs from verified FK')
        asset['metadata']['legacy_joint_proj_model_space']='SMPLest-X crop heatmap; never treated as original pixels'
        # Canonical Face72: four SMPL-X extras, then DWPose68 (contour first).
        face_order=[65,66,67,68,*range(120,137),*range(69,120)]
        person=asset['people'][0];person['face72_cam']=j137[face_order].copy();person['face72_valid']=np.ones(72,bool)
        person['metadata'].update(face72_layout='smplx137_65_136_first4_extras_then_dw68',
            face72_joint_indices_137=face_order,face72_source=map_ref)
        saved=write_record(out/(stem+'_asset.npz'),asset,schema='native_smplx_asset_v4')
        canvas=native_pad(hw);points=smplx137_to_dwpose134(uv);points=canvas.pixel_points(points)
        guide=out/(stem+'_3d_dwpose.png');render_poseguide(points,(canvas.canvas_hw[1],canvas.canvas_hw[0])).save(guide)
        records[stem]={'asset':saved,'rgb':ref(rgb),'canvas':canvas.metadata(),'guide':ref(guide),
            'mesh_error':person['metadata']['fk_mesh_max_error']}
        if stem=='0265':source=asset
        print(json.dumps({'stem':stem,'fk_mesh_error':records[stem]['mesh_error'],'guide':str(guide)}),flush=True)
    source_spec=native_pad(source['metadata']['original_hw']);loc=local_geometry(source,source_spec,renderer=raster)
    surf=loc['surfaces'][0];boxes=loc['boxes'][0];tol=1e-4*source['people'][0]['body_height']
    visible=[r['valid']&surf['valid']&(np.abs(r['z']-surf['z'])<=tol) for r in loc['regions'][0]]
    face=loc['face_points'][0][0][4:];fv=loc['face_points'][0][1][4:]
    groups=(list(range(36,42)),list(range(42,48)),[30],[48],[54]);five_ok=all(fv[g].all() for g in groups)
    person={'face_box':boxes[0].tolist(),'hand_boxes':boxes[1:].tolist(),
        'identity_valid':bool(five_ok and visible[0].any()),'face_texture_valid':bool(visible[0].any()),
        'hand_texture_valid':[bool(v.any()) for v in visible[1:]]}
    if five_ok:person['arcface_five_points_canvas_pixels']=(np.stack([face[g,:2].mean(0) for g in groups])*
        np.array([source_spec.canvas_hw[1],source_spec.canvas_hw[0]])).tolist()
    job={**appearance_models(),'role':'source_only','modalities':{'arcface':True,'face_dino':True,'hand_dino':True},
        'source_person_ids':['tiktok:00031:0','__inactive__'],'references':[{'rgb':records['0265']['rgb'],
            'canvas':source_spec.metadata(),'person_ids':['tiktok:00031:0','__inactive__'],
            'geometry_sha256':records['0265']['asset']['sha256'],'people':[person]}]}
    write(out/'source_references_job.json',job);write(out/'index.json',{'records':records,'expression_basis_sha256':basis_sha})


def hi4d_geometry():
    import torch
    import smplx
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
    # The original release pickles contain literal Chumpy shapedirs arrays.
    # Reuse the repository's audited data-only reader without importing Pro's
    # package/Adapter, installing Chumpy, or changing any model tensor.
    loader_path=ROOT/'src/pose_control/qwen21_pro/geometry.py'
    loader_spec=importlib.util.spec_from_file_location('fixedpair_literal_smpl',loader_path)
    loader=importlib.util.module_from_spec(loader_spec);sys.modules[loader_spec.name]=loader
    loader_spec.loader.exec_module(loader)
    original_smpl=smplx.SMPL
    def literal_smpl(path,**kwargs):
        return original_smpl(path,data_struct=smplx.utils.Struct(**loader.load_literal_smpl_model(path)),**kwargs)
    smplx.SMPL=literal_smpl
    spec=importlib.util.spec_from_file_location('native_exporter',ROOT/'scripts/data/build_qwen21_native_dual_cache.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(4)
    dest=HROOT/'data/geometry_v3_v3';dest.mkdir(parents=True,exist_ok=False)
    base=ROOT/'datasets/Hi4D_pilot_v1/raw/pair12/highfive12';raster=IndependentMeshRasterizer();records={}
    for frame in (11,82,112):
        raw=dict(pair='pair12',action='highfive12',frame_id=frame,camera_id=4,
            official_smpl=str(base/f'smpl/{frame:06d}.npz'),meta=str(base/'meta.npz'),
            camera_file=str(base/'cameras/rgb_cameras.npz'),rgb=str(base/f'images/4/{frame:06d}.jpg'),
            person_mask_A=str(base/f'seg/img_seg_mask/4/0/{frame:06d}.png'),
            person_mask_B=str(base/f'seg/img_seg_mask/4/1/{frame:06d}.png'),
            additional_sources={'literal_model_loader':ref(loader_path),'preparation_tool':producer_snapshot()})
        item=module.export_frame(raw,dest,ROOT/'add/tools/SMPLest-X/human_models/human_model_files/smpl',raster,1280)
        records[str(frame)]={'geometry':{'path':item['path'],'sha256':item['sha256']},'raw':raw,'metadata':item['metadata']}
        print(json.dumps({'frame':frame,'mesh_errors':item['metadata']['mesh_errors'],'camera_errors':item['metadata']['camera_mesh_errors']}),flush=True)
    selected=json.loads((HROOT/'data/selection.json').read_text());output=[]
    for chosen,frame in zip(selected['pairs'],(82,112)):
        src=records['11'];dst=records[str(frame)];sm=src['metadata'];tm=dst['metadata']
        output.append({'schema':'native_dual_pair_v3','pair_id':chosen['pair_id'],
            'source_geometry':src['geometry'],'target_geometry':dst['geometry'],
            'source_rgb':sm['rgb'],'target_rgb':tm['rgb'],
            'source_masks':[ref(src['raw']['person_mask_'+c]) for c in ('A','B')],
            'source_person_ids':sm['person_ids'],'target_person_ids':tm['person_ids'],'source_indices':[0,1],
            'kind':'camera_z','pairing':'same_subject_pair_target_frame',
            'source_canvas':native_pad(sm['original_hw']).metadata(),'target_canvas':native_pad(tm['original_hw']).metadata()})
    with (dest/'pairs.jsonl').open('x') as f:
        for r in output:f.write(json.dumps(r)+'\n')
    write(dest/'index.json',records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('lock','tiktok-import','hi4d-geometry'))
    args=parser.parse_args()
    {'lock':lock,'tiktok-import':tiktok_import,'hi4d-geometry':hi4d_geometry}[args.action]()


if __name__=='__main__':main()
