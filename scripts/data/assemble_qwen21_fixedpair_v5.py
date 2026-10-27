#!/usr/bin/env python3
"""Assemble selected V4/V5 records and exact multi-reference declarations."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def helpers():
    spec=importlib.util.spec_from_file_location('fixedpair_preparation',ROOT/'scripts/data/prepare_qwen21_fixedpair_v5.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def tiktok():
    import numpy as np
    import torch
    from PIL import Image
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_full_assets import read_record,write_record
    from src.pose_control.qwen21_full.native_full_data import assemble_full_cache,BODY_SCHEMA,PAIR_SCHEMA
    from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
    h=helpers();root=h.TROOT;idx=json.loads((root/'data/smplx_import_v3/index.json').read_text())['records']
    selection=json.loads((root/'data/selection.json').read_text());refs=h.ref(root/'data/source_references_v2.npz');refs['schema']='native_local_reference_v4'
    source=idx['0265'];dest=root/'data/body_local_v4_v1';dest.mkdir(parents=True,exist_ok=False)
    raster=IndependentMeshRasterizer('cuda');out=[]
    for selected,stem in zip(selection['pairs'],('0374','0126')):
        target=idx[stem]
        mask=ROOT/'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset/00031/masks/0265.png'
        job={'pair_id':selected['pair_id'],'source_asset':source['asset'],'driver_asset':target['asset'],
            'source_canvas':source['canvas'],'target_canvas':target['canvas'],'source_masks':[h.ref(mask)],
            'source_rgb':source['rgb'],'target_rgb':target['rgb'],'source_indices':[0,-1],'source_slot_ids':[0,1],
            'units':'model_unit','scale_status':'model_scale','references':refs,'expression_mode':'driver',
            'distances':'absent_single_person'}
        sample,local=assemble_full_cache(job,renderer=raster)
        body=write_record(dest/(stem+'_body.npz'),sample,schema=BODY_SCHEMA)
        geo=write_record(dest/(stem+'_local.npz'),local,schema='native_local_condition_v4')
        out.append({'schema':PAIR_SCHEMA,'pair_id':selected['pair_id'],'scene_kind':'single','data_profile':'full_smplx',
            'body':body,'local':geo,'references':refs,'source_rgb':source['rgb'],'target_rgb':target['rgb']})
    with (dest/'pairs.jsonl').open('x') as f:
        for row in out:f.write(json.dumps(row)+'\n')
    # Visible Source surfaces only. Original pixels, then Canvas exactly once.
    asset=read_record(source['asset'],schema='native_smplx_asset_v4');canvas=CanvasSpec.from_record(source['canvas'])
    surface=raster.render(asset['people'][0],asset['K'],canvas.original_hw)
    with Image.open(mask) as im:human=np.asarray(im.convert('L'))>0
    labels=(1,0,5,8,0,0);support=[]
    for region,label in enumerate(labels):
        value=(surface['part']==label)&surface['valid']&human if label else np.zeros(canvas.original_hw,bool)
        p=dest/f'source_support_{region}.png';Image.fromarray(value.astype(np.uint8)*255).save(p);support.append(h.ref(p))
    h.write(dest/'source_support.json',support)
    return dest/'pairs.jsonl'


def hi4d_reference_job():
    from src.pose_control.qwen21_full.native_hybrid_cache import reference_jobs
    h=helpers();root=h.HROOT
    reference_jobs(root/'data/geometry_v3_v3/pairs.jsonl',root/'data/observations_direct_v2/observations.jsonl',
        root/'data/reference_jobs_v2',modalities={'arcface':True,'face_dino':True,'hand_dino':True},
        models=h.appearance_models())


def hi4d():
    import numpy as np
    from PIL import Image
    from src.pose_control.qwen21_full.native_hybrid_cache import build_cache
    from src.pose_control.qwen21_full.native_hybrid_assets import mask_array
    h=helpers();root=h.HROOT;geo=root/'data/geometry_v3_v3/pairs.jsonl';row=h.rows(geo)[0]
    ref=h.ref(root/'data/source_references_v1.npz');ref['schema']='native_local_reference_v4'
    rm=root/'data/source_references_index.jsonl'
    with rm.open('x') as f:f.write(json.dumps({'source_geometry_sha256':row['source_geometry']['sha256'],'references':ref})+'\n')
    dest=root/'data/body_local_v41_v1'
    build_cache(geo,root/'data/observations_direct_v2/observations.jsonl',dest,references_mode='manifest',references_manifest=rm)
    with np.load(row['source_geometry']['path'],allow_pickle=False) as d:
        part=d['part'];valid=d['valid']
    masks=[mask_array(r,part.shape[-2:]) for r in row['source_masks']]
    support=[]
    for region,(person,label) in enumerate(((0,1),(1,1),(0,5),(0,8),(1,5),(1,8))):
        value=(part[person]==label)&valid[person]&masks[person]
        p=dest/f'source_support_{region}.png';Image.fromarray(value.astype(np.uint8)*255).save(p);support.append(h.ref(p))
    h.write(dest/'source_support.json',support)
    return dest/'pairs.jsonl'


def promote(kind):
    import numpy as np
    from src.pose_control.qwen21_full.native_v5.data import promote_manifest
    from src.pose_control.qwen21_full.native_full_data import NativeFullDatasetV4
    from src.pose_control.qwen21_full.native_v5.conditioning import SCHEMA
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_hybrid_cache import observation_index,observation_for
    from src.pose_control.qwen21_full.dual_data_runtime import NativeDualDataset
    from src.pose_edit.geometry import render_poseguide
    h=helpers();root=h.TROOT if kind=='tiktok' else h.HROOT
    directory=root/('data/body_local_v4_v1' if kind=='tiktok' else 'data/body_local_v41_v1')
    v4=directory/'pairs.jsonl';reader=NativeFullDatasetV4(v4,load_target=False)
    selection=json.loads((root/'data/selection.json').read_text());selected={r['pair_id']:r for r in selection['pairs']}
    masks=json.loads((directory/'source_support.json').read_text());decl={}
    for row in reader.rows:
        sample=reader[reader.rows.index(row)];c=sample['local_condition_v4'];meta=c.metadata[0]
        decl[row['pair_id']]={'detail':{'availability':'available','role':'primary_source_only','support_kind':'visible_owned',
            'source_rgb_sha256':meta['source_rgb_sha256'],'source_person_ids':meta['source_person_ids'],'masks':masks},
            'supervision':{'availability':'unavailable','reason':'velocity-only pilot; no auxiliary identity/perceptual qualification requested'}}
    dest=root/'data/native_v5_v1';result=promote_manifest({'v4_manifest':h.ref(v4),'records':decl,'feather_px':8},dest)
    if kind=='tiktok':imports=json.loads((root/'data/smplx_import_v3/index.json').read_text())['records']
    else:
        dual=NativeDualDataset(root/'data/geometry_v3_v3/pairs.jsonl',load_target=False)
        observations=observation_index(root/'data/observations_direct_v2/observations.jsonl')
    for i,row in enumerate(result):
        sample=reader[i];case=selected[row['pair_id']];source=reader.rows[i]['source_rgb'];roles=['source']
        images=[{'role':'source','file':source,'canvas':sample['source_canvas'],'canvas_id':CanvasSpec.from_record(sample['source_canvas']).fingerprint(),'space':'original'}]
        spec=CanvasSpec.from_record(sample['target_canvas'])
        if kind=='tiktok':
            stem=case['bank_record']['target_stem'];guides=[imports[stem]['guide']];roles+=['target_pose']
        else:
            raw,meta=dual.geometry_record(dual.rows[i]['target_geometry']);obs=observation_for(observations,meta)
            if not obs['person_assigned'].all():raise ValueError('Selected Hi4D guides need verified one-to-one native observations')
            points=spec.pixel_points(obs['xy_native']);points[~obs['point_valid']]=np.nan
            guides=[]
            for name,which in (('combined',None),('person_0',0),('person_1',1)):
                p=dest/(row['pair_id']+'_'+name+'_pose.png')
                if which is None:
                    a=render_poseguide(points[0],(spec.canvas_hw[1],spec.canvas_hw[0]))
                    b=render_poseguide(points[1],(spec.canvas_hw[1],spec.canvas_hw[0]))
                    from PIL import ImageChops
                    ImageChops.lighter(a,b).save(p)
                else:render_poseguide(points[which],(spec.canvas_hw[1],spec.canvas_hw[0])).save(p)
                guides.append(h.ref(p))
            roles+=['combined_pose','person_0_pose','person_1_pose']
        for role,guide in zip(roles[1:],guides):
            images.append({'role':role,'file':guide,'canvas':spec.metadata(),'canvas_id':spec.fingerprint(),'space':'canvas'})
        condition={'schema':SCHEMA,'profile':selection['profile'],'pair_id':row['pair_id'],'primary_source_index':0,
            'target_canvas_id':spec.fingerprint(),'images':images,
            'prompt':{'text':case['prompt'],'sha256':case['prompt_sha256'],'bank':selection['bank'],
                'field':'compiled_prompt_inference','schema':case['bank_record']['schema'],'pair_id':row['pair_id']}}
        row['qwen_condition']=h.write(dest/(row['pair_id']+'_qwen.json'),condition)
    for role in ('train','observation'):
        chosen=[r for r in result if selected[r['pair_id']]['role']==role]
        with (dest/(role+'.jsonl')).open('x') as f:
            for row in chosen:f.write(json.dumps(row)+'\n')
        h.write(dest/(role+'_groups.json'),{r['pair_id']:('tiktok:00031' if kind=='tiktok' else 'hi4d:pair12:highfive12:cam4') for r in chosen})
    h.write(dest/'qwen_records.json',result)


def refresh_hi4d_observations():
    """Explicit supplemental detection for exactly the three selected raw frames."""
    from src.pose_control.qwen21_full.native_hybrid_cache import import_observations,detect_missing
    h=helpers();root=h.HROOT;data=json.loads((root/'data/geometry_v3_v3/index.json').read_text())
    manifest=root/'data/selected_original_frames.jsonl';requests=[]
    declared=[]
    for value in data.values():
        raw=value['raw'];meta=value['metadata']
        declared.append({**raw,'person_A':0,'person_B':1,'dwpose_native_height':meta['original_hw'][0],
            'dwpose_native_width':meta['original_hw'][1],'dwpose_pre_crop':False,'dwpose_pre_resize':False,
            'declaration_source':'verified_original_SMPL_camera_and_official_instance_masks'})
        requests.append(['pair12','highfive12','4',str(raw['frame_id'])])
    if not manifest.exists():
      with manifest.open('x') as f:
        for row in declared:f.write(json.dumps(row)+'\n')
    else:
        if h.rows(manifest)!=declared:raise ValueError('Existing raw input declaration differs')
    geo=root/'data/geometry_v3_v3/pairs.jsonl'
    if not (root/'data/observations_declared_v1').exists():
        import_observations(geo,manifest,root/'data/observations_declared_v1',layout='dwpose134')
    detector={'dwpose_det_model':str(ROOT/'models/DWPose/yolox_l.onnx'),
        'dwpose_pose_model':str(ROOT/'models/DWPose/dw-ll_ucoco_384.onnx')}
    detect_missing(geo,root/'data/observations_declared_v1/observations.jsonl',requests,detector,root/'data/observations_direct_v2')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('tiktok','hi4d-reference-job','hi4d-detect-selected','hi4d','promote-tiktok','promote-hi4d'))
    a=p.parse_args();{'tiktok':tiktok,'hi4d-reference-job':hi4d_reference_job,'hi4d-detect-selected':refresh_hi4d_observations,'hi4d':hi4d,
        'promote-tiktok':lambda:promote('tiktok'),'promote-hi4d':lambda:promote('hi4d')}[a.action]()


if __name__=='__main__':main()
