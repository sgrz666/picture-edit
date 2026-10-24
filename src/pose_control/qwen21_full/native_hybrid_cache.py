"""Offline Hi4D Hybrid orchestration. Never renders or caches learned tokens."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import torch

from src.data.canvas_contract import CanvasSpec, letterbox
from .dual_data_runtime import NativeDualDataset
from .native_full_assets import read_record, write_record, sha256
from .native_full_data import BODY_SCHEMA, PAIR_SCHEMA
from .native_local_condition import LOCAL_SCHEMA, REFERENCE_SCHEMA
from .native_hybrid_assets import (ASSET_SCHEMA, OBS_SCHEMA, HybridConfig, file_ref, frame_key,
    geometry_key, import_frame, detect_frame, load_jsonl, read_observation, mask_array)
from .native_hybrid_local import build_local, local_observations, empty_references


def write_json(path, value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def write_jsonl(path, rows):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')


def new_output(path):
    path=Path(path).resolve()
    # All new cache products belong on the persistent data volume.
    if not str(path).startswith('/data/shangguanrz/'):
        raise ValueError('Hybrid outputs must use a new version directory under /data/shangguanrz/')
    path.mkdir(parents=True,exist_ok=False)
    return path


def unique_geometry(dataset):
    result={}
    for row in dataset.rows:
        for side in ('source','target'):
            ref=row[side+'_geometry'];_,m=dataset.geometry_record(ref);key=geometry_key(m)
            if key in result and result[key]['sha256']!=ref['sha256']:
                raise ValueError('Multiple V3 geometries for the same original frame/camera')
            result[key]=ref
    return result


def import_observations(v3_manifest, generation_manifest, output, *, layout, config=None):
    config=config or HybridConfig();dataset=NativeDualDataset(v3_manifest,load_target=False)
    indexed={}
    for row in load_jsonl(generation_manifest): indexed.setdefault(frame_key(row),[]).append(row)
    destination=new_output(output);rows=[];statuses=Counter();reports=[]
    source=file_ref(generation_manifest)
    for number,(key,ref) in enumerate(unique_geometry(dataset).items()):
        asset=import_frame(dataset,ref,indexed,source,layout,config)
        stored=write_record(destination/f'observation_{number:06d}.npz',asset,schema=ASSET_SCHEMA)
        rows.append({'schema':OBS_SCHEMA,'key':list(key),'geometry':ref,'asset':stored})
        statuses.update(x['status'] for x in asset['metadata']['assignment'])
        reports.append({'key':list(key),'assignment':asset['metadata']['assignment'],
            'candidate_failures':asset['metadata']['provenance'].get('candidate_failures',[]),
            'source_status':asset['metadata']['provenance'].get('source_status','unavailable')})
    write_jsonl(destination/'observations.jsonl',rows)
    write_json(destination/'report.json',{'schema':'native_hybrid_import_report_v1','frames':len(rows),
        'generation_rows':sum(map(len,indexed.values())),'person_status':dict(statuses),'records':reports,'config':config.record(),
        'v3_manifest':file_ref(v3_manifest),'generation_manifest':source,'runtime_acceptance':'not_run'})
    return rows


def detect_missing(v3_manifest, observations, requests, detector_config, output, *, config=None):
    """Requests is a JSON list of exact frame keys; never scan/detect everything."""
    config=config or HybridConfig();dataset=NativeDualDataset(v3_manifest,load_target=False)
    geos=unique_geometry(dataset);rows=load_jsonl(observations)
    selected={tuple(map(str,k)) for k in requests}
    if not selected or len(selected)!=len(requests): raise ValueError('Declare a nonempty unique supplemental detection key list')
    known={tuple(r['key']) for r in rows}
    if not selected<=known: raise ValueError('Supplemental key is not present in the observation index')
    destination=new_output(output);result=[];report=[]
    for number,row in enumerate(rows):
        key=tuple(row['key'])
        if key not in selected: result.append(row);continue
        asset=read_observation(row,geometry_sha=geos[key]['sha256'])
        if all(x['status']=='matched' for x in asset['metadata']['assignment']):
            raise ValueError('detect-missing only accepts explicitly missing/rejected/ambiguous records')
        updated=detect_frame(dataset,geos[key],asset,detector_config,config)
        stored=write_record(destination/f'observation_{number:06d}.npz',updated,schema=ASSET_SCHEMA)
        result.append({**row,'asset':stored});report.append({'key':list(key),'before':asset['metadata']['assignment'],
            'after':updated['metadata']['assignment']})
    write_jsonl(destination/'observations.jsonl',result)
    write_json(destination/'report.json',{'requests':report,'config':config.record(),'prior_manifest':file_ref(observations)})


def observation_index(path):
    result={}
    for row in load_jsonl(path):
        key=tuple(row['key'])
        if key in result: raise ValueError('Duplicate observation frame/camera key')
        result[key]=row
    return result


def observation_for(index,meta):
    key=geometry_key(meta)
    if key not in index: raise ValueError('Observation index must preserve missing rows explicitly: '+str(key))
    asset=read_observation(index[key],geometry_sha=meta['record_sha256'])
    if asset['metadata']['camera_id']!=meta['camera_id'] or asset['metadata']['person_ids']!=meta['person_ids']:
        raise ValueError('Observation identity/camera differs from V3')
    return asset


def official_masks(asset):
    m=asset['metadata'];provenance=m['provenance'];refs=provenance.get('masks')
    if refs is None:
        # An unavailable source chain disables only local control, never the body.
        return np.zeros((2,*m['original_hw']),bool)
    if len(refs)!=2: raise ValueError('Exactly two explicit official mask references required')
    return np.stack([mask_array(ref,m['original_hw']) for ref in refs])


def source_job(dataset, row, asset, source_data, meta, config, modalities, models):
    spec=letterbox(meta['original_hw'],dataset.reference_side)
    raw_masks=official_masks(asset)
    masks=torch.stack([spec.stack(torch.from_numpy(m).float(),nearest=True)>.5 for m in raw_masks])
    # Head masks are from the existing V3 source part asset; no local renderer.
    from .depth_geometry import onto_canvas
    part=torch.from_numpy(source_data['part']==1).float()[:,None]
    heads=onto_canvas(part,spec)[:,0]>.5
    from .native_hybrid_assets import body_anchors
    uv,av=body_anchors(source_data,meta);wrist=spec.pixel_points(uv[:,[20,21]].astype(np.float32))
    wrist[~av[:,[20,21]]]=np.nan
    obs=local_observations(asset,spec,masks,heads&masks,wrist,config)
    people=[]
    for p in range(2):
        points=obs['face'][p,4:];valid=obs['face_valid'][p,4:]
        groups=(list(range(36,42)),list(range(42,48)),[30],[48],[54])
        five_valid=all(valid[g].all() for g in groups)
        person={'face_box':obs['face_boxes'][p].tolist(),'hand_boxes':obs['hand_boxes'][p].tolist(),
            'identity_valid':bool(modalities['arcface'] and obs['face_roi'][p] and five_valid),
            'face_texture_valid':bool(modalities['face_dino'] and obs['face_roi'][p]),
            'hand_texture_valid':(obs['hand_roi'][p]&modalities['hand_dino']).tolist()}
        if five_valid:
            person['arcface_five_points_canvas_pixels']=(np.stack([points[g,:2].mean(0) for g in groups])*
                np.array([spec.canvas_hw[1],spec.canvas_hw[0]],np.float32)).tolist()
        people.append(person)
    return {**models,'role':'source_only','modalities':modalities,'source_person_ids':meta['person_ids'],
        'references':[{'rgb':{**meta['rgb'],'path':str(dataset.verify(meta['rgb']).resolve())},'canvas':spec.metadata(),
            'person_ids':meta['person_ids'],'geometry_sha256':meta['record_sha256'],'people':people}],
        'hybrid_observation':asset['metadata'],'config':config.record()}


def reference_jobs(v3_manifest, observations, output, *, modalities, models, config=None, reference_side=512):
    if set(modalities)!={'arcface','face_dino','hand_dino'} or any(type(v)!=bool for v in modalities.values()):
        raise ValueError('Declare three independent appearance requests as bools')
    config=config or HybridConfig();dataset=NativeDualDataset(v3_manifest,load_target=False,reference_side=reference_side)
    index=observation_index(observations);destination=new_output(output);jobs=[];seen=set()
    for row in dataset.rows:
        ref=row['source_geometry']
        if ref['sha256'] in seen: continue
        seen.add(ref['sha256']);data,meta=dataset.geometry_record(ref)
        asset=observation_for(index,meta)
        job=source_job(dataset,row,asset,data,meta,config,modalities,models)
        path=destination/f'source_{len(jobs):06d}.json';write_json(path,job)
        jobs.append({'source_geometry_sha256':ref['sha256'],'source_rgb_sha256':meta['rgb']['sha256'],
            'job':file_ref(path),'references_output':str(destination/f'references_{len(jobs):06d}.npz'),
            'status':'pending_explicit_extraction','availability':[p for p in job['references'][0]['people']]})
    write_jsonl(destination/'reference_jobs.jsonl',jobs)
    write_json(destination/'report.json',{'sources':len(jobs),'extraction_executed':False,'modalities':modalities,
        'config':config.record(),'v3_manifest':file_ref(v3_manifest)})


def build_cache(v3_manifest, observations, output, *, references_mode, references_manifest=None,
                config=None, max_pixels=512*512, reference_side=512):
    config=config or HybridConfig()
    if references_mode not in ('explicit_empty','manifest') or (references_mode=='manifest')!=(references_manifest is not None):
        raise ValueError('Explicit empty appearance or a complete reference manifest required')
    dataset=NativeDualDataset(v3_manifest,load_target=False,max_pixels=max_pixels,reference_side=reference_side)
    index=observation_index(observations);reference_index={}
    if references_manifest is not None:
        for r in load_jsonl(references_manifest):
            key=r['source_geometry_sha256']
            if key in reference_index: raise ValueError('Duplicate source appearance declaration')
            reference_index[key]=r['references']
    destination=new_output(output);output_rows=[];reports=[]
    for i,row in enumerate(dataset.rows):
        _,source=dataset.geometry_record(row['source_geometry']);_,target=dataset.geometry_record(row['target_geometry'])
        if source['model_family_indices']!=[1,1] or target['model_family_indices']!=[1,1]:
            raise ValueError('Hybrid profile must retain native Hi4D SMPL')
        # This is the authoritative V3 sample, with original numeric conditions.
        sample=dataset[i];sample.pop('src_image')
        sample['pair_metadata']={**sample['pair_metadata'],'source_rgb_sha256':row['source_rgb']['sha256'],
            'target_rgb_sha256':row['target_rgb']['sha256'],'data_profile':'hybrid_hi4d',
            'body_model_family':'smpl','v3_manifest_sha256':sha256(v3_manifest)}
        body=write_record(destination/f'body_{i:06d}.npz',sample,schema=BODY_SCHEMA)
        asset=observation_for(index,target)
        local,report=build_local(sample,asset,official_masks(asset),source_meta=source,target_meta=target,
            body_record_sha256=body['sha256'],config=config)
        spec=CanvasSpec.from_record(sample['source_canvas'])
        if references_mode=='explicit_empty':
            refs=empty_references(source,spec,declaration=references_mode)
            reference=write_record(destination/f'references_{i:06d}.npz',refs,schema=REFERENCE_SCHEMA)
        else:
            if row['source_geometry']['sha256'] not in reference_index: raise ValueError('Declared source features missing; no silent fallback')
            reference=reference_index[row['source_geometry']['sha256']]
            refs=read_record(reference,schema=REFERENCE_SCHEMA).validate()
        rm=refs.metadata[0]
        if (rm['source_rgb_sha256']!=source['rgb']['sha256'] or rm['source_canvas_id']!=spec.fingerprint()
            or rm['source_person_ids']!=source['person_ids'] or rm['reference_count']!=1
            or rm['reference_sources'][0]['geometry_sha256']!=source['record_sha256']):
            raise ValueError('Source references must bind this original source image, canvas and person order')
        local_ref=write_record(destination/f'local_{i:06d}.npz',local,schema=LOCAL_SCHEMA)
        rgb={s:{'path':str(dataset.verify(row[s+'_rgb']).resolve()),'sha256':row[s+'_rgb']['sha256']} for s in ('source','target')}
        output_rows.append({'schema':PAIR_SCHEMA,'scene_kind':'dual','data_profile':'hybrid_hi4d','pair_id':row['pair_id'],
            'body':body,'local':local_ref,'references':reference,'source_rgb':rgb['source'],'target_rgb':rgb['target'],
            'body_record_sha256':body['sha256'],'driver_record_sha256':row['target_geometry']['sha256'],
            'v3_pair':deepcopy(row),'v3_manifest':file_ref(v3_manifest),'body_model_family':'smpl'})
        reports.append({'pair_id':row['pair_id'],**report,'appearance':{'arcface':refs.arcface_valid.tolist(),
            'face_dino':refs.face_dino_valid.tolist(),'hand_dino':refs.hand_dino_valid.tolist()}})
    write_jsonl(destination/'pairs.jsonl',output_rows)
    write_json(destination/'report.json',{'schema':'native_hi4d_hybrid_report_v1','samples':len(output_rows),
        'records':reports,'config':config.record(),'references_mode':references_mode,'runtime_acceptance':'not_run','all_rows_retained':True,
        'observation_manifest':file_ref(observations),'v3_manifest':file_ref(v3_manifest)})
    return output_rows
