"""Reuse verified legacy physical records and assemble only missing Native pairs.

All numerical assembly comes from source/legacy/prepare_native_batch.py and the
unchanged same-fit float-depth/target-cache wrappers. The new code only binds
provenance, applies the standard lossless level-3 NPZ codec, and resumes stages.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import uuid

from data_prepare import SHA, EXPECTED_COUNTS, check_frame_binding, digest, load, publish, ref, rows, write_exclusive_bytes


def module_from(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def seal(path, root):
    path=Path(path); target=root/'source/data_producers'/(SHA(path)+'_'+path.name)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        write_exclusive_bytes(target,path.read_bytes())
    if SHA(target)!=SHA(path):
        raise ValueError('Sealed data producer differs: '+str(path))
    return ref(target)


def check_pair_binding(pair, source, target, frame, sample, policy):
    metadata=sample['pair_metadata']
    required={'pair_id':pair['pair_id'],'source_rgb_sha256':source['rgb']['sha256'],
              'target_rgb_sha256':target['rgb']['sha256'],'source_asset_sha256':source['artifacts']['asset']['sha256'],
              'driver_asset_sha256':target['artifacts']['asset']['sha256'],'geometry_method':'sapiens_v1',
              'source_loss_mask':pair['source_loss_mask'],'target_loss_mask':pair['target_loss_mask'],
              'selected_source_geometry':source['original_geometry'],'selected_target_geometry':target['original_geometry']}
    for key, expected in required.items():
        if metadata.get(key)!=expected:
            raise ValueError('Native pair provenance differs: '+key)
    if sample['source_canvas']!=source['canvas'] or sample['target_canvas']!=target['canvas']:
        raise ValueError('Native pair source/target Canvas differs')
    for local in sample['local_condition_v4'].metadata:
        if local.get('loss_mask')!=frame['loss_mask'] or local.get('hand_raw_pose_usable')!=frame.get('hand_raw_pose_usable'):
            raise ValueError('Native pair fine regional masks/raw hand validity changed')
        for key, expected in {'source_asset_sha256':source['artifacts']['asset']['sha256'],
                              'driver_record_sha256':target['artifacts']['asset']['sha256'],
                              'source_rgb_sha256':source['rgb']['sha256'],
                              'driver_rgb_sha256':target['rgb']['sha256']}.items():
            if local.get(key)!=expected:
                raise ValueError('Native local provenance differs: '+key)
        scene=local.get('local_scene_depth',{})
        if scene.get('policy')!=policy or scene.get('source_asset_sha256')!=target['artifacts']['asset']['sha256'] or scene.get('target_canvas_id')!=local.get('target_canvas_id'):
            raise ValueError('Native pair exact target float scene policy differs')


def check_feature_identity(record,models,appearance_producer):
    expected=digest({'physical_frame':record['physical_identity'],'models':models,
                     'feature_producer':appearance_producer,'batch_limit':16})
    if record['identity']!=expected:
        raise ValueError('Frozen Source feature identity differs from original physical/model/producer bindings')


def decoded_rgb_sha(image):
    return hashlib.sha256(repr(image.size).encode()+b'\0'+image.convert('RGB').tobytes()).hexdigest()


def split_inputs(args):
    frames, pairs, counts, all_ids, all_pairs, groups = {}, {}, {}, set(), set(), set()
    for split in ('train','val','test'):
        frame_rows=rows(args.data_root/split/'frames.jsonl')
        pair_rows=rows(args.data_root/split/'pairs.jsonl')
        by_id={f['frame_id']:f for f in frame_rows}
        clips={f['clip_id'] for f in frame_rows}
        count={'videos':len(clips),'frames':len(frame_rows),'pairs':len(pair_rows)}
        if count!=EXPECTED_COUNTS[split]:
            raise ValueError('Published full split counts differ: '+split+' '+str(count))
        if len(by_id)!=len(frame_rows) or set(by_id)&all_ids or clips&groups:
            raise ValueError('Video/frame split leakage or duplicate IDs')
        for f in frame_rows:
            if f['split']!=split or f['group_id']!=f['clip_id']:
                raise ValueError('Frame split/group differs')
        for p in pair_rows:
            if p['pair_id'] in all_pairs:
                raise ValueError('Duplicate original pair')
            all_pairs.add(p['pair_id'])
            source,target=(by_id[p[k]] for k in ('source_frame_id','target_frame_id'))
            if p['split']!=split or source['clip_id']!=target['clip_id'] or p['clip_id']!=source['clip_id']:
                raise ValueError('Pair split/video differs')
            if p['source_loss_mask']!=source['loss_mask'] or p['target_loss_mask']!=target['loss_mask']:
                raise ValueError('Pair original fine masks differ')
        frames.update(by_id); pairs[split]=pair_rows; counts[split]=count
        all_ids.update(by_id); groups.update(clips)
    return frames,pairs,counts


def source_records(args, frames, helper, selected_renderer):
    import torch
    from PIL import Image
    from src.pose_control.qwen21_full.native_full_assets import read_record
    from src.pose_control.qwen21_full.native_v5.data import read_sidecar
    from src.pose_control.qwen21_full.native_v5.contracts import NativeDetailReferencesV5,DETAIL_SCHEMA
    index={};pixel_owners={}
    root=args.old_root/'native_sapiens_v1'
    import appearance
    legacy_producers={}
    helper_path=Path(helper.__file__)
    for name,path in {'prepare_native_batch.py':args.legacy_root/'prepare_native_batch.py',
                      'appearance.py':args.legacy_root/'appearance.py','hia_training.py':helper_path}.items():
        original=root/'source'/(SHA(path)+'_'+name)
        if SHA(original)!=SHA(path):
            raise ValueError('Historical physical producer snapshot differs: '+name)
        legacy_producers[name]=ref(original)
    for number,frame in enumerate(frames.values(),1):
        candidates=sorted((root/'appearance'/frame['frame_id']).glob('*/ready.json'))
        valid=[]
        for marker in candidates:
            record=load(marker)
            if not record.get('source_features_complete'):
                continue
            try:
                check_frame_binding(frame,record,helper.canvas_for(frame['original_hw'],512).metadata())
            except ValueError:
                continue
            valid.append((marker,record))
        if len(valid)!=1:
            raise ValueError('Expected exactly one compatible complete Source record: '+frame['frame_id'])
        marker,record=valid[0]
        SHA.verify(record['rgb'])
        with Image.open(record['rgb']['path']) as image:
            if (image.height,image.width)!=tuple(frame['original_hw']):
                raise ValueError('Original RGB dimensions differ from physical Source')
            pixels=decoded_rgb_sha(image)
        previous=pixel_owners.get(pixels)
        if previous is not None and previous['split']!=frame['split']:
            raise ValueError('Decoded RGB pixel leakage across train/val/test')
        pixel_owners[pixels]={'frame_id':frame['frame_id'],'split':frame['split']}
        for value in record['original_geometry'].values():
            SHA.verify(value)
        clip,fid=frame['clip_id'],frame['frame_id']
        expected={'parameters':args.data_root/'geometry/smplx/sapiens_v1'/clip/(fid+'.pt'),
                  'depth':args.data_root/'geometry/depth/sapiens_v1'/clip/(fid+'_depth.png'),
                  'normal':args.data_root/'geometry/normal/sapiens_v1'/clip/(fid+'_normal.png')}
        if record['original_geometry']!={name:ref(path) for name,path in expected.items()}:
            raise ValueError('Source geometry must use exact selected v1 assets')
        for value in record['artifacts'].values():
            SHA.verify(value)
        physical=load(Path(record['artifacts']['asset']['path']).parent.parent/'ready.json')
        if physical['identity']!=record.get('physical_identity'):
            raise ValueError('Source feature physical lineage differs')
        for key in ('rgb','canvas','original_geometry','loss_mask'):
            if physical[key]!=record[key]:
                raise ValueError('Source physical/reference lineage differs: '+key)
        asset=read_record(record['artifacts']['asset'],schema='native_smplx_asset_v4')
        if asset['metadata'].get('original_geometry')!=record['original_geometry'] or asset['metadata'].get('rgb_sha256')!=record['rgb']['sha256']:
            raise ValueError('Physical asset selected-v1/RGB provenance differs')
        references=read_record(record['artifacts']['references'],schema='native_local_reference_v4').validate()
        metadata=references.metadata[0]
        if (metadata.get('source_frame_id')!=fid or metadata.get('loss_mask')!=frame['loss_mask']
            or metadata.get('hand_raw_pose_usable')!=frame.get('hand_raw_pose_usable')
            or metadata.get('source_rgb_sha256')!=record['rgb']['sha256']):
            raise ValueError('Source-only feature provenance/fine validity differs')
        if any(value.dtype!=torch.float32 for value in (references.arcface,references.face_dino,references.hand_dino)):
            raise ValueError('Frozen Source ArcFace/DINO must remain FP32')
        if metadata.get('arcface_preprocessing')!=appearance.ARCFACE_PREPROCESSING or metadata.get('dino_preprocessing')!=appearance.DINO_PREPROCESSING:
            raise ValueError('Source feature preprocessing differs from pinned original producer')
        model_root=args.legacy_repo_root/'models/antelopev2'
        models={'arcface_model':str(model_root/'glintr100.onnx'),'arcface_sha256':metadata['arcface_sha256'],
                'dino_model':str(model_root/'dinov2-giant'),'dino_files':metadata['dino_files'],'device':'cuda'}
        check_feature_identity(record,models,legacy_producers['appearance.py'])
        person=asset['people'][0]['metadata']
        expected_physical=digest({'frame':frame,'selected':record['original_geometry'],'producer':legacy_producers,
                                  'smplx_model_sha256':person['model_sha256'],'model_options':person['model_options'],
                                  'expression_basis_sha256':person['expression_basis_sha256'],'canvas_long_side':512,
                                  'appearance':{**models,'phase':'physical_assets_before_batched_source_features'}})
        if physical['identity']!=expected_physical:
            raise ValueError('Physical Source original frame/model/producer identity differs: '+fid)
        detail=read_sidecar(record['artifacts']['detail'],NativeDetailReferencesV5,DETAIL_SCHEMA)
        if detail.metadata[0]['source_rgb_sha256']!=record['rgb']['sha256']:
            raise ValueError('Source detail RGB differs')
        selected_renderer.register(asset['people'][0],asset['K'],frame['original_hw'],expected['depth'],expected['normal'])
        index[fid]=record
        if number%20==0:
            print(json.dumps({'event':'reused_source_verified','completed':number,'total':len(frames)}),flush=True)
    if len(frames)==1960:
        publish(args.run_root/'data_audits/source_frames.json',{
            'schema':'bilibili_single_source_audit_v1','passed':True,'frames':len(frames),
            'frames_decoded':len(frames),'unique_pixel_images':len(pixel_owners),'cross_split_pixel_conflicts':[],
            'hash_method':'sha256(repr((width,height))+NUL+decoded_RGB_bytes)',
            'physical_and_source_feature_identities_recomputed':True,'frozen_source_dtype':'float32',
            'fine_hand_validity_unchanged':True,'selected_geometry':'sapiens_v1','canvas_long_side':512,
            'model_constructed':False,'optimizer_created':False,'training_started':False})
    return index


def read_reused_pair(args, marker, pair, source, target, frame, policy):
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5
    from src.pose_control.qwen21_full.native_v5.conditioning import read_condition
    result=load(marker)
    if result.get('split')!=pair['split'] or result.get('group_id')!='bilibili:'+frame['group_id']:
        raise ValueError('Existing pair split/original-video group differs')
    for item in result['artifacts'].values():
        SHA.verify(item)
    if result['row'].get('references')!=source['artifacts']['references'] or result['row'].get('detail')!=source['artifacts']['detail']:
        raise ValueError('Existing pair Source feature/detail references differ')
    check=args.run_root/'native/reuse_checks'/(pair['pair_id']+'_'+result['identity']+'.jsonl')
    publish(check,[result['row']],jsonl=True)
    dataset=NativeFullDatasetV5(check,load_target=True,data_profile='full_smplx')
    sample=dataset[0]
    check_pair_binding(pair,source,target,frame,sample,policy)
    condition=read_condition(result['row']['qwen_condition'],sample)
    # read_condition validates every source/pose guide/Canvas/prompt reference.
    if load(result['row']['qwen_condition']['path']).get('profile')!='single_2ref':
        raise ValueError('Existing pair Qwen profile differs')
    return result


def verify_reuse(args):
    """CPU-only real-reader audit; never constructs a rasterizer or learned model."""
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(args.legacy_root))
    import local_scene_depth
    helper_path=args.repo_root/'experiments/6.qwen-controlnet union/hia_geometry_training_20261029/scripts/hia_training.py'
    helper=module_from(helper_path,'bilibili_single_hia_cpu_check')
    frames,pairs,counts=split_inputs(args)
    if args.verify_limit:
        wanted=set()
        for split in pairs:
            for pair in pairs[split]:
                if list((args.old_root/'native_sapiens_v1/pairs'/pair['pair_id']).glob('*/ready.json')):
                    wanted.update((pair['source_frame_id'],pair['target_frame_id']))
                    if len(wanted)>=args.verify_limit*2:
                        break
            if len(wanted)>=args.verify_limit*2:
                break
        frames={k:v for k,v in frames.items() if k in wanted}
    registry=SimpleNamespace(register=lambda *unused:None)
    prepared=source_records(args,frames,helper,registry)
    checked=0
    for split in pairs:
        for pair in pairs[split]:
            if pair['source_frame_id'] not in prepared or pair['target_frame_id'] not in prepared:
                continue
            for marker in sorted((args.old_root/'native_sapiens_v1/pairs'/pair['pair_id']).glob('*/ready.json')):
                source,target=(prepared[pair[key]] for key in ('source_frame_id','target_frame_id'))
                read_reused_pair(args,marker,pair,source,target,frames[pair['target_frame_id']],local_scene_depth.POLICY)
                checked+=1
                print(json.dumps({'event':'real_native_reuse_verified','pairs':checked,'pair_id':pair['pair_id']}),flush=True)
                if args.verify_limit and checked>=args.verify_limit:
                    return


def reuse_historical_pair(args,pair,source,target,frame,policy):
    for marker in sorted((args.old_root/'native_sapiens_v1/pairs'/pair['pair_id']).glob('*/ready.json')):
        candidate=load(marker)
        if candidate.get('row',{}).get('references')!=source['artifacts']['references']:
            continue
        # A different Source feature version is not this candidate. Once matched,
        # every SHA/Canvas/data-reader error is terminal, never a rebuild signal.
        return read_reused_pair(args,marker,pair,source,target,frame,policy)
    return None


def prepare_pairs(args):
    from src.pose_control.qwen21_full import native_full_data
    from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
    sys.path.insert(0,str(args.legacy_root))
    batch=module_from(args.legacy_root/'prepare_native_batch.py','bilibili_single_legacy_batch')
    import local_scene_depth
    from local_scene_depth import LocalSceneDepth
    from pair_target_cache import SingleTargetCache
    helper_path=args.repo_root/'experiments/6.qwen-controlnet union/hia_geometry_training_20261029/scripts/hia_training.py'
    helper=module_from(helper_path,'bilibili_single_hia_helper')
    batch.sha=SHA; helper.sha=SHA
    # Preserve legacy JSON bytes/schema while making new ready markers atomic.
    # Physical/Source producers are never rerun or changed.
    batch.write=publish
    frames,pairs,counts=split_inputs(args)
    selected=batch.SelectedGeometryRenderer(IndependentMeshRasterizer(args.device))
    renderer=helper.BoundedRenderer(selected,max_bytes=512*1024*1024)
    prepared=source_records(args,frames,helper,selected)
    sources={name:seal(path,args.run_root) for name,path in {
        'legacy_pair':args.legacy_root/'prepare_native_batch.py','legacy_appearance':args.legacy_root/'appearance.py',
        'legacy_scene':args.legacy_root/'local_scene_depth.py','legacy_target_cache':args.legacy_root/'pair_target_cache.py',
        'hia_helper':helper_path,'native_full_data':Path(native_full_data.__file__),
        'prepare_dataset':Path(__file__),'data_prepare':Path(__file__).with_name('data_prepare.py'),
        'npz_codec':Path(__file__).with_name('npz_codec.py')}.items()}
    originals={'root_frames':ref(args.data_root/'frames/frames.jsonl'),'root_pairs':ref(args.data_root/'pairs/pairs.jsonl')}
    for split in ('train','val','test'):
        originals[split+'_frames']=ref(args.data_root/split/'frames.jsonl')
        originals[split+'_pairs']=ref(args.data_root/split/'pairs.jsonl')
    original=native_full_data.assemble_full_cache
    aligned=LocalSceneDepth(original); cache=SingleTargetCache(aligned.assemble)
    native_full_data.assemble_full_cache=cache.assemble
    options=SimpleNamespace(output_root=args.run_root/'native')
    manifests={}; reused=created=0
    try:
        for split in ('train','val','test'):
            built=[]
            for number,pair in enumerate(pairs[split],1):
                source,target=(prepared[pair[key]] for key in ('source_frame_id','target_frame_id'))
                frame=frames[pair['target_frame_id']]
                # New-run resume is producer-pinned by the unchanged legacy identity.
                result=reuse_historical_pair(args,pair,source,target,frame,local_scene_depth.POLICY)
                if result is not None:
                    reused+=1
                if result is None:
                    result=batch.prepare_pair(pair,source,target,frame,options,helper,renderer,sources)
                    created+=1
                built.append(result)
                if number%20==0 or number==len(pairs[split]):
                    print(json.dumps({'event':'native_pairs','split':split,'completed':number,'total':len(pairs[split]),
                                      'old_reused':reused,'new_or_resumed':created}),flush=True)
            root=args.run_root/'native/manifests'/split
            manifests[split]={'manifest':publish(root/'pairs.jsonl',[b['row'] for b in built],True),
                              'groups':publish(root/'groups.json',{b['row']['pair_id']:b['group_id'] for b in built}),
                              'pairs':len(built)}
    finally:
        native_full_data.assemble_full_cache=original
    report={'schema':'bilibili_single_pair_preparation_v1','counts':counts,'frames_reused':len(prepared),
            'old_pair_records_reused':reused,'new_or_resumed_pair_records':created,'producer':sources,
            'inputs':originals,'lossless_npz_compresslevel':3,'local_scene_depth_policy':local_scene_depth.POLICY,
            'body_png_values_unchanged':True,'fine_hand_validity_unchanged':True,'strict_subset_created':False,
            'all_original_rows_preserved':True,'model_constructed':False,'optimizer_created':False,'training_started':False}
    report['source_records']=publish(args.run_root/'native/source_records.json',prepared)
    report['source_audit']=ref(args.run_root/'data_audits/source_frames.json')
    return manifests,counts,publish(args.run_root/'data_audits/pair_preparation.json',report)


def full_native_audit(args,manifests,spatial):
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5
    from src.pose_control.qwen21_full.native_v5.conditioning import read_condition
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig,SourceSpec,stage_preset
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    from src.pose_control.qwen21_full.native_v5.spatial_weights import read_spatial_index,read_spatial_components
    from src.data.canvas_contract import CanvasSpec
    report_path=args.run_root/'data_audits/native_full_data.json'
    if report_path.exists():
        existing=load(report_path)
        if not existing.get('passed') or existing.get('manifests')!=manifests or existing.get('spatial_weights')!=spatial:
            raise ValueError('Existing full Native data audit differs')
        return ref(report_path)
    frames,pairs,counts=split_inputs(args)
    source_manifest={}
    for fid,frame in frames.items():
        compatible=[]
        for marker in (args.old_root/'native_sapiens_v1/appearance'/fid).glob('*/ready.json'):
            record=load(marker)
            if record.get('source_features_complete') and record.get('split')==frame['split']:
                compatible.append(record)
        if len(compatible)!=1:
            raise ValueError('Ambiguous original Source during final audit')
        source_manifest[fid]=compatible[0]
    import local_scene_depth
    sources=tuple(SourceSpec('bilibili_'+s,role='train' if s=='train' else 'validation',
                  manifest=manifests[s]['manifest'],groups=manifests[s]['groups'],data_profile='full_smplx',
                  scene_kind='single',weight=1.,spatial_weights=spatial) for s in ('train','val'))
    # All local inputs are tested through the final S3 routes; no backend is constructed.
    stage=stage_preset('S3')
    stage=replace(stage,routes=tuple(r for r in stage.routes if r.branch!='interaction'))
    config=TrainingConfig(stage=stage,sources=sources,qwen_condition_profile='single_2ref')
    # Check provenance inside the real preflight reader, so large pair NPZs are
    # decoded once per split rather than repeating all train/validation reads.
    route_path=args.run_root/'data_audits'/('native_train_val_routes_'+uuid.uuid4().hex+'.json')
    spatial_index,_=read_spatial_index(spatial)
    pair_by_id={p['pair_id']:p for values in pairs.values() for p in values}
    checked=set()
    for split in ('train','val','test'):
        if [r['pair_id'] for r in rows(manifests[split]['manifest']['path'])]!=[p['pair_id'] for p in pairs[split]]:
            raise ValueError('Final Native manifest does not preserve every ordered original pair')
    original_reader=NativeFullDatasetV5.__getitem__
    def validated_reader(dataset,index):
        sample=original_reader(dataset,index)
        pid=dataset.rows[index]['pair_id'];pair=pair_by_id[pid]
        source,target=(source_manifest[pair[k]] for k in ('source_frame_id','target_frame_id'))
        check_pair_binding(pair,source,target,frames[pair['target_frame_id']],sample,local_scene_depth.POLICY)
        # Train/val condition and spatial checks are executed by DataPlan itself.
        if pair['split']=='test':
            read_condition(dataset.rows[index]['qwen_condition'],sample)
            spec=CanvasSpec.from_record(target['canvas'])
            key=target['rgb']['sha256']+':'+spec.fingerprint()
            if key not in spatial_index['entries']:
                raise ValueError('Spatial index misses actual Native test target RGB/Canvas')
            entry=spatial_index['entries'][key]
            maps=read_spatial_components(entry['file'],gt_rgb_sha256=target['rgb']['sha256'],
                                         target_canvas_id=spec.fingerprint(),canvas=spec)
            import torch
            if not torch.equal(maps['valid'],sample['valid_image_mask'].float()):
                raise ValueError('Test spatial coverage differs from actual Native image Canvas')
        checked.add(pid)
        if len(checked)%50==0:
            print(json.dumps({'event':'native_full_reader','completed':len(checked),'total':12156,'split':pair['split']}),flush=True)
        return sample
    NativeFullDatasetV5.__getitem__=validated_reader
    try:
        routes=NativeV5Trainer(config).preflight_data(route_path)
        test=NativeFullDatasetV5(manifests['test']['manifest']['path'],load_target=True,data_profile='full_smplx')
        for index in range(len(test)):
            test[index]
    finally:
        NativeFullDatasetV5.__getitem__=original_reader
    if len(checked)!=12156 or checked!=set(pair_by_id):
        raise ValueError('Full Native preflight did not read all 12156 pairs')
    report={'schema':'bilibili_single_native_data_audit_v1','passed':True,'pairs_read':len(checked),'counts':counts,
            'manifests':manifests,'spatial_weights':spatial,'test_in_training_sources':False,
            'train_val_route_audit':ref(route_path),'source_target_canvas_geometry_float_scene_verified':True,
            'all_original_rows_retained':True,'model_constructed':False,'optimizer_created':False,'training_started':False}
    return publish(report_path,report)
