"""Build real Native/SDPose conditions and SAM-bound regional supervision."""
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

from prepare import read, ref, sha, write


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def verify_artifact(reference):
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Frozen frame artifact changed: '+reference['path'])


def frame_policy(frame, sd, face):
    valid = sd['valid_mask134']; left = sum(valid[92:113])>=8; right = sum(valid[113:134])>=8
    points = [p if v else [None,None] for p,v in zip(sd['points134_raw'],valid)]
    return {**frame, 'group_id': frame['clip_id'], 'candidate': sd['points134_raw'],
        'confidence': sd['scores134_raw'], 'effective_confidence': sd['scores134_raw'],
        'valid_keypoint_mask': valid, 'loss_mask': {'body': True, 'face': sum(valid[24:92])>=10,
            'hand_a': right, 'hand_b': left}, 'hand_raw_pose_usable': {'hand_a': right, 'hand_b': left},
        'hand_body_wrist_indices': {'hand_a':4,'hand_b':7}, 'hand_block_slices': {'hand_a':[113,134],'hand_b':[92,113]},
        'hand_landmarks21': {'hand_a': points[113:134], 'hand_b': points[92:113]},
        'detection': {'primary_box': sd['bbox_xyxy'], 'face': {'landmarks5': face}},
        'pose_backend': 'sdpose', 'annotation_source': frame['sdpose']}


def build_regions(sd, body, spec, people, rgb_sha, annotation_sha, sam_quality,*,segmentation_accepted=True):
    import numpy as np
    from src.pose_control.qwen21_full.native_v5.spatial_masks import (
        BODY_JOINTS, COARSE_HAND_GAIN, _gaussian, _hand_region, _coarse_hand_region, canvas_mask)
    frame = frame_policy({'original_hw': list(spec.original_hw), 'clip_id':'region', 'sdpose':{'sha256':annotation_sha}}, sd, None)
    valid = spec.valid_mask()[0].numpy(); person = np.minimum(canvas_mask(body,spec),valid)
    arrays = {'valid':valid}
    for family,n in (('person',2),('joint',2),('hands',4),('background',1)):
        arrays[family] = np.zeros((n,*spec.canvas_hw),np.float32); arrays[family+'_quality'] = np.zeros(n,np.float32)
    arrays['person'][0] = person; arrays['person_quality'][0] = float(np.clip(sam_quality,0,1))
    points = np.asarray(sd['points134_raw'],float); scores = np.asarray(sd['scores134_raw'],float)
    known = np.asarray(sd['valid_mask134'],bool)&np.isfinite(points).all(1)&np.isfinite(scores)
    # SDPose exports heatmap maxima, which can exceed 1; only ROI q is bounded.
    scores = np.clip(scores,0,1)
    canvas_points = spec.pixel_points(np.nan_to_num(points))
    sigma = float(np.clip(.025*(sd['bbox_xyxy'][3]-sd['bbox_xyxy'][1])*spec.scale_xy[1],8,24))
    used = []
    for index in BODY_JOINTS:
        if known[index] and scores[index]>=.3:
            arrays['joint'][0] = np.maximum(arrays['joint'][0],_gaussian(spec.canvas_hw,canvas_points[index],sigma)); used.append(scores[index])
    arrays['joint'][0] = np.minimum(arrays['joint'][0],person)
    arrays['joint_quality'][0] = float(np.mean(used)) if used else 0.
    hand_status = {}
    for slot,(side,key,start) in enumerate((('left','hand_b',92),('right','hand_a',113))):
        shape,status = _hand_region(frame,key,spec); quality_scale = 1.
        if not status.get('enabled'):
            shape,status = _coarse_hand_region(frame,key,spec,person)
            quality_scale = .5
            if status.get('enabled'):
                if status.get('gain') != COARSE_HAND_GAIN: raise ValueError('Unverified coarse-hand gain')
                shape = np.clip(shape/COARSE_HAND_GAIN,0,1)
        arrays['hands'][slot] = np.minimum(shape,person)
        eligible = known[start:start+21]
        q = quality_scale*float(np.mean(scores[start:start+21][eligible])) if eligible.any() else 0.
        arrays['hands_quality'][slot] = q if arrays['hands'][slot].any() else 0.
        hand_status[side] = {'region':status,'quality':float(arrays['hands_quality'][slot]),'source':'SDPose_weak_annotation'}
    arrays['background'][0] = np.maximum(valid-person,0); arrays['background_quality'][0] = float(arrays['background'][0].any())
    metadata = {'schema':'native_region_supervision_v1','target_rgb_sha256':rgb_sha,
        'canvas':spec.metadata(),'canvas_id':spec.fingerprint(),'person_ids':people,
        'annotation_sha256':annotation_sha,'hand_slot_order':['A-left','A-right','B-left','B-right'],
        'producer':'xiaohongshu_sdpose_sam2_regions_v1','joint_indices':list(BODY_JOINTS),
        'body_source':'SAM2.1 from locked SDPose body points and bbox','hands':hand_status,
        'quality_policy':'raw SDPose heatmap scores clipped to [0,1]; original annotation unchanged',
        'intersection':'pixel SAM support before area pooling; coverage applied once'}
    metadata['segmentation_accepted']=segmentation_accepted
    if not segmentation_accepted:
        for family in ('person','joint','hands','background'):
            arrays[family].fill(0);arrays[family+'_quality'].fill(0)
        for status in metadata['hands'].values():
            status['annotation_quality']=status['quality'];status['quality']=0.;status['enabled']=False
        metadata['missing_supervision']='SAM coverage below .65; no spatial GT, no background metric, no weight redistribution'
    return arrays,metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True); parser.add_argument('--model-repo',type=Path,default=Path('/home/shangguanrz/project/pic-edit'))
    a = parser.parse_args(); sys.path.insert(0,str(a.repo)); root=a.root
    if (root/'prepared.json').exists(): raise FileExistsError('Assets/configuration already locked')
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from insightface import model_zoo
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
    from src.pose_control.qwen21_full.native_full_assets import read_record,write_record
    from src.pose_control.qwen21_full.native_full_data import assemble_full_cache
    from src.pose_control.qwen21_full.native_v5.data import training_sidecar,write_sidecar
    from src.pose_control.qwen21_full.native_v5.contracts import DETAIL_SCHEMA,SUPERVISION_SCHEMA
    from src.pose_control.qwen21_full.native_v5.conditioning import SCHEMA
    from src.pose_control.qwen21_full.region_supervision import write_region_sidecar
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    from scripts.data.build_native_spatial_weights import SamForeground
    from sam_quality import AssessedForeground
    from cache_preparation import write_once
    fraction=float(os.environ.get('REGION_PILOT_MEMORY_FRACTION','.9'))
    if not 0<fraction<=.95:raise ValueError('GPU memory fraction must be in (0,.95]')
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(fraction)
    selection=read(root/'data/selection.json'); geometry=read(root/'data/geometry.json'); initial=read(root/'data/initialization.json')
    base_path=Path('/data/shangguanrz/pic-edit-experiments/one/bilibili-region-sdpose-sft-rl-20261120-v1/configs/templates/uniform/S0.json')
    base=read(base_path); manifest=base['model_manifest']
    manifest_path=Path(manifest['path'])
    if not manifest_path.exists():
        old='/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/'
        if not str(manifest_path).startswith(old):raise FileNotFoundError(manifest_path)
        manifest_path=Path(str(manifest_path).replace(old,'/data/shangguanrz/pic-edit-experiments/one/qwen21_full_native_v1/',1))
    if sha(manifest_path)!=manifest['sha256']:raise ValueError('Frozen Qwen model manifest differs after path relocation')
    base['model_manifest']=ref(manifest_path)
    source_dir=Path(__file__).resolve().parents[1]/'source/native-builders'
    helper=load_module(source_dir/'hia_training.py','xhs_native_builder'); helper.ROOT=a.model_repo
    appearance=load_module(source_dir/'appearance.py','xhs_native_appearance')
    producer=ref(__file__); model=a.model_repo/'add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz'
    importer=helper.Importer(model,root/'data/model_metadata',producer)
    renderer=helper.BoundedRenderer(IndependentMeshRasterizer('cuda'))
    face_root=a.model_repo/'models/antelopev2'; dino=face_root/'dinov2-giant'
    models={'arcface_model':str(face_root/'glintr100.onnx'),'arcface_sha256':sha(face_root/'glintr100.onnx'),
            'dino_model':str(dino),'dino_files':{str(p.relative_to(dino)):sha(p) for p in sorted(dino.rglob('*'))
            if p.is_file() and p.suffix in ('.safetensors','.bin','.json')},'device':'cuda'}
    features=appearance.SourceFeatureExtractor(models)
    detector=model_zoo.get_model(str(face_root/'scrfd_10g_bnkps.onnx'),providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1,input_size=(640,640))
    sam_args=SimpleNamespace(sam_repo=a.model_repo/'add/tools/sam2',
        sam_checkpoint=Path('/data/shangguanrz/models/home-models/sam2.1/sam2.1_hiera_large.pt'),
        sam_config='configs/sam2.1/sam2.1_hiera_l.yaml',device='cuda')
    sam=AssessedForeground(SamForeground(sam_args))
    write_once(root/('data/asset-producer-'+producer['sha256'][:12]+'.json'),{'source':producer,'importer':ref(source_dir/'hia_training.py'),
          'appearance':ref(source_dir/'appearance.py'),'appearance_models':models,
          'sam_checkpoint':ref(sam_args.sam_checkpoint),'smplx_model':ref(model),'detector':ref(face_root/'scrfd_10g_bnkps.onnx'),
          'canvas_long_side':512,'pose_backend':'sdpose','geometry_backend':'SMPLest-X_locked_SDpose_bbox',
          'qwen_configuration_source':ref(base_path),'qwen_model_manifest':base['model_manifest']})
    built={}; entries={}
    for number,(fid,original) in enumerate(sorted(selection['frames'].items()),1):
        directory=root/f'data/native_frames/{fid}'; marker=directory/'ready.json'
        if marker.exists():
            old=read(marker)
            for value in old['artifacts'].values():
                verify_artifact(value)
            built[fid]=old; entries[old['region_key']]=old['region_entry']; continue
        attempt=directory/('build_'+uuid.uuid4().hex[:10]); attempt.mkdir(parents=True)
        sd=read(original['sdpose']['path'])
        if ref(original['rgb_path'])['sha256']!=original['rgb_sha256']: raise ValueError('Selected RGB changed')
        image=cv2.imread(original['rgb_path']); boxes,landmarks=detector.detect(image)
        face=None
        if len(boxes):
            index=int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])))
            center=landmarks[index].mean(0); bb=sd['bbox_xyxy']
            if bb[0]<=center[0]<=bb[2] and bb[1]<=center[1]<=bb[3]: face=landmarks[index].tolist()
        frame=frame_policy(original,sd,face); asset,joints=importer.convert(geometry['frames'][fid]['parameters'],
            {'producer':'SMPLest-X','crop':original['sdpose']},frame,attempt)
        person_id='xiaohongshu:'+frame['clip_id']+':0'
        asset['people'][0]['metadata']['person_id']=person_id
        asset['metadata'].update(dataset='xiaohongshu_single',camera_id=person_id+':camera',scale_group_id=person_id+':model',
            geometry_method='smplestx_sdpose_bbox',original_parameters=geometry['frames'][fid]['parameters'])
        asset_ref=write_record(attempt/'asset.npz',asset,schema='native_smplx_asset_v4')
        spec=helper.canvas_for(frame['original_hw'],512)
        with Image.open(frame['rgb_path']) as im:
            rgb=im.convert('RGB'); body,sam_report=sam(rgb,frame)
        mask=attempt/'sam_person.png'
        with mask.open('xb') as f: Image.fromarray(body*255).save(f,format='PNG')
        empty,detail=helper.source_references(asset,ref(frame['rgb_path']),spec,asset_ref,frame,renderer)
        refs=features.extract(frame,ref(frame['rgb_path']),spec,asset_ref,empty)
        refs_ref=write_record(attempt/'source_references.npz',refs,schema='native_local_reference_v4')
        detail_ref=write_sidecar(attempt/'source_detail.npz',detail,DETAIL_SCHEMA)
        with Image.open(frame['skeleton']['path']) as im: guide=spec.image(im.convert('RGB'))
        guide_path=attempt/'sdpose_guide.png'
        with guide_path.open('xb') as f: guide.save(f,format='PNG')
        arrays,metadata=build_regions(sd,body,spec,[person_id,'__inactive__'],frame['rgb_sha256'],frame['sdpose']['sha256'],
                                     sam_report['sam_score'],segmentation_accepted=sam_report['segmentation_accepted'])
        region=write_region_sidecar(attempt/'regions.npz',arrays,metadata); key=frame['rgb_sha256']+':'+spec.fingerprint()
        item={'frame':frame,'canvas':spec.metadata(),'sam_report':sam_report,'region_key':key,
              'region_entry':{'file':region,'metadata':metadata},'artifacts':{'asset':asset_ref,'mask':ref(mask),
              'references':refs_ref,'detail':detail_ref,'guide':ref(guide_path),'region':region}}
        write(marker,item); built[fid]=item; entries[key]=item['region_entry']
        print(json.dumps({'native_frame':number,'total':len(selection['frames']),'frame_id':fid}),flush=True)
    region_index=write_once(root/'data/region_index.json',{'schema':'native_region_index_v1','entries':entries})
    canonical=read(root/'data/qwen_condition_index.json') if (root/'data/qwen_condition_index.json').exists() else None
    splits=[s for s in ('train','val','test') if any(p['split']==s for p in selection['pairs'])]
    rows={s:[] for s in splits}; groups={s:{} for s in splits}; evaluation=[]
    for pair in selection['pairs']:
        source=built[pair['source_frame_id']]; target=built[pair['target_frame_id']]; frame=target['frame']; pair_id=pair['pair_id']
        directory=root/f'data/native_pairs/{pair_id}'; directory.mkdir(parents=True)
        sample,local=assemble_full_cache({'pair_id':pair_id,'source_asset':source['artifacts']['asset'],
            'driver_asset':target['artifacts']['asset'],'source_canvas':source['canvas'],'target_canvas':target['canvas'],
            'source_masks':[source['artifacts']['mask']],'source_rgb':ref(source['frame']['rgb_path']),
            'target_rgb':ref(frame['rgb_path']),'source_indices':[0,-1],'source_slot_ids':[0,1],
            'units':'model_unit','scale_status':'model_scale','references':source['artifacts']['references'],
            'expression_mode':'driver','distances':'absent_single_person'},renderer=renderer)
        local=helper.apply_regional_validity(local,frame)
        sample['pair_metadata'].update(dataset_type='xiaohongshu_native_full_smplx_v4',original_note_id=pair['note_id'])
        body_ref=write_record(directory/'body.npz',sample,schema='native_full_body_record_v4')
        local_ref=write_record(directory/'local.npz',local,schema='native_local_condition_v4')
        v4={'schema':'native_full_pair_v4','pair_id':pair_id,'scene_kind':'single','data_profile':'full_smplx',
            'body':body_ref,'local':local_ref,'references':source['artifacts']['references'],
            'source_rgb':ref(source['frame']['rgb_path']),'target_rgb':ref(frame['rgb_path'])}
        v4_path=directory/'v4.jsonl'
        with v4_path.open('x') as f: f.write(json.dumps(v4)+'\n')
        references=read_record(source['artifacts']['references'],schema='native_local_reference_v4')
        supervision=training_sidecar({**sample,'local_condition_v4':local,'local_references_v4':references},
            {'availability':'unavailable','reason':'Legacy auxiliary identity loss disabled in the regional FM contrast'})
        supervision=replace(supervision,weights=helper.mask_regional_weights(supervision.weights,frame)).validate()
        supervision_ref=write_sidecar(directory/'supervision.npz',supervision,SUPERVISION_SCHEMA)
        prompt="Change the person's pose to match <image2>. Preserve the identity, face, clothing and background in <image1>."
        import hashlib
        prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()
        bank=write(directory/'prompt-bank.json',{'schema':'xiaohongshu_native_prompt_v1','pair_id':pair_id,
            'compiled_prompt_inference':prompt,'compiled_prompt_inference_sha256':prompt_sha,'producer':'fixed_source_identity_target_sdpose'})
        source_spec=CanvasSpec.from_record(source['canvas']); target_spec=CanvasSpec.from_record(target['canvas'])
        condition={'schema':SCHEMA,'profile':'single_2ref','pair_id':pair_id,'primary_source_index':0,
            'target_canvas_id':target_spec.fingerprint(),'target_pose_backend':'sdpose','images':[
            {'role':'source','file':ref(source['frame']['rgb_path']),'canvas':source['canvas'],'canvas_id':source_spec.fingerprint(),'space':'original'},
            {'role':'target_pose','file':target['artifacts']['guide'],'canvas':target['canvas'],'canvas_id':target_spec.fingerprint(),'space':'canvas'}],
            'prompt':{'text':prompt,'sha256':prompt_sha,'field':'compiled_prompt_inference','bank':bank,'schema':'xiaohongshu_native_prompt_v1','pair_id':pair_id}}
        if canonical is not None:
            key='xiaohongshu_'+pair['split']+'\0'+pair_id
            bound=canonical['entries'][key]['condition']
            from src.pose_control.qwen21_full.native_v5.conditioning import read_condition
            read_condition(bound,{**sample,'local_condition_v4':local,'native_scene_kind':'single'})
        else:bound=write(root/f'conditions/{pair_id}.json',condition)
        row={'schema':'native_full_pair_v5','pair_id':pair_id,'v4_manifest':ref(v4_path),'references':source['artifacts']['references'],
             'detail':source['artifacts']['detail'],'supervision':supervision_ref,'qwen_condition':bound}
        split=pair['split']; rows[split].append(row); groups[split][pair_id]='xiaohongshu:'+pair['note_id']
        sd=read(frame['sdpose']['path']); points=np.asarray(sd['points134_raw'],float)
        valid=np.asarray(sd['valid_mask134'],bool)&np.isfinite(points).all(1)
        points=target_spec.pixel_points(np.nan_to_num(points)); points[~valid]=0
        evaluation.append({'pair_id':pair_id,'split':split,'group':groups[split][pair_id],'frame_id':frame['frame_id'],
            'source_rgb':ref(source['frame']['rgb_path']),'target_rgb':ref(frame['rgb_path']),
            'source_mask':source['artifacts']['mask'],
            'source_mask_accepted':source['sam_report']['segmentation_accepted'],
            'source_canvas':source['canvas'],'target_canvas':target['canvas'],'target_points134':points.tolist(),
            'target_scores134':sd['scores134_raw'],'target_valid134':valid.tolist(),'sdpose_keypoints':frame['sdpose'],
            'region':target['artifacts']['region'],'native_hand_condition':bool(local.hand_roi_valid.any())})
    sources=[]
    for split in splits:
        path=root/f'data/{split}.jsonl'
        with path.open('x') as f:
            for row in rows[split]:f.write(json.dumps(row)+'\n')
        sources.append({'source_id':'xiaohongshu_'+split,'role':'validation' if split=='val' else split,
            'manifest':ref(path),'groups':write(root/f'data/{split}_groups.json',groups[split]),
            'data_profile':'full_smplx','scene_kind':'single','weight':1. if split=='train' else None,'region_supervision':region_index})
    write(root/'data/evaluation.json',evaluation)
    experiment=selection.get('experiment',{})
    budgets=experiment.get('budgets',{'S0':100,'S1':50,'S2':100,'S3':50})
    for arm,mode in (('uniform','uniform'),('regional','region_balanced_v1')):
        for phase,budget in budgets.items():
            branches=['body'] if phase=='S0' else ['joint'] if phase=='S3' else ['face','hand','detail','poselink']
            routes=[{'name':b,'branch':b,'weight':1.,'loss':{'region_mode':mode},'controls':{'interaction':False},
                     'base_controls':{'interaction':False,'raster':False,'part_contact':False}} for b in branches]
            raw={'schema':'native_v7_training_v1','model_path':base['model_path'],'model_manifest':base['model_manifest'],
                'initialization':initial['initialization'],'sources':sources,'cache_root':str(root/'cache'),'cache_identity':None,
                'output_root':str(root/f'output/{arm}/{phase}'),'seed':42,'qwen_condition_profile':'single_2ref',
                'paper_loss':{'person_fm':0.,'contact_fm':0.,'identity_contrastive':0.,'temperature':.07},
                'stage':{'stage_id':phase+'_xiaohongshu_region_pilot','phase':phase,'routes':routes,'budget_updates':budget,'accumulation':4,
                  'exclude_groups':['geometry_dual','relation_raster','relation_tokens','interaction_controller','interaction_heads','strength_interaction'],
                  'optimizer':{'preset':'protect_inherited','multiplier':.1 if phase=='S3' else 1.}},
                'probe_every_updates':experiment.get('probe_every_updates',25),'probe_samples':experiment.get('probe_samples',5),
                'save_every_updates':experiment.get('save_every_updates',25),'generation_every_updates':0,
                'generation_steps':20,'generation_seed':42,'diagnostics':False,
                'continuous_stage_authorization':{'source':'direct_user_message','text':'小红书；最新架构和Bilibili20000步权重；开启自动化训练程序，等待1916769结束；测试集出图评测，扩大规模；空间加权和RL对照'}}
            write(root/f'configs/templates/{arm}/{phase}.json',TrainingConfigV7.from_dict(raw).record())
    write(root/'prepared.json',{'schema':'xiaohongshu_region_sdpose_pilot_v1','root':str(root),'data_root':selection['data_root'],
        'dataset_label':'Xiaohongshu','train_pairs':len(rows['train']),'val_pairs':len(rows['val']),
        'test_pairs':len(rows.get('test',[])),'train_videos':len(set(groups['train'].values())),
        'val_videos':len(set(groups['val'].values())),'test_notes':len(set(groups.get('test',{}).values())),'group_unit':'note',
        'validation_source_id':'xiaohongshu_val','initialization':initial['initialization'],'initialization_source':initial['source'],
        'region_index':region_index,'region_targets':len(entries),'budgets':budgets,'updates_per_sft_arm':sum(budgets.values()),
        'followup_updates':experiment.get('followup_updates',10),'test_source_id':'xiaohongshu_test' if 'test' in splits else None,
        'pose_backend':'sdpose','cache_prepared':False,'old_data_modified':False,
        'scope':'fixed-budget single-seed experiment; qualified body/hand subset; no test optimization'})
    print(json.dumps({'prepared':str(root/'prepared.json'),'train':len(rows['train']),'val':len(rows['val'])}),flush=True)


if __name__ == '__main__': main()
