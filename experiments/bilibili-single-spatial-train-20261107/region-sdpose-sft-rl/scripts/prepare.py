"""Create an immutable, video-balanced Bilibili/SDPose pilot and regional ROIs.

Only new artifacts are written below --root. Existing RGB, guides, annotations,
manifests, checkpoints and frozen caches are always read-only.
"""
from copy import deepcopy
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import os

REPO = Path(os.environ.get('PICTUREEDIT_REPO',str(Path(__file__).resolve().parents[4])))
sys.path.insert(0, str(REPO))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def ref(path):
    return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def write(path,value,*,lines=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        if lines:
            for row in value:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        else:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
    return ref(path)


def select_rows(rows,groups,per_video,seed,qualifier=None,unique_key=None):
    if per_video<1:raise ValueError('Positive per-video pilot size required')
    videos=defaultdict(list)
    for row in rows:videos[groups[row['pair_id']]].append(row)
    rng=random.Random(seed);selected=[]
    for video in sorted(videos):
        candidates=list(videos[video]);rng.shuffle(candidates)
        if qualifier:
            preferred=[r for r in candidates if qualifier(r)]
            if preferred:candidates=[preferred[0]]+[r for r in candidates if r['pair_id']!=preferred[0]['pair_id']]
        chosen=[];keys=set()
        for row in candidates:
            key=unique_key(row) if unique_key else row['pair_id']
            if key in keys:continue
            chosen.append(deepcopy(row));keys.add(key)
            if len(chosen)==per_video:break
        if len(chosen)!=per_video:raise ValueError('Too few distinct targets in video '+video)
        selected.extend(chosen)
    return selected


def rebind_condition(old,pose_ref):
    if old['profile']!='single_2ref' or [x['role'] for x in old['images']]!=['source','target_pose']:
        raise ValueError('This pilot uses the existing single-person two-image protocol')
    new=deepcopy(old);new['images'][1]['file']=deepcopy(pose_ref)
    new['images'][1]['space']='canvas';new['target_pose_backend']='sdpose'
    return new


def hand_key(frame,side):
    expected={'left':7,'right':4}[side]
    keys=[k for k,v in frame['hand_body_wrist_indices'].items() if v==expected]
    if len(keys)!=1:raise ValueError('Missing explicit anatomical wrist mapping')
    return keys[0]


def make_regions(frame,sd,old_maps,sample,annotation_sha):
    import numpy as np
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v5.spatial_masks import (
        BODY_JOINTS,COARSE_HAND_GAIN,_gaussian,_hand_region,_coarse_hand_region,valid_keypoints)
    spec=CanvasSpec.from_record(sample['target_canvas']);meta=sample['local_condition_v4'].metadata[0]
    if (sd['height'],sd['width'])!=spec.original_hw:raise ValueError('SDPose/RGB original Canvas differs')
    local=deepcopy(frame)
    local['candidate']=sd['points134_raw'];local['confidence']=sd['scores134_raw']
    local['effective_confidence']=sd['scores134_raw'];local['valid_keypoint_mask']=sd['valid_mask134']
    points,scores,valid_points=valid_keypoints(local)
    body=old_maps['body'][0].numpy().astype(np.float32);valid=spec.valid_mask()[0].numpy()
    body=np.minimum(body,valid)
    arrays={'valid':valid}
    for family,n in {'person':2,'joint':2,'hands':4,'background':1}.items():
        arrays[family]=np.zeros((n,*spec.canvas_hw),np.float32)
        arrays[family+'_quality']=np.zeros(n,np.float32)
    arrays['person'][0]=body;arrays['person_quality'][0]=1
    canvas_points=spec.pixel_points(points)
    height=max(1.,(sd['bbox_xyxy'][3]-sd['bbox_xyxy'][1])*spec.scale_xy[1])
    sigma=float(np.clip(.025*height,8.,24.));accepted=[]
    for i in BODY_JOINTS:
        if valid_points[i] and scores[i]>=.3:
            arrays['joint'][0]=np.maximum(arrays['joint'][0],_gaussian(spec.canvas_hw,canvas_points[i],sigma))
            accepted.append(float(scores[i]))
    arrays['joint'][0]=np.minimum(arrays['joint'][0],body)
    arrays['joint_quality'][0]=float(np.mean(accepted)) if accepted else 0
    hand_status={}
    for slot,side in enumerate(('left','right')):
        key=hand_key(frame,side)
        if frame.get(key+'_status') in ('occluded','out_of_frame') or frame.get(key+'_present') is False:
            hand_status[side]={'enabled':False,'reason':'explicit_visibility_negative'};continue
        fine,fine_status=_hand_region(frame,key,spec)
        if fine_status.get('enabled'):
            shape=fine;quality=1.;status={'source':'legacy_fine','fine':fine_status}
        else:
            coarse,coarse_status=_coarse_hand_region(local,key,spec,body)
            if not coarse_status.get('enabled'):
                hand_status[side]={'enabled':False,'source':'sdpose_coarse','coarse':coarse_status};continue
            if coarse_status.get('gain')!=COARSE_HAND_GAIN:raise ValueError('Unverified coarse-hand gain')
            shape=np.clip(coarse/COARSE_HAND_GAIN,0,1);quality=.5
            status={'source':'sdpose_coarse','coarse':coarse_status}
        arrays['hands'][slot]=np.minimum(shape,body)
        arrays['hands_quality'][slot]=quality if arrays['hands'][slot].any() else 0
        hand_status[side]={'enabled':bool(arrays['hands'][slot].any()),**status}
    arrays['background'][0]=np.maximum(valid-body,0)
    arrays['background_quality'][0]=float(arrays['background'][0].any())
    metadata={'schema':'native_region_supervision_v1','target_rgb_sha256':meta['driver_rgb_sha256'],
        'canvas':spec.metadata(),'canvas_id':spec.fingerprint(),'person_ids':meta['target_person_ids'],
        'hand_slot_order':['A-left','A-right','B-left','B-right'],'annotation_sha256':annotation_sha,
        'frame_id':frame['frame_id'],'producer':'bilibili_sdpose_regions_v1','joint_indices':list(BODY_JOINTS),
        'joint_sigma_canvas_px':sigma,'hands':hand_status,'pose_backend':'sdpose'}
    return arrays,metadata


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,default=Path('/data/shangguanrz/datasets/bilibili'))
    ap.add_argument('--base-run',type=Path,default=Path('/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261108-ungated-v1'))
    ap.add_argument('--spatial-index',type=Path,default=Path('/data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1/full_visible_region_v2_20261107/index.json'))
    ap.add_argument('--initialization',type=Path,default=Path('/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native-v7-singlepair-overfit-20261117-v1/initialization/v7.pt'))
    ap.add_argument('--per-video',type=int,default=2)
    args=ap.parse_args();root=args.root.resolve()
    if not str(root).startswith('/data/shangguanrz/'):raise ValueError('Pilot artifacts belong on persistent /data')
    if (root/'prepared.json').exists() or (root/'data').exists():raise FileExistsError('Use a fresh pilot root')
    from PIL import Image
    import numpy as np
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62 as NativeFullDatasetV7
    from src.pose_control.qwen21_full.native_v5.spatial_weights import read_spatial_components
    from src.pose_control.qwen21_full.region_supervision import write_region_sidecar
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    root.mkdir(parents=True,exist_ok=True)
    frames={r['frame_id']:r for r in map(json.loads,(args.data_root/'frames/frames.jsonl').read_text().splitlines()) if r}
    spatial=json.loads(args.spatial_index.read_text())['entries'];info={};evaluation=[];region_entries={};sources=[]
    def get_info(row):
        if row['pair_id'] not in info:
            condition=json.loads(Path(row['qwen_condition']['path']).read_text())
            pose_path=Path(condition['images'][1]['file']['path'])
            fid=next((p.name for p in pose_path.parents if p.name in frames),None)
            if fid is None:raise ValueError('Cannot bind original target frame '+str(pose_path))
            bvid=fid.rsplit('_',1)[0]
            kp=args.data_root/'geometry/skeleton/sdpose_2d_kp'/bvid/(fid+'_keypoints.json')
            sd=json.loads(kp.read_text());info[row['pair_id']]=(condition,fid,kp,sd)
        return info[row['pair_id']]
    def qualified(row):
        _,fid,_,sd=get_info(row);frame=frames[fid]
        fine=frame.get('hand_raw_pose_usable',{})
        fine=any(fine.values()) if isinstance(fine,dict) else bool(fine)
        return fine and sum(sd['valid_mask134'][92:])>=16
    for split in ('train','val'):
        rows=[json.loads(x) for x in (args.base_run/f'manifests/{split}/pairs.jsonl').read_text().splitlines() if x]
        groups=json.loads((args.base_run/f'manifests/{split}/groups.json').read_text())
        selected=select_rows(rows,groups,args.per_video,42,qualified,lambda row:get_info(row)[1])
        selection_ref=write(root/f'data/{split}_original.jsonl',selected,lines=True)
        dataset=NativeFullDatasetV7(selection_ref['path'],load_target=True,data_profile='full_smplx')
        rebound=[]
        for i,row in enumerate(selected):
            sample=dataset[i];meta=sample['local_condition_v4'].metadata[0]
            condition,fid,kp,sd=get_info(row);frame=frames[fid];canvas=CanvasSpec.from_record(sample['target_canvas'])
            key=meta['driver_rgb_sha256']+':'+meta['target_canvas_id']
            if key not in region_entries:
                maps=read_spatial_components(spatial[key]['file'],gt_rgb_sha256=meta['driver_rgb_sha256'],
                    target_canvas_id=meta['target_canvas_id'],canvas=sample['target_canvas'])
                arrays,metadata=make_regions(frame,sd,maps,sample,sha(kp))
                regional_ref=write_region_sidecar(root/f'data/regions/{fid}_{canvas.fingerprint()[:10]}.npz',arrays,metadata)
                region_entries[key]={'file':regional_ref,'metadata':metadata}
                skeleton=Path(sd['skeleton_png'])
                with Image.open(skeleton) as image:guide=canvas.image(image.convert('RGB'))
                guide_path=root/f'conditions/guides/{fid}_{canvas.fingerprint()[:10]}.png'
                guide_path.parent.mkdir(parents=True,exist_ok=True)
                with guide_path.open('xb') as f:guide.save(f,format='PNG')
            guide_path=root/f'conditions/guides/{fid}_{canvas.fingerprint()[:10]}.png'
            new_condition=rebind_condition(condition,ref(guide_path))
            condition_ref=write(root/f'conditions/{row["pair_id"]}.json',new_condition)
            rebound.append({**row,'qwen_condition':condition_ref})
            raw_points=np.asarray(sd['points134_raw'],float);valid=np.asarray(sd['valid_mask134'],bool)&np.isfinite(raw_points).all(1)
            points=canvas.pixel_points(np.nan_to_num(raw_points));points[~valid]=0
            gt_path=Path(frame.get('rgb_path') or args.data_root/f'frames/images/{fid.rsplit("_",1)[0]}/{fid}.png')
            if sha(gt_path)!=meta['driver_rgb_sha256']:raise ValueError('GT path does not match paired supervision')
            evaluation.append({'pair_id':row['pair_id'],'split':split,'group':groups[row['pair_id']],
                'frame_id':fid,'source_rgb':condition['images'][0]['file'],'target_rgb':ref(gt_path),
                'source_canvas':condition['images'][0]['canvas'],'target_canvas':canvas.metadata(),
                'target_points134':points.tolist(),'target_scores134':sd['scores134_raw'],
                'target_valid134':valid.tolist(),'sdpose_keypoints':ref(kp),'region':region_entries[key]['file'],
                'native_hand_condition':bool(sample['local_condition_v4'].hand_roi_valid.any())})
        manifest=write(root/f'data/{split}.jsonl',rebound,lines=True)
        group_ref=write(root/f'data/{split}_groups.json',{r['pair_id']:groups[r['pair_id']] for r in selected})
        sources.append({'source_id':'bilibili_'+split,'role':'train' if split=='train' else 'validation',
            'manifest':manifest,'groups':group_ref,'data_profile':'full_smplx','scene_kind':'single','weight':1.,'spatial_weights':None})
    region_index=write(root/'data/region_index.json',{'schema':'native_region_index_v1','entries':region_entries})
    for source in sources:source['region_supervision']=region_index
    write(root/'data/evaluation.json',evaluation)
    initial=ref(args.initialization);base=json.loads((args.base_run/'stages/S0/config.json').read_text())
    budgets={'S0':100,'S1':50,'S2':100,'S3':50}
    for arm,mode in (('uniform','uniform'),('regional','region_balanced_v1')):
        for phase,budget in budgets.items():
            branches=['body'] if phase=='S0' else ['joint'] if phase=='S3' else ['face','hand','detail','poselink']
            routes=[{'name':b,'branch':b,'weight':1.,'loss':{'region_mode':mode},
                'controls':{'interaction':False},'base_controls':{'interaction':False,'raster':False,'part_contact':False}} for b in branches]
            raw={'schema':'native_v7_training_v1','model_path':base['model_path'],'model_manifest':base['model_manifest'],
                'initialization':initial,'sources':sources,'cache_root':str(root/'cache'),'cache_identity':None,
                'output_root':str(root/f'output/{arm}/{phase}'),'seed':42,'qwen_condition_profile':'single_2ref',
                'paper_loss':{'person_fm':0.,'contact_fm':0.,'identity_contrastive':0.,'temperature':.07},
                'stage':{'stage_id':phase+'_bilibili_sdpose_region_pilot','phase':phase,'routes':routes,
                    'budget_updates':budget,'accumulation':4,'exclude_groups':['geometry_dual','relation_raster','relation_tokens','interaction_controller','interaction_heads','strength_interaction'],
                    'optimizer':{'preset':'protect_inherited','multiplier':.1 if phase=='S3' else 1.}},
                'probe_every_updates':25,'probe_samples':5,'save_every_updates':25,'generation_every_updates':0,
                'generation_steps':20,'generation_seed':42,'diagnostics':False,
                'continuous_stage_authorization':{'source':'direct_user_message','text':'行接入吧,就用/data/shangguanrz/datasets/bilibili 作为训练试试,然后注意保存生成的图像,使用SDpose; 最好也用下我们rl的思路训练试试,也做个对比'}}
            config=TrainingConfigV7.from_dict(raw)
            write(root/f'configs/templates/{arm}/{phase}.json',config.record())
    summary={'schema':'bilibili_region_sdpose_pilot_v1','root':str(root),'data_root':str(args.data_root),
        'train_pairs':sum(r['split']=='train' for r in evaluation),'val_pairs':sum(r['split']=='val' for r in evaluation),
        'train_videos':len({r['group'] for r in evaluation if r['split']=='train'}),
        'val_videos':len({r['group'] for r in evaluation if r['split']=='val'}),
        'region_targets':len(region_entries),'initialization':initial,'region_index':region_index,
        'budgets':budgets,'updates_per_sft_arm':sum(budgets.values()),'cache_prepared':False,
        'pose_backend':'sdpose','old_data_modified':False,'scope':'video-balanced integration pilot; not full-data or multi-seed efficacy'}
    write(root/'prepared.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
