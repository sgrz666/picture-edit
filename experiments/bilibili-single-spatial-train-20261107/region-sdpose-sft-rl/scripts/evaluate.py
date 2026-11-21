"""SDPose image evaluation, then CPU identity/background enrichment in qwen21.

Separate actions preserve the existing SDPose and Qwen package environments.
Every generated image remains at its original path and is bound by SHA256.
"""
import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys

REPO=Path(os.environ.get('PICTUREEDIT_REPO',str(Path(__file__).resolve().parents[4])))
sys.path.insert(0,str(REPO))


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def bind_models(root,name,paths,settings):
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    record={'schema':'bilibili_reward_model_identity_v1','settings':settings,
            'files':[{'path':str(p.resolve()),'sha256':file_sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))]}
    path=root/('report/model-identity-'+name+'.json')
    if path.exists():
        if read(path)!=record:raise ValueError('Frozen reward model/code identity changed: '+name)
    else:write(path,record)
    return {'path':str(path),'sha256':file_sha(path)}


def aggregate(records):
    import numpy as np
    summary={'images':len(records),'person_detected_rate':sum(bool(x.get('person_detected')) for x in records)/max(1,len(records)),
             'pose':{},'identity':{},'pixel':{}}
    for family in ('body','arm','wrist','hands','left_hand','right_hand'):
        rows=[x['pose'][family] for x in records if x['pose'][family]['applicable']]
        summary['pose'][family]={'active_images':len(rows)}
        for key in ('reward','pck_002','pck_005','recall','l2_px','l2_px_all','nme_all'):
            vals=[r[key] for r in rows if r[key] is not None]
            summary['pose'][family][key]=float(np.mean(vals)) if vals else None
    for key in ('id_cos_src','id_cos_tgt','identity_reward'):
        values=[x['identity'][key] for x in records if 'identity' in x and x['identity'][key] is not None]
        summary['identity'][key]={'count':len(values),'mean':float(np.mean(values)) if values else None}
    identity=[x['identity'] for x in records if 'identity' in x]
    summary['identity']['applicable_images']=sum(x['applicable'] for x in identity)
    summary['identity']['generated_face_detected_rate']=sum(x['generated_faces']>0 for x in identity)/len(identity) if identity else None
    for key in ('background_mse','valid_mse','background_reward'):
        values=[x['pixel'][key] for x in records if 'pixel' in x and x['pixel'][key] is not None]
        summary['pixel'][key]=float(np.mean(values)) if values else None
    return summary


def pose_action(args):
    import cv2
    import numpy as np
    from src.evaluation.sdpose import load_runtime,detect_people,DEFAULT_SCORE_THRESHOLD
    from src.pose_control.qwen21_full.region_rewards import pose_metrics
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    os.environ.setdefault('CUDA_HOME','/home/shangguanrz/nihaochen/envs/qwen21')
    from src.evaluation import sdpose as adapter
    files=list(adapter.DEFAULT_SDPOSE_CKPT_DIR.rglob('*.safetensors'))+[adapter.DEFAULT_SDPOSE_CKPT_DIR/adapter.DEFAULT_YOLO_FILENAME,Path(adapter.__file__)]
    files.extend(adapter.DEFAULT_SDPOSE_OOD_ROOT/p for p in ('models/ModifiedUNet.py','models/HeatmapHead.py','pipelines/SDPose_D_Pipeline.py','configs/_base_/datasets/coco_wholebody.py'))
    identity=bind_models(args.root,'sdpose',files,{'order':'geometry','score_threshold':DEFAULT_SCORE_THRESHOLD,
        'yolo_conf':.25,'yolo_iou':.65,'input_size':[768,1024],'bbox_padding':1.25,'timestep':999,'flip_test':True,'per_image_inference_seed':42})
    runtime=load_runtime(device=args.device)
    pairs={x['pair_id']:x for x in read(args.root/'data/evaluation.json')}
    manifest=read(args.manifest);records=[]
    for i,row in enumerate(manifest['records']):
        if file_sha(row['path'])!=row['sha256']:raise ValueError('Generated image changed')
        pair=pairs[row['pair_id']];image=cv2.imread(row['path'])
        if image is None:raise ValueError('Cannot read generated image')
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
        people=detect_people(image,runtime,sort_by='area',order='geometry')
        points=scores=valid=None
        if people:
            points=np.asarray(people[0]['points'],float);scores=np.asarray(people[0]['scores'],float)
            rect=pair['target_canvas']['valid_rect'];valid=np.isfinite(points).all(1)&np.isfinite(scores)&(scores>=DEFAULT_SCORE_THRESHOLD)
            valid&=(points[:,0]>=rect[0])&(points[:,0]<rect[2])&(points[:,1]>=rect[1])&(points[:,1]<rect[3])
        metrics=pose_metrics(pair['target_points134'],pair['target_valid134'],points,valid,pair['target_canvas']['resized_hw'])
        payload={**row,'group':pair['group'],'split':pair['split'],'pose':metrics,'person_detected':bool(people),'detected_persons':len(people),
            'predicted_points134':None if points is None else [[float(v) if math.isfinite(v) else None for v in p] for p in points],
            'predicted_scores134':None if scores is None else [float(v) if math.isfinite(v) else 0. for v in scores]}
        records.append(payload);print(json.dumps({'sdpose_evaluated':i+1,'total':len(manifest['records']),'path':row['path']}),flush=True)
    write(args.output,{'schema':'bilibili_region_sdpose_evaluation_v1','backend':'sdpose','slot_order':'geometry',
        'threshold':DEFAULT_SCORE_THRESHOLD,'model_identity':identity,'label':manifest.get('label'),'records':records,'summary':aggregate(records)})


def identity_action(args):
    import cv2
    import numpy as np
    from PIL import Image
    from insightface import model_zoo
    from insightface.utils import face_align
    from src.data.canvas_contract import CanvasSpec
    models=Path('/home/shangguanrz/project/pic-edit/models/antelopev2')
    model_identity=bind_models(args.root,'identity',[models/'scrfd_10g_bnkps.onnx',models/'glintr100.onnx'],
        {'provider':'CPUExecutionProvider','detector_input':[640,640],'identity_applicability_oracle_min':.3})
    detector=model_zoo.get_model(str(models/'scrfd_10g_bnkps.onnx'),providers=['CPUExecutionProvider'])
    encoder=model_zoo.get_model(str(models/'glintr100.onnx'),providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1,input_size=(640,640));encoder.prepare(ctx_id=-1)
    def face(image):
        bgr=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2BGR)
        boxes,landmarks=detector.detect(bgr)
        if len(boxes)==0:return 0,None
        index=int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])))
        feature=encoder.get_feat(face_align.norm_crop(bgr,landmark=landmarks[index])).reshape(-1).astype(np.float32)
        feature/=max(float(np.linalg.norm(feature)),1e-12)
        return len(boxes),feature
    @lru_cache(maxsize=256)
    def original(path,canvas_json):
        with Image.open(path) as image:canvas=CanvasSpec.from_record(json.loads(canvas_json)).image(image.convert('RGB'))
        return face(canvas)
    pairs={x['pair_id']:x for x in read(args.root/'data/evaluation.json')};report=read(args.input)
    for i,row in enumerate(report['records']):
        pair=pairs[row['pair_id']]
        source_count,source=original(pair['source_rgb']['path'],json.dumps(pair['source_canvas'],sort_keys=True))
        target_count,target=original(pair['target_rgb']['path'],json.dumps(pair['target_canvas'],sort_keys=True))
        with Image.open(row['path']) as image:generated_count,generated=face(image)
        source_target=float(source@target) if source is not None and target is not None else None
        applicable=source_target is not None and source_target>=.3
        cos_src=float(generated@source) if generated is not None and source is not None else None
        cos_tgt=float(generated@target) if generated is not None and target is not None else None
        row['identity']={'applicable':applicable,'source_faces':source_count,'target_faces':target_count,'generated_faces':generated_count,
            'source_target_oracle':source_target,'id_cos_src':cos_src,'id_cos_tgt':cos_tgt,
            'identity_reward':max(0.,min(1.,cos_src)) if applicable and cos_src is not None else 0. if applicable else None}
        spec=CanvasSpec.from_record(pair['target_canvas'])
        with Image.open(pair['target_rgb']['path']) as image:gt=np.asarray(spec.image(image.convert('RGB')),np.float32)/255
        with Image.open(row['path']) as image:gen=np.asarray(image.convert('RGB'),np.float32)/255
        with np.load(pair['region']['path'],allow_pickle=False) as archive:
            background=archive['background'][0].astype(np.float32);valid=archive['valid'].astype(np.float32)
        if gen.shape!=gt.shape:raise ValueError('Generation/GT Canvas differs')
        error=((gen-gt)**2).mean(-1)
        bg_mse=float((error*background).sum()/background.sum()) if background.sum()>0 else None
        row['pixel']={'valid_mse':float((error*valid).sum()/valid.sum()),'background_mse':bg_mse,
            'background_reward':math.exp(-bg_mse/.02) if bg_mse is not None else None}
        print(json.dumps({'identity_background_evaluated':i+1,'total':len(report['records'])}),flush=True)
    report['summary']=aggregate(report['records']);report['identity_backend']='antelopev2/glintr100_cpu';report['identity_model_identity']=model_identity
    write(args.output,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['pose','identity'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite an evaluation: '+str(args.output))
    if args.action=='pose':pose_action(args)
    else:identity_action(args)


if __name__=='__main__':main()
