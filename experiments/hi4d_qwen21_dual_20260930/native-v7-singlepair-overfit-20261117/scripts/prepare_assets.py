"""Qualify real tracked faces; add new GT supervision and Source-only repair."""
from pathlib import Path
import argparse
import itertools
import json
import sys

PROJECT=Path('/home/shangguanrz/project/pic-edit')
sys.path.insert(0,str(PROJECT if PROJECT.exists() else Path(__file__).resolve().parents[4]))


def match_faces(detections,expected,canvas):
    """Unique geometry-slot assignment; never use detector order as identity."""
    import numpy as np
    detections=np.asarray(detections)
    expected=np.asarray(expected)*np.array([canvas.canvas_hw[1],canvas.canvas_hw[0]]*2)
    boxes=canvas.pixel_points(detections[:,:4].reshape(-1,2,2)).reshape(-1,4)
    quality=np.full((2,len(boxes)),-np.inf)
    for person,region in enumerate(expected):
        span=region[2:]-region[:2]
        for index,box in enumerate(boxes):
            center=(box[:2]+box[2:])/2
            overlap=np.maximum(0,np.minimum(box[2:],region[2:])-np.maximum(box[:2],region[:2])).prod()
            union=np.maximum(box[2:]-box[:2],0).prod()+np.maximum(span,0).prod()-overlap
            iou=overlap/max(union,1e-8)
            inside=np.all(center>=region[:2]-.25*span) and np.all(center<=region[2:]+.25*span)
            if detections[index,4]>=.5 and min(box[2:]-box[:2])>=24 and inside and iou>=.15:
                quality[person,index]=iou
    candidates=sorted(((sum(quality[p,i] for p,i in enumerate(order)),order)
                       for order in itertools.permutations(range(len(boxes)),2)),reverse=True)
    if not candidates or not np.isfinite(candidates[0][0]):
        raise ValueError('No qualified unique face assignment for both tracked people')
    if len(candidates)>1 and np.isfinite(candidates[1][0]) and candidates[0][0]-candidates[1][0]<.02:
        raise ValueError('Ambiguous qualified face assignment')
    return list(candidates[0][1])


def source_repair(sample):
    import torch
    from src.data.canvas_contract import CanvasSpec
    source=CanvasSpec.from_record(sample['source_canvas'])
    target=CanvasSpec.from_record(sample['target_canvas'])
    if source.metadata()!=target.metadata():
        raise ValueError('Source/Target Canvas differ; no implicit repair resize is allowed')
    local=sample['local_condition_v4'];meta=local.metadata[0]
    masks=sample['source_person_mask'].float()
    if tuple(masks.shape)!=(2,1,*source.canvas_hw):
        raise ValueError('Explicit Source person Mask grid required')
    selected=torch.stack([masks[int(local.source_indices[0,person]),0] for person in range(2)])[None]
    selected=selected*sample['valid_image_mask'][None]*local.person_valid[:,:,None,None]
    return {'schema':'native_pose_repair_v61','role':'source_repair','mask':selected.float(),
        'metadata':({'target_canvas_id':meta['target_canvas_id'],'source_rgb_sha256s':[meta['source_rgb_sha256']],
                     'target_person_ids':meta['target_person_ids'],'driver_record_sha256':meta['driver_record_sha256']},)}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as file:json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)
    from src.pose_control.qwen21_full.native_full_assets import sha256
    return {'path':str(path.resolve()),'sha256':sha256(path)}


def load_identity(path):
    from src.pose_control.qwen21_full.native_v5.losses import frozen_evaluators
    from src.pose_control.qwen21_full.native_full_assets import sha256
    return frozen_evaluators(str(path),None,device='cpu',sha=[sha256(path),None])[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from insightface import model_zoo
    from insightface.utils import face_align
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
    from src.pose_control.qwen21_full.native_v5.data import training_sidecar,write_sidecar
    from src.pose_control.qwen21_full.native_v5.contracts import SUPERVISION_SCHEMA
    from src.pose_control.qwen21_full.native_full_assets import write_record,sha256
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    root=PROJECT/'models/antelopev2'
    detector=model_zoo.get_model(str(root/'scrfd_10g_bnkps.onnx'),providers=['CPUExecutionProvider'],sess_options=options)
    recognizer=model_zoo.get_model(str(root/'glintr100.onnx'),providers=['CPUExecutionProvider'],sess_options=options)
    detector.prepare(ctx_id=-1,input_size=(640,640));recognizer.prepare(ctx_id=-1)
    evaluator=load_identity(PROJECT/'models/face_eval/identity_encoder.ts')
    dataset=NativeFullDatasetV62(args.manifest,load_target=True,data_profile='hybrid_hi4d')
    records=[];new_rows=[];source_features=None
    for index in range(len(dataset)):
        sample=dataset[index];row=dict(dataset.rows[index]);local=sample['local_condition_v4'];meta=local.metadata[0]
        v4=next(json.loads(line) for line in Path(row['v4_manifest']['path']).read_text().splitlines()
                if json.loads(line)['pair_id']==row['pair_id'])
        if Path(v4['source_rgb']['path']).parent!=Path(v4['target_rgb']['path']).parent:
            raise ValueError('Pilot repair requires same original scene/camera folder')
        pair_dir=args.output/row['pair_id'];pair_dir.mkdir()
        extracted={}
        for role,reference,canvas,expected in (
                ('source',v4['source_rgb'],CanvasSpec.from_record(sample['source_canvas']),sample['local_references_v4'].source_face_boxes[0].numpy()),
                ('target',v4['target_rgb'],CanvasSpec.from_record(sample['target_canvas']),local.face_boxes[0].numpy())):
            image=np.asarray(Image.open(reference['path']).convert('RGB'))
            bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            boxes,points=detector.detect(bgr)
            if points is None:raise ValueError('Detector did not supply five-point landmarks')
            selected=match_faces(boxes,expected,canvas)
            features=[];parity=[]
            for person,detection in enumerate(selected):
                crop=face_align.norm_crop(bgr,landmark=points[detection],image_size=112)
                onnx=recognizer.get_feat(crop).reshape(-1);onnx=onnx/max(np.linalg.norm(onnx),1e-12)
                pixels=torch.from_numpy(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1)[None].float()/127.5-1
                with torch.no_grad():feature=evaluator(pixels).flatten().numpy()
                feature=feature/max(np.linalg.norm(feature),1e-12)
                parity.append(float(onnx@feature));features.append(feature)
                cv2.imwrite(str(pair_dir/f'{role}-person{person}-aligned.png'),crop)
            extracted[role]={'selected_detections':selected,'boxes_original':boxes.tolist(),
                'landmarks5_original':points[selected].tolist(),'features':np.stack(features),'onnx_torchscript_cosine':parity}
        # Source reference slots use source_indices, target slots use stable target IDs.
        source=extracted['source']['features'][local.source_indices[0].numpy()]
        target=extracted['target']['features']
        similarity=source@target.T
        valid=[bool(similarity[p,p]>=.30 and similarity[p,p]-similarity[1-p,p]>=.05) for p in range(2)]
        if min(extracted['source']['onnx_torchscript_cosine']+extracted['target']['onnx_torchscript_cosine'])<.999:
            raise ValueError('Existing TorchScript/ONNX identity evaluator parity failed')
        raw={'role':'training_only','coordinate_space':'original_pixels','driver_rgb_sha256':meta['driver_rgb_sha256'],
             'target_person_ids':meta['target_person_ids'],'landmarks5':extracted['target']['landmarks5_original'],
             'valid':valid,'qualification':{'min_source_target_cosine':.30,'min_identity_margin':.05,'source_target_similarity':similarity.tolist()}}
        raw_ref=write_json(pair_dir/'target-five-points.json',raw)
        declaration={'availability':'available','record':raw_ref}
        supervision=training_sidecar(sample,declaration)
        record={'pair_id':row['pair_id'],'source_target_similarity':similarity.tolist(),
                'auxiliary_valid':supervision.auxiliary_valid.tolist(),
                'source':{k:v for k,v in extracted['source'].items() if k!='features'},
                'target':{k:v for k,v in extracted['target'].items() if k!='features'}}
        records.append(record)
        write_json(pair_dir/'face-qualification.json',record)
        if not supervision.auxiliary_valid.all():
            write_json(args.output/'qualification-rejected.json',{'records':records})
            raise ValueError('Both real tracked faces must qualify; no pseudo labels are generated')
        row['supervision']=write_sidecar(pair_dir/'supervision.npz',supervision,SUPERVISION_SCHEMA)
        repair=source_repair(sample)
        row['pose_repair_v61']=write_record(pair_dir/'source-repair.npz',repair,schema='native_pose_repair_v61')
        new_rows.append(row)
    manifests={}
    for index,role in enumerate(('train','observation')):
        path=args.output/(role+'.jsonl')
        with path.open('x',encoding='utf8') as file:file.write(json.dumps(new_rows[index],ensure_ascii=False)+'\n')
        manifests[role]={'path':str(path.resolve()),'sha256':sha256(path)}
        manifests[role+'_groups']=write_json(args.output/(role+'-groups.json'),{new_rows[index]['pair_id']:'pair12/highfive12/cam4'})
        checked=NativeFullDatasetV62(path,load_target=True,data_profile='hybrid_hi4d')[0]
        assert checked['training_supervision_v5'].auxiliary_valid.all() and 'pose_repair_v61' in checked
    report={'source_manifest':{'path':str(args.manifest.resolve()),'sha256':sha256(args.manifest)},
            'manifests':manifests,'qualification':records,'training_started':False}
    write_json(args.output/'prepared.json',report)
    print(json.dumps({'prepared':str(args.output/'prepared.json'),'manifests':manifests,
                      'qualification':[{'pair_id':r['pair_id'],'auxiliary_valid':r['auxiliary_valid'],'similarity':r['source_target_similarity']} for r in records]}))


if __name__=='__main__':main()
