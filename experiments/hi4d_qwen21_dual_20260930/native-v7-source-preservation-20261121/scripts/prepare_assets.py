"""Create new Source-only face anchors/background supports and bounded configs."""
from pathlib import Path
from dataclasses import replace
import argparse
import importlib.util
import json
import sys

PROJECT=Path('/home/shangguanrz/project/pic-edit')
sys.path.insert(0,str(PROJECT if PROJECT.exists() else Path(__file__).resolve().parents[4]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--budget',type=int,default=12)
    args=parser.parse_args()
    if not 1<=args.budget<=40:parser.error('Bounded pilot budget must be 1..40')
    root=args.root.resolve();data=root/'data/assets-v1';configs=root/'configs/pilot-v1'
    if data.exists() or configs.exists():raise FileExistsError('Asset/config outputs must be new')
    data.mkdir(parents=True);configs.mkdir(parents=True)
    import torch
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from insightface import model_zoo
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
    from src.pose_control.qwen21_full.native_v62.training.io import verify,file_sha,write_json
    from src.pose_control.qwen21_full.native_v5.losses import frozen_evaluators
    from src.pose_control.qwen21_full.face_loss_alignment import align_face_rgb
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7,DataPlanV7,stage_preset_v7
    from src.pose_control.qwen21_full.native_v7.preservation_assets import (
        INDEX_SCHEMA,BACKGROUND_IMPLEMENTATION,sample_binding,background_support,write_record)
    from src.pose_control.qwen21_full.native_v7.preservation import PreservationConfig
    torch.set_num_threads(2)
    def ref(path):return {'path':str(path.resolve()),'sha256':file_sha(path)}
    baseline=args.baseline_root
    cfg_path=baseline/'configs/pilot300-native-fullgpu-v2/S3-preflight.json'
    previous=TrainingConfigV7.from_dict(json.loads(cfg_path.read_text(encoding='utf8')))
    evaluator=verify(previous.evaluators['identity'])
    model,_=frozen_evaluators(evaluator['path'],None,device='cpu',sha=[evaluator['sha256'],None])
    detector_path=PROJECT/'models/antelopev2/scrfd_10g_bnkps.onnx'
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    detector=model_zoo.get_model(str(detector_path),providers=['CPUExecutionProvider'],sess_options=options)
    detector.prepare(ctx_id=-1,input_size=(640,640),det_thresh=.5)
    matcher_path=baseline/'scripts/prepare_assets.py'
    spec=importlib.util.spec_from_file_location('previous_face_qualification',matcher_path)
    matcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(matcher)
    entries={};records=[]
    for source in previous.sources:
        dataset=NativeFullDatasetV62(verify(source.manifest)['path'],load_target=False,data_profile=source.data_profile)
        for i in range(len(dataset)):
            sample=dataset[i];row=dataset.rows[i];canvas=CanvasSpec.from_record(sample['source_canvas'])
            v4_ref=verify(row['v4_manifest'])
            v4=next(value for line in Path(v4_ref['path']).read_text(encoding='utf8').splitlines()
                if (value:=json.loads(line))['pair_id']==row['pair_id'])
            image_ref=verify(v4['source_rgb'])
            pixels=np.asarray(Image.open(image_ref['path']).convert('RGB'))
            boxes,points=detector.detect(cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
            if points is None:raise ValueError('No real Source five-point detections')
            selected=matcher.match_faces(boxes,sample['local_references_v4'].source_face_boxes[0].numpy(),canvas)
            landmarks=canvas.pixel_points(np.asarray(points)[selected])
            source_pixels=((sample['src_image'].float()+1)*.5)[None].expand(2,-1,-1,-1)
            with torch.no_grad():
                features=model(align_face_rgb(source_pixels,torch.tensor(landmarks).float())*2-1)
                features=features[-1] if isinstance(features,(tuple,list)) else features
                features=torch.nn.functional.normalize(features.flatten(1).float(),dim=-1)
            indices=sample['local_condition_v4'].source_indices[0]
            features=features[indices].contiguous()
            valid=sample['local_condition_v4'].person_valid[0].cpu().clone()
            features[~valid]=0
            metadata={**sample_binding(sample,evaluator['sha256']), 'background_margin_px':16,
                'background_implementation':BACKGROUND_IMPLEMENTATION,
                'source_qualification':{'source_image':image_ref,'detector':ref(detector_path),
                    'matcher':ref(matcher_path),'selected':selected,'boxes_original':boxes.tolist(),
                    'landmarks5_original':np.asarray(points)[selected].tolist(),
                    'feature_implementation':ref(Path(__file__))}}
            support=background_support(sample,16)
            record=write_record(data/(source.source_id+'-'+row['pair_id']+'.pt'),features,valid,support,metadata)
            entries[source.source_id+'\0'+row['pair_id']]=record
            records.append({'source_id':source.source_id,'pair_id':row['pair_id'],'record':record,
                'source_faces':int(valid.sum()),'background_token_area':float(support.sum())})
    index_ref=write_json(data/'index.json',{'schema':INDEX_SCHEMA,'evaluator':evaluator,'entries':entries})
    plans={}
    for name,preserve in (('control',None),('source',PreservationConfig(source_identity=.2)),
                          ('source-background',PreservationConfig(source_identity=.2,background=.1))):
        stage=stage_preset_v7('S3',budget=args.budget)
        stage=replace(stage,routes=tuple(replace(route,loss=replace(route.loss,auxiliary_every=1,auxiliary_sigma_max=.5))
            for route in stage.routes))
        current=replace(previous,stage=stage,gradient_report=None,output_root=str(root/'output/training'),
            preservation=preserve,preservation_index=index_ref if preserve else None,
            save_every_updates=args.budget,probe_every_updates=0,generation_every_updates=0,
            continuous_stage_authorization={'source':'direct_user_message',
                'text':'行,那你根据我们实验情况进行相应的架构或训练的改进吧🤔'})
        current.require_ready()
        plan=DataPlanV7(current)
        write_json(configs/(name+'.json'),current.record())
        plans[name]={'config':ref(configs/(name+'.json')),'data':write_json(configs/(name+'-data.json'),plan.report())}
    report={'schema':'native_v7_preservation_prepared_v1','records':records,'index':index_ref,
        'parent':previous.initialization,'source_config':ref(cfg_path),'budget':args.budget,
        'configs':plans,'target_rgb_used_for_source_features':False,'source_features_recomputed_with_pinned_evaluator':True}
    write_json(data/'prepared.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
