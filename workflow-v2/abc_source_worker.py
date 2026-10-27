#!/usr/bin/env python3
"""DWPose single-person evidence and SAM2 masks on each source's own canvas."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def require_single_person(people,role):
    if len(people)!=1:raise ValueError(f'Source {role} must contain exactly one detected person; found {len(people)}')
    return people[0]


def select_person_mask(masks,quality,points,scores):
    masks=np.asarray(masks,bool);points=np.asarray(points);scores=np.asarray(scores)
    if masks.ndim!=3 or not masks.any():raise ValueError('SAM2 returned empty person masks')
    good=(scores>.5)&np.isfinite(points).all(1)
    xy=np.round(points[good]).astype(int)
    if not len(xy):raise ValueError('Source has no reliable DWPose points for mask validation')
    xy[:,0]=xy[:,0].clip(0,masks.shape[2]-1);xy[:,1]=xy[:,1].clip(0,masks.shape[1]-1)
    support=masks[:,xy[:,1],xy[:,0]].mean(1)
    rank=np.asarray(quality)+support
    rank[~masks.reshape(len(masks),-1).any(1)]=-np.inf
    chosen=int(rank.argmax())
    if support[chosen]<.45:raise ValueError(f'SAM2 mask covers too few reliable person keypoints ({support[chosen]:.3f})')
    return masks[chosen],{'mask_index':chosen,'sam_quality':float(np.asarray(quality)[chosen]),
        'pose_support':float(support[chosen]),'mask_pixels':int(masks[chosen].sum())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text(encoding='utf-8'));settings=request['settings']
    import cv2
    import torch
    from src.pose_edit import dwpose
    from src.data.canvas_contract import aspect_preserving,native_pad
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model=build_sam2(settings.get('sam2_config','configs/sam2.1/sam2.1_hiera_l.yaml'),
        settings.get('sam2_checkpoint','/home/shangguanrz/models/sam2.1/sam2.1_hiera_large.pt'),device='cuda')
    predictor=SAM2ImagePredictor(model);sessions=dwpose.load_sessions(settings);records=[]
    for item in request['sources']:
        out=Path(item['out_dir']);out.mkdir(parents=True,exist_ok=True)
        with Image.open(item['image']) as handle:image=ImageOps.exif_transpose(handle).convert('RGB')
        rgb=np.asarray(image);person=require_single_person(dwpose.detect_people(settings,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),sessions),item['role'])
        points=np.asarray(person['points']);scores=np.asarray(person['scores'])
        # Restrict SAM foreground prompts to confident body landmarks; face/hand points
        # are validation evidence rather than dozens of near-duplicate prompts.
        good=np.flatnonzero((scores[:18]>.5)&np.isfinite(points[:18]).all(1))
        prompt=points[good]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks,quality,_=predictor.predict(point_coords=prompt if len(prompt) else None,
                point_labels=np.ones(len(prompt),np.int32) if len(prompt) else None,
                box=np.asarray(person['box_xyxy']),multimask_output=True)
        mask,checks=select_person_mask(masks,quality,points[:18],scores[:18])
        original=out/'original.png';canvas_path=out/'canvas.png';mask_original=out/'mask_original.png';mask_path=out/'mask.png'
        for path in (original,canvas_path,mask_original,mask_path,out/'keypoints.npz',out/'source.json'):
            if path.exists():raise FileExistsError(f'Source outputs already exist: {path}')
        policy=settings.get('canvas_policy','aspect_preserving')
        if policy=='native_pad':spec=native_pad((image.height,image.width))
        elif policy=='aspect_preserving':spec=aspect_preserving((image.height,image.width),
            max_pixels=int(settings.get('resolution',768))**2,pad_multiple=32)
        else:raise ValueError('ABC requires native_pad or aspect_preserving CanvasSpec')
        image.save(original);spec.image(image,fill=(128,128,128)).save(canvas_path)
        mask_image=Image.fromarray(mask.astype(np.uint8)*255);mask_image.save(mask_original);spec.image(mask_image,nearest=True).save(mask_path)
        normalised=points/np.array([image.width,image.height]);canvas_points=spec.points(normalised)*np.array(spec.canvas_hw[::-1])
        np.savez_compressed(out/'keypoints.npz',points=points,scores=scores,canvas_points=canvas_points,
            box_xyxy=np.asarray(person['box_xyxy']),original_hw=np.array([image.height,image.width]))
        result={'person_id':'source_'+item['role'],'role':item['role'],'image':str(original),
            'original':str(Path(item['image']).resolve()),'canvas_image':str(canvas_path),'canvas':spec.metadata(),
            'mask':str(mask_path),'mask_original':str(mask_original),'keypoints_npz':str(out/'keypoints.npz'),
            'detection':{'person_count':1,'box_xyxy':person['box_xyxy'],'model':'DWPose-ONNX-CPU'},
            'mask_checks':checks,'mask_model':'SAM2.1-Hiera-Large'}
        (out/'source.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        records.append(result)
    Path(request['manifest']).write_text(json.dumps({'schema':'abc_source_evidence_v1','sources':records},
        ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
