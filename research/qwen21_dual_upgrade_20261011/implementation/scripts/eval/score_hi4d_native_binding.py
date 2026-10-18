#!/usr/bin/env python3
"""Evaluate the locked Hi4D experiment after explicitly reviewing A identities.

Detection never assigns identities by target position. Manual decisions are
required for every source and generated image before final aggregation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.eval.eval_hi4d_native_binding import DEFAULT, METHODS, MAPPINGS, OPENPOSE_TO_COCO, sha, write, read_rows, points_for
from scripts.eval.score_qwen21_native_value import FaceMetric
from src.data.letterbox import Letterbox
from src.evaluation.joint_pilot_metrics import PoseMetric
from src.evaluation.dwpose_onnx.onnxdet import inference_detector
from src.evaluation.dwpose_onnx.onnxpose import inference_pose

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def cache_file(root, path):
    return root/'evaluation/detections'/f'{sha(path)}_grey128_v2.json'


def detect(root):
    pose, face = PoseMetric(), FaceMetric()
    cases = json.loads((root/'data/cases.json').read_text())
    paths = sorted({c[k]['rgb'] for c in cases for k in ('source','target')} |
                   {r['image'] for r in read_rows(root/'generations.jsonl')})
    for path in paths:
        dest = cache_file(root,path)
        if dest.exists():
            if json.loads(dest.read_text())['sha256'] != sha(path):raise ValueError('Detection cache changed')
            continue
        im=Image.open(path).convert('RGB');small=Letterbox.from_size(im.size,512).image(im,fill=(128,128,128))
        bgr=cv2.cvtColor(np.asarray(small),cv2.COLOR_RGB2BGR)
        boxes=inference_detector(pose.det,bgr)
        people=[]
        if len(boxes):
            points,scores=inference_pose(pose.pose,boxes,bgr)
            for i,(box,xy,score) in enumerate(zip(boxes,points,scores)):
                people.append({'index':i,'box_at_512':box.tolist(),
                               'coco17_at_512':xy[:17].tolist(),'coco17_scores':score[:17].tolist()})
        rec={'path':path,'sha256':sha(path),'image_size':list(im.size),'people':people,'faces':face.detect(path)}
        write(dest,rec)
        review=small.copy();draw=ImageDraw.Draw(review);font=ImageFont.truetype(FONT,16)
        for person in people:
            b=person['box_at_512'];draw.rectangle(b,outline=(255,50,50),width=2)
            draw.text((b[0],max(0,b[1]-20)),f'P{person["index"]}',fill=(255,50,50),font=font)
            for xy,s in zip(person['coco17_at_512'],person['coco17_scores']):
                if s>.3:draw.ellipse((xy[0]-1,xy[1]-1,xy[0]+1,xy[1]+1),fill='yellow')
        for f in rec['faces']['detections']:
            b=f['box_at_512'];draw.rectangle(b,outline=(30,230,255),width=2)
            draw.text((b[0],b[3]),f'F{f["index"]}',fill=(30,230,255),font=font)
        out=root/'evaluation/detection_previews'/f'{sha(path)}_grey128_v2.png';out.parent.mkdir(parents=True,exist_ok=True);review.save(out)
        print(json.dumps({'detected':path,'people':len(people),'faces':len(rec['faces']['detections'])}),flush=True)


def records(root,path):
    rec=json.loads(cache_file(root,path).read_text())
    if rec['sha256']!=sha(path):raise ValueError('Detection/image mismatch')
    return rec


def review_sheets(root):
    """Readable person/face indices for human review; no identity inference."""
    cases={c['pair_id']:c for c in json.loads((root/'data/cases.json').read_text())}
    groups=defaultdict(dict)
    for r in read_rows(root/'generations.jsonl'):groups[(r['resolution'],r['pair_id'],r['mapping'])][r['method']]=r
    for (res,pid,mapping),rows in groups.items():
        if set(rows)!=set(METHODS):continue
        case=cases[pid];paths=[case['source']['rgb'],case['target']['rgb'],*[rows[m]['image'] for m in METHODS]]
        if not all(cache_file(root,p).exists() for p in paths):continue
        sheet=Image.new('RGB',(5*512,710),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,16)
        for col,(label,path) in enumerate(zip(('A','B','RGB','合并骨架','分开骨架'),paths)):
            rec=records(root,path);prev=root/'evaluation/detection_previews'/f'{sha(path)}_grey128_v2.png'
            sheet.paste(Image.open(prev),(col*512,30));draw.text((col*512+5,5),f'{pid} {res} {mapping} {label}',font=font,fill='black')
            text=[]
            for p in rec['people']:
                b=p['box_at_512'];text.append(f'P{p["index"]}: x={b[0]:.0f}..{b[2]:.0f}; joints={sum(s>.3 for s in p["coco17_scores"])}')
            for f in rec['faces']['detections']:
                b=f['box_at_512'];text.append(f'F{f["index"]}: x={b[0]:.0f}..{b[2]:.0f}; short={f["short_side_at_512"]:.1f}; q={f["score"]:.2f}')
            draw.multiline_text((col*512+5,545),'\n'.join(text),font=font,fill='black',spacing=4)
        out=root/'review/sheets'/f'{pid}_{res}_{mapping}.png';out.parent.mkdir(parents=True,exist_ok=True);sheet.save(out)


def pose_compare(person, target_path):
    """Same anatomical COCO17 order, normalized by the unpadded frame."""
    target,scores,transform=points_for(target_path,512)
    target=target[np.asarray(OPENPOSE_TO_COCO)];scores=scores[np.asarray(OPENPOSE_TO_COCO)]
    visible=scores>.3;total=int(visible.sum())
    if person is None:return {'pose_pck_002':None,'pose_nme':None,'pose_joint_recall':0.,'pose_visible':total}
    xy=np.asarray(person['coco17_at_512']);gs=np.asarray(person['coco17_scores'])
    matched=visible&(gs>.3);x0,y0,x1,y1=transform.rect
    d=np.linalg.norm(xy[matched]-target[matched],axis=-1)/np.hypot(x1-x0,y1-y0)
    return {'pose_pck_002':float((d<.02).sum()/total) if total else None,
            'pose_nme':float(d.mean()) if len(d) else None,
            'pose_joint_recall':float(matched.sum()/total) if total else None,'pose_visible':total}


def face_info(rec,person):
    index=person.get('face_detection_index')
    if index is None:return None,['no_assigned_actual_face']
    if not isinstance(index,int) or not 0<=index<len(rec['faces']['detections']):raise ValueError('Invalid face index')
    f=rec['faces']['detections'][index]
    reasons=list(f['automatic_exclusion_reasons'])
    if not person.get('face_real'):reasons.append('manual_false_or_uncertain_detection')
    if not person.get('face_identity_usable'):reasons.append('manual_visibility_or_identity_uncertain')
    return f,reasons


def cosine(a,b):
    return float(np.dot(a['embedding'],b['embedding'])) if a is not None and b is not None else None


def background(root,case,row):
    source=Image.open(case['source']['rgb']).convert('RGB')
    transform=Letterbox.from_size(source.size,512)
    a=np.asarray(transform.image(source,fill=(128,128,128))).astype(float)
    out=np.asarray(Image.open(row['image']).convert('RGB').resize((512,512),Image.Resampling.BILINEAR)).astype(float)
    masks=np.zeros((512,512),np.uint8)
    for frame in ('source','target'):
        for path in case[frame]['masks']:
            im=Image.open(path).convert('L')
            masks|=(np.asarray(Letterbox.from_size(im.size,512).image(im,nearest=True,fill=0))>0).astype(np.uint8)
    masks=cv2.dilate(masks,np.ones((15,15),np.uint8))
    valid=np.zeros((512,512),bool);x0,y0,x1,y1=transform.rect;valid[y0:y1,x0:x1]=True
    valid&=masks==0
    return float(np.abs(a-out)[valid].mean()),int(valid.sum())


def checked_review(manual,path,rec):
    r=manual['images'].get(path)
    if not r or not r.get('reviewed') or r.get('image_sha256')!=sha(path):raise ValueError(f'Missing/changed review: {path}')
    if sorted(p['source_id'] for p in r['people'])!=[0,1]:raise ValueError('Review must declare both source identities')
    ids=[p.get('person_detection_index') for p in r['people'] if p.get('person_detection_index') is not None]
    if len(set(ids))!=len(ids) or any(not 0<=i<len(rec['people']) for i in ids):raise ValueError('Duplicate/invalid person detection')
    if r['correspondence'] not in ('correct','wrong','uncertain'):raise ValueError('Invalid correspondence')
    return r


def score(root):
    cases={c['pair_id']:c for c in json.loads((root/'data/cases.json').read_text())}
    manual=json.loads((root/'review/manual.json').read_text())
    generations=read_rows(root/'generations.jsonl')
    expected=json.loads((root/'data/jobs.json').read_text())
    keys=lambda r:(r['pair_id'],r['resolution'],r['mapping'],r['method'])
    if len(generations)!=48 or {keys(r) for r in generations}!={keys(r) for r in expected}:raise ValueError('Incomplete formal results')
    people_rows=[];rows=[]
    for row in generations:
        case=cases[row['pair_id']];sr=records(root,case['source']['rgb']);gr=records(root,row['image'])
        sm=checked_review(manual,case['source']['rgb'],sr);gm=checked_review(manual,row['image'],gr)
        assigned=MAPPINGS[row['mapping']];partial=[]
        for person in sorted(gm['people'],key=lambda p:p['source_id']):
            a=person['source_id'];idx=person.get('person_detection_index')
            pd=gr['people'][idx] if idx is not None else None
            # Only visually recognized A identities enter identity-aware pose scoring.
            if person['identity_status'] not in ('preserved','mixed'):pd=None
            metrics=pose_compare(pd,case['target']['dwpose'][assigned[a]])
            gf,greasons=face_info(gr,person)
            sp=next(p for p in sm['people'] if p['source_id']==a);sf,sreasons=face_info(sr,sp)
            raw=cosine(sf,gf);reliable=raw is not None and not (greasons+sreasons)
            p={'pair_id':row['pair_id'],'resolution':row['resolution'],'mapping':row['mapping'],'method':row['method'],
               'source_id':a,'intended_target_id':assigned[a],**person,**metrics,
               'identity_a_cosine_raw':raw,'identity_a_cosine_reliable':raw if reliable else None,
               'identity_exclusion_reasons':{'source':sreasons,'generated':greasons},
               'source_face_box':sf['box_native'] if sf else None,'generated_face_box':gf['box_native'] if gf else None,
               'image':row['image'],'image_sha256':row['image_sha256']}
            people_rows.append(p);partial.append(p)
        bg,bgn=background(root,case,row)
        valid=[p for p in partial if p['pose_pck_002'] is not None]
        rows.append({k:row[k] for k in ('pair_id','resolution','mapping','method','image','seconds','peak_cuda_gib')}|
                    {'correspondence':gm['correspondence'],'interaction':gm['interaction'],'artifacts':gm['artifacts'],
                     'clothing_preservation':gm['clothing_preservation'],'notes':gm['notes'],
                     'pose_pck_002_identified':float(np.mean([p['pose_pck_002'] for p in valid])) if valid else None,
                     'pose_identified_people':len(valid),'background_mae':bg,'background_pixels':bgn})
    # A fair pose subset needs the same source-person instance identifiable in all routes.
    buckets=defaultdict(dict)
    for p in people_rows:buckets[(p['resolution'],p['pair_id'],p['mapping'],p['source_id'])][p['method']]=p
    common_pose={k for k,v in buckets.items() if set(v)==set(METHODS) and all(p['pose_pck_002'] is not None for p in v.values())}
    common_face={k for k,v in buckets.items() if set(v)==set(METHODS) and all(p['identity_a_cosine_reliable'] is not None for p in v.values())}
    summaries=[]
    for resolution in (512,1024):
        for method in METHODS:
            rr=[r for r in rows if r['resolution']==resolution and r['method']==method]
            pp=[p for p in people_rows if p['resolution']==resolution and p['method']==method]
            common=[p for p in pp if (resolution,p['pair_id'],p['mapping'],p['source_id']) in common_pose]
            faces=[p for p in pp if (resolution,p['pair_id'],p['mapping'],p['source_id']) in common_face]
            def avg(items,key):
                vv=[p[key] for p in items if isinstance(p.get(key),(float,int)) and math.isfinite(p[key])]
                return float(np.mean(vv)) if vv else None
            summaries.append({'resolution':resolution,'method':method,'images':len(rr),
              'correspondence_correct':sum(r['correspondence']=='correct' for r in rr),
              'correspondence_wrong':sum(r['correspondence']=='wrong' for r in rr),
              'correspondence_uncertain':sum(r['correspondence']=='uncertain' for r in rr),
              'identifiable_people':sum(p['identity_status'] in ('preserved','mixed') for p in pp),
              'expected_people':len(pp),'pose_pck_002_common':avg(common,'pose_pck_002'),
              'pose_nme_common':avg(common,'pose_nme'),'pose_common_n':len(common),
              'pose_pck_002_identified':avg(pp,'pose_pck_002'),'pose_identified_n':sum(p['pose_pck_002'] is not None for p in pp),
              'face_cosine_common':avg(faces,'identity_a_cosine_reliable'),'face_common_n':len(faces),
              'face_raw_mean':avg(pp,'identity_a_cosine_raw'),'face_raw_n':sum(p['identity_a_cosine_raw'] is not None for p in pp),
              'background_mae':avg(rr,'background_mae'),'seconds':avg(rr,'seconds'),'peak_cuda_gib':max(r['peak_cuda_gib'] for r in rr),
              'artifact_images':sum(bool(r['artifacts']) for r in rr),
              'correct_without_recorded_artifacts':sum(r['correspondence']=='correct' and not r['artifacts'] for r in rr),
              'cases_both_assignments_correct':sum(all(v['correspondence']=='correct' for v in rr if v['pair_id']==pid)
                  for pid in {v['pair_id'] for v in rr}),
              'natural_interaction_images':sum(r['interaction']=='natural' for r in rr)})
    write(root/'evaluation/per_person.json',people_rows);write(root/'evaluation/per_image.json',rows)
    write(root/'evaluation/summary.json',summaries)
    write(root/'evaluation/common_subsets.json',{'pose_person_instances':[list(k) for k in sorted(common_pose)],
          'face_person_instances':[list(k) for k in sorted(common_face)],'face_minimum_for_ranking':3,
          'pose_assignment':'manual source identity, then intended target; never optimal matching by position',
          'note':'Common subsets consist of person/case/mapping instances, not independent identities.'})
    for name,items in [('per_person',people_rows),('per_image',rows),('summary',summaries)]:
        with (root/f'evaluation/{name}.csv').open('w',newline='') as f:
            fields=list(dict.fromkeys(k for item in items for k in item));w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for item in items:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in item.items()})
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--detect',action='store_true');p.add_argument('--score',action='store_true');p.add_argument('--review-sheets',action='store_true');args=p.parse_args()
    if args.detect:detect(args.output)
    if args.review_sheets:review_sheets(args.output)
    if args.score:score(args.output)


if __name__=='__main__':main()
