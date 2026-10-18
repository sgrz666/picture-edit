#!/usr/bin/env python3
"""CPU detection and manually identity-assigned scoring for six direct Hi4D cases."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.eval.eval_hi4d_separate_depth import DEFAULT,PREVIOUS,METHODS,sha,write,read_rows,verify_lock,validate_row,references,points_for
# The old root-level evaluator was removed during the earlier repository cleanup.
# Its tracked, algorithm-identical copy supplies the legacy import only inside
# this independent process; no historical source file or evaluation is changed.
if importlib.util.find_spec('eval_iper_four_arm') is None:
    from src.evaluation import joint_pilot_metrics
    sys.modules['eval_iper_four_arm']=joint_pilot_metrics
from scripts.eval.score_hi4d_native_binding import (
    detect,records,cache_file,pose_compare,face_info,cosine,background,checked_review,
)

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def reuse_reference_records(root):
    """Only original A/B detections and their earlier human assignments are reused."""
    cases=json.loads((root/'data/cases.json').read_text())
    old=json.loads((PREVIOUS/'review/manual.json').read_text())
    manual_path=root/'review/manual.json'
    manual=json.loads(manual_path.read_text()) if manual_path.exists() else {'images':{},'pairwise':{}}
    for path in {c[role]['rgb'] for c in cases for role in ('source','target')}:
        prior=old['images'][path]
        if prior['image_sha256']!=sha(path):raise ValueError('Source/target review checksum changed')
        src=cache_file(PREVIOUS,path);dest=cache_file(root,path)
        if json.loads(src.read_text())['sha256']!=sha(path):raise ValueError('Reference detection differs')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        preview=f'{sha(path)}_grey128_v2.png'
        out=root/'evaluation/detection_previews'/preview;out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PREVIOUS/'evaluation/detection_previews'/preview,out)
        manual['images'].setdefault(path,{**prior,'reused_review_source':str(PREVIOUS/'review/manual.json')})
    write(manual_path,manual)


def review_sheets(root):
    cases={c['pair_id']:c for c in json.loads((root/'data/cases.json').read_text())}
    rows=read_rows(root/'generations.jsonl')
    for pid,c in cases.items():
        rr={r['method']:r for r in rows if r['pair_id']==pid}
        if set(rr)!=set(METHODS):continue
        paths=[c['source']['rgb'],c['target']['rgb'],*[rr[m]['image'] for m in METHODS]]
        if not all(cache_file(root,p).exists() for p in paths):continue
        sheet=Image.new('RGB',(4*640,865),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,22)
        for col,(label,path) in enumerate(zip(('A：身份来源','B：目标动作','分开骨架','分开骨架＋共同深度'),paths)):
            rec=records(root,path);prev=root/'evaluation/detection_previews'/f'{sha(path)}_grey128_v2.png'
            sheet.paste(Image.open(prev).resize((640,640)),(col*640,40))
            draw.text((col*640+5,5),f'{pid} {label}',font=font,fill='black')
            lines=[]
            for p in rec['people']:
                b=p['box_at_512'];lines.append(f'P{p["index"]}: x={b[0]:.0f}..{b[2]:.0f}; 有效关节={sum(s>.3 for s in p["coco17_scores"])}')
            for f in rec['faces']['detections']:
                b=f['box_at_512'];lines.append(f'F{f["index"]}: x={b[0]:.0f}..{b[2]:.0f}; 短边={f["short_side_at_512"]:.1f}px')
            draw.multiline_text((col*640+5,687),'\n'.join(lines),font=font,fill='black',spacing=3)
        out=root/'review/sheets'/f'{pid}.jpg';out.parent.mkdir(parents=True,exist_ok=True);sheet.save(out,quality=96)
        # Identical B-coordinate upper-body windows make hand/arm occlusion
        # inspectable. This visualization is never a generation condition.
        anchors=[]
        for p in c['target']['dwpose']:
            xy,sc,_=points_for(p,1024)
            ids=np.r_[0:8,24:134]
            anchors.append(xy[ids][sc[ids]>.3])
        points=np.concatenate(anchors);low=np.floor(points.min(0)-35).clip(0,1024).astype(int)
        high=np.ceil(points.max(0)+35).clip(0,1024).astype(int)
        box=(*low,*high)
        zoom=Image.new('RGB',(4*800,850),'white');zd=ImageDraw.Draw(zoom)
        folder=root/'inputs/1024'/pid
        for col,(label,path) in enumerate(zip(('B目标','共同深度','分开骨架','加共同深度'),
                [folder/'B.png',folder/'depth_joint.png',*[rr[m]['image'] for m in METHODS]])):
            im=Image.open(path).convert('RGB').crop(box)
            im.thumbnail((800,790),Image.Resampling.LANCZOS)
            # Upscale small windows uniformly, without altering image content.
            factor=min(800/im.width,790/im.height)
            im=im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.LANCZOS)
            zoom.paste(im,(col*800+(800-im.width)//2,45+(790-im.height)//2))
            zd.text((col*800+8,5),pid+' '+label,font=font,fill='black')
        dest=root/'review/interaction_zoom'/f'{pid}.jpg';dest.parent.mkdir(parents=True,exist_ok=True);zoom.save(dest,quality=97,subsampling=0)
        write(dest.with_suffix('.json'),{'box_in_1024':list(map(int,box)),
            'selection':'union of valid target head/arms/face/hands plus35px; same window for both outputs',
            'generation_condition':False})


def score(root):
    verify_lock(root)
    cases={c['pair_id']:c for c in json.loads((root/'data/cases.json').read_text())}
    manual=json.loads((root/'review/manual.json').read_text())
    rows=read_rows(root/'generations.jsonl');jobs=json.loads((root/'data/jobs.json').read_text())
    key=lambda r:(r['pair_id'],r['method'])
    if len(rows)!=12 or {key(r) for r in rows}!={key(r) for r in jobs}:raise ValueError('Need all 12 formal outputs')
    prompts=json.loads((root/'prompts/locked_prompts.json').read_text());lock_sha=sha(root/'experiment_lock.json')
    people=[];images=[]
    for r in rows:
        validate_row(r,next(j for j in jobs if key(j)==key(r)),prompts[r['pair_id']][r['method']],references(root,r),lock_sha)
        c=cases[r['pair_id']];source=records(root,c['source']['rgb']);generated=records(root,r['image'])
        sm=checked_review(manual,c['source']['rgb'],source);gm=checked_review(manual,r['image'],generated)
        for person in sorted(gm['people'],key=lambda p:p['source_id']):
            subject=person['source_id'];idx=person.get('person_detection_index')
            pd=generated['people'][idx] if idx is not None else None
            if person['identity_status'] not in ('preserved','mixed'):pd=None
            metrics=pose_compare(pd,c['target']['dwpose'][subject])
            gf,greasons=face_info(generated,person)
            sp=next(p for p in sm['people'] if p['source_id']==subject)
            sf,sreasons=face_info(source,sp)
            raw=cosine(sf,gf);reliable=raw is not None and not (greasons+sreasons)
            people.append({'pair_id':r['pair_id'],'method':r['method'],**person,**metrics,
                'person_detected':idx is not None,'identity_a_cosine_raw':raw,
                'identity_a_cosine_reliable':raw if reliable else None,
                'face_exclusion_reasons':{'source':sreasons,'generated':greasons},
                'source_face_box':sf['box_native'] if sf else None,
                'generated_face_box':gf['box_native'] if gf else None,'image':r['image'],'image_sha256':r['image_sha256']})
        bg,bgn=background(root,c,r)
        images.append({k:r[k] for k in ('pair_id','method','image','seconds','peak_cuda_gib')}|
            {k:gm[k] for k in ('correspondence','interaction','artifacts','clothing_preservation','notes')}|
            {'background_mae':bg,'background_pixels':bgn,'detected_people':len(generated['people']),
             'face_detections':len(generated['faces']['detections'])})
        images[-1]['visually_observed_people']=gm['observed_body_count']
        images[-1]['visually_observed_extra_heads']=gm.get('extra_head_count',0)
    groups={}
    for p in people:groups.setdefault((p['pair_id'],p['source_id']),{})[p['method']]=p
    common_pose={key for key,g in groups.items() if set(g)==set(METHODS) and all(p['pose_pck_002'] is not None for p in g.values())}
    common_face={key for key,g in groups.items() if set(g)==set(METHODS) and all(p['identity_a_cosine_reliable'] is not None for p in g.values())}
    def avg(items,key):
        values=[i[key] for i in items if isinstance(i.get(key),(int,float)) and np.isfinite(i[key])]
        return float(np.mean(values)) if values else None
    summaries=[]
    for m in METHODS:
        rr=[r for r in images if r['method']==m];pp=[p for p in people if p['method']==m]
        poses=[p for p in pp if (p['pair_id'],p['source_id']) in common_pose]
        faces=[p for p in pp if (p['pair_id'],p['source_id']) in common_face]
        summaries.append({'method':m,'images':len(rr),'expected_people':12,
            'correspondence_correct':sum(r['correspondence']=='correct' for r in rr),
            'correspondence_wrong':sum(r['correspondence']=='wrong' for r in rr),
            'correspondence_uncertain':sum(r['correspondence']=='uncertain' for r in rr),
            'detected_assigned_people':sum(p['person_detected'] for p in pp),
            'pose_pck_002_common':avg(poses,'pose_pck_002'),'pose_nme_common':avg(poses,'pose_nme'),
            'pose_joint_recall_common':avg(poses,'pose_joint_recall'),'pose_common_n':len(poses),
            'pose_pck_002_identified':avg(pp,'pose_pck_002'),
            'face_cosine_common':avg(faces,'identity_a_cosine_reliable'),'face_common_n':len(faces),
            'face_raw_mean':avg(pp,'identity_a_cosine_raw'),'face_raw_n':sum(p['identity_a_cosine_raw'] is not None for p in pp),
            'background_mae':avg(rr,'background_mae'),'seconds':avg(rr,'seconds'),
            'peak_cuda_gib':max(r['peak_cuda_gib'] for r in rr),
            'artifact_images':sum(bool(r['artifacts']) for r in rr),
            'natural_interaction_images':sum(r['interaction']=='natural' for r in rr)})
    if set(manual.get('pairwise',{}))!=set(cases):raise ValueError('All six paired visual decisions required')
    counts={name:sum(v['occlusion_change']==name for v in manual['pairwise'].values())
            for name in ('improved','equal','worse','uncertain')}
    interaction_counts={name:sum(v['interaction_change']==name for v in manual['pairwise'].values())
                        for name in ('improved','equal','worse','uncertain')}
    write(root/'evaluation/per_person.json',people);write(root/'evaluation/per_image.json',images)
    write(root/'evaluation/summary.json',summaries)
    write(root/'evaluation/paired_visual_summary.json',{'occlusion_change_counts':counts,
        'interaction_change_counts':interaction_counts,'cases':manual['pairwise'],
        'note':'Visual decisions are qualitative; no automated occlusion metric or independent human panel.'})
    write(root/'evaluation/common_subsets.json',{'pose_person_instances':[list(k) for k in sorted(common_pose)],
        'face_person_instances':[list(k) for k in sorted(common_face)],'face_minimum_for_ranking':3,
        'identity_assignment':'manually identify A clothing/hair, then compare to intended official subject; never optimize matching by PCK',
        'n_definition':'person x case instances; six source identities across three reused A images',
        'face_requirement':'score >= .5, short side >=32 at512, real face and manually reliable visibility'})
    for name,items in [('per_person',people),('per_image',images),('summary',summaries)]:
        with (root/f'evaluation/{name}.csv').open('w',newline='') as f:
            fields=list(dict.fromkeys(k for row in items for k in row));w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for row in items:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    print(json.dumps({'summary':summaries,'occlusion_change_counts':counts},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--detect',action='store_true');p.add_argument('--review-sheets',action='store_true');p.add_argument('--score',action='store_true')
    a=p.parse_args()
    if a.detect:reuse_reference_records(a.output);detect(a.output)
    if a.review_sheets:review_sheets(a.output)
    if a.score:score(a.output)


if __name__=='__main__':main()
