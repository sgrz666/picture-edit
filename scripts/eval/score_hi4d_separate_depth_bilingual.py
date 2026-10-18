#!/usr/bin/env python3
"""Identity-reviewed evaluation of the four locked Hi4D bilingual arms."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.eval.eval_hi4d_separate_depth_bilingual import (
    DEFAULT,PREVIOUS,ARMS,LABELS,FONT,sha,write,read_rows,verify_lock,validate_row,references,key,target_overlay)
# Reuse exactly the previously used CPU detector, face evaluator and metrics.
from scripts.eval import score_hi4d_separate_depth as previous_score
from scripts.eval.score_hi4d_native_binding import (
    detect,records,cache_file,pose_compare,face_info,cosine,background,checked_review)

CONTRASTS={
    'language_separate':(('separate','en'),('separate','zh')),
    'language_depth':(('separate_depth','en'),('separate_depth','zh')),
    'depth_en':(('separate','en'),('separate_depth','en')),
    'depth_zh':(('separate','zh'),('separate_depth','zh')),
}


def reuse_references(root):
    cases=json.loads((root/'data/cases.json').read_text())
    old=json.loads((PREVIOUS/'review/manual.json').read_text())
    dest=root/'review/manual.json'
    manual=json.loads(dest.read_text()) if dest.exists() else {'images':{},'comparisons':{}}
    changed=not dest.exists()
    for path in {c[role]['rgb'] for c in cases for role in ('source','target')}:
        prior=old['images'][path]
        if prior['image_sha256']!=sha(path):raise ValueError('A/B original changed')
        src=cache_file(PREVIOUS,path);out=cache_file(root,path);out.parent.mkdir(parents=True,exist_ok=True)
        if json.loads(src.read_text())['sha256']!=sha(path):raise ValueError('Reference detection changed')
        shutil.copy2(src,out)
        preview=f'{sha(path)}_grey128_v2.png';p=root/'evaluation/detection_previews'/preview;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PREVIOUS/'evaluation/detection_previews'/preview,p)
        if path not in manual['images']:
            manual['images'][path]={**prior,'reused_review_source':str(PREVIOUS/'review/manual.json')}
            changed=True
    if changed:write(dest,manual)


def review_sheets(root):
    rows=read_rows(root/'generations.jsonl')
    for case in json.loads((root/'data/cases.json').read_text()):
        pid=case['pair_id'];rr={(r['method'],r['language']):r for r in rows if r['pair_id']==pid}
        if set(rr)!=set(ARMS):continue
        paths=[case['source']['rgb'],case['target']['rgb'],*[rr[a]['image'] for a in ARMS]]
        if not all(cache_file(root,p).exists() for p in paths):continue
        sheet=Image.new('RGB',(6*512,820),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,17)
        for col,(label,path) in enumerate(zip(('A','B',*[LABELS[a] for a in ARMS]),paths)):
            rec=records(root,path);preview=root/'evaluation/detection_previews'/f'{sha(path)}_grey128_v2.png'
            sheet.paste(Image.open(preview),(512*col,40));draw.text((512*col+5,5),f'{pid} {label}',font=font,fill='black')
            notes=[]
            for p in rec['people']:
                b=p['box_at_512'];nose=p['coco17_at_512'][0]
                notes.append(f'P{p["index"]} x={b[0]:.0f}..{b[2]:.0f}; nose=({nose[0]:.0f},{nose[1]:.0f})')
            for f in rec['faces']['detections']:
                b=f['box_at_512'];notes.append(f'F{f["index"]} x={b[0]:.0f}..{b[2]:.0f}; short={f["short_side_at_512"]:.1f}px')
            draw.multiline_text((512*col+5,565),'\n'.join(notes),font=font,fill='black',spacing=5)
        out=root/'review/sheets'/f'{pid}.jpg';out.parent.mkdir(parents=True,exist_ok=True);sheet.save(out,quality=97,subsampling=0)


def save_target_overlays(root):
    checks=[]
    for row in read_rows(root/'generations.jsonl'):
        merged=Image.open(root/'inputs/1024'/row['pair_id']/'combined_target.png').convert('RGB')
        image=Image.open(row['image']).convert('RGB')
        result=target_overlay(image,merged)
        path=root/'overlays'/Path(row['image']).name;path.parent.mkdir(parents=True,exist_ok=True);result.save(path)
        mask=np.asarray(merged).max(-1)>0
        assert np.array_equal(np.asarray(result)[~mask],np.asarray(image)[~mask])
        checks.append({'image':row['image'],'overlay':str(path),'overlay_sha256':sha(path),
            'target_skeleton':str(root/'inputs/1024'/row['pair_id']/'combined_target.png'),
            'target_skeleton_sha256':sha(root/'inputs/1024'/row['pair_id']/'combined_target.png'),
            'zero_background_preserved':True,'alpha':.8,'same_scene_coordinates':True})
    write(root/'checks/overlay_checks.json',checks)


def avg(items,key):
    values=[i[key] for i in items if isinstance(i.get(key),(float,int)) and np.isfinite(i[key])]
    return float(np.mean(values)) if values else None


def score(root):
    verify_lock(root)
    cases={c['pair_id']:c for c in json.loads((root/'data/cases.json').read_text())}
    manual=json.loads((root/'review/manual.json').read_text())
    rows=read_rows(root/'generations.jsonl');jobs=json.loads((root/'data/jobs.json').read_text())
    if len(rows)!=24 or {key(r) for r in rows}!={key(j) for j in jobs}:raise ValueError('24 complete unique generations required')
    prompts=json.loads((root/'prompts/locked_prompts.json').read_text());digest=sha(root/'experiment_lock.json')
    people=[];images=[]
    for r in rows:
        validate_row(r,next(j for j in jobs if key(j)==key(r)),prompts[r['pair_id']][r['method']][r['language']],references(root,r),digest)
        c=cases[r['pair_id']];source=records(root,c['source']['rgb']);gen=records(root,r['image'])
        sm=checked_review(manual,c['source']['rgb'],source);gm=checked_review(manual,r['image'],gen)
        for p in gm['people']:
            idx=p.get('person_detection_index');pd=gen['people'][idx] if idx is not None else None
            if p['identity_status'] not in ('preserved','mixed'):pd=None
            metrics=pose_compare(pd,c['target']['dwpose'][p['source_id']])
            gf,greasons=face_info(gen,p);sp=next(s for s in sm['people'] if s['source_id']==p['source_id'])
            sf,sreasons=face_info(source,sp);raw=cosine(sf,gf);valid=raw is not None and not (greasons+sreasons)
            people.append({'pair_id':r['pair_id'],'method':r['method'],'language':r['language'],**p,**metrics,
                'person_detected':idx is not None,'identity_a_cosine_raw':raw,'identity_a_cosine_reliable':raw if valid else None,
                'face_exclusion_reasons':{'source':sreasons,'generated':greasons},
                'source_face_box':sf['box_native'] if sf else None,'generated_face_box':gf['box_native'] if gf else None,
                'image':r['image'],'image_sha256':r['image_sha256']})
        bg,bgn=background(root,c,r)
        images.append({k:r[k] for k in ('pair_id','method','language','image','seconds','peak_cuda_gib')}|
            {k:gm[k] for k in ('correspondence','interaction','artifacts','clothing_preservation','notes',
                             'pose_condition_match','b_rgb_agreement','condition_conflict_notes')}|
            {'background_mae':bg,'background_pixels':bgn,'detected_people':len(gen['people']),
             'face_detections':len(gen['faces']['detections']),'observed_body_count':gm['observed_body_count'],
             'observed_extra_heads':gm.get('extra_head_count',0)})
    groups={}
    for p in people:groups.setdefault((p['pair_id'],p['source_id']),{})[(p['method'],p['language'])]=p
    common_pose={k for k,g in groups.items() if set(g)==set(ARMS) and all(p['pose_pck_002'] is not None for p in g.values())}
    common_face={k for k,g in groups.items() if set(g)==set(ARMS) and all(p['identity_a_cosine_reliable'] is not None for p in g.values())}
    summaries=[]
    for method,language in ARMS:
        rr=[r for r in images if (r['method'],r['language'])==(method,language)]
        pp=[p for p in people if (p['method'],p['language'])==(method,language)]
        poses=[p for p in pp if (p['pair_id'],p['source_id']) in common_pose]
        faces=[p for p in pp if (p['pair_id'],p['source_id']) in common_face]
        summaries.append({'method':method,'language':language,'images':len(rr),
            'correspondence_correct':sum(r['correspondence']=='correct' for r in rr),
            'correspondence_wrong':sum(r['correspondence']=='wrong' for r in rr),
            'correspondence_uncertain':sum(r['correspondence']=='uncertain' for r in rr),
            'exactly_two_people':sum(r['observed_body_count']==2 and r['observed_extra_heads']==0 for r in rr),
            'extra_people_images':sum(r['observed_body_count']>2 for r in rr),
            'artifact_images':sum(bool(r['artifacts']) for r in rr),
            'natural_interaction_images':sum(r['interaction']=='natural' for r in rr),
            'pose_pck_002_common':avg(poses,'pose_pck_002'),'pose_nme_common':avg(poses,'pose_nme'),
            'pose_joint_recall_common':avg(poses,'pose_joint_recall'),'pose_common_n':len(poses),
            'pose_pck_002_identified':avg(pp,'pose_pck_002'),
            'face_cosine_common':avg(faces,'identity_a_cosine_reliable'),'face_common_n':len(faces),
            'face_raw_mean':avg(pp,'identity_a_cosine_raw'),'face_raw_n':sum(p['identity_a_cosine_raw'] is not None for p in pp),
            'background_mae':avg(rr,'background_mae'),'seconds':avg(rr,'seconds'),
            'peak_cuda_gib':max(r['peak_cuda_gib'] for r in rr)})
    comparisons={};paired_metrics=[]
    for name,(a,b) in CONTRASTS.items():
        subsets={metric:{k for k,g in groups.items() if all(x in g and g[x][metric] is not None for x in (a,b))}
                 for metric in ('pose_pck_002','identity_a_cosine_reliable')}
        pairposes=[(g[a],g[b]) for k,g in groups.items() if k in subsets['pose_pck_002']]
        pairfaces=[(g[a],g[b]) for k,g in groups.items() if k in subsets['identity_a_cosine_reliable']]
        paired_metrics.append({'contrast':name,'from':list(a),'to':list(b),'pose_n':len(pairposes),
            'pose_pck_delta':avg([{'v':b['pose_pck_002']-a['pose_pck_002']} for a,b in pairposes],'v'),
            'face_n':len(pairfaces),'face_cosine_delta':avg([{'v':b['identity_a_cosine_reliable']-a['identity_a_cosine_reliable']} for a,b in pairfaces],'v')})
        reviewed=[]
        for pid in cases:
            v=manual['comparisons'][pid][name]
            for field in ('interaction_change','occlusion_change','correspondence_change'):
                if v[field] not in ('improved','equal','worse','uncertain'):raise ValueError('Invalid paired judgement')
            reviewed.append(v)
        comparisons[name]={field:{label:sum(v[field]==label for v in reviewed) for label in ('improved','equal','worse','uncertain')}
                           for field in ('interaction_change','occlusion_change','correspondence_change')}
    for name,data in [('per_person',people),('per_image',images),('summary',summaries),('paired_numeric',paired_metrics)]:
        write(root/f'evaluation/{name}.json',data)
        with (root/f'evaluation/{name}.csv').open('w',newline='') as f:
            fields=list(dict.fromkeys(k for r in data for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in data:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    write(root/'evaluation/common_subsets.json',{'pose_person_instances':[list(k) for k in sorted(common_pose)],
        'face_person_instances':[list(k) for k in sorted(common_face)],'face_minimum_for_ranking':3,
        'n_definition':'人物×案例，不是独立人物；六案例复用三张A，共六位源人物',
        'assignment':'按A的可辨认外观确认人物，再比较指定目标；不按最佳PCK重新配人'})
    write(root/'evaluation/paired_visual_summary.json',{'counts':comparisons,'cases':manual['comparisons'],
        'review':'AI辅助逐图观察，非独立人工盲评；不确定保留不确定'})
    print(json.dumps({'summary':summaries,'paired_counts':comparisons},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--detect',action='store_true');parser.add_argument('--review-sheets',action='store_true')
    parser.add_argument('--overlays',action='store_true');parser.add_argument('--score',action='store_true');args=parser.parse_args()
    if args.detect:reuse_references(args.output);detect(args.output)
    if args.review_sheets:review_sheets(args.output)
    if args.overlays:save_target_overlays(args.output)
    if args.score:score(args.output)


if __name__=='__main__':main()
