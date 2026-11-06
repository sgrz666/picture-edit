"""Reproducible small-sample selection and before/after hand-weight evidence."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=next(p for p in (Path.cwd(),*Path(__file__).resolve().parents) if (p/'src/data/canvas_contract.py').exists())
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from src.data.canvas_contract import CanvasSpec
from src.pose_control.qwen21_full.native_v5.spatial_masks import regional_maps,visible_hand_region,_hand_region
from src.pose_control.qwen21_full.native_v5.spatial_weights import read_spatial_components,spatial_token_weights
from scripts.data.build_native_spatial_weights import sha,publish,save_image,heatmap,digest


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def prepare(args):
    source=[json.loads(line) for line in args.frames.read_text().splitlines() if line.strip()]
    lookup={r['frame_id']:r for r in source}
    pilot=load(args.pilot_index)
    ids=pilot['config']['selected_frame_ids'];selected=[lookup[fid] for fid in ids]
    negatives=[];used=set(ids)
    for state in ('occluded','out_of_frame'):
        for side in ('hand_a','hand_b'):
            row=next(r for r in source if r[side+'_status']==state and r['frame_id'] not in used)
            selected.append(row);used.add(row['frame_id'])
            negatives.append(dict(frame_id=row['frame_id'],side=side,state=state))
    args.output_root.mkdir(parents=True,exist_ok=False)
    directory=args.output_root/'input';directory.mkdir()
    frames=directory/'frames24.jsonl'
    with frames.open('x',encoding='utf8') as stream:
        for row in selected:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    report=dict(schema='spatial_hand_ablation_selection_v1',source=ref(args.frames),pilot_index=ref(args.pilot_index),
                baseline_index=ref(args.baseline_index),frames=ref(frames),
                original_pilot_ids=ids,negative_cases=negatives,selected_ids=[r['frame_id'] for r in selected],
                annotation_summary=dict(frames=len(source),legacy_enabled_frames=sum(any(r.get('loss_mask',{}).get(k,False) and r.get('hand_raw_pose_usable',{}).get(k,False) for k in ('hand_a','hand_b')) for r in source),
                                        hand_a_present=sum(r['hand_a_present'] for r in source),hand_b_present=sum(r['hand_b_present'] for r in source)))
    publish(args.output_root/'selection.json',report)
    print(json.dumps(report),flush=True)


def tensor_maps(entry):
    m=entry['metadata']
    maps=read_spatial_components(entry['file'],gt_rgb_sha256=m['gt_rgb_sha256'],target_canvas_id=m['target_canvas_id'])
    if maps['metadata']!=m:raise ValueError('Index metadata differs from pinned NPZ')
    return maps


def token_weights(maps,spec,hand=None):
    parts={k:maps[k] for k in ('body','joint','hand')}
    if hand is not None:parts['hand']=torch.from_numpy(hand)[None]
    return spatial_token_weights(parts,maps['valid'],spec.token_hw,(.5,.5,1.))


def compare(args):
    torch.set_num_threads(4)
    selection=load(args.selection);old=load(args.baseline_index);new=load(args.candidate_index)
    if new.get('config',{}).get('hand_policy')!='visible_region_v2':
        raise ValueError('Candidate policy must be visible_region_v2')
    source=[json.loads(line) for line in Path(selection['frames']['path']).read_text().splitlines() if line.strip()]
    by_id={e['metadata']['frame_id']:e for e in new['entries'].values()}
    args.output_root.mkdir(parents=True,exist_ok=False)
    cases=[];thumbnails=[];reasons=Counter()
    for frame in source:
        fid=frame['frame_id'];entry=by_id[fid];key=entry['metadata']['gt_rgb_sha256']+':'+entry['metadata']['target_canvas_id']
        prior=old['entries'][key];a=tensor_maps(prior);b=tensor_maps(entry)
        if entry['metadata'].get('hand_policy')!='visible_region_v2':
            raise ValueError('Candidate entry policy differs')
        if prior['metadata'].get('hand_policy') not in ('legacy_strict','existing_loss_mask_and_hand_raw_pose_usable'):
            raise ValueError('Baseline policy must be legacy')
        if any(item['metadata'].get('source_frame_sha256')!=digest(frame) for item in (prior,entry)):
            raise ValueError('Ablation source annotation binding differs')
        spec=CanvasSpec.from_record(entry['metadata']['canvas'])
        for name in ('body','joint','valid'):
            if not torch.equal(a[name],b[name]):raise ValueError('Non-hand component changed: '+name+'/'+fid)
        with Image.open(Path(entry['file']['path']).parent/'body_mask.png') as im:body_original=(np.array(im)>127).astype(np.uint8)
        actual_maps,actual_status=regional_maps(frame,body_original,spec,hand_policy='visible_region_v2')
        legacy_maps,_=regional_maps(frame,body_original,spec,hand_policy='legacy_strict')
        if actual_status!=entry['metadata'].get('region_status'):
            raise ValueError('Candidate region status differs from actual maps')
        for name in ('body','joint','hand','valid'):
            if not np.array_equal(actual_maps[name].astype(np.float16),b[name][0].numpy().astype(np.float16)):
                raise ValueError('Candidate cached '+name+' differs from claimed policy')
            if not np.array_equal(legacy_maps[name].astype(np.float16),a[name][0].numpy().astype(np.float16)):
                raise ValueError('Baseline cached '+name+' differs from claimed policy')
        zero=np.zeros(spec.canvas_hw,np.float32)
        off=token_weights(b,spec,zero);old_weights=token_weights(a,spec);new_weights=token_weights(b,spec)
        hand_records={}
        for side in ('hand_a','hand_b'):
            single,status=visible_hand_region(frame,side,spec,actual_maps['body'])
            old_single,_=_hand_region(frame,side,spec)
            single=single.astype(np.float16).astype(np.float32);old_single=old_single.astype(np.float16).astype(np.float32)
            old_delta=token_weights(a,spec,old_single)-off;delta=token_weights(b,spec,single)-off
            active=F.interpolate(torch.from_numpy(single)[None,None],size=spec.token_hw,mode='area').flatten(1)>1e-5
            evidence=dict(old_enabled=bool(old_single.any()),new_enabled=status['enabled'],coarse=status['coarse'],fine=status['fine'],
                          old_isolated_latent_increment_max=float(old_delta.max()),new_isolated_latent_increment_max=float(delta.max()),
                          new_actual_latent_weight_peak=float(new_weights[active].max()) if active.any() else None,
                          body_joint_baseline_peak=float(off[active].max()) if active.any() else None)
            if status['enabled'] and float(delta.max())<=0:raise ValueError('Hand vanished after pooling: '+fid+'/'+side)
            if not status['coarse']['enabled']:reasons[status['coarse']['reason']]+=1
            hand_records[side]=evidence
        case=dict(frame_id=fid,hands=hand_records,non_hand_components_identical=True)
        cases.append(case)
        with Image.open(frame['rgb_path']) as im:rgb=spec.image(im.convert('RGB'))
        overlay=np.array(rgb).astype(np.float32);alpha=b['hand'][0].numpy()[...,None]*.6
        overlay=overlay*(1-alpha)+np.array([255,65,100])*alpha
        h,w=spec.canvas_hw
        a_pixel=1+.5*a['body'][0].numpy()+.5*a['joint'][0].numpy()+a['hand'][0].numpy()
        b_pixel=1+.5*b['body'][0].numpy()+.5*b['joint'][0].numpy()+b['hand'][0].numpy()
        panels=[rgb,heatmap(a['hand'][0].numpy()),heatmap(b['hand'][0].numpy()),Image.fromarray(np.uint8(overlay)),
                heatmap(np.minimum(a_pixel,2.5)*a['valid'][0].numpy(),0,2.5),heatmap(np.minimum(b_pixel,2.5)*b['valid'][0].numpy(),0,2.5),
                heatmap(old_weights.reshape(spec.token_hw).numpy(),0,2.5).resize((w,h),Image.Resampling.NEAREST),
                heatmap(new_weights.reshape(spec.token_hw).numpy(),0,2.5).resize((w,h),Image.Resampling.NEAREST)]
        strip=Image.new('RGB',(w*8,h+44),'white');draw=ImageDraw.Draw(strip)
        labels=('GT','Old hand','New hand','New overlay','Old weight','New weight','Old latent','New latent')
        for i,(im,label) in enumerate(zip(panels,labels)):
            strip.paste(im,(i*w,44));draw.text((i*w+5,5),label,fill='black')
        draw.text((5,24),fid+' | hand_a delta %.3f -> %.3f | hand_b delta %.3f -> %.3f'%(
                  hand_records['hand_a']['old_isolated_latent_increment_max'],hand_records['hand_a']['new_isolated_latent_increment_max'],
                  hand_records['hand_b']['old_isolated_latent_increment_max'],hand_records['hand_b']['new_isolated_latent_increment_max']),fill='black')
        path=args.output_root/(fid+'.jpg');save_image(path,strip,format='JPEG');case['preview']=ref(path)
        thumb=strip.copy();thumb.thumbnail((1600,450));thumbnails.append(thumb)
        if fid=='BV14NHK6SE9d_000286':
            # Keep whole-hand context in an exportable close-up, instead of relying
            # on the contact-sheet thumbnail to assess a few latent cells.
            points=np.asarray(frame['candidate'])[113:134]
            mapped=spec.pixel_points(points)
            x0=max(0,int(mapped[:,0].min())-12);x1=min(w,int(mapped[:,0].max())+13)
            y0=max(0,int(mapped[:,1].min())-12);y1=min(h,int(mapped[:,1].max())+13)
            zoom=Image.new('RGB',((x1-x0)*5*5,(y1-y0)*5+28),'white');draw=ImageDraw.Draw(zoom)
            for i,(im,label) in enumerate(zip((panels[0],panels[1],panels[2],panels[6],panels[7]),('GT left hand','Old hand','New hand','Old latent','New latent'))):
                cut=im.crop((x0,y0,x1,y1)).resize(((x1-x0)*5,(y1-y0)*5),Image.Resampling.NEAREST)
                zoom.paste(cut,(i*(x1-x0)*5,28));draw.text((i*(x1-x0)*5+4,7),label,fill='black')
            save_image(args.output_root/'missing_left_hand_zoom.jpg',zoom,format='JPEG')
    for item in selection['negative_cases']:
        case=next(c for c in cases if c['frame_id']==item['frame_id'])
        if case['hands'][item['side']]['new_enabled']:raise ValueError('Visibility negative was upweighted')
    prior_ids=set(selection['original_pilot_ids']);control=[c for c in cases if c['frame_id'] in prior_ids]
    stats=lambda subset:dict(frames=len(subset),old_enabled_frames=sum(any(h['old_enabled'] for h in c['hands'].values()) for c in subset),
                            new_enabled_frames=sum(any(h['new_enabled'] for h in c['hands'].values()) for c in subset),
                            old_enabled_hands=sum(h['old_enabled'] for c in subset for h in c['hands'].values()),
                            new_enabled_hands=sum(h['new_enabled'] for c in subset for h in c['hands'].values()),
                            coarse_enabled_hands=sum(h['coarse']['enabled'] for c in subset for h in c['hands'].values()),
                            fine_enabled_hands=sum(h['fine']['enabled'] for c in subset for h in c['hands'].values()))
    report=dict(schema='spatial_hand_region_ablation_v2',baseline=ref(args.baseline_index),candidate=ref(args.candidate_index),
                original_pilot=stats(control),all_samples=stats(cases),negative_cases_passed=len(selection['negative_cases']),
                non_hand_components_identical=True,coarse_rejection_reasons=dict(reasons),cases=cases,
                training_started=False,full_dataset_rebuilt=False)
    publish(args.output_root/'comparison.json',report)
    sheet=Image.new('RGB',(max(im.width for im in thumbnails),sum(im.height+18 for im in thumbnails)),'white');draw=ImageDraw.Draw(sheet);top=0
    for thumb,case in zip(thumbnails,cases):
        draw.text((4,top+2),case['frame_id'],fill='black');sheet.paste(thumb,(0,top+18));top+=thumb.height+18
    save_image(args.output_root/'contact_sheet.jpg',sheet,format='JPEG')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--frames',type=Path,required=True);a.add_argument('--pilot-index',type=Path,required=True)
    a.add_argument('--baseline-index',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True)
    a=sub.add_parser('compare');a.add_argument('--selection',type=Path,required=True);a.add_argument('--baseline-index',type=Path,required=True)
    a.add_argument('--candidate-index',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();(prepare if args.action=='prepare' else compare)(args)


if __name__=='__main__':main()
