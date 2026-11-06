"""Expand the visual review without changing model, loss or region settings."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import html
import json
import os
from pathlib import Path
import random
import sys
from urllib.parse import quote

ROOT=next(p for p in (Path.cwd(),*Path(__file__).resolve().parents) if (p/'src/data/canvas_contract.py').exists())
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from src.data.canvas_contract import CanvasSpec
from src.pose_control.qwen21_full.native_v5.spatial_masks import valid_keypoints
from scripts.data.build_native_spatial_weights import sha,publish,save_image


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def negative(row):return any(row.get(k+'_status') in ('occluded','out_of_frame') or row.get(k+'_present') is False for k in ('hand_a','hand_b'))
def fine(row):return any(row.get('loss_mask',{}).get(k,False) and row.get('hand_raw_pose_usable',{}).get(k,False) for k in ('hand_a','hand_b'))


def hand_scale(row):
    points,scores,valid=valid_keypoints(row);scales=[]
    for block in row.get('hand_block_slices',{}).values():
        ids=np.arange(*block);good=valid[ids]&(scores[ids]>=.35)
        if good.sum()>=3:
            scales.append(float(np.ptp(points[ids][good],axis=0).max())/max(row['original_hw']))
    return max(scales,default=0.)


def select_review_frames(source,excluded,count,seed):
    candidates=[r for r in source if r['frame_id'] not in excluded]
    if len({r['frame_id'] for r in source})!=len(source):raise ValueError('Duplicate source frame IDs')
    groups=defaultdict(list)
    rng=random.Random(seed);rng.shuffle(candidates)
    for row in candidates:groups[row['clip_id']].append(row)
    if not 2*len(groups)<=count<=len(candidates):raise ValueError('count must cover two frames per video and fit available new frames')
    if any(len(rows)<2 for rows in groups.values()):raise ValueError('Each video needs at least two new frames')
    selected=[];used=set()
    def add(row):selected.append(row);used.add(row['frame_id'])
    for clip in sorted(groups):add(groups[clip][0])
    for i,clip in enumerate(sorted(groups)):
        remaining=[r for r in groups[clip] if r['frame_id'] not in used]
        negatives=[r for r in remaining if negative(r)]
        fines=[r for r in remaining if fine(r)]
        if sum(negative(r) for r in selected)<12 and negatives:add(negatives[0])
        elif sum(fine(r) for r in selected)<15 and fines:add(fines[0])
        else:
            scales=[(hand_scale(r),n,r) for n,r in enumerate(remaining)]
            positive=[v for v in scales if v[0]>0]
            options=positive or scales
            add((min(options,key=lambda v:(v[0],v[1])) if i%2==0 else max(options,key=lambda v:(v[0],-v[1])))[2])
    while len(selected)<count:
        seen=Counter((r.get('hand_a_status'),r.get('hand_b_status')) for r in selected)
        remaining=[r for r in candidates if r['frame_id'] not in used]
        add(min(remaining,key=lambda r:seen[(r.get('hand_a_status'),r.get('hand_b_status'))]))
    return selected


def prepare(args):
    source=[json.loads(line) for line in args.frames.read_text().splitlines() if line.strip()]
    prior=load(args.exclude_selection)
    if prior['source']['sha256']!=sha(args.frames):raise ValueError('Previous review belongs to another source manifest')
    excluded=set(prior['selected_ids']);selected=select_review_frames(source,excluded,args.count,args.seed)
    args.output_root.mkdir(parents=True,exist_ok=False);(args.output_root/'input').mkdir()
    frames=args.output_root/'input'/f'frames{args.count}.jsonl'
    with frames.open('x',encoding='utf8') as stream:
        for row in selected:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    negatives=[dict(frame_id=r['frame_id'],side=k,state=r.get(k+'_status') or 'not_present') for r in selected for k in ('hand_a','hand_b')
               if r.get(k+'_status') in ('occluded','out_of_frame') or r.get(k+'_present') is False]
    report=dict(schema='spatial_hand_review_selection_v2',source=ref(args.frames),exclude_selection=ref(args.exclude_selection),
                baseline_index=ref(args.baseline_index),frames=ref(frames),seed=args.seed,selected_ids=[r['frame_id'] for r in selected],
                original_pilot_ids=[r['frame_id'] for r in selected],control_scope='all newly selected frames, paired against v1',
                negative_cases=negatives,video_counts=dict(Counter(r['clip_id'] for r in selected)),
                previous_overlap=len(excluded&{r['frame_id'] for r in selected}),
                status_counts={k:dict(Counter(r.get(k+'_status') for r in selected)) for k in ('hand_a','hand_b')},
                hand_scale_ratios={r['frame_id']:hand_scale(r) for r in selected})
    publish(args.output_root/'selection.json',report)
    print(json.dumps({k:report[k] for k in ('seed','video_counts','previous_overlap','status_counts')}),flush=True)


def gallery(args):
    selection=load(args.selection);report=load(args.comparison)
    if report['candidate']['sha256']!=sha(report['candidate']['path']):raise ValueError('Candidate index changed')
    index=load(report['candidate']['path']);entries={v['metadata']['frame_id']:v for v in index['entries'].values()}
    frames={r['frame_id']:r for r in (json.loads(line) for line in Path(selection['frames']['path']).read_text().splitlines() if line.strip())}
    if {c['frame_id'] for c in report['cases']}!=set(frames):raise ValueError('Review cases differ from selected frames')
    args.output_root.mkdir(parents=True,exist_ok=False);(args.output_root/'zooms').mkdir();(args.output_root/'pages').mkdir()
    negatives={c['frame_id'] for c in selection['negative_cases']};items=[];thumbs=[]
    for case in report['cases']:
        if sha(case['preview']['path'])!=case['preview']['sha256']:raise ValueError('Comparison figure changed')
        fid=case['frame_id'];frame=frames[fid];spec=CanvasSpec.from_record(entries[fid]['metadata']['canvas']);h,w=spec.canvas_hw
        with Image.open(case['preview']['path']) as im:strip=im.convert('RGB')
        if strip.size!=(w*8,h+44):raise ValueError('Unexpected comparison figure layout')
        images=[strip.crop((n*w,44,(n+1)*w,44+h)) for n in (0,1,2,6,7)]
        zooms={};points,scores,valid=valid_keypoints(frame)
        for side,block in frame['hand_block_slices'].items():
            ids=np.arange(*block);good=valid[ids]&(scores[ids]>=.35)
            if good.sum()<3:continue
            xy=spec.pixel_points(points[ids][good]);lo=np.maximum(0,np.floor(xy.min(0)-12)).astype(int)
            hi=np.minimum((w,h),np.ceil(xy.max(0)+13)).astype(int)
            if (hi<=lo).any():continue
            cw,ch=(hi-lo).tolist();factor=max(2,min(6,420//max(cw,ch)))
            zoom=Image.new('RGB',(cw*factor*5,ch*factor+30),'white');draw=ImageDraw.Draw(zoom)
            for n,(image,label) in enumerate(zip(images,('GT','Old hand','New hand','Old latent','New latent'))):
                cut=image.crop((*lo,*hi)).resize((cw*factor,ch*factor),Image.Resampling.NEAREST)
                zoom.paste(cut,(n*cw*factor,30));draw.text((n*cw*factor+4,8),label,fill='black')
            path=args.output_root/'zooms'/(fid+'_'+side+'.jpg');save_image(path,zoom,format='JPEG');zooms[side]='zooms/'+path.name
        target=quote(Path(os.path.relpath(Path(case['preview']['path']).resolve(),args.output_root.resolve())).as_posix(),safe='/')
        items.append(dict(frame_id=fid,clip_id=frame['clip_id'],negative=fid in negatives,hands=case['hands'],image=target,zooms=zooms))
        thumb=strip.copy();thumb.thumbnail((1600,430));thumbs.append((fid,thumb))
    page_files=[]
    for start in range(0,len(thumbs),10):
        chunk=thumbs[start:start+10];sheet=Image.new('RGB',(max(im.width for _,im in chunk),sum(im.height+20 for _,im in chunk)),'white')
        draw=ImageDraw.Draw(sheet);top=0
        for fid,im in chunk:draw.text((4,top+3),fid,fill='black');sheet.paste(im,(0,top+20));top+=im.height+20
        path=args.output_root/'pages'/f'page_{start//10+1:02d}.jpg';save_image(path,sheet,format='JPEG');page_files.append(ref(path))
    data=json.dumps(items,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>手部空间权重 · 100 张复核</title><style>body{font:15px system-ui,sans-serif;background:#f3f5f7;color:#17212c;margin:0}header{position:sticky;top:0;background:#fff;padding:18px 24px;border-bottom:1px solid #ccd3da;z-index:1}h1{font-size:22px;margin:0 0 8px}p{margin:8px 0}button,select{font:inherit;padding:7px 12px;margin-right:9px}main{max-width:1800px;margin:18px auto;padding:0 20px}article{background:white;border:1px solid #d3d9df;margin-bottom:24px;padding:14px}h2{font-size:16px;margin:0 0 10px}article img{display:block;width:100%;height:auto}a{color:#15569b}.good{color:#176c47}.off{color:#7d4b25}.meta{line-height:1.8}</style>
<header><h1>手部空间权重：新旧对照</h1><p>GT → 旧 Hand → 新 Hand → 叠加 → 旧权重 → 新权重 → 旧 latent → 新 latent。点击图片放大，或打开单手近图。</p>
<label>样本 <select id="filter"><option value="all">全部</option><option value="recovered">有新增手部增强</option><option value="rejected">至少一只手未增强</option><option value="negative">含遮挡／出画负例</option></select></label>
<button id="prev">上一页</button><button id="next">下一页</button><span id="status"></span><p>粗区域最高 0.65，精细区域最高 1；Loss 系数及 Body／Joint 不变。启用数量不等于检测准确率。</p></header><main id="cases"></main>
<script>const cases=__DATA__;let page=0;const size=10;const host=document.getElementById('cases');const filter=document.getElementById('filter');
function render(){let list=cases.filter(c=>filter.value==='all'||filter.value==='negative'&&c.negative||filter.value==='recovered'&&Object.values(c.hands).some(h=>!h.old_enabled&&h.new_enabled)||filter.value==='rejected'&&Object.values(c.hands).some(h=>!h.new_enabled));let pages=Math.max(1,Math.ceil(list.length/size));page=Math.min(Math.max(0,page),pages-1);host.replaceChildren();
for(const c of list.slice(page*size,(page+1)*size)){const card=document.createElement('article');const title=document.createElement('h2');title.textContent=c.frame_id+(c.negative?' · 含负例':'');card.append(title);const meta=document.createElement('div');meta.className='meta';
for(const [side,h] of Object.entries(c.hands)){const line=document.createElement('div');line.className=h.new_enabled?'good':'off';line.textContent=side+'：'+(h.new_enabled?(h.fine.enabled?(h.coarse.enabled?'精细＋粗区域':'精细区域'):'粗区域'):'未增强 / '+h.coarse.reason)+'；latent 独立增量 '+h.old_isolated_latent_increment_max.toFixed(3)+' → '+h.new_isolated_latent_increment_max.toFixed(3)+'  ';if(c.zooms[side]){const a=document.createElement('a');a.href=c.zooms[side];a.target='_blank';a.rel='noopener';a.textContent='查看这只手近图';line.append(a);}meta.append(line);}card.append(meta);const a=document.createElement('a');a.href=c.image;a.target='_blank';a.rel='noopener';const img=document.createElement('img');img.src=c.image;img.alt=c.frame_id+' 新旧空间权重对照';img.loading='lazy';a.append(img);card.append(a);host.append(card);}
document.getElementById('status').textContent='第 '+(page+1)+' / '+pages+' 页，共 '+list.length+' 张';document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=page===pages-1;}
filter.onchange=()=>{page=0;render();};document.getElementById('prev').onclick=()=>{page--;render();};document.getElementById('next').onclick=()=>{page++;render();};render();</script></html>'''.replace('__DATA__',data)
    with (args.output_root/'index.html').open('x',encoding='utf8') as stream:stream.write(document)
    publish(args.output_root/'gallery_manifest.json',dict(frames=len(items),videos=len({i['clip_id'] for i in items}),
            pages=page_files,zooms=sum(len(i['zooms']) for i in items),comparison=ref(args.comparison)))
    print(json.dumps(dict(frames=len(items),pages=len(page_files),zooms=sum(len(i['zooms']) for i in items))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--frames',type=Path,required=True);a.add_argument('--exclude-selection',type=Path,required=True)
    a.add_argument('--baseline-index',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);a.add_argument('--count',type=int,default=100);a.add_argument('--seed',type=int,default=20261107)
    a=s.add_parser('gallery');a.add_argument('--selection',type=Path,required=True);a.add_argument('--comparison',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();(prepare if args.action=='prepare' else gallery)(args)


if __name__=='__main__':main()
