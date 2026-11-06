#!/usr/bin/env python3
"""Build immutable, GT-only supervision caches with already installed SAM2.1."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for, build_prompts, regional_maps


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def publish_recoverable(path, value, *, allow_revision=False):
    """Keep truncated JSON in place and publish a separately named complete record."""
    path=Path(path)
    for candidate in [path,*sorted(path.parent.glob(path.stem+'_recovered_*'+path.suffix))]:
        if not candidate.exists():
            continue
        try:
            old=json.loads(candidate.read_text(encoding='utf-8'))
        except (json.JSONDecodeError,UnicodeError):
            continue
        if old==value:
            return candidate
        if not allow_revision:
            raise ValueError('Existing published JSON differs: '+str(candidate))
    target=path if not path.exists() else path.with_name(path.stem+'_recovered_'+uuid.uuid4().hex+path.suffix)
    publish(target,value)
    return target


def save_npz(path, maps, metadata):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **{k: np.asarray(v, dtype=np.uint8 if k == 'valid' else np.float16)
                                       for k, v in maps.items()},
                            metadata=np.array(json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False)))


def save_image(path, image, *, format='PNG'):
    with Path(path).open('xb') as stream:
        image.save(stream, format=format, **({'quality':90} if format=='JPEG' else {}))


def enabled_hand(frame):
    return any(frame.get('loss_mask', {}).get(k, False) and frame.get('hand_raw_pose_usable', {}).get(k, False)
               for k in ('hand_a', 'hand_b'))


def select_frames(rows, limit, seed):
    if limit < 0:
        raise ValueError('limit must be nonnegative; 0 means all frames')
    if not limit or limit >= len(rows):
        return list(rows)
    rng = random.Random(seed); shuffled = list(rows); rng.shuffle(shuffled)
    # Include both trustworthy hands and ordinary/disabled-hand cases. Pilot selection
    # never replaces the training split or creates a filtered training manifest.
    selected, used, clips = [], set(), set()
    hand_budget = max(1, limit//3)
    for row in shuffled:
        if enabled_hand(row) and row.get('clip_id') not in clips:
            selected.append(row); used.add(row['frame_id']); clips.add(row.get('clip_id'))
            if len(selected) == hand_budget:
                break
    if len(selected) >= limit:
        return selected
    for unique_clip in (True, False):
        for row in shuffled:
            if row['frame_id'] in used or (unique_clip and row.get('clip_id') in clips):
                continue
            selected.append(row); used.add(row['frame_id']); clips.add(row.get('clip_id'))
            if len(selected) == limit:
                return selected
    return selected


class SamForeground:
    def __init__(self, args):
        if args.sam_repo:
            sys.path.insert(0, str(args.sam_repo.resolve()))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device = args.device
        self.predictor = SAM2ImagePredictor(build_sam2(args.sam_config, str(args.sam_checkpoint), device=args.device,
                                                     apply_postprocessing=False))

    def __call__(self, image, frame):
        points, labels, box = build_prompts(frame)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.startswith('cuda') else nullcontext()
        with torch.inference_mode(), context:
            self.predictor.set_image(np.array(image, copy=True))
            masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels, box=box,
                                                     multimask_output=True)
        quality = np.asarray(scores).reshape(-1)
        masks = np.asarray(masks).reshape(-1, image.height, image.width)
        # Prefer SAM quality while requiring it to cover observed body prompts.
        px = np.clip(np.rint(points[:,0]).astype(int), 0, image.width-1)
        py = np.clip(np.rint(points[:,1]).astype(int), 0, image.height-1)
        coverage = masks[:,py,px].mean(1)
        rank = quality + .5*coverage
        chosen = int(rank.argmax()); mask = masks[chosen].astype(np.uint8)
        if not mask.any() or coverage[chosen] < .65:
            raise ValueError('SAM foreground empty or misses more than 35% of body prompts')
        return mask, dict(sam_score=float(quality[chosen]), body_prompt_coverage=float(coverage[chosen]),
                          foreground_fraction=float(mask.mean()), candidate=chosen)


def weight_pixels(maps, coefficients=(.5,.5,1.), max_weight=2.5):
    value = 1 + sum(c*maps[k] for c,k in zip(coefficients, ('body','joint','hand')))
    return maps['valid'] * np.clip(value, 1., max_weight)


def heatmap(value, lo=0., hi=1.):
    value = np.clip((value-lo)/(hi-lo), 0, 1)
    stops = np.array([[20,35,80], [35,145,190], [100,210,130], [255,205,60], [230,55,35]], np.float32)
    ix = np.minimum((value*4).astype(int),3); mix = (value*4-ix)[...,None]
    return Image.fromarray(np.uint8(stops[ix]*(1-mix)+stops[ix+1]*mix))


def previews(image, spec, maps):
    rgb = spec.image(image); pixels = np.asarray(rgb).astype(np.float32)
    overlay = pixels.copy()
    for key, color in (('body',(50,190,110)),('joint',(255,180,0)),('hand',(240,60,130))):
        alpha = maps[key][...,None] * .4
        overlay = overlay*(1-alpha)+np.array(color)*alpha
    weight = weight_pixels(maps)
    h,w = spec.canvas_hw
    latent = np.asarray(Image.fromarray(weight).resize(spec.token_hw[::-1], Image.Resampling.BOX))
    low = heatmap(latent, 0, 2.5).resize((w,h), Image.Resampling.NEAREST)
    panels = [rgb, Image.fromarray(np.uint8(overlay)), heatmap(maps['body']), heatmap(maps['joint']),
              heatmap(maps['hand']), heatmap(weight,0,2.5), low]
    strip = Image.new('RGB', (w*len(panels),h+26), 'white'); draw=ImageDraw.Draw(strip)
    for i,(panel,label) in enumerate(zip(panels,('GT','GT + regions','Body','Joint','Hand','Weight 0-2.5','Latent weight'))):
        strip.paste(panel,(i*w,26)); draw.text((i*w+6,6),label,fill='black')
    return strip, weight, latent


def process_frame(frame, args, predictor, config):
    fid = frame['frame_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', fid):
        raise ValueError('Unsafe/noncanonical frame_id')
    image_path = Path(frame['rgb_path']); rgb_sha = sha(image_path)
    if frame.get('rgb_sha256') != rgb_sha:
        raise ValueError('GT RGB changed: '+fid)
    spec = canvas_for(frame['original_hw'], args.canvas_long_side)
    meta = dict(schema='native_spatial_components_v1', frame_id=fid, gt_rgb_sha256=rgb_sha,
                target_canvas_id=spec.fingerprint(), canvas=spec.metadata(), geometry_method=args.geometry_method,
                source_frame_sha256=digest(frame), producer_config_sha256=digest(config),
                role='training_only_gt_supervision', hand_policy='existing_loss_mask_and_hand_raw_pose_usable',
                geometry_role='version_binding_only; foreground from GT RGB; joint/hand from existing 2D annotations')
    frame_directory = args.output_root/'frames'/fid
    ready = frame_directory/'ready.json'
    if frame_directory.exists() and not args.resume:
        raise FileExistsError(str(frame_directory))
    candidates=[ready,*sorted(frame_directory.glob('ready_*.json'))]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            record = json.loads(candidate.read_text())
        except (json.JSONDecodeError,UnicodeError):
            continue  # Retain interrupted marker; a new immutable attempt can finish.
        if record['identity'] != digest(meta):
            raise ValueError('Resume frame/config/Canvas identity differs: '+fid)
        if any(sha(v['path']) != v['sha256'] for v in record['artifacts'].values()):
            raise ValueError('Resume artifact changed: '+fid)
        return record
    frame_directory.mkdir(parents=True, exist_ok=True)
    directory=frame_directory/('attempt_'+uuid.uuid4().hex)
    directory.mkdir(exist_ok=False)
    with Image.open(image_path) as im:
        image=im.convert('RGB')
    if (image.height,image.width) != spec.original_hw:
        raise ValueError('GT dimensions differ from frame metadata')
    body, sam = predictor(image,frame)
    maps, status = regional_maps(frame,body,spec)
    meta.update(region_status=status, segmentation=sam)
    strip, weight, latent = previews(image,spec,maps)
    paths = dict(components=directory/'components.npz',body_mask=directory/'body_mask.png',
                 joint=directory/'joint_heatmap.png',hand=directory/'hand_soft.png',
                 weight=directory/'weight_heatmap.png',latent=directory/'latent_weight.png',preview=directory/'preview.jpg')
    save_npz(paths['components'],maps,meta)
    save_image(paths['body_mask'],Image.fromarray(body*255))
    save_image(paths['joint'],Image.fromarray(np.uint8(maps['joint']*255)))
    save_image(paths['hand'],Image.fromarray(np.uint8(maps['hand']*255)))
    save_image(paths['weight'],heatmap(weight,0,2.5))
    save_image(paths['latent'],heatmap(latent,0,2.5))
    save_image(paths['preview'],strip,format='JPEG')
    # Loss sanity uses deliberately nonuniform errors and independently accumulated
    # sums; all-one errors alone would miss a misaligned weight map.
    valid_latent = np.asarray(Image.fromarray(maps['valid']).resize(spec.token_hw[::-1],Image.Resampling.BOX))
    y,x=np.indices(spec.token_hw); error=(1+y+2*x).astype(np.float64)
    baseline=float((error*valid_latent).sum()/valid_latent.sum())
    zero=weight_pixels(maps,(0,0,0))
    zero_latent=np.asarray(Image.fromarray(zero).resize(spec.token_hw[::-1],Image.Resampling.BOX))
    identity=dict(meta); identity.pop('region_status'); identity.pop('segmentation')
    record=dict(identity=digest(identity), metadata=meta,
                artifacts={k:dict(path=str(p.resolve()),sha256=sha(p)) for k,p in paths.items()},
                statistics=dict(body_area=float(maps['body'].sum()),joint_area=float(maps['joint'].sum()),hand_area=float(maps['hand'].sum()),
                                valid_pixels=int(maps['valid'].sum()),weight_min_valid=float(weight[maps['valid']>0].min()),
                                weight_max=float(weight.max()),latent_max=float(latent.max()),
                                padding_weight_max=float(weight[maps['valid']==0].max()) if (maps['valid']==0).any() else 0.,
                                zero_coefficients_baseline_error=abs(float((error*zero_latent).sum()/zero_latent.sum())-baseline)))
    publish(ready if not ready.exists() else frame_directory/('ready_'+uuid.uuid4().hex+'.json'),record)
    return record


def run(args):
    if not args.output_root.is_absolute() or not str(args.output_root).startswith('/data/shangguanrz/'):
        raise ValueError('Output must be an absolute directory under /data/shangguanrz/')
    if args.torch_threads<1:
        raise ValueError('torch-threads must be positive')
    torch.set_num_threads(args.torch_threads)
    rows=[json.loads(line) for line in args.frames.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No source frames')
    if len({r['frame_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate frame IDs')
    selected=select_frames(rows,args.limit,args.seed)
    config=dict(schema='native_spatial_generation_v1',frames=dict(path=str(args.frames),sha256=sha(args.frames)),
                selected_frame_ids=[r['frame_id'] for r in selected],sam_checkpoint=dict(path=str(args.sam_checkpoint),sha256=sha(args.sam_checkpoint)),
                sam_config=args.sam_config,device=args.device,torch_threads=args.torch_threads,canvas_long_side=args.canvas_long_side,
                geometry_method=args.geometry_method,seed=args.seed,
                preview_coefficients=[.5,.5,1.],preview_max_weight=2.5,
                producer_files={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(), ROOT/'src/pose_control/qwen21_full/native_v5/spatial_masks.py')})
    if args.output_root.exists():
        if not args.resume:
            raise FileExistsError(str(args.output_root))
        publish_recoverable(args.output_root/'config.json',config)
    else:
        args.output_root.mkdir(parents=True,exist_ok=False);publish_recoverable(args.output_root/'config.json',config)
    predictor=SamForeground(args); records=[];started=time.time()
    with (args.output_root/('progress_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.jsonl')).open('x') as progress:
        for i,frame in enumerate(selected):
            try:
                record=process_frame(frame,args,predictor,config);records.append(record)
            except Exception as error:
                publish(args.output_root/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),
                        dict(frame_id=frame['frame_id'],error=repr(error),completed=len(records),training_started=False))
                raise
            status=dict(completed=i+1,total=len(selected),frame_id=frame['frame_id'],elapsed_sec=round(time.time()-started,2))
            progress.write(json.dumps(status)+'\n');progress.flush()
            if i<3 or (i+1)%20==0 or i+1==len(selected):
                print(json.dumps(status),flush=True)
    entries={r['metadata']['gt_rgb_sha256']+':'+r['metadata']['target_canvas_id']:
             dict(file=r['artifacts']['components'],metadata=r['metadata']) for r in records}
    if len(entries)!=len(records):
        raise ValueError('Duplicate GT/Canvas identity; inspect before publishing')
    index=dict(schema='native_spatial_index_v1',config=config,entries=entries)
    index_path=publish_recoverable(args.output_root/'index.json',index)
    summary=dict(frames=len(records),source_frames=len(rows),failed=0,elapsed_sec=round(time.time()-started,2),
                 hand_enabled_frames=sum(any(v['enabled'] for v in r['metadata']['region_status']['hands'].values()) for r in records),
                 padding_weight_max=max(r['statistics']['padding_weight_max'] for r in records),
                 zero_coefficients_baseline_max_error=max(r['statistics']['zero_coefficients_baseline_error'] for r in records),
                 index=dict(path=str(index_path),sha256=sha(index_path)),
                 training_started=False,models_downloaded=False)
    publish_recoverable(args.output_root/'summary.json',summary,allow_revision=True)
    preview_records=records if len(records)<=20 else records[::max(1,len(records)//20)][:20]
    thumbs=[]
    for r in preview_records:
        with Image.open(r['artifacts']['preview']['path']) as im:
            panel=im.convert('RGB');panel.thumbnail((1400,430));thumbs.append(panel)
    sheet=Image.new('RGB',(max(im.width for im in thumbs),sum(im.height+22 for im in thumbs)),'white');draw=ImageDraw.Draw(sheet);top=0
    for im,r in zip(thumbs,preview_records):
        draw.text((6,top+4),r['metadata']['frame_id'],fill='black');sheet.paste(im,(0,top+22));top+=im.height+22
    if not (args.output_root/'contact_sheet.jpg').exists():
        save_image(args.output_root/'contact_sheet.jpg',sheet,format='JPEG')
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--sam-checkpoint',type=Path,default=Path('/data/shangguanrz/models/home-models/sam2.1/sam2.1_hiera_large.pt'))
    p.add_argument('--sam-repo',type=Path,default=Path('/home/shangguanrz/project/pic-edit/add/tools/sam2'))
    p.add_argument('--sam-config',default='configs/sam2.1/sam2.1_hiera_l.yaml')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--torch-threads',type=int,default=4)
    p.add_argument('--canvas-long-side',type=int,default=512);p.add_argument('--geometry-method',default='sapiens_v1')
    p.add_argument('--limit',type=int,default=20);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true')
    args=p.parse_args();run(args)


if __name__=='__main__':
    main()
