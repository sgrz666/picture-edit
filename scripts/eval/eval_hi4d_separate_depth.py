#!/usr/bin/env python3
"""Hi4D native, direct correspondence: bright separate poses +/- joint depth.

Inference only. --prepare locks inputs and prompts; --validate-only also runs
coordinate/depth tests. Old Hi4D experiment and all model weights are read only.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.eval.eval_hi4d_native_binding import (
    DEFAULT as PREVIOUS, DATA, MODEL, REVISION, BODY_EDGES, COLORS,
    points_for, sha, write, read_rows, versions,
)
from src.data.letterbox import Letterbox

DEFAULT = Path('/home/shangguanrz/nihaochen/qwen21_full/hi4d_separate_depth_20261012')
METHODS = ('separate', 'separate_depth')
RESOLUTION = 1024
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FOCUS = {
    'hi4d_00': '交叉前臂的前后次序、相互接触与两人的独立手臂',
    'hi4d_01': '两人手掌相对位置及上下关系、双手接触是否自然',
    'hi4d_02': '两人交叠的手掌/前臂、手臂是否混合或穿过身体',
    'hi4d_03': '拥抱时两人的手臂环抱路径、胸前/背后的前后关系',
    'hi4d_04': '一人搭肩与另一人抱腰的不同角色、交叠手臂的前后关系',
    'hi4d_05': '近镜头手臂与伙伴腰背的遮挡、另一人的垂手是否保留',
}


def skeleton_xy(xy, scores):
    """Same coordinates/threshold as the historical renderer; no confidence dimming."""
    if xy.shape != (134, 2) or scores.shape != (134,) or not np.isfinite(xy).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid 134-point skeleton')
    valid = scores > .3
    canvas = np.zeros((1024, 1024, 3), np.uint8)
    # Fixed pastel colors keep even originally blue limbs bright on black.
    colors = [tuple(round(.75*v + .25*255) for v in c) for c in COLORS]
    lines = []
    def line(a, b, color, thickness):
        if valid[a] and valid[b]:
            cv2.line(canvas, tuple(np.rint(xy[a]).astype(int)), tuple(np.rint(xy[b]).astype(int)),
                     color, thickness, cv2.LINE_AA)
            lines.append([int(a), int(b), thickness])
    for i, (a, b) in enumerate(BODY_EDGES):
        line(a, b, colors[i], 3)
    for i in range(18):
        if valid[i]:
            cv2.circle(canvas, tuple(np.rint(xy[i]).astype(int)), 2, colors[i], -1, cv2.LINE_AA)
    for offset in (92, 113):
        for finger in range(5):
            ids = [offset, *range(offset+1+finger*4, offset+5+finger*4)]
            for j, (a, b) in enumerate(zip(ids, ids[1:])):
                line(a, b, colors[(finger*4+j) % 18], 2)
    for a, b in ((13,18),(18,19),(19,20),(20,13),(10,21),(21,22),(22,23),(23,10)):
        line(a, b, (255,255,255), 2)
    for i in range(24, 92):
        if valid[i]:
            cv2.circle(canvas, tuple(np.rint(xy[i]).astype(int)), 2, (255,255,255), -1, cv2.LINE_AA)
    return Image.fromarray(canvas), {'valid_point_indices':np.flatnonzero(valid).tolist(),
                                     'drawn_edges':lines, 'labels_drawn':False,
                                     'confidence_affects_brightness':False}


def depth_display(scene, first, second):
    if not (scene.shape == first.shape == second.shape) or scene.ndim != 2:
        raise ValueError('Depth shapes differ')
    if any(not np.isfinite(x).all() or np.any(x < 0) for x in (scene, first, second)):
        raise ValueError('Nonfinite/negative depth')
    valid = scene > 0
    union = (first > 0) | (second > 0)
    if not np.array_equal(valid, union) or not valid.any():
        raise ValueError('Scene depth foreground differs from individual surface union')
    nearest = np.minimum(np.where(first > 0, first, np.inf), np.where(second > 0, second, np.inf))
    maxerr = float(np.abs(scene[valid]-nearest[valid]).max())
    # Native assets store quantized raster depths: case04 differs by <=7.9e-5 m.
    if maxerr > 1e-4:
        raise ValueError(f'Scene depth does not select nearest surface: {maxerr}')
    lo, hi = float(scene[valid].min()), float(scene[valid].max())
    if hi <= lo:
        raise ValueError('Degenerate shared depth range')
    grey = np.zeros(scene.shape, np.uint8)
    grey[valid] = np.rint(32 + 223*(hi-scene[valid])/(hi-lo)).astype(np.uint8)
    overlap = (first>0) & (second>0)
    return Image.fromarray(grey).convert('RGB'), {
        'shared_depth_range_metres':[lo,hi], 'near_bright':True, 'foreground_grey_range':[32,255],
        'background_grey':0, 'foreground_pixels':int(valid.sum()), 'overlap_pixels':int(overlap.sum()),
        'nearest_surface_max_error_metres':maxerr, 'nearest_surface_tolerance_metres':1e-4,
        'normalization':'one range for the already jointly rendered visible scene; never per person',
    }


def prompt_for(case, method):
    if method not in METHODS:
        raise ValueError(method)
    text = [
        'Edit image 1. It is a photograph of the TWO original people whose identities, appearances and scene must be preserved. '
        'Image 2 is the target pose drawing for the first specified person, and image 3 is the target pose drawing for the second specified person. '
        'Images 2 and 3 use the SAME full-scene canvas and coordinates: interpret them together as one two-person interaction. '
        'Do not independently center, crop, resize or reposition either skeleton. Colored lines and white facial dots describe pose only, not appearance.',
        f'In image 1, assign {case["source"]["descriptors"][0]} to the skeleton in IMAGE 2. '
        f'Assign {case["source"]["descriptors"][1]} to the skeleton in IMAGE 3. '
        'Follow these explicit person-to-image assignments rather than matching by original left/right position, gender or body size.',
    ]
    if method == 'separate_depth':
        text.append('Image 4 is the JOINT depth map of the SAME target two-person interaction, aligned with images 2 and 3. '
                    'Brighter nonblack human surfaces are nearer the camera and darker human surfaces are farther away; black is background. '
                    'Use it only to interpret surface depth and which body parts occlude other parts. '
                    'It supplies no identity, clothing, skin color, lighting or output texture.')
    text.append('Repose each original person according to their assigned skeleton. Reproduce the shared target placement, distance, '
                'head and body orientation, and natural contact/interaction, while retaining each person\'s original body proportions. '
                'Preserve each original person\'s recognizable face, skin tone, hairstyle, facial hair, clothing and footwear. '
                'Do not swap or blend identities, outfits or body types. Keep the background, floor, camera view and lighting from image 1. '
                'Return ONE natural photorealistic image containing EXACTLY TWO people, with plausible anatomy and separate limbs. '
                'Do NOT display skeleton lines, colored pose strokes, landmark dots, labels, letters, numbers, depth-map greyscale, '
                'reference drawings, panels or a collage in the result.')
    return '\n\n'.join(text)


def self_check():
    # Confidence changes ABOVE the threshold must not change a single rendered pixel.
    xy = np.column_stack((np.linspace(170,800,134),np.linspace(130,900,134)))
    a, meta = skeleton_xy(xy, np.full(134,.4))
    b, _ = skeleton_xy(xy, np.full(134,.99))
    assert np.array_equal(a,b) and len(meta['valid_point_indices']) == 134
    scores = np.full(134,.99); scores[113] = .3
    _, missing = skeleton_xy(xy,scores)
    assert 113 not in missing['valid_point_indices'] and not any(113 in e[:2] for e in missing['drawn_edges'])
    # Exact shared mapping and overlap selection; never averaging surfaces.
    first = np.array([[0,2,3],[0,4,0]],np.float32)
    second = np.array([[0,3,2],[0,0,5]],np.float32)
    scene = np.minimum(np.where(first>0,first,np.inf),np.where(second>0,second,np.inf))
    scene[~np.isfinite(scene)] = 0
    im, d = depth_display(scene,first,second)
    g = np.asarray(im)[:,:,0]
    assert g[0,0]==0 and g[0,1]==255 and g[1,2]==32 and d['overlap_pixels']==2
    try:
        wrong = scene.copy(); wrong[0,1] = 3
        depth_display(wrong,first,second)
    except ValueError:
        pass
    else:
        raise AssertionError('Far surface accepted')
    # Synthetic pixel landmark and normalized coordinates undergo the same transform.
    t = Letterbox.from_size((940,1280),1024)
    mark = Image.new('L',(940,1280));mark.putpixel((470,640),255)
    mapped = np.asarray(t.image(mark,nearest=True))
    xy2 = t.points(np.array([[.5,.5]]))*1024
    assert tuple(xy2[0]) == (512.,512.) and mapped[512,512]==255
    assert Letterbox.from_size((1024,1024),1024).metadata()['valid_rect'] == [0,0,1024,1024]
    return {'passed':True,'checks':['constant_brightness_above_threshold','exact_valid_points_and_edges',
        'nearest_surface_overlaps','one_shared_depth_range','letterbox_landmark_coordinate', 'square_compatibility']}


def references(root, job):
    names = ['A','S1','S2'] + (['depth_joint'] if job['method']=='separate_depth' else [])
    return [{'index':i+1,'role':name,'path':str(root/'inputs/1024'/job['pair_id']/f'{name}.png'),
             'sha256':sha(root/'inputs/1024'/job['pair_id']/f'{name}.png')} for i,name in enumerate(names)]


def key(row):
    return row['pair_id'], row['method']


def prepare(root):
    if (root/'generations.jsonl').exists():
        raise ValueError('Cannot relock an experiment after generation started')
    root.mkdir(parents=True,exist_ok=True)
    tests = self_check();write(root/'checks/synthetic_checks.json',tests)
    cases = json.loads((PREVIOUS/'data/cases.json').read_text())
    assert [c['pair_id'] for c in cases] == [f'hi4d_{i:02d}' for i in range(6)]
    cases = [{k:v for k,v in c.items() if k!='actions'} for c in cases]
    protected = {};qc=[]
    for c in cases:
        folder = root/'inputs/1024'/c['pair_id'];folder.mkdir(parents=True,exist_ok=True)
        meta={};renders=[]
        for name,role in [('A','source'),('B','target')]:
            record=c[role]
            for p in [record['rgb'],*record['masks'],record['smpl'],record['camera'],*record.get('dwpose',[])]:
                protected[p] = sha(p)
            im=Image.open(record['rgb']).convert('RGB');t=Letterbox.from_size(im.size,1024)
            t.image(im,fill=(128,128,128)).save(folder/f'{name}.png')
            meta[name]={'transform':t.metadata(),'source':record['rgb']}
        for person,path in enumerate(c['target']['dwpose']):
            xy,scores,t = points_for(path,1024)
            if t.metadata()!=meta['B']['transform']:
                raise ValueError('Skeleton/RGB transformation mismatch')
            im, info = skeleton_xy(xy,scores)
            im.save(folder/f'S{person+1}.png');renders.append(np.asarray(im))
            info.update({'source_keypoints':path,'mapped_coordinates':xy.tolist(),'scores':scores.tolist(),
                         'transform':t.metadata(),'missing_body_points':np.flatnonzero(scores[:18]<=.3).tolist(),
                         'missing_left_hand_points':int((scores[92:113]<=.3).sum()),
                         'missing_right_hand_points':int((scores[113:134]<=.3).sum())})
            meta[f'S{person+1}']=info
        merged=np.maximum(*renders)
        Image.fromarray(merged).save(folder/'combined_qc_only.png')
        base=np.asarray(Image.open(folder/'B.png')).copy();mask=merged.max(-1)>0
        base[mask]=merged[mask];Image.fromarray(base).save(folder/'B_skeleton_overlay.png')
        pair,action,frame = c['target']['key']
        raw=DATA/'final_assets_native_v1'/pair/action/'cam4/geometry'/f'{frame}.npz'
        protected[str(raw)]=sha(raw);shutil.copyfile(raw,folder/'depth_raw.npz')
        with np.load(raw,allow_pickle=False) as z:
            size=(int(z['width']),int(z['height']))
            if size!=Image.open(c['target']['rgb']).size:
                raise ValueError('Depth/RGB native canvas differs')
            depth,di=depth_display(z['depth_scene'],z['depth_A'],z['depth_B'])
            if z['K_runtime'].shape!=(3,3) or not np.isfinite(z['K_runtime']).all():
                raise ValueError('Invalid runtime camera intrinsics')
            di['K_runtime']=z['K_runtime'].tolist()
            di['asset_version']=str(z['asset_version'])
            # Compare to camera 4 calibration, not just a hard-coded camera name.
            with np.load(c['target']['camera'],allow_pickle=False) as cam:
                ids=np.asarray(cam['ids']).astype(str)
                at=np.flatnonzero(ids=='4')
                if len(at)!=1:raise ValueError('Camera 4 calibration unavailable')
                K=np.asarray(cam['intrinsics'][int(at[0])])
                if not np.allclose(K,z['K_runtime'],atol=1e-4):raise ValueError('Depth camera intrinsics differ from camera4')
            di['camera4_intrinsics_match']=True
        depth.save(folder/'depth_native_grey.png')
        t=Letterbox.from_size(size,1024);t.image(depth,fill=(0,0,0)).save(folder/'depth_joint.png')
        di.update({'raw_source':str(raw),'transform':t.metadata(),'geometry_fill':0})
        meta['depth_joint']=di
        depthmask=np.asarray(Image.open(folder/'depth_joint.png')).max(-1)>0
        edges=cv2.Canny(depthmask.astype(np.uint8)*255,50,150)>0
        overlay=np.asarray(Image.open(folder/'B.png')).copy();overlay[edges]=(255,255,0)
        Image.fromarray(overlay).save(folder/'B_depth_outline.png')
        write(folder/'metadata.json',meta)
        qc.append({'pair_id':c['pair_id'],'depth':di,'source_descriptors':c['source']['descriptors'],
                   'target_descriptors':c['target']['descriptors'],'focus_before_generation':FOCUS[c['pair_id']],
                   'point_counts':[len(meta[f'S{s}']['valid_point_indices']) for s in (1,2)]})
    jobs=[{'pair_id':c['pair_id'],'resolution':1024,'method':m,'mapping':'direct'} for c in cases for m in METHODS]
    prompts={c['pair_id']:{m:prompt_for(c,m) for m in METHODS} for c in cases}
    write(root/'data/cases.json',cases);write(root/'data/jobs.json',jobs)
    write(root/'prompts/locked_prompts.json',prompts);write(root/'checks/input_checks.json',qc)
    # Six readable input-QC strips: B, each separate pose, overlay, shared depth, depth silhouette.
    for c in cases:
        folder=root/'inputs/1024'/c['pair_id'];names=['B','S1','S2','B_skeleton_overlay','depth_joint','B_depth_outline']
        sheet=Image.new('RGB',(6*380,430),'white');draw=ImageDraw.Draw(sheet)
        font=ImageFont.truetype(FONT,18)
        for i,n in enumerate(names):
            sheet.paste(Image.open(folder/f'{n}.png').resize((380,380)),(i*380,42))
            draw.text((i*380+8,8),f'{c["pair_id"]} {n}',font=font,fill='black')
        sheet.save(root/'checks'/f'{c["pair_id"]}_inputs.jpg',quality=95)
    previous_lock=json.loads((PREVIOUS/'experiment_lock.json').read_text())
    model_files=previous_lock['model_metadata']
    runtime_sources={str(Path(__file__).resolve()):sha(__file__),
                     str(PROJECT/'src/data/letterbox.py'):sha(PROJECT/'src/data/letterbox.py')}
    from scripts.eval.eval_qwen21_native_value import SamplingWitness
    from scripts.eval.eval_qwen21_bfs_skeleton import sample
    for fn in (SamplingWitness,sample):
        p=inspect.getfile(fn);runtime_sources[p]=sha(p)
    locked={str(p):sha(p) for d in ('data','prompts') for p in (root/d).rglob('*') if p.is_file()}
    prepared={str(p):sha(p) for p in (root/'inputs').rglob('*') if p.is_file()}
    write(root/'experiment_lock.json',{'protocol':'hi4d_separate_depth_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'runtime_versions':versions(),'model_revision':REVISION,'model':str(MODEL.resolve()),
        'protected_files':protected,'model_metadata':model_files,'runtime_sources':runtime_sources,
        'locked_data':locked,'prepared_files':prepared,'reference_order':{'separate':['A','S1','S2'],
        'separate_depth':['A','S1','S2','depth_joint']},'detailed_pose_text':False,'manual_occlusion_text':False,
        'prompt_rewrite':False,'training':False,'swap':False,'formal_expected':12})
    print(f'Prepared and locked {len(jobs)} jobs in {root}')


def verify_lock(root):
    lock=json.loads((root/'experiment_lock.json').read_text())
    if lock['runtime_versions']!=versions():raise ValueError('Runtime versions differ')
    for group in ('protected_files','model_metadata','runtime_sources','locked_data','prepared_files'):
        for path,expected in lock[group].items():
            if sha(path)!=expected:raise ValueError(f'Locked file changed: {path}')
    return lock


def validate_row(row,job,prompt,refs,lock_sha):
    for k,v in {**job,'steps':30,'seed':42,'true_cfg_scale':1.,'use_kv_cache':False,
                'control_forwards':0,'lora_loaded':False,'prompt':prompt,'references':refs,
                'lock_sha256':lock_sha,'runtime_versions':versions(),'target_rgb_in_native_references':False}.items():
        if row.get(k)!=v:raise ValueError(f'Saved configuration differs: {key(job)} {k}')
    if row['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():raise ValueError('Prompt SHA differs')
    if sha(row['image'])!=row['image_sha256'] or Image.open(row['image']).size!=(1024,1024):raise ValueError('Bad output image')
    if [r['role'] for r in refs] != (['A','S1','S2']+(['depth_joint'] if job['method']=='separate_depth' else [])):
        raise ValueError('Reference image roles differ')
    if not np.isfinite(row['seconds']) or not np.isfinite(row['peak_cuda_gib']):raise ValueError('Invalid resource measurement')


def run(args):
    import torch
    from diffusers import QwenImage21Pipeline
    from scripts.eval.eval_qwen21_bfs_skeleton import sample
    root=args.output;verify_lock(root);lock_sha=sha(root/'experiment_lock.json')
    if not (root/'checks/visual_input_review.json').exists():raise ValueError('Visual input QC must be recorded before generation')
    reviewed=json.loads((root/'checks/visual_input_review.json').read_text())
    if reviewed.get('lock_sha256')!=lock_sha or not reviewed.get('passed'):raise ValueError('Visual QC does not match lock')
    jobs=json.loads((root/'data/jobs.json').read_text())
    if args.group!='all':jobs=[j for j in jobs if j['method']==args.group]
    if args.max_samples:jobs=jobs[:args.max_samples]
    prompts=json.loads((root/'prompts/locked_prompts.json').read_text())
    index=root/'generations.jsonl';rows=read_rows(index) if index.exists() else []
    if len({key(r) for r in rows})!=len(rows):raise ValueError('Duplicate result keys')
    done={key(r):r for r in rows};pending=[]
    for j in jobs:
        if key(j) in done:validate_row(done[key(j)],j,prompts[j['pair_id']][j['method']],references(root,j),lock_sha)
        else:pending.append(j)
    if not pending:return
    torch.set_num_threads(4);torch.manual_seed(42)
    print(json.dumps({'event':'loading','cuda_free_total_gib':[n/1024**3 for n in torch.cuda.mem_get_info()],
                      'pending':len(pending),'gpu':torch.cuda.get_device_name(0)}),flush=True)
    pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),dtype=torch.bfloat16).to('cuda')
    for module in (pipe.transformer,pipe.text_encoder,pipe.vae):module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    if getattr(pipe.transformer,'peft_config',None) or getattr(pipe.transformer,'_pic_edit_qwen21_control_active',False):
        raise AssertionError('Native model has LoRA or project Adapter')
    for j in pending:
        prompt=prompts[j['pair_id']][j['method']];refs=references(root,j)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        output,witness=sample(pipe,prompt,[Image.open(r['path']).convert('RGB') for r in refs],1024)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        image=output.images[0].convert('RGB')
        if image.size!=(1024,1024) or not np.isfinite(np.asarray(image)).all():raise ValueError('Invalid sampled image')
        path=root/'images/1024'/f'{j["pair_id"]}_{j["method"]}.png';path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        row={**j,'steps':30,'seed':42,'true_cfg_scale':1.,'use_kv_cache':False,'control_forwards':0,'lora_loaded':False,
             'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'references':refs,
             'lock_sha256':lock_sha,'runtime_versions':versions(),'image':str(path),'image_sha256':sha(path),
             'seconds':seconds,'peak_cuda_gib':torch.cuda.max_memory_allocated()/1024**3,
             'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
             'correctness_observation':witness,'assignment_source_to_target':[0,1],
             'target_rgb_in_native_references':False,'model_revision':REVISION}
        validate_row(row,j,prompt,refs,lock_sha)
        with index.open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        done[key(j)]=row
        status={**j,'completed':len(done),'expected':12,'seconds':seconds,'peak_cuda_gib':row['peak_cuda_gib']}
        write(root/'status.json',status);print(json.dumps(status),flush=True);del output
    if any(p.requires_grad or p.grad is not None for m in (pipe.transformer,pipe.text_encoder,pipe.vae) for p in m.parameters()):
        raise AssertionError('Unexpected training or unfrozen parameter')
    verify_lock(root)
    write(root/'runtime_passed.json',{'passed':True,'jobs_checked':len(jobs),'formal_completed':len(done),
        'all_parameters_frozen':True,'no_optimizer':True,'no_lora':True,'no_project_adapter':True,
        'protected_inputs_unchanged':True,'raw_depths_unchanged':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--validate-only',action='store_true')
    p.add_argument('--group',choices=('all',*METHODS),default='all');p.add_argument('--max-samples',type=int,default=0)
    args=p.parse_args()
    if args.prepare:prepare(args.output)
    elif args.validate_only:verify_lock(args.output);print(json.dumps(self_check()))
    else:run(args)


if __name__=='__main__':main()
