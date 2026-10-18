#!/usr/bin/env python3
"""Locked, inference-only Hi4D person-to-pose correspondence experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA = PROJECT / 'datasets/Hi4D_pilot_v1'
DEFAULT = Path('/home/shangguanrz/nihaochen/qwen21_full/hi4d_native_binding_20261010')
MODEL = PROJECT / 'models/Qwen-Image-2.1'
REVISION = '790c92633540aa0cb11d9abf19eb46d861714758'
METHODS = ('rgb', 'combined', 'separate')
MAPPINGS = {'direct': (0, 1), 'swap': (1, 0)}
LOWRES = ('hi4d_00', 'hi4d_04')
CASES = (
    (('pair22', 'hug22', '000005'), ('pair28', 'fight28', '000081')),
    (('pair22', 'hug22', '000005'), ('pair12', 'highfive12', '000032')),
    (('pair28', 'dance28', '000006'), ('pair22', 'game22', '000014')),
    (('pair28', 'dance28', '000006'), ('pair14', 'hug14', '000034')),
    (('pair12', 'highfive12', '000028'), ('pair28', 'dance28', '000067')),
    (('pair12', 'highfive12', '000028'), ('pair22', 'hug22', '000089')),
)
DESCRIPTORS = {
    'pair22': (
        'the short-haired person in the dark burgundy sweater, patterned shorts and black athletic shoes',
        'the bob-haired person in the light blue sweater, beige trousers and tan ankle boots'),
    'pair28': (
        'the tall short blond-haired person in the dark blue long-sleeved top, beige trousers and dark sneakers',
        'the curly reddish-haired person in the white T-shirt, grey-brown trousers and light sneakers'),
    'pair12': (
        'the person with wavy dark hair and a beard, wearing a navy long-sleeved shirt, black trousers and brown sandals',
        'the short-haired person in the black short-sleeved T-shirt, black trousers and light grey sneakers'),
    'pair14': (
        'the short dark-haired person in the navy polo shirt, blue jeans and black sneakers',
        'the tied-back-haired person in the black sleeveless athletic top, black leggings and light sneakers'),
}
# Indexed by the official subject IDs, not by image left/right.
ACTIONS = (
    (
        'Occupy the right-hand target position. Stand with feet apart and knees slightly bent; turn the torso partly away from the camera towards the partner on the left. Raise one bent forearm diagonally between the two faces to meet or block the partner\'s raised arm, while the other hand stays low near the hip. Turn the head towards the partner.',
        'Occupy the left-hand target position. Face the partner on the right with a slightly bent, staggered stance. Raise one bent arm towards the partner\'s upper chest/raised forearm, extend the other arm down and outward to the image left, and look towards the partner. Keep the unequal arm poses of the two roles.'),
    (
        'Occupy the left-hand target position, facing right. Lean the torso forward slightly and extend both arms forward at roughly waist height, elbows slightly bent, palms facing down over the partner\'s open hands. Keep both feet on the floor.',
        'Occupy the right-hand target position, facing left. Remain relatively upright, with both elbows bent and both forearms extended forward at waist height, palms up beneath the partner\'s hands. Look at the partner. Reproduce the hand-to-hand contact rather than lifting the hands overhead.'),
    (
        'Occupy the left-hand target position, facing the partner on the right. Stand upright with elbows bent, both hands raised in front of the chest and palms towards the partner; reproduce the two visible hand positions, one slightly lower than the other. Keep the feet grounded and the head turned towards the partner.',
        'Occupy the right-hand target position, facing left. Bend both elbows and bring the hands in front of the upper torso to meet the partner\'s hands, keeping the lower hand near waist/low chest height. Maintain the small gap between the torsos and the mutual head orientation.'),
    (
        'Occupy the left-hand target position, facing right. Stand very close to the partner with the chest turned towards them, spread both arms outward at approximately shoulder height with soft elbows, ready to wrap them around the partner. Keep the head close to the partner and the feet grounded.',
        'Occupy the right-hand target position, facing left. Lean the upper body and head towards the partner in a close embrace, with both arms opened around their sides at shoulder/chest height. Preserve the close chest-to-chest arrangement and the distinct arm paths of the two people.'),
    (
        'Occupy the right-hand target position, facing left. Bend both elbows and place the hands around the partner\'s waist/lower torso, with the hands lower than the partner\'s raised arms. Keep an upright, close dancing stance and turn the head towards the partner.',
        'Occupy the left-hand target position, facing right. Raise both arms to wrap them over and around the partner\'s shoulders/neck, elbows bent, with hands resting near the shoulders or upper back. Lean slightly towards the partner. Preserve the shoulder-versus-waist roles and the close dancing contact.'),
    (
        'Occupy the left-hand target position, facing the partner on the right with the torso partly visible to the camera. Reach the arm nearest the camera around the partner\'s waist/lower back, keeping the other arm lower or partly occluded. Stand very close, with feet grounded and the head turned towards the partner.',
        'Occupy the right-hand target position with the torso partly turned away from the camera and the head turned left towards the partner. Keep the arms lowered alongside the body, one visible hand near the thigh, and stand close enough to receive the partner\'s arm around the waist. Do not give both people the same arm pose.'),
)
BODY_EDGES = ((1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),
              (1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17))
COLORS = ((255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
          (0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
          (0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85))
# Wholebody.__call__ inserts neck and reorders the first 18 joints to OpenPose.
OPENPOSE_TO_COCO = (0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def versions():
    import torch
    return {'python': sys.version.split()[0], 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            **{k: importlib.metadata.version(k) for k in ('diffusers', 'transformers')}}


def frame_record(key):
    pair, action, frame = key
    p = DATA / 'raw' / pair / action
    return {'key': list(key), 'rgb': str(p / 'images/4' / f'{frame}.jpg'),
            'masks': [str(p / 'seg/img_seg_mask/4' / str(i) / f'{frame}.png') for i in range(2)],
            'dwpose': [str(DATA / 'native_interaction_v2/dwpose_native' / pair / action / f'{frame}_{s}.npz') for s in ('A','B')],
            'smpl': str(p / 'smpl' / f'{frame}.npz'), 'camera': str(p / 'cameras/rgb_cameras.npz'),
            'descriptors': list(DESCRIPTORS[pair])}


def points_for(path, side):
    from src.data.letterbox import Letterbox
    with np.load(path, allow_pickle=False) as z:
        xy, scores = z['xy_native'].copy(), z['scores'].copy()
        size = int(z['width']), int(z['height'])
    if xy.shape != (134,2) or scores.shape != (134,) or not np.isfinite(xy).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid whole-body keypoints')
    transform = Letterbox.from_size(size, side)
    xy = transform.points(xy / np.asarray(size)) * side
    return xy, scores, transform


def skeleton(path, subject, side):
    """Render from coordinates at requested size; no source geometry is edited."""
    xy, scores, transform = points_for(path, side)
    canvas = np.zeros((side,side,3), dtype=np.uint8)
    scale = side / 1024
    valid = scores > .3
    width = max(1, round(2*scale))
    def line(a, b, color, thickness=width):
        if valid[a] and valid[b]:
            factor = float(np.clip(scores[a]*scores[b],0,1))
            cv2.line(canvas, tuple(np.rint(xy[a]).astype(int)), tuple(np.rint(xy[b]).astype(int)),
                     tuple(int(v*factor) for v in color), thickness, cv2.LINE_AA)
    for i,(a,b) in enumerate(BODY_EDGES): line(a,b,COLORS[i])
    for i in range(18):
        if valid[i]: cv2.circle(canvas,tuple(np.rint(xy[i]).astype(int)),max(1,round(2*scale)),COLORS[i],-1,cv2.LINE_AA)
    for offset in (92,113):
        for finger in range(5):
            ids = [offset, *range(offset+1+finger*4,offset+5+finger*4)]
            for j,(a,b) in enumerate(zip(ids,ids[1:])): line(a,b,COLORS[(finger*4+j)%18],max(1,round(scale)))
    for a,b in ((13,18),(18,19),(19,20),(20,13),(10,21),(21,22),(22,23),(23,10)):
        line(a,b,(235,235,235))
    for i in range(24,92):
        if valid[i]:
            intensity = round(255*float(np.clip(scores[i],0,1)))
            cv2.circle(canvas,tuple(np.rint(xy[i]).astype(int)),max(1,round(scale)),(intensity,)*3,-1,cv2.LINE_AA)
    core = canvas.copy()
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT,max(16,round(24*scale)))
    anchor = xy[:18][valid[:18]]
    x = max(4,min(side-65,int(anchor[:,0].min()-55*scale)))
    y = max(4,min(side-35,int(anchor[:,1].min()-35*scale)))
    draw.text((x,y),f'S{subject+1}',font=font,fill='white',stroke_width=1,stroke_fill='black')
    return image,core,transform


def build_prompt(case, method, mapping):
    assignment = MAPPINGS[mapping]
    a, b = case['source'], case['target']
    if method == 'rgb':
        roles = ('Image 1 is the photograph to edit and supplies both people and the scene. '
                 'Image 2 is a photograph of TWO DIFFERENT people and supplies only the two-person pose and interaction. '
                 f'In image 2 target role S1 is {b["descriptors"][0]}; target role S2 is {b["descriptors"][1]}.')
    elif method == 'combined':
        roles = ('Image 1 is the photograph to edit and supplies both people and the scene. '
                 'Image 2 is one black-background drawing of the target two-person pose. '
                 'The skeletons are labeled S1 and S2. Each skeleton has colored body/hand lines and facial landmark dots; '
                 'line colors identify joints, NOT clothing or identity.')
    else:
        roles = ('Image 1 is the photograph to edit and supplies both people and the scene. '
                 'Image 2 contains ONLY target skeleton S1. Image 3 contains ONLY target skeleton S2. '
                 'Images 2 and 3 use the SAME full-scene canvas and coordinates; mentally overlay them to obtain the shared interaction. '
                 'Do not center, rescale or place the two skeletons independently. '
                 'Colored lines and facial landmark dots encode pose, NOT clothing or identity.')
    text = [roles, 'Return ONE photorealistic edited image containing exactly the two original people from image 1. '
            'The person-to-pose assignment below is intentional and MUST override matching by gender, appearance, height or original left/right position.']
    for source_id, target_id in enumerate(assignment):
        text.append(f'Source person A{source_id+1} is {a["descriptors"][source_id]} in image 1. '
                    f'Assign this exact person to target role S{target_id+1}. Required pose: {case["actions"][target_id]}')
    text.append('Preserve EACH source person\'s recognizable face, skin tone, hairstyle, facial hair, clothing, footwear and body proportions. '
                'Do not exchange or blend their faces, outfits or body types. Reproduce the target two-person relative placement, distance, '
                'mutual head orientation, arm paths and visible hand/body contacts, adapting articulation naturally to the original people. '
                'Left/right target positions refer to the IMAGE; preserve anatomical left/right as shown by the reference. '
                'Keep image 1\'s camera view, background, lighting and floor. Do not copy the pose-reference people\'s identities or clothes. '
                'Produce natural anatomy without fused bodies or extra people/limbs. Do not draw skeleton lines, landmark dots, labels, numbers or panels in the output.')
    return '\n\n'.join(text)


def formal_jobs(cases):
    return [{'pair_id':c['pair_id'],'resolution':res,'method':m,'mapping':mapping}
            for res in (512,1024) for c in cases if res==1024 or c['pair_id'] in LOWRES
            for mapping in MAPPINGS for m in METHODS]


def key(row): return (row['pair_id'],row['resolution'],row['method'],row['mapping'])


def prepare(root):
    from src.data.letterbox import Letterbox
    root.mkdir(parents=True,exist_ok=True)
    cases,protected,qc = [],{},[]
    for i,(a,b) in enumerate(CASES):
        case={'pair_id':f'hi4d_{i:02d}','source':frame_record(a),'target':frame_record(b),'actions':list(ACTIONS[i])}
        # A is an appearance reference; precomputed A pose is neither required nor supplied.
        case['source'].pop('dwpose')
        assert a[0]!=b[0] and a[0] not in ('pair00','pair02') and b[0] not in ('pair00','pair02')
        for role in ('source','target'):
            record=case[role]
            for path in [record['rgb'],*record['masks'],record['smpl'],record['camera'],*record.get('dwpose',[])]:
                protected[path]=sha(path)
        cases.append(case)
        for res in (512,1024):
            folder=root/'inputs'/str(res)/case['pair_id'];folder.mkdir(parents=True,exist_ok=True)
            prepared={}
            for name,role in (('A','source'),('B','target')):
                original=Image.open(case[role]['rgb']).convert('RGB')
                t=Letterbox.from_size(original.size,res)
                image=t.image(original,fill=(128,128,128));image.save(folder/f'{name}.png')
                prepared[name]={'path':str(folder/f'{name}.png'),'sha256':sha(folder/f'{name}.png'),'transform':t.metadata()}
            rendered,cores=[],[]
            for person in (0,1):
                im,core,t=skeleton(case['target']['dwpose'][person],person,res)
                im.save(folder/f'S{person+1}.png');rendered.append(np.asarray(im));cores.append(core)
                prepared[f'S{person+1}']={'path':str(folder/f'S{person+1}.png'),'sha256':sha(folder/f'S{person+1}.png'),'transform':t.metadata()}
                xy,score,_=points_for(case['target']['dwpose'][person],res)
                if (score[:18]>.3).sum()<15: raise ValueError('Too few usable target body joints')
            combined=Image.fromarray(np.maximum(*rendered));combined.save(folder/'combined.png')
            prepared['combined']={'path':str(folder/'combined.png'),'sha256':sha(folder/'combined.png'),'transform':t.metadata()}
            # Separate rendering includes identical labels. Pixelwise maximum is the exact combined image.
            assert np.array_equal(np.asarray(combined),np.maximum(*rendered))
            write(folder/'metadata.json',prepared)
            if res==1024:
                base=np.asarray(Image.open(folder/'B.png')).copy()
                overlay=base.copy();mask=np.maximum(*cores).max(-1)>0
                overlay[mask]=np.maximum(*cores)[mask]
                Image.fromarray(overlay).save(folder/'B_skeleton_overlay.png')
                panel=Image.new('RGB',(res*4,res),(255,255,255))
                for j,name in enumerate(('A','B','combined','B_skeleton_overlay')):
                    panel.paste(Image.open(folder/f'{name}.png'),(j*res,0))
                panel.save(root/'inputs'/f'{case["pair_id"]}_qc.png')
                qc.append({'pair_id':case['pair_id'],'overlay':str(folder/'B_skeleton_overlay.png'),
                           'person_ids':'official source 0/1 -> A1/A2; official target 0/1 -> S1/S2',
                           'body_visible':[(points_for(p,res)[1][:18]>.3).sum().item() for p in case['target']['dwpose']]})
    prompts={c['pair_id']:{mapping:{m:build_prompt(c,m,mapping) for m in METHODS} for mapping in MAPPINGS} for c in cases}
    from diffusers import QwenImage21Pipeline
    runtime_sources=[Path(__file__),Path(inspect.getfile(QwenImage21Pipeline)),PROJECT/'src/data/letterbox.py']
    model_files=[MODEL/'model_index.json',MODEL/'transformer/config.json',MODEL/'transformer/diffusion_pytorch_model.safetensors.index.json']
    payload={'protocol':'hi4d_native_binding_v1','training':False,'project_adapters':False,'lora_loaded':False,
             'model':str(MODEL.resolve()),'model_revision':REVISION,'versions':versions(),'steps':30,'seed':42,
             'true_cfg_scale':1.,'use_kv_cache':False,'align_mode':'letterbox_v1','resolutions':[512,1024],
             'lowres_ids':list(LOWRES),'expected_results':48,'mappings':MAPPINGS,'methods':list(METHODS),
             'protected_files':protected,'runtime_sources':{str(p):sha(p) for p in runtime_sources},
             'model_metadata':{str(p):sha(p) for p in model_files if p.exists()},'source_distinct_images':3,
             'source_distinct_people':6,'target_distinct_images':6}
    for rel,value in [('data/cases.json',cases),('data/jobs.json',formal_jobs(cases)),('prompts/locked_prompts.json',prompts),('input_qc.json',qc)]:
        p=root/rel
        if p.exists() and json.loads(p.read_text())!=value:raise ValueError(f'Existing input differs: {p}')
        write(p,value)
    payload['locked_data']={str(root/rel):sha(root/rel) for rel in ('data/cases.json','data/jobs.json','prompts/locked_prompts.json')}
    # Include all prepared references in the immutable lock.
    payload['prepared_files']={str(p):sha(p) for p in sorted((root/'inputs').glob('*/*/*.png')) if p.name!='B_skeleton_overlay.png'}
    p=root/'experiment_lock.json'
    if p.exists() and json.loads(p.read_text())!=json.loads(json.dumps(payload)):raise ValueError('Existing experiment lock differs')
    write(p,payload)
    print(json.dumps({'prepared':True,'cases':len(cases),'jobs':48,'source_people':6}),flush=True)


def references(root,job):
    folder=root/'inputs'/str(job['resolution'])/job['pair_id']
    roles={'rgb':('A','B'),'combined':('A','combined'),'separate':('A','S1','S2')}[job['method']]
    return [{'index':i+1,'role':r,'path':str(folder/f'{r}.png'),'sha256':sha(folder/f'{r}.png')} for i,r in enumerate(roles)]


def validate_saved(row,job,prompt,refs,lock_sha):
    for k,v in {**job,'steps':30,'seed':42,'true_cfg_scale':1.,'use_kv_cache':False,'control_forwards':0,
                'lora_loaded':False,'prompt':prompt,'references':refs,'lock_sha256':lock_sha,'runtime_versions':versions()}.items():
        if row.get(k)!=v:raise ValueError(f'Resume differs: {k}')
    if sha(row['image'])!=row['image_sha256'] or Image.open(row['image']).size!=(job['resolution'],)*2:
        raise ValueError('Completed output changed')


def verify_lock(root):
    lock=json.loads((root/'experiment_lock.json').read_text())
    if versions()!=lock['versions']:raise ValueError('Runtime version changed')
    for kind in ('protected_files','runtime_sources','model_metadata','locked_data','prepared_files'):
        for p,expected in lock[kind].items():
            if sha(p)!=expected:raise ValueError(f'Locked file changed: {p}')
    return lock


def run(args):
    import torch
    from diffusers import QwenImage21Pipeline
    from scripts.eval.eval_qwen21_bfs_skeleton import sample
    root=args.output;lock=verify_lock(root);lock_sha=sha(root/'experiment_lock.json')
    prompts=json.loads((root/'prompts/locked_prompts.json').read_text())
    jobs=[j for j in json.loads((root/'data/jobs.json').read_text()) if j['resolution']==args.resolution]
    if args.max_samples:jobs=jobs[:args.max_samples]
    index=root/'generations.jsonl';existing=read_rows(index) if index.exists() else []
    if len({key(r) for r in existing})!=len(existing):raise ValueError('Duplicate output keys')
    completed={key(r):r for r in existing};pending=[]
    for j in jobs:
        if key(j) in completed:
            validate_saved(completed[key(j)],j,prompts[j['pair_id']][j['mapping']][j['method']],references(root,j),lock_sha)
        else:pending.append(j)
    if not pending:return
    torch.set_num_threads(4);torch.manual_seed(42)
    pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),dtype=torch.bfloat16).to('cuda')
    for module in (pipe.transformer,pipe.text_encoder,pipe.vae):module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    if getattr(pipe.transformer,'peft_config',None) or getattr(pipe.transformer,'_pic_edit_qwen21_control_active',False):
        raise AssertionError('Native instance has LoRA or project control')
    for j in pending:
        prompt=prompts[j['pair_id']][j['mapping']][j['method']];refs=references(root,j)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        output,witness=sample(pipe,prompt,[Image.open(r['path']).convert('RGB') for r in refs],j['resolution'])
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        rgba=output.images[0].convert('RGBA')
        image=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
        path=root/'images'/str(j['resolution'])/f'{j["pair_id"]}_{j["mapping"]}_{j["method"]}.png'
        path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        row={**j,'steps':30,'seed':42,'true_cfg_scale':1.,'use_kv_cache':False,'control_forwards':0,
             'lora_loaded':False,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
             'references':refs,'lock_sha256':lock_sha,'runtime_versions':versions(),'image':str(path),
             'image_sha256':sha(path),'seconds':seconds,'peak_cuda_gib':torch.cuda.max_memory_allocated()/1024**3,
             'correctness_observation':witness,'assignment_source_to_target':list(MAPPINGS[j['mapping']]),
             'target_rgb_in_native_references':j['method']=='rgb','model_revision':REVISION}
        validate_saved(row,j,prompt,refs,lock_sha)
        with index.open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        completed[key(j)]=row
        status={**j,'completed':len(completed),'expected':48,'seconds':seconds,'peak_cuda_gib':row['peak_cuda_gib']}
        write(root/'status.json',status);print(json.dumps(status),flush=True)
        del output
    if any(p.requires_grad or p.grad is not None for m in (pipe.transformer,pipe.text_encoder,pipe.vae) for p in m.parameters()):
        raise AssertionError('Unexpected gradient or unfrozen parameter')
    verify_lock(root)
    write(root/f'runtime_{args.resolution}_passed.json',{'passed':True,'jobs_checked':len(jobs),'all_parameters_frozen':True,
          'no_optimizer':True,'no_lora':True,'no_project_adapter':True,'protected_inputs_unchanged':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--resolution',type=int,choices=(512,1024),default=512)
    p.add_argument('--max-samples',type=int,default=0)
    args=p.parse_args()
    if args.prepare:prepare(args.output)
    elif args.validate_only:verify_lock(args.output);print('Locked inputs verified')
    else:run(args)


if __name__=='__main__':main()
