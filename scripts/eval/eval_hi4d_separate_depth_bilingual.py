#!/usr/bin/env python3
"""Locked, inference-only Hi4D thin-skeleton/depth bilingual experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.eval import eval_hi4d_separate_depth as legacy
from scripts.eval.eval_hi4d_native_binding import BODY_EDGES, COLORS, MODEL, REVISION, sha, write, read_rows, versions

DEFAULT = Path('/home/shangguanrz/nihaochen/qwen21_full/hi4d_separate_depth_prompt_v2_20261012')
PREVIOUS = legacy.DEFAULT
METHODS = ('separate', 'separate_depth')
LANGUAGES = ('en', 'zh')
ARMS = tuple((m, l) for m in METHODS for l in LANGUAGES)
LABELS = {('separate','en'):'分开骨架·英文', ('separate','zh'):'分开骨架·中文',
          ('separate_depth','en'):'＋共同深度·英文', ('separate_depth','zh'):'＋共同深度·中文'}
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
DESCRIPTORS_ZH = {
    'pair22': ('穿深酒红色毛衣、花纹短裤和黑色运动鞋的短发人物',
               '穿浅蓝色毛衣、米色长裤和棕色短靴的齐耳短发人物'),
    'pair28': ('穿深蓝色长袖上衣、米色长裤和深色运动鞋的高个金色短发人物',
               '穿白色T恤、灰棕色长裤和浅色运动鞋的红棕色卷发人物'),
    'pair12': ('穿深蓝色长袖衬衫、黑色长裤和棕色凉鞋，有深色波浪发和胡须的人物',
               '穿黑色短袖T恤、黑色长裤和浅灰色运动鞋的短发人物'),
}
# Each semantic unit is authored in both languages; no automatic rewriting.
CLAUSES = [
    ('roles',
     'Edit image 1. It is the only source of the two people, their appearances and the scene. Image 2 is the target skeleton for the first assigned person; image 3 is the target skeleton for the second assigned person. Both skeletons describe these same two original people, not additional people.',
     '编辑图1。图1是两个人物、人物外观和场景的唯一来源。图2是第一位指定人物的目标骨架，图3是第二位指定人物的目标骨架。两张骨架描述的就是图1原有的这两个人，不是额外的人物。'),
    ('shared_coordinates',
     'Images 2 and 3 share one full-scene coordinate system. Use them together to determine each person\'s target position, spacing, body and head orientation, and natural interaction. Do not independently center, crop, resize or reposition either skeleton.',
     '图2和图3使用同一个完整场景坐标系。结合两张骨架确定各自的目标位置、彼此距离、身体和头部朝向以及自然交互。不要分别将骨架居中、裁切、缩放或重新定位。'),
    ('move_existing_people',
     'Move and repose the two original people into their assigned target poses and positions. Do not leave their original bodies in place and generate other people to perform the poses. Do not retain copies, doubles or substitutes. Restore a plausible continuation of the background wherever their movement exposes their original locations.',
     '把原有的两个人移动并调整到各自指定的目标姿势和位置。不要把他们原来的身体留在原处，再另外生成人物执行目标姿势。不要保留人物的副本、分身或替身。人物移动后原位置露出的区域，应恢复为合理连续的背景。'),
    ('preserve_appearance',
     'Keep each original person\'s recognizable face, skin tone, hairstyle, facial hair, clothing, footwear, body shape and proportions. Keep the scene, floor, camera view and lighting of image 1. Do not swap or blend the two identities, outfits or body types.',
     '保留每位原人物可辨认的脸、肤色、发型、胡须、服装、鞋子、体型和身体比例。保留图1的场景、地面、相机视角和光照。不要交换或混合两个人的身份、服装或体型。'),
    ('anatomy_and_count',
     'The final photograph must contain exactly these TWO original people, with natural anatomy and plausible contact and occlusion. Do not add people, heads or limbs. A missing skeleton joint does not mean that a body part is absent; keep the person anatomically complete with plausible occlusion.',
     '最终照片必须恰好包含这两位原人物，人体结构自然，接触和遮挡合理。不要增加人物、头部或肢体。骨架中没有标出的关节不表示人体缺少该部位；保持人体结构完整，并采用合理遮挡。'),
    ('photo_only',
     'Colored skeleton lines and white facial dots provide pose only, not appearance or texture. Return one natural photorealistic image. Do not show skeleton lines, landmark dots, labels, letters, numbers, depth-map greyscale, reference drawings, panels or a collage.',
     '彩色骨架线和白色面部点只提供姿态，不提供外观或纹理。只输出一张自然真实的照片。不要显示骨架线、关键点、标签、字母、数字、深度灰度、参考示意图、分栏或拼图。'),
]
DEPTH_CLAUSE = ('joint_depth',
    'Image 4 is the joint depth map of the SAME two-person target arrangement, aligned with images 2 and 3. Brighter nonblack human surfaces are nearer the camera; darker human surfaces are farther away; black is background. Use it only for relative surface depth and which body parts occlude others. It does not supply additional people, identities, clothing, skin colors, lighting or output textures.',
    '图4是同一对人物目标布局的共同深度图，与图2、图3对齐。非黑色人体表面越亮越靠近相机，越暗越远离相机，黑色为背景。它只提供表面相对远近和肢体遮挡关系，不提供额外人物、身份、服装、肤色、光照或输出纹理。')


def key(row):
    return row['pair_id'], row['method'], row['language']


def prompt_units(case, method):
    a, b = case['source']['descriptors']
    az, bz = DESCRIPTORS_ZH[case['source']['key'][0]]
    binding = ('binding',
        f'In image 1, assign {a} to the skeleton in IMAGE 2. Assign {b} to the skeleton in IMAGE 3. Follow these appearance-based assignments, not the original left/right order, gender or body size.',
        f'图1中，{az}对应图2的骨架，{bz}对应图3的骨架。按这些外观描述确定对应关系，不按原来的左右顺序、性别或体型匹配。')
    return [CLAUSES[0], binding, CLAUSES[1], *([DEPTH_CLAUSE] if method=='separate_depth' else []), *CLAUSES[2:]]


def prompt_for(case, method, language):
    at = 1 if language=='en' else 2
    return '\n\n'.join(unit[at] for unit in prompt_units(case, method))


def skeleton_xy(xy, scores):
    valid = np.asarray(scores) > .3
    assert np.asarray(xy).shape == (134,2) and valid.shape == (134,)
    assert np.isfinite(xy).all() and np.isfinite(scores).all()
    canvas=np.zeros((1024,1024,3),np.uint8)
    colors=[tuple(round(.75*v+.25*255) for v in c) for c in COLORS]
    edges=[]
    def line(a,b,color,width):
        if valid[a] and valid[b]:
            cv2.line(canvas,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),color,width,cv2.LINE_AA)
            edges.append([int(a),int(b),width])
    for i,(a,b) in enumerate(BODY_EDGES):line(a,b,colors[i],2)
    for i in range(18):
        if valid[i]:cv2.circle(canvas,tuple(np.rint(xy[i]).astype(int)),1,colors[i],-1,cv2.LINE_AA)
    for offset in (92,113):
        for finger in range(5):
            ids=[offset,*range(offset+1+finger*4,offset+5+finger*4)]
            for j,(a,b) in enumerate(zip(ids,ids[1:])):line(a,b,colors[(finger*4+j)%18],1)
    for a,b in ((13,18),(18,19),(19,20),(20,13),(10,21),(21,22),(22,23),(23,10)):
        line(a,b,(255,255,255),1)
    for i in range(24,92):
        if valid[i]:cv2.circle(canvas,tuple(np.rint(xy[i]).astype(int)),1,(255,255,255),-1,cv2.LINE_AA)
    return Image.fromarray(canvas), {'valid_point_indices':np.flatnonzero(valid).tolist(),
        'drawn_edges':edges,'threshold':.3,'body_line_width':2,'hand_foot_line_width':1,
        'face_point_radius':1,'body_point_radius':1,'labels_drawn':False,'confidence_affects_brightness':False}


def target_overlay(image, merged):
    base=np.asarray(image.convert('RGB'));pose=np.asarray(merged)
    if base.shape!=(1024,1024,3) or pose.shape!=base.shape:raise ValueError('Overlay requires the unmodified shared1024 canvas')
    mask=pose.max(-1)>0;out=base.copy()
    out[mask]=np.rint(.8*pose[mask].astype(float)+.2*base[mask].astype(float)).astype(np.uint8)
    return Image.fromarray(out)


def references(root, job):
    names=['A','S1','S2']+(['depth_joint'] if job['method']=='separate_depth' else [])
    return [{'index':i+1,'role':n,'path':str(root/'inputs/1024'/job['pair_id']/f'{n}.png'),
             'sha256':sha(root/'inputs/1024'/job['pair_id']/f'{n}.png')} for i,n in enumerate(names)]


def self_check():
    xy=np.column_stack((np.linspace(180,800,134),np.linspace(100,900,134)))
    a,info=skeleton_xy(xy,np.full(134,.4));b,_=skeleton_xy(xy,np.full(134,.99))
    assert np.array_equal(a,b)
    scores=np.full(134,.9);scores[113]=.3;_,info=skeleton_xy(xy,scores)
    assert 113 not in info['valid_point_indices'] and not any(113 in e[:2] for e in info['drawn_edges'])
    raw=Image.new('RGB',(1024,1024),(100,80,60));pose=np.zeros((1024,1024,3),np.uint8);pose[300,400]=(250,100,50)
    over=np.asarray(target_overlay(raw,pose));assert tuple(over[300,400])==(220,96,52)
    assert tuple(over[0,0])==(100,80,60) and np.count_nonzero((over!=np.asarray(raw)).any(-1))==1
    return {'passed':True,'checks':['strict_threshold_0.3','fixed_brightness','thin_widths','overlay_80_20','black_background_not_pasted']}


def prepare(root):
    if (root/'experiment_lock.json').exists() or (root/'generations.jsonl').exists():raise ValueError('Experiment already locked; use validation/resume')
    previous=legacy.verify_lock(PREVIOUS)
    root.mkdir(parents=True,exist_ok=True)
    cases=json.loads((PREVIOUS/'data/cases.json').read_text())
    assert [c['pair_id'] for c in cases]==[f'hi4d_{i:02d}' for i in range(6)]
    protected=dict(previous['protected_files'])
    for path in (PREVIOUS/'data/cases.json', PREVIOUS/'experiment_lock.json', PREVIOUS/'generations.jsonl'):
        protected[str(path)]=sha(path)
    for row in read_rows(PREVIOUS/'generations.jsonl'):protected[row['image']]=sha(row['image'])
    old_report=json.loads((PREVIOUS/'report_checks.json').read_text())['report']
    protected[old_report]=sha(old_report)
    qc=[]
    for c in cases:
        old=PREVIOUS/'inputs/1024'/c['pair_id'];folder=root/'inputs/1024'/c['pair_id'];folder.mkdir(parents=True,exist_ok=True)
        meta=copy.deepcopy(json.loads((old/'metadata.json').read_text()));renders=[]
        for name in ('A.png','B.png','depth_joint.png','depth_raw.npz','depth_native_grey.png','B_depth_outline.png'):
            shutil.copy2(old/name,folder/name);protected[str(old/name)]=sha(old/name)
            assert sha(folder/name)==sha(old/name)
        for person in ('S1','S2'):
            info=meta[person];xy=np.asarray(info['mapped_coordinates']);scores=np.asarray(info['scores'])
            im,new=skeleton_xy(xy,scores)
            assert new['valid_point_indices']==info['valid_point_indices']
            assert [e[:2] for e in new['drawn_edges']]==[e[:2] for e in info['drawn_edges']]
            new.update({k:v for k,v in info.items() if k not in new})
            meta[person]=new;im.save(folder/f'{person}.png');renders.append(np.asarray(im))
            protected[str(old/f'{person}.png')]=sha(old/f'{person}.png')
        merged=np.maximum(*renders);Image.fromarray(merged).save(folder/'combined_target.png')
        target_overlay(Image.open(folder/'B.png'),merged).save(folder/'B_target_overlay.png')
        write(folder/'metadata.json',meta)
        qc.append({'pair_id':c['pair_id'],'valid_points':[len(meta[p]['valid_point_indices']) for p in ('S1','S2')],
                   'coordinates_and_valid_set_unchanged':True,'depth_bytes_unchanged':True,'known_pose_defects_retained':True})
        sheet=Image.new('RGB',(5*380,430),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,20)
        for i,name in enumerate(('B','S1','S2','B_target_overlay','depth_joint')):
            sheet.paste(Image.open(folder/f'{name}.png').resize((380,380)),(380*i,45));draw.text((380*i+6,6),name,font=font,fill='black')
        out=root/'checks'/f'{c["pair_id"]}_inputs.jpg';out.parent.mkdir(parents=True,exist_ok=True);sheet.save(out,quality=96)
    jobs=[{'pair_id':c['pair_id'],'resolution':1024,'method':m,'language':l,'mapping':'direct'} for c in cases for m,l in ARMS]
    prompts={c['pair_id']:{m:{l:prompt_for(c,m,l) for l in LANGUAGES} for m in METHODS} for c in cases}
    semantics={c['pair_id']:{m:[{'id':u[0],'en':u[1],'zh':u[2]} for u in prompt_units(c,m)] for m in METHODS} for c in cases}
    write(root/'data/cases.json',cases);write(root/'data/jobs.json',jobs)
    write(root/'prompts/locked_prompts.json',prompts);write(root/'prompts/semantic_alignment.json',semantics)
    write(root/'checks/input_checks.json',qc);write(root/'checks/synthetic_checks.json',self_check())
    from scripts.eval.eval_qwen21_bfs_skeleton import sample
    from scripts.eval.eval_qwen21_native_value import SamplingWitness
    runtime={str(Path(__file__).resolve()):sha(__file__)}
    for obj in (legacy,sample,SamplingWitness):
        path=inspect.getfile(obj);runtime[path]=sha(path)
    runtime[str(PROJECT/'src/data/letterbox.py')]=sha(PROJECT/'src/data/letterbox.py')
    locked={str(p):sha(p) for directory in ('data','prompts') for p in (root/directory).rglob('*') if p.is_file()}
    prepared={str(p):sha(p) for p in (root/'inputs').rglob('*') if p.is_file()}
    write(root/'experiment_lock.json',{'protocol':'hi4d_bilingual_thin_v2','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'runtime_versions':versions(),'model':str(MODEL.resolve()),'model_revision':REVISION,
        'protected_files':protected,'model_metadata':previous['model_metadata'],'runtime_sources':runtime,
        'locked_data':locked,'prepared_files':prepared,'formal_expected':24,'training':False,'swap':False,
        'prompt_rewrite':False,'detailed_pose_text':False,'manual_occlusion_text':False,
        'reference_order':{'separate':['A','S1','S2'],'separate_depth':['A','S1','S2','depth_joint']},
        'overlay':'actual input target skeletons; unchanged scene coordinates;80% skeleton/20% generated RGB; display only'})
    print(json.dumps({'prepared':24,'root':str(root)},ensure_ascii=False),flush=True)


def verify_lock(root):
    return legacy.verify_lock(root)


def validate_row(row,job,prompt,refs,lock_sha):
    expected={**job,'steps':30,'seed':42,'true_cfg_scale':1.,'use_kv_cache':False,
        'control_forwards':0,'lora_loaded':False,'prompt':prompt,'references':refs,
        'lock_sha256':lock_sha,'runtime_versions':versions(),'target_rgb_in_native_references':False}
    for k,v in expected.items():
        if row.get(k)!=v:raise ValueError(f'Saved configuration differs: {key(job)} {k}')
    if row['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():raise ValueError('Prompt digest differs')
    if sha(row['image'])!=row['image_sha256'] or Image.open(row['image']).size!=(1024,1024):raise ValueError('Invalid saved output')


def run(args):
    import torch
    from diffusers import QwenImage21Pipeline
    from scripts.eval.eval_qwen21_bfs_skeleton import sample
    root=args.output;verify_lock(root);lock_sha=sha(root/'experiment_lock.json')
    approved=json.loads((root/'checks/pre_generation_review.json').read_text())
    if not approved.get('passed') or approved['lock_sha256']!=lock_sha:raise ValueError('Input and bilingual review required')
    jobs=json.loads((root/'data/jobs.json').read_text())
    if args.group!='all':jobs=[j for j in jobs if j['method']==args.group]
    if args.language!='all':jobs=[j for j in jobs if j['language']==args.language]
    if args.max_samples:jobs=jobs[:args.max_samples]
    prompts=json.loads((root/'prompts/locked_prompts.json').read_text())
    rows=read_rows(root/'generations.jsonl') if (root/'generations.jsonl').exists() else [];done={key(r):r for r in rows}
    if len(done)!=len(rows):raise ValueError('Duplicate generation records')
    pending=[]
    for j in jobs:
        if key(j) in done:validate_row(done[key(j)],j,prompts[j['pair_id']][j['method']][j['language']],references(root,j),lock_sha)
        else:pending.append(j)
    if not pending:return
    torch.set_num_threads(4);torch.manual_seed(42)
    print(json.dumps({'event':'loading','pending':len(pending),'cuda_free_total_gib':[x/2**30 for x in torch.cuda.mem_get_info()]}),flush=True)
    pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),dtype=torch.bfloat16).to('cuda')
    for module in (pipe.transformer,pipe.text_encoder,pipe.vae):module.eval().requires_grad_(False)
    if getattr(pipe.transformer,'peft_config',None) or getattr(pipe.transformer,'_pic_edit_qwen21_control_active',False):raise AssertionError('Unexpected control/LoRA')
    pipe.set_progress_bar_config(disable=True)
    for job in pending:
        prompt=prompts[job['pair_id']][job['method']][job['language']];refs=references(root,job)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        output,witness=sample(pipe,prompt,[Image.open(r['path']).convert('RGB') for r in refs],1024)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        image=output.images[0].convert('RGB');assert image.size==(1024,1024)
        path=root/'images/1024'/f'{job["pair_id"]}_{job["method"]}_{job["language"]}.png';path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        if witness['native_transformer_calls']!=30 or witness['control_context_installed'] or witness['target_token_count']!=4096:raise AssertionError('Sampling protocol violation')
        if witness['native_prefix_image_token_counts']!=[4096*len(refs)]:raise AssertionError('Reference image token count differs')
        row={**job,'steps':30,'seed':42,'true_cfg_scale':1.,'use_kv_cache':False,'control_forwards':0,'lora_loaded':False,
            'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'references':refs,'lock_sha256':lock_sha,
            'runtime_versions':versions(),'image':str(path),'image_sha256':sha(path),'seconds':seconds,
            'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'correctness_observation':witness,'assignment_source_to_target':[0,1],'target_rgb_in_native_references':False,'model_revision':REVISION}
        validate_row(row,job,prompt,refs,lock_sha)
        with (root/'generations.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
        done[key(job)]=row
        write(root/'status.json',{'completed':len(done),'expected':24,'latest':job,'seconds':seconds,'peak_cuda_gib':row['peak_cuda_gib']})
        print(json.dumps({'completed':len(done),'expected':24,**job,'seconds':seconds},ensure_ascii=False),flush=True)
        del output
    assert all(not p.requires_grad and p.grad is None for m in (pipe.transformer,pipe.text_encoder,pipe.vae) for p in m.parameters())
    verify_lock(root)
    write(root/'runtime_passed.json',{'passed':True,'formal_completed':len(done),'all_frozen':True,'no_optimizer':True,
        'no_lora':True,'no_project_adapter':True,'old_experiment_preserved':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--group',choices=('all',*METHODS),default='all')
    parser.add_argument('--language',choices=('all',*LANGUAGES),default='all')
    parser.add_argument('--max-samples',type=int,default=0)
    args=parser.parse_args()
    if args.prepare:prepare(args.output)
    elif args.validate_only:verify_lock(args.output);print(json.dumps(self_check()))
    else:run(args)


if __name__=='__main__':main()
