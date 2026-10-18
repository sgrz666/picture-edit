#!/usr/bin/env python3
"""Five locked native Hi4D input combinations; English, inference only."""
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
from src.data.letterbox import Letterbox
from scripts.eval import eval_hi4d_separate_depth_bilingual as previous
from scripts.eval.eval_hi4d_native_binding import MODEL, REVISION, sha, write, read_rows, versions

DEFAULT = Path('/home/shangguanrz/nihaochen/qwen21_full/hi4d_depth_combinations_20261013')
PREVIOUS = previous.DEFAULT
METHODS = ('skeleton', 'skeleton_joint_depth', 'joint_depth', 'separate_depth', 'joint_separate_depth')
ARMS = tuple((m, 'en') for m in METHODS)
LABELS = dict(zip(ARMS, ('分开骨架', '骨架＋合并深度', '合并深度', '分开深度', '合并＋分开深度')))
FONT = previous.FONT
REF_NAMES = {
    'skeleton': ['A', 'S1', 'S2'],
    'skeleton_joint_depth': ['A', 'S1', 'S2', 'depth_joint'],
    'joint_depth': ['A', 'depth_joint'],
    'separate_depth': ['A', 'D1', 'D2'],
    'joint_separate_depth': ['A', 'depth_joint', 'D1', 'D2'],
}
COMMON = {
    'edit': 'Edit IMAGE 1 into one natural photograph of exactly the SAME TWO people. IMAGE 1 is the sole source of their identities, faces, skin tones, hairstyles, facial hair, clothes, footwear and body proportions, and of the scene, camera view and lighting.',
    'coordinates': 'All geometry references share one full-scene canvas. Use their existing coordinates together for the target positions, distance, body articulation, head directions and interaction. Preserve each original person\'s own body proportions while adopting the assigned pose. Do not treat separate references as separate scenes or independently center or enlarge their people.',
    'same_people': 'Every geometry reference describes the target arrangement of these SAME TWO original people. Repeated views of a figure do not represent additional people. Geometry provides pose and spatial structure only, not identity, clothing, skin color or photographic texture.',
    'move': 'Move and repose the two original people into their assigned target positions. Remove their old bodies from the old positions; do not leave the originals standing there while adding doubles, substitutes or other people to perform the target poses. Restore the newly exposed areas as a plausible continuation of the original background.',
    'preserve': 'Keep each person\'s recognizable face, hair, skin tone, clothing, footwear and body shape from IMAGE 1. Keep the original background, floor, camera view and lighting. Keep the two identities distinct, with natural anatomy and plausible contact and occlusion. There must be exactly two people, without duplicated heads, hands, arms, legs or other extra body parts. Incomplete reference marks do not imply missing anatomy.',
    'canvas': 'Black areas in the geometry references are empty reference canvas, not the background of the output. Preserve the background and lighting of image 1. Do not copy the grey shading or black canvas into the edited photograph.',
    'output': 'Return only the edited IMAGE 1 as a single natural photorealistic photograph. Do not draw skeleton lines, facial landmark dots, labels, reference figures, grey geometric bodies, panels or a collage.',
}


def key(row):
    return row['pair_id'], row['method'], row['language']


def prompt_units(case, method):
    a, b = case['source']['descriptors']
    right = case['pair_id'] in ('hi4d_00', 'hi4d_04')
    loc1, loc2 = ('right', 'left') if right else ('left', 'right')
    if method in ('skeleton', 'skeleton_joint_depth'):
        roles = 'IMAGE 2 contains only the first target skeleton; IMAGE 3 contains only the second target skeleton. Colored lines and white dots specify joints and facial landmarks, not appearance.'
        binding = f'In IMAGE 1, assign {a} to the skeleton in IMAGE 2. Assign {b} to the skeleton in IMAGE 3. These explicit assignments override guesses based on original left/right order, gender or relative body size.'
        if method == 'skeleton_joint_depth':
            roles += ' IMAGE 4 is the jointly rendered depth map of the same two-person target interaction, aligned with the two skeleton images.'
    elif method == 'joint_depth':
        roles = 'IMAGE 2 is the jointly rendered depth map of the two-person target interaction. Its two human figures are target pose guides for the two people in IMAGE 1.'
        binding = f'In IMAGE 1, assign {a} to the target figure whose HEAD is farther to the {loc1} in IMAGE 2. Assign {b} to the other figure, whose HEAD is farther to the {loc2} in IMAGE 2. Left/right here identifies the target heads in IMAGE 2 only; it does not refer to the people\'s original positions in IMAGE 1 or change the target arrangement.'
    elif method == 'separate_depth':
        roles = 'IMAGE 2 shows only the first target person\'s depth surface; IMAGE 3 shows only the second target person\'s depth surface. Each individual surface is rendered on the same full-scene canvas and retains its target position.'
        binding = f'In IMAGE 1, assign {a} to the target depth figure in IMAGE 2. Assign {b} to the target depth figure in IMAGE 3. These explicit assignments override guesses based on original left/right order, gender or relative body size.'
    elif method == 'joint_separate_depth':
        roles = 'IMAGE 2 is the jointly rendered depth map of the target two-person interaction. IMAGE 3 shows only the first target person\'s depth surface; IMAGE 4 shows only the second target person\'s depth surface. These are joint and individual views of the SAME TWO target figures, not four different people. IMAGE 2 specifies their jointly visible surfaces and mutual occlusion; IMAGES 3 and 4 identify each figure separately.'
        binding = f'In IMAGE 1, assign {a} to the target depth figure in IMAGE 3, which is the figure whose HEAD is farther to the {loc1} in IMAGE 2. Assign {b} to the target depth figure in IMAGE 4, whose HEAD is farther to the {loc2} in IMAGE 2. The individual images retain their full-scene positions; they are not centered portraits.'
    else:
        raise ValueError(method)
    units = [('edit', COMMON['edit']), ('roles', roles), ('binding', binding),
             ('coordinates', COMMON['coordinates']), ('same_people', COMMON['same_people'])]
    if method != 'skeleton':
        units.append(('depth', 'All depth references use the same camera and grey scale: brighter human surfaces are nearer the camera and darker human surfaces are farther away. Use depth for relative surface distance and occlusion. Grey is a measurement code, not skin, clothing, lighting or the intended photographic appearance.'))
    units.extend((k, COMMON[k]) for k in ('move', 'preserve', 'canvas', 'output'))
    return units


def prompt_for(case, method):
    return '\n\n'.join(text for _, text in prompt_units(case, method))


def depth_grey(depth, lo, hi):
    depth = np.asarray(depth)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth < 0) or not hi > lo:
        raise ValueError('Invalid depth or shared normalization range')
    mask = depth > 0
    if np.any(depth[mask] < lo - 1e-4) or np.any(depth[mask] > hi + 1e-4):
        raise ValueError('Individual depth exceeds the locked joint range')
    grey = np.zeros(depth.shape, np.uint8)
    grey[mask] = np.clip(np.rint(32 + 223 * (hi-depth[mask])/(hi-lo)), 32, 255).astype(np.uint8)
    return Image.fromarray(grey).convert('RGB')


def references(root, job):
    return [{'index':i+1, 'role':name,
             'path':str(root/'inputs/1024'/job['pair_id']/f'{name}.png'),
             'sha256':sha(root/'inputs/1024'/job['pair_id']/f'{name}.png')}
            for i, name in enumerate(REF_NAMES[job['method']])]


def self_check():
    a = np.array([[0., 1., 3.], [0., 0., 2.]], np.float32)
    b = np.array([[0., 2., 2.], [3., 0., 0.]], np.float32)
    scene = np.minimum(np.where(a > 0, a, np.inf), np.where(b > 0, b, np.inf))
    scene[~np.isfinite(scene)] = 0
    ga = np.asarray(depth_grey(a, 1., 3.))[..., 0]
    gb = np.asarray(depth_grey(b, 1., 3.))[..., 0]
    gs = np.asarray(depth_grey(scene, 1., 3.))[..., 0]
    assert ga[0,1] == 255 and ga[0,2] == 32 and ga[0,0] == 0
    assert ga[1,2] == gb[0,1] and np.array_equal(np.maximum(ga, gb), gs)
    assert np.array_equal(scene > 0, (a > 0) | (b > 0))
    previous.self_check()
    return {'passed':True, 'shared_depth_scale':True, 'nearer_surface_brighter':True,
            'individual_occluded_surfaces_retained':True, 'thin_skeleton_and_target_overlay':True}


def prepare(root):
    if (root/'experiment_lock.json').exists() or (root/'generations.jsonl').exists():
        raise ValueError('Experiment is already locked; use validation or resume')
    old = previous.verify_lock(PREVIOUS)
    cases = json.loads((PREVIOUS/'data/cases.json').read_text())
    assert [c['pair_id'] for c in cases] == [f'hi4d_{i:02d}' for i in range(6)]
    protected = dict(old['protected_files'])
    for p in (PREVIOUS/'data/cases.json', PREVIOUS/'experiment_lock.json', PREVIOUS/'generations.jsonl',
              PREVIOUS/'report_checks.json', PREVIOUS/'preview_report_checks.json'):
        protected[str(p)] = sha(p)
    for row in read_rows(PREVIOUS/'generations.jsonl'):
        protected[row['image']] = sha(row['image'])
    for name in ('report_checks.json', 'preview_report_checks.json'):
        p = Path(json.loads((PREVIOUS/name).read_text())['report']);protected[str(p)] = sha(p)
    qc = []
    for c in cases:
        pid = c['pair_id'];oldfolder = PREVIOUS/'inputs/1024'/pid;folder = root/'inputs/1024'/pid
        folder.mkdir(parents=True, exist_ok=True)
        meta = copy.deepcopy(json.loads((oldfolder/'metadata.json').read_text()))
        for name in ('A.png','B.png','S1.png','S2.png','depth_joint.png','depth_raw.npz','depth_native_grey.png',
                     'B_depth_outline.png','combined_target.png','B_target_overlay.png'):
            src = oldfolder/name;shutil.copy2(src, folder/name);protected[str(src)] = sha(src)
            assert sha(folder/name) == sha(src)
        for person in ('S1', 'S2'):
            info = meta[person];xy = np.asarray(info['mapped_coordinates']);scores = np.asarray(info['scores'])
            drawing, fresh = previous.skeleton_xy(xy, scores)
            assert np.array_equal(np.asarray(drawing), np.asarray(Image.open(folder/f'{person}.png')))
            assert fresh['valid_point_indices'] == info['valid_point_indices']
            assert fresh['body_line_width'] == 2 and fresh['face_point_radius'] == 1
        with np.load(folder/'depth_raw.npz', allow_pickle=False) as z:
            joint, di = previous.legacy.depth_display(z['depth_scene'], z['depth_A'], z['depth_B'])
            lo, hi = meta['depth_joint']['shared_depth_range_metres']
            assert di['shared_depth_range_metres'] == [lo, hi]
            assert np.array_equal(np.asarray(joint), np.asarray(Image.open(folder/'depth_native_grey.png')))
            transform = Letterbox.from_size(joint.size, 1024)
            assert transform.metadata() == meta['B']['transform'] == meta['depth_joint']['transform']
            assert np.array_equal(np.asarray(transform.image(joint, fill=(0,0,0))), np.asarray(Image.open(folder/'depth_joint.png')))
            with np.load(c['target']['smpl']) as smpl, np.load(c['target']['camera']) as cam:
                at = np.flatnonzero(cam['ids'] == 4);assert len(at) == 1
                K, E = cam['intrinsics'][at[0]], cam['extrinsics'][at[0]]
                assert np.allclose(K, z['K_runtime'], atol=1e-4)
                head3d = smpl['joints_3d'][:,15,:]
                q = head3d @ E[:,:3].T + E[:,3];assert np.all(q[:,2] > 0)
                q = q @ K.T;native_xy = q[:,:2] / q[:,2:]
                head_xy = transform.points(native_xy / np.asarray(joint.size)) * 1024
            expect_right = pid in ('hi4d_00', 'hi4d_04')
            assert bool(head_xy[0,0] > head_xy[1,0]) == expect_right
            masks = [np.asarray(Image.open(p).convert('L')) > 0 for p in c['target']['masks']]
            individual_info = []
            for i, name in enumerate(('depth_A', 'depth_B')):
                depth = z[name];native = depth_grey(depth, lo, hi)
                native.save(folder/f'D{i+1}_native.png')
                transform.image(native, fill=(0,0,0)).save(folder/f'D{i+1}.png')
                fg = depth > 0
                ious = [float((fg&m).sum()/(fg|m).sum()) for m in masks]
                if ious[i] < .6 or ious[i] <= ious[1-i]:raise ValueError(f'Depth person mapping uncertain: {pid} {name} {ious}')
                info = {'raw_key':name, 'person_index':i, 'shared_depth_range_metres':[lo,hi],
                        'transform':transform.metadata(), 'full_individual_surface':True,
                        'segmentation_iou':ious, 'projected_smpl_head_at_1024':head_xy[i].tolist()}
                meta[f'D{i+1}'] = info;individual_info.append(info)
            meta['target_head_mapping'] = {'source_to_target':[0,1], 'source_person_1_head_side':'right' if expect_right else 'left',
                'projected_smpl_heads':head_xy.tolist(), 'source':'original target SMPL person indices; no pose refit'}
        write(folder/'metadata.json', meta)
        sheet = Image.new('RGB',(7*340,405),'white');draw = ImageDraw.Draw(sheet);font = ImageFont.truetype(FONT,22)
        for i, name in enumerate(('B','S1','S2','depth_joint','D1','D2','B_target_overlay')):
            sheet.paste(Image.open(folder/f'{name}.png').resize((340,340)),(i*340,45))
            draw.text((i*340+5,8),f'{pid} {name}',font=font,fill='black')
        preview = root/'checks'/f'{pid}_inputs.jpg';preview.parent.mkdir(parents=True,exist_ok=True);sheet.save(preview,quality=96)
        qc.append({'pair_id':pid, 'individual_depth':individual_info, 'depth_joint_bytes_unchanged':True,
                   'skeleton_bytes_coordinates_and_valid_set_unchanged':True,
                   'target_head_mapping':meta['target_head_mapping'], 'joint_nearest_surface_check':di})
    jobs = [{'pair_id':c['pair_id'],'resolution':1024,'method':m,'language':'en','mapping':'direct'} for c in cases for m in METHODS]
    prompts = {c['pair_id']:{m:{'en':prompt_for(c,m)} for m in METHODS} for c in cases}
    semantic = {c['pair_id']:{m:dict(prompt_units(c,m)) for m in METHODS} for c in cases}
    write(root/'data/cases.json',cases);write(root/'data/jobs.json',jobs)
    write(root/'prompts/locked_prompts.json',prompts);write(root/'prompts/semantic_units.json',semantic)
    write(root/'checks/input_checks.json',qc);write(root/'checks/synthetic_checks.json',self_check())
    from scripts.eval.eval_qwen21_bfs_skeleton import sample
    from scripts.eval.eval_qwen21_native_value import SamplingWitness
    runtime = {str(Path(__file__).resolve()):sha(__file__)}
    for obj in (previous, previous.legacy, sample, SamplingWitness, Letterbox):
        p = inspect.getfile(obj);runtime[p] = sha(p)
    write(root/'experiment_lock.json', {
        'protocol':'hi4d_depth_combinations_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'model':str(MODEL.resolve()),'model_revision':REVISION,'runtime_versions':versions(),
        'model_metadata':old['model_metadata'],'protected_files':protected,'runtime_sources':runtime,
        'locked_data':{str(p):sha(p) for d in ('data','prompts') for p in (root/d).rglob('*') if p.is_file()},
        'prepared_files':{str(p):sha(p) for p in (root/'inputs').rglob('*') if p.is_file()},
        'formal_expected':30,'reference_order':REF_NAMES,'training':False,'swap':False,'prompt_rewrite':False,
        'detailed_pose_text':False,'manual_occlusion_text':False,'language':'en','all_five_groups_regenerated':True,
        'overlay':'B target skeleton, unchanged coordinates, 80% skeleton/20% output, display only; not depth-route input'})
    print(json.dumps({'prepared':30,'root':str(root)},ensure_ascii=False),flush=True)


def verify_lock(root):
    return previous.verify_lock(root)


def validate_row(row, job, prompt, refs, lock_sha):
    previous.validate_row(row, job, prompt, refs, lock_sha)
    if row.get('model_revision') != REVISION:raise ValueError('Base revision differs')
    expected = previous.versions()
    if row.get('runtime_versions') != expected:raise ValueError('Environment differs')


def run(args):
    import torch
    from diffusers import QwenImage21Pipeline
    from scripts.eval.eval_qwen21_bfs_skeleton import sample
    root=args.output;verify_lock(root);lock_sha=sha(root/'experiment_lock.json')
    approved=json.loads((root/'checks/pre_generation_review.json').read_text())
    if not approved.get('passed') or approved['lock_sha256']!=lock_sha:raise ValueError('Input/prompt review required')
    jobs=json.loads((root/'data/jobs.json').read_text())
    if args.variant!='all':jobs=[j for j in jobs if j['method']==args.variant]
    prompts=json.loads((root/'prompts/locked_prompts.json').read_text())
    rows=read_rows(root/'generations.jsonl') if (root/'generations.jsonl').exists() else []
    done={key(r):r for r in rows}
    if len(done)!=len(rows):raise ValueError('Duplicate results')
    pending=[]
    for j in jobs:
        if key(j) in done:validate_row(done[key(j)],j,prompts[j['pair_id']][j['method']]['en'],references(root,j),lock_sha)
        else:pending.append(j)
    if not pending:return
    torch.set_num_threads(4);torch.manual_seed(42)
    print(json.dumps({'event':'loading','pending':len(pending),'cuda_free_total_gib':[x/2**30 for x in torch.cuda.mem_get_info()]}),flush=True)
    pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),dtype=torch.bfloat16).to('cuda')
    for m in (pipe.transformer,pipe.text_encoder,pipe.vae):m.eval().requires_grad_(False)
    if getattr(pipe.transformer,'peft_config',None) or getattr(pipe.transformer,'_pic_edit_qwen21_control_active',False):raise AssertionError('Unexpected Adapter/LoRA')
    pipe.set_progress_bar_config(disable=True)
    for j in pending:
        prompt=prompts[j['pair_id']][j['method']]['en'];refs=references(root,j)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        output,witness=sample(pipe,prompt,[Image.open(r['path']).convert('RGB') for r in refs],1024)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        image=output.images[0].convert('RGB');assert image.size==(1024,1024)
        path=root/'images/1024'/f'{j["pair_id"]}_{j["method"]}_en.png';path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        if witness['native_transformer_calls']!=30 or witness['control_context_installed'] or witness['target_token_count']!=4096:raise AssertionError('Invalid native sampling')
        if witness['native_prefix_image_token_counts']!=[4096*len(refs)]:raise AssertionError('Reference count differs')
        row={**j,'steps':30,'seed':42,'true_cfg_scale':1.,'use_kv_cache':False,'control_forwards':0,'lora_loaded':False,
            'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'references':refs,'lock_sha256':lock_sha,
            'runtime_versions':versions(),'image':str(path),'image_sha256':sha(path),'seconds':seconds,
            'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'correctness_observation':witness,'assignment_source_to_target':[0,1],
            'target_rgb_in_native_references':False,'model_revision':REVISION}
        validate_row(row,j,prompt,refs,lock_sha)
        with (root/'generations.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
        done[key(j)]=row
        write(root/'status.json',{'completed':len(done),'expected':30,'latest':j,'seconds':seconds,'peak_cuda_gib':row['peak_cuda_gib']})
        print(json.dumps({'completed':len(done),'expected':30,**j,'seconds':seconds},ensure_ascii=False),flush=True)
        del output
    assert all(not p.requires_grad and p.grad is None for m in (pipe.transformer,pipe.text_encoder,pipe.vae) for p in m.parameters())
    verify_lock(root)
    write(root/'runtime_passed.json',{'passed':True,'formal_completed':len(done),'all_frozen':True,'no_optimizer':True,
        'no_lora':True,'no_project_adapter':True,'old_experiment_preserved':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--validate-only',action='store_true')
    p.add_argument('--variant',choices=('all',*METHODS),default='all');args=p.parse_args()
    if args.prepare:prepare(args.output)
    elif args.validate_only:
        lock=verify_lock(args.output);assert lock['formal_expected']==30
        print(json.dumps({'lock_valid':True,'self_check':self_check()},ensure_ascii=False))
    else:run(args)


if __name__=='__main__':main()
