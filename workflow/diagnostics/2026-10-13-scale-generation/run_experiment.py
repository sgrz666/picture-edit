#!/usr/bin/env python3
"""Generate a frozen-prompt source-scale/anchor ablation with the existing native worker."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
WORKFLOW=HERE.parents[1]
sys.path.insert(0,str(WORKFLOW))
from pose_transfer.common import (atomic_json,file_lock,finish_stage,make_request,read_json,
    sha256_file,stage_outputs,stage_valid)
from pose_transfer.geometry import valid_points
from pose_transfer.prompts import validate_generator_prompt
from geometry_ablation import fit_to_source_anchor,render_clipped_guide
import run as workflow_cli

CASES=('hard_01','hard_02','hard_04','hard_05','medium_02')
ROOT=WORKFLOW/'runs/scale_anchor_20261013'
POSE=WORKFLOW/'diagnostics/2026-10-13-scale'
SCHEMA='source_anchor_geometry_ablation_v1'


def prepare(name,verify=False):
    baseline=WORKFLOW/'runs'/name
    destination=ROOT/name
    old=read_json(baseline/'generation/generation.json')
    settings=read_json(baseline/'run.json')['settings']
    for ref in old['reference_images']:
        file={'A_reference':baseline/'inputs/A_reference.png',
            'B_3d_dwpose':baseline/'geometry/B_3d_dwpose.png'}[ref['role']]
        if sha256_file(file)!=ref['sha256']:raise RuntimeError(f'{name}: baseline reference changed')
    prompt=baseline/'prompt/generation_prompt.txt'
    if sha256_file(prompt)!=old['prompt_sha256']:
        raise RuntimeError(f'{name}: baseline prompt changed')
    validate_generator_prompt(prompt.read_text())
    original_canvas=baseline/'generation/final_canvas.png'
    if sha256_file(original_canvas)!=read_json(baseline/'generation/stage.json')['outputs']['final_canvas']['sha256']:
        raise RuntimeError(f'{name}: baseline generated canvas changed')
    arrays=baseline/'geometry/B_keypoints.npz'
    visible=POSE/f'{name}_visible_pose.npz'
    prep_identity={'experiment':SCHEMA,'operation':'A shoulder-hip scale + A pelvis + raster clipping',
        'min_source_score':0.3,
        'inputs_sha256':{'baseline_keypoints':sha256_file(arrays),'source_visible_pose':sha256_file(visible),
            'A_reference':sha256_file(baseline/'inputs/A_reference.png'),
            'source_preprocess':sha256_file(baseline/'inputs/source_preprocess.json'),
            'frozen_prompt':sha256_file(prompt)},
        'code':{'geometry_ablation.py':sha256_file(HERE/'geometry_ablation.py'),
            'run_experiment.py':sha256_file(__file__),
            'pose_transfer/geometry.py':sha256_file(WORKFLOW/'pose_transfer/geometry.py')}}
    prep_request=make_request('source_anchor_geometry',prep_identity,destination,{})
    prep_record=destination/'geometry/ablation_stage.json'
    if not stage_valid(prep_record,prep_request,destination):
        if verify:raise RuntimeError(f'{name}: ablation preparation is absent or stale')
        for directory in ('inputs','geometry','prompt','generation'):
            (destination/directory).mkdir(parents=True,exist_ok=True)
        with np.load(arrays) as saved:original={key:saved[key].copy() for key in saved.files}
        with np.load(visible) as measured:
            source=measured['source_raw133'];scores=measured['source_scores']
        aligned,operation=fit_to_source_anchor(original['keypoints_generation_px'],source,scores)
        for source_file,target in [(baseline/'inputs/A_reference.png',destination/'inputs/A_reference.png'),
            (baseline/'inputs/source_preprocess.json',destination/'inputs/source_preprocess.json'),
            (prompt,destination/'prompt/generation_prompt.txt')]:
            shutil.copyfile(source_file,target)
        with Image.open(destination/'inputs/A_reference.png') as im:a=im.convert('RGB')
        guide=render_clipped_guide(aligned,a.size)
        guide.save(destination/'geometry/B_3d_dwpose.png')
        Image.blend(a,guide,0.45).save(destination/'geometry/A_poseguide_alignment.png')
        original['keypoints_generation_px']=aligned
        original['valid_generation']=valid_points(aligned,*a.size)
        np.savez_compressed(destination/'geometry/B_keypoints.npz',**original)
        operation.update({'baseline_run':str(baseline),'A_reference_sha256':sha256_file(baseline/'inputs/A_reference.png'),
            'source_pose_npz':str(visible),'source_pose_sha256':sha256_file(visible),
            'original_keypoints_sha256':sha256_file(arrays),
            'joints_in_frame':int(original['valid_generation'].sum()),
            'body_joints_in_frame':int(original['valid_generation'][:18].sum()),
            'canvas_size':list(a.size),'clipping_policy':'raster clipping only; off-frame joint coordinates retained'})
        atomic_json(destination/'geometry/operation.json',operation)
        atomic_json(destination/'prompt/generation_prompt.json',{'source':'baseline_prompt_frozen',
            'source_path':str(prompt),'source_sha256':sha256_file(prompt),
            'generation_prompt':str(destination/'prompt/generation_prompt.txt'),
            'sha256':sha256_file(destination/'prompt/generation_prompt.txt'),
            'PE_rerun':False,'baseline_PE_source':old.get('prompt_source')})
        outputs={'A_reference':destination/'inputs/A_reference.png',
            'source_preprocess':destination/'inputs/source_preprocess.json',
            'B_3d_dwpose':destination/'geometry/B_3d_dwpose.png',
            'B_keypoints':destination/'geometry/B_keypoints.npz',
            'A_alignment':destination/'geometry/A_poseguide_alignment.png',
            'operation':destination/'geometry/operation.json',
            'generation_prompt':destination/'prompt/generation_prompt.txt',
            'generation_prompt_provenance':destination/'prompt/generation_prompt.json'}
        finish_stage(prep_record,prep_request,outputs,operation,destination)
    paths=stage_outputs(prep_record,destination)
    paths['run_dir']=str(destination)
    request=workflow_cli.stage_request('generate',settings,paths,destination)
    before=read_json(baseline/'generation/stage.json')['identity']
    if request['identity']['model_identity']!=before['model_identity']:
        raise RuntimeError(f'{name}: Qwen model artifacts differ from baseline')
    if request['identity']['settings']!=before['settings']:
        raise RuntimeError(f'{name}: generation settings differ from baseline')
    request['identity']['experiment_geometry']=prep_identity
    if sha256_file(paths['generation_prompt'])!=old['prompt_sha256']:
        raise RuntimeError('Frozen prompt is not byte-identical to baseline')
    return destination,settings,request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--verify',action='store_true',help='Verify existing prep/generation hashes without inference')
    args=parser.parse_args()
    with file_lock(WORKFLOW/'.workflow.lock'):
        for name in args.cases:
            target=ROOT/name
            manifest=target/'experiment.json'
            state={'schema':SCHEMA,'case':name,'baseline_run':str(WORKFLOW/'runs'/name),
                'status':'preparing','started_at_unix':time.time(),
                'variables_changed':['guide scale','guide pelvis anchor','off-canvas segment raster clipping'],
                'fixed':['A image','source canvas/preprocess','exact baseline prompt','model','seed','steps','CFG','KV cache'],
                'limits':'Uniform 2D source-scale/anchor ablation; B bone proportions retained; PE not rerun'}
            if manifest.exists() and read_json(manifest).get('schema')!=SCHEMA:
                raise RuntimeError(f'Refusing to overwrite a different experiment at {target}')
            try:
                if not args.verify:atomic_json(manifest,state)
                target,settings,request=prepare(name,args.verify)
                record=target/'generation/stage.json'
                cached=stage_valid(record,request,target)
                if not cached:
                    if args.verify:raise RuntimeError(f'{name}: generation absent or stale')
                    state.update(status='generating',generation_request=str(target/'generation/request.json'))
                    atomic_json(manifest,state)
                    atomic_json(target/'generation/request.json',request)
                    print(f'GENERATE {name} -> {target}',flush=True)
                    with (target/'generation/run.log').open('w') as log:
                        done=subprocess.run([settings['qwen_python'],'-u',str(WORKFLOW/'generate_worker.py'),
                            '--request',str(target/'generation/request.json')],cwd=WORKFLOW,
                            stdout=log,stderr=subprocess.STDOUT)
                    if done.returncode:
                        tail='\n'.join((target/'generation/run.log').read_text(errors='replace').splitlines()[-25:])
                        raise RuntimeError(tail)
                    if not stage_valid(record,request,target):raise RuntimeError('Generated artifact evidence failed validation')
                outputs=stage_outputs(record,target)
                print(f'{"VERIFIED" if args.verify else "CACHE_HIT" if cached else "COMPLETE"} {name} {outputs["final_image"]}',flush=True)
                if not args.verify:
                    state.update(status='complete',cache_hit=cached,outputs=outputs,finished_at_unix=time.time())
                    atomic_json(manifest,state)
            except Exception as error:
                if not args.verify:
                    state.update(status='failed',error=str(error),finished_at_unix=time.time())
                    atomic_json(manifest,state)
                raise
        if not args.verify:
            completed={p.parent.name:read_json(p) for p in ROOT.glob('*/experiment.json')}
            atomic_json(ROOT/'summary.json',{'schema':SCHEMA,'cases':completed})


if __name__=='__main__':main()
