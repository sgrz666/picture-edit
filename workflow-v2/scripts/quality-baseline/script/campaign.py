"""Frozen, resumable 68-image quality campaign using the authoritative ABC runner."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
WORKFLOW=HERE.parents[2]
ROOT=WORKFLOW.parent
sys.path.insert(0,str(WORKFLOW));sys.path.insert(0,str(ROOT))
from abc_runner import sha256,asset_identity,code_identity
from abc_variants import generation_key
from src.pose_edit.common import atomic_json,file_lock

BASE=Path('/data/shangguanrz/pic-edit-workflow-v2/validation')
CHECKPOINT=Path('/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-fixedpair-v5_20261025_v1/training/checkpoints/update_00000300_5821758668f84b5e8b4d99dc5499c83e.pt')
CHECKPOINT_SHA='6a340c0a9bde9fbea943bfb26cbfce425a893ac2201d1b23651e0508e4bd20e2'
CASES={
 'standing':dict(c='standing_C.jpg',official=str(BASE/'standing_hi4d.json'),prompt='Follow the non-contact standing arrangement in the retargeted skeletons.'),
 'handshake':dict(c='handshake_C.jpg',official=None,prompt='Make A and B shake hands exactly as the retargeted skeletons specify.'),
 'hug':dict(c='hug_C.png',official=str(ROOT/'workflow-multiple/pairs/hug_pair00_084_c4_r000'),prompt='Make A and B embrace exactly as the retargeted hugging skeletons specify.'),
}
CODE_PATTERNS=['workflow-v2/**/*.py','workflow-v2/**/*.sh','workflow-v2/config.json',
 'src/pose_edit/*.py','src/data/canvas_contract.py','src/pose_control/qwen21_full/**/*.py',
 'src/pose_control/qwen21_full/native_v5/**/*.py','src/pose_control/qwen21_pro/geometry.py']


def matrix():
    jobs=[]
    def add(phase,case,resolution,steps,seed,variant='full',baseline=False,diagnostic='none',swap=None):
        label='qwen_only' if baseline else variant
        if diagnostic!='none':label=diagnostic
        if swap:label=swap+'_'+label
        jobs.append(dict(phase=phase,case=case,resolution=resolution,steps=steps,seed=seed,variant=variant,
            mode='qwen_baseline' if baseline else 'native_v5',diagnostic=diagnostic,swap=swap,
            tag=f's{seed}_{label}',label=label))
    for phase,res,steps,seeds in (('smoke',512,12,(42,)),('formal',768,30,(42,43,44))):
        for case in CASES:
            for seed in seeds:
                for variant,baseline in (('full',True),('full',False),('body_interaction',False),('no_source_detail',False)):
                    add(phase,case,res,steps,seed,variant,baseline)
    for swap in ('source_swap','role_swap'):
        for baseline in (True,False):add('swap','standing',768,30,42,baseline=baseline,swap=swap)
    for case in ('handshake','hug'):
        for seed in (42,43):
            for variant in ('body_only','no_hand','no_face'):add('diagnose',case,768,30,seed,variant)
    for seed in (42,43):
        for diagnostic in ('photo_prompt','monochrome_guides'):add('diagnose','hug',768,30,seed,baseline=True,diagnostic=diagnostic)
    return jobs


def source_files(settings):
    files=set()
    def include(path):
        path=Path(path).resolve()
        if path.is_file():files.add(path)
        else:files.update(p.resolve() for p in path.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts)
    include(WORKFLOW/'config.json');include(CHECKPOINT)
    for p in (BASE/'inputs').iterdir():
        if p.is_file():include(p)
    for key in ('qwen_model','pe_model','smplest_weights','multihmr_weights','smplx_model','sam2_checkpoint',
                'dwpose_det_model','dwpose_pose_model','arcface_model','appearance_dino_model','native_dino_model','hi4d_model_root','fcl_root'):
        include(settings[key])
    from hi4d import resolve_record
    for case in CASES.values():
        if case['official']:
            include(Path(case['official'])/'pair.json' if Path(case['official']).is_dir() else case['official'])
            record=resolve_record(Path(case['official']))
            for key in ('official_smpl','meta','camera_file','rgb'):include(record[key])
    return sorted(files)


def freeze(out):
    record=out/'freeze.json'
    if record.exists():
        frozen=json.loads(record.read_text());assert_frozen(frozen,full=True);return frozen
    settings=json.loads((WORKFLOW/'config.json').read_text())
    if sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('Pinned checkpoint SHA mismatch')
    files={str(p):dict(sha256=sha256(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in source_files(settings)}
    frozen=dict(schema='abc_quality_freeze_v1',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True),code=code_identity(CODE_PATTERNS),
        assets=files,checkpoint=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,settings=settings,matrix=matrix(),
        source_count_training=1,source_count_inference=2,created_unix=time.time(),training_executed=False)
    frozen['environments']=environment_identity(settings)
    atomic_json(record,frozen)
    atomic_json(out/'input_index.json',dict(cases=CASES,source_a=str(BASE/'inputs/A.png'),source_b=str(BASE/'inputs/B.png'),
        handshake_scope='Historical upper-body photograph; inferred legs are not annotated ground truth',
        official_finger_scope='Hi4D SMPL has no fine finger pose; corresponding conditions remain unavailable.'))
    return frozen


def environment_identity(settings):
    code=('import sys,json,importlib.metadata as m; '
          'print(json.dumps(dict(python=sys.version,packages={d.metadata["Name"]:d.version for d in m.distributions()}),sort_keys=True))')
    return {key:json.loads(subprocess.check_output([settings[key],'-B','-c',code],text=True))
            for key in ('qwen_python','geometry_python','multihmr_python','sam_python')}


def assert_frozen(frozen,*,full=False):
    if code_identity(CODE_PATTERNS)!=frozen['code']:raise RuntimeError('Frozen runtime code changed; start a new campaign')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=frozen['git_head']:
        raise RuntimeError('Git HEAD changed during campaign')
    if full and environment_identity(frozen['settings'])!=frozen['environments']:
        raise RuntimeError('Frozen Python/package versions changed')
    for path,identity in frozen['assets'].items():
        p=Path(path);s=p.stat()
        if s.st_size!=identity['bytes'] or s.st_mtime_ns!=identity['mtime_ns']:
            raise RuntimeError('Frozen asset changed: '+path)
        if full and sha256(p)!=identity['sha256']:raise RuntimeError('Frozen asset SHA changed: '+path)


def arguments(out,job):
    case=CASES[job['case']];a,b=('B.png','A.png') if job['swap']=='source_swap' else ('A.png','B.png')
    group=f"{job['case']}_{job['resolution']}"+(('_'+job['swap']) if job['swap'] else '')
    args=['--source-a',str(BASE/'inputs'/a),'--source-b',str(BASE/'inputs'/b),'--interaction',str(BASE/'inputs'/case['c']),
          '--user-prompt',case['prompt'],'--role-map','A=right,B=left' if job['swap']=='role_swap' else 'A=left,B=right',
          '--name',group,'--run-root',str(out/'runs'),'--resolution',str(job['resolution']),'--steps',str(job['steps']),
          '--seed',str(job['seed']),'--mode',job['mode'],'--variant',job['variant'],'--generation-tag',job['tag'],
          '--diagnostic',job['diagnostic'],'--asset-snapshot',str(out/'freeze.json'),
          '--asset-snapshot-sha256',sha256(out/'freeze.json')]
    if case['official']:args+=['--hi4d-record',case['official']]
    # Only same-resolution A/B observations are reused, in a new geometry stage.
    cached=out/'runs'/f"standing_{job['resolution']}"
    if (cached/'geometry/stage.json').is_file() and group!=cached.name:args+=['--sources-from',str(cached)]
    if job['mode']=='native_v5':args+=['--checkpoint',str(CHECKPOINT),'--observe-native']
    return args,group


def paths(out,job):
    _,group=arguments(out,job);key=generation_key(job['mode'],job['variant'],job['tag'])
    run=out/'runs'/group
    return run,run/'generation'/key,run/'evaluation'/key


def run_job(out,job,frozen,*,verify=False):
    args,group=arguments(out,job);identity=f"{group}__{job['mode']}__{job['tag']}"
    record=out/'jobs'/(identity+'.json')
    prior=json.loads(record.read_text()) if record.exists() else None
    command=[frozen['settings']['qwen_python'],'-B','-u',str(WORKFLOW/'run.py'),*args]
    if verify or (prior is not None and prior.get('status')=='complete'):
        # Use the original complete request, including its original source cache binding.
        original=prior['command'] if prior else command
        result=subprocess.run([*original,'--verify'],cwd=ROOT,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Read-only job verification failed: '+identity+'\n'+result.stdout+result.stderr)
        return prior or dict(job=job,status='verified')
    assert_frozen(frozen)
    attempt=time.strftime('%Y%m%dT%H%M%S')+'_'+str(time.time_ns()%1000000)
    log=out/'logs'/(identity+'__'+attempt+'.log');started=time.time()
    running=dict(job=job,command=command,status='running',started_unix=started,log=str(log))
    atomic_json(record,running);atomic_json(out/'current.json',running)
    print('START '+identity,flush=True)
    with log.open('x') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'})
    running.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,seconds=time.time()-started)
    if result.returncode==0:
        assert_frozen(frozen)
        check=subprocess.run([*command,'--verify'],cwd=ROOT,capture_output=True,text=True)
        (out/'logs'/(identity+'__'+attempt+'_verify.log')).write_text(check.stdout+check.stderr)
        running.update(verified=check.returncode==0)
        if check.returncode:running['status']='failed_verification'
    atomic_json(record,running);atomic_json(out/'current.json',running)
    print('END '+identity+' '+running['status'],flush=True)
    if running['status']!='complete':raise RuntimeError('Campaign stopped; inspect '+str(log))
    return running


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',default='quality_baseline_20261028_v1')
    parser.add_argument('--phase',choices=('all','smoke','formal','swap','diagnose','report'),default='all')
    parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',args.campaign):parser.error('Unsafe campaign name')
    out=BASE/args.campaign
    if args.verify:
        frozen=json.loads((out/'freeze.json').read_text());assert_frozen(frozen,full=True)
        jobs=[json.loads(p.read_text()) for p in sorted((out/'jobs').glob('*.json'))]
        if len(jobs)!=68 or any(j['status']!='complete' for j in jobs):raise RuntimeError('All 68 jobs must complete before campaign verification')
        for item in jobs:run_job(out,item['job'],frozen,verify=True)
        assert_frozen(frozen,full=True)
        from quality_report import audit
        print(json.dumps(audit(out,write=False),indent=2));return
    out.mkdir(parents=True,exist_ok=True)
    with file_lock(out/'.campaign.lock'):
        for name in ('jobs','logs','report'): (out/name).mkdir(exist_ok=True)
        frozen=freeze(out)
        selected=[j for j in matrix() if args.phase=='all' or j['phase']==args.phase]
        for number,job in enumerate(selected):
            run_job(out,job,frozen)
            completed=[json.loads(p.read_text()) for p in (out/'jobs').glob('*.json')]
            atomic_json(out/'progress.json',dict(expected=68,complete=sum(x['status']=='complete' for x in completed),
                phases=dict(Counter(x['job']['phase'] for x in completed if x['status']=='complete'))))
            if number+1==len(selected) or selected[number+1]['phase']!=job['phase']:
                from quality_report import build_report
                build_report(out)
        from quality_report import build_report
        build_report(out)


if __name__=='__main__':main()
