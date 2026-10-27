"""One resumable A+B+C entry point with content-addressed stage evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from abc_contracts import parse_role_map
from src.pose_edit.common import atomic_json,file_lock

MODES=('geometry_only','qwen_baseline','native_v5')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def asset_identity(path):
    path=Path(path).resolve()
    if path.is_file():return {'sha256':sha256(path)}
    if not path.is_dir():raise FileNotFoundError(path)
    return {str(p.relative_to(path)):sha256(p) for p in sorted(path.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-a','source-b','interaction','checkpoint'):
        parser.add_argument('--'+key,type=Path)
    parser.add_argument('--pair',help=argparse.SUPPRESS)
    parser.add_argument('--hi4d-record',type=Path,help='Official Hi4D frame JSON or existing dual pair directory; prefer official body/contact geometry')
    parser.add_argument('--sources-from',type=Path,help='Reuse verified source-only A/B estimates from an existing completed geometry run')
    parser.add_argument('--user-prompt',default='')
    parser.add_argument('--mode',choices=MODES,default='native_v5')
    parser.add_argument('--role-map',default='A=left,B=right')
    parser.add_argument('--name',default='abc_20261026')
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--run-root',type=Path,default=Path('/data/shangguanrz/pic-edit-workflow-v2/runs'))
    parser.add_argument('--resolution',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--seed',type=int)
    for name in ('check','verify','force'):parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args(argv)
    if args.pair:parser.error('workflow-v2 uses A+B+C; legacy --pair remains in workflow-multiple')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',args.name):parser.error('Use a simple --name without path separators')
    try:parse_role_map(args.role_map)
    except ValueError as error:parser.error(str(error))
    if args.force and args.verify:parser.error('--force and --verify are mutually exclusive')
    if not args.check:
        if not all((args.source_a,args.source_b,args.interaction)):parser.error('--source-a, --source-b and --interaction are required')
        if args.mode=='native_v5' and not args.checkpoint:parser.error('--checkpoint is required for native_v5')
    return args

def stage_outputs(directory):
    return {str(p.relative_to(directory)):{'sha256':sha256(p),'bytes':p.stat().st_size}
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name not in ('stage.json','request.json')
        and '__pycache__' not in p.parts}

def execute_stage(run_dir,name,identity,request,worker,*,verify=False):
    directory=Path(run_dir)/name;record=directory/'stage.json';fingerprint=digest(identity)
    if record.is_file():
        old=json.loads(record.read_text(encoding='utf8'))
        if old['fingerprint']!=fingerprint or old['outputs']!=stage_outputs(directory):
            raise RuntimeError(f'{name}: evidence changed; use a new --name')
        print(f'{name} CACHE_HIT',flush=True)
        return old
    if verify:raise RuntimeError(f'{name}: missing evidence; use a new --name or resume without --verify')
    work_directory=directory
    prior=directory/'request.json'
    if directory.exists() and any(directory.iterdir()):
        if name.startswith('generation/'):
            work_directory=directory/'attempts'/('attempt_'+time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
        elif not prior.is_file() or json.loads(prior.read_text(encoding='utf8')).get('fingerprint')!=fingerprint:
            raise RuntimeError(f'{name}: incomplete stage has different inputs; use a new --name')
    directory.mkdir(parents=True,exist_ok=True)
    work_directory.mkdir(parents=True,exist_ok=True)
    request={**request,'output_dir':str(work_directory.resolve()),'identity':identity,'fingerprint':fingerprint}
    atomic_json(work_directory/'request.json',request)
    worker(request)
    if work_directory!=directory:
        for filename in ('final_image.png','final_canvas.png','generation.json'):
            artifact=work_directory/filename
            if artifact.is_file():
                alias=directory/filename
                if alias.exists():raise RuntimeError(f'{alias} already exists; use a new --name')
                alias.symlink_to(artifact.relative_to(directory))
    outputs=stage_outputs(directory)
    if not outputs:raise RuntimeError(f'{name}: worker produced no artifacts')
    result=dict(schema='abc_stage_v1',name=name,fingerprint=fingerprint,identity=identity,
        outputs=outputs,finished_at_unix=time.time())
    atomic_json(record,result)
    print(f'{name} COMPLETE',flush=True)
    return result

def subprocess_worker(script,settings):
    def run(request):
        directory=Path(request['output_dir']);path=directory/'request.json'
        log=directory/'worker.log'
        with log.open('a',encoding='utf8') as stream:
            process=subprocess.run([settings['qwen_python'],'-u',str(HERE/script),'--request',str(path)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode:
            tail='\n'.join(log.read_text(encoding='utf8',errors='replace').splitlines()[-35:])
            raise RuntimeError(f'{script} failed ({process.returncode}):\n{tail}')
    return run

def code_identity(patterns):
    files=set()
    for pattern in patterns:files.update(p for p in ROOT.glob(pattern) if p.is_file())
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)}

def settings_for(args):
    settings=json.loads(args.config.read_text(encoding='utf8'))
    for key in ('resolution','steps','seed'):
        if getattr(args,key) is not None:settings[key]=getattr(args,key)
    if settings['resolution']<256 or settings['steps']<1:raise ValueError('resolution >=256 and positive steps required')
    if settings.get('true_cfg_scale')!=1.0:raise ValueError('This native-canvas sampler requires true_cfg_scale=1.0')
    if settings.get('memory_mode')!='sequential':raise ValueError('memory_mode must be sequential')
    return settings

def dependency_check(settings,mode,checkpoint=None,official=False):
    names=['qwen_python','geometry_python','multihmr_python','sam_python','smplest_root',
        'sam2_checkpoint','dwpose_det_model','dwpose_pose_model','fcl_root']
    if official:names=[n for n in names if n!='multihmr_python']+['hi4d_model_root']
    if mode!='geometry_only':names+=['qwen_model','pe_model']
    for name in names:
        if not Path(settings[name]).exists():raise FileNotFoundError(f'{name}: {settings[name]}')
    if checkpoint and not Path(checkpoint).is_file():raise FileNotFoundError(checkpoint)
    subprocess.run([settings['qwen_python'],'-B','-c',
        'import torch; x=torch.ones(1,device="cuda"); print("CUDA_ALLOCATION_OK",x.item(),torch.cuda.mem_get_info())'],check=True)
    print('WORKFLOW_V2_ASSETS_OK',flush=True)

def run(args):
    settings=settings_for(args)
    from hi4d import resolve_record
    official=resolve_record(args.hi4d_record) if args.hi4d_record else None
    if args.check:
        dependency_check(settings,args.mode,args.checkpoint,official=official is not None);return
    inputs={key:str(getattr(args,key).expanduser().resolve()) for key in ('source_a','source_b','interaction')}
    for path in inputs.values():
        if not Path(path).is_file():raise FileNotFoundError(path)
    if args.checkpoint and not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    name=args.name+(time.strftime('_rerun_%Y%m%dT%H%M%S') if args.force else '')
    run_dir=args.run_root.expanduser().resolve()/name
    role_mapping=parse_role_map(args.role_map)
    request=dict(schema='abc_workflow_request_v1',project_root=str(ROOT),run_dir=str(run_dir),
        settings=settings,inputs=inputs,role_mapping=role_mapping,mode=args.mode,
        official_record=official,
        checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,user_prompt=args.user_prompt)
    source_binding=None
    if args.sources_from:
        from abc_contracts import resolve_source_cache
        request['source_cache'],source_binding=resolve_source_cache(args.sources_from,inputs,settings)
    geometry_keys=('resolution','canvas_policy','person_detection_confidence','smplest_checkpoint',
        'interaction_fov','interaction_K','contact_candidate_threshold_m','contact_max_shift_m','target_margin')
    geometry_identity=dict(schema='abc_geometry_v1',inputs={k:sha256(v) for k,v in inputs.items()},
        roles=role_mapping,settings={k:settings.get(k) for k in geometry_keys},
        code=code_identity(['workflow-v2/abc_contracts.py','workflow-v2/abc_geometry_worker.py','workflow-v2/abc_source_worker.py',
            'workflow-v2/abc_estimate_worker.py','workflow-v2/abc_hi4d_worker.py','workflow-v2/hi4d.py','workflow-v2/abc_native_assets.py','workflow-v2/retarget/*.py','workflow-v2/render/*.py',
            'src/pose_edit/*.py','src/data/canvas_contract.py','src/pose_control/qwen21_pro/geometry.py',
            'src/pose_control/qwen21_full/native_full_assets.py','src/pose_control/qwen21_full/*condition*.py',
            'src/pose_control/qwen21_full/*geometry*.py','src/pose_control/qwen21_full/body_pose.py','src/pose_control/qwen21_full/conditions.py']),
        models={k:asset_identity(settings[k]) for k in ('sam2_checkpoint','dwpose_det_model','dwpose_pose_model','fcl_root')},
        estimator_assets={k:asset_identity(settings[k]) for k in (('smplest_weights','smplx_model') if official else ('smplest_weights','multihmr_weights','smplx_model'))})
    if source_binding is not None:geometry_identity['source_cache']=source_binding
    if official is not None:
        geometry_identity['official_hi4d']=dict(frame={k:official[k] for k in ('pair','action','frame_id','camera_id')},
            assets={k:asset_identity(official[k]) for k in ('official_smpl','meta','camera_file','rgb')},
            models=asset_identity(settings['hi4d_model_root']))
    def execute():
        geom=execute_stage(run_dir,'geometry',geometry_identity,request,
            subprocess_worker('abc_geometry_worker.py',settings),verify=args.verify)
        request['scene']=str(run_dir/'geometry/scene.json')
        if args.mode=='geometry_only':return {'geometry':geom['fingerprint']}
        prompt_identity=dict(geometry=geom['fingerprint'],user_prompt=args.user_prompt,
            pe=asset_identity(settings['pe_model']),seed=settings['pe_seed'],max_new_tokens=settings['pe_max_new_tokens'],
            code=code_identity(['workflow-v2/abc_prompt_worker.py','workflow-v2/abc_contracts.py','src/pose_edit/prompt_contract.py']))
        prompt=execute_stage(run_dir,'prompt',prompt_identity,request,
            subprocess_worker('abc_prompt_worker.py',settings),verify=args.verify)
        request['prompt']=str(run_dir/'prompt/generation_prompt.txt')
        generation_identity=dict(geometry=geom['fingerprint'],prompt=prompt['fingerprint'],mode=args.mode,
            settings={k:settings[k] for k in ('resolution','steps','seed','true_cfg_scale','memory_mode')},
            qwen=asset_identity(settings['qwen_model']),
            code=code_identity(['workflow-v2/abc_generate_worker.py','workflow-v2/abc_native*.py',
                'src/pose_control/qwen21_full/native_v5/*.py','src/pose_control/qwen21_full/*.py']))
        if args.mode=='native_v5':
            generation_identity['checkpoint']=asset_identity(args.checkpoint)
            generation_identity['appearance_models']={key:asset_identity(settings[key]) for key in ('arcface_model','native_dino_model')}
        generation=execute_stage(run_dir,'generation/'+args.mode,generation_identity,request,
            subprocess_worker('abc_generate_worker.py',settings),verify=args.verify)
        request['image']=str(run_dir/'generation'/args.mode/'final_image.png')
        evaluation=execute_stage(run_dir,'evaluation/'+args.mode,
            {'generation':generation['fingerprint'],'code':code_identity(['workflow-v2/abc_evaluate.py']),
             'detector':geometry_identity['models'],'appearance':asset_identity(settings['appearance_dino_model']),
             'identity_model':asset_identity(settings['arcface_model'])},
            request,subprocess_worker('abc_evaluate.py',settings),verify=args.verify)
        return dict(geometry=geom['fingerprint'],prompt=prompt['fingerprint'],
            generation=generation['fingerprint'],evaluation=evaluation['fingerprint'])
    if args.verify:
        result=execute()
    else:
        run_dir.mkdir(parents=True,exist_ok=True)
        with file_lock(run_dir/'.workflow.lock'):
            try:
                result=execute()
                atomic_json(run_dir/('run_'+args.mode+'.json'),dict(schema='abc_workflow_v1',mode=args.mode,
                    status='complete',stages=result,request=request,verification='validated',quality='see_evaluation',finished_at_unix=time.time()))
            except Exception as error:
                atomic_json(run_dir/('run_'+args.mode+'.json'),dict(schema='abc_workflow_v1',mode=args.mode,
                    status='failed',error=str(error),request=request,finished_at_unix=time.time()))
                raise
    print(('VERIFIED ' if args.verify else 'WORKFLOW_FINISHED ')+str(run_dir),flush=True)
    return result

def main():
    try:run(parse_args())
    except (RuntimeError,ValueError,OSError) as error:
        print('WORKFLOW_FAILED: '+str(error),file=sys.stderr,flush=True);raise SystemExit(1)
