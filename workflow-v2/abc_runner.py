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
from abc_variants import VARIANTS, DIAGNOSTICS, control_records, generation_key
from src.pose_edit.common import atomic_json,file_lock

MODES=('geometry_only','qwen_baseline','native_v5')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def asset_identity(path,*,snapshot=None):
    path=Path(path).resolve()
    def content(file):
        if snapshot is None:return sha256(file)
        identity=snapshot.get(str(file.resolve()));stat=file.stat()
        if identity is None or stat.st_size!=identity['bytes'] or stat.st_mtime_ns!=identity['mtime_ns']:
            raise ValueError('Frozen asset changed or is absent from snapshot: '+str(file))
        return identity['sha256']
    if path.is_file():return {'sha256':content(path)}
    if not path.is_dir():raise FileNotFoundError(path)
    return {str(p.relative_to(path)):content(p) for p in sorted(path.rglob('*'))
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
    parser.add_argument('--variant',choices=VARIANTS,default='full')
    parser.add_argument('--generation-tag',help='Independent sibling generation/evaluation stage')
    parser.add_argument('--diagnostic',choices=DIAGNOSTICS,default='none')
    parser.add_argument('--observe-native',action='store_true',help='Record scalar residual contributions without changing sampling')
    parser.add_argument('--asset-snapshot',type=Path,help='Campaign-owned, fully hashed frozen model manifest')
    parser.add_argument('--asset-snapshot-sha256',help='Exact SHA of the immutable asset snapshot')
    for name in ('check','verify','force'):parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args(argv)
    if args.pair:parser.error('workflow-v2 uses A+B+C; legacy --pair remains in workflow-multiple')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',args.name):parser.error('Use a simple --name without path separators')
    try:parse_role_map(args.role_map)
    except ValueError as error:parser.error(str(error))
    if args.force and args.verify:parser.error('--force and --verify are mutually exclusive')
    try:generation_key(args.mode,args.variant,args.generation_tag)
    except ValueError as error:parser.error(str(error))
    if args.diagnostic!='none' and not args.generation_tag:parser.error('--diagnostic requires an explicit --generation-tag')
    if args.mode!='native_v5' and args.observe_native:parser.error('--observe-native requires --mode native_v5')
    if bool(args.asset_snapshot)!=bool(args.asset_snapshot_sha256):parser.error('Asset snapshot needs both path and pinned SHA256')
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
    snapshot=None
    if args.asset_snapshot:
        if sha256(args.asset_snapshot)!=args.asset_snapshot_sha256:raise ValueError('Frozen asset snapshot SHA changed')
        manifest=json.loads(args.asset_snapshot.read_text(encoding='utf8'))
        if manifest.get('schema')!='abc_quality_freeze_v1':raise ValueError('Unknown asset snapshot schema')
        snapshot=manifest['assets']
    asset=lambda path:asset_identity(path,snapshot=snapshot)
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
        variant=args.variant,generation_tag=args.generation_tag,diagnostic=args.diagnostic,observe_native=args.observe_native,
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
        models={k:asset(settings[k]) for k in ('sam2_checkpoint','dwpose_det_model','dwpose_pose_model','fcl_root')},
        estimator_assets={k:asset(settings[k]) for k in (('smplest_weights','smplx_model') if official else ('smplest_weights','multihmr_weights','smplx_model'))})
    if source_binding is not None:geometry_identity['source_cache']=source_binding
    if official is not None:
        geometry_identity['official_hi4d']=dict(frame={k:official[k] for k in ('pair','action','frame_id','camera_id')},
            assets={k:asset(official[k]) for k in ('official_smpl','meta','camera_file','rgb')},
            models=asset(settings['hi4d_model_root']))
    def execute():
        geom=execute_stage(run_dir,'geometry',geometry_identity,request,
            subprocess_worker('abc_geometry_worker.py',settings),verify=args.verify)
        request['scene']=str(run_dir/'geometry/scene.json')
        if args.mode=='geometry_only':return {'geometry':geom['fingerprint']}
        prompt_identity=dict(geometry=geom['fingerprint'],user_prompt=args.user_prompt,
            pe=asset(settings['pe_model']),seed=settings['pe_seed'],max_new_tokens=settings['pe_max_new_tokens'],
            code=code_identity(['workflow-v2/abc_prompt_worker.py','workflow-v2/abc_contracts.py','src/pose_edit/prompt_contract.py']))
        prompt=execute_stage(run_dir,'prompt',prompt_identity,request,
            subprocess_worker('abc_prompt_worker.py',settings),verify=args.verify)
        request['prompt']=str(run_dir/'prompt/generation_prompt.txt')
        output_key=generation_key(args.mode,args.variant,args.generation_tag)
        native_controls,base_controls=control_records(args.variant)
        generation_identity=dict(geometry=geom['fingerprint'],prompt=prompt['fingerprint'],mode=args.mode,
            variant=args.variant,diagnostic=args.diagnostic,observe_native=args.observe_native,
            controls=native_controls if args.mode=='native_v5' else None,
            base_controls=base_controls if args.mode=='native_v5' else None,
            settings={k:settings[k] for k in ('resolution','steps','seed','true_cfg_scale','memory_mode')},
            qwen=asset(settings['qwen_model']),
            code=code_identity(['workflow-v2/abc_generate_worker.py','workflow-v2/abc_variants.py','workflow-v2/abc_feature_cache.py','workflow-v2/abc_native*.py',
                'src/pose_control/qwen21_full/native_v5/*.py','src/pose_control/qwen21_full/*.py']))
        if args.mode=='native_v5':
            generation_identity['checkpoint']=asset(args.checkpoint)
            generation_identity['appearance_models']={key:asset(settings[key]) for key in ('arcface_model','native_dino_model')}
        generation=execute_stage(run_dir,'generation/'+output_key,generation_identity,request,
            subprocess_worker('abc_generate_worker.py',settings),verify=args.verify)
        request['image']=str(run_dir/'generation'/output_key/'final_image.png')
        evaluation=execute_stage(run_dir,'evaluation/'+output_key,
            {'generation':generation['fingerprint'],'code':code_identity(['workflow-v2/abc_evaluate.py','workflow-v2/abc_quality.py']),
             'detector':geometry_identity['models'],'appearance':asset(settings['appearance_dino_model']),
             'identity_model':asset(settings['arcface_model'])},
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
                atomic_json(run_dir/('run_'+generation_key(args.mode,args.variant,args.generation_tag)+'.json'),dict(schema='abc_workflow_v1',mode=args.mode,
                    status='complete',stages=result,request=request,verification='validated',quality='see_evaluation',finished_at_unix=time.time()))
            except Exception as error:
                atomic_json(run_dir/('run_'+generation_key(args.mode,args.variant,args.generation_tag)+'.json'),dict(schema='abc_workflow_v1',mode=args.mode,
                    status='failed',error=str(error),request=request,finished_at_unix=time.time()))
                raise
    print(('VERIFIED ' if args.verify else 'WORKFLOW_FINISHED ')+str(run_dir),flush=True)
    return result

def main():
    try:run(parse_args())
    except (RuntimeError,ValueError,OSError) as error:
        print('WORKFLOW_FAILED: '+str(error),file=sys.stderr,flush=True);raise SystemExit(1)
