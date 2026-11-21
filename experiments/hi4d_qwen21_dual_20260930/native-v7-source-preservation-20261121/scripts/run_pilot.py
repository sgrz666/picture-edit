"""Run two bounded S3 continuations with identical data/noise/optimizer policy."""
from pathlib import Path
from dataclasses import replace
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback

PROJECT=Path('/home/shangguanrz/project/pic-edit')
sys.path.insert(0,str(PROJECT if PROJECT.exists() else Path(__file__).resolve().parents[4]))


def validate_pair(control,modified):
    excluded={'preservation','preservation_index','gradient_report'}
    comparable=lambda value:{k:v for k,v in value.items() if k not in excluded}
    budget=control['stage']['budget_updates']
    if (control.get('preservation') is not None or not modified.get('preservation') or not modified.get('preservation_index')
            or control['stage']['phase']!='S3' or type(budget)!=int or not 1<=budget<=40
            or comparable(control)!=comparable(modified)):
        raise ValueError('Paired pilot must differ only in its bound preservation objective')
    return budget


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--source-gradient',type=Path,required=True)
    parser.add_argument('--memory-fraction',type=float,default=.8)
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch('[a-zA-Z0-9-]+',args.name):parser.error('Plain new run name required')
    if not 0<args.memory_fraction<=.9:parser.error('Pilot cap must be positive and at most .9')
    root=args.root.resolve()
    if args.detach:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:]]
        command.remove('--detach')
        with (root/'logs'/(args.name+'-driver.log')).open('xb') as stream:
            process=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        with (root/'logs'/(args.name+'-pid.json')).open('x') as stream:
            json.dump({'pid':process.pid,'argv':command,'maximum_additional_optimizer_updates':80},stream)
        print(json.dumps({'pid':process.pid}));return
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    from src.pose_control.qwen21_full.native_v62.training.io import verify,write_json
    from src.pose_control.qwen21_full.native_v62.training.checkpointing import load_committed
    report_root=root/'report'/args.name
    report_root.mkdir(exist_ok=False)
    helper_path=args.baseline_root/'scripts/run_training.py'
    spec=importlib.util.spec_from_file_location('prior_pilot_boundaries',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    read=lambda path:json.loads(path.read_text(encoding='utf8'))
    raw={name:read(root/'configs/pilot-v1'/(filename+'.json')) for name,filename in (('control','control'),('source-background','source-background'))}
    budget=validate_pair(raw['control'],raw['source-background'])
    configs={name:TrainingConfigV7.from_dict(value) for name,value in raw.items()}
    if any(Path(c.output_root)!=root/'output/training' for c in configs.values()):
        raise ValueError('Pilot output must belong to this experiment')
    completed={}
    def run(command,label,resource=None):
        command=helper.child_command(root,command,resource=resource,memory_fraction=args.memory_fraction)
        logfile=root/'logs'/(args.name+'-'+label+'.log')
        print(json.dumps({'event':label,'argv':command}),flush=True)
        with logfile.open('x',encoding='utf8') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        parent=configs['control'].initialization
        saved,ref,_=load_committed(parent,schema=configs['control'].schema)
        expected=saved['training_state']['global_update']+budget
        del saved
        for name,cfg in configs.items():
            gradient=args.source_gradient if name=='source-background' else report_root/'control-gradient.json'
            if name=='control':
                run(['preflight','--level','grad','--config',root/'configs/pilot-v1/control.json','--report',gradient],
                    'control-grad',root/'report'/(args.name+'-control-grad-resources.json'))
            grad_ref=verify({'path':str(gradient.resolve())})
            grad=read(gradient)
            if not (grad.get('passed') is True and grad.get('production_passed') is True
                    and grad.get('schema')=='native_v7_gradient_preflight_v1'
                    and grad.get('config_sha256')==cfg.signature()
                    and grad.get('initialization_sha256')==ref['sha256']):
                raise ValueError('Gradient report does not accept this exact objective/parent')
            cfg=replace(cfg,gradient_report=grad_ref)
            ready=report_root/(name+'-ready.json');write_json(ready,cfg.record())
            before=set((root/'output/training/segments').glob('*'))
            run(['advance-stage','--checkpoint',ref['path'],'--config',ready],name+'-train',
                root/'report'/(args.name+'-'+name+'-train-resources.json'))
            segment,summary=helper.new_stage_summary(root/'output/training/segments',before,cfg.record())
            checkpoint=summary['checkpoint']
            saved,actual,_=load_committed(checkpoint['checkpoint'],schema=cfg.schema)
            helper.check_completed_stage(summary,saved['training_state'],cfg.record(),cfg.signature(),expected,ref)
            if checkpoint['checkpoint']!=actual:raise ValueError('Completed checkpoint reference differs')
            del saved
            generation=replace(cfg,initialization=actual)
            generation_config=report_root/(name+'-generation.json');write_json(generation_config,generation.record())
            image_report=report_root/(name+'-images.json')
            run([str(helper_path),'generate','--config',generation_config,'--checkpoint',actual['path'],
                '--output',root/'output'/args.name/name,'--report',image_report],name+'-generation')
            completed[name]={'checkpoint':checkpoint,'generation_config':verify({'path':str(generation_config)}),
                'generation_report':verify({'path':str(image_report)}),'segment':str(segment),'global_update':expected}
            write_json(report_root/(name+'-completed.json'),completed[name])
        write_json(report_root/'completed.json',{'arms':completed,'updates_per_arm':budget,
            'shared_parent':ref,'policy':'same_parent_data_seed_gate_budget_only_preservation_differs',
            'generation_quality_claimed':False})
        print(json.dumps({'event':'completed','report':str(report_root/'completed.json')}),flush=True)
    except BaseException as error:
        write_json(report_root/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),
            'completed_arms':completed,'artifacts_preserved':True})
        raise


if __name__=='__main__':main()
