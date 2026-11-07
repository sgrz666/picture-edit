"""Compare old/new audit metrics on real Adapter shapes, without optimizer or Qwen."""
import argparse
from dataclasses import fields,is_dataclass,replace
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

from supervisor import file_ref,read_ref,write_new_json

CHANGED = ('trainer.py','validation.py','parameters.py')
TRAINING = Path('src/pose_control/qwen21_full/native_v5/training')


def load_baseline(repo,name):
    key='src.pose_control.qwen21_full.native_v5.training._speed_baseline_'+name
    spec=importlib.util.spec_from_file_location(key,Path(repo)/TRAINING/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
    return module


def assert_tree_equal(a,b):
    import numpy as np
    import torch
    if type(a) is not type(b):raise AssertionError('Sample tree type differs')
    if torch.is_tensor(a):
        assert a.shape==b.shape and a.dtype==b.dtype
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a,np.ndarray):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    elif is_dataclass(a):
        for f in fields(a):assert_tree_equal(getattr(a,f.name),getattr(b,f.name))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_tree_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree_equal(x,y)
    else:assert a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-repo',required=True)
    parser.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--initialization',default='/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1/initialization/native_v5_init_v2.pt')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();repo=Path(args.repo_root).resolve();baseline=Path(args.baseline_repo).resolve()
    # This is a separate short-lived audit process; do not alter the running job's environment.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','backend:cudaMallocAsync')
    sys.path.insert(0,str(repo))
    import torch
    torch.set_num_threads(4)
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    from src.pose_control.qwen21_full.native_v5.training.parameters import ParameterCatalog
    from src.pose_control.qwen21_full.native_v5.training import validation
    from src.pose_control.qwen21_full.native_v5.training.config import stage_preset
    from src.pose_control.qwen21_full.native_v5.training.data import sample_status
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5
    from run_plan import INACTIVE_GROUPS
    before={p.relative_to(baseline).as_posix():file_ref(p)['sha256'] for p in (baseline/'src').rglob('*.py')}
    after={p.relative_to(repo).as_posix():file_ref(p)['sha256'] for p in (repo/'src').rglob('*.py')}
    changed=sorted(name for name in before.keys()|after.keys() if before.get(name)!=after.get(name))
    if changed!=sorted((TRAINING/name).as_posix() for name in CHANGED):
        raise AssertionError('Unexpected production source changes: '+str(changed))
    manifest=Path(args.run_root)/'native/manifests/train/pairs.jsonl'
    dataset=NativeFullDatasetV5(manifest,load_target=True,data_profile='full_smplx')
    examples=[]
    for index in (0,len(dataset)//2,len(dataset)-1):
        first,second=dataset[index],dataset[index]
        assert_tree_equal(first,second)
        status=sample_status(first,stage_preset('S0').routes[0],TrainingConfig())
        assert status['bucket'][2]==second['native_scene_kind']
        examples.append({'index':index,'pair_id':dataset.rows[index]['pair_id'],'all_fields_bitwise_equal':True})
    del dataset,first,second
    old_validation=load_baseline(baseline,'validation');old_parameters=load_baseline(baseline,'parameters')
    saved=torch.load(args.initialization,map_location='cpu',weights_only=True,mmap=True)
    runtime=NativeFullRuntimeV5.from_checkpoint(saved).to('cuda');del saved
    catalog=ParameterCatalog(runtime);old_catalog=old_parameters.ParameterCatalog(runtime)
    assert catalog.record()==old_catalog.record()
    generator=torch.Generator(device='cuda').manual_seed(42)
    records=[]
    for phase in ('S0','S1','S2','S3'):
        stage=stage_preset(phase)
        stage=replace(stage,routes=tuple(r for r in stage.routes if r.branch!='interaction'),exclude_groups=INACTIVE_GROUPS)
        active=catalog.apply(stage)
        for i,name in enumerate(sorted(active)):
            p=catalog.parameters[name]
            p.grad=None if i%5==0 else torch.zeros_like(p) if i%5==1 else torch.randn(p.shape,device='cuda',dtype=p.dtype,generator=generator)*.001
        last=catalog.parameters[sorted(active)[-1]]
        if last.grad is None:last.grad=torch.ones_like(last)*.001
        common={'active':active,'base_parameters':{},'evaluators':[],'pipe':SimpleNamespace(vae=None,text_encoder=None)}
        backend=SimpleNamespace(catalog=catalog,**common);old_backend=SimpleNamespace(catalog=old_catalog,**common)
        gradients=lambda:((n,p.grad) for n,p in sorted(catalog.parameters.items()) if p.grad is not None)
        before_hash=validation.tensor_digest(gradients())
        expected=old_validation.audit_gradients(old_backend)
        actual=validation.audit_gradients(backend)
        assert actual==expected
        old_times=[];new_times=[]
        for repeat in range(args.repeats):
            for function,owner,durations in ((old_validation.audit_gradients,old_backend,old_times),
                                             (validation.audit_gradients,backend,new_times)):
                torch.cuda.synchronize();start=time.perf_counter();function(owner);torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
        assert validation.tensor_digest(gradients())==before_hash
        records.append({'phase':phase,'active_parameters':len(active),'catalog_parameters':len(catalog.rows),
            'gradient_elements':sum(p.grad.numel() for p in catalog.parameters.values() if p.grad is not None),
            'metrics_exact':True,'gradient_bytes_unchanged':True,'baseline_seconds':old_times,'optimized_seconds':new_times,
            'baseline_median_seconds':statistics.median(old_times),'optimized_median_seconds':statistics.median(new_times),
            'audit_speedup':statistics.median(old_times)/statistics.median(new_times)})
        print(records[-1],flush=True)
    new_code={name:file_ref(repo/TRAINING/name) for name in CHANGED}
    result={'schema':'native_training_throughput_audit_v1','passed':True,'optimizer_created':False,
        'optimizer_step_executed':False,'baseline_repo':str(baseline),'repo_root':str(repo),'new_code':new_code,
        'baseline_code':{name:file_ref(baseline/TRAINING/name) for name in CHANGED},'producer':file_ref(__file__),
        'initialization':file_ref(args.initialization),'sample_manifest':file_ref(manifest),'sample_checks':examples,
        'changed_src_files':changed,'unchanged_src_files':len(before)-len(changed),'device':torch.cuda.get_device_name(0),
        'records':records,'scope':'Isolated audit on real Adapter parameter shapes with synthetic gradient buffers; not production FM training or a quality claim.'}
    print(write_new_json(args.output,result),flush=True)


if __name__=='__main__':main()
