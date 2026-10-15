"""Reject accidental changes or duplicate steps when resuming the bounded pilot."""
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import torch


def pilot_code_sha256(project):
    """Include the complete Qwen adapter package, including reused dependencies."""
    project=Path(project)
    paths=list((project/'src/pose_control/qwen21_full').rglob('*.py'))
    paths += [project/path for path in ('scripts/train/train_qwen21_dual_v8_pilot.py',
        'src/data/hi4d_smplx_dataset.py','scripts/precompute/precompute_hi4d_smplx_controls.py')]
    paths += [path for name in ('src/__init__.py','src/data/__init__.py','src/pose_control/__init__.py')
        if (path:=project/name).exists()]
    return {path.relative_to(project).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def tensor_cache_sha256(value):
    """Hash values canonically; torch ZIP archive names are not data identity."""
    def describe(item):
        if torch.is_tensor(item):
            raw=item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return dict(tensor_dtype=str(item.dtype),shape=list(item.shape),bytes_sha256=hashlib.sha256(raw).hexdigest())
        if isinstance(item,Mapping):
            if any(not isinstance(key,str) for key in item):raise ValueError('cache keys must be strings')
            return {key:describe(child) for key,child in item.items()}
        if isinstance(item,(tuple,list)):return [describe(child) for child in item]
        if item is None or type(item) in (str,int,float,bool):return item
        raise ValueError(f'unsupported cache value: {type(item).__name__}')
    encoded=json.dumps(describe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_resume_contract(saved, current, *, requested_steps):
    previous=saved.get('run_contract')
    if not isinstance(previous,Mapping) or not isinstance(current,Mapping) or not current:
        raise ValueError('resume requires a complete immutable run contract')
    for key in sorted(set(previous)|set(current)):
        if key not in previous or key not in current or previous[key]!=current[key]:
            raise ValueError(f'resume contract mismatch: {key}')
    step=saved.get('step')
    if type(step) is not int or not 0<step<=requested_steps<=300:
        raise ValueError('resume step must be positive and cannot exceed the requested pilot steps')
    return step


def validate_optimizer_contract(state, contract, *, parameters=None, checkpoint_step=None):
    if not isinstance(state,Mapping) or len(state.get('param_groups',[]))!=1:
        raise ValueError('resume optimizer requires the original single parameter group')
    if contract.get('name')!='AdamW':raise ValueError('resume optimizer must be AdamW')
    group=state['param_groups'][0]
    for key,expected in contract.items():
        if key=='name':continue
        actual=group.get(key)
        if key=='betas' and actual is not None:actual=list(actual)
        if actual!=expected:raise ValueError(f'resume optimizer mismatch: {key}')
    if parameters is None:return
    parameters=list(parameters)
    ids=group.get('params',[])
    if ids!=list(range(len(parameters))) or any(type(key) is not int for key in ids):
        raise ValueError('resume optimizer parameter count/order is invalid')
    states=state.get('state')
    if not isinstance(states,Mapping) or not states or not set(states)<=set(ids):
        raise ValueError('resume optimizer state keys are missing or unknown')
    expected_keys={'step','exp_avg','exp_avg_sq'}|({'max_exp_avg_sq'} if group.get('amsgrad') else set())
    for key,p in zip(ids,parameters):
        if key not in states:continue  # Unused parameters legitimately have no moments.
        record=states[key]
        if not isinstance(record,Mapping) or set(record)!=expected_keys:
            raise ValueError(f'resume optimizer state fields are invalid: {key}')
        step=record['step']
        if torch.is_tensor(step):
            if step.numel()!=1 or not bool(torch.isfinite(step).all()):raise ValueError('resume optimizer step is invalid')
            step=float(step)
        if type(step) not in (int,float) or not math.isfinite(step) or step<=0 or int(step)!=step or (checkpoint_step is not None and step!=checkpoint_step):
            raise ValueError('resume optimizer step is invalid')
        for name in expected_keys-{'step'}:
            value=record[name]
            if not torch.is_tensor(value) or value.shape!=p.shape or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'resume optimizer moment is invalid: {key}.{name}')
            if name!='exp_avg' and bool((value<0).any()):raise ValueError(f'resume optimizer squared moment is negative: {key}.{name}')


def validate_metrics_resume(path, step):
    path=Path(path)
    if not path.exists():return  # A fresh directory starts after the parent checkpoint.
    try:
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        steps=[row['step'] for row in rows]
    except (ValueError,KeyError,TypeError) as exc:
        raise ValueError('resume metrics are incomplete or malformed') from exc
    if not steps or steps[-1]!=step:raise ValueError('resume checkpoint step disagrees with metrics tail')
    if any(type(value) is not int or value<1 for value in steps) or any(b!=a+1 for a,b in zip(steps,steps[1:])):
        raise ValueError('resume metrics must contain consecutive, unique steps')
