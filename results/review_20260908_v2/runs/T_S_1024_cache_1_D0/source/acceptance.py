"""CPU-only final audit of the completed experiment, independent of UI labels."""
import argparse
import csv
import json
import math
from pathlib import Path
from .common import quality_manifest,training_manifest,write_json,sha256

def audit(root):
    manifests=quality_manifest()+training_manifest()+training_manifest(True)
    contracts={c['id']:c for c in manifests}
    expected=list(contracts)
    expected += [f'STABLE_M_{r}_{i}' for r in (512,768,1024) for i in range(3)]
    expected += [f'ADAPTER_INFER_{r}_{t}' for r in (512,768,1024) for t in ('S0','D0')]
    expected += [f'Q4_{t}_{r}_{s}' for t in ('S2','D1') for r in (768,1024) for s in (42,1024)]
    errors=[];status_counts={};checked=[]
    for name in expected:
        d=root/'runs'/name
        if not (d/'summary.json').exists():errors.append(f'Missing run {name}');continue
        c=json.loads((d/'config.json').read_text(encoding='utf-8'));s=json.loads((d/'summary.json').read_text(encoding='utf-8'))
        if name in contracts and any(c.get(k)!=value for k,value in contracts[name].items()):errors.append(f'{name}: manifest configuration mismatch')
        status_counts[s['status']]=status_counts.get(s['status'],0)+1
        if s['status']!='PASS':errors.append(f'{name}: {s["status"]}');continue
        if s.get('nvml_error'):errors.append(f'{name}: NVML error')
        with (d/'gpu_samples.csv').open() as f:
            values=[float(r['used_gib']) for r in csv.DictReader(f)]
        if not values or abs(max(values)-s['peak_nvml_gib'])>1e-6:errors.append(f'{name}: NVML summary mismatch')
        if c['kind']=='train':
            v=json.loads((d/'validation.json').read_text(encoding='utf-8'))
            if v['backbone_before']!=v['backbone_after']:errors.append(f'{name}: base weights changed')
            module_keys={'transformer','vae','lmm','connector'} if c['encoding']=='online' else {'transformer','vae'}
            if set(v['backbone_before'])!=module_keys:errors.append(f'{name}: incomplete frozen-module evidence')
            if not s.get('validation_passed') or v.get('zero_max_abs_diff') is None or not 0<=v['zero_max_abs_diff']<=.01 or v['gradients'].get('backbone_gradients_absent') is not True:errors.append(f'{name}: frozen/zero equivalence gate failed')
            if v['gradients']['heads_nonzero']!=[True]*6 or not v['gradients']['body_nonzero']:errors.append(f'{name}: gradient gate failed')
            if v['optimizer_dtypes']!=['torch.float32']:errors.append(f'{name}: optimizer precision mismatch')
            updates=json.loads((d/'updates.json').read_text())
            expected_count=100 if c['group']=='stable' else 25
            if len(updates)!=expected_count or s['train_steps']!=expected_count:errors.append(f'{name}: update count mismatch')
            if c['checkpointing'] and min(v['recomputation'].values())<2:errors.append(f'{name}: no actual checkpoint recomputation')
            if not c['checkpointing'] and v['recomputation']!={'dit':1,'adapter':1}:errors.append(f'{name}: unexpected recomputation')
            shapes=s.get('semantic_shapes',{})
            if set(shapes)!=set(c.get('training_tasks',[c['task']])):errors.append(f'{name}: missing semantic evidence')
            for task,shape in shapes.items():
                if set(shape)!={'seq','pooled','ref','target'} or shape['seq'][1]<=64 or shape['ref'][-2:]!=[c['resolution']//8]*2 or shape['target']!=shape['ref']:errors.append(f'{name}: invalid real condition shapes')
            if c['encoding']=='online':
                comparison=v.get('cached_online_comparison') or {}
                if set(comparison)!={'seq','pooled','ref','target'} or not all(x.get('allclose') is True for x in comparison.values()):errors.append(f'{name}: cache comparison failed')
            if any(not math.isfinite(u[k]) for u in updates for k in ('loss','gradient_norm')):errors.append(f'{name}: nonfinite optimization record')
            if c.get('alternate'):
                if [u['task'] for u in updates]!=['S0','D0']*50:errors.append(f'{name}: alternation incorrect')
                for task in ('S0','D0'):
                    if not (d/'task_evidence'/task/'control_metadata.json').exists():errors.append(f'{name}: {task} controls missing')
        if c['kind'] in ('quality','adapter_infer'):
            if not (d/'output.png').exists() or not (d/'output_content.png').exists():errors.append(f'{name}: output missing')
        checked.append(name)
    a=root/'runs/Q0_S0_42_stretch_35_4.5/output.png';b=root/'runs/REPRO_Q0_S0_42/output.png'
    repro=a.exists() and b.exists() and sha256(a)==sha256(b)
    if not repro:errors.append('Reproduction output hashes do not match')
    capacity=json.loads((root/'capacity.json').read_text()) if (root/'capacity.json').exists() else []
    if len(capacity)!=3 or {r.get('resolution') for r in capacity}!={512,768,1024}:errors.append('All three resolution capacity gates not yet passed')
    for r in capacity:
        numeric=('stable_peak_nvml_gib','steady_peak_allocated_gib','steady_peak_reserved_gib','lifecycle_peak_nvml_gib','suggested_capacity_gib','repeat_variation','step_p50_seconds')
        if any(not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]) for k in numeric):errors.append('Invalid capacity numeric fields');continue
        res=r['resolution'];stable=[];lifecycle=[];cache_tasks=set();infer_tasks=set()
        for p in (root/'runs').glob('*/summary.json'):
            cfg=json.loads((p.parent/'config.json').read_text(encoding='utf-8'));summary=json.loads(p.read_text())
            if cfg.get('resolution')!=res or summary['status']!='PASS':continue
            if cfg.get('group')=='stable':stable.append(summary);lifecycle.append(summary)
            if cfg.get('kind')=='cache':
                if summary.get('cache_roundtrip_exact'):cache_tasks.add(cfg['task'])
                lifecycle.append(summary)
            if cfg.get('kind')=='adapter_infer':infer_tasks.add(cfg['task']);lifecycle.append(summary)
        if len(stable)!=3 or cache_tasks!={'S0','D0'} or infer_tasks!={'S0','D0'}:errors.append(f'{res}: capacity prerequisite evidence missing');continue
        peak=max(s['peak_nvml_gib'] for s in lifecycle);peaks=[s['peak_nvml_gib'] for s in stable]
        variation=(max(peaks)-min(peaks))/min(peaks)
        if r['lifecycle_peak_nvml_gib']!=peak or r['stable_peak_nvml_gib']!=max(peaks) or variation>.05 or abs(r['repeat_variation']-variation)>1e-8 or r['suggested_capacity_gib']!=math.ceil(peak+max(2,.1*peak)):errors.append(f'{res}: capacity calculation mismatch')
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8')) if (root/'scores.json').exists() else {}
    quality_ids=[name for name in expected if name.startswith('Q')]
    if any(name not in scores for name in quality_ids):errors.append('Some native editing images lack visual scores')
    for name in quality_ids:
        score=scores.get(name,{})
        for key in ('action','identity','clothing','background','anatomy'):
            value=score.get(key)
            if key not in score or not (key=='identity' and value is None) and (type(value) is not int or value not in (0,1,2)):errors.append(f'{name}: invalid score {key}')
        if any(type(score.get(k)) is not bool for k in ('added_person','missing_person','wrong_person')) or not score.get('reviewer') or not score.get('note'):errors.append(f'{name}: incomplete score provenance/flags')
    selection_path=root/'quality_selection.json'
    if not selection_path.exists():errors.append('Quality selection missing')
    else:
        selection=json.loads(selection_path.read_text(encoding='utf-8'))
        for name in quality_ids:
            if name.startswith('Q4_') and (root/'runs'/name/'config.json').exists():
                cfg=json.loads((root/'runs'/name/'config.json').read_text(encoding='utf-8'))
                if any(cfg.get(k)!=v for k,v in selection['parameters'].items()):errors.append(f'{name}: selected parameters mismatch')
    result=dict(passed=not errors,expected_primary_runs=len(expected),checked_primary_runs=len(checked),status_counts=status_counts,
                errors=errors,reproducible_output_exact=repro,quality_score_source='AI visual initial review; human review pending',
                interpretation='Acceptance of experiment implementation and resource measurement, not pose-edit capability')
    write_json(root/'ACCEPTANCE.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();result=audit(a.root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
