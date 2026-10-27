"""CPU-only final audit of the completed experiment, independent of UI labels."""
import argparse
import csv
import json
from pathlib import Path
from .common import quality_manifest,training_manifest,write_json,sha256

def audit(root):
    expected=[c['id'] for c in quality_manifest()]+[c['id'] for c in training_manifest()]+[c['id'] for c in training_manifest(True)]
    expected += [f'STABLE_M_{r}_{i}' for r in (512,768,1024) for i in range(3)]
    expected += [f'ADAPTER_INFER_{r}_{t}' for r in (512,768,1024) for t in ('S0','D0')]
    expected += [f'Q4_{t}_{r}_{s}' for t in ('S2','D1') for r in (768,1024) for s in (42,1024)]
    errors=[];status_counts={};checked=[]
    for name in expected:
        d=root/'runs'/name
        if not (d/'summary.json').exists():errors.append(f'Missing run {name}');continue
        c=json.loads((d/'config.json').read_text(encoding='utf-8'));s=json.loads((d/'summary.json').read_text(encoding='utf-8'))
        status_counts[s['status']]=status_counts.get(s['status'],0)+1
        if s['status']!='PASS':errors.append(f'{name}: {s["status"]}');continue
        if s.get('nvml_error'):errors.append(f'{name}: NVML error')
        with (d/'gpu_samples.csv').open() as f:
            values=[float(r['used_gib']) for r in csv.DictReader(f)]
        if not values or abs(max(values)-s['peak_nvml_gib'])>1e-6:errors.append(f'{name}: NVML summary mismatch')
        if c['kind']=='train':
            v=json.loads((d/'validation.json').read_text(encoding='utf-8'))
            if v['backbone_before']!=v['backbone_after']:errors.append(f'{name}: base weights changed')
            if v['gradients']['heads_nonzero']!=[True]*6 or not v['gradients']['body_nonzero']:errors.append(f'{name}: gradient gate failed')
            if v['optimizer_dtypes']!=['torch.float32']:errors.append(f'{name}: optimizer precision mismatch')
            updates=json.loads((d/'updates.json').read_text())
            expected_count=100 if c['group']=='stable' else 25
            if len(updates)!=expected_count or s['train_steps']!=expected_count:errors.append(f'{name}: update count mismatch')
            if c['checkpointing'] and min(v['recomputation'].values())<2:errors.append(f'{name}: no actual checkpoint recomputation')
            if c['encoding']=='online' and not all(x['allclose'] for x in v['cached_online_comparison'].values()):errors.append(f'{name}: cache comparison failed')
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
    if len(capacity)!=3:errors.append('All three resolution capacity gates not yet passed')
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8')) if (root/'scores.json').exists() else {}
    quality_ids=[name for name in expected if name.startswith('Q')]
    if any(name not in scores for name in quality_ids):errors.append('Some native editing images lack visual scores')
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
