"""Read-only health inspection of a Native run; no model construction or optimizer."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.run_root).resolve()
    journal=[json.loads(p.read_text(encoding='utf8'))['record'] for p in sorted((root/'journal').glob('*.json'))]
    reference=next(row['plan'] for row in reversed(journal) if row['event'] in ('plan','code_revision'))
    plan=json.loads(Path(reference['path']).read_text(encoding='utf8'));repo=Path(plan['repo_root'])
    sys.path.insert(0,str(repo/'experiments/bilibili-single-spatial-training/scripts'));sys.path.insert(0,str(repo))
    from supervisor import status,verify_runtime_code,read_ref,write_new_json,file_ref
    state=status(root);verify_runtime_code(plan);errors=[];warnings=[]
    if not state['owner_alive']:errors.append('Supervisor is not alive')
    if state['status'] not in ('running','waiting_resources','completed'):errors.append('Run status: '+state['status'])
    if state['failure_counts']:warnings.append({'failure_counts':state['failure_counts']})
    paths=sorted(root.glob('stages/*/segments/*/metrics.jsonl'),key=lambda p:p.stat().st_mtime_ns)
    if not paths:raise ValueError('No formal training metrics yet')
    path=paths[-1];rows=[];partial=0
    # Read a single bounded file snapshot; a concurrent partial append is not a committed record.
    with path.open('rb') as stream:raw=stream.read()
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):partial+=1;continue
        rows.append(json.loads(line)['record'])
    updates=[r for r in rows if r.get('optimizer_updated')]
    probes=[r for r in rows if r.get('kind')=='held_out_loss_probe']
    if not updates:raise ValueError('No observed formal optimizer updates')
    if any(b['global_update']!=a['global_update']+1 for a,b in zip(updates,updates[1:])):errors.append('Non-contiguous update counters in current segment')
    if any(not math.isfinite(r['gradient_norm']) or r['gradient_norm']<=0 for r in updates):errors.append('Invalid gradient norm')
    if any(not math.isfinite(loss['weighted_micro_objective']) for r in updates for loss in r['losses']):errors.append('Nonfinite training objective')
    train_ids={source['source_id'] for source in plan.get('sources',[])}
    execution=json.loads((path.parents[2]/'execution_config.json').read_text(encoding='utf8'))
    train_ids={source['source_id'] for source in execution['sources'] if source['role']=='train'}
    val_ids={source['source_id'] for source in execution['sources'] if source['role']=='validation'}
    for record in updates:
        if len(record['window'])!=execution['stage']['accumulation']:errors.append('Incomplete accumulation window');break
        if any(item['source'] not in train_ids for item in record['window']):errors.append('Held-out source appeared in optimizer sampling');break
    validation=[];selection=None
    for probe in probes:
        if any(record.get('source_id') not in val_ids for record in probe['records']):errors.append('Validation source mismatch')
        if not probe['records'] or any(not math.isfinite(r['objective']) for r in probe['records']):errors.append('Nonfinite validation')
        if selection is None:selection=probe.get('selection')
        elif selection!=probe.get('selection'):errors.append('Fixed validation selection changed')
        validation.append({'update':probe['update'],'objective_mean':statistics.mean(r['objective'] for r in probe['records']),
                           'samples':len(probe['records'])})
    checkpoint=None
    commits=[value for value in state['stage_commits'].values() if value]
    if commits:
        from src.pose_control.qwen21_full.native_v5.training.checkpointing import load_committed
        import torch
        torch.set_num_threads(1)
        commit=max(commits,key=lambda v:v['global_update'])
        saved,ref,tail=load_committed(commit['checkpoint']);training=saved['training_state']
        gradient=read_ref(training['gradient_report'])
        if gradient.get('frozen_before')!=training['frozen_state'] or gradient.get('frozen_after')!=training['frozen_state']:
            errors.append('Frozen checkpoint state differs from production acceptance')
        guard=training['validation_state']
        if not guard.get('passed') or not guard.get('finite') or guard.get('triggered'):errors.append('Committed validation guard not passing')
        used=training.get('frozen_cache_state',{}).get('used',{})
        for cache_ref in used.values():
            from supervisor import checked_ref
            checked_ref(cache_ref)
        checkpoint={'global_update':training['global_update'],'stage_update':training['stage_update'],
                    'checkpoint':ref,'commit':file_ref(Path(ref['path']).with_suffix('.commit.json')),
                    'validation_state':guard,'used_cache_files_sha_verified':len(used),
                    'optimizer_state_present':bool(training['optimizer_state']),
                    'rng_present':bool(training['rng']),'uncommitted_metrics_tail_bytes':tail['uncommitted_tail_bytes']}
        del saved
    recent=updates[-20:];seconds=[r['update_seconds'] for r in recent]
    cache_count=sum(1 for _ in (root/'frozen_qwen/entries').glob('*.json'))
    result={'schema':'native_training_health_inspection_v1','checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'passed':not errors,'errors':errors,'warnings':warnings,'runtime_repo':str(repo),'status':state['status'],
            'task':state.get('task'),'supervisor_pid':state['pid'],'observed_successful_updates':updates[-1]['global_update'],
            'committed_updates':state['committed_updates'],'total_budget':plan['total_updates'],
            'recent_seconds_per_update_mean':statistics.mean(seconds),'recent_seconds_per_update_median':statistics.median(seconds),
            'recent_train_objective_mean':statistics.mean(sum(x['weighted_micro_objective'] for x in r['losses']) for r in recent),
            'validation':validation,'checkpoint':checkpoint,'cached_pairs':cache_count,'partial_append_lines_ignored':partial,
            'metrics':str(path),'generation_not_due_yet':updates[-1]['global_update']<execution['generation_every_updates'],
            'no_model_or_optimizer_constructed':True}
    reference=write_new_json(args.output,result)
    print(json.dumps({'report':reference,'summary':{k:result[k] for k in ('passed','errors','observed_successful_updates',
          'committed_updates','recent_seconds_per_update_mean','recent_train_objective_mean','validation','cached_pairs')}}),flush=True)


if __name__=='__main__':main()
