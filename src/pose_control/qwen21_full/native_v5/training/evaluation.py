"""Bounded, group-balanced evaluation and resumable validation loss guard."""
from copy import deepcopy
import math
from .config import digest


def fixed_evaluation_selection(config,data):
    role='observation' if config.experiment_mode=='fixed_pair_overfit' else 'validation'
    key=(config.seed,config.probe_samples,role)
    cached=getattr(data,'_fixed_evaluation_selections',{})
    if key in cached:return deepcopy(cached[key])
    groups={};seen=set()
    for row in data.records:
        pair=(row['source_id'],row['index'])
        if row['role']!=role or pair in seen:continue
        seen.add(pair)
        group=data.group_indices[row['source_id']][row['pair_id']]
        groups.setdefault(group,[]).append({k:row[k] for k in ('source_id','index','pair_id')})
    def rank(row):return digest([config.seed,row['source_id'],row['pair_id']])
    for rows in groups.values():rows.sort(key=rank)
    ordered=sorted(groups,key=lambda g:digest([config.seed,g]));selected=[];targets=set()
    def target(row):
        source,index=row['source_id'],row['index'];raw=data.datasets[source].rows[index]
        value=raw.get('target_rgb_sha256') or raw.get('driver_rgb_sha256')
        if value is None:
            value=data.sample(source,index)['local_condition_v4'].metadata[0]['driver_rgb_sha256']
        return value
    while len(selected)<config.probe_samples and any(groups.values()):
        for group in ordered:
            candidates=groups[group]
            if not candidates:continue
            chosen=next((i for i,row in enumerate(candidates) if target(row) not in targets),0)
            row=candidates.pop(chosen);targets.add(target(row));selected.append(row)
            if len(selected)==config.probe_samples:break
    if not selected:raise ValueError('Missing bounded '+role+' evaluation examples')
    cached[key]=deepcopy(selected);data._fixed_evaluation_selections=cached
    return selected


class ValidationGuard:
    def __init__(self,config,*,selection=None,state=None):
        self.ratio=config.validation_loss_limit_ratio;self.patience=config.validation_failure_patience
        self.state=deepcopy(state) if state is not None else {
            'schema':'native_v5_validation_state_v1','enabled':self.ratio is not None,
            'baseline_loss':None,'last_loss':None,'consecutive_failures':0,'passed':False,'finite':True,
            'triggered':False,'last_probe_update':None,'last_generation_update':None,
            'loss_limit_ratio':self.ratio,'failure_patience':self.patience,'selection':selection or []}
        if state is not None:
            if (self.state.get('loss_limit_ratio')!=self.ratio or self.state.get('failure_patience')!=self.patience or
                    (selection is not None and self.state.get('selection')!=selection)):
                raise ValueError('Resumed validation policy/selection differs')
            if self.state.get('triggered'):raise ValueError('Cannot resume a triggered validation guard')

    def observe(self,report):
        values=[r['objective'] for r in report['records'] if 'objective' in r]
        finite=bool(values) and all(math.isfinite(v) for v in values)
        loss=sum(values)/len(values) if finite else None
        self.state.update(last_loss=loss,finite=finite,last_probe_update=report['update'])
        if not finite:
            self.state.update(triggered=True,passed=False)
            raise RuntimeError('validation loss is nonfinite or empty')
        if self.state['baseline_loss'] is None:self.state['baseline_loss']=loss
        failed=self.ratio is not None and loss>self.ratio*self.state['baseline_loss']
        self.state['consecutive_failures']=self.state['consecutive_failures']+1 if failed else 0
        self.state['triggered']=self.state['consecutive_failures']>=self.patience
        self.state['passed']=not self.state['triggered']
        if self.state['triggered']:raise RuntimeError('validation loss exceeded baseline patience')
        return self.state
