"""Canonical name-based parameter catalog and fixed AdamW superset."""
from dataclasses import dataclass,asdict
import math
import torch
from .config import digest

HEAD_PREFIXES=('base.body.output_heads.','face_controller.output_heads.','face_controller.static_output.',
    'hand_controller.output_heads.','hand_controller.static_output.','interaction_head.output_heads.',
    'base.depth.projection.','base.identity_fusion.output.')
BODY_GROUPS={'native_body','body21','geometry_dual','champ_cnn','champ_attention','detail_cnn','depth_fusion','strength_body'}

@dataclass(frozen=True)
class ParameterRow:
    name: str
    runtime_group: str
    role: str
    lineage: str
    shape: tuple
    dtype: str
    elements: int
    optimizer_group: str

def role_for(name,group):
    if group=='inactive':return 'inactive'
    if name.startswith(HEAD_PREFIXES):return 'head'
    if group.startswith('strength_'):return 'strength'
    if group in ('native_body','face_controller','hand_controller','interaction_controller','depth_fusion') or name.startswith('base.identity_fusion.'):
        return 'controller'
    return 'encoder'

class ParameterCatalog:
    def __init__(self,runtime):
        params=dict(runtime.named_parameters());groups=runtime.named_parameter_groups();seen={};rows=[]
        report=runtime.inheritance_report or {};fresh=set(report.get('new_initialized',()))
        inherited=set(report.get('inherited',()))|set(report.get('adapted',()))
        object_names={}
        for group,items in groups.items():
            for name,p in items:
                if name in seen or id(p) in object_names:raise ValueError('Duplicate/shared parameter registered twice: '+name)
                if params.get(name) is not p:raise ValueError('Catalog path does not identify runtime parameter: '+name)
                seen[name]=group;object_names[id(p)]=name
                role=role_for(name,group);lineage='new' if name in fresh else 'inherited' if name in inherited else 'unknown'
                rows.append(ParameterRow(name,group,role,lineage,tuple(p.shape),str(p.dtype),p.numel(),group+'|'+role+'|'+lineage))
        if set(seen)!=set(params):raise ValueError('Catalog must cover every Adapter parameter')
        self.rows=tuple(sorted(rows,key=lambda r:r.name));self.parameters=params;self.runtime=runtime

    def record(self):return [asdict(r) for r in self.rows]
    def signature(self):return digest(self.record())

    def active_names(self,stage):
        branches={r.branch for r in stage.routes};groups=set()
        if stage.phase=='S0':groups=BODY_GROUPS.copy()
        elif stage.phase=='S3':groups={r.runtime_group for r in self.rows if r.runtime_group not in ('inactive','geometry_shared')}
        else:
            for branch in branches:
                family={'face':{'face_heads'},'hand':{'hand_heads'},'detail':{'face_heads','hand_heads'},'interaction':{'interaction_heads'}}[branch]
                groups |= family
                if stage.phase=='S2':
                    if branch in ('face','hand'):
                        groups|={branch+'_encoder',branch+'_controller'}|{r.runtime_group for r in self.rows if r.runtime_group.startswith('strength_'+branch)}
                    elif branch=='detail':groups|={'detail_bridge','face_controller','hand_controller','strength_detail'}
                    else:groups|={'geometry_dual','relation_raster','relation_tokens','interaction_controller','strength_interaction'}
        if stage.unfreeze_geometry:
            if stage.phase not in ('S0','S3'):raise ValueError('Shared Geometry unfreeze is explicit S0/S3 only')
            groups.add('geometry_shared')
        known={r.runtime_group for r in self.rows}
        if (set(stage.include_groups)|set(stage.exclude_groups))-known:raise ValueError('Unknown runtime group override')
        if set(stage.include_groups)-groups:raise ValueError('Stage override would unfreeze an unrelated/frozen branch')
        groups|=set(stage.include_groups);groups-=set(stage.exclude_groups);groups.discard('inactive')
        names={r.name for r in self.rows if r.runtime_group in groups and r.role!='inactive'}
        if stage.phase=='S1':names={r.name for r in self.rows if r.name in names and r.role=='head'}
        if not names:raise ValueError('Stage has no active parameters')
        return names

    def apply(self,stage):
        active=self.active_names(stage)
        coarse={k:any(r.name in active for r in self.rows if r.runtime_group==k) for k in self.runtime.named_parameter_groups()}
        coarse['inactive']=False
        self.runtime.set_trainability('joint',overrides=coarse)
        for name,p in self.parameters.items():
            p.requires_grad_(name in active);p.grad=None
        self.runtime.train(self.runtime.training)
        self.assert_flags(active)
        return active

    def assert_flags(self,active):
        if any(dict(self.runtime.named_parameters()).get(n) is not p for n,p in self.parameters.items()):raise RuntimeError('Parameter objects changed after catalog binding')
        wrong=[n for n,p in self.parameters.items() if p.requires_grad!=(n in active)]
        if wrong:raise RuntimeError('Frozen mask changed after installation: '+str(wrong[:8]))

    def base_lr(self,row,spec):
        if spec.preset=='legacy_flat':rate=1e-4
        elif row.role in ('head','strength'):rate=1e-4
        elif spec.preset=='full_legacy':rate=5e-5 if row.role=='encoder' else 1e-5
        elif row.role=='controller' or row.lineage=='new':rate=5e-5
        else:rate=1e-5
        rate=spec.group_lrs.get(row.optimizer_group,spec.group_lrs.get(row.runtime_group,rate))
        return rate*spec.multiplier

    def group_spec(self,spec):
        known={r.runtime_group for r in self.rows}|{r.optimizer_group for r in self.rows}
        if set(spec.group_lrs)-known:raise ValueError('Unknown optimizer LR override')
        groups={}
        for r in self.rows:
            if r.role=='inactive':continue
            group=groups.setdefault(r.optimizer_group,{'name':r.optimizer_group,'names':[],'lr':self.base_lr(r,spec),'weight_decay':spec.weight_decay})
            group['names'].append(r.name)
        return [groups[k] for k in sorted(groups)]

    def build_optimizer(self,spec):
        groups=[{**g,'params':[self.parameters[n] for n in g['names']]} for g in self.group_spec(spec)]
        return torch.optim.AdamW(groups,betas=tuple(spec.betas),eps=spec.eps)

    def set_stage_lrs(self,optimizer,spec):
        expected=self.group_spec(spec)
        if [g['name'] for g in optimizer.param_groups]!=[g['name'] for g in expected]:raise ValueError('Optimizer superset group order changed')
        for group,new in zip(optimizer.param_groups,expected):
            if group['names']!=new['names']:raise ValueError('Optimizer named membership changed')
            group['lr']=new['lr'];group['weight_decay']=new['weight_decay']

class UpdateScheduler:
    def __init__(self,optimizer,spec,budget):
        self.optimizer=optimizer;self.spec=spec;self.budget=budget;self.completed=0
        self.base_lrs=[g['lr'] for g in optimizer.param_groups];self._apply()

    def factor(self):
        warm=self.spec.warmup_updates
        if self.spec.scheduler=='constant':return 1.
        if self.completed<warm:return (self.completed+1)/max(1,warm)
        if self.spec.scheduler=='warmup_constant':return 1.
        fraction=(self.completed-warm)/max(1,self.budget-warm)
        return .5*(1+math.cos(math.pi*min(1.,fraction)))

    def _apply(self):
        for group,base in zip(self.optimizer.param_groups,self.base_lrs):group['lr']=base*self.factor()
    def step(self):self.completed+=1;self._apply()
    def state_dict(self):return {'completed':self.completed,'base_lrs':self.base_lrs,'budget':self.budget,'spec':asdict(self.spec)}
    def load_state_dict(self,state):
        if digest(state['spec'])!=digest(asdict(self.spec)) or state['budget']!=self.budget or state['base_lrs']!=self.base_lrs:raise ValueError('Scheduler signature differs')
        self.completed=state['completed'];self._apply()
