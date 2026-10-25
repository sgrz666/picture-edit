"""Pure-stdlib training configuration. Resolve is not permission to train."""
from dataclasses import dataclass,field,asdict
from pathlib import Path
import hashlib,json,math

SCHEMA='native_v5_training_v1'
BRANCHES=('body','face','hand','detail','interaction','joint')

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def positive(value,name,*,zero=False):
    if type(value) not in (float,int) or not math.isfinite(value) or value<(0 if zero else 1e-300):raise ValueError('Invalid '+name)

def construct(cls,raw):
    unknown=set(raw)-set(cls.__dataclass_fields__)
    if unknown:raise ValueError('Unknown '+cls.__name__+' fields: '+str(sorted(unknown)))
    return cls(**raw)

def reference_syntax(ref):
    if ref is None:return
    if not isinstance(ref,dict) or not ref.get('path') or not Path(ref['path']).is_absolute():raise ValueError('Artifact reference needs an absolute path')
    sha=ref.get('sha256')
    if sha is not None and (not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)):raise ValueError('Invalid SHA256 reference')

@dataclass(frozen=True)
class LossSpec:
    velocity: float=1.
    face: float=0.
    hand: float=0.
    identity: float=0.
    perceptual: float=0.
    auxiliary_every: int=4
    auxiliary_sigma_max: float=.35

    def __post_init__(self):
        for k in ('velocity','face','hand','identity','perceptual'):positive(getattr(self,k),k,zero=True)
        if self.velocity<=0 or type(self.auxiliary_every)!=int or self.auxiliary_every<1 or not 0<self.auxiliary_sigma_max<=1:raise ValueError('Invalid loss/gate')

@dataclass(frozen=True)
class RouteSpec:
    name: str
    branch: str
    weight: float | None=None
    source_ids: tuple=()
    loss: LossSpec=field(default_factory=LossSpec)
    controls: dict=field(default_factory=dict)
    base_controls: dict=field(default_factory=dict)

    def __post_init__(self):
        if not self.name or '\0' in self.name or self.branch not in BRANCHES:raise ValueError('Explicit route name/branch required')
        if self.weight is not None:positive(self.weight,'route weight')
        if len(set(self.source_ids))!=len(self.source_ids):raise ValueError('Duplicate route source')
        booleans={'face','hand','detail','interaction'}
        numeric={'body_strength','interaction_strength','detail_strength','face_strength','hand_strength','face_identity_gain','face_texture_gain','face_geometry_gain','hand_geometry_gain','hand_texture_gain'}
        if set(self.controls)-booleans-numeric:raise ValueError('Unknown Native V5 route control')
        for k,v in self.controls.items():
            if k in booleans:
                if type(v)!=bool:raise ValueError('Control switch requires bool')
            else:positive(v,k,zero=True)
        base_bool={'detail','interaction','raster','part_contact','identity'};base_num={'depth_strength','relation_strength','interaction_strength','identity_strength'}
        if set(self.base_controls)-base_bool-base_num-{'depth_mode'}:raise ValueError('Unknown base route control')
        for k,v in self.base_controls.items():
            if k in base_bool and type(v)!=bool:raise ValueError('Base switch requires bool')
            if k in base_num:positive(v,k,zero=True)
        if self.base_controls.get('depth_mode','full_geometry') not in ('off','scene_only','full_geometry'):raise ValueError('Invalid explicit depth mode')
        if self.base_controls.get('interaction_strength',1.)!=1.:raise ValueError('Set Interaction strength in Native controls, not the inactive legacy fusion')

@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    role: str='train'
    manifest: dict | None=None
    groups: dict | None=None               # JSON pair_id -> dataset/original-video-or-sequence group
    data_profile: str='auto'
    scene_kind: str='any'
    weight: float | None=None

    def __post_init__(self):
        if not self.source_id or '\0' in self.source_id or self.role not in ('train','validation') or self.data_profile not in ('auto','full_smplx','hybrid_hi4d') or self.scene_kind not in ('any','single','dual'):
            raise ValueError('Invalid source declaration')
        if self.source_id=='champ_motions_example':raise ValueError('Unpaired CHAMP motion examples are not supervised training data')
        if self.weight is not None:positive(self.weight,'source weight')
        reference_syntax(self.manifest);reference_syntax(self.groups)

@dataclass(frozen=True)
class OptimizerSpec:
    preset: str='protect_inherited'
    multiplier: float=1.
    group_lrs: dict=field(default_factory=dict)
    betas: tuple=(.9,.999)
    eps: float=1e-8
    weight_decay: float=.01
    clip_norm: float=1.
    scheduler: str='constant'
    warmup_updates: int=0

    def __post_init__(self):
        if self.preset not in ('protect_inherited','full_legacy','legacy_flat') or self.scheduler not in ('constant','warmup_constant','warmup_cosine'):raise ValueError('Unknown optimizer/LR preset')
        positive(self.multiplier,'LR multiplier');positive(self.eps,'eps');positive(self.clip_norm,'clip')
        positive(self.weight_decay,'decay',zero=True)
        if len(self.betas)!=2 or any(not 0<=v<1 for v in self.betas) or type(self.warmup_updates)!=int or self.warmup_updates<0:raise ValueError('Invalid AdamW/warmup')
        for k,v in self.group_lrs.items():positive(v,'group LR '+k)

@dataclass(frozen=True)
class StageSpec:
    stage_id: str='S0_body'
    phase: str='S0'
    routes: tuple=()
    budget_updates: int | None=None
    accumulation: int=1
    max_attempts: int | None=None
    optimizer: OptimizerSpec=field(default_factory=OptimizerSpec)
    unfreeze_geometry: bool=False
    include_groups: tuple=()
    exclude_groups: tuple=()

    def __post_init__(self):
        if not self.stage_id or self.phase not in ('S0','S1','S2','S3') or not self.routes:raise ValueError('Stage/phase/routes required')
        if len({r.name for r in self.routes})!=len(self.routes):raise ValueError('Duplicate route name')
        if len(self.routes)>1 and any(r.weight is None for r in self.routes):raise ValueError('Multi-route weights must be explicit')
        if self.budget_updates is not None and (type(self.budget_updates)!=int or self.budget_updates<1):raise ValueError('Positive explicit update budget required')
        if type(self.accumulation)!=int or self.accumulation<1:raise ValueError('Invalid accumulation')
        if self.max_attempts is not None and (type(self.max_attempts)!=int or self.max_attempts<1):raise ValueError('Invalid attempt bound')
        if set(self.include_groups)&set(self.exclude_groups):raise ValueError('Conflicting freeze policy')
        if self.phase=='S0' and any(r.branch!='body' for r in self.routes):raise ValueError('S0 is Body/Geometry/Depth only')
        if self.phase in ('S1','S2') and any(r.branch in ('body','joint') for r in self.routes):raise ValueError('S1/S2 use explicit local/Interaction routes')

@dataclass(frozen=True)
class TrainingConfig:
    schema: str=SCHEMA
    stage: StageSpec=field(default_factory=lambda:stage_preset('S0'))
    sources: tuple=()
    initialization: dict | None=None
    model_path: str='/home/shangguanrz/project/pic-edit/models/Qwen-Image-2.1'
    model_manifest: dict | None=None       # JSON list of file references, pins Qwen config and shards
    cache_root: str | None=None
    cache_identity: dict | None=None
    evaluators: dict | None=None
    output_root: str | None=None
    gradient_report: dict | None=None
    body_baseline_report: dict | None=None
    seed: int=42
    prompt: str='Change the poses of the existing people. Preserve identities, clothing, faces, hands, and background.'
    save_every_updates: int=100
    probe_every_updates: int=0
    probe_samples: int=8
    num_workers: int=0
    micro_batch: int=1
    world_size: int=1
    eligibility_min_spatial_cells: int=4
    eligibility_feather_px: int=8

    def __post_init__(self):
        if self.schema!=SCHEMA or self.num_workers!=0 or self.micro_batch!=1 or self.world_size!=1:raise ValueError('V5 preparation supports one process/GPU, B1, workers0')
        if self.eligibility_min_spatial_cells<1 or not 0<=self.eligibility_feather_px<=32:raise ValueError('Invalid eligibility grid rules')
        if type(self.seed)!=int or not 0<=self.seed<2**63 or any(type(x)!=int for x in (self.save_every_updates,self.probe_every_updates,self.probe_samples)) or self.save_every_updates<1 or self.probe_every_updates<0 or self.probe_samples<1:raise ValueError('Invalid seed/interval')
        for ref in (self.initialization,self.model_manifest,self.cache_identity,self.gradient_report,self.body_baseline_report):reference_syntax(ref)
        for name in ('model_path','cache_root','output_root'):
            if getattr(self,name) is not None and not Path(getattr(self,name)).is_absolute():raise ValueError('Absolute '+name+' required')
        if self.evaluators:
            for ref in self.evaluators.values():reference_syntax(ref)
        if len({s.source_id for s in self.sources})!=len(self.sources):raise ValueError('Duplicate source IDs')
        train=[s for s in self.sources if s.role=='train']
        if len(train)>1 and any(s.weight is None for s in train):raise ValueError('Multi-source weights must be explicit')
        ids={s.source_id for s in train}
        if any(set(r.source_ids)-ids for r in self.stage.routes):raise ValueError('Route references unknown training sources')

    def record(self):return asdict(self)
    def signature(self):
        value=self.record();value.pop('gradient_report');value.pop('body_baseline_report')
        return digest(value)
    def missing(self):
        missing=[k for k in ('initialization','model_manifest','cache_root','cache_identity','output_root') if not getattr(self,k)]
        if self.stage.budget_updates is None:missing.append('stage.budget_updates')
        if not any(s.role=='train' for s in self.sources):missing.append('train source')
        if not any(s.role=='validation' for s in self.sources):missing.append('validation source')
        for s in self.sources:
            if not s.manifest:missing.append(s.source_id+'.manifest')
            if not s.groups:missing.append(s.source_id+'.groups')
        return missing
    def require_ready(self):
        if self.missing():raise ValueError('Preparation is not train-ready: '+', '.join(self.missing()))
        if not str(Path(self.output_root)).startswith('/data/shangguanrz/'):raise ValueError('New training artifacts belong under /data/shangguanrz/')

    @classmethod
    def from_dict(cls,raw):
        raw=dict(raw);stage=dict(raw.get('stage',{}))
        if stage:
            stage['routes']=tuple(construct(RouteSpec,{**r,'source_ids':tuple(r.get('source_ids',())),
                'loss':construct(LossSpec,r.get('loss',{}))}) for r in stage.get('routes',()))
            stage['optimizer']=construct(OptimizerSpec,{**stage.get('optimizer',{}),'betas':tuple(stage.get('optimizer',{}).get('betas',(.9,.999)))})
            for k in ('include_groups','exclude_groups'):stage[k]=tuple(stage.get(k,()))
            raw['stage']=construct(StageSpec,stage)
        raw['sources']=tuple(construct(SourceSpec,s) for s in raw.get('sources',()))
        return construct(cls,raw)


def stage_preset(phase,branch=None,*,budget=None):
    if phase=='S0':routes=(RouteSpec('body','body',1.),)
    elif phase in ('S1','S2'):
        selected=(branch,) if branch else ('face','hand','detail','interaction')
        if any(b not in ('face','hand','detail','interaction') for b in selected):raise ValueError('Explicit specialty branch required')
        routes=tuple(RouteSpec(b,b,1.,loss=LossSpec(face=.5 if b in ('face','detail') else 0.,hand=.5 if b in ('hand','detail') else 0.)) for b in selected)
    elif phase=='S3':routes=(RouteSpec('joint','joint',1.),)
    else:raise ValueError('Unknown stage')
    return StageSpec(stage_id=phase+'_'+(branch or ('body' if phase=='S0' else 'joint' if phase=='S3' else 'heads')),
        phase=phase,routes=routes,budget_updates=budget,optimizer=OptimizerSpec(multiplier=.1 if phase=='S3' else 1.))
