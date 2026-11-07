"""Append-only frozen inputs on demand; the optimizer's sample order never changes."""
from copy import deepcopy
import json
from pathlib import Path
import torch
from .config import digest
from .data import DataPlan,route_options
from .frozen_inputs import FrozenInputPlan,progress
from .io import read_json_ref,verify
from .cache_store import FrozenCacheStore
from ..data import NativeFullDatasetV5


def body_status(joint):
    result=deepcopy(joint)
    result['cells']=[0]*6
    result['effective_regions']={'face':[False]*2,'hand':[False]*4,'detail':[False]*6}
    result['branch_usable']={'body':joint['branch_usable']['body'],'face':False,'hand':False,
                             'detail':False,'interaction':False,'joint':joint['branch_usable']['body']}
    return result


class QualifiedBodyPlan(DataPlan):
    """Project the already-read single-person joint qualification to body-only S0."""
    def __init__(self,config,qualification):
        if config.stage.phase!='S0' or len(config.stage.routes)!=1 or config.stage.routes[0].branch!='body':
            raise ValueError('Qualified body plan is S0/body only')
        options,base=route_options(config.stage.routes[0],'single')
        if (options.face or options.hand or options.detail or options.interaction or options.body_strength!=1 or
                not base.identity or base.identity_strength!=1 or base.depth_mode!='full_geometry'):
            raise ValueError('Nonstandard S0 controls require full data qualification')
        self.config=config;self.frozen=FrozenInputPlan(config,qualification,lazy=True)
        self.datasets={};self.group_indices={};self.status={};self.records=[];self.pools={};self.bindings={};self.spatial={}
        self.conditions=self.frozen.conditions;self.condition_bindings=self.frozen.condition_bindings
        self.identity=self.frozen.identity;self.signature=digest(self.identity);self.spatial_indices={}
        for source in config.sources:
            self.datasets[source.source_id]=NativeFullDatasetV5(source.manifest['path'],load_target=True,data_profile=source.data_profile)
            groups,_=read_json_ref(source.groups);self.group_indices[source.source_id]=groups
            self.spatial_indices[source.source_id]=self.identity['sources'][source.source_id].get('spatial_weights',{}).get('entries',{})
        route=config.stage.routes[0]
        for row in self.frozen.qualified_report['rows']:
            if row['route']!='joint':continue
            if row['bucket'][2]!='single' or not row['branch_usable']['body']:
                raise ValueError('Qualification is not the standard single-person body superset')
            status=body_status({k:v for k,v in row.items() if k not in ('source_id','role','index','pair_id','route')})
            if row['role']=='train' and route.source_ids and row['source_id'] not in route.source_ids:
                status.update(eligible=False,reasons=status['reasons']+['source_not_selected_for_route'])
            value={**row,**status,'route':route.name};self.records.append(value)
            self.status[(route.name,row['source_id'],row['index'])]=status
        if len(self.records)!=sum(len(dataset) for dataset in self.datasets.values()):raise ValueError('Qualified S0 coverage differs')
        for source in config.sources:
            if source.role!='train' or route.source_ids and source.source_id not in route.source_ids:continue
            pool=[i for i in range(len(self.datasets[source.source_id])) if self.status[(route.name,source.source_id,i)]['eligible']]
            if not pool:raise ValueError('Empty qualified S0 training source')
            self.pools[(route.name,source.source_id)]=pool
        if not any(row['role']=='validation' and row['eligible'] for row in self.records):raise ValueError('No qualified S0 validation')
        progress('qualified_s0_plan_ready',pairs=len(self.records),full_native_decode=False)

    def target_rgb_sha(self,source,index):return self.frozen.ensure_metadata(source,index)['meta']['driver_rgb_sha256']

    def sample(self,source,index):
        sample=self.datasets[source][index]
        if self.spatial_indices[source]:
            meta=sample['local_condition_v4'].metadata[0];key=meta['driver_rgb_sha256']+':'+meta['target_canvas_id']
            if key not in self.spatial_indices[source]:raise ValueError('Spatial index misses actual sample/Canvas')
            self.spatial[(source,index)]=self.spatial_indices[source][key]
            self._attach_spatial(source,index,sample)
        return sample


def training_data_plan(config):
    if config.cache_identity is not None and config.stage.phase=='S0':
        index,_=read_json_ref(config.cache_identity)
        if index.get('loading_mode')=='on_demand':return QualifiedBodyPlan(config,index['qualification'])
    return DataPlan(config)


class OnDemandCache:
    def __init__(self,config,data,descriptor,state=None):
        self.config=config;self.descriptor=descriptor;self.pipe=None
        self.frozen=getattr(data,'frozen',None) or FrozenInputPlan(config,descriptor['qualification'],lazy=True)
        expected={'model_manifest_sha256':descriptor['model_manifest_sha256'],'profile':config.qwen_condition_profile,
                  'prompt_sha256':digest(config.prompt),'sources':data.identity['sources']}
        preparation,ref=read_json_ref(descriptor['entry_preparation'])
        if (Path(ref['path']).resolve()!=Path(config.cache_root).resolve()/'preparation.json' or
                preparation['identity']!=expected or preparation['identity_sha256']!=digest(expected)):
            raise ValueError('On-demand cache preparation binding differs')
        self.store=FrozenCacheStore(config.cache_root,expected,resume=True);self.used={}
        self.indices={source:{row['pair_id']:i for i,row in enumerate(dataset.rows)} for source,dataset in self.frozen.datasets.items()}
        if state is not None:
            if state.get('schema')!='native_on_demand_used_cache_v1':raise ValueError('Missing on-demand cache checkpoint state')
            for key,expected_ref in state['used'].items():
                source,pair_id=key.split('\0');index=self.indices[source][pair_id]
                binding=self.binding(source,index);entry=self.store.get(binding)
                if entry is None or entry['file']!=expected_ref:raise ValueError('Previously used frozen cache missing/changed: '+key)
                self.used[key]=expected_ref

    def binding(self,source,index):
        binding=self.frozen.binding(source,index,self.descriptor['model_manifest_sha256'],digest(self.config.prompt))
        ref,spec=self.frozen.conditions[(source,index)]
        for item in (ref,spec['prompt']['bank'],*[x['file'] for x in spec['images']]):self.frozen.sha.verify(item)
        return binding

    def ensure(self,pairs):
        pending=[];seen=set()
        for source,index in pairs:
            binding=self.binding(source,index);key=binding['key']
            if key in seen:continue
            seen.add(key)
            if not self.store._record(binding).exists():
                if key in self.used:raise ValueError('Missing frozen cache used before checkpoint: '+key)
                pending.append((source,index,binding))
        if not pending:return
        from .backend import load_cache_pipeline,encode_frozen_pair
        from .validation import preserved_rng
        progress('on_demand_encoding',requested=len(pending))
        with preserved_rng(),torch.no_grad():
            try:
                if self.pipe is None:
                    self.pipe=load_cache_pipeline(self.config)
                    self.pipe.text_encoder.requires_grad_(False).eval();self.pipe.vae.requires_grad_(False).eval()
                else:
                    self.pipe.text_encoder.to('cuda');self.pipe.vae.to('cuda')
                versions=[(p,p._version) for model in (self.pipe.text_encoder,self.pipe.vae) for p in model.parameters()]
                for source,index,binding in pending:
                    spec=self.frozen.conditions[(source,index)][1]
                    pair=encode_frozen_pair(self.pipe,self.frozen.datasets[source][index],self.config,spec)
                    self.store.put(binding,pair)
                    progress('on_demand_pair_committed',source=source,pair_id=self.frozen.datasets[source].rows[index]['pair_id'])
                if any(p.requires_grad or p.grad is not None or p._version!=v for p,v in versions):
                    raise RuntimeError('Frozen cache encoder weights changed')
            finally:
                if self.pipe is not None:
                    self.pipe.text_encoder.to('cpu');self.pipe.vae.to('cpu');torch.cuda.empty_cache()

    def read(self,source,index):
        binding=self.binding(source,index);self.ensure([(source,index)])
        entry=self.store.get(binding)
        if entry is None:raise ValueError('Requested cache was not committed')
        previous=self.used.get(binding['key'])
        if previous is not None and previous!=entry['file']:raise ValueError('Used frozen cache SHA changed')
        self.used[binding['key']]=entry['file'];return entry

    def state(self):return {'schema':'native_on_demand_used_cache_v1','used':dict(self.used)}
