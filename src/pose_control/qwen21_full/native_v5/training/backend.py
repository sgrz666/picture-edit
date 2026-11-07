"""Lazy native Qwen execution adapter. No model/cache/optimizer on import."""
from pathlib import Path
from dataclasses import asdict
import json
import torch
from .io import verify,read_json_ref,file_sha,write_json
from .config import digest
from .parameters import ParameterCatalog
from ..runtime import NativeFullRuntimeV5
from ..integration import install_native_full_v5,bind_native_full_v5
from ..data import collate_native_full_v5
from ...native_full_data import identity_from_reference_v4
from ...qwen_runtime import transformer_inputs,dataset_image_to_pil
from ..losses import frozen_evaluators

def code_identity():
    root=Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob('*.py'))}

def model_identity(config):
    model,bound=read_json_ref(config.model_manifest)
    if not isinstance(model,list) or not model:raise ValueError('Qwen manifest must list config/index/weight shard file references')
    model_files=[verify(r) for r in model]
    model_root=Path(config.model_path).resolve()
    if any(model_root not in Path(r['path']).resolve().parents for r in model_files):raise ValueError('Qwen manifest files must belong to the declared model root')
    required={p.resolve() for p in model_root.rglob('*') if p.is_file() and p.suffix in ('.safetensors','.bin','.json','.model','.txt','.tiktoken') and p.resolve()!=Path(bound['path']).resolve()}
    if not any(p.suffix in ('.safetensors','.bin') for p in required) or not required<={Path(r['path']).resolve() for r in model_files}:raise ValueError('Qwen manifest omits native pipeline weights/config/tokenizer assets')
    return bound,model_files

def asset_identity(config,data_plan):
    bound,model_files=model_identity(config)
    cache,cache_ref=read_json_ref(config.cache_identity)
    if config.qwen_condition_profile=='legacy_single':
        if cache.get('schema')!='native_v5_frozen_cache_index_v1' or cache.get('prompt_sha256')!=digest(config.prompt):
            raise ValueError('Frozen legacy cache prompt identity differs')
    else:
        from ..conditioning import CACHE_SCHEMA
        conditions=getattr(data_plan,'condition_bindings',None)
        if conditions is None:conditions={s+'\0'+data_plan.datasets[s].rows[i]['pair_id']:ref['sha256'] for (s,i),(ref,_) in data_plan.conditions.items()}
        if (cache.get('schema')!=CACHE_SCHEMA or cache.get('profile')!=config.qwen_condition_profile or cache.get('condition_bindings')!=conditions):
            raise ValueError('Frozen multi-reference cache condition/Prompt identity differs')
    if cache.get('model_manifest_sha256')!=bound['sha256']:raise ValueError('Frozen cache model identity differs')
    evaluators={}
    required={name for r in config.stage.routes for name in ('identity','perceptual') if getattr(r.loss,name)>0}
    for name in required:
        if not config.evaluators or name not in config.evaluators:raise ValueError('Missing requested frozen evaluator: '+name)
        evaluators[name]=verify(config.evaluators[name])
    return {'data':data_plan.identity,'model_manifest':bound,'model_files':model_files,'cache_identity':cache_ref,
        'evaluators':evaluators,'code':code_identity()},cache

def load_training_pipeline(config):
    from diffusers import QwenImage21Pipeline
    return QwenImage21Pipeline.from_pretrained(config.model_path,text_encoder=None,vae=None,
        device_map='cuda',dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True)

def load_cache_pipeline(config):
    from diffusers import QwenImage21Pipeline
    return QwenImage21Pipeline.from_pretrained(config.model_path,transformer=None,
        device_map='cuda',dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True)

class NativeBackend:
    def __init__(self,config,data_plan,*,saved=None,purpose='train'):
        self.config=config;self.data=data_plan;self.device=torch.device('cuda')
        self.identity,self.cache=asset_identity(config,data_plan)
        self.initialization=verify(config.initialization)
        self.identity['initialization']=self.initialization
        saved=saved if saved is not None else torch.load(self.initialization['path'],map_location='cpu',mmap=True,weights_only=True)
        self.runtime=NativeFullRuntimeV5.from_checkpoint(saved)
        if self.runtime.config.verification_only and purpose!='grad':raise ValueError('Verification-only checkpoint cannot start training')
        if (self.runtime.config.min_spatial_cells,self.runtime.config.feather_px)!=(config.eligibility_min_spatial_cells,config.eligibility_feather_px):
            raise ValueError('Data eligibility rules differ from the checkpoint local configuration')
        from .data import route_options
        needs_local=any((o.face and o.face_strength>0) or (o.hand and o.hand_strength>0)
                        for r in config.stage.routes for o,_ in (route_options(r,'single'),route_options(r,'dual')))
        for dataset in data_plan.datasets.values() if needs_local else ():
            for i in range(len(dataset)):
                sample=dataset[i];c=sample['local_condition_v4']
                from .data import route_options
                for route in config.stage.routes:
                    options,_=route_options(route,sample['native_scene_kind'])
                    local=(options.face and options.face_strength>0) or (options.hand and options.hand_strength>0)
                    if local and c.mesh_valid.any() and self.runtime.bps_convention is None:
                        raise ValueError('Bind a verified BPS convention during explicit model initialization before local training/gradient acceptance')
                    if options.face and options.strength('face_geometry')>0:
                        for p in range(2):
                            meta=c.metadata[0];used=meta['expression_mode']=='driver' or (meta['expression_mode']=='source' and meta.get('expression_retargeted',[False,False])[p])
                            if used and c.expression_valid[0,p] and not (self.runtime.face_parameter_basis or {}).get('expression_basis_sha256'):
                                raise ValueError('Active Face expression requires an explicitly bound encoder basis')
        self.runtime.to(self.device)
        self.catalog=ParameterCatalog(self.runtime);self.active=self.catalog.apply(config.stage)
        self.pipe=load_training_pipeline(config)
        self.pipe.transformer.requires_grad_(False).eval().to(self.device)
        self.lazy_frozen_components=True;self.frozen_component_digests={}
        self.runtime.train()
        self.base_state_names=set(self.pipe.transformer.state_dict())
        self.installation=install_native_full_v5(self.pipe.transformer,self.runtime)
        self.pipe.transformer.enable_gradient_checkpointing();self.catalog.assert_flags(self.active)
        self.base_parameters={n:p for n,p in self.pipe.transformer.named_parameters() if not n.startswith(('native_body.','native_v5_'))}
        self.evaluators=[None,None]
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
        self.identity['native_qwen_implementation']=file_sha(__import__('inspect').getfile(QwenImage21Transformer2DModel))
        self.on_demand=None
        if self.cache.get('loading_mode')=='on_demand':
            from .streaming import OnDemandCache
            state=saved.get('training_state',{}).get('frozen_cache_state')
            if 'training_state' in saved and state is None:raise ValueError('Streaming checkpoint lacks used-cache state')
            self.on_demand=OnDemandCache(config,data_plan,self.cache,state)

    def ensure_cache_pairs(self,pairs):
        if getattr(self,'on_demand',None) is not None:self.on_demand.ensure(pairs)

    def cache_state(self):return self.on_demand.state() if getattr(self,'on_demand',None) is not None else None

    def ensure_evaluators(self,required):
        if not required:return tuple(self.evaluators)
        refs=[self.config.evaluators.get(k) if k in required else None for k in ('identity','perceptual')]
        missing={k for i,k in enumerate(('identity','perceptual')) if k in required and self.evaluators[i] is None}
        if missing:
            refs=[self.config.evaluators.get(k) if k in missing else None for k in ('identity','perceptual')]
            models=frozen_evaluators(*(verify(r)['path'] if r else None for r in refs),device=self.device,
                sha=[verify(r)['sha256'] if r else None for r in refs])
            for i,m in enumerate(models):
                if m is not None:self.evaluators[i]=m
        self.ensure_vae().to(self.device)
        return tuple(self.evaluators)

    def ensure_vae(self):
        if self.pipe.vae is None:
            from diffusers import AutoencoderKLQwenImage21
            from .validation import tensor_digest
            vae=AutoencoderKLQwenImage21.from_pretrained(self.config.model_path,subfolder='vae',
                torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True)
            vae.requires_grad_(False).eval()
            self.pipe.vae=vae
            self.frozen_component_digests['vae']=tensor_digest(vae.state_dict().items())
        return self.pipe.vae

    def pair(self,source,index):
        sample=self.data.sample(source,index);key=source+'\0'+sample['pair_metadata']['pair_id']
        entry=self.on_demand.read(source,index) if getattr(self,'on_demand',None) is not None else self.cache.get('entries',{}).get(key)
        if entry is None:raise ValueError('Missing frozen latent cache; run explicit cache preparation first: '+key)
        ref=verify(entry['file']);p=Path(ref['path']);root=Path(self.config.cache_root).resolve()
        if root not in p.resolve().parents:raise ValueError('Cache entry escapes declared cache root')
        c=sample['local_condition_v4'].metadata[0]
        if any(entry.get(k)!=c.get(k) for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')):raise ValueError('Frozen latent cache belongs to another sample')
        pair=torch.load(p,map_location='cpu',weights_only=True)
        if tuple(pair['source_token_hw'])!=sample['source_token_hw'] or tuple(pair['target_token_hw'])!=sample['target_token_hw']:raise ValueError('Cached token grid differs')
        for name in ('source_latent','target_latent'):
            if pair[name].shape[-1]!=64:raise ValueError('Frozen cache must retain native Qwen64 latents')
        pair={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in pair.items()}
        if self.config.qwen_condition_profile!='legacy_single':
            from ..conditioning import validate_pair
            validate_pair(pair)
            ref,spec=self.data.conditions[(source,index)]
            if entry.get('condition_sha256')!=ref['sha256'] or pair.get('prompt_sha256')!=spec['prompt']['sha256']:
                raise ValueError('Frozen pair uses another prompt or reference order')
        spatial=sample.get('spatial_weights_v5')
        batch=collate_native_full_v5([{k:v for k,v in sample.items() if k!='spatial_weights_v5'}])
        if spatial is not None:
            batch['spatial_weights_v5']={**spatial,'target_token_hw':tuple(pair['target_token_hw'])}
        return sample,batch,pair

    def forward(self,batch,pair,route,sigma,noise,options,base):
        from torch.nn import functional as F
        identity=identity_from_reference_v4(batch,pair['source_latent'],pair['source_token_hw']) if base.identity and base.identity_strength else None
        coverage=F.avg_pool2d(batch['valid_image_mask'].to(self.device).float(),16).flatten(1)
        target=pair['target_latent'];noisy=(1-sigma[:,None,None])*target.float()+sigma[:,None,None]*noise.float()
        prepared=self.runtime.encode(batch,target_token_hw=tuple(pair['target_token_hw']),identity=identity,
            source_latent=pair['source_latent'],options=options,base_options=base)
        with bind_native_full_v5(self.pipe.transformer,prepared,options=options,valid_mask=coverage):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=self.pipe.transformer(**transformer_inputs(pair,noisy.to(target.dtype),sigma))[0]
        return output[:,-target.shape[1]:],noise.float()-target.float(),coverage,noisy

def encode_frozen_pair(pipe,sample,config,spec=None):
    if spec is not None:
        from ..conditioning import encode_pair
        return encode_pair(pipe,sample,spec)
    from ..conditioning import encode_legacy_source
    pair=encode_legacy_source(pipe,sample,config.prompt)
    image=dataset_image_to_pil(sample['tgt_image']);device=next(pipe.vae.parameters()).device
    pixels=pipe.image_processor.preprocess(image.convert('RGBA'),width=image.width,height=image.height)
    pixels=pixels.unsqueeze(2).to(device=device,dtype=pipe.vae.dtype)
    pair['target_latent']=pipe._encode_vae_image(pixels,torch.Generator(device=device).manual_seed(0)).flatten(3).squeeze(2).transpose(1,2)
    return pair

def prepare_cache(config,data_plan,output,*,resume=False,mode='eager'):
    """Explicit future frozen preprocessing, separate from training and preflight."""
    from .cache_store import FrozenCacheStore
    from .frozen_inputs import progress
    root=Path(output).resolve()
    if not str(root).startswith('/data/shangguanrz/'):raise ValueError('New cache must use persistent data volume')
    manifest,_=model_identity(config)
    identity={'model_manifest_sha256':manifest['sha256'],'profile':config.qwen_condition_profile,
        'prompt_sha256':digest(config.prompt),'sources':data_plan.identity['sources']}
    store=FrozenCacheStore(root,identity,resume=resume)
    if mode=='on_demand':
        from ..conditioning import CACHE_SCHEMA
        if not hasattr(data_plan,'qualification'):raise ValueError('On-demand encoding requires full data qualification')
        value={'schema':CACHE_SCHEMA,'model_manifest_sha256':manifest['sha256'],'profile':config.qwen_condition_profile,
               'condition_bindings':data_plan.condition_bindings,'entries':{},'loading_mode':'on_demand',
               'entry_preparation':verify({'path':str(root/'preparation.json')}),'qualification':data_plan.qualification,
               'pair_count':sum(len(dataset) for dataset in data_plan.datasets.values())}
        path=root/'on_demand_index.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf8'))!=value:raise ValueError('On-demand descriptor changed')
            return verify({'path':str(path)})
        progress('on_demand_cache_ready',pairs=value['pair_count'],all_pairs_preencoded=False)
        return write_json(path,value)
    if mode!='eager':raise ValueError('Unknown frozen encoding mode')
    # Audit every committed pair before loading any GPU component or adding files.
    pending=[]
    entries={};conditions={}
    total=sum(len(dataset) for dataset in data_plan.datasets.values());scanned=0
    for source,dataset in data_plan.datasets.items():
        for i in range(len(dataset)):
            ref,spec=data_plan.conditions[(source,i)] if config.qwen_condition_profile!='legacy_single' else (None,None)
            if hasattr(data_plan,'binding'):
                binding=data_plan.binding(source,i,manifest['sha256'],digest(config.prompt));key=binding['key']
            else:
                sample=dataset[i];key=source+'\0'+sample['pair_metadata']['pair_id'];c=sample['local_condition_v4'].metadata[0]
                binding={'key':key,**{k:c[k] for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')},
                    'source_token_hw':list(sample['source_token_hw']),'target_token_hw':list(sample['target_token_hw']),
                    'model_manifest_sha256':manifest['sha256'],'condition_sha256':ref['sha256'] if ref else None,
                    'prompt_sha256':spec['prompt']['sha256'] if spec else digest(config.prompt)}
            if ref:conditions[key]=ref['sha256']
            entry=store.get(binding)
            if entry is None:pending.append((source,i,binding,spec))
            else:entries[key]=entry
            scanned+=1
            if scanned==1 or scanned%250==0 or scanned==total:progress('frozen_cache_scan',completed=scanned,total=total,reused=len(entries))
    if (root/'index.json').exists() and pending:raise ValueError('Published frozen index is missing pair commits')
    if pending:
        progress('loading_frozen_text_encoder_vae',pending=len(pending),total=total)
        pipe=load_cache_pipeline(config)
        pipe.text_encoder.requires_grad_(False).eval();pipe.vae.requires_grad_(False).eval()
        with torch.no_grad():
            for source,i,binding,spec in pending:
                pair=encode_frozen_pair(pipe,data_plan.datasets[source][i],config,spec)
                entries[binding['key']]=store.put(binding,pair)
                done=len(entries)
                if done==1 or done%20==0 or done==total:progress('frozen_cache_encoded',completed=done,total=total)
    if config.qwen_condition_profile!='legacy_single':
        from ..conditioning import CACHE_SCHEMA
        return store.publish({'schema':CACHE_SCHEMA,'model_manifest_sha256':manifest['sha256'],
            'profile':config.qwen_condition_profile,'condition_bindings':conditions,'entries':entries})
    return store.publish({'schema':'native_v5_frozen_cache_index_v1','model_manifest_sha256':manifest['sha256'],
        'prompt_sha256':digest(config.prompt),'entries':entries})
