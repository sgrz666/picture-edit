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
    if cache.get('schema')!='native_v5_frozen_cache_index_v1' or cache.get('model_manifest_sha256')!=bound['sha256'] or cache.get('prompt_sha256')!=digest(config.prompt):
        raise ValueError('Frozen cache model/prompt identity differs')
    evaluators={}
    required={name for r in config.stage.routes for name in ('identity','perceptual') if getattr(r.loss,name)>0}
    for name in required:
        if not config.evaluators or name not in config.evaluators:raise ValueError('Missing requested frozen evaluator: '+name)
        evaluators[name]=verify(config.evaluators[name])
    return {'data':data_plan.identity,'model_manifest':bound,'model_files':model_files,'cache_identity':cache_ref,
        'evaluators':evaluators,'code':code_identity()},cache

class NativeBackend:
    def __init__(self,config,data_plan,*,saved=None,purpose='train'):
        from diffusers import QwenImage21Pipeline
        self.config=config;self.data=data_plan;self.device=torch.device('cuda')
        self.identity,self.cache=asset_identity(config,data_plan)
        self.initialization=verify(config.initialization)
        self.identity['initialization']=self.initialization
        saved=saved if saved is not None else torch.load(self.initialization['path'],map_location='cpu',mmap=True,weights_only=True)
        self.runtime=NativeFullRuntimeV5.from_checkpoint(saved)
        if self.runtime.config.verification_only and purpose!='grad':raise ValueError('Verification-only checkpoint cannot start training')
        if (self.runtime.config.min_spatial_cells,self.runtime.config.feather_px)!=(config.eligibility_min_spatial_cells,config.eligibility_feather_px):
            raise ValueError('Data eligibility rules differ from the checkpoint local configuration')
        for dataset in data_plan.datasets.values():
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
        self.pipe=QwenImage21Pipeline.from_pretrained(config.model_path,dtype=torch.bfloat16,local_files_only=True)
        self.pipe.transformer.requires_grad_(False).eval().to(self.device)
        self.pipe.text_encoder.requires_grad_(False).eval().to('cpu');self.pipe.vae.requires_grad_(False).eval().to('cpu')
        self.runtime.train()
        self.base_state_names=set(self.pipe.transformer.state_dict())
        self.installation=install_native_full_v5(self.pipe.transformer,self.runtime)
        self.pipe.transformer.enable_gradient_checkpointing();self.catalog.assert_flags(self.active)
        self.base_parameters={n:p for n,p in self.pipe.transformer.named_parameters() if not n.startswith(('native_body.','native_v5_'))}
        self.evaluators=[None,None]
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
        self.identity['native_qwen_implementation']=file_sha(__import__('inspect').getfile(QwenImage21Transformer2DModel))

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
        self.pipe.vae.to(self.device)
        return tuple(self.evaluators)

    def pair(self,source,index):
        sample=self.data.sample(source,index);key=source+'\0'+sample['pair_metadata']['pair_id']
        entry=self.cache.get('entries',{}).get(key)
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
        return sample,collate_native_full_v5([sample]),pair

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

def prepare_cache(config,data_plan,output):
    """Explicit future frozen preprocessing, separate from training and preflight."""
    from diffusers import QwenImage21Pipeline
    from ...qwen_runtime import Qwen21IPERCache
    root=Path(output).resolve()
    if not str(root).startswith('/data/shangguanrz/'):raise ValueError('New cache must use persistent data volume')
    root.mkdir(parents=True,exist_ok=False)
    manifest,_=model_identity(config)
    pipe=QwenImage21Pipeline.from_pretrained(config.model_path,dtype=torch.bfloat16,local_files_only=True)
    pipe.text_encoder.requires_grad_(False).eval().to('cuda');pipe.vae.requires_grad_(False).eval().to('cuda')
    cache=Qwen21IPERCache(root/'frozen',resolution='native',prompt=config.prompt,canvas_policy='aspect_preserving');entries={}
    with torch.no_grad():
        for source,dataset in data_plan.datasets.items():
            for i in range(len(dataset)):
                sample=dataset[i];batch=collate_native_full_v5([sample]);pair=cache.pair(pipe,batch,torch.device('cuda'))
                path=root/('pair_'+digest([source,sample['pair_metadata']['pair_id']])[:24]+'.pt')
                with path.open('xb') as f:torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in pair.items()},f)
                c=sample['local_condition_v4'].metadata[0]
                entries[source+'\0'+sample['pair_metadata']['pair_id']]={'file':{'path':str(path),'sha256':file_sha(path)},
                    **{k:c[k] for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')}}
    return write_json(root/'index.json',{'schema':'native_v5_frozen_cache_index_v1','model_manifest_sha256':manifest['sha256'],
        'prompt_sha256':digest(config.prompt),'entries':entries})
