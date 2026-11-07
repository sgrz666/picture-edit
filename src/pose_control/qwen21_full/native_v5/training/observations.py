"""Fixed-seed generation, with training RNG and module modes restored."""
from pathlib import Path
from .validation import preserved_rng,tensor_digest
from .data import route_options
from .config import RouteSpec,digest
from .io import file_sha,write_json,verify
from .evaluation import fixed_evaluation_selection


def generation_probe(backend,config,data,update,output,*,baseline=False,seeds=None,ablations=False):
    import torch
    from dataclasses import replace,asdict
    import json,uuid
    from ..generation import generate_native
    records=[];output=Path(output);was_training=backend.runtime.training
    seeds=(config.generation_seed,) if seeds is None else tuple(seeds)
    selection=fixed_evaluation_selection(config,data)
    runtime_sha=tensor_digest((n,t) for n,t in backend.runtime.state_dict().items() if torch.is_tensor(t))
    with preserved_rng(),torch.no_grad():
        try:
            backend.ensure_vae()
            backend.runtime.eval()
            for selected in selection:
                source=selected['source_id']
                for index in (selected['index'],):
                    sample,batch,pair=backend.pair(source,index)
                    phase_body=config.stage.phase=='S0' and not baseline
                    route=RouteSpec('generation','body' if phase_body else 'joint',1.)
                    options,base=route_options(route,sample['native_scene_kind'])
                    variants=[('native',options,base,True)]
                    if baseline:variants.insert(0,('qwen_only',options,base,False))
                    if ablations:
                        variants.extend((('depth_off',options,replace(base,depth_strength=0.),True),
                            ('local_off',replace(options,face=False,hand=False,detail=False),base,True)))
                        if sample['native_scene_kind']=='dual':
                            variants.append(('interaction_off',replace(options,interaction=False),replace(base,interaction=False,raster=False,part_contact=False),True))
                    for seed in seeds:
                        for label,controls,base_controls,enabled in variants:
                            folder=output/f'update_{update:06d}'/source/sample['pair_metadata']['pair_id']
                            folder.mkdir(parents=True,exist_ok=True)
                            path=folder/f'{label}_seed{seed}.png'
                            commit=folder/f'{label}_seed{seed}.commit.json'
                            binding={'config_sha256':config.signature(),'data_signature':getattr(data,'signature',digest(selection)),
                                'runtime_sha256':runtime_sha,'update':update,'source_id':source,'pair_id':sample['pair_metadata']['pair_id'],
                                'variant':label,'seed':seed,'steps':config.generation_steps,'controls':asdict(controls),
                                'base_controls':asdict(base_controls),'enabled':enabled}
                            if commit.exists():
                                saved=json.loads(commit.read_text(encoding='utf8'))
                                if saved.get('schema')!='native_v5_generation_commit_v1' or saved.get('binding')!=binding:
                                    raise ValueError('Committed generation binding differs: '+str(commit))
                                record=saved['record']
                                if Path(record['path']).resolve().parent!=folder.resolve():raise ValueError('Committed generation escapes output folder')
                                verify({'path':record['path'],'sha256':record['sha256']})
                                records.append(record);continue
                            # A crash before its commit can leave an orphan image. Preserve
                            # it and give the replacement a distinct name.
                            if path.exists():path=folder/f'{label}_seed{seed}_{uuid.uuid4().hex}.png'
                            im=generate_native(backend.pipe,backend.runtime,batch,pair,options=controls,
                                base_options=base_controls,seed=seed,steps=config.generation_steps,enabled=enabled)
                            with path.open('xb') as f:im.save(f,format='PNG')
                            record={'source_id':source,'pair_id':sample['pair_metadata']['pair_id'],
                                'variant':label,'seed':seed,'steps':config.generation_steps,'path':str(path),
                                'sha256':file_sha(path),'size':list(im.size),'target_rgb_used_as_condition':False}
                            write_json(commit,{'schema':'native_v5_generation_commit_v1','binding':binding,'record':record})
                            records.append(record)
        finally:backend.runtime.train(was_training)
    return {'kind':'fixed_seed_generation','update':update,'records':records,'selection':selection,'training_rng_preserved':True}
