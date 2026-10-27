"""Fixed-seed generation, with training RNG and module modes restored."""
from pathlib import Path
from .validation import preserved_rng
from .data import route_options
from .config import RouteSpec
from .io import file_sha


def generation_probe(backend,config,data,update,output,*,baseline=False,seeds=None,ablations=False):
    import torch
    from dataclasses import replace
    from ..generation import generate_native
    records=[];output=Path(output);was_training=backend.runtime.training
    seeds=(config.generation_seed,) if seeds is None else tuple(seeds)
    with preserved_rng(),torch.no_grad():
        try:
            backend.runtime.eval()
            for source,dataset in data.datasets.items():
                for index in range(len(dataset)):
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
                            if path.exists():raise FileExistsError('Never overwrite a generation: '+str(path))
                            im=generate_native(backend.pipe,backend.runtime,batch,pair,options=controls,
                                base_options=base_controls,seed=seed,steps=config.generation_steps,enabled=enabled)
                            with path.open('xb') as f:im.save(f,format='PNG')
                            records.append({'source_id':source,'pair_id':sample['pair_metadata']['pair_id'],
                                'variant':label,'seed':seed,'steps':config.generation_steps,'path':str(path),
                                'sha256':file_sha(path),'size':im.size,'target_rgb_used_as_condition':False})
        finally:backend.runtime.train(was_training)
    return {'kind':'fixed_seed_generation','update':update,'records':records,'training_rng_preserved':True}
