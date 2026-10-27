"""Optional held-out loss probes, independent seeds and preserved training RNG."""
import torch
from .validation import preserved_rng
from .data import route_options
from .objective import compute_micro_loss,counts_for_window

def loss_probe(backend,config,data,update):
    records=[];was_training=backend.runtime.training
    with preserved_rng(),torch.no_grad():
        try:
            backend.runtime.eval()
            for route_index,route in enumerate(config.stage.routes):
                role='observation' if config.experiment_mode=='fixed_pair_overfit' else 'validation'
                selected=[r for r in data.records if r['role']==role and r['route']==route.name and r['eligible']][:config.probe_samples]
                if not selected:
                    if role=='observation':
                        records.append({'route':route.name,'status':'observation_route_unavailable'});continue
                    raise ValueError('Missing held-out probe examples: '+route.name)
                for i,row in enumerate(selected):
                    source,index=row['source_id'],row['index'];sample,batch,pair=backend.pair(source,index)
                    options,base=route_options(route,sample['native_scene_kind'])
                    sigma=torch.tensor([(.2,.5,.8)[i%3]],device=backend.device)
                    generator=torch.Generator(device=backend.device).manual_seed(24042+100*route_index+i)
                    noise=torch.randn(pair['target_latent'].shape,device=backend.device,dtype=pair['target_latent'].dtype,generator=generator)
                    entry={'route':route.name,'sigma':float(sigma[0]),'status':row}
                    counts=counts_for_window([entry],{route.name:route},update)
                    required={k for k in ('identity','perceptual') if counts[k]}
                    evaluators=backend.ensure_evaluators(required)
                    output,target,coverage,noisy=backend.forward(batch,pair,route,sigma,noise,options,base)
                    loss,parts=compute_micro_loss(backend.pipe,output,target,coverage,batch['training_supervision_v5'].to(backend.device),
                        noisy,sigma,(batch['tgt_image'].to(backend.device).float()+1)*.5,route.loss,counts,update,evaluators)
                    records.append({'route':route.name,'source_id':source,'index':index,'sigma':float(sigma[0]),'objective':float(loss),
                        'parts':{k:float(v) if torch.is_tensor(v) else v for k,v in parts.items()}})
        finally:backend.runtime.train(was_training)
    return {'kind':'same_scene_observation_loss' if config.experiment_mode=='fixed_pair_overfit' else 'held_out_loss_probe',
        'update':update,'records':records,'training_rng_preserved':True,'generation_executed':False}
