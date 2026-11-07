"""Stage/data/parameters/objectives are independent. Nothing runs on construction."""
from pathlib import Path
from dataclasses import asdict
from .config import TrainingConfig,SCHEMA,digest
from .io import write_json,new_segment,Journal,verify


class NativeV5Trainer:
    def __init__(self,config):
        self.config=config if isinstance(config,TrainingConfig) else TrainingConfig.from_dict(config)

    def prepare(self,output=None):
        result={'schema':SCHEMA,'action':'prepare','config':self.config.record(),'config_sha256':self.config.signature(),
            'missing_bindings':self.config.missing(),'assets_checked':False,'training_ready':False,
            'model_constructed':False,'optimizer_created':False,'training_started':False,
            'policy':{'manual_stage_transitions':True,'stable_optimizer_superset':True,'checkpoint_boundary_only':True}}
        if output is not None:write_json(output,result)
        return result

    def preflight_data(self,output):
        from .streaming import training_data_plan
        try:result=training_data_plan(self.config).report()
        except Exception as error:
            write_json(output,{'level':'data','passed':False,'error':str(error),'sources':[asdict(s) for s in self.config.sources],
                'original_manifests_preserved':True,'model_constructed':False,'gradient_passed':False})
            raise
        write_json(output,result);return result

    def preflight_grad(self,output):
        from .streaming import training_data_plan
        from .backend import NativeBackend
        from .validation import gradient_preflight
        self.config.require_ready();data=training_data_plan(self.config);backend=NativeBackend(self.config,data,purpose='grad')
        result=gradient_preflight(backend,self.config,data);write_json(output,result);return result

    def cache_prepare(self,output,*,resume=False,qualification=None,mode='eager'):
        from .data import DataPlan
        from .backend import prepare_cache
        if qualification is not None:
            from .frozen_inputs import FrozenInputPlan
            data=FrozenInputPlan(self.config,qualification,lazy=mode=='on_demand')
        else:data=DataPlan(self.config)
        return prepare_cache(self.config,data,output,resume=resume,mode=mode)

    def _start(self,mode,checkpoint=None):
        import random
        import numpy as np
        import torch
        from .streaming import training_data_plan
        from .backend import NativeBackend
        from .sampling import DeterministicSampler,derived_seed
        from .parameters import UpdateScheduler
        from .checkpointing import load_committed,validate_resume,restore_training_randomness
        from .validation import validate_gate,check_body_baseline,check_warmed_heads,frozen_digest,assert_optimizer_fp32
        config=self.config;config.require_ready();random.seed(config.seed);np.random.seed(config.seed%(2**32));torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
        if mode!='resume' and not config.gradient_report:raise ValueError('Explicit matching gradient report required before model/optimizer construction')
        if mode=='train' and config.stage.phase not in ('S0','S1'):raise ValueError('S2/S3 start through explicit advance-stage')
        saved=parent=tail=None
        if mode!='train':
            saved,parent,tail=load_committed(checkpoint)
            if mode=='advance-stage' and verify(config.initialization)!=parent:raise ValueError('Advance initialization must be the exact parent checkpoint')
        data=training_data_plan(config);backend=NativeBackend(config,data,saved=saved,purpose='train')
        catalog=backend.catalog;active=backend.active
        if backend.runtime.config.local_mode=='static' and any(r.role=='head' and r.runtime_group in ('face_heads','hand_heads') and r.name in active for r in catalog.rows):
            body_frozen=all(not p.requires_grad for p in backend.runtime.body.parameters())
            if body_frozen and not any(h.weight.detach().count_nonzero() for h in backend.runtime.body.output_heads):raise ValueError('Static local Head cannot train through a frozen zero Body output')
        if mode=='train':
            if config.stage.phase not in ('S0','S1'):raise ValueError('S2/S3 start through explicit advance-stage, not a fresh reset presented as continuation')
            check_body_baseline(config,backend.initialization)
            gradient=validate_gate(config.gradient_report,backend,config,data)
            origin=backend.initialization;global_update=stage_update=attempts=0
        elif mode=='resume':
            state=validate_resume(saved,config,backend.identity,catalog)
            origin=state['stage_origin'];gradient=validate_gate(state['gradient_report'],backend,config,data,stage_origin=origin)
            global_update=state['global_update'];stage_update=state['stage_update'];attempts=state['attempts']
        else:
            state=saved['training_state'];previous=TrainingConfig.from_dict(state['config'])
            allowed={'S0':('S0','S1'),'S1':('S1','S2'),'S2':('S2','S3'),'S3':('S3',)}
            if config.stage.phase not in allowed[previous.stage.phase]:raise ValueError('Invalid manual stage transition')
            if config.stage.phase=='S1' and previous.stage.phase=='S0':check_body_baseline(config,parent)
            if state['catalog_sha256']!=catalog.signature():raise ValueError('Stage advance cannot change the Adapter parameter catalog')
            if state['stage_update']<1:raise ValueError('Parent stage has no successful updates')
            # The matching new-stage gradient report is the activation gate.
            # Its explicit command is the human stage decision; no auto advance.
            gradient=validate_gate(config.gradient_report,backend,config,data)
            if config.stage.phase in ('S2','S3'):
                check_warmed_heads(catalog,active,data)
            origin=parent;global_update=state['global_update'];stage_update=attempts=0
        optimizer=catalog.build_optimizer(config.stage.optimizer)
        if mode!='train':
            expected=[g['names'] for g in optimizer.param_groups]
            if [g['names'] for g in saved['training_state']['optimizer_state']['param_groups']]!=expected:raise ValueError('Optimizer superset named membership differs')
            optimizer.load_state_dict(saved['training_state']['optimizer_state'])
            catalog.set_stage_lrs(optimizer,config.stage.optimizer)
        scheduler=UpdateScheduler(optimizer,config.stage.optimizer,config.stage.budget_updates)
        sampler=DeterministicSampler(config,data.pools)
        noise_generator=torch.Generator(device=backend.device).manual_seed(derived_seed(config.seed,config.stage.stage_id+':noise'))
        if mode=='resume':
            scheduler.load_state_dict(saved['training_state']['scheduler_state']);sampler.load_state_dict(saved['training_state']['sampler_state'])
        assert_optimizer_fp32(optimizer);optimizer.zero_grad(set_to_none=True);catalog.assert_flags(active)
        backend.ensure_evaluators({k for r in config.stage.routes for k in ('identity','perceptual') if getattr(r.loss,k)>0})
        segment=new_segment(config.output_root,mode,parent)
        write_json(segment/'resolved.json',{'config':config.record(),'identity':backend.identity,'catalog':catalog.record(),
            'active_names':sorted(active),'optimizer_groups':catalog.group_spec(config.stage.optimizer),'gradient_gate':gradient,
            'stage_origin':origin,'parent_checkpoint':parent,'old_log_tail':tail,'environment_policy':'strict_resume; frozen_Qwen_eval; B1; workers0'})
        if mode=='advance-stage':write_json(segment/'stage_transition.json',{'parent':parent,'old_stage':saved['training_state']['config']['stage'],
            'new_stage':asdict(config.stage),'optimizer_state_retained':True,'new_state_policy':'AdamW lazily initializes first gradient',
            'manual_command':config.continuous_stage_authorization is None,
            'continuous_stage_authorization':config.continuous_stage_authorization,'new_gradient_report':gradient})
        journal=Journal(segment/'metrics.jsonl')
        if mode!='train':restore_training_randomness(saved['training_state'],noise_generator)
        frozen=frozen_digest(backend)
        if mode=='resume' and saved['training_state'].get('frozen_state')!=frozen:raise ValueError('Resumed frozen digest differs')
        return dict(backend=backend,data=data,catalog=catalog,active=active,optimizer=optimizer,scheduler=scheduler,sampler=sampler,
            noise_generator=noise_generator,segment=segment,journal=journal,gradient=gradient,origin=origin,parent=parent,
            global_update=global_update,stage_update=stage_update,attempts=attempts,frozen=frozen,
            validation_state=saved['training_state'].get('validation_state') if mode=='resume' else None)

    def run(self,*,mode='train',checkpoint=None,stop_after_update=None):
        if mode not in ('train','resume','advance-stage'):raise ValueError('Explicit execution mode required')
        import torch
        import time
        from .data import route_options
        from .objective import counts_for_window,compute_micro_loss
        from .validation import audit_gradients,assert_optimizer_fp32,frozen_digest
        from .checkpointing import save_snapshot
        from .evaluation import ValidationGuard,fixed_evaluation_selection
        config=self.config;ctx=self._start(mode,checkpoint);backend=ctx['backend'];runtime=backend.runtime
        if stop_after_update is not None and (type(stop_after_update)!=int or stop_after_update<=ctx['global_update']):raise ValueError('Stop boundary must be beyond restored global update')
        routes={r.name:r for r in config.stage.routes};budget=config.stage.budget_updates
        max_attempts=config.stage.max_attempts or budget*config.stage.accumulation*10
        micro_position=0;last_snapshot={'checkpoint':ctx['parent']} if mode=='resume' else None
        has_loss=config.validation_loss_limit_ratio is not None or config.probe_every_updates>0
        if mode=='resume' and config.validation_loss_limit_ratio is not None and not ctx.get('validation_state'):
            raise ValueError('Enabled guard requires the committed validation state on resume')
        selection=fixed_evaluation_selection(config,ctx['data']) if has_loss or config.generation_every_updates else []
        guard=ValidationGuard(config,selection=selection,state=ctx.get('validation_state'))
        def probe_loss():
            from .probes import loss_probe
            report=loss_probe(backend,config,ctx['data'],ctx['global_update'])
            if guard.state['enabled']:guard.observe(report)
            else:guard.state['last_probe_update']=ctx['global_update']
            ctx['journal'].append(report)
        def generate():
            from .observations import generation_probe
            ctx['journal'].append(generation_probe(backend,config,ctx['data'],ctx['global_update'],Path(config.output_root)/'output/images'))
            guard.state['last_generation_update']=ctx['global_update']
        def failure(error):
            write_json(ctx['segment']/'failure.json',{'error':str(error),'micro_position':micro_position,
                'last_committed_snapshot':last_snapshot,'partial_window_not_checkpointed':micro_position!=0,
                'validation_state':guard.state})
        def save():
            if frozen_digest(backend)!=ctx['frozen']:raise RuntimeError('Frozen state changed')
            return save_snapshot(Path(config.output_root)/'checkpoints',runtime,ctx['optimizer'],ctx['scheduler'],ctx['sampler'],ctx['noise_generator'],
                config=config,identity=backend.identity,catalog=ctx['catalog'],active=ctx['active'],global_update=ctx['global_update'],
                stage_update=ctx['stage_update'],attempts=ctx['attempts'],journal=ctx['journal'],micro_position=micro_position,
                gradient_report=ctx['gradient'],stage_origin=ctx['origin'],parent=ctx['parent'],frozen_state=ctx['frozen'],validation_state=guard.state,
                frozen_cache_state=backend.cache_state() if hasattr(backend,'cache_state') else None)
        if guard.state['enabled'] and guard.state['baseline_loss'] is None:
            try:probe_loss()
            except Exception as error:failure(error);raise
        while ctx['stage_update']<budget and ctx['attempts']<max_attempts and (stop_after_update is None or ctx['global_update']<stop_after_update):
            update_started=time.perf_counter()
            window=[]
            while len(window)<config.stage.accumulation and ctx['attempts']<max_attempts:
                route_name,source,index=ctx['sampler'].draw();route=routes[route_name];ctx['attempts']+=1
                status=ctx['data'].status[(route_name,source,index)]
                # Data preflight already verified this scene kind. Decode the full
                # sample once, in backend.pair, rather than twice per micro-batch.
                options,base=route_options(route,status['bucket'][2])
                sigma=torch.rand(1,device=backend.device,generator=ctx['noise_generator'])
                branch_names={'body':('body',),'face':('face_identity','face_texture','face_geometry'),
                    'hand':('hand_geometry','hand_texture'),'detail':('detail',),'interaction':('interaction',),
                    'joint':('body','interaction','detail','face_identity','face_texture','face_geometry','hand_geometry','hand_texture')}[route.branch]
                enabled=any(runtime.strength(b,l,sigma,options).any() for b in branch_names for l in runtime.config.injection_layers)
                if not enabled:
                    ctx['journal'].append({'attempt':ctx['attempts'],'optimizer_updated':False,'reason':'all_route_schedules_zero',
                        'route':route_name,'source_id':source,'index':index,'sigma':float(sigma[0])});continue
                window.append({'route':route_name,'source':source,'index':index,'sigma':float(sigma[0]),'status':status})
            if len(window)!=config.stage.accumulation:
                ctx['journal'].append({'optimizer_updated':False,'reason':'partial_window_discarded_at_attempt_bound','entries':window});break
            counts=counts_for_window(window,routes,ctx['global_update']+1);ctx['optimizer'].zero_grad(set_to_none=True)
            if hasattr(backend,'ensure_cache_pairs'):backend.ensure_cache_pairs([(entry['source'],entry['index']) for entry in window])
            values=[]
            try:
                for i,entry in enumerate(window):
                    micro_position=i+1;route=routes[entry['route']]
                    sample,batch,pair=backend.pair(entry['source'],entry['index'])
                    if sample['native_scene_kind']!=entry['status']['bucket'][2]:raise ValueError('Sample scene kind differs from verified preflight')
                    options,base=route_options(route,sample['native_scene_kind'])
                    sigma=torch.tensor([entry['sigma']],device=backend.device)
                    noise=torch.randn(pair['target_latent'].shape,device=backend.device,dtype=pair['target_latent'].dtype,generator=ctx['noise_generator'])
                    required={k for k in ('identity','perceptual') if getattr(route.loss,k)>0 and counts[k]}
                    evaluators=backend.ensure_evaluators(required)
                    prediction,target,coverage,noisy=backend.forward(batch,pair,route,sigma,noise,options,base)
                    supervision=batch['training_supervision_v5'].to(backend.device)
                    loss,parts=compute_micro_loss(backend.pipe,prediction,target,coverage,supervision,noisy,sigma,
                        (batch['tgt_image'].to(backend.device).float()+1)*.5,route.loss,counts,ctx['global_update']+1,evaluators,
                        spatial=batch.get('spatial_weights_v5'))
                    if not torch.isfinite(loss) or not loss.requires_grad:raise FloatingPointError('Invalid/no-gradient objective for an eligible route')
                    loss.backward();values.append({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items()})
                gradients=audit_gradients(backend);active_params=[ctx['catalog'].parameters[n] for n in sorted(ctx['active'])]
                norm=torch.nn.utils.clip_grad_norm_(active_params,config.stage.optimizer.clip_norm)
                if not torch.isfinite(norm) or norm==0:raise FloatingPointError('Invalid accumulated gradient')
                ctx['optimizer'].step();ctx['scheduler'].step();assert_optimizer_fp32(ctx['optimizer'])
                ctx['global_update']+=1;ctx['stage_update']+=1;micro_position=0
                ctx['journal'].append({'global_update':ctx['global_update'],'stage_update':ctx['stage_update'],'attempts':ctx['attempts'],
                    'optimizer_updated':True,'update_seconds':time.perf_counter()-update_started,'window':window,'losses':values,'gradient_norm':float(norm),'gradients':gradients,
                    'learning_rates':{g['name']:g['lr'] for g in ctx['optimizer'].param_groups}})
                finished=ctx['stage_update']==budget
                due_loss=(config.probe_every_updates and ctx['stage_update']%config.probe_every_updates==0) or (has_loss and finished)
                # Enabled guard also validates every save, before it can be committed.
                if guard.state['enabled'] and ctx['stage_update']%config.save_every_updates==0:due_loss=True
                if due_loss and guard.state['last_probe_update']!=ctx['global_update']:probe_loss()
                if config.generation_every_updates and (ctx['stage_update']%config.generation_every_updates==0 or finished):
                    if guard.state['last_generation_update']!=ctx['global_update']:generate()
                if ctx['stage_update']%config.save_every_updates==0:last_snapshot=save()
            except Exception as error:
                failure(error)
                raise
        ctx['optimizer'].zero_grad(set_to_none=True);micro_position=0
        try:
            if has_loss and guard.state['last_probe_update']!=ctx['global_update']:probe_loss()
            if config.generation_every_updates and ctx['stage_update']==budget and guard.state['last_generation_update']!=ctx['global_update']:generate()
            last_snapshot=save()
        except Exception as error:failure(error);raise
        write_json(ctx['segment']/'summary.json',{'requested_updates':budget,'completed_stage_updates':ctx['stage_update'],
            'global_updates':ctx['global_update'],'attempts':ctx['attempts'],'checkpoint':last_snapshot,'automatic_stage_advance':False,
            'budget_completed':ctx['stage_update']==budget,'requested_stop_boundary':stop_after_update,'validation_state':guard.state})
        if ctx['stage_update']!=budget and (stop_after_update is None or ctx['global_update']<stop_after_update):raise RuntimeError('Attempt bound reached before requested updates; partial immutable checkpoint retained')
        return last_snapshot
