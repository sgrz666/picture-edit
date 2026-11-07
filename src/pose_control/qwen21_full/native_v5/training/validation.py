"""Explicit no-optimizer acceptance; static/data reports can never pass this gate."""
from contextlib import contextmanager
import hashlib
import torch
from .config import digest,canonical
from .io import read_json_ref
from .checkpointing import capture_rng,restore_rng,environment
from .data import route_options

def tensor_digest(items):
    h=hashlib.sha256()
    for name,tensor in sorted(items,key=lambda x:x[0]):
        h.update(canonical([name,list(tensor.shape),str(tensor.dtype)]).encode())
        raw=tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        for offset in range(0,raw.numel(),16*1024*1024):h.update(raw[offset:offset+16*1024*1024].cpu().numpy().tobytes())
    return h.hexdigest()

def frozen_digest(backend):
    native=[(n,p) for n,p in backend.runtime.named_parameters() if n not in backend.active]
    native += [('buffer:'+n,b) for n,b in backend.runtime.named_buffers()]
    state=backend.pipe.transformer.state_dict()
    base=[(n,state[n]) for n in backend.base_state_names if torch.is_tensor(state[n])]
    auxiliary=[]
    for scope,model in (('vae',backend.pipe.vae),('text_encoder',backend.pipe.text_encoder),
        *[('evaluator_'+str(i),m) for i,m in enumerate(backend.evaluators) if m is not None]):
        if model is None:continue
        if scope in ('vae','text_encoder') and getattr(backend,'lazy_frozen_components',False):
            expected=getattr(backend,'frozen_component_digests',{}).get(scope)
            if expected is not None and tensor_digest(model.state_dict().items())!=expected:raise RuntimeError('Frozen lazy component changed: '+scope)
            continue
        auxiliary.extend((scope+':'+n,t) for n,t in model.state_dict().items() if torch.is_tensor(t))
    return {'native':tensor_digest(native),'qwen':tensor_digest(base),'frozen_auxiliary':tensor_digest(auxiliary),'architecture':digest(backend.runtime.get_extra_state())}

def _gradient_metrics_to_cpu(packed):
    """One scalar-result transfer per device, without changing gradient buffers."""
    return packed.detach().cpu().tolist()


def audit_gradients(backend,*,require_any=True):
    backend.catalog.assert_flags(backend.active)
    unexpected=[n for n,p in backend.catalog.parameters.items() if n not in backend.active and p.grad is not None]
    unexpected += ['qwen:'+n for n,p in backend.base_parameters.items() if p.requires_grad or p.grad is not None]
    for scope,model in (('vae',backend.pipe.vae),('text_encoder',backend.pipe.text_encoder),
        *[('evaluator_'+str(i),m) for i,m in enumerate(backend.evaluators) if m is not None]):
        if model is None:continue
        unexpected.extend(scope+':'+n for n,p in model.named_parameters() if p.requires_grad or p.grad is not None)
    if unexpected:raise RuntimeError('Frozen parameter received gradient: '+str(unexpected[:8]))
    groups={};metrics={}
    for r in backend.catalog.rows:
        p=backend.catalog.parameters[r.name]
        # Preserve the old parameter-order error precedence. Earlier nonfinite
        # gradients still fail before a later illegal parameter dtype.
        if p.dtype!=torch.float32:break
        if p.grad is None:continue
        g=p.grad.detach()
        value=torch.stack((torch.isfinite(g).all().to(torch.float32),
                           g.count_nonzero().ne(0).to(torch.float32),g.norm()))
        groups.setdefault(g.device,[]).append((r.name,value))
    for values in groups.values():
        packed=_gradient_metrics_to_cpu(torch.stack([value for _,value in values]))
        metrics.update((name,value) for (name,_),value in zip(values,packed))
    rows={};nonzero=False
    for r in backend.catalog.rows:
        p=backend.catalog.parameters[r.name];g=p.grad
        if p.dtype!=torch.float32:raise ValueError('Adapter parameters must be FP32')
        if g is not None and not metrics[r.name][0]:raise FloatingPointError('Nonfinite gradient: '+r.name)
        status='not_called_or_no_gradient' if g is None else 'zero' if not metrics[r.name][1] else 'nonzero'
        if status=='nonzero':nonzero=True
        rows[r.name]={'status':status,'norm':None if g is None else metrics[r.name][2],'active':r.name in backend.active,'group':r.runtime_group,'role':r.role}
    if require_any and not nonzero:raise RuntimeError('No active finite nonzero gradient')
    return rows

def assert_optimizer_fp32(optimizer):
    for state in optimizer.state.values():
        for k in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if k in state and state[k].dtype!=torch.float32:raise ValueError('AdamW moments must remain FP32')


def check_warmed_heads(catalog,active,data):
    """Require warmup for usable training routes, not dormant single/dual heads."""
    required=set()
    for row in data.records:
        if row.get('role')!='train' or not row.get('eligible'):continue
        branches=row['branch_usable']
        if branches.get('face'):required.add('face_heads')
        if branches.get('hand'):required.add('hand_heads')
        if branches.get('interaction'):required.add('interaction_heads')
        if branches.get('detail'):
            detail=row['effective_regions']['detail']
            if any(detail[:2]):required.add('face_heads')
            if any(detail[2:]):required.add('hand_heads')
    relevant=required & {r.runtime_group for r in catalog.rows if r.name in active and r.role=='head'}
    for group in sorted(relevant):
        if not any(catalog.parameters[r.name].detach().count_nonzero() for r in catalog.rows
                if r.runtime_group==group and r.role=='head' and r.name in active):
            raise ValueError('Selected branch Head has not been warmed: '+group)
    return sorted(relevant)

@contextmanager
def preserved_rng():
    state=capture_rng()
    try:yield
    finally:restore_rng(state)

@contextmanager
def head_probe(backend):
    heads=[(r.name,backend.catalog.parameters[r.name]) for r in backend.catalog.rows if r.role=='head' and r.name in backend.active]
    saved={n:p.detach().cpu().clone() for n,p in heads};generator=torch.Generator(device=backend.device).manual_seed(7829)
    try:
        with torch.no_grad():
            for _,p in heads:p.copy_(torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=generator)*.001)
        yield
    finally:
        with torch.no_grad():
            for n,p in heads:p.copy_(saved[n].to(p.device))

def validate_gate(report_ref,backend,config,data,*,stage_origin=None):
    report,bound=read_json_ref(report_ref)
    expected=stage_origin or backend.initialization
    checks={'schema':'native_v5_gradient_preflight_v1','level':'grad','passed':True,'production_passed':True,
        'initialization_sha256':expected['sha256'],'config_sha256':config.signature(),
        'data_signature':data.signature,'identity_sha256':digest(backend.identity),'catalog_sha256':backend.catalog.signature()}
    checks['environment_sha256']=digest(environment())
    if any(report.get(k)!=v for k,v in checks.items()):raise ValueError('Missing/mismatched real gradient acceptance gate')
    return bound

def check_body_baseline(config,initialization):
    if config.stage.phase=='S0':return None
    if not config.body_baseline_report:raise ValueError('Starting beyond S0 requires a bound Body baseline report and manual acceptance')
    report,bound=read_json_ref(config.body_baseline_report)
    accepted=report.get('manual_acknowledgment') is True
    if report.get('acceptance_kind')=='technical_training_gate_v1':
        accepted=False
        if config.continuous_stage_authorization is not None and report.get('continuous_stage_authorization')==config.continuous_stage_authorization:
            rebuilt=build_automatic_body_gate(report['s0_checkpoint'],report['s0_gradient_report'],config.continuous_stage_authorization)
            accepted=rebuilt==report
    if not accepted and config.continuous_stage_authorization is not None:
        if (report.get('continuous_stage_authorization')==config.continuous_stage_authorization and
                report.get('acceptance_kind')=='technical_pilot_gate'):
            gradient,_=read_json_ref(report['s0_gradient_report'])
            accepted=gradient.get('level')=='grad' and gradient.get('passed') is True and gradient.get('production_passed') is True
    if report.get('body_accepted') is not True or not accepted or report.get('checkpoint_sha256')!=initialization['sha256']:
        raise ValueError('Body baseline was not accepted for this initialization')
    return bound

def build_automatic_body_gate(checkpoint_ref,gradient_report_ref,authorization):
    """Bind completed production S0 evidence; this never invents a human acknowledgment."""
    import math
    from types import SimpleNamespace
    from .config import TrainingConfig
    from .checkpointing import load_committed
    from .parameters import ParameterCatalog
    if (not isinstance(authorization,dict) or authorization.get('source')!='direct_user_message' or
            not isinstance(authorization.get('text'),str) or not authorization['text'].strip()):
        raise ValueError('Explicit continuous stage authorization required')
    saved,checkpoint,_=load_committed(checkpoint_ref);state=saved['training_state']
    config=TrainingConfig.from_dict(state['config'])
    if (config.stage.phase!='S0' or config.stage.budget_updates!=10000 or state['stage_update']!=10000 or
            state['global_update']<state['stage_update'] or config.continuous_stage_authorization!=authorization):
        raise ValueError('Automatic Body gate requires completed authorized S0 budget 10000')
    if (config.signature()!=state['config_sha256'] or digest(state['identity'])!=state['identity_sha256'] or
            digest(state['catalog'])!=state['catalog_sha256']):raise ValueError('Checkpoint config/data/catalog binding differs')
    expected=ParameterCatalog.active_names(SimpleNamespace(rows=[SimpleNamespace(**r) for r in state['catalog']]),config.stage)
    if state['active_names']!=sorted(expected):raise ValueError('Checkpoint frozen mask differs')
    gradient,gradient_ref=read_json_ref(gradient_report_ref)
    from .io import verify
    if verify(state['gradient_report'])!=gradient_ref:raise ValueError('Checkpoint gradient report binding differs')
    checks={'schema':'native_v5_gradient_preflight_v1','level':'grad','passed':True,'production_passed':True,
        'config_sha256':config.signature(),'data_signature':digest(state['identity']['data']),
        'identity_sha256':state['identity_sha256'],'catalog_sha256':state['catalog_sha256'],
        'initialization_sha256':state['stage_origin']['sha256'],'optimizer_created':False,'optimizer_step_executed':False}
    if any(gradient.get(k)!=v for k,v in checks.items()) or gradient.get('verification_only') is not False:
        raise ValueError('Missing or mismatched production gradient evidence')
    frozen=state.get('frozen_state')
    if not frozen or gradient.get('frozen_before')!=frozen or gradient.get('frozen_after')!=frozen:
        raise ValueError('Frozen production state changed')
    validation=state.get('validation_state',{})
    if (validation.get('enabled') is not True or validation.get('passed') is not True or
            validation.get('finite') is not True or validation.get('triggered') is not False or
            validation.get('last_probe_update')!=state['global_update'] or
            validation.get('loss_limit_ratio')!=config.validation_loss_limit_ratio or
            validation.get('failure_patience')!=config.validation_failure_patience or
            validation.get('consecutive_failures',config.validation_failure_patience)>=config.validation_failure_patience or
            any(type(validation.get(k)) not in (int,float) or not math.isfinite(validation[k]) for k in ('baseline_loss','last_loss'))):
        raise ValueError('Missing passed finite final validation state')
    return {'schema':'native_v5_body_acceptance_v1','acceptance_kind':'technical_training_gate_v1','body_accepted':True,
        'checkpoint_sha256':checkpoint['sha256'],'s0_checkpoint':checkpoint,'s0_gradient_report':gradient_ref,
        'continuous_stage_authorization':authorization,'completed_stage_updates':state['stage_update'],
        'config_sha256':state['config_sha256'],'identity_sha256':state['identity_sha256'],
        'data_signature':gradient['data_signature'],'frozen_state':frozen,'validation_state':validation}

@contextmanager
def prefix_trace(backend,pair,noisy,sigma):
    from ...native_integration import _prepare_execution
    from ...qwen_runtime import transformer_inputs
    values=transformer_inputs(pair,noisy.to(pair['target_latent'].dtype),sigma)
    target=_prepare_execution(backend.pipe.transformer,values,None,False,None,1.).target_mask
    observed={};handles=[]
    for index,block in enumerate(backend.pipe.transformer.transformer_blocks):
        def hook(module,args,result,index=index):
            if result.shape[1]!=target.numel():raise ValueError('Block output does not match joint text/reference/target layout')
            # Detach BEFORE indexing. Indexing a grad-tracked block output
            # saves its mask inside a non-reentrant checkpoint; the temporary
            # observation hook is absent during recomputation.
            observed[str(index)]=tensor_digest([('text_and_reference_prefix',result.detach()[:,~target])])
        handles.append(block.register_forward_hook(hook))
    try:yield observed,values
    finally:
        for handle in handles:handle.remove()

def gradient_preflight(backend,config,data):
    from .objective import compute_micro_loss,counts_for_window
    routes={r.name:r for r in config.stage.routes};records=[];passed=True
    with preserved_rng():
        backend.ensure_evaluators({k for r in config.stage.routes for k in ('identity','perceptual') if getattr(r.loss,k)>0})
        before=frozen_digest(backend)
        cases=[]
        for route in config.stage.routes:
            representatives={}
            for (name,source),pool in data.pools.items():
                if name!=route.name:continue
                for i in pool:
                    status=data.status[(name,source,i)]
                    key=(source,status['bucket'][2],status['bucket'][3],tuple(k for k,v in status['branch_usable'].items() if v),
                        digest(status['capabilities']),bool(status['auxiliary_faces']))
                    representatives.setdefault(key,(source,i))
            if not representatives or len(representatives)>(config.gradient_case_budget or config.probe_samples):
                passed=False;records.append({'route':route.name,'status':'not_exercised_or_budget_too_small','required_cases':len(representatives)});continue
            cases.extend((route,source,i) for source,i in representatives.values())
        for route,source,index in cases:
            sample,batch,pair=backend.pair(source,index)
            options,base=route_options(route,sample['native_scene_kind'])
            sigma=torch.tensor([.2 if route.loss.identity or route.loss.perceptual else .5],device=backend.device)
            generator=torch.Generator(device=backend.device).manual_seed(24042)
            noise=torch.randn(pair['target_latent'].shape,device=backend.device,dtype=pair['target_latent'].dtype,generator=generator)
            noisy=(1-sigma[:,None,None])*pair['target_latent'].float()+sigma[:,None,None]*noise.float()
            with torch.no_grad(),prefix_trace(backend,pair,noisy,sigma) as (baseline_prefix,values):
                with torch.autocast('cuda',dtype=torch.bfloat16):backend.pipe.transformer(**values,native_enabled=False)
            entry={'route':route.name,'sigma':float(sigma[0]),'status':data.status[(route.name,source,index)]}
            counts=counts_for_window([entry],routes,route.loss.auxiliary_every)
            required={k for k in ('identity','perceptual') if counts[k]}
            evaluators=backend.ensure_evaluators(required)
            def backward_once():
                backend.runtime.zero_grad(set_to_none=True)
                with prefix_trace(backend,pair,noisy,sigma) as (controlled_prefix,_):
                    output,target,coverage,actual_noisy=backend.forward(batch,pair,route,sigma,noise,options,base)
                if baseline_prefix!=controlled_prefix:raise RuntimeError('Native control changed a text/reference prefix')
                supervision=batch['training_supervision_v5'].to(backend.device)
                loss,_=compute_micro_loss(backend.pipe,output,target,coverage,supervision,actual_noisy,sigma,
                    (batch['tgt_image'].to(backend.device).float()+1)*.5,route.loss,counts,route.loss.auxiliary_every,evaluators,
                    spatial=batch.get('spatial_weights_v5'))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preflight objective')
                loss.backward();return audit_gradients(backend)
            first=backward_once()
            active_head=[v for n,v in first.items() if v['active'] and v['role']=='head' and v['status']=='nonzero']
            if not active_head:passed=False
            with head_probe(backend):probe=backward_once()
            status=entry['status'];regions=status['effective_regions'];expected=set()
            if route.branch in ('body','joint'):expected.add('native_body')
            if route.branch in ('body','joint'):
                expected|={'body21','geometry_shared'}
                if sample['native_scene_kind']=='dual':expected.add('geometry_dual')
                typed=sample.get('dual_depth',sample.get('depth_condition_v2'))
                depth_valid=typed.scene_valid.any() if hasattr(typed,'scene_valid') else typed.valid.any() if typed is not None else False
                if base.depth_mode!='off' and base.depth_strength and depth_valid:
                    expected|={'champ_cnn','champ_attention','depth_fusion'}
                    if base.detail:expected.add('detail_cnn')
            if route.branch in ('face','joint') and any(regions['face']):expected.add('face_controller')
            if route.branch in ('hand','joint') and any(regions['hand']):expected.add('hand_controller')
            if route.branch in ('face','joint') and any(regions['face']):expected.add('face_encoder')
            if route.branch in ('hand','joint') and any(regions['hand']):expected.add('hand_encoder')
            if route.branch in ('detail','joint') and any(regions['detail']):
                expected.add('detail_bridge')
                if any(regions['detail'][:2]):expected.add('face_controller')
                if any(regions['detail'][2:]):expected.add('hand_controller')
            if route.branch in ('interaction','joint') and status['branch_usable']['interaction']:
                expected|={'interaction_controller','geometry_dual'}
                depth=sample['dual_depth']
                if base.raster and depth.camera_available.any():expected.add('relation_raster')
                if base.part_contact and (depth.part_distance_valid.any() or depth.contacts.valid.any() or depth.part_contact[...,1].any()):expected.add('relation_tokens')
            active_groups={r.runtime_group for r in backend.catalog.rows if r.name in backend.active}
            expected &= active_groups
            observed={v['group'] for v in probe.values() if v['status']=='nonzero'}
            missing=sorted(expected-observed)
            if missing:passed=False
            backend.runtime.zero_grad(set_to_none=True)
            records.append({'route':route.name,'source_id':source,'index':index,'first':first,'probe':probe,
                'prefix_unchanged':True,'prefix_block_digests':baseline_prefix,
                'expected_upstream_groups':sorted(expected),'missing_upstream_groups':missing,
                'head_gradient_observed':bool(active_head),'upstream_probe_groups':sorted({v['group'] for v in probe.values() if v['role'] not in ('head','strength') and v['status']=='nonzero'})})
        after=frozen_digest(backend)
        if before!=after:raise RuntimeError('Frozen weights/buffers or architecture changed during validation')
    return {'schema':'native_v5_gradient_preflight_v1','level':'grad','passed':passed,'production_passed':passed and not backend.runtime.config.verification_only,
        'initialization_sha256':backend.initialization['sha256'],'config_sha256':config.signature(),'data_signature':data.signature,
        'identity_sha256':digest(backend.identity),'catalog_sha256':backend.catalog.signature(),'records':records,
        'environment_sha256':digest(environment()),
        'frozen_before':before,'frozen_after':after,'optimizer_created':False,'optimizer_step_executed':False,
        'verification_only':backend.runtime.config.verification_only}
