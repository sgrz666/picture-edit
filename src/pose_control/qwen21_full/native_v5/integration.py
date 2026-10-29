"""One V5 installation; all streams read the same post-block target Hidden."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,replace,asdict
from functools import partial
import inspect,weakref
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from ..native_integration import _prepare_execution
from ..native_body import _IdentitySafeAdd
from .contracts import PreparedNativeFullControlV5,NativeControlOptionsV5
from .observation import observe_residual


@dataclass(frozen=True)
class Binding:
    control: PreparedNativeFullControlV5
    options: NativeControlOptionsV5
    valid_mask: torch.Tensor | None
    conv_input_gate: bool


@dataclass(frozen=True)
class Execution:
    body: object
    prepared: PreparedNativeFullControlV5 | None
    options: NativeControlOptionsV5
    static_geometry: tuple=()


def body_delta(body,query,geometry,layer,sigma):
    """Same Native Body algebra, with its Query also reused by Interaction."""
    if layer not in body.config.injection_layers: return query.new_zeros(*query.shape[:2],body.config.hidden_dim)
    g=geometry.tokens;time=body.time_projection(sigma.float()[:,None])[:,None]
    fused=query+g+query*body.geometry_modulation(g).tanh()
    fused=body.fusion(fused+body.layer_embedding.weight[layer]+time)
    return body.output_heads[6*layer//body.config.num_layers](fused)


class BlockMixin:
    def forward(self,*args,native_v5_execution=None,**kwargs):
        original=super().forward(*args,**kwargs);owner=self._native_v5_owner()
        if owner is None: raise RuntimeError('V5 transformer owner missing')
        execution=native_v5_execution or owner._native_v5_execution.get()
        if execution is None or not execution.body.enabled: return original
        e=execution.body;p=execution.prepared;options=execution.options;layer=self._native_v5_layer
        config=owner._native_v5_config;strength=owner.native_v5_strength
        with torch.autocast(device_type=original.device.type,enabled=False):
            hidden=original[:,e.target_mask].float()
            query=owner.native_body.query_projection(owner.native_body.query_norm(hidden))
            geometry=execution.static_geometry[6*layer//config.num_layers] if execution.static_geometry else e.geometry
            body_gain=strength('body',layer,e.sigma,options)
            delta=body_delta(owner.native_body,query,geometry,layer,e.sigma)*body_gain[:,None,None] if body_gain.any() else hidden.new_zeros(hidden.shape)
            if e.valid_mask is not None: delta=delta*e.valid_mask[...,None]
            observe_residual('body',delta,layer=layer,sigma=e.sigma,gain=body_gain)
            if layer in config.injection_layers:
                gain=strength('interaction',layer,e.sigma,options)
                if gain.any() and p.relation_valid.any():
                    update=owner.native_v5_interaction(query,p,layer=layer,sigma=e.sigma)*gain[:,None,None]
                    if e.valid_mask is not None: update=update*e.valid_mask[...,None]
                    observe_residual('interaction',update,layer=layer,sigma=e.sigma,gain=gain)
                    delta=delta+update
                else:
                    observe_residual('interaction',None,layer=layer,sigma=e.sigma,gain=gain,condition_valid=p.relation_valid.any())
                if config.local_mode=='dynamic' and p.local is not None:
                    for family,regions in (('face',(0,1)),('hand',(2,3,4,5))):
                        if not getattr(options,family) or getattr(options,family+'_strength')==0:
                            observe_residual(family,None,layer=layer,sigma=e.sigma,condition_valid=False)
                            continue
                        update=getattr(owner,'native_v5_'+family)(hidden,p.local,layer=layer,sigma=e.sigma,
                            target_hw=p.body.target_token_hw,regions=regions,strength=strength,options=options)
                        if e.valid_mask is not None: update=update*(e.valid_mask!=0)[...,None]
                        observe_residual(family,update,layer=layer,sigma=e.sigma)
                        delta=delta+update
        result=original.clone();result[:,e.target_mask]=_IdentitySafeAdd.apply(original[:,e.target_mask],delta.to(original.dtype))
        return result


class TransformerMixin:
    def forward(self,*args,native_full_control=None,native_full_options=None,native_valid_mask=None,
                native_conv_input_gate=True,native_enabled=True,**kwargs):
        if self._native_v5_execution.get() is not None: raise RuntimeError('Reentrant V5 forward unsupported')
        if 'native_geometry' in kwargs or getattr(self,'_pic_edit_qwen21_control_active',False): raise ValueError('Do not stack V5 and another control installer')
        binding=self._native_v5_binding.get()
        if binding is not None:
            if native_full_control is not None: raise ValueError('Duplicate V5 binding')
            native_full_control=binding.control;native_full_options=binding.options
            native_valid_mask=binding.valid_mask;native_conv_input_gate=binding.conv_input_gate
        options=native_full_options or NativeControlOptionsV5();options.validate()
        if native_enabled:
            if not isinstance(native_full_control,PreparedNativeFullControlV5): raise ValueError('Prepared V5 conditions required')
            native_full_control.validate()
            if any(m['options']!=asdict(options) for m in native_full_control.metadata): raise ValueError('Prepared options differ; re-encode when enabled modalities change')
        bound=self._native_v5_signature.bind(self,*args,**kwargs);bound.apply_defaults()
        features=None if native_full_control is None else native_full_control.body
        body=_prepare_execution(self,bound.arguments,features,native_enabled,native_valid_mask,1.,native_conv_input_gate)
        static=[];config=self._native_v5_config
        if native_enabled and config.local_mode=='static' and native_full_control.local is not None:
            for layer in config.injection_layers:
                value=features.geometry
                for family,regions in (('face',(0,1)),('hand',(2,3,4,5))):
                    if not getattr(options,family) or getattr(options,family+'_strength')==0: continue
                    delta=getattr(self,'native_v5_'+family).static_delta_v5(native_full_control.local,
                        layer=layer,sigma=body.sigma,target_hw=features.target_token_hw,regions=regions,
                        strength=self.native_v5_strength,options=options)
                    value=_IdentitySafeAdd.apply(value,delta)
                static.append(self.native_body.prepare_geometry(replace(features,geometry=value),
                    conv_input_valid=(body.valid_mask!=0) if native_conv_input_gate and body.valid_mask is not None else None))
        execution=Execution(body,native_full_control,options,tuple(static));token=self._native_v5_execution.set(execution)
        function=getattr(self,'_gradient_checkpointing_func',None);checkpointing=torch.is_grad_enabled() and self.gradient_checkpointing
        try:
            if checkpointing:
                standard=getattr(function,'__module__','')=='diffusers.models.modeling_utils' and getattr(function,'__name__','')=='_gradient_checkpointing_func'
                explicit=isinstance(function,partial) and function.func is torch_checkpoint and function.keywords.get('use_reentrant') is False
                if not (standard or explicit): raise ValueError('V5 requires non-reentrant checkpointing')
                def checkpoint_bound(block,*block_args): return function(partial(block,native_v5_execution=execution),*block_args)
                self._gradient_checkpointing_func=checkpoint_bound
            return super().forward(*args,**kwargs)
        finally:
            if checkpointing: self._gradient_checkpointing_func=function
            self._native_v5_execution.reset(token)


def install_native_full_v5(model,runtime):
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from .runtime import NativeFullRuntimeV5
    if not isinstance(model,QwenImage21Transformer2DModel) or not isinstance(runtime,NativeFullRuntimeV5): raise ValueError('Native Qwen21 and V5 required')
    if hasattr(model,'native_body') or hasattr(model,'control_blocks'): raise ValueError('Install V5 once on a clean Qwen')
    if not model.config.causal_condition or model.inner_dim!=runtime.config.hidden_dim or len(model.transformer_blocks)!=runtime.config.num_layers:
        raise ValueError('Native causal Qwen21 dimensions differ')
    if any(p.dtype!=torch.float32 for p in runtime.parameters()): raise ValueError('Keep V5 modules FP32')
    before=dict(model.named_parameters());flags={id(p):p.requires_grad for p in runtime.parameters()};signature=inspect.signature(type(model).forward)
    model.requires_grad_(False)
    for name,module in (('native_body',runtime.body),('native_v5_face',runtime.face_controller),('native_v5_hand',runtime.hand_controller),
                        ('native_v5_interaction',runtime.interaction_head),('native_v5_strength',runtime.strength)):
        model.add_module(name,module)
    for p in runtime.parameters(): p.requires_grad_(flags[id(p)])
    model._native_v5_config=runtime.config;model._native_v5_signature=signature
    model._native_v5_execution=ContextVar('native_v5_execution_'+str(id(model)),default=None)
    model._native_v5_binding=ContextVar('native_v5_binding_'+str(id(model)),default=None)
    model.__class__=type('NativeFullV5'+type(model).__name__,(TransformerMixin,type(model)),{})
    for i,block in enumerate(model.transformer_blocks):
        block._native_v5_owner=weakref.ref(model);block._native_v5_layer=i
        block.__class__=type('NativeFullV5'+type(block).__name__,(BlockMixin,type(block)),{})
    if any(dict(model.named_parameters())[n] is not p for n,p in before.items()): raise RuntimeError('A base Qwen parameter was replaced')
    return {'architecture':'native_full_v5','batch_size':1,'kv_cache':False,'new_qwen_tokens':0,
            'shared_query':'Body+Interaction','base_frozen':True,'requires_grad_preserved':True,'local_mode':runtime.config.local_mode}


@contextmanager
def bind_native_full_v5(model,control,*,options=None,valid_mask=None,conv_input_gate=True):
    if not isinstance(model,TransformerMixin) or model._native_v5_binding.get() is not None: raise ValueError('Clean installed V5 binding required')
    token=model._native_v5_binding.set(Binding(control,options or NativeControlOptionsV5(),valid_mask,conv_input_gate))
    try: yield model
    finally: model._native_v5_binding.reset(token)
