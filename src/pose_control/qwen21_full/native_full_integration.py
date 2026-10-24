"""V4-only Qwen installation: one Body, immutable checkpoint-owned execution."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
import inspect
import weakref
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .native_integration import _prepare_execution
from .native_body import _IdentitySafeAdd
from .native_local_condition import PreparedNativeFullControlV4
from .native_local_control import schedule_value
from .native_full_runtime import FullControlOptionsV4


@dataclass(frozen=True)
class _FullBinding:
    control: PreparedNativeFullControlV4
    options: FullControlOptionsV4
    valid_mask: torch.Tensor | None
    strength: float
    conv_input_gate: bool


@dataclass(frozen=True)
class _FullExecution:
    body: object
    control: PreparedNativeFullControlV4
    options: FullControlOptionsV4
    config: object


class _FullBlockMixin:
    def forward(self,*args,native_full_execution=None,**kwargs):
        original=super().forward(*args,**kwargs)
        owner=self._native_full_owner()
        if owner is None: raise RuntimeError('V4 transformer owner is missing')
        execution=native_full_execution or owner._native_full_execution.get()
        if execution is None or not execution.body.enabled: return original
        e=execution.body;layer=self._native_full_layer;options=execution.options
        local=execution.control.local
        common=dict(prepared=e.geometry,layer=layer,sigma=e.sigma,target_mask=e.target_mask,valid_mask=e.valid_mask,strength=e.strength)
        face=options.face and options.face_strength>0
        hand=options.hand and options.hand_strength>0
        if local is None or execution.config.local_mode=='static' or not (face or hand) or layer not in execution.config.injection_layers:
            return owner.native_body(original,**common)
        with torch.autocast(device_type=original.device.type,enabled=False):
            # All three read the same unmodified post-block Hidden. Sum FP32,
            # cast once, and write only target cells (text/reference prefix intact).
            delta=owner.native_body(original,**common,return_delta=True)
            hidden=original[:,e.target_mask]
            for enabled,name,regions,strength,knots in (
                (face,'face',(0,1),options.face_strength,execution.config.face_schedule),
                (hand,'hand',(2,3,4,5),options.hand_strength,execution.config.hand_schedule)):
                if enabled:
                    update=getattr(owner,'native_'+name+'_controller')(hidden,local,layer=layer,sigma=e.sigma,
                        target_hw=execution.control.body.target_token_hw,regions=regions)
                    if e.valid_mask is not None: update=update*(e.valid_mask!=0)[...,None]
                    delta=delta+update*strength*schedule_value(e.sigma,knots)[:,None,None]
        result=original.clone()
        result[:,e.target_mask]=_IdentitySafeAdd.apply(original[:,e.target_mask],delta.to(original.dtype))
        return result


class _FullTransformerMixin:
    def forward(self,*args,native_full_control=None,native_full_options=None,native_valid_mask=None,
                native_strength=1.,native_conv_input_gate=False,native_enabled=True,**kwargs):
        if self._native_full_execution.get() is not None: raise RuntimeError('Reentrant V4 forward is unsupported')
        if 'native_geometry' in kwargs or getattr(self,'_pic_edit_qwen21_control_active',False):
            raise ValueError('Do not combine V4 with another Native/legacy installation')
        binding=self._native_full_binding.get()
        if binding is not None:
            if native_full_control is not None: raise ValueError('Duplicate V4 execution binding')
            native_full_control=binding.control;native_full_options=binding.options
            native_valid_mask=binding.valid_mask;native_strength=binding.strength;native_conv_input_gate=binding.conv_input_gate
        options=native_full_options or FullControlOptionsV4()
        if native_enabled:
            if not isinstance(native_full_control,PreparedNativeFullControlV4): raise ValueError('V4 prepared control required')
            native_full_control.validate()
            if any(m['options']!=dict(face=options.face,hand=options.hand,face_strength=options.face_strength,hand_strength=options.hand_strength) for m in native_full_control.metadata):
                raise ValueError('Prepare local controls with the same execution switches/strengths')
        bound=self._native_full_signature.bind(self,*args,**kwargs);bound.apply_defaults()
        features=None if native_full_control is None else native_full_control.body
        config=self._native_full_config
        if native_enabled and config.local_mode=='static':
            value=features.geometry
            for delta,strength,knots in ((native_full_control.static_face,options.face_strength,config.face_schedule),
                                         (native_full_control.static_hand,options.hand_strength,config.hand_schedule)):
                if delta is not None:
                    value=_IdentitySafeAdd.apply(value,delta*strength*schedule_value(bound.arguments['timestep'],knots)[:,None,None])
            features=replace(features,geometry=value)
        body=_prepare_execution(self,bound.arguments,features,native_enabled,native_valid_mask,native_strength,native_conv_input_gate)
        execution=_FullExecution(body,native_full_control,options,config)
        token=self._native_full_execution.set(execution)
        function=getattr(self,'_gradient_checkpointing_func',None)
        checkpointing=torch.is_grad_enabled() and self.gradient_checkpointing
        try:
            if checkpointing:
                standard=getattr(function,'__module__','')=='diffusers.models.modeling_utils' and getattr(function,'__name__','')=='_gradient_checkpointing_func'
                explicit=isinstance(function,partial) and function.func is torch_checkpoint and function.keywords.get('use_reentrant') is False
                if not (standard or explicit): raise ValueError('V4 requires non-reentrant checkpointing')
                def checkpoint_bound(block,*block_args):
                    return function(partial(block,native_full_execution=execution),*block_args)
                self._gradient_checkpointing_func=checkpoint_bound
            return super().forward(*args,**kwargs)
        finally:
            if checkpointing: self._gradient_checkpointing_func=function
            self._native_full_execution.reset(token)


def install_native_full_v4(model,runtime):
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from .native_full_runtime import NativeFullRuntimeV4
    if not isinstance(runtime,NativeFullRuntimeV4) or not isinstance(model,QwenImage21Transformer2DModel): raise ValueError('Native Qwen21 and Full V4 required')
    if hasattr(model,'native_body') or hasattr(model,'control_blocks'): raise ValueError('Install V4 exactly once on a clean native Qwen')
    if (not model.config.causal_condition or model.inner_dim!=runtime.config.hidden_dim
            or len(model.transformer_blocks)!=runtime.config.num_layers):
        raise ValueError('V4 requires the declared causal Qwen21 Hidden/layer configuration')
    if any(p.dtype!=torch.float32 for p in runtime.parameters()): raise ValueError('V4 modules must stay FP32; cast Qwen before installation')
    before=dict(model.named_parameters());flags={id(p):p.requires_grad for p in runtime.parameters()}
    signature=inspect.signature(type(model).forward)
    model.requires_grad_(False)
    for name,module in (('native_body',runtime.body),('native_face_controller',runtime.face_controller),('native_hand_controller',runtime.hand_controller)):
        model.add_module(name,module)
    for p in runtime.parameters(): p.requires_grad_(flags[id(p)])
    model._native_full_config=runtime.config
    model._native_full_signature=signature
    model._native_full_execution=ContextVar('native_full_execution_'+str(id(model)),default=None)
    model._native_full_binding=ContextVar('native_full_binding_'+str(id(model)),default=None)
    model.__class__=type('NativeFullV4'+type(model).__name__,(_FullTransformerMixin,type(model)),{})
    for layer,block in enumerate(model.transformer_blocks):
        block._native_full_owner=weakref.ref(model);block._native_full_layer=layer
        block.__class__=type('NativeFullV4'+type(block).__name__,(_FullBlockMixin,type(block)),{})
    if any(dict(model.named_parameters())[n] is not p for n,p in before.items()): raise RuntimeError('V4 replaced a Qwen parameter')
    return {'architecture':'native_full_v4','base_frozen':True,'requires_grad_preserved':True,'batch_size':1,'kv_cache':False,
            'verification_only':runtime.config.verification_only,
            'local_layers':list(runtime.config.injection_layers),'local_mode':runtime.config.local_mode,'new_qwen_tokens':0}


@contextmanager
def bind_native_full_v4(model,control,*,options=None,valid_mask=None,strength=1.,conv_input_gate=True):
    if not isinstance(model,_FullTransformerMixin): raise ValueError('Install V4 before binding')
    if model._native_full_binding.get() is not None: raise ValueError('Duplicate V4 binding')
    token=model._native_full_binding.set(_FullBinding(control,options or FullControlOptionsV4(),valid_mask,strength,conv_input_gate))
    try: yield model
    finally: model._native_full_binding.reset(token)
