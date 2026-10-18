"""Post-block control for QwenImage21's validated native target suffix.

Training owns one context per forward/backward graph. Native prefix KV caching
and CFG are left to the pipeline; inference may reuse this context across steps.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import inspect
import math

import torch

from .conditions import ControlStrengths


class DualV8ControlContext(AbstractContextManager):
    def __init__(self, transformer, adapter, prepared, *, sigma, strengths=None):
        self.transformer = transformer
        self.adapter = adapter
        self.prepared = prepared
        self.sigma_provider = sigma
        self.strengths = ControlStrengths() if strengths is None else strengths
        self.handles = []
        self.call_counts = {int(i): 0 for i in adapter.layer_indices}
        self.forward_count = 0
        self._training_forward_seen = False
        self._active = False
        self._owns_lock = False
        self._timestep = self._sigma = None
        self._sequence_tokens = self._target_tokens = None

    @staticmethod
    def _batch_scalar(value, reference, name):
        tensor = torch.as_tensor(value,device=reference.device,dtype=torch.float32).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(reference.shape[0])
        if tensor.shape != (reference.shape[0],) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{name} must contain one finite value per batch row')
        # A scheduler may mutate its tensors on the next step. Training checkpoint
        # recomputation must retain this forward's actual numerical conditions.
        return tensor.detach().clone()

    def _before_transformer(self,module,args,kwargs):
        if torch.is_grad_enabled() and self._training_forward_seen:
            raise RuntimeError('one training forward per context; complete backward before creating the next context')
        inputs = dict(inspect.signature(module.forward).bind_partial(*args,**kwargs).arguments)
        inputs.update(inputs.pop('kwargs',{}))
        shapes = inputs.get('img_shapes')
        hidden = inputs.get('hidden_states')
        prompt = inputs.get('encoder_hidden_states')
        mask = inputs.get('img_mask')
        if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[-1] != 64:
            raise ValueError('QwenImage21 packed image latents must be [B,N,64]')
        batch = hidden.shape[0]
        if (not shapes or len(shapes) != batch or not shapes[0]
                or any(row != shapes[0] for row in shapes)):
            raise ValueError('native img_shapes must have an identical image layout for each batch row')
        if any(len(shape)!=3 or shape[0]!=1 or min(shape)<1 for shape in shapes[0]):
            raise ValueError('native image shapes require single frames and positive token dimensions')
        _,h,w = shapes[0][-1]
        if (h,w) != self.prepared.target_hw or self.prepared.dense.shape[0] != batch:
            raise ValueError('prepared geometry and native target token grid/batch differ')
        target_count = h*w
        image_count = sum(math.prod(shape) for shape in shapes[0])
        if target_count%4 or image_count != hidden.shape[1]:
            raise ValueError('native packed image count or four-token image slots are inconsistent')
        if (not torch.is_tensor(prompt) or prompt.ndim!=3 or prompt.shape[0]!=batch
                or not torch.is_tensor(mask) or mask.dtype!=torch.bool
                or mask.shape!=(batch,prompt.shape[1]+target_count//4)):
            raise ValueError('native prompt/image-slot layout is inconsistent')
        if not torch.equal(mask,mask[:1].expand_as(mask)):
            raise ValueError('native image slots must be shared across the batch')
        if not bool(mask[:,-target_count//4:].all()):
            raise ValueError('target image slots must form the native target suffix')
        expanded = torch.repeat_interleave(mask[0],torch.where(mask[0],4,1))
        if int(expanded.sum()) != image_count:
            raise ValueError('native image slots do not match packed image count')
        _,target_mask = module.build_token_metadata(expanded,shapes[0])
        if bool(target_mask[:-target_count].any()) or not bool(target_mask[-target_count:].all()):
            raise ValueError('native target metadata does not describe a contiguous target suffix')
        cached = inputs.get('kv_cache_mode') == 'cached'
        self._sequence_tokens = target_count if cached else expanded.numel()
        self._target_tokens = target_count
        self._timestep = self._batch_scalar(inputs['timestep'],hidden,'model timestep')
        actual_sigma = self.sigma_provider(inputs) if callable(self.sigma_provider) else self.sigma_provider
        self._sigma = self._batch_scalar(actual_sigma,hidden,'flow sigma')
        if bool(((self._sigma<0)|(self._sigma>1)).any()):
            raise ValueError('flow sigma must be in [0,1]')
        self._training_forward_seen = torch.is_grad_enabled()
        self.forward_count += 1
        if getattr(self.adapter,'capture_residual_maps',False):
            self.adapter.last_residual_maps={}

    def _after_block(self,layer):
        def inject(module,args,output):
            if not self._active or self._timestep is None:
                raise RuntimeError('dual context must remain active through forward and backward')
            expected = (self.prepared.dense.shape[0],self._sequence_tokens,self.adapter.hidden_size)
            if not torch.is_tensor(output) or tuple(output.shape) != expected:
                raise ValueError(f'unexpected native block output; expected {expected}')
            target = output[:,-self._target_tokens:]
            delta = self.adapter.residual(layer,target,self.prepared,self._timestep,self._sigma,self.strengths)
            if delta.shape != target.shape:
                raise ValueError('dual residual must have the native target shape')
            injected_delta=delta.to(target)
            if getattr(self.adapter,'capture_residual_maps',False):
                self.adapter.last_residual_maps[layer]=dict(
                    rms=injected_delta.detach().float().square().mean(-1).sqrt().reshape(
                        delta.shape[0],*self.prepared.target_hw).cpu(),
                    sigma=float(self._sigma[0]),timestep=float(self._timestep[0]),
                    target_hw=self.prepared.target_hw)
            self.call_counts[layer] += 1
            updated = target+injected_delta
            return updated if self._sequence_tokens==self._target_tokens else torch.cat(
                (output[:,:-self._target_tokens],updated),dim=1)
        return inject

    def _after_transformer(self,module,args,output):
        value = output[0] if isinstance(output,(tuple,list)) else output.sample
        if value.requires_grad:
            def require_live_context(gradient):
                if not self._active:
                    raise RuntimeError('dual context must remain active until backward finishes')
                return gradient
            value.register_hook(require_live_context)

    def __enter__(self):
        if self._active or getattr(self.transformer,'_pic_edit_qwen21_control_active',False):
            raise RuntimeError('another Qwen control context is active')
        if self.strengths.disabled:
            return self
        config = self.transformer.config
        if (len(self.transformer.transformer_blocks)!=32 or config.patch_size!=1
                or config.in_channels!=64 or not config.causal_condition
                or self.transformer.inner_dim!=self.adapter.hidden_size):
            raise ValueError('dual_v8 requires the native causal 32-block QwenImage21 layout')
        if any(i<0 or i>=32 for i in self.adapter.layer_indices):
            raise ValueError('invalid native block index')
        self.transformer._pic_edit_qwen21_control_active = True
        self._owns_lock = self._active = True
        try:
            self.handles.append(self.transformer.register_forward_pre_hook(self._before_transformer,with_kwargs=True))
            for layer in self.adapter.layer_indices:
                self.handles.append(self.transformer.transformer_blocks[layer].register_forward_hook(self._after_block(layer)))
            self.handles.append(self.transformer.register_forward_hook(self._after_transformer))
        except Exception:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self,exception_type,exception,traceback):
        self._active = False
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if self._owns_lock:
            self.transformer._pic_edit_qwen21_control_active = False
            self._owns_lock = False
