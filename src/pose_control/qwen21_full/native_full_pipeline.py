"""One request/step V4 binding; the actual Qwen reference prefix is reused."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from .native_full_runtime import FullControlOptionsV4
from .native_dual_depth import DualControlOptions
from .native_full_data import identity_from_reference_v4


@contextmanager
def bind_native_full_pipeline_v4(transformer,runtime,batch,*,options=None,base_options=None):
    if runtime.training: raise ValueError('Inference-only binding; train via encode once per step with autograd')
    options=options or FullControlOptionsV4();base_options=base_options or DualControlOptions()
    prepared=None;report={'condition_encodes':0,'reference_images':1,'qwen_tokens_added':0,'architecture':'native_full_v4'}
    def prepare(module,args,kwargs):
        nonlocal prepared
        if torch.is_grad_enabled() or args: raise ValueError('Use named Qwen inputs inside inference_mode/no_grad')
        shapes=kwargs.get('img_shapes')
        if not isinstance(shapes,(list,tuple)) or len(shapes)!=1 or len(shapes[0])!=2: raise ValueError('V4 requires B=1 and one source image')
        source,target=shapes[0]
        if tuple(source[1:])!=tuple(batch['source_token_hw']) or tuple(target[1:])!=tuple(batch['target_token_hw']): raise ValueError('Actual Qwen source/target canvas differs')
        hidden=kwargs['hidden_states'];n=source[1]*source[2]
        if hidden.shape!=(1,n+target[1]*target[2],64): raise ValueError('Expected native64-channel raw image prefix')
        if prepared is None:
            identity=identity_from_reference_v4(batch,hidden[:,:n],tuple(source[1:])) if base_options.identity and base_options.identity_strength else None
            prepared=runtime.encode(batch,target_token_hw=tuple(target[1:]),options=options,base_options=base_options,identity=identity)
            report['condition_encodes']+=1
        if 'native_geometry' in kwargs or 'native_full_control' in kwargs: raise ValueError('Duplicate Native control binding')
        kwargs.update(native_full_control=prepared,native_full_options=options,native_enabled=True,
            native_valid_mask=F.avg_pool2d(batch['valid_image_mask'].to(hidden.device).float(),16).flatten(1),native_conv_input_gate=True)
        return args,kwargs
    handle=transformer.register_forward_pre_hook(prepare,with_kwargs=True)
    try: yield report
    finally: handle.remove()
