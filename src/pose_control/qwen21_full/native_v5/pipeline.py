"""Single primary Qwen reference binding. Static conditioning encodes once/request."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from ..native_full_data import identity_from_reference_v4
from .contracts import NativeControlOptionsV5
from ..native_dual_depth import DualControlOptions


@contextmanager
def bind_native_full_pipeline_v5(transformer,runtime,batch,*,options=None,base_options=None):
    if runtime.training: raise ValueError('Inference binding requires eval; training uses encode with autograd')
    options=options or NativeControlOptionsV5();base_options=base_options or DualControlOptions()
    prepared=None;report={'architecture':'native_full_v5','condition_encodes':0,'primary_reference_images':1,'qwen_tokens_added':0}
    def prepare(module,args,kwargs):
        nonlocal prepared
        if torch.is_grad_enabled() or args: raise ValueError('Use named Qwen inputs under no_grad for inference')
        shapes=kwargs.get('img_shapes')
        if not isinstance(shapes,(tuple,list)) or len(shapes)!=1 or len(shapes[0])!=2: raise ValueError('V5 requires B=1 and one primary Source')
        source,target=shapes[0];n=source[1]*source[2];hidden=kwargs['hidden_states']
        if tuple(source[1:])!=tuple(batch['source_token_hw']) or tuple(target[1:])!=tuple(batch['target_token_hw']) or hidden.shape!=(1,n+target[1]*target[2],64):
            raise ValueError('Actual Qwen64 prefix/grid differs from declared Canvas')
        if prepared is None:
            primary=hidden[:,:n]
            identity=identity_from_reference_v4(batch,primary,tuple(source[1:])) if base_options.identity and base_options.identity_strength else None
            prepared=runtime.encode(batch,target_token_hw=tuple(target[1:]),identity=identity,source_latent=primary,options=options,base_options=base_options)
            report['condition_encodes']+=1
        if 'native_full_control' in kwargs or 'native_geometry' in kwargs: raise ValueError('Duplicate V5 binding')
        kwargs.update(native_full_control=prepared,native_full_options=options,native_enabled=True,
            native_valid_mask=F.avg_pool2d(batch['valid_image_mask'].to(hidden.device).float(),16).flatten(1),native_conv_input_gate=True)
        return args,kwargs
    handle=transformer.register_forward_pre_hook(prepare,with_kwargs=True)
    try: yield report
    finally: handle.remove()
