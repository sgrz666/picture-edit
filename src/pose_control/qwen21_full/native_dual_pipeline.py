"""One inference-call binding: reuse the REAL Qwen reference prefix latent once."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F

from .dual_data_runtime import identity_from_reference
from .native_dual_depth import DualControlOptions


@contextmanager
def bind_native_dual_pipeline(transformer, runtime, batch, *, options=None):
    """No second VAE pass and no target RGB; only the declared reference slice is read.

    This memoization is inference-only and lives for one pipeline call. Training
    calls runtime.encode once per step and never caches learned outputs.
    """
    options = options or DualControlOptions()
    if runtime.training:
        raise ValueError('Pipeline binding is inference-only; training must encode with autograd per step')
    prepared = None
    report = {'condition_encodes': 0, 'reference_images': 1, 'new_qwen_tokens': 0}

    def prepare(module, args, kwargs):
        nonlocal prepared
        if torch.is_grad_enabled():
            raise ValueError('Inference binding requires no_grad/inference_mode')
        if args:
            raise ValueError('Qwen pipeline must pass named transformer inputs')
        shapes = kwargs.get('img_shapes')
        if not isinstance(shapes, (list, tuple)) or len(shapes) != 1 or len(shapes[0]) != 2:
            raise ValueError('V3 requires B=1, exactly one reference grid, and one target grid')
        reference, target = shapes[0]
        if tuple(reference[1:]) != tuple(batch['source_token_hw']) or tuple(target[1:]) != tuple(batch['target_token_hw']):
            raise ValueError('Actual pipeline reference/target canvas differs; no implicit mask resizing')
        hidden = kwargs['hidden_states']
        n = reference[1]*reference[2]
        if hidden.shape != (1, n+target[1]*target[2], 64):
            raise ValueError('Expected raw native Qwen 64-channel reference+target latents')
        if prepared is None:
            identity = identity_from_reference(batch, hidden[:, :n], tuple(reference[1:])) if options.identity else None
            prepared = runtime.encode(batch, target_token_hw=tuple(target[1:]), identity=identity, options=options)
            report['condition_encodes'] += 1
        if kwargs.get('native_geometry') is not None:
            raise ValueError('A second Native binding would duplicate control')
        kwargs['native_geometry'] = prepared
        kwargs['native_enabled'] = True
        kwargs['native_valid_mask'] = F.avg_pool2d(batch['valid_image_mask'].to(hidden.device).float(), 16).flatten(1)
        kwargs['native_conv_input_gate'] = True
        return args, kwargs

    handle = transformer.register_forward_pre_hook(prepare, with_kwargs=True)
    try:
        yield report
    finally:
        handle.remove()
