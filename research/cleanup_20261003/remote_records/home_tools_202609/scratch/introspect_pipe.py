import inspect
import json
import sys

import torch
import diffusers
from diffusers import QwenImage21Pipeline

print('diffusers', diffusers.__version__)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device count', torch.cuda.device_count())
free, total = torch.cuda.mem_get_info()
print(f'VRAM free {free/2**30:.2f} GiB / total {total/2**30:.2f} GiB')

sig = inspect.signature(QwenImage21Pipeline.__call__)
print('\n=== QwenImage21Pipeline.__call__ 参数 ===')
for name, p in sig.parameters.items():
    if name == 'self':
        continue
    d = '' if p.default is inspect._empty else f' = {p.default!r}'
    print(f'  {name}: {p.annotation if p.annotation is not inspect._empty else "?"}{d}')

print('\n=== CPU offload 能力 ===')
for m in ('enable_model_cpu_offload', 'enable_sequential_cpu_offload',
          'enable_attention_slicing', 'enable_vae_slicing', 'enable_vae_tiling'):
    print(f'  {m}: {hasattr(QwenImage21Pipeline, m)}')

print('\n=== 组件 ===')
print(' ', list(QwenImage21Pipeline._optional_components), '| required:', list(QwenImage21Pipeline._required_components) if hasattr(QwenImage21Pipeline, "_required_components") else 'n/a')

print('\n=== from_pretrained 关键参数 ===')
sp = inspect.signature(QwenImage21Pipeline.from_pretrained)
for n in ('torch_dtype', 'dtype', 'device_map', 'low_cpu_mem_usage'):
    print(f'  {n}: {"YES" if n in sp.parameters else "no"}')
