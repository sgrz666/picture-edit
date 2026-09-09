import sys
import torch

print('Python:', sys.version)
print('Torch version:', torch.__version__)
print('Device name:', torch.cuda.get_device_name(0))
print('CUDA Capability:', torch.cuda.get_device_capability(0))
print('Total memory (GiB):', torch.cuda.get_device_properties(0).total_memory / (1024**3))

q = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.bfloat16)
out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
print('SDPA works successfully, output shape:', out.shape)

packages = ['accelerate', 'smplx', 'trimesh', 'pytorch3d', 'flash_attn']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'available')
        print(pkg + ': ' + str(ver))
    except Exception as err:
        print(pkg + ': NOT installed (' + str(err) + ')')
