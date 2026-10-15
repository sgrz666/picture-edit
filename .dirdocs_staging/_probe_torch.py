import torch, json, os, glob
E='/home/shangguanrz/project/pic-edit/experiments'
f=E+'/iper_adapter_finetune_v1/condition_cache/001_10_source_motion_f000050.pt'
d=torch.load(f, map_location='cpu')
print('type', type(d))
if isinstance(d, dict):
    for k,v in d.items():
        if torch.is_tensor(v): print('  ',k, tuple(v.shape), v.dtype)
        else: print('  ',k, type(v), str(v)[:200])
print()
print('=== iper_v6_comparative_3000 (archived) ===')
C='/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/iper_v6_comparative_3000'
for p in sorted(glob.glob(C+'/*')):
    print('  ', 'DIR ' if os.path.isdir(p) else os.path.getsize(p), os.path.basename(p))
for d2 in glob.glob(C+'/*/'):
    ch=sorted(os.listdir(d2))
    print(os.path.basename(d2), len(ch), ch[:8])
for jf in glob.glob(C+'/**/*.json', recursive=True):
    print('---', jf)
    print(open(jf,errors='replace').read()[:1800])
