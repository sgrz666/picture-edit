import re
a = open('/home/shangguanrz/project/pic-edit/train_iper_qwen21_body.py').read()
b = open('/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment/code/overfit_inter01.py').read()

def grab(src, name):
    i = src.find(f'def {name}(')
    if i < 0: return None
    j = src.find('\ndef ', i + 5)
    return src[i:j if j > 0 else len(src)]

oa = grab(a, 'loss_for_batch')
ob = grab(b, 'loss_for_batch')
print('=== 原始 train_iper_qwen21_body.py::loss_for_batch ===')
print(oa)
print()
print('=== 本次 overfit_inter01.py::loss_for_batch ===')
print(ob)
print()
print('=== 关键行提取 ===')
for tag, src in (('原始', oa), ('本次', ob)):
    for ln in src.split('\n'):
        s = ln.strip()
        if any(k in s for k in ('sigma =', 'noisy =', 'velocity =', 'weight =',
                                'loss =', 'prediction =', 'interpolate', 'autocast')):
            print(f'  [{tag}] {s}')
