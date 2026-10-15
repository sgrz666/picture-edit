import os

import numpy as np

H = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'
cands = {
    'geometry(target hug#108)': f'{H}/final_assets_native_v1/pair00/hug00/cam4/geometry/000108.npz',
    'geometry(target hug#104)': f'{H}/final_assets_native_v1/pair00/hug00/cam4/geometry/000104.npz',
    'geometry(source taichi#41)': f'{H}/final_assets_native_v1/pair00/taichi00/cam4/geometry/000041.npz',
    'dwpose target A #108': f'{H}/native_interaction_v2/dwpose_native/pair00/hug00/000108_A.npz',
    'dwpose target B #108': f'{H}/native_interaction_v2/dwpose_native/pair00/hug00/000108_B.npz',
    'smpl target #108': f'{H}/raw/pair00/hug00/smpl/000108.npz',
    'cameras rgb': f'{H}/raw/pair00/hug00/cameras/rgb_cameras.npz',
    'seg mask A target#108 (raw)': f'{H}/raw/pair00/hug00/seg/img_seg_mask/4/0/000108.png',
    'seg mask B target#108 (raw)': f'{H}/raw/pair00/hug00/seg/img_seg_mask/4/1/000108.png',
}
for name, p in cands.items():
    print('=' * 72)
    print(name)
    print('  ', p)
    if not os.path.isfile(p):
        print('   MISSING')
        continue
    if p.endswith('.png'):
        from PIL import Image
        im = Image.open(p)
        a = np.asarray(im)
        print('   PNG size=%s mode=%s unique=%s' % (im.size, im.mode, np.unique(a)[:6]))
        continue
    try:
        d = np.load(p, allow_pickle=True)
        for k in d.files:
            arr = d[k]
            print('   %-30s shape=%s dtype=%s' % (k, getattr(arr, 'shape', None), getattr(arr, 'dtype', None)))
    except Exception as exc:
        print('   ERR', exc)

# 是否存在 normal 类文件
print()
print('=' * 72)
print('搜索 normal / depth / render 相关文件')
import subprocess
for pat in ('*normal*', '*depth*', '*render*', '*contact*'):
    r = subprocess.run(['bash', '-lc', f'find {H} -maxdepth 6 -iname "{pat}" 2>/dev/null | head -4'],
                       capture_output=True, text=True)
    hits = [x for x in r.stdout.strip().split('\n') if x]
    print(f'  {pat:12s} -> {len(hits)} 例')
    for h in hits[:2]:
        print('      ', h)
