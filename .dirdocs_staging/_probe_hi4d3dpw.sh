#!/bin/bash
R=/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1
C=/data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls
echo "=== cache dir ==="
ls $C | head -5
ls $C | wc -l
du -sh --si $C
python3 -c "
import torch, glob
fs = sorted(glob.glob('$C/control_*.pt'))
print('n_files', len(fs))
p = fs[0]
d = torch.load(p, map_location='cpu')
print('FILE', p)
for k, v in d.items():
    print(k, tuple(v.shape), v.dtype)
import os
print('size_MB', round(os.path.getsize(p)/1e6,1))
"
echo "=== raw hug00 counts ==="
ls $R/raw/pair00/hug00/images/4 | wc -l
ls $R/raw/pair00/hug00/smpl | wc -l
ls $R/raw/pair00/hug00/seg/img_seg_mask/4
echo "=== processed channel counts hug00 ==="
ls $R/processed_512_1500/pair00/hug00/depth | wc -l
ls $R/processed_512_1500/pair00/hug00/normal | wc -l
ls $R/processed_512_1500/pair00/hug00/person_id | wc -l
ls $R/processed_512_1500/pair00/hug00/scene_person_id | wc -l
ls $R/processed_512_1500/pair00/hug00/smpl_mask | wc -l
ls $R/processed_512_1500/pair00/hug00/occlusion | wc -l
ls $R/processed_512_1500/pair00/hug00/pose/A | wc -l
ls $R/processed_512_1500/pair00/hug00/pose/A_npy | wc -l
ls $R/processed_512_1500/pair00/hug00/person_mask/A | wc -l
head -c 1500 $R/processed_512_1500/pair00/hug00/sequence_meta.json
echo
python3 -c "
from PIL import Image
import numpy as np, glob
R='$R/processed_512_1500/pair00/hug00'
for ch in ['rgb','depth','normal','person_id','scene_person_id','smpl_mask','occlusion']:
    fs=sorted(glob.glob(R+'/'+ch+'/*.*'))
    f=fs[0]
    if f.endswith('.npy'):
        a=np.load(f); print(ch, f.split('/')[-1], a.shape, a.dtype, a.min(), a.max())
    else:
        im=Image.open(f); a=np.asarray(im); print(ch, f.split('/')[-1], im.size, im.mode, 'uniqvals', np.unique(a)[:6])
f=sorted(glob.glob(R+'/pose/A_npy/*.npy'))[0]
a=np.load(f); print('pose_A_npy', f.split('/')[-1], a.shape, a.dtype)
"
